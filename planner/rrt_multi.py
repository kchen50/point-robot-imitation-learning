import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Optional progress bar (fallback to no-op if tqdm not installed)
try:
    from tqdm import tqdm as _tqdm
except Exception:
    class _tqdm:  # type: ignore
        def __init__(self, total=None, desc=None, dynamic_ncols=True):
            self.total = total
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix_str(self, s):
            pass
        def close(self):
            pass

def _limit_blas_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _collect_worker(
    xml_path,
    num_trajectories,
    seed,
    steps_per_action=5,
    time_limit_seconds=30.0,
    kdtree_rebuild_every=64,
    randomize_start=True,
    min_plan_len=1,
):
    _limit_blas_threads()
    import numpy as np  # delayed import in worker
    import mujoco
    try:
        from rrt_planner import RRTPlanner
    except Exception:
        from planner.rrt_planner import RRTPlanner

    planner = RRTPlanner(
        xml_path=xml_path,
        steps_per_action=steps_per_action,
        time_limit_seconds=time_limit_seconds,
        kdtree_rebuild_every=kdtree_rebuild_every,
    )
    return planner.collect(
        num_trajectories=num_trajectories,
        seed=seed,
        randomize_start=randomize_start,
        min_plan_len=min_plan_len,
        progress_every=None,
        show_progress=False,
    )


def collect_parallel_chunks(
    xml_path,
    total_trajectories,
    chunk_size=5,
    num_workers=None,
    base_seed=0,
    steps_per_action=5,
    time_limit_seconds=30.0,
    kdtree_rebuild_every=64,
    randomize_start=True,
    min_plan_len=1,
    verbose=True,
):
    if total_trajectories <= 0:
        return []
    if num_workers is None or num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 1))

    counts = [chunk_size] * (total_trajectories // chunk_size)
    rem = total_trajectories % chunk_size
    if rem:
        counts.append(rem)
    if not counts:
        return []

    seed_seq = np.random.SeedSequence(base_seed)
    seeds = list(seed_seq.generate_state(len(counts), dtype=np.uint32).astype(int))

    ctx = mp.get_context("spawn")
    start_time = time.time()
    done = 0
    all_trajs = []

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        pbar = _tqdm(total=total_trajectories, desc="Collect (parallel RRT)", dynamic_ncols=True) if verbose else None
        futures = []
        for n, s in zip(counts, seeds):
            futures.append(
                ex.submit(
                    _collect_worker,
                    xml_path,
                    n,
                    int(s),
                    steps_per_action,
                    time_limit_seconds,
                    kdtree_rebuild_every,
                    randomize_start,
                    min_plan_len,
                )
            )

        for idx, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            all_trajs.extend(batch)
            done += len(batch)
            if verbose:
                elapsed = time.time() - start_time
                if pbar is not None:
                    pbar.update(len(batch))
                    pbar.set_postfix_str(f"chunks {idx}/{len(counts)} | {elapsed:.2f}s")
                else:
                    print(f"[parallel] {done}/{total_trajectories} collected after {elapsed:.2f}s (chunks done: {idx}/{len(counts)})")
        if pbar is not None:
            pbar.close()

    return all_trajs[:total_trajectories]


if __name__ == "__main__":
    XML = "/home/kchen/MLAI/point-robot-imitation-learning/point_robot_nav.xml"
    trajs = collect_parallel_chunks(
        xml_path=XML,
        total_trajectories=50,
        chunk_size=5,
        num_workers=8,
        base_seed=123,
        steps_per_action=5,
        time_limit_seconds=30.0,
        randomize_start=True,
        min_plan_len=1,
        verbose=True,
    )
    print("Total trajectories:", len(trajs))

