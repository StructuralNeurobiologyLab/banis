import dask
from distributed import LocalCluster, progress, as_completed
from tqdm import tqdm
from tqdm.dask import TqdmCallback
from dask import delayed, compute, persist
from dask.distributed import Client
import time


# Create some simulated tasks
def work(x):
    time.sleep(1)
    return x * x

if __name__ == '__main__':
    # Start a local distributed cluster
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)

    print("computing with persist")
    tasks = [dask.delayed(work)(i) for i in range(20)]
    x = persist(tasks)  # start computation in the background
    progress(x)
    results1 = client.gather(x)

    print("computing with tqdm (doesn't work)")
    tasks = [dask.delayed(work)(i) for i in range(20)]
    with TqdmCallback(desc="Distributed compute", total=len(tasks), mininterval=0.5):
        results = client.compute(tasks, sync=True)

    print("computing with as_completed")
    tasks = [dask.delayed(work)(i) for i in range(20)]
    futures = client.compute(tasks)
    for future in tqdm(
            as_completed(futures),
            total=len(futures),
            smoothing=0,
            desc="Predicting chunks"
    ):
        pass
