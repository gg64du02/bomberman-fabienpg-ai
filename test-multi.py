from multiprocessing import Pool, TimeoutError
import time
import os

# st_time=time.time()


def f(x):
    return x*x

if __name__ == '__main__':
    pool = Pool(processes=8)              # start 4 worker processes

    # print "[0, 1, 4,..., 81]"
    print(pool.map(f, range(10)))

    # print same numbers in arbitrary order
    for i in pool.imap_unordered(f, range(10)):
        print(i)

    while(1):
        end_time = time.time()
        # # evaluate "f(20)" asynchronously
        # res = pool.apply_async(f, (20,))      # runs in *only* one process
        # print(res.get(timeout=1)    )          # prints "400"
        #
        # # evaluate "os.getpid()" asynchronously
        # res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        # print(res.get(timeout=1)     )         # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(8)]
        print([res.get(timeout=1) for res in multiple_results])


        print(format(1000 * (end_time - time.time())))



    # make a single worker sleep for 10 secs
    res = pool.apply_async(time.sleep, (10,))
    # res = pool.apply_async(f, (20,))
    try:
        print(res.get(timeout=1))
    except TimeoutError:
        print("We lacked patience and got a multiprocessing.TimeoutError")