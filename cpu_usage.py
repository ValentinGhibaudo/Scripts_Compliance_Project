import time
import psutil
import  numpy as np

import threading


# samnode 10.69.168.40

# TODO red this
# https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.cpu_percent


def start_measurement(func, args):

    cpu_list = []
    mem_list = []
    currentProcess = psutil.Process()
    running = True
    def cpu_sniff():
        while running:
            cpu = currentProcess.cpu_percent(interval=0.1)
            cpu_list.append(cpu)
            mem = currentProcess.memory_info()
            mem_list.append(mem.rss)

    thread = threading.Thread(target=cpu_sniff)
    thread.start()

    time.sleep(1.5)

    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()

    running = False
    thread.join()
    # print(cpu_list)
    # print(mem_list)

    d = dict(
        duration_s=t1-t0,
        cpu=max(cpu_list) - min(cpu_list),
        mem=max(mem_list) - min(mem_list),
    )
    return d






def my_func(a, b):
    for i in range(50):

        c = np.fft.fft(a) * np.fft.fft(b)


def test_start_measurement():
    a = np.random.uniform(size=1024**2)
    b = np.random.uniform(size=1024**2)
    d = start_measurement(my_func, (a, b))
    print(d)


if __name__ == "__main__":
    test_start_measurement()
