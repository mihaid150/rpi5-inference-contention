import os
import sys

if __name__ == "__main__":
    wanted = {int(x) for x in sys.argv[1].split(",")}
    if not wanted or not wanted.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Requested CPUs outside effective allowance")
    os.sched_setaffinity(0, wanted)
    os.execv(sys.argv[2], sys.argv[2:])
