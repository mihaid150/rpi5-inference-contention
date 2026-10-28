import os
from pathlib import Path


def point(active, target):
    active = Path(active)
    temp = active.with_name(active.name + ".next")
    os.symlink(str(Path(target).resolve()), temp)
    os.replace(temp, active)
