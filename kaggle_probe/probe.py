import sys, platform, subprocess
from importlib.metadata import version, PackageNotFoundError
print("PYTHON", sys.version)
print("PYTAG  cp%d%d" % sys.version_info[:2])
print("PLATFORM", platform.platform(), platform.machine())
print("GLIBC", platform.libc_ver())
for p in ["torch", "numpy", "gymnasium", "gym", "stable-baselines3", "sb3-contrib",
          "cloudpickle", "farama-notifications", "pandas", "matplotlib",
          "typing-extensions", "pyarrow", "python-dotenv", "arc-agi"]:
    try:
        print(f"PKG {p:22s} {version(p)}")
    except PackageNotFoundError:
        print(f"PKG {p:22s} MISSING")
import torch
print("TORCH cuda_build", torch.version.cuda, "| threads", torch.get_num_threads())
print(subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True).stdout)
