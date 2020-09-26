## Installation

https://www.tensorflow.org/install

### 0. Optional: GPU Support

https://www.tensorflow.org/install/gpu

1. Install Visual Studion 2017 Free Version
   and C++ Redistributable:

   - [https://www.techspot.com/downloads/6278-visual-studio.html](https://www.techspot.com/downloads/6278-visual-studio.html)
   - [https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

2. Install CUDA Toolkit 10.1
[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

3. Install NVIDIA cuDNN version 7
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Download (you need an account) and copy the dll, include, and lib files in the corresponding directories of the CUDA Toolkit installation directory

4. On Windows: Modify environment variables (paths must match your installation directory):
- Add variable
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
```

Add 2 entries to PATH variable:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp
```

### 1. Installation
1. Create a virtual environment and activate it (e.g. with conda or virtualenv)

```console
conda create -n tf python=3.8
conda activate tf
```

2. Install with
```console
pip install tensorflow
```

### 2. Verification:
```python
import tensorflow as tf
print(tf.__version__)

# test gpu
physical_devices = tf.config.list_physical_devices("GPU")
print(len(physical_devices))
```