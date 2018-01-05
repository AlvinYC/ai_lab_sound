# Description

Setup development environment upon Ubuntu Desktop 16.04 with LightDM GUI

## Install Ubuntu through USB or CDROM

## Step 1 - Login X-Window
* Press ``` CTRL-ALT-T ``` to launch Terminal shell 

* Install Ubuntu system update
    ```
    $ sudo apt-get update
    $ sudo apt-get upgrade
    ```

* Install OpenSSH server, allow remote ssh login
    ```
    $ sudo apt-get install -y openssh-server
    ```

* Verify remote ssh connection at notebook, and copy public key.
* Install essential toolkits
    ```
    $ sudo apt-get install -y build-essential curl unzip dkms pkg-config linux-headers-generic \
                              vim git tree libssl-dev zlib1g-dev libbz2-dev \
                              libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
                              libncursesw5-dev xz-utils tk-dev
    ```

* You might consider to modify /etc/default/locale, change all ``` lzh_TW ``` to ``` en_US.UTF-8 ``` 

* Reboot


## Step 2 - Download NVIDIA cuda drivers
* Download from local server, or from NVDIA developer program
    ```
    $ scp -r isaac:/home/dataset/GPU .
    ```

* Remove *incompatible* open source nvidia driver from kernel
    ```
    $ sudo bash -c "echo -e \"blacklist nouveau\nalias nouveau off\" > /etc/modprobe.d/nvidia.conf"
    $ sudo update-initramfs -u
    $ sudo reboot
    ```

## Step 3 - Install NVIDIA drivers
* Use SSH connect to target workstation
* Stop X-Window
    ```
    $ sudo service lightdm stop
    ```

* Update Ubuntu compatible NVidia driver 
    ```
    $ sudo ubuntu-drivers autoinstall
    $ sudo reboot
    ```

* Use SSH connect to target workstation again
* Verify driver installed correctly. 
    ```
    $ nvidia-smi
    ```

* Install CUDA driver and its patch
    ```
    //// Note: 
    ////    1. Do NOT install Graphics Driver, use above latest driver instead 
    ////    2. Did NOT install sample code
    ////    3. Yes to other options
    $ sudo sh GPU/cuda_8.0.61_375.26_linux-run
    $ sudo sh GPU/cuda_8.0.61.2_linux-run
    ```

* Install CUDNN driver for Neural Network accerlation.
    ```
    $ sudo tar zxvf GPU/cudnn-8.0-linux-x64-v6.0.tgz -C /usr/local/
    ```

* Install NVIDIA CUDA Profile Tools
    ```
    $ sudo apt-get install libcupti-dev
    ```

* Add cuda to library path, so other program can load library correctly
    ```
    $ sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
    $ sudo ldconfig
    ```

* Add cuda to execution path
    ```
    $ echo "export PATH=\"/usr/local/cuda/bin:\$PATH\"" >> ~/.bashrc
    ```

## Step 4 - Install Pyenv virtual envrionment & Python packages

* Install Pyenv
    ```
    $ curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
    $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    $ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
    $ source ~/.bashrc
    ```

* Install Python 3
    ```
    $ pyenv install 3.5.4
    $ pyenv global 3.5.4
    $ pyenv version
    $ python -V
    ```

* Install 3rd party Python library
    ```
    $ pip install --upgrade pip numpy ipykernel jupyter matplotlib Pillow
    $ pip install --upgrade pandas scipy scikit-learn gpustat
    ```

* Fix pip format error
    ```
    $ mkdir -p ~/.pip && echo -e "[list]\nformat = columns" > ~/.pip/pip.conf
    ```

* Install OpenCV
    ```
    $ sudo apt-get install -y ffmpeg openexr webp libgtk2.0-0
    $ pip install https://bazel.blob.core.windows.net/opencv/opencv_python-3.3.0-cp35-cp35m-linux_x86_64.whl
    ```

* Install Tensorflow with GPU support
    ```
    $ pip install --upgrade tensorflow-gpu
    ```

* Test if everything works fine, launch python with below scripts
    ```Python
    import numpy as np
    import cv2
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    ```

* Assume everything working, GPU information should be addtional printed upon Tensorflow session() called
    ```
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
    name: TITAN Xp
    major: 6 minor: 1 memoryClockRate (GHz) 1.582
    pciBusID 0000:09:00.0
    Total memory: 11.90GiB
    Free memory: 11.74GiB
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
    I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN Xp, pci bus id: 0000:09:00.0)
    b'Hello, TensorFlow!'
    ```

## Step 5 - Burn GPU and monitor 

* Download ` gpu-bench.py ` script
* Compute N iteration of matrix multiplication, say ` 2000 `
    ```
    $ time python gpu-bench.py 2000
    ```

* Open another terminal or SSH shell to watch GPU utilization, focus on GPU-Util percentage and Power voltage
    ```
    $ watch -n 1 nvidia-smi

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 0000:01:00.0     Off |                  N/A |
    | 20%   51C    P2   237W / 250W |  10695MiB / 11172MiB |    100%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |    0      1062    G   /usr/lib/xorg/Xorg                              62MiB |
    |    0      1780    C   python3                                      10629MiB |
    +-----------------------------------------------------------------------------+

    $ watch -n 1 -c gpustat

    [0] GeForce GTX 1080 Ti | 31'C,   0 % |    74 / 11172 MB | root(62M)
    ```

* Updates:
    - 2017/10/24:
        - In case that ubuntu autoupdate is enabled, your   nVidia driver will be auto installed/updated to 384   (from 375). You might encounter the error message     'Failed to initialize NVML: Driver/library version  mismatch' while executing 'nvidia-smi' command.  Simple route to make it work again: just reboot the  machine.
        - Please be noted that nVidia 384 driver's  accompanying CUDA version is 9.0. TensorFlow 1.4 has     offical python wheel install package for CUDA8 +    cuDNN6, but haven't provided CUDA9 + cuDNN7 yet (you   need to build from source).