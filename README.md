# VSports

## How to install

### Install TinyXML, Eigen, OpenGL, assimp, Python3, etc...

```bash
sudo apt-get install libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui
```

### Install boost with python3 (1.66)

We strongly recommand that you install boost libraries from the **source code**
(not apt-get, etc...).

- Download boost sources with the version over 1.66.(https://www.boost.org/users/history/version_1_66_0.html)

- Compile and Install the sources

```bash
cd /path/to/boost_1_xx/
./bootstrap.sh --with-python=python3
sudo ./b2 --with-python --with-filesystem --with-system --with-regex install
```

- Check yourself that the libraries are installed well in your directory `/usr/local/`. (or `/usr/`)

If installed successfully, you should have something like

Include

* `/usr/local/include/boost/`
* `/usr/local/include/boost/python/`
* `/usr/local/include/boost/python/numpy`

Lib 

* `/usr/local/lib/libboost_filesystem.so`
* `/usr/local/lib/libboost_python3.so`
* `/usr/local/lib/libboost_numpy3.so`


### Install DART 6.3

Please refer to http://dartsim.github.io/ (Install version 6.3)

If you are trying to use latest version, rendering codes should be changed according to the version. It is recommended to use the exact 6.3 version.

Manual from DART(http://dartsim.github.io/install_dart_on_ubuntu.html)
1. install required dependencies

```bash
sudo apt-get install build-essential cmake pkg-config git
sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install libopenscenegraph-dev
```
2. install DART v6.3.0

```bash
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.3.0
mkdir build
cd build
cmake ..
make -j4
sudo make install
```



## How to install

### Install TinyXML, Eigen, OpenGL, assimp, Python3, etc...

```bash
sudo apt-get install libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui
```

### Install boost with python3 (1.66)

We strongly recommand that you install boost libraries from the **source code**
(not apt-get, etc...).

- Download boost sources with the version over 1.66.(https://www.boost.org/users/history/version_1_66_0.html)

- Compile and Install the sources

```bash
cd /path/to/boost_1_xx/
./bootstrap.sh --with-python=python3
sudo ./b2 --with-python --with-filesystem --with-system --with-regex install
```

- Check yourself that the libraries are installed well in your directory `/usr/local/`. (or `/usr/`)

If installed successfully, you should have something like

Include

* `/usr/local/include/boost/`
* `/usr/local/include/boost/python/`
* `/usr/local/include/boost/python/numpy`

Lib 

* `/usr/local/lib/libboost_filesystem.so`
* `/usr/local/lib/libboost_python3.so`
* `/usr/local/lib/libboost_numpy3.so`


### Install DART 6.3

Please refer to http://dartsim.github.io/ (Install version 6.3)

If you are trying to use latest version, rendering codes should be changed according to the version. It is recommended to use the exact 6.3 version.

Manual from DART(http://dartsim.github.io/install_dart_on_ubuntu.html)
1. install required dependencies

```bash
sudo apt-get install build-essential cmake pkg-config git
sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install libopenscenegraph-dev
```
2. install DART v6.3.0

```bash
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.3.0
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

### Install PIP things

You should first activate virtualenv.
```bash
virtualenv /path/to/venv --python=python3
source /path/to/venv/bin/activate
```
- pytorch(https://pytorch.org/)

```bash
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision
```

- numpy, matplotlib

```bash
pip3 install numpy matplotlib ipython
```


## How to compile and run

### Compile and Run

```bash
mkdir build
cd build
cmake .. 
make -j8
```

- Run Training
```bash
cd pyvs
source /path/to/virtualenv/
python3 LatentARTraining.py -motion basket_0
```

