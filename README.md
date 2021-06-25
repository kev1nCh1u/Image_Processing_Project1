# Image_Processing_Project1

![Alt text](img/surf_left_image_keypoints.jpg?raw=true "Title")
![Alt text](img/surf_good_matches.jpg?raw=true "Title")

## start
    mkdir build
    cd build
    cmake ..
    make
    ./ImageProject1

## python install
    pip install numpy
    pip install opencv-python
    pip install opencv-contrib-python

## cpp install
https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html

    cmake -GNinja -DOPENCV_ENABLE_NONFREE:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master

    ninja

    sudo apt install libgtk2.0-dev

    sudo apt install pkg-config

## git ignore defore
    git rm -rf --cached .

## tools
vcpkg is a cross-platform open source package manager by Microsoft.
https://github.com/microsoft/vcpkg

OpenMP(OMP)
https://docs.microsoft.com/zh-tw/cpp/parallel/openmp/reference/openmp-functions?view=msvc-160
