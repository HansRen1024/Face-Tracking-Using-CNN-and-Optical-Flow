### 2021.06.06 Pytorch version: https://github.com/HansRen1024/C-OF. Code is released (20210816).

### 2021.05.10 Pytorch version will be released soon.

### 2019.01.25 UPDATE

Some guys are not familiar with ncnn, and compile this repo with an error of **could not find net.h**. Please move to https://github.com/Tencent/ncnn to find what is ncnn and how to install it. Download the version we used from https://github.com/Tencent/ncnn/archive/refs/tags/20180830.zip.

### 2018.12.20 IMPORTANT UPDATE

I extremely optimized the code, all useless contents were removed. Right now, tracking speed is approximate **3ms**. Besides, I deleted initialization funtion and OpenCV 3.x is supported now. 

RK3399 20+ ms/frame

### Face-Tracking-Using-Optical-Flow-and-CNN

I optimized OpenTLD making it run faster and better for face tracking.

This version of TLD is faster and more stable than that in OpenCV. I delete some funtions to make it run faster. What is more, use RNet to judge the face that TLD produced to avoid TLD tracking a wrong target. In order to get a stable bounding box, I fix the width and height that MTCNN provides. Running time on my PC(Intel® Xeon(R) CPU E5-2673 v3 @ 2.40GHz × 48) is about 16ms(MTCNN, ncnn), 30ms(TLD initialization), 10ms(TLD tracking) on an image of 320*240 resolution. Besides, MTCNN can be replaced by PCN or any other face/object detection algorithms.

中文介绍地址：https://blog.csdn.net/renhanchi/article/details/85089265

### Installing

~~OpenCV 2.4.X is required!~~(Now OpenCV 3.x is supported)

Install ncnn firstly, and reset ncnn's include and lib pathes in CMakeLists.txt.

```shell
mkdir build
cd build
cmake ..
make
cd ..
./demo
```

### Examples

![image](https://github.com/HansRen1024/Face-Tracking-Based-on-OpenTLD-and-RNet/blob/master/example/saved_1.gif)

![image](https://github.com/HansRen1024/Face-Tracking-Based-on-OpenTLD-and-RNet/blob/master/example/saved_.gif)

### References

https://github.com/Tencent/ncnn

https://github.com/CongWeilin/mtcnn-caffe

https://github.com/alantrrs/OpenTLD
