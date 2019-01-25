### 2019.01.25 UPDATE

Many people do not familiar with NCNN, and compile this repo with an error that **could not find net.h**. Please move on https://github.com/Tencent/ncnn to find what is NCNN and how to install it.

### 2018.12.20 IMPORTANT UPDATE

I extremely optimized OpenTLD, all useless contents were removed. This time, tracking speed is about **3ms**. Besides, I deleted initialization funtion and OpenCV 3.x is supported now. 

RK3399 20+ms/frame

### Face-Tracking-Based-on-OpenTLD-and-RNet

I optimized OpenTLD making it run faster and better for face tracking.

This version of TLD below is faster and more stable than that in OpenCV. I delete some funtions to make it run faster. What is more, use RNet to judge the face that TLD produced to avoid TLD tracking a wrong target. In order to get a stable bounding box, I fix the width and height that MTCNN provides when initialising. Running speeds on my PC(Intel® Xeon(R) CPU E5-2673 v3 @ 2.40GHz × 48) are about 16ms(MTCNN, ncnn), 30ms(TLD initialization), 10ms(TLD tracking) for an image in 320*240 resolutions. Besides, MTCNN can be replaced by PCN or any other face detection algorithms.

中文地址：https://blog.csdn.net/renhanchi/article/details/85089265

### Installing

OpenCV 2.4.X is required!(Now OpenCV 3.x is supported)

Install NCNN firstly, and reset your NCNN path in CMakeLists.txt.

```shell
mkdir build
cd build
cmake ..
make
cd ..
./demo
```

### Examples

![image](https://github.com/HandsomeHans/Face-Tracking-Based-on-OpenTLD-and-RNet/blob/master/example/saved_1.gif)

![image](https://github.com/HandsomeHans/Face-Tracking-Based-on-OpenTLD-and-RNet/blob/master/example/saved_.gif)

### References

https://github.com/Tencent/ncnn

https://github.com/CongWeilin/mtcnn-caffe

https://github.com/alantrrs/OpenTLD
