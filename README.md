For YOLOv3 object detection project I used pretrained YOLOv3-416 model. It works successfully.  But the weight file is too large to push to the Github repo hence I wanted to remark its link here.

|  Model |   Train|  Test  | mAP  | FLOPS |  FPS |  Cfg |Weights|
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------- |
|  YOLOv3 - 416 | COCO trainval  |  test-dev |  55.3 | 65.86 Bn  |   35|  [cfg](http://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg "cfg") |[weights](http://pjreddie.com/darknet/yolo/ "weights->YOLOv3-416")|

</br>

**Here are the results:**
</br>

![](https://snipboard.io/chgCus.jpg)
</br>

![](https://snipboard.io/OMkbgZ.jpg)

------------

##### Android Cam
I tried to detect object with yolov3 pretrained 416 model on android cam. It works and pretty accurate but there are some points that the model can't detect the precise parts on the frame. So, yolov3 is insufficient to detect an object or objects. There is an example below which is a book cover for OBLOMOV . We can clearly see that there is a man and his dog in a small room but the model can't detect the dog .

![](https://snipboard.io/z2YwvE.jpg)