import sys
sys.path.append("/home/xingduan/caffe_pyt/python")
import torch
import numpy as np
import caffe
#model = torch.load("./trained_weights_our_dataset/mobilenet0.25_epoch_140.pth", map_location = "cpu")
#net = caffe.Net("FaceDetector_aoru_test.prototxt", caffe.TEST)

model = torch.load("./torch_models/mobilenet0.25_epoch_150.pth", map_location = "cpu")
net = caffe.Net("./caffe_models/FaceDetector_aoru_test.prototxt", caffe.TEST)
import cv2 as cv
import numpy as np
to_save_keys = net.params.keys()
ori_keys = model.keys()
i = 0
# export LD_LIBRARY_PATH=/home/xingduan/miniconda2/envs/anti_spoofing_pyt/lib:$LD_LIBRARY_PATH
for key_and_type in to_save_keys:
    key = "_".join(key_and_type.split("_")[:-1])
    tp = key_and_type.split("_")[-1]

    if tp == "conv":
        # weight
        print(key_and_type)
        net.params[key_and_type][0].data[:] = model["module." + key + ".weight"].cpu().numpy()
        #print(model[key + ".weight"].cpu().numpy())
        i+=1
        # bias
        if (key + ".bias") in ori_keys:
            net.params[key_and_type][1].data[:] = model["module." + key + ".bias"].cpu().numpy()
            i+=1
    elif tp == "bn":
        print(key_and_type)
        #print(net.params[key_and_type][2].data[:].shape, model[key + ".running_mean"].cpu().numpy().shape)
        # weight
        net.params[key_and_type][0].data[:] = model["module." + key + ".running_mean"].cpu().numpy()
        net.params[key_and_type][1].data[:] = model["module." + key + ".running_var"].cpu().numpy()
        net.params[key_and_type][2].data[:] = 1
        # scale
        #net.params[key_and_type][0].data[:] = model[key + ".weight"].view(1, -1, 1, 1).cpu().numpy()
        #net.params[key_and_type][1].data[:] = model[key + ".bias"].view(1, -1, 1, 1).cpu().numpy()
        #net.params[key_and_type][2].data[:] = model[key + ".running_mean"].view(1, -1, 1, 1).cpu().numpy()
        #net.params[key_and_type][3].data[:] = model[key + ".running_var"].view(1, -1, 1, 1).cpu().numpy()
        i+=4
    elif tp == "scale":
        print(key_and_type)
        net.params[key_and_type][0].data[:] = model["module." + key + ".weight"].cpu().numpy()
        net.params[key_and_type][1].data[:] = model["module." + key + ".bias"].cpu().numpy()
    elif tp == "deconv":
        print(key_and_type)
        # weight
        net.params[key_and_type][0].data[:] = model["module." + key + ".weight"].cpu().numpy()
        i+=1
        # bias
        if (key + ".bias") in ori_keys:
           net.params[key_and_type][1].data[:] = model["module." + key + ".bias"].cpu().numpy()
           i+=1
    else:
        print(key_and_type , "no", "*" * 10)
net.params["face_rpn_landmark_pred_stride8"][0].data[:] = model["module.LandmarkHead.0.conv1x1.weight"].cpu().numpy()
net.params["face_rpn_landmark_pred_stride8"][1].data[:] = model["module.LandmarkHead.0.conv1x1.bias"].cpu().numpy()
net.params["face_rpn_landmark_pred_stride16"][0].data[:] = model["module.LandmarkHead.1.conv1x1.weight"].cpu().numpy()
net.params["face_rpn_landmark_pred_stride16"][1].data[:] = model["module.LandmarkHead.1.conv1x1.bias"].cpu().numpy()
net.params["face_rpn_landmark_pred_stride32"][0].data[:] = model["module.LandmarkHead.2.conv1x1.weight"].cpu().numpy()
net.params["face_rpn_landmark_pred_stride32"][1].data[:] = model["module.LandmarkHead.2.conv1x1.bias"].cpu().numpy()
net.params["face_rpn_bbox_pred_stride8"][0].data[:] = model["module.BboxHead.0.conv1x1.weight"].cpu().numpy()
net.params["face_rpn_bbox_pred_stride8"][1].data[:] = model["module.BboxHead.0.conv1x1.bias"].cpu().numpy()
net.params["face_rpn_bbox_pred_stride16"][0].data[:] = model["module.BboxHead.1.conv1x1.weight"].cpu().numpy()
net.params["face_rpn_bbox_pred_stride16"][1].data[:] = model["module.BboxHead.1.conv1x1.bias"].cpu().numpy()
net.params["face_rpn_bbox_pred_stride32"][0].data[:] = model["module.BboxHead.2.conv1x1.weight"].cpu().numpy()
net.params["face_rpn_bbox_pred_stride32"][1].data[:] = model["module.BboxHead.2.conv1x1.bias"].cpu().numpy()
img_raw = cv.imread('/home/xingduan/xa/retinaface/Pytorch_Retinaface/data/widerface/val/images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_240.jpg', cv.IMREAD_COLOR)
x = np.ones((1,3,640,640), dtype = np.float32)
im = np.float32(img_raw)
#im -= (104, 117, 123)
im = im.transpose(2, 0, 1)[np.newaxis, :, :, :]
data = net.blobs['input0']
data.reshape(*x.shape)
data.data[...] = x 
y = net.forward()
print(y)
net.save("./caffe_models/FaceDetector_aoru_test.caffemodel")
