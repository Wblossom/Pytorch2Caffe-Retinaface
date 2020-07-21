import sys
sys.path.append("/home/xingduan/caffe_pyt/python")
import torch
import numpy as np
import caffe
from utils.box_utils import decode, decode_landm
from prior_box import PriorBox
import cv2 as cv
from config import cfg_mnet
import torch.nn.functional as F
from utils.nms.py_cpu_nms import py_cpu_nms
#model = torch.load("./trained_weights_our_dataset/mobilenet0.25_epoch_230.pth", map_location = "cpu")
net = caffe.Net("./caffe_models/FaceDetector_aoru_test.prototxt","./caffe_models/FaceDetector_aoru_test.caffemodel", caffe.TEST)
img_raw = cv.imread('test.jpg')
img_raw = cv.resize(img_raw, (1216, 768))
im = np.float32(img_raw)
im_height, im_width, _ = im.shape
scale = torch.Tensor([im.shape[1], im.shape[0], im.shape[1], im.shape[0]])
im -= (104, 117, 123)
im = im.transpose(2, 0, 1)[np.newaxis, :, :, :]
data = net.blobs['input0']
data.reshape(*im.shape)
data.data[...] = im

output = net.forward()
face_rpn_landmark_pred_stride8 = torch.from_numpy(output["face_rpn_landmark_pred_stride8"])
face_rpn_landmark_pred_stride16 = torch.from_numpy(output["face_rpn_landmark_pred_stride16"])
face_rpn_landmark_pred_stride32 = torch.from_numpy(output["face_rpn_landmark_pred_stride32"])
face_rpn_bbox_pred_stride8 = torch.from_numpy(output["face_rpn_bbox_pred_stride8"])
face_rpn_bbox_pred_stride16 = torch.from_numpy(output["face_rpn_bbox_pred_stride16"])
face_rpn_bbox_pred_stride32 = torch.from_numpy(output["face_rpn_bbox_pred_stride32"])
face_rpn_cls_prob_reshape_stride8 = torch.from_numpy(output["face_rpn_cls_prob_reshape_stride8"])
face_rpn_cls_prob_reshape_stride16 = torch.from_numpy(output["face_rpn_cls_prob_reshape_stride16"])
face_rpn_cls_prob_reshape_stride32 = torch.from_numpy(output["face_rpn_cls_prob_reshape_stride32"])
#face_rpn_cls_prob_reshape_stride8 = torch.from_numpy(output["softmax1"])
#face_rpn_cls_prob_reshape_stride16 = torch.from_numpy(output["softmax2"])
#face_rpn_cls_prob_reshape_stride32 = torch.from_numpy(output["softmax3"])
pred_ldmk = torch.cat([face_rpn_landmark_pred_stride8.view(1,20,-1), face_rpn_landmark_pred_stride16.view(1,20,-1), face_rpn_landmark_pred_stride32.view(1,20,-1)], dim = -1)
pred_bbox = torch.cat([face_rpn_bbox_pred_stride8.view(1,8,-1), face_rpn_bbox_pred_stride16.view(1,8,-1), face_rpn_bbox_pred_stride32.view(1,8,-1)], dim = -1)
pred_cla = torch.cat([face_rpn_cls_prob_reshape_stride8.view(1,4,-1), face_rpn_cls_prob_reshape_stride16.view(1,4,-1), face_rpn_cls_prob_reshape_stride32.view(1,4,-1)], dim = -1)
print(pred_cla[0,:,0])
landms = pred_ldmk.permute(0,2,1).contiguous().view(1,-1,10)
loc = pred_bbox.permute(0,2,1).contiguous().view(1,-1,4)
pred_cla = torch.cat([pred_cla[:,0:4:2,:], pred_cla[:,1:5:2,:]], dim = 1)
pred_cla  = pred_cla.permute(0,2,1).contiguous().view(1,-1,2)
conf = pred_cla
print(conf[0,0,:])
priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
priors = priorbox.forward()
prior_data = priors.data

boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
boxes = boxes * scale / 1
boxes = boxes.cpu().numpy()

scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
scale1 = torch.Tensor([im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                        im.shape[3], im.shape[2], im.shape[3], im.shape[2],
                        im.shape[3], im.shape[2]])
landms = landms * scale1 / 1
landms = landms.cpu().numpy()

inds = np.where(scores > 0.02)[0]
boxes = boxes[inds]
landms = landms[inds]
scores = scores[inds]

order = scores.argsort()[::-1][:5000]
boxes = boxes[order]
landms = landms[order]
scores = scores[order]

dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
keep = py_cpu_nms(dets, 0.4)

dets = dets[keep, :]
landms = landms[keep]

dets = dets[:750, :]
landms = landms[:750, :]


dets = np.concatenate((dets, landms), axis=1)

for b in dets:
    if b[4] < 0.6:
        continue
    text = "{:.4f}".format(b[4])
    b = list(map(int, b))
    cv.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cx = b[0]
    cy = b[1] + 12

    cv.putText(img_raw, text, (cx, cy), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    cv.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    cv.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    cv.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    cv.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    cv.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
cv.imwrite("img_caffe.jpg", img_raw)
print("save over")
