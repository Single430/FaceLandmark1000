import torch
import onnx
import onnxruntime
import numpy as np
from itertools import product as product
from math import ceil
import cv2


class FaceDetector(object):
    def __init__(self):
        self.model_path = r'model/FaceDetector.onnx'
        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)
        self.ort_session = onnxruntime.InferenceSession(self.model_path)
        self.cfg = self.config()
        self.conf_threshold = 0.5
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.vis_threshold = 0.6
        self.image_size = (640, 640)

    def run(self, image):
        ori_height, ori_width = image.shape[:2]
        processed_image, scale, img_height, img_width = self.preprocess(image)
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(processed_image)}
        locations, confidences, landmarks = self.ort_session.run(None, ort_inputs)
        detections, landmarks = self.postprocess(processed_image, locations, confidences, landmarks, scale, img_height,
                                                 img_width)

        detections[:, 0] = detections[:, 0] * ori_width / self.image_size[0]
        detections[:, 1] = detections[:, 1] * ori_height / self.image_size[1]
        detections[:, 2] = detections[:, 2] * ori_width / self.image_size[0]
        detections[:, 3] = detections[:, 3] * ori_height / self.image_size[1]

        return detections, landmarks

    def show_result(self, image, detections):
        for d in detections:
            if self.vis_threshold > d[4]:
                continue
            image = cv2.rectangle(image, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 2)
        cv2.imshow('', image)
        cv2.waitKey(1)

    def preprocess(self, image):
        image = cv2.resize(image, self.image_size)
        img = np.float32(image)
        img_height, img_width, _ = img.shape
        scale = torch.Tensor([img_width, img_height, img_width, img_height])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.detach()
        img = img.to('cuda')
        scale = scale.to('cuda')
        return img, scale, img_height, img_width

    def postprocess(self, image, locations, confidences, landmarks, scale, img_height, img_width):
        priorbox = PriorBox(self.cfg, image_size=self.image_size)
        priors = priorbox.forward()
        priors = priors.to('cuda')
        resize = 1
        prior_data = priors.data
        locations = torch.from_numpy(locations).to('cuda')
        confidences = torch.from_numpy(confidences).to('cuda')
        landmarks = torch.from_numpy(landmarks).to('cuda')
        boxes = self.decode(locations.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = confidences.squeeze(0).data.cpu().numpy()[:, 1]
        landmarks = self.decode_landmarks(landmarks.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2]])
        scale1 = scale1.to('cuda')
        landmarks = landmarks * scale1 / resize
        landmarks = landmarks.cpu().numpy()
        inds = np.where(scores > self.conf_threshold)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(detections, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        detections = detections[keep, :]
        landmarks = landmarks[keep]

        # keep top-K faster NMS
        detections = detections[:self.keep_top_k, :]
        landmarks = landmarks[:self.keep_top_k, :]

        # detections = np.concatenate((detections, landmarks), axis=1)
        return detections, landmarks

    def decode(self, loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landmarks(self, pre, priors, variances):
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)
        return landms

    def config(self):
        cfg = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
        }

        return cfg

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    image = cv2.imread('data/1.jpg')
    handle = FaceDetector()
    handle.run(image)
