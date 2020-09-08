import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import os
import time
import math
import threading
from gps import *

gpsd = None
class GpsPoller(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        global gpsd  # bring it in scope
        gpsd = gps(mode=WATCH_ENABLE)  # starting the stream of info
        self.current_value = None
        self.running = True  # setting the thread running to true

    def run(self):
        global gpsd
        while gpsp.running:
            gpsd.next()  # this will continue to loop and grab EACH set of gpsd info to clear the buffer

def readCoordinates():
    lat = gpsd.fix.latitude
    lon = gpsd.fix.longitude
    gps_time = str(gpsd.utc) + ' ' + str(gpsd.fix.time)
    speed = float("{0:.4f}".format(gpsd.fix.speed))
    alt = float("{0:.4f}".format(gpsd.fix.altitude))
    climb = float("{0:.4f}".format(gpsd.fix.climb))
    track = gpsd.fix.track
    sats = gpsd.satellites
    eps = gpsd.fix.eps
    epx = gpsd.fix.epx
    epv = gpsd.fix.epv
    ept = gpsd.fix.ept
    fixtype = gpsd.fix.mode

    if (math.isnan(lat)):
        lat = "NAN"
    else:
        lat = "%s " % lat

    if (math.isnan(lon)):
        lon = "NAN"
    else:
        lon = "%s " % lon

    if (math.isnan(speed)):
        speed = "NAN"
    else:
        speed = "%s km/h" % speed

    if (math.isnan(alt)):
        alt = "NAN"
    else:
        alt = "%s m" % alt

    if (math.isnan(climb)):
        climb = "NAN"
    else:
        climb = "%s m/s" % climb

    if (math.isnan(track)):
        track = "NAN"
    else:
        track = "%s" % track

    if (math.isnan(eps)):
        eps = "NAN"
    else:
        eps = "%s" % eps

    if (math.isnan(epx)):
        epx = "NAN"
    else:
        epx = "%s" % epx

    if (math.isnan(epv)):
        epv = "NAN"
    else:
        epv = "%s" % epv

    if (math.isnan(ept)):
        ept = "NAN"
    else:
        ept = "%s" % ept

    # sats_str = ','.join(sats)

    if fixtype == 1:
        fixtype = "No Fix"
    else:
        fixtype = "%sD" % fixtype

    coords = [gps_time, lat, lon, alt, speed, climb, track, eps, epx, epv, ept, fixtype]

    return coords

def detect_turn_signal(img0, opt, model, half, webcam, device):
    out, source, view_img, save_txt, imgsz, save_img= \
        opt.output, img0, False, False, opt.img_size, True
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    # Padded resize
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    vid_cap = None
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
#     t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#     t2 = time_synchronized()

    # Apply Classifier
#     if classify:
#         pred = apply_classifier(pred, modelc, img, img0)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # save target image
                target_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                cv2.imwrite('./target/signal.jpg'.format(int(xyxy[1])), target_img)
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

#         # Print time (inference + NMS)
#         print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
        if view_img:
            cv2.imshow(p, img0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        # cv2.imwrite('./target/test.jpg'.format(int(xyxy[1])), img0)
        return img0

def detect_lp_num_near(img0, opt, model, model_2, half, webcam, device):
    print('near')
    # global car_data
    out, source, view_img, save_txt, imgsz, save_img= \
        opt.output, img0, False, False, opt.img_size, True
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 固定車牌為紅色
    colors = [[0, 0, 255]]
#     img0 = cv2.GaussianBlur(img0, (3, 3), 0)
#     ret, img0 = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY)
#     cv2.imwrite('./target/result.jpg', img0)

    # Run inference
#     t0 = time.time()
    # Padded resize
#     a = 0
#     for i in img0.shape:
#         if i > a:
#             a = i
#     l = 640
#     r = 640 / a
#     img0 = cv2.resize(img0, None, fx=r, fy=r, interpolation=cv2.INTER_LINEAR)
#     ret, img0 = cv2.threshold(img0, 120, 255, cv2.THRESH_BINARY)
#     # I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#     img0 = cv2.GaussianBlur(img0, (33, 33), 0)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
#     img0 = cv2.filter2D(img0, -1, kernel=kernel)
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    vid_cap = None
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
#     cv2.imwrite('./target/lp_b_{}.jpg'.format(count), img0)
    # Inference
#     t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#     t2 = time_synchronized()

    # Apply Classifier
#     if classify:
#         pred = apply_classifier(pred, modelc, img, img0)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            r_dict = {}
            result = []
            conf_l = []
            for *xyxy, conf, cls in reversed(det):
                # save target image
                target_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                # cv2.imwrite('./target/lp.jpg'.format(int(xyxy[1])), target_img)
#                 if save_txt:  # Write to file
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

#                 if save_img or view_img:  # Add bbox to image
#                     label = '%s %.2f' % (names[int(cls)], conf)
#                     # plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=3)
#                     # cv2.imwrite('./target/result.jpg'.format(int(xyxy[1])), img0)
#                     result.append((float(xyxy[0]), names[int(cls)]))
#                     conf_l.append(conf)
                label = '%s %.2f' % (names[int(cls)], conf)
                result.append((float(xyxy[0]), names[int(cls)]))
                conf_l.append(conf)
            result_l = [s[1] for s in sorted(result, key=lambda x: x[0])]
            result_s = ''
            conf_avg = sum(conf_l) / len(conf_l)
            for r in result_l:
                result_s += r
#             print(result_s)
#             print(conf_avg)
            if len(result_s) in [6, 7] and conf_avg > 0.9:
                print('=========================='+result_s)
                print('==========================', conf_avg)
                return result_s, conf_avg
    fake_lp = '0'
    fake_conf = 0.0
    return fake_lp, fake_conf
    
def detect_lp_num(img0, opt, model, model_3, half, webcam, device):
    print('far')
    out, source, view_img, save_txt, imgsz, save_img= \
        opt.output, img0, False, False, opt.img_size, True
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 固定車牌為紅色
    colors = [[0, 0, 255]]
#     img0 = cv2.GaussianBlur(img0, (3, 3), 0)
#     ret, img0 = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY)
#     cv2.imwrite('./target/result.jpg', img0)

    # Run inference
#     t0 = time.time()
    # Padded resize
    a = 0
    for i in img0.shape:
        if i > a:
            a = i
    l = 640
    r = 640 / a
    img0 = cv2.resize(img0, None, fx=r, fy=r, interpolation=cv2.INTER_LINEAR)
    # ret, img0 = cv2.threshold(img0, 120, 255, cv2.THRESH_BINARY)
    # I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    img0 = cv2.GaussianBlur(img0, (33, 33), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    img0 = cv2.filter2D(img0, -1, kernel=kernel)
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    vid_cap = None
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    cv2.imwrite('./target/lp_b.jpg', img0)
    # count += 1
    # Inference
#     t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#     t2 = time_synchronized()

    # Apply Classifier
#     if classify:
#         pred = apply_classifier(pred, modelc, img, img0)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            r_dict = {}
            result = []
            conf_l = []
            for *xyxy, conf, cls in reversed(det):
                # save target image
                target_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                # cv2.imwrite('./target/lp.jpg'.format(int(xyxy[1])), target_img)
#                 if save_txt:  # Write to file
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

#                 if save_img or view_img:  # Add bbox to image
#                     label = '%s %.2f' % (names[int(cls)], conf)
#                     # plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=3)
#                     # cv2.imwrite('./target/result.jpg'.format(int(xyxy[1])), img0)
#                     result.append((float(xyxy[0]), names[int(cls)]))
#                     conf_l.append(conf)
                label = '%s %.2f' % (names[int(cls)], conf)
                result.append((float(xyxy[0]), names[int(cls)]))
                conf_l.append(conf)
            result_l = [s[1] for s in sorted(result, key=lambda x: x[0])]
            result_s = ''
            conf_avg = sum(conf_l) / len(conf_l)
            for r in result_l:
                result_s += r
#             print(result_s)
#             print(conf_avg)
            if len(result_s) in [6, 7] and conf_avg > 0.85:
                print('=========================='+result_s)
                print('==========================', conf_avg)
                return result_s, conf_avg
    fake_lp = '0'
    fake_conf = 0.0
    return fake_lp, fake_conf
    
def detect_lp(img0, opt, model, model_2, model_3, half, webcam, device, far=True):
    out, source, view_img, save_txt, imgsz, save_img= \
        opt.output, img0, False, False, opt.img_size, True
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 固定車牌為紅色
    colors = [[0, 0, 255]]

    # Run inference
#     t0 = time.time()
    # Padded resize
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    vid_cap = None
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
#     t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#     t2 = time_synchronized()

    # Apply Classifier
#     if classify:
#         pred = apply_classifier(pred, modelc, img, img0)
    
    label_num = '0'
    conf_num = 0.0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            test = ''
            for *xyxy, conf, cls in reversed(det):
                # save target image
                target_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                if test != cls:
                    cv2.imwrite('./target/lp.jpg'.format(int(xyxy[1])), target_img)
                    test = cls
                if far:
                    label_num, conf_num = detect_lp_num(img0=target_img, opt=opt, model=model_2, model_3=model_3, half=half, webcam=webcam, device=device)
                else:
                    label_num, conf_num = detect_lp_num_near(img0=target_img, opt=opt, model=model_3, model_2=model_2, half=half, webcam=webcam, device=device)
#                 if save_txt:  # Write to file
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     with open(txt_path + '.txt', 'a') as f:
#                         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

#                 if save_img or view_img:  # Add bbox to image
#                     if len(label_num) >= 6:
#                         label = '%s %.2f' % (label_num, conf_num)
#                     else:
#                         label = '%s %.2f' % (names[int(cls)], conf)
#                     plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=1)
                if len(label_num) >= 6:
                    label = '%s %.2f' % (label_num, conf_num)
                else:
                    label = '%s %.2f' % (names[int(cls)], conf)
                # plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=1)
            return xyxy, label, label_num, conf_num

#         # Print time (inference + NMS)
#         print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
#         if view_img:
#             cv2.imshow(p, img0)
#             if cv2.waitKey(1) == ord('q'):  # q to quit
#                 raise StopIteration
        # cv2.imwrite('./target/test.jpg'.format(int(xyxy[1])), img0)
        return False, False, label_num, conf_num

def detect_car_type(opt, model, model_1, model_2, model_3, half, webcam, device):
    out, source, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.view_img, opt.save_txt, opt.img_size
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
#         modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors_num = [[0, 255, 0]]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    count_json = 1
    count = 0
    gps_col = ['gps_time', 'lat', 'lon', 'alt', 'speed', 'climb', 'track', 'eps', 'epx', 'epv', 'ept', 'fixtype']
    for path, img, im0s, vid_cap in dataset:
#         print(img.shape)
#         print(im0s.shape)
        json_data = {'iot_id': '', 'timestamp': '', 'gps_time': '','lat': '', 'lon': '', 'alt': '',
                     'speed': '', 'climb': '', 'track': '', 'eps': '', 'epx': '', 'epv': '',
                    'ept': '', 'fixtype': '', 'jpg_path': '', 'cars': []}
        json_data['iot_id'] = opt.iot_id
        # json data
        timestamp = time.time()
        if count_json == 1 and count == 0:
            time_name = timestamp
        json_data['timestamp'] = timestamp
        # call gps
        coords = readCoordinates()
        for g_i, g_data in enumerate(gps_col):
            json_data[g_data] = coords[g_i]
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
#         if classify:
#             pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    car_data_col = ['cm', 'cm_conf', 'lp', 'lp_cof', 'xywh', 'color']
                    # save target image
                    target_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    # car ratio
                    ratio = (target_img.shape[0]*target_img.shape[1])/(im0.shape[0]*im0.shape[1])
#                     print(ratio)
#                     print(target_img.shape)
#                     print(((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1]))))
#                     x1, y1 = ((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])))
#                     im_shape = im0.shape
#                     print('imhape: ', im_shape)
#                     print(((im_shape[1]/2), 0))
#                     x2, y2 = ((im_shape[1]/2), 0)
#                     # slope
#                     m = slope(x1, y1, x2, y2)
#                     print(m)
                    cv2.imwrite('./target/car.jpg'.format(int(xyxy[1])), target_img)
#                     print('im0_0', im0.shape)
                    if ratio < 0.15:
                        # im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], label_num, conf_num = detect_lp(img0=target_img, opt=opt, model=model_1, model_2=model_2, model_3=model_3, half=half, webcam=webcam, device=device)
                        xyxy_lp, label_lp, label_num, conf_num = detect_lp(img0=target_img, opt=opt, model=model_1, model_2=model_2, model_3=model_3, half=half, webcam=webcam, device=device)
                    else:
                        # im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], label_num, conf_num = detect_lp(img0=target_img, opt=opt, model=model_1, model_2=model_2, model_3=model_3, half=half, webcam=webcam, device=device, far=False)
                        xyxy_lp, label_lp, label_num, conf_num = detect_lp(img0=target_img, opt=opt, model=model_1, model_2=model_2, model_3=model_3, half=half, webcam=webcam, device=device)
#                     print('im0_1', im0.shape)
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

#                     if save_img or view_img:  # Add bbox to image
#                         label = '%s %.2f' % (names[int(cls)], conf)
#                         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    if label_lp:
                        plot_one_box(xyxy_lp, target_img, label=label_lp, color=colors_num[0], line_thickness=1)
                        im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = target_img
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    place = [round(float(x), 2) for x in xywh]
                    car_data = [names[int(cls)], round(float(conf), 2), label_num, round(float(conf_num), 2), place, '']
                    car_data_dict = {car_data_col[i]: j for i, j in enumerate(car_data)}
                    json_data['cars'].append(car_data_dict)
                    # print('wh = ', xywh)

            # Print time (inference + NMS)
            t2 = time_synchronized()
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'images':
#                     cv2.imwrite(save_path, im0)
#                 else:
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer

#                         fourcc = 'mp4v'  # output video codec
#                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
#                     vid_writer.write(im0)
        # save json
        if count > 100:
            count_json += 1
            count = 0
        jpg_name = '1.jpg'.format(timestamp)
        json_data['jpg_path'] = jpg_name
        cv2.imwrite('/home/jn/hunter_data/{}/{}/jpg/{}'.format(opt.iot_id, opt.dir, jpg_name), im0)
        print('json=', json_data)
        with open('/home/jn/hunter_data/{}/{}/{}_{}.json'.format(opt.iot_id, opt.dir, opt.iot_id, count_json), 'a', encoding='utf-8') as f:
            f.write(str(json_data) + '\n')
        count += 1
        
#     if save_txt or save_img:
#         print('Results saved to %s' % Path(out))
#         if platform.system() == 'Darwin' and not opt.update:  # MacOS
#             os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def set_model(save_img=False):
    out, source, weights_0, weights_1, weights_2, weights_3,view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights_0, opt.weights_1, opt.weights_2, opt.weights_3, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_0 = attempt_load(weights_0, map_location=device)  # car_type model
    model_1 = attempt_load(weights_1, map_location=device)  # lp model
    model_2 = attempt_load(weights_2, map_location=device)  # lp_content_far model
    model_3 = attempt_load(weights_3, map_location=device)  # lp_content_near model
    # time.sleep(200)
    detect_car_type(opt=opt, model=model_0, model_1=model_1, model_2=model_2, model_3=model_3, half=half, webcam=webcam, device=device)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_0', nargs='+', type=str, default='weights/s_car_type_best.pt', help='model.pt path(s)')
    parser.add_argument('--weights_1', nargs='+', type=str, default='weights/s_lp_best.pt', help='model.pt path(s)')
    parser.add_argument('--weights_2', nargs='+', type=str, default='weights/s_lp_num_blur_best.pt', help='model.pt path(s)')
    parser.add_argument('--weights_3', nargs='+', type=str, default='weights/lp_num_best_b.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--dir', default='test')
    parser.add_argument('--iot-id', default='test')
    opt = parser.parse_args()
    print(opt)
    gpsp = GpsPoller()
    gpsp.start()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                set_model()
                strip_optimizer(opt.weights)
        else:
            set_model()
