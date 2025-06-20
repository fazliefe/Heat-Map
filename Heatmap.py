import argparse
import time
from pathlib import Path

import cv2
#if you do not have any error with OMP do not use it
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from deep_sort_tracking_id import xyxy_to_xywh,xyxy_to_tlwh,compute_color_for_labels,\
    draw_border,UI_box
parser = argparse.ArgumentParser()
opt = parser.parse_args()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
count_up=0
count_down=0
object_counter={}
object_counter1={}
#line= [(77, 935), (1819,927)]
line=[(77,935),(1819,927)]
classes_to_filter=[0]
import time


# Converts bounding box from xyxy format to xywh format
def xyxy_to_xywh(*xyxy): #4
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_bottom = max([xyxy[1].item(), xyxy[3].item()])
    bbox_height = abs(xyxy[1].item() - xyxy[3].item())
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())*0.65
    bbox_h = bbox_height / 8
    
    x_c = (bbox_left + bbox_w / 1)
    y_c = (bbox_bottom - bbox_height / 16)
    w = bbox_w 
    h = bbox_h
    
    return x_c, y_c, w, h        
        


# Loads class names from the given file
def load_classes(path):
    # Loads *.names file at 'path'
    with open("data/coco.names", 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


# Determines the direction of movement between two points
def get_direction(point1, point2):
    direction_str = ""
   #calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""
        
    #calculate x axis direction
    if point1[0] > point2[0]:
         direction_str += "East"
    elif point1[0] < point2[0]:
         direction_str += "West"
    else:
         direction_str += ""
    return direction_str


# Draws bounding boxes and object trails on the image
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= opt.trailslen)
          #speed_line_queue[id] = [] ##

        color = compute_color_for_labels(object_id[i])

        # add center to buffer
        data_deque[id].appendleft(center)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img


# Main function for video detection
def video_detection(save_img=False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
     ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
  
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                      max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                      nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                      use_cuda=True)
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    vcap = cv2.VideoCapture("rtsp://admin:Istech2021@192.168.1.188")

    # Determine dimensions of video
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    global global_img_np_array
    
    super_imposed_img = None
    global_img_np_array = np.ones([height, width], dtype = np.uint8)
    img_np_array = np.ones([height, width], dtype = int)

  # Load YOLO model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
      model = TracedModel(model, device, opt.img_size)

    if half:
      model.half()  # to FP16

  # Second-stage classifier
    classify = False
    if classify:
      modelc = load_classifier(name='resnet101', n=0)  # initialize
      modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=640)
    else:
        dataset = LoadImages(source, img_size=640)

    names = load_classes(names)
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
                
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
   
        # Iterate over the final detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p) 
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                xywh_bboxs = []
                confs = [] 
                oids = []
                        
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # Tracker inference
                outputs = deepsort.update(xywhs, confss, oids, im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    
                    draw_boxes(im0, bbox_xyxy, names , object_id , identities)

                    # Extract tracked object's bounding box coordinates
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        print("global image np array",global_img_np_array)
                        # Increment frequency counter for whole bounding box
                        global_img_np_array[y1:y2, x1:x2] += 2
                        print("global image np array",global_img_np_array)

                    # Heatmap array preprocessing
                    if global_img_np_array.size != 0:
                        global_img_np_array_norm = (global_img_np_array - global_img_np_array.min()) / (global_img_np_array.max() - global_img_np_array.min()) * 255
                    else:
                        global_img_np_array_norm = global_img_np_array

                    global_img_np_array_norm = global_img_np_array_norm.astype('uint8')
                    print("global image np array norm",global_img_np_array_norm)

                    # Apply Gaussian blur and draw heatmap
                    global_img_np_array_norm = cv2.GaussianBlur(global_img_np_array_norm,(9,9), 0)
                    heatmap_img = cv2.applyColorMap(global_img_np_array_norm, cv2.COLORMAP_JET)

                    # Overlay heatmap on video frames
                    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, im0, 0.5, 0)
                    cv2.imshow('Heatmap', super_imposed_img)
                    vid_writer.release()  # release previous video writer
            key = cv2.waitKey(1)  # Wait for 5 seconds
            if key == ord('q'):
                cv2.destroyAllWindows()
                    
            print(f'{s}Done. ({(1E3 * (t2-t1)):.1f}ms) Inference,({(1E3 * (t3-t2)):.1f}ms)NMS')
            
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, super_imposed_img)
                    print(f" The image with the result is saved in: {save_path}")
                    
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            global_img_np_array = np.ones([int(h),int(w)], dtype=np.uint32)

                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        
                    vid_writer.write(super_imposed_img)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/t*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        
    print(f'Done. ({time.time()-t0:.3f}s)')


# Class to define all the default argument settings for the detection process
class Args:
    def __init__(self):
        self.cfg = 'deep_sort.yaml'
        self.names = 'coco.names'
        self.weights = 'yolov7-tiny.pt'
        self.source = 'rtsp://admin:Istech2021@192.168.1.188'  # camera file path or image directory path
        #self.source = 'mall.mp4'  
        self.output = 'inference/output'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'gpu'
        self.view_img = False
        self.save_txt = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.exist_ok = False
        self.project = 'runs/detect'
        self.name = 'exp'
        self.nosave = False
        self.no_trace = False
        self.trailslen = False
        self.classes = classes_to_filter


opt = Args()

