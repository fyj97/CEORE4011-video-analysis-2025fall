# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import cv2
import csv
import torch
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import (check_img_size, non_max_suppression,
                                  scale_coords, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device
from ultralytics import YOLO as YOLOv8
from speed_est.mapper import PixelMapper

def is_inside_roi(x, y, roi_polygon):
    return cv2.pointPolygonTest(roi_polygon, (x, y), False) >= 0

def detect(opt):
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.yolo_model, device=device, dnn=opt.dnn)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)
    model.model.float()
    model(torch.zeros(1, 3, *imgsz).to(device))

    mobility_model = YOLOv8("./best.pt")
    id_mobility_result = {}
    id_vote_record = defaultdict(list)
    id_start_frame = {}
    id_end_frame = {}

    wait_zone_status = {}
    wait_zone_records = defaultdict(list)

    categories = ['pedestrians', 'children_wo_disability', 'elderly_wo_disability', 'non_vulnerable', 'with_disability']

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(opt.deep_sort_model, device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET)

    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride, auto=pt and not model.jit)
    vid_path, vid_writer = None, None
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    roi_polygon = np.array([[579, 600], [1042, 581], [867, 436], [529, 452]], dtype=np.int32)
    mapper_points = roi_polygon.tolist()
    mapper_points[-2], mapper_points[-1] = mapper_points[-1], mapper_points[-2]
    mapper = PixelMapper(mapper_points)

    waiting_zone = np.array([[599, 716], [1278, 712], [1042, 581], [579, 600]], dtype=np.int32)
    mobility_zone = waiting_zone
    fps = 30

    real_csv_path = save_dir / 'roi_realworld_coords.csv'
    real_csv_file = open(real_csv_path, 'w', newline='')
    real_writer = csv.writer(real_csv_file)
    real_writer.writerow(['frame', 'id', 'real_x', 'real_y'])

    first_frame_saved = False

    # Get total frames for progress bar by opening the video file directly
    total_frames = None
    if os.path.isfile(opt.source):
        try:
            cap = cv2.VideoCapture(opt.source)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total_frames > 0:
                    print(f"Total frames: {total_frames}")
        except:
            pass

    for frame_idx, (path, img, im0s, vid_cap, _) in enumerate(tqdm(dataset, desc="Processing frames", total=total_frames)):
        if vid_cap:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)

        im0 = im0s.copy()
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment, visualize=False)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0])[0]

        if not first_frame_saved:
            cv2.polylines(im0, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.polylines(im0, [waiting_zone], isClosed=True, color=(255, 0, 255), thickness=2)
            cv2.polylines(im0, [mobility_zone], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(str(save_dir / 'first_frame_with_roi.jpg'), im0)
            first_frame_saved = True

        if len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            xywhs = xyxy2xywh(pred[:, 0:4])
            confs = pred[:, 4]
            clss = pred[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

            for output in outputs:
                x1, y1, x2, y2, track_id, cls = output
                id = int(track_id)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                bottom_center = (int((x1 + x2) / 2), int(y2))

                if id not in id_start_frame:
                    id_start_frame[id] = frame_idx
                id_end_frame[id] = frame_idx

                in_wait = is_inside_roi(*bottom_center, waiting_zone)
                if in_wait and not wait_zone_status.get(id, False):
                    wait_zone_status[id] = True
                    wait_zone_records[id].append({'entry': frame_idx})
                elif not in_wait and wait_zone_status.get(id, False):
                    wait_zone_status[id] = False
                    if wait_zone_records[id] and 'entry' in wait_zone_records[id][-1] and 'exit' not in wait_zone_records[id][-1]:
                        wait_zone_records[id][-1]['exit'] = frame_idx

                if is_inside_roi(*bottom_center, roi_polygon):
                    real_x, real_y = mapper.transform(list(bottom_center))
                    real_writer.writerow([frame_idx, id, round(real_x, 2), round(real_y, 2)])

                in_mobility = is_inside_roi(*bottom_center, mobility_zone)
                if in_mobility and id not in id_mobility_result:
                    crop = im0[y1:y2, x1:x2]
                    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue
                    try:
                        results = mobility_model.predict(crop, imgsz=224, conf=0.001, verbose=False)
                        if not results or not hasattr(results[0], 'boxes'):
                            continue
                        if len(results[0].boxes) > 0:
                            cls_idx = int(results[0].boxes.cls[0].item())
                            cls_name = categories[cls_idx] if cls_idx < len(categories) else 'unknown'
                        else:
                            cls_name = 'unknown'
                        id_vote_record[id].append(cls_name)
                        if len(id_vote_record[id]) >= 5:
                            final_class = Counter(id_vote_record[id]).most_common(1)[0][0]
                            id_mobility_result[id] = final_class
                    except Exception as e:
                        print(f"[EXCEPTION] Mobility model failed for ID {id}: {e}")
                        continue

                label = f'id:{id} [{id_mobility_result.get(id, "...")}]'
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            deepsort.increment_ages()

        if opt.save_vid:
            if vid_path != path:
                vid_path = path
                w, h = im0.shape[1], im0.shape[0]
                out_path = str(save_dir / Path(path).name)
                os.makedirs(Path(out_path).parent, exist_ok=True)
                vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

    real_csv_file.close()

    with open(save_dir / 'id_mobility_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'class', 'start_frame', 'end_frame', 'start_time_sec', 'end_time_sec'])
        for id, cls in id_mobility_result.items():
            start = id_start_frame.get(id, -1)
            end = id_end_frame.get(id, -1)
            writer.writerow([id, cls, start, end, round(start/fps, 2), round(end/fps, 2)])

    with open(save_dir / 'waiting_area_times.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'entry_frame', 'exit_frame', 'entry_time_sec', 'exit_time_sec', 'duration_sec'])
        for id, records in wait_zone_records.items():
            for record in records:
                entry = record.get('entry', -1)
                exit = record.get('exit', entry)
                duration = round((exit - entry) / fps, 2)
                writer.writerow([id, entry, exit, round(entry/fps, 2), round(exit/fps, 2), duration])

    print(f"[DONE] Results saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5/weights/crowdhuman_yolov5m.pt')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='test.mp4')
    parser.add_argument('--output', type=str, default='inference/output')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true')
    parser.add_argument('--save-vid', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--project', default='runs/track')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--config_deepsort', type=str, default='deep_sort/configs/deep_sort.yaml')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    with torch.no_grad():
        detect(opt)
