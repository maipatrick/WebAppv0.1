import os
import cv2
import numpy as np
from rtmlib import PoseTracker, BodyWithFeet, draw_skeleton
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.signal import savgol_filter


def process_video_multi_person_rstLIB(input_path: str):
    assert os.path.exists(input_path), f"❌ File not found: {input_path}"

    # === Load video ===
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base, ext = os.path.splitext(input_path)
    #output_path = f"{base}_pose{ext}"
    #fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext.lower() != ".avi" else "XVID"))
    #writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #print(f"📹 Input: {input_path}")
    #print(f"💾 Output: {output_path}")

    # === RTMLib Pose Estimation ===
    pose_tracker = PoseTracker(
        BodyWithFeet,
        det_frequency=5,
        mode='balanced',
        backend='onnxruntime',
        device='cpu',
        tracking=False
    )

    # === Deep SORT Tracker ===
    tracker = DeepSort(max_age=30, n_init=3)
    pose_data = {}  # person_id: {coords: [], scores: []}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints_list, scores_list = pose_tracker(frame)
        keypoints = np.array(keypoints_list)  # (N, K, 2)
        scores = np.array(scores_list)        # (N, K)

        detections = []
        for i, kps in enumerate(keypoints):
            if np.isnan(kps).any():
                continue
            x1, y1 = np.min(kps, axis=0)
            x2, y2 = np.max(kps, axis=0)
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            conf = float(np.nanmean(scores[i]))
            detections.append((bbox, conf, {'keypoints': kps, 'score': scores[i]}))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            detection = t.det_class  # dict: {'keypoints', 'score'}
            if detection is None or 'keypoints' not in detection:
                continue

            kps = detection['keypoints']
            score = detection['score']

            # Save data for smoothing later
            # if track_id not in pose_data:
            #     pose_data[track_id] = {'coords': [], 'scores': []}
            # pose_data[track_id]['coords'].append(kps)
            # pose_data[track_id]['scores'].append(score)
            if track_id not in pose_data:
                pose_data[track_id] = {'coords': [], 'scores': [], 'frames': []}
            pose_data[track_id]['coords'].append(kps)
            pose_data[track_id]['scores'].append(score)
            pose_data[track_id]['frames'].append(frame_idx)

            # Draw skeleton + ID
            draw_skeleton(frame, np.array([kps]), np.array([score]), kpt_thr=0.1)
            l, t_, r, b = t.to_ltrb()
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t_) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #writer.write(frame)
        frame_idx += 1
        #if frame_idx % 30 == 0:
        #    print(f"🧠 Frame {frame_idx}/{total_frames}")

    cap.release()
    #writer.release()
    #print("✅ Saved video with skeletons:", output_path)

    # === Smooth keypoints ===
    #print("🔄 Smoothing landmarks with Savitzky-Golay filter...")

    for track_id, person in pose_data.items():
        coords = np.array(person['coords'])  # (T, K, 2)
        scores = np.array(person['scores'])  # (T, K)
        T, K, _ = coords.shape

        if T < 5:
            continue  # not enough frames

        for k in range(K):
            for d in range(2):  # x and y
                seq = coords[:, k, d]
                conf = scores[:, k]

                # Replace low confidence with NaN
                seq[conf < 0.3] = np.nan
                if np.all(np.isnan(seq)):
                    continue

                # Interpolate missing
                nans = np.isnan(seq)
                seq[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), seq[~nans])

                # Smooth with Savitzky-Golay
                window = min(7, len(seq) if len(seq) % 2 == 1 else len(seq)-1)
                if window >= 5:
                    coords[:, k, d] = savgol_filter(seq, window_length=window, polyorder=2)

        pose_data[track_id]['coords'] = coords


    #print("🎉 Done smoothing. Everything worked!")
    capture_length = frame_idx / (1/fps)
    return pose_data, fps, total_frames, capture_length
    

pose_data, fps, total_frames, capture_length = process_video_multi_person_rstLIB(r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\Testdata\linear\USE.mp4")