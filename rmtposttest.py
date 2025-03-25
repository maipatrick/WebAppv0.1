import cv2
from pathlib import Path
from rtmlib import PoseTracker, BodyWithFeet



def main(video_path):
    video_path = Path(video_path)
    output_path = video_path.with_name(video_path.stem + '_output.mp4')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Set up pose tracker  ('lightweight', 'balanced', 'performance') 
    pose_tracker = PoseTracker(
        BodyWithFeet,
        det_frequency=1,
        mode="balanced",
        backend="onnxruntime",
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        tracking=True,
        to_openpose=False
    )

    # Get keypoints structure
    model = BodyWithFeet
    #keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id is not None]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    # get the current frame number
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        keypoints_all, scores_all = pose_tracker(frame)
        # store for each frame the keypoints_all in a list oe dict
        for keypoints in keypoints_all:
            X = keypoints[:, 0]
            Y = keypoints[:, 1]
            # draw circles with cv2.circle
            for i, (x, y) in enumerate(zip(X, Y)):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            #draw_skel(frame, [X], [Y], model)
            #draw_keypts(frame, [X], [Y], scores_all, cmap_str='RdYlGn')
        print(f"\rProcessing frame {frame_number}/{total_number_of_frames}...", end="")
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Output saved to: {output_path}")

#main(r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\linear\Video.mp4")
main (r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\linear\7. SSL 12 kg.MOV")