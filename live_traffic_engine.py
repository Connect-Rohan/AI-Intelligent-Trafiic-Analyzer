from ultralytics import YOLO
import cv2
import os


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"

WINDOW_SECONDS = 5

VEHICLE_CLASSES = {
    2: "Cars",
    3: "Motorcycles",
    5: "Buses",
    7: "Trucks"
}


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)


# ============================================================
# OPEN VIDEO
# ============================================================

def get_video_info(video_path):

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps

    cap.release()

    return fps, total_frames, duration


# ============================================================
# ANALYZE ONE 5-SECOND WINDOW
# ============================================================

def analyze_video_window(
    video_path,
    start_second,
    window_seconds=WINDOW_SECONDS
):

    fps, total_frames, duration = get_video_info(
        video_path
    )

    if start_second >= duration:
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    start_frame = int(
        start_second * fps
    )

    end_frame = int(
        (start_second + window_seconds) * fps
    )

    end_frame = min(
        end_frame,
        total_frames
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )


    # ========================================================
    # UNIQUE TRACKED VEHICLES
    # ========================================================

    tracked_vehicles = {}

    frame_count = 0


    # ========================================================
    # PROCESS WINDOW
    # ========================================================

    while True:

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if current_frame >= end_frame:
            break

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1


        # ====================================================
        # YOLO TRACKING
        #
        # GMC IS DISABLED.
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml"
        )


        for result in results:

            if result.boxes is None:
                continue


            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                if class_id not in VEHICLE_CLASSES:
                    continue


                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                if box.id is None:
                    continue

                track_id = int(
                    box.id[0]
                )


                vehicle_type = VEHICLE_CLASSES[
                    class_id
                ]


                # ------------------------------------------------
                # SAVE UNIQUE VEHICLE
                # ------------------------------------------------

                tracked_vehicles[track_id] = vehicle_type


    cap.release()


    # ========================================================
    # COUNT UNIQUE VEHICLES
    # ========================================================

    counts = {

        "Cars": 0,

        "Motorcycles": 0,

        "Buses": 0,

        "Trucks": 0
    }


    for vehicle_type in tracked_vehicles.values():

        counts[vehicle_type] += 1


    total_vehicles = sum(
        counts.values()
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "Cars": counts["Cars"],

        "Motorcycles":
            counts["Motorcycles"],

        "Buses": counts["Buses"],

        "Trucks": counts["Trucks"],

        "Total_Vehicles":
            total_vehicles,

        "Start_Second":
            start_second,

        "End_Second":
            min(
                start_second + window_seconds,
                duration
            ),

        "Frames_Processed":
            frame_count,

        "FPS":
            fps,

        "Video_Duration":
            duration
    }