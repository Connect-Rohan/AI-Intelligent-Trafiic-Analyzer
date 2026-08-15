import cv2
from ultralytics import YOLO


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "yolo11n.pt"

VEHICLE_CLASSES = {
    2: "Cars",
    3: "Motorcycles",
    5: "Buses",
    7: "Trucks"
}


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)


# ==========================================
# OPEN VIDEO
# ==========================================

def open_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    duration = (
        total_frames / fps
    )


    return (
        cap,
        fps,
        duration
    )


# ==========================================
# ANALYZE 5-SECOND VIDEO WINDOW
# ==========================================

def analyze_video_window(
    cap,
    fps,
    start_second,
    window_seconds=5
):

    # --------------------------------------
    # CALCULATE FRAME RANGE
    # --------------------------------------

    start_frame = int(
        start_second * fps
    )

    end_frame = int(
        (start_second + window_seconds)
        * fps
    )


    # --------------------------------------
    # MOVE VIDEO TO START
    # --------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )


    # --------------------------------------
    # VEHICLE COUNTERS
    # --------------------------------------

    max_counts = {

        "Cars": 0,

        "Motorcycles": 0,

        "Buses": 0,

        "Trucks": 0

    }


    frames_processed = 0


    # ======================================
    # PROCESS VIDEO
    # ======================================

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


        frames_processed += 1


        # ==================================
        # YOLO DETECTION ONLY
        #
        # IMPORTANT:
        # No model.track()
        # No GMC
        # No optical flow
        # ==================================

        results = model.predict(
            frame,
            verbose=False
        )


        frame_counts = {

            "Cars": 0,

            "Motorcycles": 0,

            "Buses": 0,

            "Trucks": 0

        }


        # ==================================
        # READ DETECTIONS
        # ==================================

        for result in results:

            if result.boxes is None:

                continue


            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )


                if class_id not in VEHICLE_CLASSES:

                    continue


                vehicle_type = (
                    VEHICLE_CLASSES[
                        class_id
                    ]
                )


                frame_counts[
                    vehicle_type
                ] += 1


        # ==================================
        # KEEP MAXIMUM OBSERVED COUNT
        #
        # This avoids adding the same
        # vehicle once for every frame.
        # ==================================

        for vehicle_type in max_counts:

            if (
                frame_counts[
                    vehicle_type
                ]
                >
                max_counts[
                    vehicle_type
                ]
            ):

                max_counts[
                    vehicle_type
                ] = frame_counts[
                    vehicle_type
                ]


    # ======================================
    # TOTAL VEHICLES
    # ======================================

    total_vehicles = sum(
        max_counts.values()
    )


    # ======================================
    # RETURN RESULT
    # ======================================

    return {

        "Cars":
            max_counts["Cars"],

        "Motorcycles":
            max_counts["Motorcycles"],

        "Buses":
            max_counts["Buses"],

        "Trucks":
            max_counts["Trucks"],

        "Total_Vehicles":
            total_vehicles,

        "Start_Second":
            start_second,

        "End_Second":
            start_second + window_seconds,

        "Frames_Processed":
            frames_processed

    }