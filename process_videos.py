from ultralytics import YOLO
import cv2
import os


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "yolo11n.pt"

VIDEOS = {
    "Route A": "videos/route_a.mp4",
    "Route B": "videos/route_b.mp4"
}

OUTPUT_FOLDER = "processed_videos"

# Process only this many seconds
PROCESS_SECONDS = 5


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)


# ==========================================
# VEHICLE CLASSES
# ==========================================

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ==========================================
# PROCESS VIDEOS
# ==========================================

for route_name, video_path in VIDEOS.items():

    print("\n================================")
    print(f"PROCESSING {route_name}")
    print("================================")

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"ERROR: Could not open {video_path}"
        )

        continue


    # ======================================
    # VIDEO INFORMATION
    # ======================================

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30


    # ======================================
    # LIMIT FRAMES
    # ======================================

    max_frames = int(
        fps * PROCESS_SECONDS
    )


    print(
        f"FPS: {fps:.1f}"
    )

    print(
        f"Processing only "
        f"{PROCESS_SECONDS} seconds "
        f"({max_frames} frames)"
    )


    # ======================================
    # OUTPUT FILE
    # ======================================

    filename = (
    route_name
    .lower()
    .replace(" ", "_")
    + "_processed.avi"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    # ======================================
    # VIDEO WRITER
    # ======================================

    fourcc = cv2.VideoWriter_fourcc(
    *"XVID"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


    # ======================================
    # PROCESS FRAMES
    # ======================================

    frame_count = 0

    while frame_count < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1


        # ==================================
        # YOLO TRACKING
        # ==================================

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )


        vehicle_count = 0


        # ==================================
        # DRAW DETECTIONS
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
                    VEHICLE_CLASSES[class_id]
                )


                vehicle_count += 1


                # --------------------------
                # BOUNDING BOX
                # --------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # --------------------------
                # LABEL
                # --------------------------

                label = vehicle_type


                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                    label += (
                        f" ID:{track_id}"
                    )


                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ==================================
        # TOP INFORMATION PANEL
        # ==================================

        cv2.rectangle(
            frame,
            (0, 0),
            (450, 90),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            route_name,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Vehicles: {vehicle_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ==================================
        # SAVE FRAME
        # ==================================

        writer.write(
            frame
        )


        # ==================================
        # PROGRESS
        # ==================================

        if frame_count % 30 == 0:

            percentage = (
                frame_count
                / max_frames
                * 100
            )

            print(
                f"Progress: "
                f"{percentage:.0f}%"
            )


    # ======================================
    # RELEASE
    # ======================================

    cap.release()

    writer.release()


    print(
        f"✓ Saved: {output_path}"
    )


print("\n================================")
print("ALL ROUTES PROCESSED")
print("================================")