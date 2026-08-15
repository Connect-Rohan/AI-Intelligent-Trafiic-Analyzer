from ultralytics import YOLO
import cv2
import os
import json


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"

VIDEOS = {
    "Route A": "videos/route_a.mp4",
    "Route B": "videos/route_b.mp4"
}

OUTPUT_FOLDER = "processed_videos"

# Process maximum 20 seconds from each video
PROCESS_SECONDS = 20

# Analyze traffic every 5 seconds
WINDOW_SECONDS = 5


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("YOLO model loaded successfully.")


# ============================================================
# VEHICLE CLASSES
# ============================================================

# COCO class IDs used by YOLO
VEHICLE_CLASSES = {

    2: "Cars",

    3: "Motorcycles",

    5: "Buses",

    7: "Trucks"
}


# ============================================================
# STORE ALL TRAFFIC ANALYSIS
# ============================================================

traffic_analysis = {}


# ============================================================
# PROCESS EACH ROUTE
# ============================================================

for route_name, video_path in VIDEOS.items():

    print("\n" + "=" * 40)
    print(f"PROCESSING {route_name}")
    print("=" * 40)


    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not os.path.exists(video_path):

        print(
            f"ERROR: Video not found: {video_path}"
        )

        traffic_analysis[route_name] = []

        continue


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"ERROR: Could not open {video_path}"
        )

        traffic_analysis[route_name] = []

        continue


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    video_duration = (
        total_frames / fps
    )


    # --------------------------------------------------------
    # ACTUAL PROCESSING DURATION
    # --------------------------------------------------------

    processing_duration = min(
        PROCESS_SECONDS,
        video_duration
    )


    max_frames = int(
        processing_duration * fps
    )


    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Video duration: "
        f"{video_duration:.2f} seconds"
    )

    print(
        f"Processing only: "
        f"{processing_duration:.2f} seconds"
    )

    print(
        f"Maximum frames: "
        f"{max_frames}"
    )


    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )


    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


    if not writer.isOpened():

        print(
            f"ERROR: Could not create "
            f"output video: {output_path}"
        )

        cap.release()

        traffic_analysis[route_name] = []

        continue


    # ========================================================
    # ANALYSIS WINDOWS
    # ========================================================

    route_windows = []


    # Current 5-second window
    window_start_seconds = 0

    window_end_seconds = WINDOW_SECONDS


    # Current window vehicle counts
    window_counts = {

        "Cars": 0,

        "Motorcycles": 0,

        "Buses": 0,

        "Trucks": 0
    }


    # Number of frames analyzed in current window
    window_frame_count = 0


    # Total frames processed
    frame_count = 0


    # ========================================================
    # PROCESS VIDEO FRAME BY FRAME
    # ========================================================

    while frame_count < max_frames:


        ret, frame = cap.read()


        if not ret:

            break


        frame_count += 1


        # ----------------------------------------------------
        # CURRENT VIDEO TIME
        # ----------------------------------------------------

        current_seconds = (
            frame_count / fps
        )


        # ====================================================
        # YOLO DETECTION
        # ====================================================

        # IMPORTANT:
        # We intentionally use model()
        # instead of model.track().
        #
        # This avoids the GMC / optical-flow
        # OpenCV error that happened on Route B.

        results = model(
            frame,
            verbose=False
        )


        # ====================================================
        # FRAME COUNTS
        # ====================================================

        frame_counts = {

            "Cars": 0,

            "Motorcycles": 0,

            "Buses": 0,

            "Trucks": 0
        }


        # ====================================================
        # PROCESS YOLO DETECTIONS
        # ====================================================

        for result in results:


            if result.boxes is None:
                continue


            for box in result.boxes:


                # --------------------------------------------
                # CLASS ID
                # --------------------------------------------

                class_id = int(
                    box.cls[0]
                )


                # --------------------------------------------
                # IGNORE NON-VEHICLES
                # --------------------------------------------

                if class_id not in VEHICLE_CLASSES:

                    continue


                vehicle_type = (
                    VEHICLE_CLASSES[class_id]
                )


                # --------------------------------------------
                # COUNT VEHICLE
                # --------------------------------------------

                frame_counts[
                    vehicle_type
                ] += 1


                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

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


                # --------------------------------------------
                # LABEL
                # --------------------------------------------

                label = vehicle_type


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # ADD FRAME COUNTS TO CURRENT 5-SECOND WINDOW
        # ====================================================

        window_counts["Cars"] += (
            frame_counts["Cars"]
        )

        window_counts["Motorcycles"] += (
            frame_counts["Motorcycles"]
        )

        window_counts["Buses"] += (
            frame_counts["Buses"]
        )

        window_counts["Trucks"] += (
            frame_counts["Trucks"]
        )


        window_frame_count += 1


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (500, 145),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            route_name,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Time: {current_seconds:.1f}s",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Cars: {frame_counts['Cars']}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Motorcycles: {frame_counts['Motorcycles']}",
            (180, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Buses: {frame_counts['Buses']}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Trucks: {frame_counts['Trucks']}",
            (180, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SAVE PROCESSED FRAME
        # ====================================================

        writer.write(
            frame
        )


        # ====================================================
        # CHECK WHETHER 5-SECOND WINDOW IS COMPLETE
        # ====================================================

        window_complete = (
            current_seconds >= window_end_seconds
        )


        video_finished = (
            frame_count >= max_frames
        )


        if window_complete or video_finished:


            # =================================================
            # CALCULATE WINDOW AVERAGE
            # =================================================

            if window_frame_count > 0:

                average_cars = (
                    window_counts["Cars"]
                    / window_frame_count
                )

                average_motorcycles = (
                    window_counts["Motorcycles"]
                    / window_frame_count
                )

                average_buses = (
                    window_counts["Buses"]
                    / window_frame_count
                )

                average_trucks = (
                    window_counts["Trucks"]
                    / window_frame_count
                )

            else:

                average_cars = 0

                average_motorcycles = 0

                average_buses = 0

                average_trucks = 0


            # =================================================
            # ROUND AVERAGES
            # =================================================

            average_cars = round(
                average_cars
            )

            average_motorcycles = round(
                average_motorcycles
            )

            average_buses = round(
                average_buses
            )

            average_trucks = round(
                average_trucks
            )


            # =================================================
            # TOTAL VEHICLES
            # =================================================

            total_vehicles = (

                average_cars

                + average_motorcycles

                + average_buses

                + average_trucks
            )


            # =================================================
            # CREATE ANALYSIS RECORD
            # =================================================

            window_data = {

                "start": round(
                    window_start_seconds,
                    2
                ),

                "end": round(
                    min(
                        window_end_seconds,
                        processing_duration
                    ),
                    2
                ),

                "Cars": average_cars,

                "Motorcycles": average_motorcycles,

                "Buses": average_buses,

                "Trucks": average_trucks,

                "Total_Vehicles": total_vehicles
            }


            route_windows.append(
                window_data
            )


            # =================================================
            # PRINT RESULT
            # =================================================

            print(
                f"{route_name} | "
                f"{window_data['start']:.0f}-"
                f"{window_data['end']:.0f}s | "
                f"Cars: {average_cars} | "
                f"Motorcycles: {average_motorcycles} | "
                f"Buses: {average_buses} | "
                f"Trucks: {average_trucks} | "
                f"TOTAL: {total_vehicles}"
            )


            # =================================================
            # PREPARE NEXT WINDOW
            # =================================================

            window_start_seconds = (
                window_end_seconds
            )

            window_end_seconds += (
                WINDOW_SECONDS
            )


            window_counts = {

                "Cars": 0,

                "Motorcycles": 0,

                "Buses": 0,

                "Trucks": 0
            }


            window_frame_count = 0


    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()

    writer.release()


    # ========================================================
    # SAVE ROUTE ANALYSIS
    # ========================================================

    traffic_analysis[
        route_name
    ] = route_windows


    # ========================================================
    # ROUTE SUMMARY
    # ========================================================

    print()

    print(
        f"✓ {route_name} complete"
    )

    print(
        f"✓ Processed video saved: "
        f"{output_path}"
    )

    print(
        f"✓ Analysis windows: "
        f"{len(route_windows)}"
    )


# ============================================================
# SAVE JSON
# ============================================================

analysis_path = os.path.join(
    OUTPUT_FOLDER,
    "traffic_analysis.json"
)


with open(
    analysis_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        traffic_analysis,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 40)

print(
    "TRAFFIC VIDEO PROCESSING COMPLETE"
)

print("=" * 40)

print()

print(
    "Analysis saved to:"
)

print(
    analysis_path
)


print()


for route_name, windows in traffic_analysis.items():

    print(
        f"{route_name}: "
        f"{len(windows)} analysis windows"
    )


    for window in windows:

        print(
            f"  "
            f"{window['start']:.0f}-"
            f"{window['end']:.0f} sec → "
            f"{window['Total_Vehicles']} vehicles "
            f"("
            f"Cars: {window['Cars']}, "
            f"Motorcycles: {window['Motorcycles']}, "
            f"Buses: {window['Buses']}, "
            f"Trucks: {window['Trucks']}"
            f")"
        )