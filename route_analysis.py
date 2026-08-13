from ultralytics import YOLO
import cv2
import csv
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "yolo11n.pt"

VIDEOS = {
    "Route A": "videos/route_a.mp4",
    "Route B": "videos/route_b.mp4"
}

OUTPUT_FOLDER = "data"

# Save one observation every 5 seconds
INTERVAL_SECONDS = 5


# ==========================================
# CREATE DATA FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# LOAD YOLO
# ==========================================

model = YOLO(MODEL_PATH)


# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ==========================================
# ANALYZE ROUTE
# ==========================================

def analyze_route(route_name, video_path):

    print("\n========================================")
    print(f"ANALYZING {route_name}")
    print("========================================")

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():

        print(f"ERROR: Cannot open {video_path}")

        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_number = 0

    # Number of frames in one observation window
    interval_frames = int(
        fps * INTERVAL_SECONDS
    )

    # Track IDs seen during current interval
    interval_vehicle_ids = {
        "Car": set(),
        "Motorcycle": set(),
        "Bus": set(),
        "Truck": set()
    }

    # Track all vehicles throughout the video
    all_vehicle_ids = set()

    traffic_data = []

    while True:

        ret, frame = video.read()

        if not ret:
            break

        frame_number += 1

        # ==========================================
        # YOLO TRACKING
        # ==========================================

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        current_counts = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0
        }

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in VEHICLE_CLASSES:
                    continue

                vehicle_type = VEHICLE_CLASSES[class_id]

                # ----------------------------------
                # TRACKING ID
                # ----------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                    # Add this vehicle to interval
                    interval_vehicle_ids[
                        vehicle_type
                    ].add(track_id)

                    # Add route-wide ID
                    all_vehicle_ids.add(
                        (vehicle_type, track_id)
                    )

                # ----------------------------------
                # COUNT CURRENTLY VISIBLE VEHICLES
                # ----------------------------------

                current_counts[
                    vehicle_type
                ] += 1

                # ----------------------------------
                # DRAW BOX
                # ----------------------------------

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

                # ----------------------------------
                # LABEL
                # ----------------------------------

                label = vehicle_type

                if box.id is not None:

                    label += f" ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # ==========================================
        # SAVE DATA EVERY 5 SECONDS
        # ==========================================

        if frame_number % interval_frames == 0:

            cars = len(
                interval_vehicle_ids["Car"]
            )

            motorcycles = len(
                interval_vehicle_ids["Motorcycle"]
            )

            buses = len(
                interval_vehicle_ids["Bus"]
            )

            trucks = len(
                interval_vehicle_ids["Truck"]
            )

            total = (
                cars
                + motorcycles
                + buses
                + trucks
            )

            time_seconds = frame_number / fps

            # --------------------------------------
            # TRAFFIC DENSITY
            # --------------------------------------

            # Prototype density score.
            # Later we will calculate this using
            # road/camera capacity.
            density = min(
                total / 50,
                1.0
            )

            # --------------------------------------
            # CONGESTION LABEL
            # --------------------------------------

            if density < 0.30:

                congestion = "Low"

            elif density < 0.60:

                congestion = "Moderate"

            elif density < 0.80:

                congestion = "High"

            else:

                congestion = "Critical"

            traffic_data.append({

                "Route": route_name,

                "Time_Seconds":
                    round(time_seconds, 2),

                "Cars": cars,

                "Motorcycles":
                    motorcycles,

                "Buses":
                    buses,

                "Trucks":
                    trucks,

                "Total_Vehicles":
                    total,

                "Density":
                    round(density, 2),

                "Congestion_Level":
                    congestion
            })

            # --------------------------------------
            # PRINT DATA
            # --------------------------------------

            print(
                f"{route_name} | "
                f"Time: {time_seconds:.1f}s | "
                f"Cars: {cars} | "
                f"Bikes: {motorcycles} | "
                f"Buses: {buses} | "
                f"Trucks: {trucks} | "
                f"Total: {total} | "
                f"Congestion: {congestion}"
            )

            # --------------------------------------
            # RESET INTERVAL
            # --------------------------------------

            interval_vehicle_ids = {

                "Car": set(),

                "Motorcycle": set(),

                "Bus": set(),

                "Truck": set()
            }

        # ==========================================
        # DISPLAY STATISTICS
        # ==========================================

        visible_total = sum(
            current_counts.values()
        )

        cv2.putText(
            frame,
            route_name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Visible Vehicles: {visible_total}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to stop",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Traffic Route Analysis",
            frame
        )

        # ==========================================
        # STOP
        # ==========================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # ==========================================
    # RELEASE VIDEO
    # ==========================================

    video.release()

    cv2.destroyAllWindows()

    # ==========================================
    # SAVE CSV
    # ==========================================

    output_file = os.path.join(

        OUTPUT_FOLDER,

        route_name.lower()
        .replace(" ", "_")
        + ".csv"
    )

    if traffic_data:

        fieldnames = traffic_data[0].keys()

        with open(
            output_file,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                traffic_data
            )

    # ==========================================
    # SUMMARY
    # ==========================================

    print("\n----------------------------------------")

    print(
        f"{route_name} COMPLETED"
    )

    print(
        f"Unique vehicles tracked: "
        f"{len(all_vehicle_ids)}"
    )

    print(
        f"Dataset rows: "
        f"{len(traffic_data)}"
    )

    print(
        f"Saved: {output_file}"
    )

    print("----------------------------------------")


# ==========================================
# PROCESS BOTH ROUTES
# ==========================================

for route_name, video_path in VIDEOS.items():

    analyze_route(
        route_name,
        video_path
    )


print("\n========================================")
print("ALL ROUTES COMPLETED")
print("========================================")