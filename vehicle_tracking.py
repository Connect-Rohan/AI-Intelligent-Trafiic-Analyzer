from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("videos/traffic.mp4")

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:

    ret, frame = video.read()

    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # Current frame counts
    counts = {
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

            # Ignore objects that aren't vehicles
            if class_id not in vehicle_classes:
                continue

            vehicle_name = vehicle_classes[class_id]

            counts[vehicle_name] += 1

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{vehicle_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Total vehicles currently visible
    total = sum(counts.values())

    # Display statistics
    cv2.putText(
        frame,
        f"Cars: {counts['Car']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {counts['Motorcycle']}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {counts['Bus']}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {counts['Truck']}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles in Frame: {total}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Show video
    cv2.imshow(
        "AI Traffic Vehicle Tracking",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()