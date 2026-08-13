from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("videos/traffic.mp4")

# Vehicle classes in the COCO dataset
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

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Counters
    counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    # Process detections
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id in vehicle_classes:
                vehicle_name = vehicle_classes[class_id]
                counts[vehicle_name] += 1

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Vehicle label
                cv2.putText(
                    frame,
                    vehicle_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Total vehicles
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
        f"Total Vehicles: {total}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Show video
    cv2.imshow("AI Traffic Vehicle Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()