from video_live_analyzer import (
    open_video,
    analyze_video_window
)


# ==========================================
# ROUTE A
# ==========================================

cap_a, fps_a, duration_a = open_video(
    "videos/route_a.mp4"
)

print("\n==============================")
print("ROUTE A")
print("==============================")

print(
    f"FPS: {fps_a:.2f}"
)

print(
    f"Duration: {duration_a:.2f} seconds"
)


result_a = analyze_video_window(
    cap_a,
    fps_a,
    start_second=0,
    window_seconds=5
)

print(
    result_a
)

cap_a.release()


# ==========================================
# ROUTE B
# ==========================================

cap_b, fps_b, duration_b = open_video(
    "videos/route_b.mp4"
)

print("\n==============================")
print("ROUTE B")
print("==============================")

print(
    f"FPS: {fps_b:.2f}"
)

print(
    f"Duration: {duration_b:.2f} seconds"
)


result_b = analyze_video_window(
    cap_b,
    fps_b,
    start_second=0,
    window_seconds=5
)

print(
    result_b
)

cap_b.release()