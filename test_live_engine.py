from live_traffic_engine import analyze_video_window


ROUTE_A_VIDEO = "videos/route_a.mp4"
ROUTE_B_VIDEO = "videos/route_b.mp4"


print()
print("==============================")
print("LIVE TRAFFIC ENGINE TEST")
print("==============================")


# ============================================================
# ROUTE A
# ============================================================

print()
print("ROUTE A")
print("==============================")


route_a = analyze_video_window(
    ROUTE_A_VIDEO,
    start_second=0,
    window_seconds=5
)


print(route_a)


# ============================================================
# ROUTE B
# ============================================================

print()
print("ROUTE B")
print("==============================")


route_b = analyze_video_window(
    ROUTE_B_VIDEO,
    start_second=0,
    window_seconds=5
)


print(route_b)


print()
print("==============================")
print("TEST COMPLETE")
print("==============================")