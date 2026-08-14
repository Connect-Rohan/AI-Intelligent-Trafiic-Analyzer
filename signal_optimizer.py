# ==========================================
# SIGNAL OPTIMIZER
# ==========================================

TOTAL_SIGNAL_TIME = 90


def optimize_signals(route_a, route_b):

    total = (
        route_a["Total_Vehicles"] +
        route_b["Total_Vehicles"]
    )

    if total == 0:

        return {
            "Route A": 45,
            "Route B": 45
        }

    share_a = (
        route_a["Total_Vehicles"] / total
    )

    share_b = (
        route_b["Total_Vehicles"] / total
    )

    green_a = round(
        share_a * TOTAL_SIGNAL_TIME
    )

    green_b = round(
        share_b * TOTAL_SIGNAL_TIME
    )

    return {

        "Route A": {
            "Traffic_Share": share_a,
            "Green_Time": green_a
        },

        "Route B": {
            "Traffic_Share": share_b,
            "Green_Time": green_b
        }

    }