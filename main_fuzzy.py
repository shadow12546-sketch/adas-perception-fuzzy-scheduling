import argparse
import csv
import time
from pathlib import Path

import cv2

from modules.object_detection import ObjectDetector
from modules.lane_detection import LaneDetector
from modules.depth_estimation import DepthEstimator
from modules.bird_eye_view import BirdEyeView
from utils.drawing import (
    draw_detections,
    draw_lanes,
    draw_fps,
    draw_info_panel,
    create_side_by_side,
)
from utils.video import VideoProcessor


def trapezoid(x, a, b, c, d):
    """Return the membership value of x in a trapezoidal fuzzy set."""
    if x < a or x > d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if a <= x < b:
        return 1.0 if a == b else (x - a) / (b - a)

    if c < x <= d:
        return 1.0 if c == d else (d - x) / (d - c)

    return 0.0


def fuzzy_depth_interval(motion, load):
    """
    Mamdani fuzzy controller.

    Inputs:
        motion: normalized frame-to-frame motion, from 0 to 1
        load: frame processing time divided by the target frame time

    Output:
        Number of frames until the next depth update, from 1 to 5
    """
        # Motion ranges based on the observed scores:
    # median was about 0.095; the 90th percentile was about 0.125.
    motion_low = trapezoid(motion, 0.00, 0.00, 0.075, 0.09)
    motion_medium = trapezoid(motion, 0.075, 0.09, 0.12, 0.14)
    motion_high = trapezoid(motion, 0.12, 0.14, 1.00, 1.00)

    # Load ranges based on depth-update frames:
    # typical processing time was about 0.13 sec, or 1.3 times the 0.1 sec budget.
    load_low = trapezoid(load, 0.0, 0.0, 0.65, 0.9)
    load_medium = trapezoid(load, 0.65, 0.9, 1.05, 1.25)
    load_high = trapezoid(load, 1.05, 1.25, 4.0, 4.0)


    # Rules map to frequent, normal, or slow depth updates.
    rules = [
        ("slow", min(motion_low, load_low)),
        ("normal", min(motion_low, load_medium)),
        ("slow", min(motion_low, load_high)),
        ("frequent", min(motion_medium, load_low)),
        ("normal", min(motion_medium, load_medium)),
        ("slow", min(motion_medium, load_high)),
        ("frequent", motion_high),
    ]

    output_values = [1.0 + i * 0.05 for i in range(81)]
    aggregated = [0.0] * len(output_values)

    for output_set, rule_strength in rules:
        for i, value in enumerate(output_values):
            if output_set == "frequent":
                membership = trapezoid(value, 1.0, 1.0, 1.5, 2.5)
            elif output_set == "normal":
                membership = trapezoid(value, 1.5, 2.5, 3.5, 4.5)
            else:
                membership = trapezoid(value, 3.5, 4.5, 5.0, 5.0)

            # Mamdani implication uses min; outputs combine with max.
            aggregated[i] = max(
                aggregated[i],
                min(rule_strength, membership),
            )

    total_membership = sum(aggregated)
    if total_membership == 0:
        return 3

    # Centroid defuzzification converts the fuzzy output to one interval.
    crisp_interval = sum(
        value * membership
        for value, membership in zip(output_values, aggregated)
    ) / total_membership

    return max(1, min(5, round(crisp_interval)))


def run_pipeline(
    video_path: str,
    output_path: str = None,
    show_window: bool = True,
    skip_depth: bool = False,
):
    print("ADAS Perception Pipeline with Fuzzy Depth Scheduling")

    print("Loading object detector...")
    detector = ObjectDetector()

    print("Loading lane detector...")
    lane_detector = LaneDetector()

    print("Loading depth estimator...")
    depth_estimator = DepthEstimator() if not skip_depth else None

    print("Loading bird's eye view...")
    bev = BirdEyeView(output_size=(400, 600))

    print("\nAll modules loaded. Starting pipeline...\n")

    video = VideoProcessor(video_path, output_path)
    if not video.open():
        return

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    # The current KITTI videos are 10 FPS: 0.1 seconds per input frame.
    # Change this if you use videos with a different frame rate.
    target_frame_time = 0.1
    last_processing_time = target_frame_time

    previous_small_gray = None

    # The first frame always runs depth estimation.
    # Later frames use the interval chosen after the previous depth update.
    depth_interval = 3
    next_depth_frame = 1

    fuzzy_log = []

    while True:
        ret, frame = video.read_frame()
        if not ret:
            print("\nVideo processing complete.")
            break

        # Start timing before per-frame processing.
        frame_start = time.perf_counter()

        # Estimate motion from small grayscale frames to limit overhead.
        small_frame = cv2.resize(frame, (320, 96))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if previous_small_gray is None:
            motion_score = 0.0
        else:
            motion_score = (
                cv2.absdiff(gray, previous_small_gray).mean() / 255.0
            )

        previous_small_gray = gray

        # High motion forces depth estimation on the current frame.
        motion_high_membership = trapezoid(
            motion_score, 0.12, 0.14, 1.00, 1.00
        )
        high_motion = (
            motion_high_membership >= 0.85 or motion_score >= 0.14
        )

        frame_count += 1

        # Calculate displayed FPS every 10 frames.
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_timer
            fps = 10 / elapsed if elapsed > 0 else 0.0
            fps_timer = time.time()

        # Step 1 — Object Detection (crop bottom 15% to exclude the hood)
        detection_frame = frame[: int(frame.shape[0] * 0.85), :]
        detections = detector.detect(detection_frame)

        # Step 2 — Lane Detection
        lane_result = lane_detector.detect(frame)

        # Step 3 — Persistent fuzzy-controlled depth schedule
        depth_map = None
        run_depth_now = False

        if depth_estimator:
            run_depth_now = (
                frame_count >= next_depth_frame
                or high_motion
            )

            if run_depth_now:
                depth_map = depth_estimator.estimate(frame)

        # Step 4 — Draw results
        output_frame = frame.copy()
        output_frame = draw_lanes(output_frame, lane_result)
        output_frame = draw_detections(
            output_frame,
            detections,
            depth_map,
            frame_count,
        )
        output_frame = draw_fps(output_frame, fps)
        output_frame = draw_info_panel(output_frame, detections)

        # Step 5 — Bird's-eye view
        bev_frame = bev.transform(frame)
        bev_frame = bev.draw_objects_on_bev(
            bev_frame,
            detections,
            frame.shape,
        )

        # Step 6 — Combine views
        combined = create_side_by_side(output_frame, bev_frame)

        # Set up the video writer on the first frame.
        if frame_count == 1 and output_path:
            height, width = combined.shape[:2]
            video.setup_writer(width, height)

        video.write_frame(combined)

        # Measure full frame-processing time, including depth if it ran.
        last_processing_time = time.perf_counter() - frame_start

        # When depth runs, choose and hold the interval until the next update.
        if run_depth_now:
            load_ratio = last_processing_time / target_frame_time
            load_ratio = min(load_ratio, 4.0)

            depth_interval = fuzzy_depth_interval(
                motion_score,
                load_ratio,
            )
            next_depth_frame = frame_count + depth_interval

        # Save per-frame controller values for later analysis.
        fuzzy_log.append(
            [
                frame_count,
                round(motion_score, 5),
                round(last_processing_time, 5),
                depth_interval,
                high_motion,
                run_depth_now,
            ]
        )

        if show_window:
            cv2.imshow("ADAS Perception Pipeline", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        if frame_count % 30 == 0:
            print(
                f"Processed {frame_count}/{video.total_frames} frames "
                f"| FPS: {fps:.1f}"
            )

    video.release()

    # Save the CSV log next to the output video.
    if output_path:
        log_path = Path(output_path).with_name(
            Path(output_path).stem + "_log.csv"
        )
    else:
        log_path = Path("fuzzy_log.csv")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "frame",
                "motion_score",
                "processing_seconds",
                "depth_interval",
                "high_motion_safety_override",
                "depth_ran",
            ]
        )
        writer.writerows(fuzzy_log)

    print(f"\nTotal frames processed: {frame_count}")
    if output_path:
        print(f"Output saved to: {output_path}")
    print(f"Fuzzy schedule log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADAS Pipeline with Fuzzy Depth Scheduling"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/traffic.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/result_fuzzy.mp4",
        help="Path to save output",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without display window",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation for faster FPS",
    )
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        show_window=not args.no_window,
        skip_depth=args.skip_depth,
    )
 