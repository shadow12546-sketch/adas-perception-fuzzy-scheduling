import cv2
import time
import argparse

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


def run_pipeline(
    video_path: str,
    output_path: str = None,
    show_window: bool = True,
    skip_depth: bool = False,
):
    print("ADAS Perception Pipeline - Baseline Profiling")

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

    stage_seconds = {
        "object_detection": 0.0,
        "lane_detection": 0.0,
        "depth_estimation": 0.0,
        "visualization_and_bev": 0.0,
        "video_write": 0.0,
    }
    depth_calls = 0
    profile_start = time.perf_counter()

    while True:
        ret, frame = video.read_frame()
        if not ret:
            print("\nVideo processing complete.")
            break

        frame_count += 1

        # Calculate FPS every 10 frames.
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_timer
            fps = 10 / elapsed if elapsed > 0 else 0.0
            fps_timer = time.time()

        # Step 1 — Object Detection (crop bottom 15% to exclude the hood)
        detection_frame = frame[: int(frame.shape[0] * 0.85), :]

        stage_start = time.perf_counter()
        detections = detector.detect(detection_frame)
        stage_seconds["object_detection"] += (
            time.perf_counter() - stage_start
        )

        # Step 2 — Lane Detection
        stage_start = time.perf_counter()
        lane_result = lane_detector.detect(frame)
        stage_seconds["lane_detection"] += (
            time.perf_counter() - stage_start
        )

        # Step 3 — Depth Estimation (baseline: every 3 frames)
        depth_map = None
        if depth_estimator and frame_count % 3 == 0:
            stage_start = time.perf_counter()
            depth_map = depth_estimator.estimate(frame)
            stage_seconds["depth_estimation"] += (
                time.perf_counter() - stage_start
            )
            depth_calls += 1

        # Time drawing and bird's-eye-view processing together.
        stage_start = time.perf_counter()

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

        bev_frame = bev.transform(frame)
        bev_frame = bev.draw_objects_on_bev(
            bev_frame,
            detections,
            frame.shape,
        )

        combined = create_side_by_side(output_frame, bev_frame)

        stage_seconds["visualization_and_bev"] += (
            time.perf_counter() - stage_start
        )

        # Time writer setup and frame writing.
        stage_start = time.perf_counter()

        if frame_count == 1 and output_path:
            height, width = combined.shape[:2]
            video.setup_writer(width, height)

        video.write_frame(combined)

        stage_seconds["video_write"] += (
            time.perf_counter() - stage_start
        )

        if show_window:
            cv2.imshow("ADAS Perception Pipeline - Baseline Profile", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        if frame_count % 30 == 0:
            print(
                f"Processed {frame_count}/{video.total_frames} frames "
                f"| FPS: {fps:.1f}"
            )

    video.release()

    print(f"\nTotal frames processed: {frame_count}")
    if output_path:
        print(f"Output saved to: {output_path}")

    total_seconds = time.perf_counter() - profile_start

    print("\nProfiling summary:")
    print(f"Total processing time: {total_seconds:.2f} seconds")
    print(f"Depth calls: {depth_calls}")

    for stage_name, seconds in stage_seconds.items():
        average = seconds / frame_count if frame_count else 0.0
        print(
            f"{stage_name}: {seconds:.2f} seconds total, "
            f"{average:.4f} seconds/frame"
        )

    measured_seconds = sum(stage_seconds.values())
    other_seconds = max(0.0, total_seconds - measured_seconds)
    print(
        f"Video reading and loop overhead: {other_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile the baseline ADAS perception pipeline"
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
        default="output/result_profile.mp4",
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
        help="Skip depth estimation",
    )
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        show_window=not args.no_window,
        skip_depth=args.skip_depth,
    )

