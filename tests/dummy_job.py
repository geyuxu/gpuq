"""Dummy job script for testing gpuq."""
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", default="unnamed")
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    print(f"[START] {args.job_name} (duration={args.duration}s)")
    for i in range(args.duration):
        print(f"  tick {i+1}/{args.duration}")
        time.sleep(1)

    if args.fail:
        print(f"[FAIL] {args.job_name}")
        sys.exit(1)

    print(f"[DONE] {args.job_name}")


if __name__ == "__main__":
    main()
