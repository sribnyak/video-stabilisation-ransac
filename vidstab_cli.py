import argparse

from vidstab import VideoStabiliser, open_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str,
                        help="path to source video")
    parser.add_argument("-d", "--dst", type=str, default="out.mp4",
                        help="path to save the stabilised video to")
    parser.add_argument("-f", "--focal_length", type=int, default=705,
                        help="focal length")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()

    video = open_video(args.src)
    stabiliser = VideoStabiliser(focal_length=args.focal_length,
                                 verbose=args.verbose)
    stabiliser.stabilise(video)
    video.save(args.dst)
    print(f"Stabilised video is saved to {args.dst}")
