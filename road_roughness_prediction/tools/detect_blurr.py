"""Detect blurring frames from video"""
import argparse
import cv2


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def show_filtered_image(path, thr=0., wait=1):
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        var_lap = variance_of_laplacian(gray)
        if var_lap > thr:
            print(f'var_lap {var_lap}')
            frame_ = cv2.resize(frame, None, fx=0.5, fy=0.5)
            cv2.imshow('frame', frame_)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--thr', type=float, default=0.)
    parser.add_argument('--wait', type=int, default=1)
    args = parser.parse_args()
    show_filtered_image(args.path, args.thr, args.wait)

if __name__ == '__main__':
    main()
