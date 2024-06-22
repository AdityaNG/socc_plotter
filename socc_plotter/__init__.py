import sys
import warnings

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict


def check_opencv_installation():
    try:
        import cv2

        # Check if 'opencv-python-headless' is installed
        try:
            pkg_resources.require("opencv-python-headless")
        except (DistributionNotFound, VersionConflict):
            warnings.warn("Could not find opencv-python-headless")
            # 'opencv-python-headless' is not found or version conflict,
            # check for 'opencv-python'
            try:
                pkg_resources.require("opencv-python")
                print("Warning: opencv-python (GUI version) is installed.")
                print(
                    "Consider uninstalling it and installing",
                    "opencv-python-headless for non-GUI environments.",
                )
            except (DistributionNotFound, VersionConflict):
                print(
                    "Error: Neither opencv-python-headless nor opencv-python",
                    "is installed.",
                )
                print(
                    "Please install opencv-python-headless by running: ",
                    "pip install opencv-python-headless",
                )
                sys.exit(1)

        # Additional check to ensure cv2 module is actually available
        try:
            version = cv2.__version__
            assert version is not None
        except AttributeError:
            print("Error: OpenCV version information is not accessible.")

    except ImportError:
        print("Error: OpenCV is not installed.")
        print(
            "Please install opencv-python-headless by running:",
            "pip install opencv-python-headless",
        )
        sys.exit(1)


check_opencv_installation()
