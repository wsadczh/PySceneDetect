# -*- coding: utf-8 -*-
#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2021 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
""" ``scenedetect.backends.opencv`` Module

This module contains the :py:class:`VideoStreamCv2` class, which provides an OpenCV
based video decoder (based on cv2.VideoCapture).
"""

import math
from typing import Tuple, Union, Optional
import os.path

import cv2
from numpy import ndarray

from scenedetect.frame_timecode import FrameTimecode, MINIMUM_FRAMES_PER_SECOND_FLOAT
from scenedetect.platform import get_aspect_ratio, logger
from scenedetect.video_stream import VideoStream, SeekError, VideoOpenFailure


def open_video(path: Optional[str], device: Optional[int], framerate: Optional[float]):
    if path is not None and device is not None:
        raise ValueError("Only one of path or device can be specified!")
    if path is not None:
        return VideoStreamCv2(path_or_device=path, framerate=framerate)
    return VideoStreamCv2(path_or_device=device, framerate=framerate)


class VideoStreamCv2(VideoStream):
    """ OpenCV VideoCapture backend. """

    def __init__(self, path_or_device: Union[str, int], framerate: Optional[float] = None):
        """Open a new OpenCV backend.

        Arguments:
            path_or_device: Path to video, or device ID as integer.

            TODO: Split VideoStreamCv2 up into a child class VideoStreamCv2Device which overrides
            methods in VideoStreamCv2 rather than having to branch. Can then consider renaming this
            to VideoStreamCv2File, and also have one for image sequence if required (since that has
            implications for the seek method as well).
            """
        super().__init__()

        self._path_or_device = path_or_device
        self._is_device = isinstance(self._path_or_device, int)

        # Initialized in _open_capture:
        self._cap = None    # Reference to underlying cv2.VideoCapture object.
        self._frame_rate = None

        # VideoCapture state
        self._has_seeked = False
        self._has_grabbed = False

        self._open_capture(framerate)

    @property
    def capture(self) -> cv2.VideoCapture:
        """Returns reference to underlying VideoCapture object.

        Do not seek nor call the read/grab methods through the VideoCapture otherwise the
        VideoStreamCv2 object will be in an inconsistent state."""
        return self._cap

    #
    # VideoStream Methods/Properties
    #

    @property
    def frame_rate(self) -> float:
        """Framerate in frames/sec."""
        return self._frame_rate

    @property
    def path(self) -> str:
        """Video or device path."""
        if self._is_device:
            return "Device %d" % self._path_or_device
        return self._path_or_device

    @property
    def name(self) -> str:
        """Name of the video, without extension, or device."""
        if self._is_device:
            return self.path
        name = os.path.basename(self.path)
        last_dot_pos = name.rfind('.')
        if last_dot_pos >= 0:
            name = name[:last_dot_pos]
        return name

    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise.

        Always False if opening a device/webcam."""
        return not self._is_device

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return (math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def duration(self) -> Optional[FrameTimecode]:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        if self._is_device:
            return None
        return self.base_timecode + math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return get_aspect_ratio(self._cap)

    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp of the last frame which was
        decoded by calling `read` with advance=True.

        This method will always return 0 (e.g. be equal to `base_timecode`) if no frames
        have been `read`."""
        if self.frame_number < 1:
            return self.base_timecode
        return self.base_timecode + (self.frame_number - 1)

    @property
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in milliseconds.
        The first frame has a time of 0.0 ms.

        This method will always return 0.0 if no frames have been `read`."""
        return self._cap.get(cv2.CAP_PROP_POS_MSEC)

    @property
    def frame_number(self) -> int:
        """Current position within stream in frames as an int.

        1 indicates the first frame was just decoded by the last call to `read` with advance=True,
        whereas 0 indicates that no frames have been `read`.

        This method will always return 0 if no frames have been `read`."""
        return math.trunc(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek(self, target: Union[FrameTimecode, float, int]):
        """Seek to the given timecode. If given as a frame number, represents the current seek
        pointer (e.g. if seeking to 0, the next frame decoded will be the first frame).

        This means that, for 1-based indices (first frame is frame #1), the target frame number
        needs to be converted to 0-based by subtracting one.  For example, if we want to seek to
        the first frame, we call seek(0) followed by read(). At this point, frame_number will be 1.
        If we call seek(4) (the *fifth* frame) and then read(), frame_number will be 5.

        Seeking past the end of video shall be equivalent to seeking to the last frame.

        Not supported if the VideoStream is a device/camera.  Untested with web streams.

        Arguments:
            target: Target position in video stream to seek to. Interpreted based on type.
              If FrameTimecode, backend can seek using any representation (preferably native when
              VFR support is added).
              If float, interpreted as time in seconds.
              If int, interpreted as frame number, starting from 0.
        Raises:
            SeekError if an unrecoverable error occurs while seeking, or seeking is not
            supported (either by the backend entirely, or if the input is a stream).
        """
        if self._is_device:
            raise SeekError("Cannot seek if input is a device!")
        if target < 0:
            raise ValueError("Target seek position cannot be negative!")

        # Have to seek one behind and call grab() after to that the VideoCapture
        # returns a valid timestamp when using CAP_PROP_POS_MSEC.
        target_frame_cv2 = (self.base_timecode + target).get_frames()
        if target_frame_cv2 > 0:
            target_frame_cv2 -= 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_cv2)
        if target > 0:
            self._cap.grab()
            self._has_grabbed = True
            self._has_seeked = False
        else:
            self._has_grabbed = False
            self._has_seeked = True

    def reset(self):
        """ Close and re-open the VideoStream (should be equivalent to calling `seek(0)`). """
        self.seek(0)
        self._cap.release()
        self._open_capture(self._frame_rate)

    def read(self, decode: bool = True, advance: bool = True) -> Union[ndarray, bool]:
        """ Return next frame (or current if advance = False), or False if end of video.

        If decode = False, a boolean indicating if the next frame was advanced or not is returned.

        If decode and advance are both False, equivalent to a no-op, and the return value should
        be discarded/ignored.
        """
        if not self._cap.isOpened():
            return False
        if advance:
            self._has_grabbed = self._cap.grab()
            self._has_seeked = False
        if decode and self._has_grabbed:
            _, frame = self._cap.retrieve()
            return frame
        return self._has_grabbed

    #
    # Private Methods
    #

    def _open_capture(self, framerate: Optional[float] = None):
        """Opens capture referenced by this object and resets internal state."""
        if self._is_device and self._path_or_device < 0:
            raise ValueError("Invalid/negative device ID specified.")
        # Check if files exist if passed video file is not an image sequence
        # (checked with presence of % in filename) or not a URL (://).
        if not self._is_device and not ('%' in self._path_or_device
                                        or '://' in self._path_or_device):
            if not os.path.exists(self._path_or_device):
                raise IOError("Video file not found.")

        cap = cv2.VideoCapture(self._path_or_device)
        if not cap.isOpened():
            raise VideoOpenFailure("isOpened() returned False when opening OpenCV VideoCapture!")

        # Display a warning if the video codec type seems unsupported (#86).
        if int(abs(cap.get(cv2.CAP_PROP_FOURCC))) == 0:
            logger.error(
                "Video codec detection failed, output may be incorrect.\nThis could be caused"
                " by using an outdated version of OpenCV, or using codecs that currently are"
                " not well supported (e.g. VP9).\n"
                "As a workaround, consider re-encoding the source material before processing.\n"
                "For details, see https://github.com/Breakthrough/PySceneDetect/issues/86")

        # Ensure the framerate is correct to avoid potential divide by zero errors. This can be
        # addressed in the PyAV backend if required since it supports integer timebases.
        if not framerate:
            framerate = cap.get(cv2.CAP_PROP_FPS)
            if framerate < MINIMUM_FRAMES_PER_SECOND_FLOAT:
                raise VideoOpenFailure(
                    "Unable to obtain video framerate! Check the file/device/stream, or set the"
                    " `framerate` to assume a given framerate.")

        self._cap = cap
        self._frame_rate = framerate
        self._has_seeked = False
        self._has_grabbed = False