#ifndef VIDEO_STAB_V2_H
#define VIDEO_STAB_V2_H

#include <iostream>
#include <opencv2/opencv.hpp>

class VideoStabV2 {
public:
    VideoStabV2() = default;

    cv::Mat stabilize(cv::Mat prev_frame, cv::Mat cur_frame);

private:
    cv::Mat prev_grey;
    cv::Mat cur_grey;

    int k = 1;

    cv::Mat last_xform;
};

#endif// VIDEO_STAB_V2_H
