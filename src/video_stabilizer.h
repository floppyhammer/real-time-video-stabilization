#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

class VideoStabilizer {
public:
    VideoStabilizer() = default;

    cv::Mat stabilize(cv::Mat prev_frame, cv::Mat cur_frame);

private:
    cv::Mat prev_grey;
    cv::Mat cur_grey;

    int k = 1;

    cv::Mat last_xform;
};
