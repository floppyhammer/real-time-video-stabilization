#ifndef VIDEO_STAB_H
#define VIDEO_STAB_H

#include <iostream>
#include <opencv2/opencv.hpp>

class VideoStab {
public:
    VideoStab();

    cv::Mat stabilize(cv::Mat prev_frame, cv::Mat cur_frame);

private:
    cv::Mat prev_frame_gray;
    cv::Mat cur_frame_gray;

    int k = 1;

    const int HORIZONTAL_BORDER_CROP = 30;

    cv::Mat smoothedMat;
    cv::Mat affine;

    cv::Mat smoothedFrame;
    cv::Mat croppedFrame;

    double dx;
    double dy;
    double da;
    double ds_x;
    double ds_y;

    double sx;
    double sy;

    double scaleX;
    double scaleY;
    double theta;
    double transX;
    double transY;

    double diff_scaleX;
    double diff_scaleY;
    double diff_transX;
    double diff_transY;
    double diff_theta;

    double errscaleX;
    double errscaleY;
    double errtheta;
    double errtransX;
    double errtransY;

    double Q_scaleX;
    double Q_scaleY;
    double Q_theta;
    double Q_transX;
    double Q_transY;

    double R_scaleX;
    double R_scaleY;
    double R_theta;
    double R_transX;
    double R_transY;

    double sum_scaleX;
    double sum_scaleY;
    double sum_theta;
    double sum_transX;
    double sum_transY;

    // Kalman Filter implementation
    void kalman_filter(double *scaleX, double *scaleY, double *theta, double *transX, double *transY);
};

#endif// VIDEO_STAB_H
