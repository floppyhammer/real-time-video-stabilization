#include "video_stab.h"
#include <cmath>

// Parameters for Kalman Filter
constexpr double Q1 = 0.004;
constexpr double R1 = 0.5;

constexpr double MAX_ROTATION_DEGREE = 5.0;
constexpr double MAX_TRANSLATION_FACTOR = 0.05;
constexpr double MAX_SCALE = 1.5;
constexpr double MIN_SCALE = 0.75;

VideoStab::VideoStab() {
    smoothedMat.create(2, 3, CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errtheta = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_theta = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_theta = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_theta = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 0;
    scaleY = 0;
    theta = 0;
    transX = 0;
    transY = 0;
}

cv::Mat VideoStab::stabilize(cv::Mat prev_frame, cv::Mat cur_frame) {
//    cv::Size new_size = cv::Size((int) round(original_prev_frame.cols / (float) down_sampling_factor), (int) round(original_prev_frame.rows / (float) down_sampling_factor));

//    cv::Mat prev_frame, cur_frame;
//    cv::resize(original_prev_frame, prev_frame, new_size);
//    cv::resize(cur_frame, cur_frame, new_size);

    cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cvtColor(cur_frame, cur_frame_gray, cv::COLOR_BGR2GRAY);

    int vert_border = HORIZONTAL_BORDER_CROP * prev_frame.rows / prev_frame.cols;

    std::vector<cv::Point2f> features1, features2;

    // Estimate the features in prev_frame_gray and cur_frame_gray
    if (1) {
        goodFeaturesToTrack(prev_frame_gray, features1, 200, 0.01, 30);
    } else {
        std::vector<cv::KeyPoint> keyPoints;
        cv::FAST(prev_frame_gray, keyPoints, 20);

        cv::KeyPoint::convert(keyPoints, features1);
    }

    if (features1.empty()) { return cur_frame; }

    std::vector<uchar> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(prev_frame_gray, cur_frame_gray, features1, features2, status, err);

    std::vector<cv::Point2f> goodFeatures1, goodFeatures2;
    goodFeatures1.reserve(features1.size());
    goodFeatures2.reserve(features2.size());

    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            goodFeatures1.push_back(features1[i]);
            goodFeatures2.push_back(features2[i]);
        }
    }

    if (goodFeatures1.empty()) {
        return cur_frame;
    }

    // All the parameters scale, angle, and translation are stored in affine
    affine = estimateAffinePartial2D(goodFeatures1, goodFeatures2);

    if (affine.empty()) {
        return cur_frame;
    }

    dx = affine.at<double>(0, 2);
    dy = affine.at<double>(1, 2);
    da = atan2(affine.at<double>(1, 0), affine.at<double>(0, 0));
    ds_x = affine.at<double>(0, 0) / cos(da);
    ds_y = affine.at<double>(1, 1) / cos(da);

    sx = ds_x;
    sy = ds_y;

    sum_transX += dx;
    sum_transY += dy;
    sum_theta += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;

    // Don't calculate the predicted state of Kalman Filter on 1st iteration
    if (k == 1) {
        k++;
    } else {
        kalman_filter(&scaleX, &scaleY, &theta, &transX, &transY);
    }

    diff_scaleX = scaleX - sum_scaleX;
    diff_scaleY = scaleY - sum_scaleY;
    diff_transX = transX - sum_transX;
    diff_transY = transY - sum_transY;
    diff_theta = theta - sum_theta;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_theta;

    // Creating the smoothed parameters matrix
    smoothedMat.at<double>(0, 0) = sx * cos(da);
    smoothedMat.at<double>(0, 1) = sx * -sin(da);
    smoothedMat.at<double>(1, 0) = sy * sin(da);
    smoothedMat.at<double>(1, 1) = sy * cos(da);

    smoothedMat.at<double>(0, 2) = dx;
    smoothedMat.at<double>(1, 2) = dy;

    // Warp the previous frame using the smoothed parameters
    warpAffine(prev_frame, smoothedFrame, smoothedMat, cur_frame.size());

    // Draw the view rect
//    cv::rectangle(smoothedFrame, centered_view_rect, cv::Scalar(0, 0, 255));

    // Crop the smoothed frame a little to eliminate black region due to Kalman Filter
    croppedFrame = smoothedFrame(cv::Range(vert_border, smoothedFrame.rows - vert_border),
                                 cv::Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols - HORIZONTAL_BORDER_CROP));

    resize(croppedFrame, croppedFrame, cur_frame.size());

    // Show both the unstabilized and stabilized frames
    if (1) {
        cv::Mat canvas = cv::Mat::zeros(cur_frame.rows * 2 + 10, cur_frame.cols * 2 + 10,
                                        cur_frame.type());

        cur_frame.copyTo(canvas(cv::Range(0, cur_frame.rows), cv::Range(0, cur_frame.cols)));

        smoothedFrame.copyTo(canvas(cv::Range(0, cur_frame.rows),
                                    cv::Range(cur_frame.cols + 10, cur_frame.cols * 2 + 10)));

        croppedFrame.copyTo(canvas(cv::Range(cur_frame.rows + 10, cur_frame.rows * 2 + 10),
                                   cv::Range(0, cur_frame.cols)));

        if (canvas.cols > 1920) {
            resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
        }

        imshow("1 original & 2 stabilized & 3 cropped", canvas);

        cv::waitKey(10);
    }

    return smoothedFrame;
}

void VideoStab::kalman_filter(double *scaleX, double *scaleY, double *theta, double *transX, double *transY) {
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_theta = *theta;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errtheta = errtheta + Q_theta;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_theta = frame_1_errtheta / (frame_1_errtheta + R_theta);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *theta = frame_1_theta + gain_theta * (sum_theta - frame_1_theta);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = (1 - gain_scaleX) * frame_1_errscaleX;
    errscaleY = (1 - gain_scaleY) * frame_1_errscaleX;
    errtheta = (1 - gain_theta) * frame_1_errtheta;
    errtransX = (1 - gain_transX) * frame_1_errtransX;
    errtransY = (1 - gain_transY) * frame_1_errtransY;
}
