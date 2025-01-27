#include "video_stab.h"
#include <cmath>

// Parameters for Kalman Filter
constexpr double Q1 = 0.004;
constexpr double R1 = 0.5;

constexpr double MAX_ROTATION_DEGREE = 15.0;
constexpr double MAX_TRANSLATION_FACTOR = 0.1;

VideoStab::VideoStab() {
    smoothedMat.create(2, 3, CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errthetha = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_thetha = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_thetha = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_thetha = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 0;
    scaleY = 0;
    thetha = 0;
    transX = 0;
    transY = 0;
}

cv::Mat VideoStab::stabilize(cv::Mat original_frame_1, cv::Mat original_frame_2, int down_sampling_factor) {
    cv::Size new_size = cv::Size((int) round(original_frame_1.cols / (float) down_sampling_factor), (int) round(original_frame_1.rows / (float) down_sampling_factor));

    cv::Mat frame_1, frame_2;
    cv::resize(original_frame_1, frame_1, new_size);
    cv::resize(original_frame_2, frame_2, new_size);

    cvtColor(frame_1, frame1, cv::COLOR_BGR2GRAY);
    cvtColor(frame_2, frame2, cv::COLOR_BGR2GRAY);

    int vert_border = HORIZONTAL_BORDER_CROP * frame_1.rows / frame_1.cols;

    std::vector<cv::Point2f> features1, features2;
    std::vector<cv::Point2f> goodFeatures1, goodFeatures2;

    // Estimate the features in frame1 and frame2
    if (1) {
        goodFeaturesToTrack(frame1, features1, 200, 0.01, 30);
    } else {
        std::vector<cv::KeyPoint> keyPoints;
        cv::FAST(frame1, keyPoints, 10);

        cv::KeyPoint::convert(keyPoints, features1);
    }

    if (features1.empty()) { return original_frame_2; }

    std::vector<uchar> status;
    std::vector<float> err;
    calcOpticalFlowPyrLK(frame1, frame2, features1, features2, status, err);

    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            goodFeatures1.push_back(features1[i]);
            goodFeatures2.push_back(features2[i]);
        }
    }

    if (goodFeatures1.empty()) {
        return original_frame_2;
    }

    // All the parameters scale, angle, and translation are stored in affine
    affine = estimateAffinePartial2D(goodFeatures1, goodFeatures2);

    if (affine.empty()) {
        return original_frame_2;
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
    sum_thetha += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;

    // Don't calculate the predicted state of Kalman Filter on 1st iteration
    if (k == 1) {
        k++;
    } else {
        kalman_filter(&scaleX, &scaleY, &thetha, &transX, &transY);
    }

    diff_scaleX = scaleX - sum_scaleX;
    diff_scaleY = scaleY - sum_scaleY;
    diff_transX = transX - sum_transX;
    diff_transY = transY - sum_transY;
    diff_thetha = thetha - sum_thetha;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_thetha;

    auto max_rotation_radian = MAX_ROTATION_DEGREE / 180.0 * 3.1415926;

    auto max_translation = MAX_TRANSLATION_FACTOR * frame_2.rows;

    // Clamp transform
    da = da > max_rotation_radian ? max_rotation_radian : da;
    da = da < -max_rotation_radian ? -max_rotation_radian : da;
    dx = dx > max_translation ? max_translation : dx;
    dx = dx < -max_translation ? -max_translation : dx;
    dy = dy > max_translation ? max_translation : dy;
    dy = dy < -max_translation ? -max_translation : dy;

    // Creating the smoothed parameters matrix
    smoothedMat.at<double>(0, 0) = sx * cos(da);
    smoothedMat.at<double>(0, 1) = sx * -sin(da);
    smoothedMat.at<double>(1, 0) = sy * sin(da);
    smoothedMat.at<double>(1, 1) = sy * cos(da);

    smoothedMat.at<double>(0, 2) = dx;
    smoothedMat.at<double>(1, 2) = dy;

    // Warp the new frame using the smoothed parameters
    warpAffine(original_frame_2, smoothedFrame, smoothedMat, original_frame_2.size());

    // Crop the smoothed frame a little to eliminate black region due to Kalman Filter
    //    smoothedFrame = smoothedFrame(Range(vert_border, smoothedFrame.rows - vert_border), Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols - HORIZONTAL_BORDER_CROP));

    // Crop frame
    {
        // Create a white mask
        cv::Mat rect_mask(frame_2.size(), CV_8UC1, cv::Scalar(1));

        cv::Mat transformed;
        warpAffine(rect_mask, transformed, smoothedMat, rect_mask.size());
        rect_mask = rect_mask.mul(transformed);

        // Convert mask to square shape by using the smallest of the dimensions
        const auto min_dim = std::min(rect_mask.rows, rect_mask.cols);

        cv::Mat square_mask;
        resize(rect_mask, square_mask, {min_dim, min_dim});

        // Find the largest inscribed square of the stabilized frame, starting
        // from the bottom-right corner
        // Initialize a matrix with the same size as the mask and initial values of
        // 0, i.e. black
        cv::Mat s(square_mask.size(), CV_32SC1, cv::Scalar(0.0));

        for (auto row = square_mask.rows - 1; row > 0; --row) {
            for (auto col = square_mask.cols - 1; col > 0; --col) {
                if (square_mask.at<uchar>(row, col) == 0) {
                    continue;
                }

                // If we're dealing with the bottom-right corner, we can't use the
                // bottom, right, or bottom-right cells, so we just set the value to 1
                if (row == square_mask.rows - 1 || col == square_mask.cols - 1) {
                    s.at<int>(row, col) = 1;
                    continue;
                }

                // Otherwise, calculate the value of this cell by following the formula:
                // S[x, y] = min(S[x + 1, y], S[x, y + 1], S[x + 1, y + 1]) + 1
                s.at<int>(row, col) =
                        std::min(s.at<int>(row + 1, col), std::min(s.at<int>(row, col + 1), s.at<int>(row + 1, col + 1))) + 1;
            }
        }

        // Create a region that represents the largest inscribed square
        double square_min, square_max;
        cv::Point square_min_idx, square_max_idx;
        // Find the global min and max in the count matrix
        cv::minMaxLoc(s, &square_min, &square_max, &square_min_idx, &square_max_idx);
        cv::Rect square(square_max_idx, cv::Size(static_cast<int>(square_max), static_cast<int>(square_max)));

        // Scale the square region
        const cv::Point2f scale(static_cast<float>(rect_mask.cols) / static_cast<float>(square_mask.cols),
                                static_cast<float>(rect_mask.rows) / static_cast<float>(square_mask.rows));

        cv::Rect2f scaled_rect(
                cv::Point2f(scale.x * square_max_idx.x, scale.y * square_max_idx.y),
                cv::Size2f(scale.x * square.width, scale.y * square.height));

        //        if (firstTime) {
        //            firstTime = false;
        //            currentRect = scaled_rect;
        //        }

        //        targetRect = scaled_rect;

        //        currentRect.x = currentRect.x + (targetRect.x - currentRect.x) * 0.75;
        //        currentRect.y = currentRect.y + (targetRect.y - currentRect.y) * 0.75;
        //        currentRect.width = currentRect.width + (targetRect.width - currentRect.width) * 0.75;
        //        currentRect.height = currentRect.height + (targetRect.height - currentRect.height) * 0.75;
        //
        //        scaled_rect = currentRect;

        cv::Rect final_scaled_rect(
                cv::Point2f(static_cast<int>(round(scaled_rect.x * down_sampling_factor)), static_cast<int>(round(scaled_rect.y * down_sampling_factor))),
                cv::Size2f(static_cast<int>(round(scaled_rect.width * down_sampling_factor)), static_cast<int>(round(scaled_rect.height * down_sampling_factor))));

        if (final_scaled_rect.area() == 0) {
            return original_frame_2;
        }

        // Crop the stabilized frames to the largest inscribed square
        cv::rectangle(smoothedFrame, final_scaled_rect, cv::Scalar(0, 0, 255));

        croppedFrame = smoothedFrame(final_scaled_rect);

        resize(croppedFrame, croppedFrame, original_frame_2.size());
    }

    // Resize smoothed frame to match the original frame.
    resize(smoothedFrame, smoothedFrame, original_frame_2.size());

    // Show both the unstabilized and stabilized frames
    if (1) {
        cv::Mat canvas = cv::Mat::zeros(original_frame_2.rows * 2 + 10, original_frame_2.cols * 2 + 10, original_frame_2.type());

        original_frame_2.copyTo(canvas(cv::Range(0, original_frame_2.rows), cv::Range(0, original_frame_2.cols)));

        smoothedFrame.copyTo(canvas(cv::Range(0, original_frame_2.rows), cv::Range(original_frame_2.cols + 10, original_frame_2.cols * 2 + 10)));

        croppedFrame.copyTo(canvas(cv::Range(original_frame_2.rows + 10, original_frame_2.rows * 2 + 10), cv::Range(0, original_frame_2.cols)));

        if (canvas.cols > 1920) {
            resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
        }

        imshow("1 original & 2 stabilized & 3 cropped", canvas);

        cv::waitKey(10);
    }

    return smoothedFrame;
}

void VideoStab::kalman_filter(double *scaleX, double *scaleY, double *thetha, double *transX, double *transY) {
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *thetha = frame_1_thetha + gain_thetha * (sum_thetha - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = (1 - gain_scaleX) * frame_1_errscaleX;
    errscaleY = (1 - gain_scaleY) * frame_1_errscaleX;
    errthetha = (1 - gain_thetha) * frame_1_errthetha;
    errtransX = (1 - gain_transX) * frame_1_errtransX;
    errtransY = (1 - gain_transY) * frame_1_errtransY;
}
