/*
Thanks to Nghia Ho for his excellent code.
And,I modified the smooth step using a simple kalman filter .
So,It can processes live video streaming.
modified by chen jia.
email:chenjia2013@foxmail.com
*/

#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "video_stabilizer.h"

using namespace std;
using namespace cv;

// This video stabilization smooths the global trajectory using a sliding average window

// In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.
const int HORIZONTAL_BORDER_CROP = 20;

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct Trajectory {
    Trajectory() = default;

    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    // "+"
    friend Trajectory operator+(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
    }

    //"-"
    friend Trajectory operator-(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
    }

    //"*"
    friend Trajectory operator*(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
    }

    //"/"
    friend Trajectory operator/(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
    }

    //"="
    Trajectory operator=(const Trajectory &rx) {
        x = rx.x;
        y = rx.y;
        a = rx.a;
        return Trajectory(x, y, a);
    }

    double x;
    double y;
    double a; // angle
};

cv::Mat VideoStabilizer::stabilize(cv::Mat prev, cv::Mat cur) {
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
    cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da)
    double a = 0;
    double x = 0;
    double y = 0;

    // Step 3 - Smooth out the trajectory using an averaging window
    Trajectory X;                          //posteriori state estimate
    Trajectory X_;                         //priori estimate
    Trajectory P;                          // posteriori estimate error covariance
    Trajectory P_;                         // priori estimate error covariance
    Trajectory K;                          //gain
    Trajectory z;                          //actual measurement
    double pstd = 4e-2;                    //can be changed
    double cstd = 0.25;                    //can be changed
    Trajectory Q(pstd, pstd, pstd);        // process noise covariance
    Trajectory R(cstd, cstd, cstd);        // measurement noise covariance

    // Get the vertical border (Make sure the aspect ratio is correct)
    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols;

    // Vector from prev to cur
    vector<Point2f> prev_corner, cur_corner;
    vector<uchar> status;
    vector<float> err;

    goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
    calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

    vector<Point2f> prev_corner2, cur_corner2;
    prev_corner2.reserve(prev_corner.size());
    cur_corner2.reserve(cur_corner.size());

    // Weed out bad matches
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            prev_corner2.push_back(prev_corner[i]);
            cur_corner2.push_back(cur_corner[i]);
        }
    }

    // Rigid transform, translation + rotation only, no scaling/shearing
    Mat xform = estimateAffinePartial2D(prev_corner2, cur_corner2);

    // In rare cases no transform is found. We'll just use the last known good transform.
    if (xform.data == nullptr) {
        last_xform.copyTo(xform);
    }

    xform.copyTo(last_xform);

    // Decompose T
    double dx = xform.at<double>(0, 2);
    double dy = xform.at<double>(1, 2);
    double da = atan2(xform.at<double>(1, 0), xform.at<double>(0, 0));

    // Accumulated frame to frame transform
    x += dx;
    y += dy;
    a += da;

    z = Trajectory(x, y, a);

    if (k == 1) {
        // Initial guesses
        X = Trajectory(0, 0, 0);//Initial estimate,  set 0
        P = Trajectory(1, 1, 1);//set error variance,set 1
    } else {
        // Time update (prediction)
        X_ = X;    //X_(k) = X(k-1);
        P_ = P + Q;//P_(k) = P(k-1)+Q;
        // Measurement update（correction）
        K = P_ / (P_ + R);                 //gain;K(k) = P_(k)/( P_(k)+R );
        X = X_ + K * (z - X_);             //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k));
        P = (Trajectory(1, 1, 1) - K) * P_;//P(k) = (1-K(k))*P_(k);
    }

    // target - current
    double diff_x = X.x - x;
    double diff_y = X.y - y;
    double diff_a = X.a - a;

    dx = dx + diff_x;
    dy = dy + diff_y;
    da = da + diff_a;

    xform.at<double>(0, 0) = cos(da);
    xform.at<double>(0, 1) = -sin(da);
    xform.at<double>(1, 0) = sin(da);
    xform.at<double>(1, 1) = cos(da);

    xform.at<double>(0, 2) = dx;
    xform.at<double>(1, 2) = dy;

    // Get stabilized frame.
    Mat new_cur;
    warpAffine(prev, new_cur, xform, cur.size());

    // Crop black parts.
    Rect view_rect = Rect(Point2i(HORIZONTAL_BORDER_CROP, vert_border),
                          Size(new_cur.cols - HORIZONTAL_BORDER_CROP * 2, new_cur.rows - vert_border * 2));
    Mat new_cur_cropped = new_cur(view_rect);

    // Resize new_cur back to cur size, for better side by side comparison
    resize(new_cur_cropped, new_cur_cropped, cur.size());

    // Now draw the original and stabilized side by side for coolness
    {
        cv::Mat canvas = cv::Mat::zeros(new_cur.rows * 2 + 10, new_cur.cols * 2 + 10,
                                        new_cur.type());

        prev.copyTo(canvas(cv::Range(0, new_cur.rows), cv::Range(0, new_cur.cols)));

        cv::rectangle(new_cur, view_rect, cv::Scalar(0, 0, 255));
        new_cur.copyTo(canvas(cv::Range(0, new_cur.rows),
                              cv::Range(new_cur.cols + 10, new_cur.cols * 2 + 10)));

        new_cur_cropped.copyTo(canvas(cv::Range(new_cur.rows + 10, new_cur.rows * 2 + 10),
                                      cv::Range(0, new_cur.cols)));

        if (canvas.cols > 1920) {
            resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
        }

        imshow("1 original & 2 stabilized & 3 cropped", canvas);

        waitKey(10);
    }

    cur_grey.copyTo(prev_grey);

    k++;

    return new_cur;
}
