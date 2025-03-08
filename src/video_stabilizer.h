#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>

// In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.
constexpr int HORIZONTAL_BORDER_CROP = 40;

struct Trajectory {
    Trajectory() = default;

    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    friend Trajectory operator+(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
    }

    friend Trajectory operator-(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
    }

    friend Trajectory operator*(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
    }

    friend Trajectory operator/(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
    }

    Trajectory operator=(const Trajectory &rx) {
        x = rx.x;
        y = rx.y;
        a = rx.a;
        return Trajectory(x, y, a);
    }

    double x = 0;
    double y = 0;
    double a = 0;// angle
};

class VideoStabilizer {
public:
    VideoStabilizer() = default;

    cv::Mat stabilize(cv::Mat prev_frame, cv::Mat cur_frame);

private:
    cv::Mat prev_grey;
    cv::Mat cur_grey;

    int k = 1;

    cv::Mat last_xform;

    // Step 1 - Get previous to current frame transformation (dx, dy, da)
    double a = 0;
    double x = 0;
    double y = 0;

    // Step 3 - Smooth out the trajectory using an averaging window
    Trajectory X;// posteriori state estimate
    Trajectory P;// posteriori estimate error covariance

    // Debug data
    std::ofstream out_trajectory;
    std::ofstream out_smoothed_trajectory;
};
