#include "video_stabilizer.h"
#include "video_stabilizer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    VideoStabilizer stabilizer;

    cv::VideoCapture cap(0);

    cv::Mat prev_frame, cur_frame;

    cap >> prev_frame;

    while (true) {
        cap >> cur_frame;

        if (cur_frame.data == nullptr) {
            break;
        }

        cv::Mat new_frame = stabilizer.stabilize(prev_frame, cur_frame);

        prev_frame = cur_frame.clone();
    }

    return 0;
}
