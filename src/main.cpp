#include "video_stab.h"
#include "video_stab_v2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
    VideoStab stab;

    cv::VideoCapture cap(0);

    cv::Mat prev_frame, cur_frame;

    cap >> prev_frame;

    while (true) {
        cap >> cur_frame;

        if (cur_frame.data == nullptr) {
            break;
        }

        cv::Mat smoothedFrame = stab.stabilize(prev_frame, cur_frame);

        prev_frame = cur_frame.clone();
    }

    return 0;
}
