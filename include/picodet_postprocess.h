#pragma once
#ifndef PICODET_POSTPROCESS
#define PICODET_POSTPROCESS

#include <cmath>
#include "fastdeploy/vision.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"


using namespace std;

void NMS(fastdeploy::vision::DetectionResult* result, float iou_threshold);

void SortDetectionResult(fastdeploy::vision::DetectionResult* output);


#endif // !PICODET_POSTPROCESS



