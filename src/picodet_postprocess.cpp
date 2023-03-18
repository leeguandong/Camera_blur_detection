#include "picodet_postprocess.h"


void NMS(fastdeploy::vision::DetectionResult* result, float iou_threshold) {
	SortDetectionResult(result);

	std::vector<float> area_of_boxes(result->boxes.size());
	std::vector<int> suppressed(result->boxes.size(), 0);
	for (size_t i = 0; i < result->boxes.size(); ++i) {
		area_of_boxes[i] = (result->boxes[i][2] - result->boxes[i][0]) *
			(result->boxes[i][3] - result->boxes[i][1]);
	}

	for (size_t i = 0; i < result->boxes.size(); ++i) {
		if (suppressed[i] == 1) {
			continue;
		}
		for (size_t j = i + 1; j < result->boxes.size(); ++j) {
			if (suppressed[j] == 1) {
				continue;
			}
			float xmin = std::max(result->boxes[i][0], result->boxes[j][0]);
			float ymin = std::max(result->boxes[i][1], result->boxes[j][1]);
			float xmax = std::min(result->boxes[i][2], result->boxes[j][2]);
			float ymax = std::min(result->boxes[i][3], result->boxes[j][3]);
			float overlap_w = std::max(0.0f, xmax - xmin);
			float overlap_h = std::max(0.0f, ymax - ymin);
			float overlap_area = overlap_w * overlap_h;
			float overlap_ratio =
				overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
			if (overlap_ratio > iou_threshold) {
				suppressed[j] = 1;
			}
		}
	}
	fastdeploy::vision::DetectionResult backup(*result);
	result->Clear();
	result->Reserve(suppressed.size());
	for (size_t i = 0; i < suppressed.size(); ++i) {
		if (suppressed[i] == 1) {
			continue;
		}
		result->boxes.emplace_back(backup.boxes[i]);
		result->scores.push_back(backup.scores[i]);
		result->label_ids.push_back(backup.label_ids[i]);
	}
}

void Merge(fastdeploy::vision::DetectionResult* result, size_t low, size_t mid, size_t high) {
	std::vector<std::array<float, 4>> & boxes = result->boxes;
	std::vector<float>& scores = result->scores;
	std::vector<int32_t>& label_ids = result->label_ids;
	std::vector<std::array<float, 4>> temp_boxes(boxes);
	std::vector<float> temp_scores(scores);
	std::vector<int32_t> temp_label_ids(label_ids);
	size_t i = low;
	size_t j = mid + 1;
	size_t k = i;
	for (; i <= mid && j <= high; k++) {
		if (temp_scores[i] >= temp_scores[j]) {
			scores[k] = temp_scores[i];
			label_ids[k] = temp_label_ids[i];
			boxes[k] = temp_boxes[i];
			i++;
		}
		else {
			scores[k] = temp_scores[j];
			label_ids[k] = temp_label_ids[j];
			boxes[k] = temp_boxes[j];
			j++;
		}
	}
	while (i <= mid) {
		scores[k] = temp_scores[i];
		label_ids[k] = temp_label_ids[i];
		boxes[k] = temp_boxes[i];
		k++;
		i++;
	}
	while (j <= high) {
		scores[k] = temp_scores[j];
		label_ids[k] = temp_label_ids[j];
		boxes[k] = temp_boxes[j];
		k++;
		j++;
	}
}

void MergeSort(fastdeploy::vision::DetectionResult* result, size_t low, size_t high) {
	if (low < high) {
		size_t mid = (high - low) / 2 + low;
		MergeSort(result, low, mid);
		MergeSort(result, mid + 1, high);
		Merge(result, low, mid, high);
	}
}


void SortDetectionResult(fastdeploy::vision::DetectionResult* result) {
	size_t low = 0;
	size_t high = result->scores.size();
	if (high == 0) {
		return;
	}
	high = high - 1;
	MergeSort(result, low, high);
}

