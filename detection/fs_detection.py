import os
from detections.base import DetectionEvaluator as fs_detection


class DetectionEvaluator:
    def __init__(self, model, attack_string_hash, dataset_name):
        result_folder_detection = os.path.join("results", "fs")
        csv_fname = "%s_detection.csv" % (attack_string_hash)
        self.de = fs_detection(model, result_folder_detection, csv_fname, dataset_name)

    def build_detection_dataset(self, X_test_all, Y_test_all, Y_test_all_pred, selected_idx,
                                X_test_adv_discretized_list, Y_test_adv_discretized_pred_list, attack_string_list,
                                attack_string_hash, clip=-1, Y_test_target_next=None, Y_test_target_ll=None):
        self.de.build_detection_dataset(X_test_all, Y_test_all, Y_test_all_pred, selected_idx,
                                        X_test_adv_discretized_list, Y_test_adv_discretized_pred_list,
                                        attack_string_list,
                                        attack_string_hash, clip, Y_test_target_next, Y_test_target_ll)

    def evaluate_detections(self, dataname):
        if(dataname=='mnist'):
            detection="FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr=0.05;"
        elif(dataname=='cifar10'):
            detection="FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.2;"
        elif (dataname == 'svhn'):
            detection = "FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.1;"
        else:
            detection='FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2&distance_measure=l1&fpr=0.1;'
        self.de.evaluate_detections(detection)
