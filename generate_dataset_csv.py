from data_loading import generate_data_DPR_csv, generate_data_paired_sy_csv, generate_data_GT_csv

# generate dpr_train.csv and dpr_test.csv on the upper level folder
# dataset_path = '/media/hdr/oo/Datasets/DPR/imgs/DPR_dataset/'
# generate_data_DPR_csv(dataset_path)

#generate train.csv and test.csv
dataset_path = '/dataset/disk2/dataset/DATA_pose_15/Syn_data/'
generate_data_paired_sy_csv(dataset_path)

#generate train.csv and test.csv
# dataset_path = '/media/xteam1/6699a23c-17ee-4ba0-a18f-d5b89d4917ee/dataset/GT/nonlinear/'
# generate_data_GT_csv(dataset_path)
