import argparse
from data_loading import generate_data_DPR_csv, generate_data_paired_sy_csv, generate_data_GT_csv, generate_test_own

def main():
    parser = argparse.ArgumentParser(description='HD-Net')

    parser.add_argument('--dataset_path', type=str, default='/media/hdr/oo/Datasets/DATA_pose_15/Syn_data/',
                    help='dataset path for generating csv files for training and testing')
    args = parser.parse_args()

    # generate dpr_train.csv and dpr_test.csv on the upper level folder
    # dataset_path = '/media/hdr/oo/Datasets/DPR/imgs/DPR_dataset/'
    # generate_data_DPR_csv(dataset_path)

    #generate train.csv and test.csv
    # dataset_path = args.dataset_path
    # generate_data_paired_sy_csv(dataset_path)

    #generate train.csv and test.csv
    # dataset_path = '/media/xteam1/6699a23c-17ee-4ba0-a18f-d5b89d4917ee/dataset/GT/nonlinear/'
    # generate_data_GT_csv(dataset_path)

    own_data_path = './data/sample/'
    generate_test_own(own_data_path)

if __name__ == '__main__':
    main()
