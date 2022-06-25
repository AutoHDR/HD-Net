# HD-Net

<center><img src="data/results/_img3.png " width="80%"></center>
From top to bottom are the input face, predicted albedo, predicted shading, visualized light and predicted normal.

## Training
Please see each subsection for training on different datasets. Available training datasets:

* [250k synthetic face images](https://drive.google.com/file/d/1UQONt9Usk3PKztSIoXeNUEUqD5s6z69e/view?usp=sharing)
* [DPR](https://drive.google.com/drive/folders/10luekF8vV5vo2GFYPRCe9Rm2Xy2DwHkT?usp=sharing)
* [FFHQ](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP) 
* [Real Face Image] (Real paired face image needs warped and unwarped function for alignment from [Nonlinear_Face_3DMM](https://github.com/tranluan/Nonlinear_Face_3DMM))

#### Generating the training and test csv
Run
```
python generate_dataset_csv.py \
   --dataset_path $DATA_DIR
```
#### Training

Run
```
python train_main.py \
   --dataset_path $DATA_DIR \
   --batch_size 8 \
   --log_dir  $LOG_DIR  \
   --stage $STAGE_NAME \
   --datadir $DTU_DIR
```

#### Testing on sample data

Run
```
python test_main.py 
```

Thanks to [**SfSNet-PyTorch**](https://github.com/bhushan23/SfSNet-PyTorch), [**DPR**](https://github.com/zhhoper/DPR), [**Nonlinear_Face_3DMM**](https://github.com/tranluan/Nonlinear_Face_3DMM) and [**3DMM**](https://github.com/MichaelMure/3DMM) our code is partially borrowing from them.

