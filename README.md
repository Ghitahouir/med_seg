# Semantic segmentation of muscles on abdominal CT scans 

This model is trained on the "sarco dataset 1 & 2 improved" dataset to perform semantic segmentation. This dataset is private and is available in the drive in Milvue Links/Algorithm/Sarco/data. In this same location you will find the raw inference data (CT_scans) and the cleaned inference data - only DICOMs that correspond to the values and transforms of the training set (clean_CT_scans). The inputs are DICOM files (.dcm) and the labels are NIfTI files (.nib). You will find some good checkpoints in Milvue Links/Algorithm/Sarco/checkpoints. You will also find some predictions made by previous models under 'preds_from_...' in the data tab of the drive. 
To run this code: 
- Save all the scripts in a directory. <br/>
- Create a data directory and save the 'sarco dataset 1 & 2 improved' in it. <br/>
- Save also the 'clean_CT_scan' directory which contains appropriate inference data (also see my report for further explanation). <br>
- Create a directory called 'checkpoints' there too to save the checkpoints. <br/>
- cd to the directory where the training.py and inference.py scripts are saved. 

Here are the command lines explained. 

## **Simple use with default parameters :**

### Training: 

> **dirpath** (type=str) <br>
> help="the dirpath where the training/valid/test data is stored" <p>

> **data_dir** (type=str) <br>
> help="the dirpath where to save the checkpoints from the training" <p>

> **wandb_project** (type=str) <br>
> help="name of the wandb project to log into" <p>

> **gpu** (type=int) <br>
> help="choose which gpu to use" <p>

**Example**

```console

python training.py --data_dir="./data/dataset 1 & 2 improved"  --dirpath=./checkpoints --wandb_project=Trains --gpu=0 

```

### Inference: 

> **infer_path** (type=str) <br>
> help="the dirpath where the training/valid/test data is stored" <p>

> **run_id** (type=str) <br>
> help="the dirpath where to save the checkpoints from the training" <p>

> **wandb_project** (type=str) <br>
> help="name of the wandb project to log into" <p>

> **gpu** (type=int) <br>
> help="choose which gpu to use" <p>

```console

python inference.py --infer_path=./data/clean_CT_scans scan_ABDO_SS_IV_PACS_CD_of_2.16.840.1.113669.632.20.1532476995.53703847810000378356 --run_id=h8hv6uh5 --wandb_project=Predictions --dest_dir=./data --gpu=0 

```

## **Basic parameters that can be parsed :**

### Training: 

> **max_epochs** (type=int) <br>
> help="number of epochs, -1 for unlimited training." <p>

> **learning_rate** (type=float) <br>
> help="learning rate for the training (if not using scheduler)" <p>

> **batch_size** (type=int) <br>
> help="batch size, must be even and preferably a power of 2" <p>

> **backbone** (type=str) <br>
> help="size of the efficientnet backbone (from b0 to b7)" <p>

> **model_choice** (type=str) <br>
> help="whether to use a unet or a segresnet" <p>

```console

python training.py --max_epochs=50 --learning_rate=1e-4 --batch_size=8 --backbone=efficientnet-b3 --gpu=0 --model_choice=segresnet

```


### Inference: 

> **ckpt_choice** (type=str) <br>
> help="one of 'last' or 'best'" <p>

> **log_wandb_project** (type=str) <br>
> help="name of the Project where you want to log and visualize the predicted data" <p>

> **ckpt_wandb_project** (type=str) <br>
> help="name of the wandb Project where you trained your model" <p>

```console

python inference.py --ckpt_choice= =best --log_wandb_project=Predictions --ckpt_wandb_project=Predictions 

```

## **Data augmentation parameters :**

### Training: 

> **do_flips** (type=str) <br>
> help="one of 'yes', 'true', 't'... whether to perform flips and rotations as data augmentation" <p>

> **do_elastic_transforms** (type=str) <br>
> help="whether to perform elastic transforms as data augmentation" <p>

> **aug_prob** (type=float) <br>
> help="probability of applying previous data augmentation on training set" <p>

> **mixup** (type=str) <br>
> help="one of 'no', 'input' or 'manifold', choice of regularization you wish to perform (see also my report)" <p>

> **mixup_alpha** (type=float) <br>
> help="hyperparameter for Input Mixup and Manifold Mixup regularization" <p>

```console

python training.py --do_flips=yes --do_elastic_transforms=true  --aug_prob=0.5 --mixup=manifold --mixup_alpha=1

```

## **Optimization parameters :**

### Training: 

> **resume_training** (type=str) <br>
> help="whether to resume training from a previous model" <p>

> **ckpt_path_to_resume_from** (type=float) <br>
> help="path to the saved checkpoints you wish to use to restart the model. In the last pushed code, resuming from a checkpoint restarts at the last epoch of the chosen run and reloads all the hyperparameters and metrics as they were at the last epoch of the run. You can manually change a hyperparameter by parsing it and changing its original value. (Note: here, *h8hv6uh5* is a possible run ID and *Trains* is the name of the wandb project this run was trained in. This is the general syntax I decided to use to save my checkpoints: each directory is named *runID*_*wandbproject* and consists in two checkpoints namely 'best' checkpoint and 'last' checkpoint). " <p>

> **early_stopping** (type=str) <br>
> help="whether to perform early stopping" <p>

> **patience** (type=float) <br>
> help="early stopping patience: if the monitored metric, here the best average precision of the val set, does not increase for *patience*'s value epoch, the training is stopped" <p>

```console

python training.py --resume_training=yes --ckpt_path_to_resume_from=./checkpoints/h8hv6uh5_Trains--early_stopping=yes --patience=7 

```
## **Stacking parameters :**

### Training: 

> **channel_3d** (type=str) <br>
> help="whether to stack slices on 3 channels: the model uses the SarcoDataModule_3D and loads data 3 slices at a time. The checkpoints corresponding to such runs will have a '_3d' appended to their names. " <p>

> **add_predicted_data** (type=str) <br>
> help="whether to perform self-distillation (see also my report)" <p>

> **path_to_prediction** (type=str) <br>
> help="path to the predictions you wish to add to the training and/or val set" <p>


```console

python training.py --channel_3d=yes --add_predicted_data=yes --path_to_prediction=./data/clean_CT_scans/scan_ABDO_SS_IV_PACS_CD_of_2.16.840.1.113669.632.20.1532476995.53703847810000378356

```

**Note:** It is only useful to stack the data to 3D when adding predictions. It will also work without performing self-distillation but will not change anything to the training. See my report for further explanation. 

## **Common bugs / Miscellaneous remarks :**

- Be careful with the collate function (collate_fn) of the Dataloader. The data must have the exact same shape, same dictionary keys, same array shape to fit in the same batch. The batch size must necessarily be an even number, preferably a power of 2. 
- Be careful with the Monai transforms that sometimes behave not as you would expect it. Make sure to test each transform separately in a notebook and to make sure that it does what you think it does before adding it to the pipeline. 
- About medical data: pay attention to the encoding, to the dtype of the images, to the values of the pixels (are they Hounsfields Units? do you have to rescale them using the Rescale Intercept and Slope?). 
- Be careful with the the way you compute your metrics. Most metrics used need the arrays to have specific dtypes and values for the inputs and especially the labels (only binary, only int, must be flatten...). Make sure to compute them at each step and to average and plot them only at the end of each epoch. 
- There is still a problem in the inference pipeline. I still have a bug on the inverse method of my custom Cropd (and therefor Cropd_3D) transform. I didn't have the time to fix it. The Monai Transform *Invertd* is supposed to invert all the transforms applied to the inference set by calling the inverse method of all the transforms that inherit from Monai's *InvertibleTransform* class. Monai's *Invertd* still bugs in sarco_infdatamodule.py because of the Cropd and Cropd_3D transforms. For now I have just commented it. 
- In a Pytorch Lightning Module you can only use a pl_bolts scheduler. Others create bugs. About schedulers, I have used one from pl_bolts library, for now it is commented but you can uncomment all the lines referring to the scheduler. 