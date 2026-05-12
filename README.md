## AGNI Pipeline - A Machine Learning-Augmented Flame Front Detection in High-speed Imaging

This repository (AGNI Framework) contains the integration of all three-stage classifiers for detecting the flame front(edge detection) in high-speed imaging. AGNI is modular, and automated. We
achieve this by using a hierarchical classification approach that integrates three components: a
usability classifier that removes non-informative images such as background data, and quenched
flames, a segmentation model for extracting the flame boundary and a third artifact removal
module that eliminates shot noise, double flame edges and other aberrations to produce a physically consistent flame front.

## Description

Stage 1 Classifier: The input data consists of the flame in one of the following three states at any point in any given cycle: i) initial
growth, ii) quasi-spherical state, and iii) the turbulent state.Images of the quasi-spherical flame from its nascent
state to the initial turbulent state would form the images of interest. In contrast, background images and quenched
flames are unusable for edge detection. To ensure that these usable images are consistently identified, we use the
sequence information as an additional feature. The sequence information provides positional information along with the
relevant image pixel data, helping the Model detect the correct set of images and mark them as usable.

Stage 2 Edge detection: The second stage of the framework uses the images that are labelled as usable by the first stage. These images are
usable because they contain the relevant flame-front information and are guaranteed to exclude background or quenched
flame images, where boundary edge detection becomes irrelevant. The objective of the second stage is to detect the boundary flame fronts from the given image data.We chose the U-Net image segmentation
model to ensure that the pixel label was assigned based on a) the local information (the immediate pixel neighbors) and
b) the global information (the overall flamefront).The U-Net architecture uses a ResNet34 encoder pre-trained on ImageNet as the backbone.

Stage 3 Artifacts removal: To ensure that the output flame edge is consistent with the physics of outward flame propagation, we use the third
stage classifier to remove the artifacts such as shot noise, double flame edges and late flames.The contour refinement Model is trained to distinguish the true flame edge from the other external flame edges (shot
noise) and internal flame edges (double flames). We train the Model to recognize the characteristics of the valid flame
fronts using the cycle data. We also ensure that the Model produces a continuous flame front. Thus, the resulting output
would be a clean, physics-consistent, uninterrupted flame boundary, free of artifacts. We use a U-Net Model similar to stage 2 for the artifact removal phase.

## Framework implementation
The pipeline is designed to operate in two modes: inference-only and training followed by inference. This dual-mode capability demonstrates the flexibility of the framework to process user-provided high-speed OH∗ chemiluminescence images without requiring retraining, while also supporting full model development workflows when training data is available.


## Folder Structure
```
.
├── agni.py						# main .py file which orchestrate all three stages
├── README.md
├── requirements.txt
├── Stage1						# Classifier 1, classifies an image to either usable or unsuable 		
│   ├── data
│   │   ├── final_cleaned_image_sequence_cycles.csv	# Labelled cycle data to train classifier 1
│   │   └── output_baseline				# Actual input images
│   ├── results						# folder to save the.csv of stage 1 classifier output	
│   │                                    		  when tested on modes Inference, Training+Inference 
│   │  
│   ├── src						# Stage 1 training files
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── infer.py
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── transforms.py
│   └── stage1.py					# Stage 1 entry file
├── Stage2						# Segmentation classifier to detect the flame front
│   ├── Inference					# Folder generated when the framework is evaluated in inference mode
│   │   ├── Stage1_Output				# Output of Stage 1 in inference mode
│   │   │   └── final_usable_unusable_images.csv
│   │   ├── Stage2_Output				# Output of Stage 2 in inference mode
│   │   │   ├── masks
│   │   │   ├── Masks_Pixel_Result.csv
│   │   │   └── overlay
│   │   └── Stage3_Output				# Output of Stage 3 inference on Stage 2 Masks 
│   │       ├── Masks
│   │       └── Overlay
│   ├── results						# Result/Output of Stage 2 when tested on Training+Inference
│   │   ├── checkpoints
│   │   │   └── edge_model_1.pth
│   │   ├── masks
│   │   ├── Masks_Pixel_Result.csv
│   │   └── overlay
│   ├── src						# Stage 2 training files
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── infer.py
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── transform.py
│   └── stage2.py					# Stage 2 entry file
├── Stage3						# Classifier 3 to remove artifacts from Stage 2 mask
│   ├── checkpoints					# Used only in inference mode. So the .pth of Stage 3 is mandatory
│   │   └── best_unet.pth
│   ├── __init__.py
│   ├── src						# Stage 3 training files. If needed, Stage 3 can be trained again
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── train.py
│   └── stage3.py					# Stage 3 entry file
└── structure.txt

26 directories, 37 files

```

## Setup

Create virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt


/scratch/krishna/agni-p1-pipeline/output_baseline
/scratch/krishna/agni-p1-pipeline/results_baseline/Training/Stage1_Output/models/best_model_fold6.pth

/scratch/krishna/agni-p1-pipeline/Stage2/results/checkpoints/edge_model_1.pth

/scratch/krishna/agni-p1-pipeline/Stage3/checkpoints/best_unet.pth


rm -rf pymp-* tmp*

python agni.py \
  --stage all \
  --mode train_infer \
  --input_dir output_baseline \
  --output_dir results \
  --csv_path path/to/labels.csv \
  --mask_dir white_boundaries \
  --min_img 1 \
  --max_img 20000 \
  --s1_epochs 20 \
  --s1_batch_size 16 \
  --s1_lr 1e-4 \
  --s1_kfolds 10 \
  --s2_epochs 120 \
  --s2_batch_size 4 \
  --s2_lr 5e-5 \
  --s2_pos_weight 8.0 \
  --s2_bce_weight 0.75
