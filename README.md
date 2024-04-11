This repository contains the code for fulfillment of the Midterm Project for the NYU Course Deep Learning with Prof Chinmay Hegde. The task is to create Resnet for classification on the CIFAR-10 dataset while keeping the trainable parameter count in the model below 5 mio. 

Files in this repository: 
- models.py contains models used for training
- scriptArchtest.ipynb is the notebook used for conducting the test for optimal amoutns of residual layers and blocks
- scriptDataAlterationTest.ipynb is the notebook used for testing the optimal data augmentation strategy
- scriptTrainParamstest.ipynb is the notebook used for testing various training settings and paramaters
- trainFinalModel.py is a python script to train the final model for 200 epochs
- Resnet3_443Exp_Final_best_modelNoDropouts.pth Model Checkpoint for performing the inference on the test dataset.
- submissions.csv containing ID and Labels for the custom test dataset from kaggle 


