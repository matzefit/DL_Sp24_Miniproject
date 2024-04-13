This repository contains the code for fulfillment of the Midterm Project for the NYU Course Deep Learning with Prof Chinmay Hegde. The task is to create Resnet for classification on the CIFAR-10 dataset while keeping the trainable parameter count in the model below 5 mio. 

Files in this repository: 
- models.py contains models used for training
- scriptArchtest.ipynb is the notebook used for conducting the test for optimal amoutns of residual layers and blocks
- scriptDataAlterationTest.ipynb is the notebook used for testing the optimal data augmentation strategy
- scriptTrainParamstest.ipynb is the notebook used for testing various training settings and paramaters
- scriptFinalModel.ipynb is the notebook used for running the final model training for 200 epochs. 
- Resnet3_443_Exp_Final_best_model_secondRun.pth is the best Model Checkpoint for performing the inference on the test dataset.
- submissionRerunBestModel.csv contains the predicted ID and Labels for the custom test dataset from kaggle, generated through the use of the best model checkpoint.


Additional: 
- data folder contains the CIFAR-10 data including the custom made test data for participating in the kaggle competition created for this midterm.
- augmentViz.ipynb is a notebook for visualizing the applied data augmentation strategies in the training dataset. 


To reproduce the best model checkpoint (Resnet3_443_Exp_Final_best_model_secondRun.pth) simply open and run scriptFinalModel.ipynb. 
The Output of this notebook is the prediction of test data labels (submissionRerunBestModel.csv) computed through the use of the best model checkpoint (Resnet3_443_Exp_Final_best_model_secondRun.pth)

