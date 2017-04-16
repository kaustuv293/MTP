# MTP
This repository contains codes for my thesis titled 'Learning Temporal Connectionism of Visual Abstractions using Deep Neural Architectures for Monitorring of Minimally Invasive Surgical Interventions towards Workflow Modelling'. The thesis proposes use of deep neural network based visual semantic search architectures for recognizing and tracking of different tools and various phases in surgical videos towards summarizarion and analysis of minimally invasive surgical workflow.

# Description of codes
  1. Training data for CNN: 4-D tensor (nSamples x nChannels x width x height), Training label: 2-D tensor (nSamples x nClasses)
  2. finetune.lua : Finetunes the CNN for tool detection
  3. test.lua : Evaluates the performance of CNN
  4. trainLSTM : Trains a deep LSTM network for tool detection
  5. testLSTM : Evaluates peformance of LSTM
  6. lstmTrainGen : Dataprepration for training LSTM
  
# Dataset used
The networks were trained on the m2ccai16-tool and workflow dataset(http://camma.u-strasbg.fr/datasets).
