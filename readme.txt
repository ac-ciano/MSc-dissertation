Project Title: Suicide risk prediction in social media using
Machine Learning and Large Language Models

This repository contains the code and experiments conducted.

Project Structure
The repository is organized as follows:

.
├── preprocessing.ipynb
├── requirements.txt           (Python dependencies)
├── config.py                  (Centralized API configuration)
├── .env.example               (Template for environment variables)
├── .env                       (Your actual API keys)
├── .gitignore                 (Excludes sensitive files from version control)
└── src/
    ├── func.py
    ├── K-fold
    │   └── {loss_function}--{learning_rate}
    │       ├── ...
    │       └── large-cross-entropy--5e-5
    │           ├── ... (files for 5-fold cross-validation runs)
    │           └── ...-alldata.py (final model finetuning on all data)
    └── NO-fold
        ├── Google/            (Gemini API scripts)
        ├── DeepSeek/          (DeepSeek API scripts)
        ├── Grok/              (xAI/Grok API scripts)
        ├── ... (scripts for LLM prompting)
        └── ... (scripts for pseudo-labelling)

Key Components
1. Configuration Management
config.py: Centralized configuration module that manages API keys and environment variables. All scripts that interact with LLM APIs (xAI/Grok, Google/Gemini, DeepSeek) import their credentials from this module.

.env: Local environment file containing your actual API keys. This file is git-ignored and must be created from .env.example.

.env.example: Template file showing the required environment variables. Copy this to .env and fill in your actual API keys.

2. Data Preprocessing
preprocessing.ipynb: A Jupyter Notebook located in the root directory. This notebook contains all the steps used to preprocess the raw data for the models. Note: The dataset itself is not included in this repository.

3. Core Code
src/func.py: This is a crucial script containing shared utility functions and prompts that are imported and used across various models and experiments, particularly for the LLM prompting phase.

4. Model Fine-Tuning (K-Fold Cross-Validation)
src/K-fold/: This directory houses all experiments related to the fine-tuning of the RoBERTa model using k-fold cross-validation.

Subdirectories: The subdirectories are named using the convention {loss_function}--{learning_rate} to clearly identify the hyperparameters used for each experiment (e.g., large-cross-entropy--5e-5).

Final Model Training: Within these subdirectories, files with the suffix all or alldata signify the final stage of fine-tuning. In this stage, the best-performing model architecture was trained on the entire training dataset without a separate validation set.

Iterative Fine-Tuning: Some folders may contain a nested shell structure, indicating further fine-tuning experiments that were performed based on a previous model's checkpoint.

5. LLM Prompting and Pseudo-Labelling
src/NO-fold/: This directory contains the scripts that do not use a cross-validation structure. Its primary purpose includes:

Scripts for prompting various Large Language Models (LLMs).

Code that combines the outputs from the LLMs to generate pseudo-labels for subsequent experiments.

Getting Started
To replicate the results or run the experiments, follow these general steps:

Prerequisites: Ensure you have Python 3.8+ and the necessary libraries installed.

1. Install Dependencies:
   pip install -r requirements.txt

2. Configure API Keys:
   This project uses several LLM APIs (xAI/Grok, Google/Gemini, DeepSeek). To set up your API keys:
   
   a. Copy the example environment file:
      copy .env.example .env    (Windows)
      cp .env.example .env      (Linux/Mac)
   
   b. Edit the .env file and add your actual API keys:
      XAI_API_KEY=your_xai_api_key_here
      GOOGLE_API_KEY=your_google_api_key_here
      DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   IMPORTANT: Never commit the .env file to version control. It is already included in .gitignore.
   
   The config.py module handles loading these environment variables automatically.

3. Data: The necessary raw data can be obtained by contacting "hialex.li@connect.polyu.hk" directly, placing "xinhong.chen@my.cityu.edu.hk" in CC. It will be required to sign a DUA that they will provide.

4. Preprocessing: Run the preprocessing.ipynb notebook to process the data and prepare it for the models.

5. Run Experiments: Navigate to the relevant experiment directory (src/K-fold or src/NO-fold) and execute the desired Python scripts.
