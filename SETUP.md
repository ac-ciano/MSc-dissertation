# Setup Guide

This guide will help you set up the environment and configure API keys for the MSc Dissertation project on suicide risk prediction using Machine Learning and Large Language Models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API Configuration](#api-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)
- API keys for the following services (only if running LLM experiments):
  - xAI (Grok) API
  - Google (Gemini) API
  - DeepSeek API

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ac-ciano/MSc-dissertation.git
cd MSc-dissertation
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- pandas, numpy (data processing)
- torch, transformers (deep learning)
- scikit-learn (metrics and evaluation)
- openai (API client for xAI and DeepSeek)
- google-generativeai (Google Gemini API)
- python-dotenv (environment variable management)

## API Configuration

The project uses a centralized configuration system to manage API keys securely.

### 1. Create Environment File

Copy the example environment file to create your own:

**Windows:**
```cmd
copy .env.example .env
```

**Linux/Mac:**
```bash
cp .env.example .env
```

### 2. Add Your API Keys

Open the `.env` file in a text editor and replace the placeholder values with your actual API keys:

```env
# xAI (Grok) API Configuration
XAI_API_KEY=xai-your_actual_key_here
XAI_BASE_URL=https://api.x.ai/v1

# Google (Gemini) API Configuration
GOOGLE_API_KEY=your_actual_google_api_key_here

# DeepSeek API Configuration
DEEPSEEK_API_KEY=sk-your_actual_deepseek_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 3. Obtaining API Keys

#### xAI (Grok) API
1. Visit [xAI Console](https://console.x.ai/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (format: `xai-...`)

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

#### DeepSeek API
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Sign up or log in
3. Go to API Keys section
4. Generate a new key
5. Copy the key (format: `sk-...`)

### 4. Verify Configuration

The `config.py` module automatically loads and validates your API keys. When you run any script that uses LLM APIs, it will check if the required keys are present.

**Security Best Practices:**
- ✅ **NEVER** commit your `.env` file to version control
- ✅ The `.gitignore` file already excludes `.env` files
- ✅ Only share `.env.example` as a template
- ✅ Keep your API keys confidential
- ✅ Rotate keys if they are accidentally exposed

## Project Configuration System

The project uses a modular configuration approach:

- **config.py**: Central configuration module that reads from `.env`
- **Config class**: Provides methods to access API configurations:
  - `Config.get_xai_config()` - Returns xAI API configuration
  - `Config.get_google_config()` - Returns Google API configuration
  - `Config.get_deepseek_config()` - Returns DeepSeek API configuration
  - `Config.validate_api_keys()` - Validates required API keys

All scripts in `src/NO-fold/Grok/`, `src/NO-fold/Google/`, and `src/NO-fold/DeepSeek/` automatically import and use this configuration.

## Data Setup

The raw dataset is not included in this repository due to data use agreements. To obtain access:

1. Contact **hialex.li@connect.polyu.hk**
2. CC **xinhong.chen@my.cityu.edu.hk**
3. Request access to the suicide risk prediction dataset
4. Sign the Data Use Agreement (DUA) they provide
5. Once received, place the data in the appropriate directory and run `preprocessing.ipynb`

## Running Experiments

### RoBERTa Fine-Tuning (K-Fold)

Navigate to the appropriate directory and run the training script:

```bash
cd src/K-fold/large-cross-entropy--5e-5
python finetune_roberta.py
```

### LLM Inference

For running LLM-based experiments:

```bash
cd src/NO-fold/Grok
python grok-3.py
```

```bash
cd src/NO-fold/Google
python gemini-2.5-flash.py
```

```bash
cd src/NO-fold/DeepSeek
python deepseek-reasoner.py
```

## Troubleshooting

### Missing API Keys Error
```
ValueError: Missing required API key(s): XAI_API_KEY
```
**Solution**: Ensure your `.env` file exists and contains the correct API key.

### Import Error for dotenv
```
ModuleNotFoundError: No module named 'dotenv'
```
**Solution**: Install python-dotenv:
```bash
pip install python-dotenv
```

### API Rate Limiting
If you encounter rate limit errors, the scripts include automatic retry logic. You may need to:
- Wait and retry later
- Upgrade your API tier
- Reduce concurrent requests

### Virtual Environment Issues
If packages aren't found after installation, ensure your virtual environment is activated:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

## Need Help?

For issues related to:
- **Code/Repository**: Open an issue on GitHub
- **Data Access**: Contact the data providers listed above
- **API Issues**: Consult the respective API documentation:
  - [xAI API Docs](https://docs.x.ai/)
  - [Google Gemini Docs](https://ai.google.dev/docs)
  - [DeepSeek API Docs](https://platform.deepseek.com/api-docs/)

---

## License & Citation

Please refer to the main README for license information and citation guidelines.
