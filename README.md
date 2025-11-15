# Suicide Risk Prediction in Social Media using Machine Learning and Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code and experiments for an MSc dissertation project on predicting suicide risk levels from social media posts using both traditional Machine Learning models (RoBERTa) and Large Language Models (LLMs).

## 🎯 Project Overview

The project explores automated suicide risk assessment across four risk levels:
- **Indicator**: Posts showing risk indicators
- **Ideation**: Suicidal thoughts
- **Behavior**: Suicidal behaviors
- **Attempt**: Suicide attempts

We compare multiple approaches:
- Fine-tuned RoBERTa models with various loss functions (cross-entropy, ordinal loss, Soft-F1)
- State-of-the-art LLMs (Grok-3, Gemini 2.5, DeepSeek)
- Pseudo-labeling techniques using LLM outputs

## 📁 Repository Structure

```
.
├── preprocessing.ipynb        # Data preprocessing pipeline
├── requirements.txt           # Python dependencies
├── config.py                  # API configuration management
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── SETUP.md                  # Detailed setup instructions
└── src/
    ├── func.py               # Shared utility functions and prompts
    ├── K-fold/              # RoBERTa k-fold cross-validation experiments
    │   ├── cross-entropy--*/
    │   ├── ordinal--*/
    │   ├── soft_f1--*/
    │   └── large-*/         # Larger model variants
    └── NO-fold/             # LLM prompting and pseudo-labeling
        ├── Google/          # Gemini API scripts
        ├── DeepSeek/        # DeepSeek API scripts
        ├── Grok/            # xAI/Grok API scripts
        └── ...              # Combined results and analysis
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) API keys for LLM experiments: xAI, Google, DeepSeek

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ac-ciano/MSc-dissertation.git
   cd MSc-dissertation
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys** (for LLM experiments)
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## 🔑 API Configuration

This project uses a centralized, secure configuration system for managing API keys:

```python
# config.py handles all API credentials
from config import Config

# Access API configurations
xai_config = Config.get_xai_config()
google_config = Config.get_google_config()
deepseek_config = Config.get_deepseek_config()
```

**Security Features:**
- ✅ Environment variables via `.env` file
- ✅ Automatic `.gitignore` protection
- ✅ API key validation
- ✅ No hardcoded credentials

## 📊 Data Access

The dataset is not publicly available and requires a Data Use Agreement (DUA).

**To obtain access:**
1. Contact: **hialex.li@connect.polyu.hk**
2. CC: **xinhong.chen@my.cityu.edu.hk**
3. Sign the provided DUA
4. Follow preprocessing instructions in `preprocessing.ipynb`

## 🧪 Experiments

### RoBERTa Fine-Tuning (K-Fold Cross-Validation)

The project explores various loss functions and model sizes:

- **Cross-Entropy Loss**: Standard classification
- **Ordinal Loss**: Leverages ordinal nature of risk levels
- **Soft-F1 Loss**: Optimizes for F1 score directly
- **Large Models**: RoBERTa-large variants

Example:
```bash
cd src/K-fold/large-cross-entropy--5e-5
python finetune_roberta.py
```

### LLM Inference

Prompt-based evaluation using state-of-the-art models:

```bash
# Grok-3 (xAI)
cd src/NO-fold/Grok
python grok-3.py

# Gemini 2.5 Flash (Google)
cd src/NO-fold/Google
python gemini-2.5-flash.py

# DeepSeek Reasoner
cd src/NO-fold/DeepSeek
python deepseek-reasoner.py
```

### Pseudo-Labeling

Combine LLM predictions to generate pseudo-labels:

```bash
cd src/NO-fold
python alldata_create_combined_csv.py
```

## 📈 Results

Detailed results and metrics are stored in:
- `src/K-fold/results.ipynb` - Cross-validation analysis
- `src/*_metrics.csv` - Performance metrics
- `src/figures/` - Visualizations

Key findings include model comparisons across:
- Accuracy
- Macro F1-score
- Per-class performance (Precision, Recall, F1)
- Confusion matrices

## 🛠️ Key Components

### Configuration (`config.py`)
Centralized API key management with validation and environment variable loading.

### Utilities (`src/func.py`)
Shared functions including:
- Data loading utilities
- Prompt engineering templates
- Label extraction from LLM responses
- Evaluation metrics

### Training Scripts
- `finetune_roberta*.py` - Model training with different loss functions
- `evaluate_roberta*.py` - Model evaluation
- `*-alldata.py` - Final model training on full dataset

### Analysis Notebooks
- `plot_roberta_metrics.ipynb` - RoBERTa performance visualization
- `plot_LLMs_metrics.ipynb` - LLM comparison
- `test_set_evaluate.ipynb` - Test set analysis

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{ciano2025suicide,
  title={Suicide Risk Prediction in Social Media using Machine Learning and Large Language Models},
  author={[Your Name]},
  year={2025},
  school={[Your University]}
}
```

## 🤝 Contributing

This is an academic research project. For questions or suggestions:
- Open an issue on GitHub
- Contact the repository maintainer

## ⚠️ Ethical Considerations

This research deals with sensitive mental health data. Please ensure:
- Compliance with data use agreements
- Ethical review board approval for any adaptations
- Responsible use of suicide risk prediction models
- Privacy protection for individuals

## 📄 License

[Specify your license here - e.g., MIT, GPL, etc.]

## 🙏 Acknowledgments

- Data providers: [Specify if allowed by DUA]
- Funding sources: [If applicable]
- Computational resources: [If applicable]

---

**⚠️ Disclaimer**: This project is for research purposes only. The models and predictions should not be used as substitutes for professional mental health assessment or intervention.
