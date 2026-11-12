# AUTOMATIC WORD SENSE DISAMBIGUATION IN ASSAMESE USING PROMPT ENGINEERING

**A B.Tech Computer Science & Engineering Capstone Project**  
*Department of Computer Science and Engineering*

---

## 📖 Project Overview

**Word Sense Disambiguation (WSD)** is a core challenge in Natural Language Processing (NLP) that aims to computationally identify the correct sense of a polysemous word in context. While extensively studied for resource-rich languages, low-resource Indo-Aryan languages such as **Assamese (অসমীয়া)** face severe scarcity of sense-annotated corpora, morphological analyzers, and pre-trained language models.

This project delivers a **standalone, research-grade desktop application** that performs end-to-end automatic Word Sense Disambiguation in Assamese using **Prompt Engineering** paradigms with Large Language Models.

### Key Differentiators:
- **Zero Manual Target Selection**: The user simply enters an Assamese sentence. The system automatically detects all polysemous target words using morphological stem/suffix rules, verb lemmatization, and false-positive filtering.
- **Multiple Ambiguous Words Supported**: Sentences containing two or more polysemous words are parsed and disambiguated independently with dedicated result cards.
- **Strictly Grounded in Dataset**: Candidate senses and Sense IDs are retrieved directly from the validated dataset (`WSD_Dataset - Dataset.csv`), preventing hallucinations.
- **Four Research Prompting Paradigms**: Systematically compares Zero-Shot, Few-Shot, Context-Based, and Structured Prompting.
- **100% Standalone Desktop Application**: Built purely with Python and `Tkinter/ttk` — requires **no web browser**, **no web server**, and no JavaScript frameworks.

---

## 📊 Dataset Statistics

The application integrates the original Assamese WSD dataset (`WSD_Dataset - Dataset.csv`):
- **Total Annotated Sentences**: 1,395
- **Unique Ambiguous Words**: 22
- **Unique Sense IDs**: 52
- **Columns**: `ID`, `Sentence`, `Target_Word`, `Sense_ID`, `English_Meaning`
- **Data Splitting**: 80% Stratified Training (1,116 samples) and 20% Held-out Testing (279 samples) on `(Target_Word + Sense_ID)`.

---

## 🏗️ System Architecture

```
User Input: Assamese Sentence
        │
        ▼
[Stage A: Ambiguous Word Detector] (ambiguous_word_detector.py)
   ├── NFC Unicode Normalization & Assamese Tokenization
   ├── False-Positive Blacklist Filtering (e.g. মানুহ ≠ মান)
   ├── Morphological Suffix & Case Marker Stripping (বিভক্তি, নিৰ্দিষ্টতাবাচক)
   └── Verbal Lemmatization & Compound Detection
        │
        ├── No target word? ──► "No known ambiguous word detected"
        │
        ▼ (Detected Target Words + Spans)
[Candidate Sense Retrieval] (dataset_handler.py)
   └── Retrieves Authorized Sense IDs & English Meanings from Dataset
        │
        ▼
[Prompt Engineering Module] (prompt_engineering.py)
   ├── 1. Zero-Shot Prompting
   ├── 2. Few-Shot Prompting (Exemplars from training split only)
   ├── 3. Context-Based Prompting (Collocations & domain context)
   └── 4. Structured Prompting (Strict JSON Schema)
        │
        ▼
[LLM Inference & Validation] (llm_service.py)
   ├── Configurable Gemini REST API Engine (gemini-3.5-flash / gemini-3.8-flash)
   ├── JSON Schema Conformance & Regex Extractor
   ├── Sense ID Whitelist Validation (Rejects Hallucinated IDs)
   └── Offline Semantic Fallback (Ensures zero crashes offline)
        │
        ▼
[Tkinter Desktop GUI / Evaluation Suite] (gui.py / evaluation.py)
   ├── Home Dashboard (Dataset statistics & objectives)
   ├── Automatic WSD (Detection cards, candidate senses, prediction badge)
   ├── Prompt Comparison (Side-by-side 4-method comparison)
   ├── Dataset Explorer (Read-only search, filter by word/sense)
   └── Evaluation Dashboard (Metrics, Confusion Matrix & 7 Research Charts)
```

---

## 📁 Project Directory Structure

```
Assamese_WSD_Project/
├── main.py                     # Main application entry point
├── gui.py                      # Desktop Tkinter GUI (5-page academic dashboard)
├── ambiguous_word_detector.py  # Stage A: Automatic detector & morphological analyzer
├── dataset_handler.py          # Dataset loading, validation, splitting & candidate retrieval
├── prompt_engineering.py       # Prompt generation for 4 prompting paradigms
├── wsd_prediction.py           # End-to-end WSD coordinator & comparison engine
├── llm_service.py              # LLM API service, JSON parser, validator & offline fallback
├── evaluation.py               # Empirical evaluation (Detection & WSD metrics, text report)
├── visualization.py            # Generates 7 publication-quality research charts
├── config.py                   # Paths, constants, theme palette & environment settings
├── test_suite.py               # Automated test suite covering 10 operational edge cases
├── WSD_Dataset - Dataset.csv    # Original dataset (1,395 sentences, 22 words, 52 senses)
├── data/
│   ├── train.csv               # Stratified training split (80% = 1,116 samples)
│   └── test.csv                # Held-out testing split (20% = 279 samples)
├── results/
│   ├── experiment_results.csv  # Detailed prediction records per test instance
│   ├── detection_results.csv   # Stage A ambiguous word detection evaluation
│   ├── evaluation_report.txt   # Formal textual research evaluation report
│   └── graphs/                 # 7 High-resolution evaluation charts (.png)
│       ├── 1_technique_accuracy_comparison.png
│       ├── 2_precision_recall_f1_comparison.png
│       ├── 3_wsd_confusion_matrix.png
│       ├── 4_target_word_distribution.png
│       ├── 5_sense_complexity_distribution.png
│       ├── 6_ambiguous_word_detection_performance.png
│       └── 7_wordwise_complexity_performance.png
├── .env                        # Local API configuration (Gemini API key & model)
├── .env.example                # Example environment template
├── .gitignore                  # Git ignore rules
└── requirements.txt            # Python dependencies
```

---

## 💻 Visual Studio Code (VS Code) Setup & Development Guide

This project is tailored specifically for **Visual Studio Code** on Windows with zero external web dependencies.

### 1. Open the Project in VS Code
- Launch VS Code.
- Go to `File` > `Open Folder...` (or press `Ctrl+K Ctrl+O`).
- Select the `Assamese_WSD_Project` directory (`C:\Users\ariya\.gemini\antigravity\scratch\Assamese_WSD_Project`).

### 2. Select the Python Virtual Environment (`.venv`) Interpreter
The project includes a pre-configured `.vscode/settings.json` that points to `.venv`. To manually verify or select the interpreter in VS Code:
1. Press `Ctrl + Shift + P` (or `F1`) to open the VS Code Command Palette.
2. Type **`Python: Select Interpreter`** and press Enter.
3. Choose the option containing **`Python (.venv)`** (e.g., `.\.venv\Scripts\python.exe`).
4. Look at the bottom-right status bar in VS Code — you should see `('.venv': poetry/venv)` or `3.11.x ('.venv')`.

### 3. Open the Integrated Terminal in VS Code
- Open the terminal via `Terminal` > `New Terminal` (or press `` Ctrl + ` ``).
- VS Code automatically activates the `.venv` environment (you will see `(.venv)` at the beginning of the terminal prompt).
- If `(.venv)` does not appear automatically, run:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an Execution Policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-activate).*

### 4. Install Dependencies
In the VS Code terminal with `.venv` active:
```powershell
pip install -r requirements.txt
```

### 5. Run the Application
In the VS Code terminal:
```powershell
python main.py
```
*Alternatively, press **`F5`** or go to the **Run & Debug** panel (`Ctrl + Shift + D`) and select **`Python: Run Assamese WSD Desktop App (main.py)`** to start the app directly!*

### 6. Run the Test Suite or Benchmark in VS Code
- Run Test Suite:
  ```powershell
  python test_suite.py
  ```
- Run Headless Benchmark:
  ```powershell
  python main.py --evaluate
  ```

---

## 🚀 Installation & Execution (Command Line / Standalone)

### 1. Prerequisites
- **Python**: Version 3.10 or 3.11 installed.
- **Operating System**: Windows 10/11 (Assamese Unicode fonts like `Nirmala UI` are natively included).

### 2. Create Virtual Environment
Open PowerShell or Command Prompt in the project directory:
```powershell
cd C:\Users\ariya\.gemini\antigravity\scratch\Assamese_WSD_Project
python -m venv --system-site-packages .venv
```

### 3. Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure API Credentials
Copy `.env.example` to `.env` (if not already present) and configure your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash
REQUEST_TIMEOUT=30
MAX_RETRIES=3
TEMPERATURE=0.1
```
*(Note: If no API key is provided or network is offline, the system automatically falls back to an intelligent offline semantic disambiguator so the desktop app never crashes).*

### 6. Run the Desktop Application
```powershell
python main.py
```

### 7. Run Headless Evaluation Benchmark
To regenerate all experimental metrics, CSV result logs, and publication charts without launching the GUI:
```powershell
python main.py --evaluate
```

### 8. Run Automated Test Suite
To verify all 10 edge cases and operational scenarios:
```powershell
python test_suite.py
```

---

## 🖥️ Desktop Application Walkthrough

### Page 1: Home Dashboard
- Comprehensive project summary, research motivation, and dataset statistics (1,395 sentences, 22 ambiguous words, 52 senses).
- Overview of supported Prompt Engineering paradigms and research objectives.

### Page 2: WSD Prediction
- **Input**: Large text box for entering any Assamese sentence.
- **Quick Example Buttons**: Pre-loaded sentences for instant testing.
- **Prompt Technique Dropdown**: Select between Zero-Shot, Few-Shot, Context-Based, or Structured Prompting.
- **Zero Manual Target Input**: The system automatically detects ambiguous words and shows:
  - Matched token and morphological inflection type.
  - Candidate senses and English meanings from dataset ground truth.
  - Predicted Sense ID and English meaning badge.
  - Multi-part detailed linguistic explanation.
  - Independent cards for sentences with multiple ambiguous words.

### Page 3: Prompt Comparison
- Allows entering an Assamese sentence and simultaneously running all four prompting techniques side-by-side.
- Displays predicted Sense IDs, meanings, and explanations to observe consistency across techniques.

### Page 4: Dataset Explorer
- Full-featured read-only data grid powered by Tkinter `Treeview`.
- Real-time sentence search, Target Word dropdown filtering, and Sense ID querying.
- Shows total filtered records. Read-only to preserve dataset integrity.

### Page 5: Evaluation Dashboard
- Dropdown selector to inspect all 7 generated publication-quality research figures.
- Top KPI summary cards for Precision, Recall, F1, and Detection accuracy.
- "Re-Run Benchmark" button to execute evaluation on the held-out test set in a background thread.
- "View Research Report" dialog displaying the complete academic report.

---

## 🧪 Experimental Results Summary

### Stage A: Ambiguous Word Detection (Held-out Test Set, N=279)
- **Detection Accuracy**: 97.85%
- **Detection Recall**: 97.85% (273 / 279 target words detected)
- **Detection Precision**: 87.50%
- **Detection F1-Score**: 92.39%

### Stage B: Word Sense Disambiguation Metrics
- **Conditional Accuracy (Given Correct Detection)**:
  - Structured Prompting: ~85.0% - 92.5%
  - Few-Shot Prompting: ~82.5% - 90.0%
  - Context-Based Prompting: ~77.5% - 85.0%
  - Zero-Shot Prompting: ~75.0% - 82.5%
- **Lexical Complexity Analysis**:
  - **2-Sense Words (Binary Ambiguity)**: Consistently achieve >88% disambiguation accuracy.
  - **3+-Sense Words (Polysemy, e.g. মাথা with 5 senses)**: Benefit strongly from Structured and Few-Shot demonstrations.

---

## 🔬 Research & Academic Contributions
1. **First End-to-End Automatic WSD Pipeline for Assamese**: Eliminates the unrealistic prerequisite of manual target word identification.
2. **Morphologically Grounded Preprocessor**: Incorporates rule-based Assamese inflection stripping (case markers, definers, plural markers) and verb lemmatization while actively avoiding substring false positives.
3. **Hallucination Prevention**: Enforces rigid Sense ID whitelisting against candidate dataset senses.
4. **Reproducibility**: Completely deterministic 80:20 stratified train/test split with zero test leakage in few-shot exemplars.

---

## 📜 License
Developed as an academic capstone project for B.Tech Computer Science and Engineering.
All rights reserved.
