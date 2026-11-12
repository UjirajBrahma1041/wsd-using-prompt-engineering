"""
Configuration module for Assamese Word Sense Disambiguation (WSD) Project.
Handles file paths, environment variables, model configurations, and UI styling constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Define Project Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Ensure Matplotlib config dir is writable
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib_cache")

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# File and Directory Paths
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

ORIGINAL_DATASET_PATH = BASE_DIR / "WSD_Dataset - Dataset.csv"
TRAIN_DATASET_PATH = DATA_DIR / "train.csv"
TEST_DATASET_PATH = DATA_DIR / "test.csv"

EXPERIMENT_RESULTS_PATH = RESULTS_DIR / "experiment_results.csv"
DETECTION_RESULTS_PATH = RESULTS_DIR / "detection_results.csv"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.txt"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3.5-flash")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "8"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Prompting Technique Constants
PROMPT_ZERO_SHOT = "Zero-Shot Prompting"
PROMPT_FEW_SHOT = "Few-Shot Prompting"
PROMPT_CONTEXT_BASED = "Context-Based Prompting"
PROMPT_STRUCTURED = "Structured Prompting"

PROMPT_TECHNIQUES = [
    PROMPT_ZERO_SHOT,
    PROMPT_FEW_SHOT,
    PROMPT_CONTEXT_BASED,
    PROMPT_STRUCTURED,
]

# Random Seed for Reproducibility
RANDOM_SEED = 42

# UI Theme / Palette (Professional Academic Dashboard)
THEME = {
    "primary": "#1E3A8A",       # Deep Academic Blue
    "primary_light": "#3B82F6", # Bright Blue
    "primary_dark": "#172554",  # Very dark blue
    "secondary": "#0284C7",     # Cerulean Blue
    "accent": "#0D9488",        # Teal accent
    "bg_main": "#F8FAFC",       # Slate 50 background
    "bg_card": "#FFFFFF",       # White card background
    "bg_sidebar": "#0F172A",    # Dark Slate 900 sidebar
    "sidebar_text": "#F1F5F9",  # Slate 100 text for sidebar
    "sidebar_active": "#1E293B",# Active sidebar button
    "sidebar_hover": "#334155", # Hover sidebar button
    "text_dark": "#0F172A",     # Primary dark text
    "text_muted": "#64748B",    # Muted secondary text
    "border": "#E2E8F0",        # Slate 200 border
    "success": "#16A34A",       # Green 600
    "warning": "#D97706",       # Amber 600
    "danger": "#DC2626",        # Red 600
    "font_family": "Segoe UI",  # Clean Windows font
    "font_assamese": "Nirmala UI", # Best Assamese rendering font on Windows
}
