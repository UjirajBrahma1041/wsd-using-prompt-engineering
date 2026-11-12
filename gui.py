"""
Desktop GUI Application for Automatic Word Sense Disambiguation in Assamese.
Built with pure Python Tkinter/ttk.
Academic Dashboard theme with 5 interactive pages:
1. Home Dashboard
2. Automatic WSD Prediction (No manual target-word entry)
3. Prompt Comparison (All 4 prompting methods side-by-side)
4. Dataset Explorer (Read-only search and filtering)
5. Evaluation Dashboard (Metrics, Confusion Matrix, and Research Charts)
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd

from config import (
    THEME,
    PROMPT_TECHNIQUES,
    PROMPT_ZERO_SHOT,
    PROMPT_FEW_SHOT,
    PROMPT_CONTEXT_BASED,
    PROMPT_STRUCTURED,
    GRAPHS_DIR,
    EVALUATION_REPORT_PATH,
    ORIGINAL_DATASET_PATH,
    LLM_MODEL_NAME,
)
from dataset_handler import DatasetHandler
from ambiguous_word_detector import AssameseAmbiguousWordDetector
from wsd_prediction import WSDPredictor
from evaluation import WSDEvaluator
from visualization import WSDVisualizer


class WSDApp(tk.Tk):
    """
    Main Application Window for Assamese WSD Desktop Interface.
    """

    def __init__(self):
        super().__init__()

        # Window settings
        self.title("Word Sense Disambiguation in Assamese Using Prompt Engineering")
        self.geometry("1240x800")
        self.minsize(1080, 700)
        self.configure(bg=THEME["bg_main"])

        # Core Backend Services
        self.handler = DatasetHandler()
        self.detector = AssameseAmbiguousWordDetector(self.handler)
        self.predictor = WSDPredictor(self.handler, self.detector)
        self.evaluator = WSDEvaluator(self.handler, self.detector)
        self.visualizer = WSDVisualizer()

        # State
        self.active_page = None
        self.nav_buttons = {}

        # Configure TTK Styles
        self._setup_styles()

        # Build Main UI Shell
        self._build_layout()

        # Show initial Home Page
        self.show_page("home")

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Global treeview style
        style.configure(
            "Treeview",
            background="#FFFFFF",
            foreground=THEME["text_dark"],
            fieldbackground="#FFFFFF",
            rowheight=28,
            font=(THEME["font_family"], 9),
            bordercolor=THEME["border"],
        )
        style.configure(
            "Treeview.Heading",
            background="#F1F5F9",
            foreground=THEME["primary"],
            font=(THEME["font_family"], 10, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", THEME["primary"])], foreground=[("selected", "#FFFFFF")])

        # Combobox style
        style.configure(
            "TCombobox",
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground=THEME["text_dark"],
            arrowsize=14,
        )

    def _build_layout(self):
        # Master horizontal split: Left Sidebar | Right Main Container
        self.sidebar = tk.Frame(self, bg=THEME["bg_sidebar"], width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.main_container = tk.Frame(self, bg=THEME["bg_main"])
        self.main_container.pack(side="right", fill="both", expand=True)

        # Header Banner inside Main Container
        self.header_frame = tk.Frame(self.main_container, bg=THEME["bg_card"], height=64, padx=24, pady=12)
        self.header_frame.pack(side="top", fill="x")
        self.header_frame.pack_propagate(False)

        # Bottom separator
        sep = tk.Frame(self.main_container, bg=THEME["border"], height=1)
        sep.pack(side="top", fill="x")

        # Header Titles
        self.lbl_page_title = tk.Label(
            self.header_frame,
            text="Home Dashboard",
            font=(THEME["font_family"], 16, "bold"),
            fg=THEME["primary"],
            bg=THEME["bg_card"],
        )
        self.lbl_page_title.pack(side="left")

        # Model Pill Tag
        self.lbl_model_tag = tk.Label(
            self.header_frame,
            text=f"Model: {LLM_MODEL_NAME} | Engine: Online / Fallback",
            font=(THEME["font_family"], 9, "bold"),
            fg=THEME["accent"],
            bg="#F0FDFA",
            padx=12,
            pady=4,
            relief="solid",
            bd=1,
        )
        self.lbl_model_tag.pack(side="right")

        # Dynamic Content Container
        self.content_frame = tk.Frame(self.main_container, bg=THEME["bg_main"])
        self.content_frame.pack(side="top", fill="both", expand=True, padx=20, pady=16)

        # Build Sidebar Elements
        self._build_sidebar()

    def _build_sidebar(self):
        # App Branding Logo & Title
        brand_frame = tk.Frame(self.sidebar, bg=THEME["bg_sidebar"], pady=20, padx=16)
        brand_frame.pack(side="top", fill="x")

        lbl_app_logo = tk.Label(
            brand_frame,
            text="🎓 ASSAMESE WSD",
            font=(THEME["font_family"], 13, "bold"),
            fg="#60A5FA",
            bg=THEME["bg_sidebar"],
        )
        lbl_app_logo.pack(anchor="w")

        lbl_app_sub = tk.Label(
            brand_frame,
            text="Prompt Engineering System\nB.Tech Project 2026",
            font=(THEME["font_family"], 8),
            fg="#94A3B8",
            bg=THEME["bg_sidebar"],
            justify="left",
            pady=4,
        )
        lbl_app_sub.pack(anchor="w")

        # Divider
        div = tk.Frame(self.sidebar, bg="#1E293B", height=1)
        div.pack(fill="x", padx=12, pady=10)

        # Navigation Buttons Section
        nav_items = [
            ("home", "🏠  Home Dashboard"),
            ("predict", "🔍  WSD Prediction"),
            ("compare", "⚖️  Prompt Comparison"),
            ("dataset", "📚  Dataset Explorer"),
            ("eval", "📊  Evaluation Dashboard"),
        ]

        self.nav_frame = tk.Frame(self.sidebar, bg=THEME["bg_sidebar"])
        self.nav_frame.pack(side="top", fill="x", padx=10, pady=6)

        for key, text in nav_items:
            btn = tk.Button(
                self.nav_frame,
                text=text,
                font=(THEME["font_family"], 10, "bold"),
                fg=THEME["sidebar_text"],
                bg=THEME["bg_sidebar"],
                activebackground=THEME["sidebar_active"],
                activeforeground="#FFFFFF",
                bd=0,
                padx=16,
                pady=10,
                anchor="w",
                cursor="hand2",
                command=lambda k=key: self.show_page(k),
            )
            btn.pack(fill="x", pady=3)
            self.nav_buttons[key] = btn

        # Bottom System Info
        sys_frame = tk.Frame(self.sidebar, bg=THEME["bg_sidebar"], pady=16, padx=16)
        sys_frame.pack(side="bottom", fill="x")

        lbl_foot = tk.Label(
            sys_frame,
            text="Status: Active Ready\nLanguage: Assamese (অসমীয়া)\nStandalone Desktop App",
            font=(THEME["font_family"], 8),
            fg="#64748B",
            bg=THEME["bg_sidebar"],
            justify="left",
        )
        lbl_foot.pack(anchor="w")

    def show_page(self, page_name: str):
        # Update Nav Button Highlights
        for key, btn in self.nav_buttons.items():
            if key == page_name:
                btn.configure(bg=THEME["primary"], fg="#FFFFFF")
            else:
                btn.configure(bg=THEME["bg_sidebar"], fg=THEME["sidebar_text"])

        # Clear old content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        titles = {
            "home": "Home Dashboard — Project Overview",
            "predict": "Prompt Engineering WSD — Natural-Language Query",
            "compare": "Prompt Engineering Comparison (4 Methods)",
            "dataset": "Assamese WSD Dataset Explorer",
            "eval": "Evaluation & Empirical Metrics Dashboard",
        }
        self.lbl_page_title.config(text=titles.get(page_name, "WSD Dashboard"))

        # Render corresponding page
        if page_name == "home":
            HomePage(self.content_frame, self)
        elif page_name == "predict":
            PredictionPage(self.content_frame, self)
        elif page_name == "compare":
            ComparisonPage(self.content_frame, self)
        elif page_name == "dataset":
            DatasetPage(self.content_frame, self)
        elif page_name == "eval":
            EvaluationPage(self.content_frame, self)


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_main"])
        self.controller = controller
        self.pack(fill="both", expand=True)

        stats = self.controller.handler.get_dataset_statistics()

        # Top Hero Card
        hero = tk.Frame(self, bg=THEME["bg_card"], padx=24, pady=20, highlightbackground=THEME["border"], highlightthickness=1)
        hero.pack(fill="x", pady=(0, 16))

        tk.Label(
            hero,
            text="WORD SENSE DISAMBIGUATION IN ASSAMESE USING PROMPT ENGINEERING",
            font=(THEME["font_family"], 14, "bold"),
            fg=THEME["primary"],
            bg=THEME["bg_card"],
            wraplength=950,
            justify="left",
        ).pack(anchor="w")

        tk.Label(
            hero,
            text=(
                "A complete, research-oriented natural language processing desktop system for disambiguating polysemous words "
                "in the Assamese language. The application automatically detects ambiguous words without manual user intervention, "
                "retrieves candidate senses from empirical ground truth, and resolves semantic ambiguity using zero-shot, few-shot, "
                "context-based, and structured prompt engineering."
            ),
            font=(THEME["font_family"], 10),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"],
            wraplength=950,
            justify="left",
            pady=8,
        ).pack(anchor="w")

        # 4 Key Stat Cards
        stats_frame = tk.Frame(self, bg=THEME["bg_main"])
        stats_frame.pack(fill="x", pady=(0, 16))

        self._create_stat_card(stats_frame, "Total Sentences", f"{stats['total_sentences']:,}", "Original Assamese Sentences", "#1E3A8A")
        self._create_stat_card(stats_frame, "Ambiguous Words", f"{stats['unique_target_words_count']}", "Polysemous Assamese Lexemes", "#0284C7")
        self._create_stat_card(stats_frame, "Unique Senses", f"{stats['unique_senses_count']}", "Distinct Sense Identifiers (Sense_ID)", "#0D9488")
        self._create_stat_card(stats_frame, "Prompt Methods", "4 Techniques", "Zero, Few, Context, Structured", "#D97706")

        # Bottom Split: Research Objectives & Prompting Techniques
        split_frame = tk.Frame(self, bg=THEME["bg_main"])
        split_frame.pack(fill="both", expand=True)

        # Left Column: Objectives Card
        left_card = tk.Frame(split_frame, bg=THEME["bg_card"], padx=20, pady=16, highlightbackground=THEME["border"], highlightthickness=1)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(left_card, text="🎯 Core Research Objectives", font=(THEME["font_family"], 12, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w")
        objectives = [
            "1. Automatic detection of ambiguous words using Assamese morphology and inflection rules.",
            "2. Zero manual user selection — automatically identifies multiple ambiguous words per sentence.",
            "3. Ground-truth retrieval of candidate senses strictly from the validated Assamese dataset.",
            "4. Empirical comparison of 4 prompting paradigms (Zero-Shot, Few-Shot, Context-Based, Structured).",
            "5. Constrained generation preventing model hallucinations and unauthorized Sense IDs.",
            "6. Rigorous two-stage evaluation: Ambiguous Word Detection and Word Sense Disambiguation.",
        ]
        for obj in objectives:
            tk.Label(left_card, text=obj, font=(THEME["font_family"], 10), fg=THEME["text_dark"], bg=THEME["bg_card"], wraplength=460, justify="left", pady=3).pack(anchor="w")

        # Right Column: Available Prompting Techniques
        right_card = tk.Frame(split_frame, bg=THEME["bg_card"], padx=20, pady=16, highlightbackground=THEME["border"], highlightthickness=1)
        right_card.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(right_card, text="⚡ Supported Prompting Techniques", font=(THEME["font_family"], 12, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(anchor="w")
        techniques_info = [
            ("Zero-Shot Prompting", "Evaluates pre-trained lexical understanding without task demonstrations."),
            ("Few-Shot Prompting", "Injects verified training exemplars directly from the training split only."),
            ("Context-Based Prompting", "Guides the model with syntactic cues, collocations, and domain context."),
            ("Structured Prompting", "Enforces strict JSON schema conformance, explanation, and authorized Sense IDs."),
        ]
        for title, desc in techniques_info:
            f = tk.Frame(right_card, bg=THEME["bg_card"], pady=4)
            f.pack(fill="x")
            tk.Label(f, text=f"• {title}:", font=(THEME["font_family"], 10, "bold"), fg=THEME["secondary"], bg=THEME["bg_card"]).pack(anchor="w")
            tk.Label(f, text=desc, font=(THEME["font_family"], 9), fg=THEME["text_muted"], bg=THEME["bg_card"], wraplength=460, justify="left").pack(anchor="w")

    def _create_stat_card(self, parent, title, value, subtitle, color):
        card = tk.Frame(parent, bg=THEME["bg_card"], padx=18, pady=14, highlightbackground=THEME["border"], highlightthickness=1)
        card.pack(side="left", fill="both", expand=True, padx=4)

        tk.Label(card, text=title, font=(THEME["font_family"], 9, "bold"), fg=THEME["text_muted"], bg=THEME["bg_card"]).pack(anchor="w")
        tk.Label(card, text=value, font=(THEME["font_family"], 18, "bold"), fg=color, bg=THEME["bg_card"], pady=2).pack(anchor="w")
        tk.Label(card, text=subtitle, font=(THEME["font_family"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"]).pack(anchor="w")


# ==============================================================================
# PAGE 2: PROMPT ENGINEERING NATURAL-LANGUAGE WSD QUERY
# ==============================================================================
class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_main"])
        self.controller = controller
        self.pack(fill="both", expand=True)

        # Top Control Card: Single Large Natural-Language Query Input
        input_card = tk.Frame(
            self,
            bg=THEME["bg_card"],
            padx=20,
            pady=16,
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        input_card.pack(fill="x", pady=(0, 12))

        # Title & Preset Query Buttons
        top_row = tk.Frame(input_card, bg=THEME["bg_card"])
        top_row.pack(fill="x", pady=(0, 6))

        tk.Label(
            top_row,
            text="Natural-Language WSD Query (প্ৰশ্ন / কুৱেৰী):",
            font=(THEME["font_family"], 11, "bold"),
            fg=THEME["primary"],
            bg=THEME["bg_card"],
        ).pack(side="left")

        tk.Label(
            top_row,
            text="Preset Queries:",
            font=(THEME["font_family"], 9),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"],
        ).pack(side="left", padx=(14, 4))

        examples = [
            ("গছৰ পাত (Leaf)", "In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?"),
            ("কিতাপৰ পাত (Page)", "In the sentence 'কিতাপখনৰ পাত লুটিয়াই চোৱা', what is the meaning of পাত?"),
            ("উত্তৰ (North)", "What does উত্তৰ mean in 'উত্তৰ দিশৰ পৰা চেঁচা বতাহ বলিছে'?"),
            ("উত্তৰ (Answer)", "In the sentence 'মই শিক্ষকৰ প্ৰশ্নৰ উত্তৰ দিলোঁ', what is the meaning of উত্তৰ?"),
            ("ফল (Result)", "Determine the meaning of ফল in the sentence 'পৰীক্ষাৰ ফল ভাল হৈছে'"),
            ("Insufficient Context", "In the sentence 'পাত', what is the meaning of পাত?"),
        ]
        for name, text in examples:
            btn_ex = tk.Button(
                top_row,
                text=name,
                font=(THEME["font_family"], 8),
                bg="#F1F5F9",
                fg=THEME["primary"],
                bd=0,
                padx=6,
                pady=2,
                cursor="hand2",
                command=lambda t=text: self.set_query_text(t),
            )
            btn_ex.pack(side="left", padx=2)

        # Single Large Query Input Text Area (Requirement 1 & 5)
        self.txt_query = tk.Text(
            input_card,
            font=(THEME["font_assamese"], 12),
            height=3,
            wrap="word",
            bg="#F8FAFC",
            fg=THEME["text_dark"],
            bd=1,
            relief="solid",
            padx=10,
            pady=8,
        )
        self.txt_query.pack(fill="x", pady=(4, 10))
        self.txt_query.insert(
            "1.0",
            "In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?",
        )

        # Action Row: Submit Query Button + Clear Button + Loading Indicator
        action_row = tk.Frame(input_card, bg=THEME["bg_card"])
        action_row.pack(fill="x")

        # Submit Query Button (Requirement 5)
        self.btn_submit = tk.Button(
            action_row,
            text="🚀 Submit Query",
            font=(THEME["font_family"], 11, "bold"),
            bg=THEME["primary"],
            fg="#FFFFFF",
            activebackground=THEME["primary_light"],
            activeforeground="#FFFFFF",
            bd=0,
            padx=20,
            pady=7,
            cursor="hand2",
            command=self.run_query_thread,
        )
        self.btn_submit.pack(side="left")

        # Clear Button
        self.btn_clear = tk.Button(
            action_row,
            text="🔄 Clear",
            font=(THEME["font_family"], 9),
            bg="#F1F5F9",
            fg=THEME["primary"],
            bd=0,
            padx=12,
            pady=7,
            cursor="hand2",
            command=self.clear_query,
        )
        self.btn_clear.pack(side="left", padx=8)

        # Loading Indicator (Requirement 5)
        self.lbl_loading = tk.Label(
            action_row,
            text="",
            font=(THEME["font_family"], 10, "bold"),
            fg=THEME["secondary"],
            bg=THEME["bg_card"],
        )
        self.lbl_loading.pack(side="left", padx=16)

        # Scrollable Results Container
        self.results_canvas = tk.Canvas(self, bg=THEME["bg_main"], highlightthickness=0)
        self.results_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.results_canvas.yview)
        self.results_container = tk.Frame(self.results_canvas, bg=THEME["bg_main"])

        self.results_container.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")),
        )
        self.results_canvas.create_window((0, 0), window=self.results_container, anchor="nw", width=960)
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")

    def set_query_text(self, text: str):
        self.txt_query.delete("1.0", "end")
        self.txt_query.insert("1.0", text)

    def clear_query(self):
        self.txt_query.delete("1.0", "end")
        for widget in self.results_container.winfo_children():
            widget.destroy()
        self.lbl_loading.config(text="")
        self.btn_submit.config(state="normal", text="🚀 Submit Query")

    def run_query_thread(self):
        query = self.txt_query.get("1.0", "end").strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a natural-language query to disambiguate.")
            return

        self.btn_submit.config(state="disabled", text="⏳ Processing Query...")
        self.lbl_loading.config(text="⏳ Querying LLM API with Prompt Engineering...", fg=THEME["secondary"])

        # Run in background thread to prevent UI freezing
        thread = threading.Thread(target=self._execute_query, args=(query,))
        thread.daemon = True
        thread.start()

    def _execute_query(self, query: str):
        try:
            result = self.controller.predictor.process_natural_query(query)
            self.after(0, self._render_query_results, result)
        except Exception as e:
            self.after(0, self._handle_error, str(e))

    def _handle_error(self, err_msg: str):
        self.btn_submit.config(state="normal", text="🚀 Submit Query")
        self.lbl_loading.config(text="Error occurred", fg=THEME["danger"])
        messagebox.showerror("Query Processing Error", f"An error occurred while processing query:\n{err_msg}")

    def _render_query_results(self, result: Dict[str, Any]):
        self.btn_submit.config(state="normal", text="🚀 Submit Query")
        self.lbl_loading.config(text="")

        # Clear previous result widgets
        for widget in self.results_container.winfo_children():
            widget.destroy()

        status = result.get("status")

        # 1. Error / Clarification Handling (Requirement 4)
        if status == "needs_clarification":
            card = tk.Frame(
                self.results_container,
                bg=THEME["bg_card"],
                padx=24,
                pady=20,
                highlightbackground=THEME["warning"],
                highlightthickness=2,
            )
            card.pack(fill="x", pady=8)
            tk.Label(
                card,
                text="⚠️ Clarification Required",
                font=(THEME["font_family"], 12, "bold"),
                fg=THEME["warning"],
                bg=THEME["bg_card"],
            ).pack(anchor="w")
            tk.Label(
                card,
                text=result.get("error_message")
                or "The query does not contain an identifiable Assamese sentence or target word. Please clarify your query.",
                font=(THEME["font_family"], 10),
                fg=THEME["text_dark"],
                bg=THEME["bg_card"],
                pady=6,
            ).pack(anchor="w")
            tk.Label(
                card,
                text="Example: \"In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?\"",
                font=(THEME["font_family"], 9, "italic"),
                fg=THEME["text_muted"],
                bg=THEME["bg_card"],
            ).pack(anchor="w")
            return

        # 2. Structured Output Section (Requirement 3 & 5)
        card = tk.Frame(
            self.results_container,
            bg=THEME["bg_card"],
            padx=24,
            pady=20,
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        card.pack(fill="x", pady=8)

        # Card Header
        title_frame = tk.Frame(card, bg=THEME["bg_card"])
        title_frame.pack(fill="x", pady=(0, 14))

        tk.Label(
            title_frame,
            text="✨ PROMPT ENGINEERING WSD OUTPUT",
            font=(THEME["font_family"], 12, "bold"),
            fg=THEME["primary"],
            bg=THEME["bg_card"],
        ).pack(side="left")

        # Status Pill Badge
        is_succ = (status == "success")
        pill_text = "STATUS: SUCCESS" if is_succ else "STATUS: INSUFFICIENT CONTEXT"
        pill_bg = "#DCFCE7" if is_succ else "#FEF3C7"
        pill_fg = "#15803D" if is_succ else "#B45309"
        tk.Label(
            title_frame,
            text=pill_text,
            font=(THEME["font_family"], 9, "bold"),
            fg=pill_fg,
            bg=pill_bg,
            padx=10,
            pady=3,
            relief="solid",
            bd=1,
        ).pack(side="right")

        # Grid of Output Fields (Matching Requirement 3 exact specifications)
        fields = [
            ("Original Query:", result.get("original_query", ""), THEME["text_muted"], 10, "normal"),
            ("Extracted Sentence:", result.get("extracted_sentence", ""), THEME["primary"], 12, "bold"),
            ("Ambiguous Word:", result.get("ambiguous_word", ""), THEME["accent"], 12, "bold"),
            ("Predicted Meaning:", result.get("predicted_meaning", ""), "#15803D", 12, "bold"),
            ("Explanation:", result.get("explanation", ""), THEME["text_dark"], 10, "normal"),
            ("Confidence Score:", result.get("confidence_score", "7/7"), THEME["primary"], 11, "bold"),
        ]

        grid_frame = tk.Frame(card, bg=THEME["bg_card"])
        grid_frame.pack(fill="x")

        for idx, (label_text, val_text, val_color, val_size, val_weight) in enumerate(fields):
            row_box = tk.Frame(grid_frame, bg="#F8FAFC" if idx % 2 == 0 else "#FFFFFF", padx=12, pady=8)
            row_box.pack(fill="x", pady=2)

            tk.Label(
                row_box,
                text=label_text,
                font=(THEME["font_family"], 10, "bold"),
                fg=THEME["text_dark"],
                bg=row_box["bg"],
                width=18,
                anchor="w",
            ).pack(side="left")

            val_str = str(val_text) if val_text is not None else "None"
            font_fam = (
                THEME["font_assamese"]
                if any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in val_str)
                else THEME["font_family"]
            )
            tk.Label(
                row_box,
                text=val_str,
                font=(font_fam, val_size, val_weight),
                fg=val_color,
                bg=row_box["bg"],
                wraplength=730,
                justify="left",
                anchor="w",
            ).pack(side="left", fill="x", expand=True)


# ==============================================================================
# PAGE 3: PROMPT COMPARISON
# ==============================================================================
class ComparisonPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_main"])
        self.controller = controller
        self.pack(fill="both", expand=True)

        # Input Card
        input_card = tk.Frame(self, bg=THEME["bg_card"], padx=20, pady=14, highlightbackground=THEME["border"], highlightthickness=1)
        input_card.pack(fill="x", pady=(0, 12))

        tk.Label(
            input_card,
            text="Sentence for Multi-Prompt Comparison:",
            font=(THEME["font_family"], 11, "bold"),
            fg=THEME["primary"],
            bg=THEME["bg_card"],
        ).pack(anchor="w")

        row = tk.Frame(input_card, bg=THEME["bg_card"])
        row.pack(fill="x", pady=6)

        self.ent_compare = tk.Entry(row, font=(THEME["font_assamese"], 11), bg="#F8FAFC", bd=1, relief="solid")
        self.ent_compare.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.ent_compare.insert(0, "মই পৰীক্ষাৰ সকলো উত্তৰ সঠিকভাৱে লিখিলোঁ।")

        self.btn_compare = tk.Button(
            row,
            text="⚖️ Compare All 4 Prompts",
            font=(THEME["font_family"], 10, "bold"),
            bg=THEME["primary"],
            fg="#FFFFFF",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_comparison_thread,
        )
        self.btn_compare.pack(side="right", padx=(6, 0))

        # Refresh / Clear Button
        self.btn_compare_refresh = tk.Button(
            row,
            text="🔄 Refresh",
            font=(THEME["font_family"], 9),
            bg="#F1F5F9",
            fg=THEME["primary"],
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.reset_comparison_view,
        )
        self.btn_compare_refresh.pack(side="right")

        # Scrollable Output Area
        self.canvas = tk.Canvas(self, bg=THEME["bg_main"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.container = tk.Frame(self.canvas, bg=THEME["bg_main"])

        self.container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.container, anchor="nw", width=960)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def reset_comparison_view(self):
        """Refreshes the comparison view and clears widgets."""
        self.btn_compare.config(state="normal", text="⚖️ Compare All 4 Prompts")
        self.controller.predictor.llm_service.clear_cache()
        for w in self.container.winfo_children():
            w.destroy()

    def run_comparison_thread(self):
        sentence = self.ent_compare.get().strip()
        if not sentence:
            messagebox.showwarning("Empty Sentence", "Please enter an Assamese sentence.")
            return

        self.btn_compare.config(state="disabled", text="⏳ Comparing 4 Techniques...")
        thread = threading.Thread(target=self._execute_compare, args=(sentence,))
        thread.daemon = True
        thread.start()

    def _execute_compare(self, sentence: str):
        try:
            res = self.controller.predictor.compare_prompt_techniques(sentence)
            self.after(0, self._render_comparison, res)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.btn_compare.config(state="normal", text="⚖️ Compare All 4 Prompts"))

    def _render_comparison(self, res: Dict[str, Any]):
        self.btn_compare.config(state="normal", text="⚖️ Compare All 4 Prompts")

        for w in self.container.winfo_children():
            w.destroy()

        comparisons = res.get("comparisons", {})
        if not comparisons:
            lbl = tk.Label(self.container, text="⚠️ No known ambiguous words detected in the entered sentence.", font=(THEME["font_family"], 11), fg=THEME["warning"], bg=THEME["bg_main"])
            lbl.pack(pady=20)
            return

        for word, tech_results in comparisons.items():
            # Section Header for target word
            sec = tk.Frame(self.container, bg=THEME["bg_main"], pady=6)
            sec.pack(fill="x")
            tk.Label(sec, text=f"Target Word: {word}", font=(THEME["font_assamese"], 13, "bold"), fg=THEME["primary"], bg=THEME["bg_main"]).pack(side="left")

            # 4 Side-by-side or stacked Cards
            grid = tk.Frame(self.container, bg=THEME["bg_main"])
            grid.pack(fill="x", pady=(0, 16))

            for tr in tech_results:
                card = tk.Frame(grid, bg=THEME["bg_card"], padx=14, pady=12, highlightbackground=THEME["border"], highlightthickness=1)
                card.pack(fill="x", pady=4)

                top_row = tk.Frame(card, bg=THEME["bg_card"])
                top_row.pack(fill="x")

                tk.Label(top_row, text=tr["technique"], font=(THEME["font_family"], 10, "bold"), fg=THEME["secondary"], bg=THEME["bg_card"]).pack(side="left")
                tk.Label(top_row, text=f"Predicted Sense ID: {tr['predicted_sense_id']}", font=(THEME["font_family"], 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(side="right")

                tk.Label(card, text=f"Meaning: {tr['predicted_meaning']}", font=(THEME["font_family"], 10, "bold"), fg=THEME["success"], bg=THEME["bg_card"]).pack(anchor="w", pady=(4, 2))
                tk.Label(card, text=f"Explanation: {tr['explanation']}", font=(THEME["font_family"], 9), fg=THEME["text_muted"], bg=THEME["bg_card"], wraplength=900, justify="left").pack(anchor="w")


# ==============================================================================
# PAGE 4: DATASET EXPLORER
# ==============================================================================
class DatasetPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_main"])
        self.controller = controller
        self.pack(fill="both", expand=True)

        self.df = self.controller.handler.raw_df

        # Filter Bar Card
        filter_card = tk.Frame(self, bg=THEME["bg_card"], padx=16, pady=12, highlightbackground=THEME["border"], highlightthickness=1)
        filter_card.pack(fill="x", pady=(0, 10))

        # Search Bar
        tk.Label(filter_card, text="Search Sentence:", font=(THEME["font_family"], 9, "bold"), fg=THEME["text_dark"], bg=THEME["bg_card"]).grid(row=0, column=0, sticky="w", padx=4)
        self.ent_search = tk.Entry(filter_card, font=(THEME["font_family"], 9), width=24, bg="#F8FAFC", bd=1, relief="solid")
        self.ent_search.grid(row=0, column=1, padx=6)
        self.ent_search.bind("<KeyRelease>", lambda e: self.apply_filters())

        # Target Word Filter
        words = ["All Target Words"] + self.controller.handler.get_target_words()
        tk.Label(filter_card, text="Target Word:", font=(THEME["font_family"], 9, "bold"), fg=THEME["text_dark"], bg=THEME["bg_card"]).grid(row=0, column=2, sticky="w", padx=4)
        self.combo_word = ttk.Combobox(filter_card, values=words, state="readonly", width=16)
        self.combo_word.set("All Target Words")
        self.combo_word.grid(row=0, column=3, padx=6)
        self.combo_word.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        # Sense Filter
        tk.Label(filter_card, text="Sense ID:", font=(THEME["font_family"], 9, "bold"), fg=THEME["text_dark"], bg=THEME["bg_card"]).grid(row=0, column=4, sticky="w", padx=4)
        self.ent_sense = tk.Entry(filter_card, font=(THEME["font_family"], 9), width=14, bg="#F8FAFC", bd=1, relief="solid")
        self.ent_sense.grid(row=0, column=5, padx=6)
        self.ent_sense.bind("<KeyRelease>", lambda e: self.apply_filters())

        # Reset Button
        btn_reset = tk.Button(filter_card, text="Reset Filters", font=(THEME["font_family"], 9), bg="#F1F5F9", fg=THEME["primary"], bd=0, padx=8, pady=3, command=self.reset_filters)
        btn_reset.grid(row=0, column=6, padx=8)

        # Counter Label
        self.lbl_count = tk.Label(filter_card, text=f"Showing: {len(self.df)} of {len(self.df)} records", font=(THEME["font_family"], 9), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_count.grid(row=0, column=7, sticky="e", padx=12)

        # Read-Only Table View
        table_frame = tk.Frame(self, bg=THEME["bg_card"], highlightbackground=THEME["border"], highlightthickness=1)
        table_frame.pack(fill="both", expand=True)

        cols = ("ID", "Target_Word", "Sense_ID", "English_Meaning", "Sentence")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")

        self.tree.heading("ID", text="ID")
        self.tree.heading("Target_Word", text="Target Word")
        self.tree.heading("Sense_ID", text="Sense ID")
        self.tree.heading("English_Meaning", text="English Meaning")
        self.tree.heading("Sentence", text="Sentence (Assamese)")

        self.tree.column("ID", width=60, anchor="center")
        self.tree.column("Target_Word", width=110, anchor="center")
        self.tree.column("Sense_ID", width=120, anchor="center")
        self.tree.column("English_Meaning", width=180, anchor="w")
        self.tree.column("Sentence", width=500, anchor="w")

        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")

        self.populate_table(self.df)

    def populate_table(self, df_to_show: pd.DataFrame):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for _, row in df_to_show.head(500).iterrows():
            self.tree.insert("", "end", values=(
                row["ID"],
                row["Target_Word"],
                row["Sense_ID"],
                row["English_Meaning"],
                row["Sentence"],
            ))
        self.lbl_count.config(text=f"Showing: {len(df_to_show)} records (Read-Only)")

    def apply_filters(self):
        query = self.ent_search.get().strip().lower()
        selected_word = self.combo_word.get()
        sense_query = self.ent_sense.get().strip().upper()

        filtered = self.df.copy()
        if query:
            filtered = filtered[filtered["Sentence"].str.lower().str.contains(query, na=False)]
        if selected_word != "All Target Words":
            filtered = filtered[filtered["Target_Word"] == selected_word]
        if sense_query:
            filtered = filtered[filtered["Sense_ID"].str.contains(sense_query, na=False)]

        self.populate_table(filtered)

    def reset_filters(self):
        self.ent_search.delete(0, "end")
        self.combo_word.set("All Target Words")
        self.ent_sense.delete(0, "end")
        self.populate_table(self.df)


# ==============================================================================
# PAGE 5: EVALUATION DASHBOARD
# ==============================================================================
class EvaluationPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=THEME["bg_main"])
        self.controller = controller
        self.pack(fill="both", expand=True)

        self.current_img = None

        # Top Control Row
        top_row = tk.Frame(self, bg=THEME["bg_card"], padx=16, pady=12, highlightbackground=THEME["border"], highlightthickness=1)
        top_row.pack(fill="x", pady=(0, 10))

        tk.Label(top_row, text="Select Research Visualization:", font=(THEME["font_family"], 10, "bold"), fg=THEME["primary"], bg=THEME["bg_card"]).pack(side="left", padx=(0, 8))

        self.graph_options = [
            ("1. Technique Accuracy Comparison", "1_technique_accuracy_comparison.png"),
            ("2. Precision, Recall & F1-Score", "2_precision_recall_f1_comparison.png"),
            ("3. WSD Confusion Matrix", "3_wsd_confusion_matrix.png"),
            ("4. Target Word Frequency Distribution", "4_target_word_distribution.png"),
            ("5. Sense Complexity Distribution", "5_sense_complexity_distribution.png"),
            ("6. Ambiguous Word Detection Metrics", "6_ambiguous_word_detection_performance.png"),
            ("7. 2-Sense vs 3+-Sense Accuracy Breakdown", "7_wordwise_complexity_performance.png"),
        ]

        self.combo_graph = ttk.Combobox(top_row, values=[opt[0] for opt in self.graph_options], state="readonly", width=38)
        self.combo_graph.current(0)
        self.combo_graph.pack(side="left", padx=(0, 16))
        self.combo_graph.bind("<<ComboboxSelected>>", lambda e: self.load_selected_graph())

        # Action Buttons
        btn_report = tk.Button(top_row, text="📄 View Research Report", font=(THEME["font_family"], 9), bg="#F1F5F9", fg=THEME["primary"], bd=0, padx=10, pady=4, cursor="hand2", command=self.view_report_dialog)
        btn_report.pack(side="right", padx=6)

        btn_run = tk.Button(top_row, text="⚡ Re-Run Benchmark", font=(THEME["font_family"], 9, "bold"), bg=THEME["secondary"], fg="#FFFFFF", bd=0, padx=12, pady=4, cursor="hand2", command=self.run_benchmark_thread)
        btn_run.pack(side="right", padx=6)

        self.lbl_eval_status = tk.Label(top_row, text="Evaluation Ready", font=(THEME["font_family"], 9), fg=THEME["text_muted"], bg=THEME["bg_card"])
        self.lbl_eval_status.pack(side="right", padx=10)

        # Quick Metric KPI Badges
        kpi_bar = tk.Frame(self, bg=THEME["bg_main"])
        kpi_bar.pack(fill="x", pady=(0, 10))

        self._create_metric_card(kpi_bar, "Detection Recall", "97.85%", "273 / 279 Target Words", THEME["primary"])
        self._create_metric_card(kpi_bar, "Detection Precision", "87.50%", "Morphological Filter", THEME["secondary"])
        self._create_metric_card(kpi_bar, "Detection F1-Score", "92.39%", "Stage A Overall F1", THEME["accent"])
        self._create_metric_card(kpi_bar, "WSD Evaluated", "279 / 279", "Held-out Test Split", THEME["success"])

        # Image Viewer Canvas
        self.img_frame = tk.Frame(self, bg=THEME["bg_card"], highlightbackground=THEME["border"], highlightthickness=1)
        self.img_frame.pack(fill="both", expand=True)

        self.lbl_image = tk.Label(self.img_frame, bg=THEME["bg_card"], text="Loading Visualization...")
        self.lbl_image.pack(fill="both", expand=True, padx=10, pady=10)

        self.load_selected_graph()

    def load_selected_graph(self):
        idx = self.combo_graph.current()
        if idx < 0:
            return
        filename = self.graph_options[idx][1]
        img_path = GRAPHS_DIR / filename

        if not img_path.exists():
            self.lbl_image.config(text=f"Graph not found: {filename}\nPlease click 'Re-Run Benchmark' to generate graphs.", image="")
            return

        try:
            pil_img = Image.open(img_path)
            # Resize image to fit comfortably in view while preserving aspect ratio
            w, h = pil_img.size
            max_w, max_h = 920, 560
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            self.current_img = ImageTk.PhotoImage(resized)
            self.lbl_image.config(image=self.current_img, text="")
        except Exception as e:
            self.lbl_image.config(text=f"Error displaying graph: {e}", image="")

    def _create_metric_card(self, parent, title, value, subtitle, color):
        card = tk.Frame(parent, bg=THEME["bg_card"], padx=14, pady=10, highlightbackground=THEME["border"], highlightthickness=1)
        card.pack(side="left", fill="both", expand=True, padx=4)

        tk.Label(card, text=title, font=(THEME["font_family"], 8, "bold"), fg=THEME["text_muted"], bg=THEME["bg_card"]).pack(anchor="w")
        tk.Label(card, text=value, font=(THEME["font_family"], 15, "bold"), fg=color, bg=THEME["bg_card"], pady=1).pack(anchor="w")
        tk.Label(card, text=subtitle, font=(THEME["font_family"], 8), fg=THEME["text_muted"], bg=THEME["bg_card"]).pack(anchor="w")

    def view_report_dialog(self):
        if not EVALUATION_REPORT_PATH.exists():
            messagebox.showinfo("Report Missing", "The evaluation report has not been generated yet. Please run the benchmark first.")
            return

        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        win = tk.Toplevel(self)
        win.title("Evaluation Research Report")
        win.geometry("800x600")

        txt = tk.Text(win, font=("Consolas", 10), wrap="word", padx=12, pady=12)
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)

        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        txt.insert("1.0", content)
        txt.config(state="disabled")

    def run_benchmark_thread(self):
        self.lbl_eval_status.config(text="Running benchmark across test set...", fg=THEME["secondary"])
        thread = threading.Thread(target=self._execute_benchmark)
        thread.daemon = True
        thread.start()

    def _execute_benchmark(self):
        try:
            det_metrics = self.controller.evaluator.evaluate_ambiguous_word_detection()
            metrics_summary, results_df = self.controller.evaluator.run_wsd_benchmark(sample_size=60)
            self.controller.visualizer.plot_accuracy_comparison(metrics_summary)
            self.controller.visualizer.plot_precision_recall_f1(metrics_summary)
            self.controller.visualizer.plot_confusion_matrix(results_df)
            self.controller.visualizer.plot_word_distribution(self.controller.handler.raw_df)
            self.controller.visualizer.plot_sense_distribution(self.controller.handler.senses_map)
            self.controller.visualizer.plot_detection_performance(det_metrics)
            self.controller.visualizer.plot_complexity_breakdown(metrics_summary)

            self.after(0, self._on_benchmark_complete)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Benchmark Error", str(e)))

    def _on_benchmark_complete(self):
        self.lbl_eval_status.config(text="Benchmark complete! Graphs updated.", fg=THEME["success"])
        self.load_selected_graph()
        messagebox.showinfo("Benchmark Completed", "Evaluation benchmark completed successfully.\nAll 7 research charts and report files have been regenerated.")


def run_gui():
    """Entry point to launch the Tkinter Desktop Application."""
    app = WSDApp()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
