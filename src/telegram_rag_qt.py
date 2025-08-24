#!/usr/bin/env python3
"""
Telegram RAG System - Qt for Python Interface
=============================================

A modern desktop application built with Qt for Python (PySide6) that provides
a comprehensive interface for the Telegram RAG system with OpenVINO integration.

This application preserves all functionality from the Gradio interface while
providing enhanced desktop features and better user experience.

Features:
- Model Management with real-time status
- Telegram Message Download and Processing
- Vector Store Operations
- Advanced Question Answering with Streaming
- GPU Diagnostics and Optimization
- Dark/Light Theme Support
- Dockable Panels and Advanced Layout

Based on: https://doc.qt.io/qtforpython-6/
"""

import sys
import os
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading
import time

# Qt Imports
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QTabWidget, QTextEdit, QLineEdit, QPushButton, QLabel, QSlider,
        QComboBox, QCheckBox, QProgressBar, QSplitter, QGroupBox,
        QScrollArea, QFrame, QSpacerItem, QSizePolicy, QMenuBar, QMenu,
        QStatusBar, QToolBar, QDockWidget, QTreeWidget, QTreeWidgetItem,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
        QFileDialog, QDialog, QDialogButtonBox, QFormLayout, QSpinBox,
        QTextBrowser, QPlainTextEdit, QListWidget, QListWidgetItem
    )
    from PySide6.QtCore import (
        Qt, QThread, QObject, Signal, QTimer, QSettings, QSize,
        QPropertyAnimation, QEasingCurve, QRect, QThreadPool, QRunnable
    )
    from PySide6.QtGui import (
        QFont, QIcon, QPixmap, QPalette, QColor, QAction, QTextCursor,
        QSyntaxHighlighter, QTextCharFormat, QTextDocument
    )
except ImportError:
    print("PySide6 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"])
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *

# Import our existing backend modules
import sys
from pathlib import Path

# Add examples directory to path for gradio module
examples_path = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from telegram_rag_gradio import (
    get_available_openvino_llm_models, get_model_display_name,
    get_available_devices, initialize_models, initialize_rag,
    download_ov_model_if_needed, get_ov_model_path,
    download_messages, process_messages, query_messages,
    answer_question, answer_question_stream, check_gpu_info,
    get_user_channels,  # New function for channel discovery
    embedding, reranker, llm, retriever, rag_chain,
    models_dir, data_dir, telegram_data_dir, vector_store_path,
    DEFAULT_LANGUAGE, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANK_MODEL, DEFAULT_DEVICE, DEFAULT_EMBEDDING_TYPE
)
import telegram_rag_gradio as gr_backend
from npu_models import (
    get_npu_models, is_npu_device, get_npu_model_path, is_npu_compatible_model,
    add_npu_models_to_config
)
from llm_config import SUPPORTED_LLM_MODELS, SUPPORTED_EMBEDDING_MODELS, SUPPORTED_RERANK_MODELS


class StreamingWorker(QRunnable):
    """Worker for handling streaming responses in a separate thread"""
    
    def __init__(self, callback, question, channel, temperature, num_context, show_retrieved, repetition_penalty):
        super().__init__()
        self.callback = callback
        self.question = question
        self.channel = channel
        self.temperature = temperature
        self.num_context = num_context
        self.show_retrieved = show_retrieved
        self.repetition_penalty = repetition_penalty
        
    def run(self):
        """Execute the streaming response generation"""
        try:
            # Use the existing streaming function
            for partial_response in answer_question_stream(
                self.question, self.channel, self.temperature, 
                self.num_context, self.show_retrieved, self.repetition_penalty
            ):
                self.callback.emit(partial_response)
            
            # Ensure completion is signaled even if no explicit completion marker
            self.callback.force_finish()
            
        except Exception as e:
            self.callback.emit(f"Error: {str(e)}")
            self.callback.force_finish()


class ModelWorker(QObject):
    """Worker for model operations that need to run in background"""
    
    # Signals for communication with main thread
    progress_updated = Signal(int, str)  # progress percentage, status message
    model_loaded = Signal(str, bool)     # model_type, success
    operation_completed = Signal(str)    # result message
    error_occurred = Signal(str)         # error message
    channels_fetched = Signal(list)      # list of channel data
    
    def __init__(self):
        super().__init__()
        self.should_stop = False
        
    def load_models(self, device, embedding_type):
        """Load models in background thread"""
        try:
            self.progress_updated.emit(10, "Initializing model loading...")
            
            # Initialize models
            initialize_models(device, embedding_type)
            self.progress_updated.emit(60, "Models initialized...")
            
            # Initialize RAG system
            rag_ready = initialize_rag()
            self.progress_updated.emit(90, "RAG system ready..." if rag_ready else "RAG system not ready...")
            
            self.progress_updated.emit(100, "✅ Model loading complete!")
            self.operation_completed.emit("Models loaded successfully!")
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading models: {str(e)}")
    
    def download_messages_async(self, channels, limit, hours):
        """Download messages asynchronously"""
        try:
            # Convert list back to comma-separated string for the download function
            if isinstance(channels, list):
                channels_str = ",".join(channels)
                num_channels = len(channels)
            else:
                channels_str = channels
                num_channels = len([c.strip() for c in channels.split(",") if c.strip()])
            
            self.progress_updated.emit(10, f"Starting download from {num_channels} channels...")
            
            # Run the async download function with string parameter
            result = download_messages(channels_str, limit, hours)
            
            self.progress_updated.emit(100, "Download complete!")
            self.operation_completed.emit(str(result[0]))  # result[0] is the status message
            
        except Exception as e:
            self.error_occurred.emit(f"Error downloading messages: {str(e)}")
    
    def process_messages_async(self):
        """Process messages asynchronously"""
        try:
            self.progress_updated.emit(10, "Starting message processing...")
            
            result = process_messages()
            
            self.progress_updated.emit(100, "Processing complete!")
            self.operation_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing messages: {str(e)}")
    
    def fetch_channels_async(self):
        """Fetch user's channels asynchronously"""
        try:
            self.progress_updated.emit(10, "Fetching your channels...")
            
            status, channels = get_user_channels()
            
            if "Error" in status:
                self.error_occurred.emit(status)
            else:
                self.progress_updated.emit(100, "Channels fetched successfully!")
                self.channels_fetched.emit(channels)
                self.operation_completed.emit(status)
            
        except Exception as e:
            self.error_occurred.emit(f"Error fetching channels: {str(e)}")


class StatusPanel(QWidget):
    """Panel for displaying system status and logs"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Status section
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        
        self.model_status_label = QLabel("Models: Not Loaded")
        self.rag_status_label = QLabel("RAG System: Not Ready") 
        self.device_status_label = QLabel("Device: AUTO")
        
        status_layout.addWidget(self.model_status_label)
        status_layout.addWidget(self.rag_status_label)
        status_layout.addWidget(self.device_status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        
        # Logs section
        logs_group = QGroupBox("Activity Logs")
        logs_layout = QVBoxLayout(logs_group)
        
        self.logs_text = QTextEdit()
        self.logs_text.setMaximumHeight(200)
        self.logs_text.setReadOnly(True)
        
        # Clear logs button
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.clear_logs)
        
        logs_layout.addWidget(self.logs_text)
        logs_layout.addWidget(clear_logs_btn)
        
        # Add to main layout
        layout.addWidget(status_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(logs_group)
        layout.addStretch()
    
    def update_model_status(self, status):
        self.model_status_label.setText(f"Models: {status}")
        self.log_message(f"Model Status: {status}")
    
    def update_rag_status(self, status):
        self.rag_status_label.setText(f"RAG System: {status}")
        self.log_message(f"RAG Status: {status}")
    
    def update_device_status(self, device):
        self.device_status_label.setText(f"Device: {device}")
        self.log_message(f"Device Changed: {device}")
    
    def show_progress(self, percentage, message=""):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(message)
        if percentage >= 100:
            QTimer.singleShot(2000, self.hide_progress)  # Hide after 2 seconds
    
    def hide_progress(self):
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.append(f"[{timestamp}] {message}")
    
    def clear_logs(self):
        self.logs_text.clear()


class ModelConfigPanel(QWidget):
    """Panel for model configuration and management"""
    
    # Signals
    model_reload_requested = Signal(str, str, str, str, str, str, str)  # device, embedding_type, etc.
    device_changed = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.available_models = {}
        self.setup_ui()
        self.load_available_models()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # OpenVINO Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("docs/images/OpenVINO_logo.png")
        if not logo_pixmap.isNull():
            # Scale the logo to a reasonable size while maintaining aspect ratio
            scaled_pixmap = logo_pixmap.scaled(200, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet("margin: 10px 0px;")
        else:
            # Fallback if image not found
            logo_label.setText("OpenVINO")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0068b5; text-align: center; margin: 10px 0px;")
            logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)
        
        # Header
        header_label = QLabel("Model Configuration")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Model selection section
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)
        
        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(SUPPORTED_LLM_MODELS.keys()))
        self.language_combo.setCurrentText(DEFAULT_LANGUAGE)
        model_layout.addRow("Language:", self.language_combo)
        
        # LLM Model selection
        self.llm_model_combo = QComboBox()
        model_layout.addRow("LLM Model:", self.llm_model_combo)
        
        # LLM Precision
        self.llm_precision_combo = QComboBox()
        self.llm_precision_combo.addItems(["int4", "int8", "fp16"])
        self.llm_precision_combo.setCurrentText("int4")
        model_layout.addRow("LLM Precision:", self.llm_precision_combo)
        
        # Embedding Model
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.addItems(list(SUPPORTED_EMBEDDING_MODELS[DEFAULT_LANGUAGE].keys()))
        self.embedding_model_combo.setCurrentText(DEFAULT_EMBEDDING_MODEL)
        model_layout.addRow("Embedding Model:", self.embedding_model_combo)
        
        # Reranker Model
        self.reranker_model_combo = QComboBox()
        self.reranker_model_combo.addItems(list(SUPPORTED_RERANK_MODELS.keys()))
        self.reranker_model_combo.setCurrentText(DEFAULT_RERANK_MODEL)
        model_layout.addRow("Reranker Model:", self.reranker_model_combo)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        self.device_combo.setCurrentText(DEFAULT_DEVICE)
        self.device_combo.currentTextChanged.connect(self.device_changed.emit)
        # Also refresh models when device changes (handles NPU-specific lists)
        self.device_changed.connect(self.on_device_changed)
        model_layout.addRow("Device:", self.device_combo)
        
        # Embedding implementation
        self.embedding_type_combo = QComboBox()
        self.embedding_type_combo.addItems([
            "TextEmbeddingPipeline (Latest)",
            "OpenVINO GenAI", 
            "Legacy OpenVINO"
        ])
        self.embedding_type_combo.setCurrentText("TextEmbeddingPipeline (Latest)")
        model_layout.addRow("Embedding Implementation:", self.embedding_type_combo)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.refresh_models_btn = QPushButton("🔄 Refresh Models")
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        
        self.reload_models_btn = QPushButton("Reload Models")
        self.reload_models_btn.clicked.connect(self.reload_models)
        self.reload_models_btn.setStyleSheet("QPushButton { background-color: #007acc; color: white; font-weight: bold; }")
        
        self.gpu_diagnostics_btn = QPushButton("🔍 GPU Diagnostics")
        self.gpu_diagnostics_btn.clicked.connect(self.show_gpu_diagnostics)
        
        buttons_layout.addWidget(self.refresh_models_btn)
        buttons_layout.addWidget(self.reload_models_btn)
        buttons_layout.addWidget(self.gpu_diagnostics_btn)
        
        # Model status display
        self.model_status_text = QTextEdit()
        self.model_status_text.setMaximumHeight(150)
        self.model_status_text.setReadOnly(True)
        self.update_model_status_display()
        
        # Add to layout
        layout.addWidget(model_group)
        layout.addLayout(buttons_layout)
        layout.addWidget(QLabel("Current Model Status:"))
        layout.addWidget(self.model_status_text)
        layout.addStretch()
        
        # Connect signals
        self.language_combo.currentTextChanged.connect(self.update_model_choices)
        self.llm_model_combo.currentTextChanged.connect(self.update_precision_choices)
    
    def load_available_models(self):
        """Load available models from OpenVINO collection"""
        try:
            # Use current device selection to fetch appropriate models (handles NPU)
            current_device = None
            try:
                current_device = self.device_combo.currentText()
            except Exception:
                current_device = None
            if current_device:
                self.available_models = get_available_openvino_llm_models(device=current_device)
            else:
                self.available_models = get_available_openvino_llm_models()
            self.update_llm_model_choices()
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def update_llm_model_choices(self):
        """Update LLM model choices"""
        self.llm_model_combo.clear()
        for model_id in self.available_models.keys():
            # Works for both regular and NPU models (repo_id keys)
            display_name = get_model_display_name(model_id)
            self.llm_model_combo.addItem(display_name, model_id)
        
        # Set default
        default_index = self.llm_model_combo.findData(DEFAULT_LLM_MODEL)
        if default_index >= 0:
            self.llm_model_combo.setCurrentIndex(default_index)
        elif self.llm_model_combo.count() > 0:
            # Fallback to first item if default not present (e.g., NPU list)
            self.llm_model_combo.setCurrentIndex(0)
    
    def update_model_choices(self, language):
        """Update model choices based on language"""
        # Update embedding models
        self.embedding_model_combo.clear()
        self.embedding_model_combo.addItems(list(SUPPORTED_EMBEDDING_MODELS[language].keys()))
    
    def update_precision_choices(self):
        """Update precision choices based on selected LLM"""
        current_model = self.llm_model_combo.currentData()
        self.llm_precision_combo.clear()
        if current_model and current_model in self.available_models:
            precisions = self.available_models[current_model]
            # If precisions is a list (regular models), use it; otherwise default to INT4 (NPU)
            if isinstance(precisions, list):
                self.llm_precision_combo.addItems(precisions)
            else:
                self.llm_precision_combo.addItems(["int4"])  # NPU models are INT4
        else:
            self.llm_precision_combo.addItems(["int4"])  # Safe default
    
    def refresh_models(self):
        """Refresh available models"""
        self.load_available_models()
        self.update_precision_choices()
    
    def reload_models(self):
        """Emit signal to reload models"""
        # Get embedding type mapping
        embedding_type_mapping = {
            "TextEmbeddingPipeline (Latest)": "text_embedding_pipeline",
            "OpenVINO GenAI": "openvino_genai",
            "Legacy OpenVINO": "legacy"
        }
        
        embedding_type = embedding_type_mapping.get(
            self.embedding_type_combo.currentText(), 
            "text_embedding_pipeline"
        )
        
        self.model_reload_requested.emit(
            self.language_combo.currentText(),
            self.llm_model_combo.currentData() or self.llm_model_combo.currentText(),
            self.embedding_model_combo.currentText(),
            self.reranker_model_combo.currentText(),
            self.llm_precision_combo.currentText(),
            self.device_combo.currentText(),
            embedding_type
        )

    def on_device_changed(self, device):
        """Handle device change to refresh model list (including NPU handling)"""
        try:
            self.load_available_models()
            self.update_precision_choices()
        except Exception as e:
            print(f"Error updating models for device '{device}': {e}")
    
    def show_gpu_diagnostics(self):
        """Show GPU diagnostics dialog"""
        gpu_info = check_gpu_info()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("GPU Diagnostics")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        gpu_text = QTextBrowser()
        gpu_text.setMarkdown(gpu_info)
        layout.addWidget(gpu_text)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        dialog.exec()
    
    def update_model_status_display(self):
        """Update the model status display"""
        # This will be called from main window when models are loaded
        pass


class TelegramPanel(QWidget):
    """Panel for Telegram operations"""
    
    # Signals
    download_requested = Signal(list, int, int)  # channels, limit, hours
    process_requested = Signal()
    channels_fetch_requested = Signal()  # New signal for fetching channels
    
    def __init__(self):
        super().__init__()
        self.user_channels = []  # Store fetched channels
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Channel Discovery Section
        discovery_group = QGroupBox("📡 Discover Your Telegram Channels")
        discovery_layout = QVBoxLayout(discovery_group)
        
        discovery_info = QLabel("First, fetch your subscribed channels and select which ones to download messages from.")
        discovery_info.setWordWrap(True)
        discovery_info.setStyleSheet("color: #666666; font-size: 12px; margin-bottom: 8px;")
        discovery_layout.addWidget(discovery_info)
        
        # Fetch channels button
        fetch_buttons_layout = QHBoxLayout()
        self.fetch_channels_btn = QPushButton("🔍 Fetch My Channels")
        self.fetch_channels_btn.clicked.connect(self.fetch_channels)
        self.fetch_channels_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.refresh_channels_btn = QPushButton("🔄 Refresh")
        self.refresh_channels_btn.clicked.connect(self.fetch_channels)
        self.refresh_channels_btn.setEnabled(False)
        
        fetch_buttons_layout.addWidget(self.fetch_channels_btn)
        fetch_buttons_layout.addWidget(self.refresh_channels_btn)
        fetch_buttons_layout.addStretch()
        
        discovery_layout.addLayout(fetch_buttons_layout)
        
        # Channel selection section
        self.channels_group = QGroupBox("Select Channels to Download")
        channels_layout = QVBoxLayout(self.channels_group)
        
        # Selection controls
        selection_controls = QHBoxLayout()
        self.select_all_btn = QPushButton("✅ Select All")
        self.select_all_btn.clicked.connect(self.select_all_channels)
        self.select_all_btn.setEnabled(False)
        
        self.select_none_btn = QPushButton("❌ Select None")
        self.select_none_btn.clicked.connect(self.select_no_channels)
        self.select_none_btn.setEnabled(False)
        
        self.channel_count_label = QLabel("No channels loaded")
        self.channel_count_label.setStyleSheet("color: #666666; font-size: 12px;")
        
        selection_controls.addWidget(self.select_all_btn)
        selection_controls.addWidget(self.select_none_btn)
        selection_controls.addStretch()
        selection_controls.addWidget(self.channel_count_label)
        
        channels_layout.addLayout(selection_controls)
        
        # Channel list
        self.channels_list = QListWidget()
        self.channels_list.setMaximumHeight(200)
        self.channels_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
                background-color: #ffffff;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        channels_layout.addWidget(self.channels_list)
        
        self.channels_group.setEnabled(False)  # Disabled until channels are fetched
        discovery_layout.addWidget(self.channels_group)
        
        # Download Messages Section
        download_group = QGroupBox("📥 Download Messages from Selected Channels")
        download_layout = QFormLayout(download_group)
        
        self.limit_slider = QSlider(Qt.Horizontal)
        self.limit_slider.setRange(1, 1000)
        self.limit_slider.setValue(100)
        self.limit_label = QLabel("100")
        self.limit_slider.valueChanged.connect(lambda v: self.limit_label.setText(str(v)))
        
        limit_layout = QHBoxLayout()
        limit_layout.addWidget(self.limit_slider)
        limit_layout.addWidget(self.limit_label)
        download_layout.addRow("Messages per Channel:", limit_layout)
        
        self.hours_slider = QSlider(Qt.Horizontal)
        self.hours_slider.setRange(1, 168)
        self.hours_slider.setValue(24)
        self.hours_label = QLabel("24")
        self.hours_slider.valueChanged.connect(lambda v: self.hours_label.setText(str(v)))
        
        hours_layout = QHBoxLayout()
        hours_layout.addWidget(self.hours_slider)
        hours_layout.addWidget(self.hours_label)
        download_layout.addRow("Hours to Look Back:", hours_layout)
        
        self.download_btn = QPushButton("📥 Download Messages")
        self.download_btn.clicked.connect(self.download_messages)
        self.download_btn.setEnabled(False)  # Disabled until channels are selected
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        download_layout.addRow(self.download_btn)
        
        self.download_status = QTextEdit()
        self.download_status.setMaximumHeight(100)
        self.download_status.setReadOnly(True)
        download_layout.addRow("Download Status:", self.download_status)
        
        # Process Messages Section
        process_group = QGroupBox("🔄 Process Downloaded Messages")
        process_layout = QVBoxLayout(process_group)
        
        process_info = QLabel("Convert downloaded messages into a searchable vector store for AI-powered querying.")
        process_info.setWordWrap(True)
        process_info.setStyleSheet("color: #666666; margin-bottom: 8px;")
        process_layout.addWidget(process_info)
        
        self.process_btn = QPushButton("🔄 Process Messages")
        self.process_btn.clicked.connect(self.process_messages)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        process_layout.addWidget(self.process_btn)
        
        self.process_status = QTextEdit()
        self.process_status.setMaximumHeight(100)
        self.process_status.setReadOnly(True)
        process_layout.addWidget(self.process_status)
        
        # Add to main layout
        layout.addWidget(discovery_group)
        layout.addWidget(download_group)
        layout.addWidget(process_group)
        layout.addStretch()
    
    def fetch_channels(self):
        """Request to fetch user's channels"""
        self.channels_fetch_requested.emit()
        self.fetch_channels_btn.setEnabled(False)
        self.fetch_channels_btn.setText("Fetching...")
    
    def on_channels_fetched(self, channels):
        """Handle fetched channels data"""
        self.user_channels = channels
        self.populate_channels_list()
        
        # Enable UI elements
        self.fetch_channels_btn.setEnabled(True)
        self.fetch_channels_btn.setText("🔍 Fetch My Channels")
        self.refresh_channels_btn.setEnabled(True)
        self.channels_group.setEnabled(True)
        
        # Update status
        self.channel_count_label.setText(f"{len(channels)} channels found")
    
    def populate_channels_list(self):
        """Populate the channels list with checkboxes"""
        self.channels_list.clear()
        
        for channel in self.user_channels:
            # Create list item
            item = QListWidgetItem()
            
            # Create checkbox widget
            checkbox = QCheckBox()
            
            # Format channel display name
            display_name = channel["name"]
            if channel.get("username"):
                display_name += f" (@{channel['username']})"
            
            # Add channel type and member count info
            info_parts = []
            if channel["type"] == "channel":
                info_parts.append("📢 Channel")
            else:
                info_parts.append("👥 Group")
            
            if channel.get("members_count", 0) > 0:
                info_parts.append(f"{channel['members_count']:,} members")
            
            if channel.get("is_verified"):
                info_parts.append("✅ Verified")
            
            info_text = " • ".join(info_parts)
            full_text = f"{display_name}\n{info_text}"
            
            checkbox.setText(full_text)
            checkbox.setStyleSheet("""
                QCheckBox {
                    font-size: 13px;
                    color: #333333;
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
            """)
            
            # Store channel data in checkbox
            checkbox.setProperty("channel_data", channel)
            checkbox.stateChanged.connect(self.on_channel_selection_changed)
            
            # Add to list
            self.channels_list.addItem(item)
            self.channels_list.setItemWidget(item, checkbox)
            
            # Set proper item size (60px height should accommodate 2 lines + padding)
            item.setSizeHint(QSize(400, 60))
    
    def select_all_channels(self):
        """Select all channels"""
        for i in range(self.channels_list.count()):
            item = self.channels_list.item(i)
            checkbox = self.channels_list.itemWidget(item)
            if checkbox:
                checkbox.setChecked(True)
    
    def select_no_channels(self):
        """Deselect all channels"""
        for i in range(self.channels_list.count()):
            item = self.channels_list.item(i)
            checkbox = self.channels_list.itemWidget(item)
            if checkbox:
                checkbox.setChecked(False)
    
    def on_channel_selection_changed(self):
        """Handle channel selection changes"""
        selected_count = self.get_selected_channels_count()
        
        # Enable/disable download button based on selection
        self.download_btn.setEnabled(selected_count > 0)
        
        # Enable/disable selection buttons
        total_count = self.channels_list.count()
        self.select_all_btn.setEnabled(total_count > 0 and selected_count < total_count)
        self.select_none_btn.setEnabled(selected_count > 0)
        
        # Update status
        if selected_count > 0:
            self.download_btn.setText(f"📥 Download from {selected_count} Channel{'s' if selected_count != 1 else ''}")
        else:
            self.download_btn.setText("📥 Download Messages")
    
    def get_selected_channels_count(self):
        """Get count of selected channels"""
        count = 0
        for i in range(self.channels_list.count()):
            item = self.channels_list.item(i)
            checkbox = self.channels_list.itemWidget(item)
            if checkbox and checkbox.isChecked():
                count += 1
        return count
    
    def get_selected_channels(self):
        """Get list of selected channel usernames/identifiers"""
        selected = []
        for i in range(self.channels_list.count()):
            item = self.channels_list.item(i)
            checkbox = self.channels_list.itemWidget(item)
            if checkbox and checkbox.isChecked():
                channel_data = checkbox.property("channel_data")
                if channel_data:
                    # Use username if available, otherwise use the channel name
                    identifier = channel_data.get("username") or channel_data.get("name")
                    if identifier:
                        selected.append(identifier)
        return selected
    
    def download_messages(self):
        """Handle download messages request"""
        selected_channels = self.get_selected_channels()
        
        if not selected_channels:
            self.download_status.setText("Please select at least one channel")
            return
        
        limit = self.limit_slider.value()
        hours = self.hours_slider.value()
        
        self.download_requested.emit(selected_channels, limit, hours)
    
    def process_messages(self):
        """Handle process messages request"""
        self.process_requested.emit()
    
    def update_download_status(self, status):
        """Update download status display"""
        self.download_status.setText(status)
    
    def update_process_status(self, status):
        """Update process status display"""
        self.process_status.setText(status)


class QueryPanel(QWidget):
    """Panel for querying messages"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Query input section
        query_group = QGroupBox("Query Processed Messages")
        query_layout = QFormLayout(query_group)
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your search query")
        query_layout.addRow("Search Query:", self.query_input)
        
        self.channel_filter = QLineEdit()
        self.channel_filter.setPlaceholderText("Enter channel name to filter results (optional)")
        query_layout.addRow("Filter by Channel:", self.channel_filter)
        
        self.num_results_slider = QSlider(Qt.Horizontal)
        self.num_results_slider.setRange(1, 20)
        self.num_results_slider.setValue(5)
        self.num_results_label = QLabel("5")
        self.num_results_slider.valueChanged.connect(lambda v: self.num_results_label.setText(str(v)))
        
        results_layout = QHBoxLayout()
        results_layout.addWidget(self.num_results_slider)
        results_layout.addWidget(self.num_results_label)
        query_layout.addRow("Number of Results:", results_layout)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_messages)
        query_layout.addRow(self.search_btn)
        
        # Results display
        self.query_results = QTextEdit()
        self.query_results.setReadOnly(True)
        
        # Add to layout
        layout.addWidget(query_group)
        layout.addWidget(QLabel("Search Results:"))
        layout.addWidget(self.query_results)
    
    def search_messages(self):
        """Perform search query"""
        query = self.query_input.text().strip()
        if not query:
            self.query_results.setText("Please enter a search query")
            return
            
        channel = self.channel_filter.text().strip()
        num_results = self.num_results_slider.value()
        
        try:
            results = query_messages(query, channel, num_results)
            self.query_results.setText(results)
        except Exception as e:
            self.query_results.setText(f"Error performing search: {str(e)}")


class TypingIndicator(QWidget):
    """Animated typing indicator for assistant responses"""
    
    def __init__(self):
        super().__init__()
        self.text = "✨ Generating response..."
        self.setup_ui()
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_dots)
        self.animation_timer.start(500)  # Update every 500ms
        self.dot_count = 0
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        
        # Create typing bubble
        bubble = QFrame()
        bubble.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 12px;
                padding: 12px 16px;
            }
        """)
        bubble_layout = QHBoxLayout(bubble)
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        
        # Typing text
        self.typing_label = QLabel("✨ Generating response")
        self.typing_label.setStyleSheet("""
            color: #6c757d;
            font-size: 14px;
            font-style: italic;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        """)
        
        bubble_layout.addWidget(self.typing_label)
        
        layout.addWidget(bubble)
        layout.addStretch()
        bubble.setMaximumWidth(200)
    
    def animate_dots(self):
        """Animate the typing dots"""
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count
        self.typing_label.setText(f"✨ Generating response{dots}")
    
    def update_text(self, text):
        """Replace typing indicator with actual response"""
        # Stop animation
        self.animation_timer.stop()
        
        # Transform into a regular message
        self.text = text
        
        # Clear current layout
        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create message bubble like ChatMessage
        bubble = QFrame()
        bubble.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 12px;
                padding: 8px;
            }
        """)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(8, 6, 8, 6)
        
        # Sender label
        sender_label = QLabel("Assistant")
        sender_label.setStyleSheet("""
            font-weight: 600; 
            font-size: 11px; 
            color: #666666;
            margin-bottom: 2px;
        """)
        
        # Message text with markdown support
        self.message_text_widget = QTextBrowser()
        self.message_text_widget.setMarkdown(text)
        self.message_text_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.message_text_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.message_text_widget.setReadOnly(True)
        self.message_text_widget.setOpenExternalLinks(True)  # Enable clickable links
        
        # Style and size the text widget for better markdown rendering
        self.message_text_widget.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                color: #2c2c2c;
                font-size: 14px;
                line-height: 1.5;
                padding: 4px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                color: #1a1a1a;
                margin-top: 8px;
                margin-bottom: 4px;
            }
            QTextBrowser strong {
                font-weight: 600;
                color: #1a1a1a;
            }
            QTextBrowser em {
                font-style: italic;
                color: #4a4a4a;
            }
            QTextBrowser ul, QTextBrowser ol {
                margin-left: 16px;
                margin-top: 4px;
                margin-bottom: 4px;
            }
            QTextBrowser li {
                margin-bottom: 2px;
            }
            QTextBrowser code {
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }
            QTextBrowser blockquote {
                border-left: 3px solid #e0e0e0;
                padding-left: 12px;
                margin-left: 8px;
                font-style: italic;
                color: #666666;
            }
        """)
        
        self.adjust_message_height()
        
        # Timestamp
        timestamp_label = QLabel(datetime.now().strftime("%H:%M"))
        timestamp_label.setStyleSheet("""
            color: #999999;
            font-size: 10px;
            font-weight: 400;
            margin-top: 2px;
        """)
        
        # Add content to bubble
        bubble_layout.addWidget(sender_label)
        bubble_layout.addWidget(self.message_text_widget)
        bubble_layout.addWidget(timestamp_label)
        
        # Add to layout
        layout.addWidget(bubble)
        layout.addStretch()
        bubble.setMaximumWidth(500)
        bubble.setMinimumWidth(100)
    
    def adjust_message_height(self):
        """Dynamically adjust message height based on content"""
        if hasattr(self, 'message_text_widget'):
            document = self.message_text_widget.document()
            document.setTextWidth(500)  # Match bubble width
            height = document.size().height()
            
            # Set constraints
            min_height = 30
            max_height = 400
            
            final_height = max(min_height, min(height + 20, max_height))
            self.message_text_widget.setFixedHeight(int(final_height))


class ChatMessage(QWidget):
    """Custom widget for displaying a chat message bubble"""
    
    def __init__(self, text, is_user=True, timestamp=None):
        super().__init__()
        self.text = text
        self.is_user = is_user
        self.timestamp = timestamp or datetime.now()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)  # Smaller outer margins
        
        # Create message bubble
        bubble = QFrame()
        bubble.setFrameStyle(QFrame.Box)
        bubble.setLineWidth(1)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(8, 6, 8, 6)  # Smaller margins for compact design
        
        # Message text with dynamic sizing and markdown support
        message_text = QTextBrowser()
        message_text.setMarkdown(self.text)
        message_text.setOpenExternalLinks(True)  # Enable clickable links
        
        # Calculate initial height based on content
        self.adjust_message_height(message_text)
        
        message_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        message_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        message_text.setReadOnly(True)
        
        # Store reference for dynamic updates
        self.message_text_widget = message_text
        
        # Ensure text color is always readable with markdown support
        message_text.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                color: #000000;
                font-size: 14px;
                line-height: 1.5;
                padding: 4px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                color: #1a1a1a;
                margin-top: 8px;
                margin-bottom: 4px;
            }
            QTextBrowser strong {
                font-weight: 600;
                color: #1a1a1a;
            }
            QTextBrowser em {
                font-style: italic;
                color: #4a4a4a;
            }
            QTextBrowser code {
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }
        """)
        
        # Timestamp with better styling
        timestamp_label = QLabel(self.timestamp.strftime("%H:%M"))
        timestamp_label.setStyleSheet("""
            color: rgba(255,255,255,0.7);
            font-size: 10px;
            font-weight: 400;
            margin-top: 2px;
        """ if self.is_user else """
            color: #999999;
            font-size: 10px;
            font-weight: 400;
            margin-top: 2px;
        """)
        
        # User/Assistant label
        sender_label = QLabel("You" if self.is_user else "Assistant")
        sender_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        # Styling based on sender
        if self.is_user:
            # User message - right aligned, modern blue with subtle shadow
            bubble.setStyleSheet("""
                QFrame {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #4fc3f7, stop:1 #2196f3);
                    border: none;
                    border-radius: 12px;
                    padding: 8px;
                }
            """)
            sender_label.setStyleSheet("""
                font-weight: 600; 
                font-size: 11px; 
                color: #ffffff;
                margin-bottom: 2px;
            """)
            # Override text color for user messages with markdown support
            message_text.setStyleSheet("""
                QTextBrowser {
                    background-color: transparent;
                    border: none;
                    color: #ffffff;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 4px;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                }
                QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                    color: #ffffff;
                    margin-top: 8px;
                    margin-bottom: 4px;
                }
                QTextBrowser strong {
                    font-weight: 600;
                    color: #ffffff;
                }
                QTextBrowser em {
                    font-style: italic;
                    color: #f0f0f0;
                }
                QTextBrowser ul, QTextBrowser ol {
                    margin-left: 16px;
                    margin-top: 4px;
                    margin-bottom: 4px;
                }
                QTextBrowser li {
                    margin-bottom: 2px;
                }
                QTextBrowser code {
                    background-color: rgba(255,255,255,0.2);
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 13px;
                    color: #ffffff;
                }
                QTextBrowser blockquote {
                    border-left: 3px solid rgba(255,255,255,0.3);
                    padding-left: 12px;
                    margin-left: 8px;
                    font-style: italic;
                    color: #f0f0f0;
                }
            """)
            layout.addStretch()
            layout.addWidget(bubble)
            bubble.setMaximumWidth(450)  # Responsive width
            bubble.setMinimumWidth(100)
        else:
            # Assistant message - left aligned, clean white with subtle border
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #ffffff;
                    border: 1px solid #e8e8e8;
                    border-radius: 12px;
                    padding: 8px;
                }
            """)
            sender_label.setStyleSheet("""
                font-weight: 600; 
                font-size: 11px; 
                color: #666666;
                margin-bottom: 2px;
            """)
            # Ensure text is dark for assistant messages with markdown support
            message_text.setStyleSheet("""
                QTextBrowser {
                    background-color: transparent;
                    border: none;
                    color: #2c2c2c;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 4px;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                }
                QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                    color: #1a1a1a;
                    margin-top: 8px;
                    margin-bottom: 4px;
                }
                QTextBrowser strong {
                    font-weight: 600;
                    color: #1a1a1a;
                }
                QTextBrowser em {
                    font-style: italic;
                    color: #4a4a4a;
                }
                QTextBrowser ul, QTextBrowser ol {
                    margin-left: 16px;
                    margin-top: 4px;
                    margin-bottom: 4px;
                }
                QTextBrowser li {
                    margin-bottom: 2px;
                }
                QTextBrowser code {
                    background-color: #f4f4f4;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 13px;
                }
                QTextBrowser blockquote {
                    border-left: 3px solid #e0e0e0;
                    padding-left: 12px;
                    margin-left: 8px;
                    font-style: italic;
                    color: #666666;
                }
            """)
            layout.addWidget(bubble)
            layout.addStretch()
            bubble.setMaximumWidth(500)  # Larger for assistant responses
            bubble.setMinimumWidth(100)
        
        # Add content to bubble
        bubble_layout.addWidget(sender_label)
        bubble_layout.addWidget(message_text)
        bubble_layout.addWidget(timestamp_label)
    
    def adjust_message_height(self, text_widget):
        """Dynamically adjust message height based on content"""
        # Calculate required height based on content
        document = text_widget.document()
        document.setTextWidth(350 if self.is_user else 400)  # Match bubble width
        height = document.size().height()
        
        # Set minimum and maximum constraints
        min_height = 30
        max_height = 400  # Allow larger messages
        
        # Apply calculated height with constraints
        final_height = max(min_height, min(height + 20, max_height))  # +20 for padding
        text_widget.setFixedHeight(int(final_height))
    
    def update_text(self, text):
        """Update message text (for streaming) with dynamic sizing"""
        self.text = text
        
        # Update text and dynamically adjust size
        if hasattr(self, 'message_text_widget'):
            self.message_text_widget.setMarkdown(text)
            
            # Dynamically adjust height as content grows
            self.adjust_message_height(self.message_text_widget)
            
            # Reapply styling to ensure text remains visible with markdown support
            if self.is_user:
                self.message_text_widget.setStyleSheet("""
                    QTextBrowser {
                        background-color: transparent;
                        border: none;
                        color: #ffffff;
                        font-size: 14px;
                        line-height: 1.5;
                        padding: 4px;
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                    }
                    QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                        color: #ffffff;
                        margin-top: 8px;
                        margin-bottom: 4px;
                    }
                    QTextBrowser strong {
                        font-weight: 600;
                        color: #ffffff;
                    }
                    QTextBrowser em {
                        font-style: italic;
                        color: #f0f0f0;
                    }
                    QTextBrowser code {
                        background-color: rgba(255,255,255,0.2);
                        padding: 2px 4px;
                        border-radius: 3px;
                        font-family: 'Consolas', 'Monaco', monospace;
                        font-size: 13px;
                        color: #ffffff;
                    }
                """)
            else:
                self.message_text_widget.setStyleSheet("""
                    QTextBrowser {
                        background-color: transparent;
                        border: none;
                        color: #2c2c2c;
                        font-size: 14px;
                        line-height: 1.5;
                        padding: 4px;
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                    }
                    QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                        color: #1a1a1a;
                        margin-top: 8px;
                        margin-bottom: 4px;
                    }
                    QTextBrowser strong {
                        font-weight: 600;
                        color: #1a1a1a;
                    }
                    QTextBrowser em {
                        font-style: italic;
                        color: #4a4a4a;
                    }
                    QTextBrowser code {
                        background-color: #f4f4f4;
                        padding: 2px 4px;
                        border-radius: 3px;
                        font-family: 'Consolas', 'Monaco', monospace;
                        font-size: 13px;
                    }
                """)
        else:
            # Fallback for older messages
            for child in self.findChildren(QTextBrowser):
                child.setMarkdown(text)
                break


class QAPanel(QWidget):
    """Panel for Question Answering with chat interface and conversation history"""
    
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.conversation_history = []
        self.current_streaming_message = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with chat info
        header_layout = QHBoxLayout()
        chat_label = QLabel("💬 Chat with your Telegram Messages")
        chat_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        
        # Clear chat button
        self.clear_chat_btn = QPushButton("🗑️ Clear Chat")
        self.clear_chat_btn.clicked.connect(self.clear_chat)
        self.clear_chat_btn.setMaximumWidth(100)
        
        # Export chat button
        self.export_chat_btn = QPushButton("💾 Export")
        self.export_chat_btn.clicked.connect(self.export_chat)
        self.export_chat_btn.setMaximumWidth(80)
        
        header_layout.addWidget(chat_label)
        header_layout.addStretch()
        header_layout.addWidget(self.export_chat_btn)
        header_layout.addWidget(self.clear_chat_btn)
        
        # Chat display area with modern styling
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #fafafa;
                border: none;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
        """)
        
        # Chat container
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()  # Push messages to top
        self.chat_scroll.setWidget(self.chat_container)
        
        # Welcome message
        self.add_welcome_message()
        
        # Input area
        input_group = QGroupBox("Send Message")
        input_layout = QVBoxLayout(input_group)
        
        # Question input with improved styling
        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText("Type your question about the Telegram messages here...")
        self.question_input.setMaximumHeight(80)  # Compact input height
        self.question_input.setStyleSheet("""
            QPlainTextEdit {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QPlainTextEdit:focus {
                border-color: #2196f3;
            }
        """)
        
        # Enhanced input controls
        controls_layout = QHBoxLayout()
        
        # Channel filter (compact)
        self.qa_channel_filter = QLineEdit()
        self.qa_channel_filter.setPlaceholderText("Filter by channel (optional)")
        self.qa_channel_filter.setMaximumWidth(200)
        
        # Send button
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        # Settings button for parameters
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        
        controls_layout.addWidget(QLabel("Channel:"))
        controls_layout.addWidget(self.qa_channel_filter)
        controls_layout.addStretch()
        controls_layout.addWidget(self.settings_btn)
        controls_layout.addWidget(self.send_btn)
        
        input_layout.addWidget(self.question_input)
        input_layout.addLayout(controls_layout)
        
        # Parameters (initially hidden)
        self.setup_parameters()
        
        # Add to main layout
        layout.addLayout(header_layout)
        layout.addWidget(self.chat_scroll, 1)  # Give chat area most space
        layout.addWidget(input_group)
        
        # Connect Enter key to send message
        self.question_input.installEventFilter(self)
        
        # Auto-scroll to bottom when new messages are added
        self.chat_layout.addStretch()
    
    def filter_thinking_blocks(self, text):
        """Remove <think></think> blocks from text if show_thinking is False"""
        if self.show_thinking:
            return text
        
        # Remove thinking blocks using regex
        # Pattern to match <think>...</think> blocks (including multiline and variations)
        # This handles variations like <think>, </think>, <THINK>, etc.
        pattern = r'<\s*think\s*>.*?</\s*think\s*>'
        filtered_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any extra whitespace that might be left
        filtered_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', filtered_text.strip())
        
        return filtered_text
    
    def setup_parameters(self):
        """Setup generation parameters (collapsible)"""
        # Default parameter values
        self.temperature = 0.7
        self.num_context = 5
        self.repetition_penalty = 1.1
        self.show_retrieved = False
        self.show_thinking = False  # Option to show/hide <think> blocks
    
    def add_welcome_message(self):
        """Add a welcome message to start the chat"""
        welcome_text = """# 👋 Welcome to your Telegram RAG Assistant!

I'm here to help you explore and analyze your Telegram messages using advanced AI. Here's what I can do:

## 🔍 Search & Analysis
- **Find specific topics** or keywords across all channels
- **Summarize discussions** from particular time periods  
- **Identify trending topics** and key insights

## 💬 Example Questions
- *"What were the main topics in the guardian channel this week?"*
- *"Show me recent AI and machine learning discussions"*
- *"Summarize the latest financial news and updates"*

## ✨ Getting Started
Simply type your question below and I'll search through your processed Telegram messages to provide **detailed and contextual answers**.

## ⚙️ Settings & Features
Use the **Settings** button to customize:
- Response temperature and context
- Whether to show AI thinking process
- Number of retrieved messages

> **Note**: Responses now support full markdown formatting including **bold text**, *italics*, `code snippets`, lists, and more!

Ready to explore your data? Ask me anything!"""
        
        welcome_msg = ChatMessage(welcome_text, is_user=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, welcome_msg)
        self.conversation_history.append({"role": "assistant", "content": welcome_text, "timestamp": datetime.now()})
    
    def eventFilter(self, obj, event):
        """Handle keyboard events for the input field"""
        if obj == self.question_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                # Ctrl+Enter to send message
                self.send_message()
                return True
            elif event.key() == Qt.Key_Return and event.modifiers() == Qt.NoModifier:
                # Enter for new line (default behavior)
                return False
        return super().eventFilter(obj, event)
    
    def send_message(self):
        """Send a user message and get AI response"""
        question = self.question_input.toPlainText().strip()
        if not question:
            return
        
        # Add user message to chat
        user_msg = ChatMessage(question, is_user=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_msg)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": question, 
            "timestamp": datetime.now()
        })
        
        # Clear input
        self.question_input.clear()
        
        # Disable send button
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Generating...")
        
        # Add typing indicator for assistant
        typing_msg = TypingIndicator()
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, typing_msg)
        self.current_streaming_message = typing_msg
        
        # Scroll to bottom
        self.scroll_to_bottom()
        
        # Get parameters
        channel = self.qa_channel_filter.text().strip()
        
        # Create streaming worker
        streaming_callback = StreamingCallback(show_thinking=self.show_thinking)
        streaming_callback.text_updated.connect(self.update_streaming_message)
        streaming_callback.finished.connect(self.message_finished)
        
        worker = StreamingWorker(
            streaming_callback, question, channel, self.temperature, 
            self.num_context, self.show_retrieved, self.repetition_penalty
        )
        
        self.thread_pool.start(worker)
        
        # Safety timeout to re-enable button (60 seconds)
        self.safety_timer = QTimer()
        self.safety_timer.setSingleShot(True)
        self.safety_timer.timeout.connect(self.force_finish_generation)
        self.safety_timer.start(60000)  # 60 seconds
    
    def update_streaming_message(self, text):
        """Update the streaming assistant message"""
        if self.current_streaming_message:
            # Text is already filtered by StreamingCallback if needed
            self.current_streaming_message.update_text(text)
            self.scroll_to_bottom()
    
    def message_finished(self):
        """Handle completion of message generation"""
        # Stop safety timer
        if hasattr(self, 'safety_timer'):
            self.safety_timer.stop()
        
        # Re-enable send button
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        
        # Add final message to conversation history
        if self.current_streaming_message:
            final_text = self.current_streaming_message.text
            # Text is already filtered by StreamingCallback if needed
            self.conversation_history.append({
                "role": "assistant",
                "content": final_text,
                "timestamp": datetime.now()
            })
        
        self.current_streaming_message = None
        self.scroll_to_bottom()
    
    def force_finish_generation(self):
        """Force finish generation if it takes too long"""
        print("Safety timeout: Forcing generation to finish")
        
        # Update current message if it exists
        if self.current_streaming_message:
            if hasattr(self.current_streaming_message, 'text') and self.current_streaming_message.text:
                # Keep existing text
                pass
            else:
                # Add timeout message
                self.current_streaming_message.update_text("Response generation timed out. Please try again.")
        
        # Call the normal finish method
        self.message_finished()
    
    def scroll_to_bottom(self):
        """Smooth scroll chat to bottom to show latest messages"""
        scrollbar = self.chat_scroll.verticalScrollBar()
        
        # Create smooth scrolling animation
        self.scroll_animation = QPropertyAnimation(scrollbar, b"value")
        self.scroll_animation.setDuration(300)  # 300ms animation
        self.scroll_animation.setStartValue(scrollbar.value())
        self.scroll_animation.setEndValue(scrollbar.maximum())
        self.scroll_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Small delay to ensure layout is updated
        QTimer.singleShot(50, self.scroll_animation.start)
    
    def clear_chat(self):
        """Clear the chat history"""
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 
            "Clear Chat History",
            "Are you sure you want to clear the entire chat history?\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove all chat messages except the stretch
            while self.chat_layout.count() > 1:
                child = self.chat_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            # Clear history and add welcome message
            self.conversation_history.clear()
            self.add_welcome_message()
    
    def export_chat(self):
        """Export chat history to a file"""
        if not self.conversation_history:
            QMessageBox.information(self, "Export Chat", "No chat history to export.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat History",
            f"telegram_rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    # Export as JSON
                    export_data = []
                    for msg in self.conversation_history:
                        export_data.append({
                            "role": msg["role"],
                            "content": msg["content"],
                            "timestamp": msg["timestamp"].isoformat()
                        })
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                else:
                    # Export as text
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("Telegram RAG Chat History\n")
                        f.write("=" * 50 + "\n\n")
                        
                        for msg in self.conversation_history:
                            timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                            role = "User" if msg["role"] == "user" else "Assistant"
                            f.write(f"[{timestamp}] {role}:\n")
                            f.write(f"{msg['content']}\n\n")
                
                QMessageBox.information(self, "Export Successful", f"Chat history exported to:\n{filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export chat history:\n{str(e)}")
    
    def show_settings(self):
        """Show generation parameters settings dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Generation Settings")
        dialog.setModal(True)
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout(dialog)
        
        # Temperature
        temp_slider = QSlider(Qt.Horizontal)
        temp_slider.setRange(1, 10)
        temp_slider.setValue(int(self.temperature * 10))
        temp_label = QLabel(f"{self.temperature:.1f}")
        temp_slider.valueChanged.connect(lambda v: temp_label.setText(f"{v/10:.1f}"))
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(temp_slider)
        temp_layout.addWidget(temp_label)
        layout.addRow("Temperature (creativity):", temp_layout)
        
        # Context messages
        context_slider = QSlider(Qt.Horizontal)
        context_slider.setRange(1, 100)
        context_slider.setValue(self.num_context)
        context_label = QLabel(str(self.num_context))
        context_slider.valueChanged.connect(lambda v: context_label.setText(str(v)))
        
        context_layout = QHBoxLayout()
        context_layout.addWidget(context_slider)
        context_layout.addWidget(context_label)
        layout.addRow("Context Messages:", context_layout)
        
        # Repetition penalty
        rep_slider = QSlider(Qt.Horizontal)
        rep_slider.setRange(100, 150)
        rep_slider.setValue(int(self.repetition_penalty * 100))
        rep_label = QLabel(f"{self.repetition_penalty:.2f}")
        rep_slider.valueChanged.connect(lambda v: rep_label.setText(f"{v/100:.2f}"))
        
        rep_layout = QHBoxLayout()
        rep_layout.addWidget(rep_slider)
        rep_layout.addWidget(rep_label)
        layout.addRow("Repetition Penalty:", rep_layout)
        
        # Show retrieved context
        retrieved_checkbox = QCheckBox()
        retrieved_checkbox.setChecked(self.show_retrieved)
        layout.addRow("Show Retrieved Context:", retrieved_checkbox)
        
        # Show thinking process
        thinking_checkbox = QCheckBox()
        thinking_checkbox.setChecked(self.show_thinking)
        thinking_checkbox.setToolTip("Show the AI's internal reasoning process (content between <think></think> tags)")
        layout.addRow("Show Thinking Process:", thinking_checkbox)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)
        
        # Apply settings if accepted
        if dialog.exec() == QDialog.Accepted:
            self.temperature = temp_slider.value() / 10.0
            self.num_context = context_slider.value()
            self.repetition_penalty = rep_slider.value() / 100.0
            self.show_retrieved = retrieved_checkbox.isChecked()
            self.show_thinking = thinking_checkbox.isChecked()


class StreamingCallback(QObject):
    """Callback handler for streaming text updates with thinking block filtering"""
    
    text_updated = Signal(str)
    finished = Signal()
    
    def __init__(self, show_thinking=False):
        super().__init__()
        self.last_text = ""
        self.accumulated_text = ""
        self.show_thinking = show_thinking
        self.in_thinking_block = False
        self.thinking_buffer = ""
        self.post_thinking_started = False
    
    def emit(self, text):
        """Emit text update signal with real-time thinking block filtering"""
        self.accumulated_text = text
        
        if self.show_thinking:
            # If showing thinking is enabled, just emit everything
            self.text_updated.emit(text)
            self.last_text = text
        else:
            # Filter out thinking blocks in real-time
            filtered_text = self._filter_thinking_realtime(text)
            if filtered_text != self.last_text:  # Only emit if content changed
                self.text_updated.emit(filtered_text)
                self.last_text = filtered_text
            elif not self.last_text and not filtered_text:
                # If no content yet and filtered text is empty, emit empty to maintain UI state
                self.text_updated.emit("")
        
        # Check for completion conditions
        if (text.endswith("Complete!") or 
            "Error:" in text or 
            text.endswith("---") or
            "Failed to" in text):
            self.finished.emit()
    
    def _filter_thinking_realtime(self, text):
        """Filter thinking blocks in real-time during streaming"""
        # First, remove any complete thinking blocks
        complete_think_pattern = r'<\s*think\s*>.*?</\s*think\s*>'
        filtered_text = re.sub(complete_think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Check if there's an unclosed thinking block
        # Look for <think> that doesn't have a corresponding </think>
        think_start_pattern = r'<\s*think\s*>'
        think_end_pattern = r'</\s*think\s*>'
        
        # Find all opening and closing tags
        start_matches = list(re.finditer(think_start_pattern, filtered_text, re.IGNORECASE))
        end_matches = list(re.finditer(think_end_pattern, filtered_text, re.IGNORECASE))
        
        # If we have more opening tags than closing tags, there's an unclosed block
        if len(start_matches) > len(end_matches):
            # Find the position of the last unclosed <think> tag
            last_start = start_matches[-1].start()
            # Remove everything from that point onward
            filtered_text = filtered_text[:last_start]
        
        # Also handle cases where we might be in the middle of typing <think>
        # Remove any partial <think tag at the end
        partial_tag_pattern = r'<\s*t(?:h(?:i(?:n(?:k)?)?)?)?\s*$'
        filtered_text = re.sub(partial_tag_pattern, '', filtered_text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        filtered_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', filtered_text.strip())
        
        return filtered_text
    
    def force_finish(self):
        """Force completion signal"""
        self.finished.emit()


class TelegramRAGMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("TelegramRAG", "MainApp")
        self.model_worker = None
        self.setup_ui()
        self.setup_workers()
        self.connect_signals()  # Move here after workers are set up
        self.load_settings()
        
    def setup_ui(self):
        self.setWindowTitle("Telegram RAG System - Qt Interface")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)  # Set a comfortable default size
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Configuration and Status
        left_panel = QWidget()
        left_panel.setMinimumWidth(420)  # Ensure minimum width for proper display
        left_panel.setMaximumWidth(500)  # Prevent it from getting too wide
        left_layout = QVBoxLayout(left_panel)
        
        # Model configuration
        self.model_config_panel = ModelConfigPanel()
        left_layout.addWidget(self.model_config_panel)
        
        # Status panel
        self.status_panel = StatusPanel()
        left_layout.addWidget(self.status_panel)
        
        # Right panel - Main functionality
        right_panel = QTabWidget()
        
        # Telegram operations tab
        self.telegram_panel = TelegramPanel()
        right_panel.addTab(self.telegram_panel, "Telegram Operations")
        
        # Query tab
        self.query_panel = QueryPanel()
        right_panel.addTab(self.query_panel, "Query Messages")
        
        # Q&A tab
        self.qa_panel = QAPanel()
        right_panel.addTab(self.qa_panel, "Question Answering")
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (left panel: 450px, right panel: remainder)
        splitter.setSizes([450, 1150])
        splitter.setStretchFactor(0, 0)  # Left panel fixed-ish
        splitter.setStretchFactor(1, 1)  # Right panel stretches
        
        # Setup menu bar
        self.setup_menu_bar()
        
        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Center window on screen
        self.center_on_screen()
    
    def setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Theme toggle
        theme_action = QAction("Toggle Dark/Light Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_workers(self):
        """Setup background workers"""
        self.model_worker = ModelWorker()
        
        # Connect worker signals
        self.model_worker.progress_updated.connect(self.status_panel.show_progress)
        self.model_worker.operation_completed.connect(self.on_operation_completed)
        self.model_worker.error_occurred.connect(self.on_error_occurred)
    
    def connect_signals(self):
        """Connect all UI signals"""
        # Model configuration signals
        self.model_config_panel.model_reload_requested.connect(self.reload_models)
        self.model_config_panel.device_changed.connect(self.status_panel.update_device_status)
        
        # Telegram operations signals
        self.telegram_panel.download_requested.connect(self.download_messages)
        self.telegram_panel.process_requested.connect(self.process_messages)
        self.telegram_panel.channels_fetch_requested.connect(self.fetch_channels)
        
        # Connect model worker channels signal to telegram panel
        self.model_worker.channels_fetched.connect(self.telegram_panel.on_channels_fetched)
    
    def reload_models(self, language, llm_model, embedding_model, reranker_model, llm_precision, device, embedding_type):
        """Reload models with new configuration"""
        self.status_panel.log_message(f"Reloading models: {llm_model}, Device: {device}")
        
        # Update backend model paths before loading (supports NPU)
        try:
            # Determine if NPU and if selected model is NPU-optimized
            if is_npu_device(device) and is_npu_compatible_model(llm_model):
                # Ensure NPU models are added to config
                add_npu_models_to_config()
                # Find NPU model info by repo_id
                npu_list = get_npu_models("llm")
                selected_info = None
                for m in npu_list:
                    if m.get("repo_id") == llm_model or m.get("name") == llm_model:
                        selected_info = m
                        break
                if selected_info:
                    # Set LLM path to NPU-optimized model
                    gr_backend.llm_model_dir = get_npu_model_path(selected_info, models_dir)
                else:
                    # Fallback to standard OV model repo
                    gr_backend.llm_model_dir = download_ov_model_if_needed(llm_model, llm_precision, "llm") or get_ov_model_path(llm_model, llm_precision)
            else:
                # Standard OpenVINO preconverted model
                gr_backend.llm_model_dir = download_ov_model_if_needed(llm_model, llm_precision, "llm") or get_ov_model_path(llm_model, llm_precision)
            
            # Embedding and reranker (standard OV repos)
            gr_backend.embedding_model_dir = download_ov_model_if_needed(embedding_model, "int8", "embedding") or Path(embedding_model)
            gr_backend.rerank_model_dir = download_ov_model_if_needed(reranker_model, "int8", "rerank") or Path(reranker_model)
        except Exception as e:
            self.status_panel.log_message(f"Error preparing model paths: {e}")
        
        # Update status
        self.status_panel.update_model_status("Loading...")
        
        # Start model loading in background (will use updated paths)
        thread = threading.Thread(
            target=self.model_worker.load_models,
            args=(device, embedding_type)
        )
        thread.start()
    
    def download_messages(self, channels, limit, hours):
        """Download messages in background"""
        self.status_panel.log_message(f"Downloading messages from {len(channels)} channels")
        
        # Start download in background
        thread = threading.Thread(
            target=self.model_worker.download_messages_async,
            args=(channels, limit, hours)
        )
        thread.start()
    
    def process_messages(self):
        """Process messages in background"""
        self.status_panel.log_message("Processing messages into vector store")
        
        # Start processing in background
        thread = threading.Thread(
            target=self.model_worker.process_messages_async
        )
        thread.start()
    
    def fetch_channels(self):
        """Fetch user's channels in background"""
        self.status_panel.log_message("Fetching user's Telegram channels")
        
        # Start fetching in background
        thread = threading.Thread(
            target=self.model_worker.fetch_channels_async
        )
        thread.start()
    
    def on_operation_completed(self, message):
        """Handle completed operations"""
        self.status_panel.log_message(f"Operation completed: {message}")
        self.status_bar.showMessage(message, 5000)
        
        # Update status panels if needed
        if "download" in message.lower():
            self.telegram_panel.update_download_status(message)
        elif "process" in message.lower():
            self.telegram_panel.update_process_status(message)
        elif "model" in message.lower():
            self.status_panel.update_model_status("Loaded")
    
    def on_error_occurred(self, error_message):
        """Handle errors"""
        self.status_panel.log_message(f"Error: {error_message}")
        self.status_bar.showMessage(f"Error: {error_message}", 5000)
        
        # Show error dialog
        QMessageBox.critical(self, "Error", error_message)
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setModal(True)
        layout = QFormLayout(dialog)
        
        # Add some basic settings
        theme_combo = QComboBox()
        theme_combo.addItems(["Auto", "Light", "Dark"])
        layout.addRow("Theme:", theme_combo)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)
        
        dialog.exec()
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        app = QApplication.instance()
        palette = app.palette()
        
        if palette.color(QPalette.Window).lightness() > 128:
            # Switch to dark theme
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
            dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            app.setPalette(dark_palette)
        else:
            # Switch to light theme
            app.setPalette(QApplication.style().standardPalette())
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Telegram RAG System", 
                         """<h2>Telegram RAG System</h2>
                         <p>A modern desktop application for Telegram message analysis using OpenVINO and Qt for Python.</p>
                         <p><b>Features:</b></p>
                         <ul>
                         <li>TextEmbeddingPipeline integration</li>
                         <li>Real-time streaming responses</li>
                         <li>GPU optimization and diagnostics</li>
                         <li>Modern Qt interface</li>
                         </ul>
                         <p><b>Built with:</b> Qt for Python (PySide6), OpenVINO, LangChain</p>
                         <p><a href="https://doc.qt.io/qtforpython-6/">Qt for Python Documentation</a></p>""")
    
    def load_settings(self):
        """Load application settings"""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = self.settings.value("windowState") 
        if state:
            self.restoreState(state)
    
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def center_on_screen(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            window_geometry = self.geometry()
            
            # Calculate center position
            x = (screen_geometry.width() - window_geometry.width()) // 2
            y = (screen_geometry.height() - window_geometry.height()) // 2
            
            # Move window to center
            self.move(x, y)
    
    def closeEvent(self, event):
        """Handle application close event"""
        self.save_settings()
        event.accept()


def main():
    """Main application entry point"""
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Telegram RAG System")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TelegramRAG")
    
    # Set application icon if available
    # app.setWindowIcon(QIcon("icon.png"))
    
    # Create and show main window
    window = TelegramRAGMainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 