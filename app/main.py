#!/usr/bin/env python3
import sys
import signal
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QFileDialog, QFrame, QSplitter, QTextEdit, QMessageBox,
    QProgressBar, QGroupBox, QFormLayout, QSpinBox, QComboBox,
    QDialog, QDialogButtonBox, QCheckBox, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal, QMimeData, QUrl, QProcess, QObject, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QPalette, QTextCursor, QIcon

# Import MultiFileSelector component
from multi_file_selector import MultiFileSelector
from skimage.color import label2rgb
import numpy as np

# System monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU monitoring will be disabled.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import nvidia_ml_py3 as pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False

if not GPUTIL_AVAILABLE and not PYNVML_AVAILABLE:
    print("Warning: Neither GPUtil nor nvidia-ml-py3 available. GPU monitoring will be disabled.")

# Import viewer component - from same directory
try:
    from multi_viewer import MultiViewer, ViewerConfig, MultiviewerConfig, create_multiviewer
except ImportError as e:
    print(f"Warning: Could not import MultiViewer: {e}")
    MultiViewer = None

# Import core module - no longer directly imported, using process method
# try:
#     from core import run as run_core
# except ImportError as e:
#     print(f"Warning: Could not import core module: {e}")
#     run_core = None

class ConsoleWindow(QDialog):
    """Console window for displaying background running status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Console Output")
        self.setMinimumSize(600, 400)
        self.resize(800, 500)
        
        # Layout
        layout = QVBoxLayout(self)
        
        # Control button area
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_console)
        button_layout.addWidget(self.clear_btn)
        
        self.auto_scroll_checkbox = QCheckBox("Auto Scroll")
        self.auto_scroll_checkbox.setChecked(True)
        button_layout.addWidget(self.auto_scroll_checkbox)
        
        button_layout.addStretch()
        
        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Console text area
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setFont(QFont("Consolas", 10))
        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.console_text)
        
        # Set window properties
        self.setModal(False)  # Non-modal window, can operate with main window simultaneously
    
    def append_text(self, text):
        """Add text to console"""
        self.console_text.append(f"[{self.get_timestamp()}] {text}")
        
        # Auto scroll to bottom
        if self.auto_scroll_checkbox.isChecked():
            scrollbar = self.console_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def clear_console(self):
        """Clear console"""
        self.console_text.clear()
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def closeEvent(self, event):
        """Override close event, hide window instead of destroying"""
        self.hide()
        event.ignore()







class ProcessRunner(QThread):
    """Class for running processing using QThread (replacing QProcess for Nuitka compatibility)"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    console_output = Signal(str)
    finished_signal = Signal(bool, str)
    result_path_generated = Signal(str, str)  # New signal: type, path
    y_value_generated = Signal(str)  # New signal: Y value
    dataframe_generated = Signal(object)  # New signal: DataFrame result
    
    def __init__(self, pre_image_path, pre_mask_path, post_image_path, post_mask_path, parent=None):
        super().__init__(parent)
        self.pre_image_path = pre_image_path
        self.pre_mask_path = pre_mask_path
        self.post_image_path = post_image_path
        self.post_mask_path = post_mask_path
        self._stop_requested = False
        
        # Store generated paths
        self.generated_paths = {
            'pre_bc_path': None,
            'pre_bone_path': None,
            'post_bc_path': None,
            'post_bone_path': None,
            'pre_results_path': None,
            'post_results_path': None
        }
    
    def start_processing(self):
        """Start processing"""
        if self.isRunning():
            return  # Thread already running
        
        self.status_updated.emit("Preparing to start processing...")
        self.console_output.emit("=== Preparing to start processing ===")
        self.progress_updated.emit(5)
        
        # Check if image files exist
        if not os.path.exists(self.pre_image_path):
            error_msg = f"Pre image file not found: {self.pre_image_path}"
            self.console_output.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
            return
            
        if not os.path.exists(self.post_image_path):
            error_msg = f"Post image file not found: {self.post_image_path}"
            self.console_output.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
            return
        
        # Check mask files
        if not os.path.exists(self.pre_mask_path):
            error_msg = f"Pre mask file not found: {self.pre_mask_path}"
            self.console_output.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
            return
            
        if not os.path.exists(self.post_mask_path):
            error_msg = f"Post mask file not found: {self.post_mask_path}"
            self.console_output.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
            return
        
        self.console_output.emit(f"Pre image: {self.pre_image_path}")
        self.console_output.emit(f"Pre mask: {self.pre_mask_path}")
        self.console_output.emit(f"Post image: {self.post_image_path}")
        self.console_output.emit(f"Post mask: {self.post_mask_path}")
        self.status_updated.emit("Starting processing...")
        self.progress_updated.emit(10)
        
        # Start the thread
        self.start()
    
    def run(self):
        """Thread run method"""
        try:
            self.console_output.emit("Starting processing...")
            
            # Import core module
            try:
                from core import run
                self.console_output.emit("Successfully imported core module")
            except ImportError as e:
                error_msg = f"Failed to import core module: {e}"
                self.console_output.emit(error_msg)
                self.finished_signal.emit(False, error_msg)
                return
            
            # Redirect print output to our signal
            import sys
            from io import StringIO
            
            class SignalStream:
                def __init__(self, signal, thread_instance):
                    self.signal = signal
                    self.thread_instance = thread_instance
                    self.buffer = ""
                
                def write(self, text):
                    if text and text.strip():
                        # Handle percentage progress messages
                        if "percentage@" in text:
                            parts = text.strip().split("@")
                            if len(parts) == 2:
                                try:
                                    progress_value = int(parts[1])
                                    self.thread_instance.progress_updated.emit(progress_value)
                                    self.signal.emit(f"Progress: {progress_value}%")
                                except ValueError:
                                    self.signal.emit(text.strip())
                            return
                        
                        # Handle callback messages
                        if "callback@" in text:
                            parts = text.strip().split("@")
                            if len(parts) == 3:
                                path_type = parts[1]
                                path_value = parts[2]
                                self.signal.emit(f"Detected path generation: {path_type} -> {path_value}")
                                self.thread_instance.result_path_generated.emit(path_type, path_value)
                                if path_type == "y":
                                    self.thread_instance.y_value_generated.emit(path_value)
                            return
                        
                        self.signal.emit(text.strip())
                        
                        # Update progress based on output
                        self._update_progress_from_output(text.strip())
                
                def flush(self):
                    pass
                
                def _update_progress_from_output(self, line):
                    """Update progress based on output"""
                    line_lower = line.lower()
                    
                    if "starting processing task" in line_lower:
                        self.thread_instance.progress_updated.emit(15)
                    elif "calculating file md5" in line_lower:
                        self.thread_instance.progress_updated.emit(25)
                    elif "checking segmentation files" in line_lower:
                        self.thread_instance.progress_updated.emit(35)
                    elif "calculating pre phase metrics" in line_lower:
                        self.thread_instance.progress_updated.emit(45)
                    elif "calculating post phase metrics" in line_lower:
                        self.thread_instance.progress_updated.emit(75)
                    elif "calculating final result" in line_lower:
                        self.thread_instance.progress_updated.emit(90)
                    elif "processing task complete" in line_lower:
                        self.thread_instance.progress_updated.emit(100)
            
            # Temporarily redirect stdout
            original_stdout = sys.stdout
            signal_stream = SignalStream(self.console_output, self)
            sys.stdout = signal_stream
            
            try:
                # Execute processing
                self.console_output.emit("Starting core processing...")
                result = run(self.pre_image_path, self.pre_mask_path, self.post_image_path, self.post_mask_path)
                self.console_output.emit(f"Processing result: {result}")
                self.console_output.emit("Processing completed successfully")

                # Handle the result based on its type
                if isinstance(result, dict) and "Probability" in result and "DataFrame" in result:
                    # New format: dictionary with Probability and DataFrame
                    probability = result["Probability"]
                    dataframe = result["DataFrame"]

                    self.console_output.emit(f"Probability: {probability}")
                    self.console_output.emit(f"DataFrame shape: {dataframe.shape}")

                    # Emit the results
                    self.y_value_generated.emit(str(probability))
                    self.dataframe_generated.emit(dataframe)
                else:
                    # Legacy format: single value
                    self.y_value_generated.emit(str(result))

                self.finished_signal.emit(True, "Processing completed successfully")
                
            finally:
                # Restore stdout
                sys.stdout = original_stdout
                
        except Exception as e:
            import traceback
            error_msg = f"Error during processing: {str(e)}"
            self.console_output.emit(error_msg)
            self.console_output.emit(traceback.format_exc())
            self.finished_signal.emit(False, error_msg)
    
    def stop_processing(self):
        """Stop processing thread"""
        if self.isRunning():
            self.console_output.emit("Stopping processing...")
            self._stop_requested = True
            self.quit()
            if self.wait(3000):
                self.console_output.emit("Process stopped")
            else:
                self.console_output.emit("Forcefully stopping process")
                self.terminate()
                self.wait()


class ResultDisplayWidget(QGroupBox):
    """Result display component"""

    def __init__(self, parent=None):
        super().__init__("Processing Results", parent)
        # 设置一致的字体大小，与Visualization面板保持一致
        self.setStyleSheet("QGroupBox { font-size: 12px; }")
        self.init_ui()

    def get_theme_colors(self):
        """Get colors based on current theme"""
        palette = self.palette()
        is_dark_theme = palette.color(QPalette.Window).lightness() < 128

        if is_dark_theme:
            return {
                'text_color': palette.color(QPalette.Text).name(),
                'prob_bg': 'rgba(0, 120, 212, 0.2)',
                'prob_border': '#0078d4',
                'prob_text': '#5dade2',
                'cutoff_bg': 'rgba(255, 152, 0, 0.2)',
                'cutoff_border': '#ff9800',
                'cutoff_text': '#ffb74d',
                'success_bg': 'rgba(40, 167, 69, 0.2)',
                'success_border': '#28a745',
                'success_text': '#4caf50',
                'warning_bg': 'rgba(255, 193, 7, 0.2)',
                'warning_border': '#ffc107',
                'warning_text': '#ffeb3b'
            }
        else:
            return {
                'text_color': '#333',
                'prob_bg': '#f0f8ff',
                'prob_border': '#0078d4',
                'prob_text': '#0078d4',
                'cutoff_bg': '#fff3e0',
                'cutoff_border': '#ff9800',
                'cutoff_text': '#e65100',
                'success_bg': '#e8f5e8',
                'success_border': '#28a745',
                'success_text': '#28a745',
                'warning_bg': '#fff3cd',
                'warning_border': '#ffc107',
                'warning_text': '#856404'
            }

    def init_ui(self):
        """Initialize interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)  # Reduce top and bottom margins
        layout.setSpacing(4)  # Reduce spacing further

        # First row: pCR Probability
        prob_layout = QHBoxLayout()
        prob_layout.setSpacing(8)

        self.prob_label = QLabel("pCR Probability:")
        self.prob_label.setFont(QFont("Arial", 10, QFont.Bold))
        prob_layout.addWidget(self.prob_label)

        self.y_value_label = QLabel("Calculating...")
        self.y_value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        prob_layout.addWidget(self.y_value_label)
        prob_layout.addStretch()

        # Second row: Clinical Cut-off
        cutoff_layout = QHBoxLayout()
        cutoff_layout.setSpacing(8)

        self.cutoff_title_label = QLabel("Clinical Cut-off:")
        self.cutoff_title_label.setFont(QFont("Arial", 10, QFont.Bold))
        cutoff_layout.addWidget(self.cutoff_title_label)

        self.cutoff_value_label = QLabel("0.270")
        self.cutoff_value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        cutoff_layout.addWidget(self.cutoff_value_label)
        cutoff_layout.addStretch()

        layout.addLayout(prob_layout)
        layout.addLayout(cutoff_layout)

        # 设置整个组件的最大高度 - 调整为两行
        self.setMaximumHeight(80)  # 增加高度以适应两行

        # Apply initial theme colors
        self.update_theme_colors()

    def update_theme_colors(self):
        """Update colors based on current theme"""
        colors = self.get_theme_colors()

        # Update label colors
        self.prob_label.setStyleSheet(f"color: {colors['text_color']};")
        self.cutoff_title_label.setStyleSheet(f"color: {colors['text_color']};")

        # Update value label styles
        self.y_value_label.setStyleSheet(f"""
            QLabel {{
                background-color: {colors['prob_bg']};
                border: 1px solid {colors['prob_border']};
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: bold;
                color: {colors['prob_text']};
                text-align: left;
            }}
        """)

        self.cutoff_value_label.setStyleSheet(f"""
            QLabel {{
                background-color: {colors['cutoff_bg']};
                border: 1px solid {colors['cutoff_border']};
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: bold;
                color: {colors['cutoff_text']};
                text-align: left;
            }}
        """)

    def update_y_value(self, y_value):
        """Update Y value display"""
        colors = self.get_theme_colors()

        try:
            # Try to convert y_value to float and format for display
            if isinstance(y_value, (int, float)):
                formatted_value = f"{float(y_value):.3f}"
            else:
                # If it's a string, try to convert
                formatted_value = f"{float(y_value):.3f}"

            self.y_value_label.setText(formatted_value)
            self.y_value_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {colors['success_bg']};
                    border: 1px solid {colors['success_border']};
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-size: 12px;
                    font-weight: bold;
                    color: {colors['success_text']};
                }}
            """)
        except (ValueError, TypeError) as e:
            # If conversion fails, just display the original value
            self.y_value_label.setText(str(y_value))
            self.y_value_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {colors['warning_bg']};
                    border: 1px solid {colors['warning_border']};
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-size: 12px;
                    font-weight: bold;
                    color: {colors['warning_text']};
                }}
            """)

    def reset_display(self):
        """Reset display"""
        colors = self.get_theme_colors()
        self.y_value_label.setText("Calculating...")
        self.y_value_label.setStyleSheet(f"""
            QLabel {{
                background-color: {colors['prob_bg']};
                border: 1px solid {colors['prob_border']};
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: bold;
                color: {colors['prob_text']};
            }}
        """)


class SystemMonitorWidget(QGroupBox):
    """System monitor component showing CPU and GPU usage"""

    def __init__(self, parent=None):
        super().__init__("System Monitor", parent)
        self.init_ui()
        self.init_monitoring()

    def get_theme_colors(self):
        """Get colors based on current theme"""
        palette = self.palette()
        is_dark_theme = palette.color(QPalette.Window).lightness() < 128

        if is_dark_theme:
            return {
                'default_bg': 'rgba(0, 120, 212, 0.2)',
                'default_border': '#0078d4',
                'default_text': '#5dade2',
                'progress_bg': palette.color(QPalette.Window).name(),
                'progress_border': palette.color(QPalette.Mid).name()
            }
        else:
            return {
                'default_bg': '#f0f8ff',
                'default_border': '#0078d4',
                'default_text': '#0078d4',
                'progress_bg': '#f0f0f0',
                'progress_border': '#ccc'
            }

    def init_ui(self):
        """Initialize interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        colors = self.get_theme_colors()

        # CPU usage display
        cpu_layout = QHBoxLayout()
        cpu_layout.setContentsMargins(0, 0, 0, 0)

        cpu_label = QLabel("CPU:")
        cpu_label.setFont(QFont("Arial", 9, QFont.Bold))
        cpu_label.setFixedWidth(35)
        cpu_layout.addWidget(cpu_label)

        self.cpu_value = QLabel("---%")
        self.cpu_value.setStyleSheet(f"""
            QLabel {{
                background-color: {colors['default_bg']};
                border: 1px solid {colors['default_border']};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
                color: {colors['default_text']};
                min-width: 50px;
            }}
        """)
        cpu_layout.addWidget(self.cpu_value)

        # CPU progress bar
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        self.cpu_progress.setFixedHeight(12)
        self.cpu_progress.setTextVisible(False)
        self.cpu_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {colors['progress_border']};
                border-radius: 3px;
                background-color: {colors['progress_bg']};
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.7 #FFC107, stop:1 #F44336);
                border-radius: 2px;
            }}
        """)
        cpu_layout.addWidget(self.cpu_progress)

        layout.addLayout(cpu_layout)

        # GPU memory display
        gpu_layout = QHBoxLayout()
        gpu_layout.setContentsMargins(0, 0, 0, 0)

        gpu_label = QLabel("GPU:")
        gpu_label.setFont(QFont("Arial", 9, QFont.Bold))
        gpu_label.setFixedWidth(35)
        gpu_layout.addWidget(gpu_label)

        self.gpu_value = QLabel("--- MB")
        self.gpu_value.setStyleSheet(f"""
            QLabel {{
                background-color: {colors['default_bg']};
                border: 1px solid {colors['default_border']};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
                color: {colors['default_text']};
                min-width: 50px;
            }}
        """)
        gpu_layout.addWidget(self.gpu_value)

        # GPU progress bar
        self.gpu_progress = QProgressBar()
        self.gpu_progress.setRange(0, 100)
        self.gpu_progress.setFixedHeight(12)
        self.gpu_progress.setTextVisible(False)
        self.gpu_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {colors['progress_border']};
                border-radius: 3px;
                background-color: {colors['progress_bg']};
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2196F3, stop:0.7 #FF9800, stop:1 #F44336);
                border-radius: 2px;
            }}
        """)
        gpu_layout.addWidget(self.gpu_progress)
        
        layout.addLayout(gpu_layout)
        
        # Set fixed height for compact display
        self.setMaximumHeight(85)
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        self.gpu_count = 0
        
        if PYNVML_AVAILABLE:
            try:
                if 'nvidia_ml_py3' in sys.modules:
                    import nvidia_ml_py3 as pynvml
                else:
                    import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.gpu_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"NVIDIA-ML initialized successfully. Found {self.gpu_count} GPU(s)")
            except Exception as e:
                print(f"NVIDIA-ML initialization failed: {e}")
                self.gpu_available = False
        
        if not self.gpu_available and GPUTIL_AVAILABLE:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_available = True
                    self.gpu_count = len(gpus)
                    print(f"GPUtil initialized successfully. Found {self.gpu_count} GPU(s)")
                else:
                    print("GPUtil: No GPUs found")
            except Exception as e:
                print(f"GPUtil initialization failed: {e}")
                self.gpu_available = False
        
        if not self.gpu_available:
            print("No GPU monitoring available")
    
    def init_monitoring(self):
        """Initialize monitoring timer"""
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_system_info)
        self.monitor_timer.start(2000)  # Update every 2 seconds
    
    def update_system_info(self):
        """Update system information"""
        # Update CPU usage
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_value.setText(f"{cpu_percent:.1f}%")
                self.cpu_progress.setValue(int(cpu_percent))
                
                # Update CPU label color based on usage
                if cpu_percent < 50:
                    color = "#4CAF50"  # Green
                elif cpu_percent < 80:
                    color = "#FFC107"  # Yellow
                else:
                    color = "#F44336"  # Red
                
                self.cpu_value.setStyleSheet(f"""
                    QLabel {{
                        background-color: {color}20;
                        border: 1px solid {color};
                        border-radius: 3px;
                        padding: 2px 8px;
                        font-size: 10px;
                        font-weight: bold;
                        color: {color};
                        min-width: 50px;
                    }}
                """)
            except Exception as e:
                self.cpu_value.setText("Error")
                print(f"CPU monitoring error: {e}")
        
        # Update GPU memory usage
        if self.gpu_available:
            try:
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        used_mb = gpu.memoryUsed
                        total_mb = gpu.memoryTotal
                        usage_percent = (used_mb / total_mb) * 100
                        
                        self.gpu_value.setText(f"{used_mb:.0f}/{total_mb:.0f} MB")
                        self.gpu_progress.setValue(int(usage_percent))
                elif PYNVML_AVAILABLE:
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
                    info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_mb = info.used // 1024 // 1024
                    total_mb = info.total // 1024 // 1024
                    usage_percent = (info.used / info.total) * 100
                    
                    self.gpu_value.setText(f"{used_mb}/{total_mb} MB")
                    self.gpu_progress.setValue(int(usage_percent))
                
                # Update GPU label color based on usage
                if usage_percent < 50:
                    color = "#2196F3"  # Blue
                elif usage_percent < 80:
                    color = "#FF9800"  # Orange
                else:
                    color = "#F44336"  # Red
                
                self.gpu_value.setStyleSheet(f"""
                    QLabel {{
                        background-color: {color}20;
                        border: 1px solid {color};
                        border-radius: 3px;
                        padding: 2px 8px;
                        font-size: 10px;
                        font-weight: bold;
                        color: {color};
                        min-width: 50px;
                    }}
                """)
                
            except Exception as e:
                self.gpu_value.setText("No GPU")
                self.gpu_progress.setValue(0)
        else:
            self.gpu_value.setText("N/A")
            self.gpu_progress.setValue(0)


class LeftPanel(QWidget):
    """Left control panel"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer()  # Add timer for elapsed time tracking
        self.start_time = 0  # Record start time
        self.timer.timeout.connect(self.update_elapsed_time)  # Connect timer signal
        self.init_ui()
    
    def init_ui(self):
        """Initialize interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # File selection area
        file_group = QGroupBox("Image Upload")
        file_group.setMaximumHeight(280)  # Reduced height since only 2 files per scan now
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setSpacing(12)
        
        # First file selector - Pre
        self.file_selector1 = MultiFileSelector("Pre-treatment")
        self.file_selector1.setMinimumHeight(120)  # Reduced since only 2 files now
        file_layout.addWidget(self.file_selector1)
        
        # Second file selector - Post
        self.file_selector2 = MultiFileSelector("Post-treatment")
        self.file_selector2.setMinimumHeight(120)  # Reduced since only 2 files now
        file_layout.addWidget(self.file_selector2)
        
        layout.addWidget(file_group)
        
        # Control button area
        control_group = QGroupBox("Processing")
        control_group.setMaximumHeight(120)  # Set maximum height to make it more compact
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(6, 6, 6, 6)  # Reduced margins
        control_layout.setSpacing(6)  # Reduced spacing
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)  # Reduced spacing
        
        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(30)  # More compact height
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;  /* Consistent font size */
                font-weight: bold;
                padding: 4px 12px;  /* Compact padding */
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        button_layout.addWidget(self.run_btn)
        
        self.console_btn = QPushButton("Console")
        self.console_btn.setMinimumHeight(30)  # More compact height
        self.console_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;  /* Consistent font size */
                font-weight: bold;
                padding: 4px 12px;  /* Compact padding */
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        button_layout.addWidget(self.console_btn)
        
        control_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(20)  # Increase height for better text visibility
        control_layout.addWidget(self.progress_bar)
        
        # Status and timer layout
        status_timer_layout = QHBoxLayout()
        status_timer_layout.setContentsMargins(0, 0, 0, 0)
        status_timer_layout.setSpacing(6)
        
        # Status display
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_timer_layout.addWidget(self.status_label)

        # Spacer
        status_timer_layout.addStretch()

        # Timer display
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timer_label.setVisible(False)  # Initially hidden
        status_timer_layout.addWidget(self.timer_label)

        # Apply initial theme colors
        self.update_status_timer_colors()
        
        control_layout.addLayout(status_timer_layout)
        
        layout.addWidget(control_group)
        
        # System monitor area
        # self.system_monitor = SystemMonitorWidget()
        # layout.addWidget(self.system_monitor)

        # DataFrame display area
        dataframe_group = QGroupBox("Results DataFrame")
        # Remove fixed height, let it size based on content
        dataframe_layout = QVBoxLayout(dataframe_group)
        dataframe_layout.setContentsMargins(6, 6, 6, 6)
        dataframe_layout.setSpacing(6)

        # Create table widget for DataFrame display
        from PySide6.QtWidgets import QTableWidget, QHeaderView
        self.dataframe_table = QTableWidget()
        self.dataframe_table.setAlternatingRowColors(True)
        self.dataframe_table.setSelectionBehavior(QTableWidget.SelectRows)

        # Set compact row height
        self.dataframe_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        self.dataframe_table.verticalHeader().setVisible(False)  # Hide row numbers

        # Set size policy to minimize vertical space
        self.dataframe_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Initially hide the table
        self.dataframe_table.setVisible(False)
        dataframe_layout.addWidget(self.dataframe_table)

        # Add placeholder label
        self.dataframe_placeholder = QLabel("DataFrame results will appear here after processing")
        self.dataframe_placeholder.setAlignment(Qt.AlignCenter)
        dataframe_layout.addWidget(self.dataframe_placeholder)

        layout.addWidget(dataframe_group)

        # Processing Results area (moved from right panel)
        self.result_display = ResultDisplayWidget()
        self.result_display.setVisible(False)  # Initially hidden
        layout.addWidget(self.result_display)

        # Spring to push content to top
        layout.addStretch()

        # Apply initial theme colors for placeholder and table
        self.update_placeholder_colors()
        self.update_dataframe_table_colors()

    def get_theme_colors(self):
        """Get colors based on current theme"""
        palette = self.palette()
        is_dark_theme = palette.color(QPalette.Window).lightness() < 128

        if is_dark_theme:
            return {
                'placeholder_color': palette.color(QPalette.Text).name(),
                'status_color': palette.color(QPalette.Text).name(),
                'timer_bg': 'rgba(0, 120, 212, 0.2)',
                'timer_border': '#0078d4',
                'timer_text': '#5dade2',
                'success_bg': 'rgba(40, 167, 69, 0.2)',
                'success_border': '#28a745',
                'success_text': '#4caf50',
                'error_bg': 'rgba(220, 53, 69, 0.2)',
                'error_border': '#dc3545',
                'error_text': '#f44336',
                # Table colors for dark theme
                'table_bg': palette.color(QPalette.Base).name(),
                'table_text': palette.color(QPalette.Text).name(),
                'table_border': palette.color(QPalette.Mid).name(),
                'table_header_bg': palette.color(QPalette.Window).name(),
                'table_selected_bg': palette.color(QPalette.Highlight).name(),
                'table_alternate_bg': palette.color(QPalette.AlternateBase).name()
            }
        else:
            return {
                'placeholder_color': '#999',
                'status_color': '#666',
                'timer_bg': '#f0f8ff',
                'timer_border': '#0078d4',
                'timer_text': '#0078d4',
                'success_bg': '#e8f5e8',
                'success_border': '#28a745',
                'success_text': '#28a745',
                'error_bg': '#ffeaea',
                'error_border': '#dc3545',
                'error_text': '#dc3545',
                # Table colors for light theme
                'table_bg': 'white',
                'table_text': '#333',
                'table_border': '#ddd',
                'table_header_bg': '#f5f5f5',
                'table_selected_bg': '#e3f2fd',
                'table_alternate_bg': '#f9f9f9'
            }

    def update_placeholder_colors(self):
        """Update placeholder colors based on theme"""
        colors = self.get_theme_colors()
        self.dataframe_placeholder.setStyleSheet(f"""
            QLabel {{
                color: {colors['placeholder_color']};
                font-size: 11px;
                font-style: italic;
                padding: 20px;
            }}
        """)

    def update_dataframe_table_colors(self):
        """Update DataFrame table colors based on theme"""
        colors = self.get_theme_colors()
        self.dataframe_table.setStyleSheet(f"""
            QTableWidget {{
                border: 1px solid {colors['table_border']};
                border-radius: 4px;
                background-color: {colors['table_bg']};
                color: {colors['table_text']};
                font-size: 10px;
                gridline-color: {colors['table_border']};
                alternate-background-color: {colors['table_alternate_bg']};
                selection-background-color: {colors['table_selected_bg']};
            }}
            QTableWidget::item {{
                padding: 1px 3px;
                border-bottom: 1px solid {colors['table_border']};
                color: {colors['table_text']};
            }}
            QTableWidget::item:selected {{
                background-color: {colors['table_selected_bg']};
            }}
            QHeaderView::section {{
                background-color: {colors['table_header_bg']};
                color: {colors['table_text']};
                padding: 2px 3px;
                border: 1px solid {colors['table_border']};
                font-weight: bold;
                font-size: 9px;
            }}
            QScrollBar:vertical {{
                background: {colors['table_header_bg']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['table_border']};
                border-radius: 4px;
                min-height: 15px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {colors['table_text']};
            }}
        """)

    def update_status_timer_colors(self):
        """Update status and timer colors based on theme"""
        colors = self.get_theme_colors()

        self.status_label.setStyleSheet(f"color: {colors['status_color']}; font-style: italic; font-size: 10px;")

        self.timer_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['timer_text']};
                font-family: 'Courier New', monospace;
                font-size: 10px;
                font-weight: bold;
                background-color: {colors['timer_bg']};
                border: 1px solid {colors['timer_border']};
                border-radius: 3px;
                padding: 1px 4px;
            }}
        """)

    def start_timer(self):
        """Start timer"""
        from time import time
        colors = self.get_theme_colors()

        self.start_time = time()
        self.timer_label.setVisible(True)
        self.timer_label.setText("00:00:00")
        # Set running style
        self.timer_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['timer_text']};
                font-family: 'Courier New', monospace;
                font-size: 10px;
                font-weight: bold;
                background-color: {colors['timer_bg']};
                border: 1px solid {colors['timer_border']};
                border-radius: 3px;
                padding: 1px 4px;
            }}
        """)
        self.timer.start(1000)  # Update every second

    def stop_timer(self, success=True):
        """Stop timer"""
        colors = self.get_theme_colors()
        self.timer.stop()
        # Don't hide the timer, just update its style to show it's completed
        if success:
            # Green style for successful completion
            self.timer_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['success_text']};
                    font-family: 'Courier New', monospace;
                    font-size: 10px;
                    font-weight: bold;
                    background-color: {colors['success_bg']};
                    border: 1px solid {colors['success_border']};
                    border-radius: 3px;
                    padding: 1px 4px;
                }}
            """)
        else:
            # Red style for failed completion
            self.timer_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['error_text']};
                    font-family: 'Courier New', monospace;
                    font-size: 10px;
                    font-weight: bold;
                    background-color: {colors['error_bg']};
                    border: 1px solid {colors['error_border']};
                    border-radius: 3px;
                    padding: 1px 4px;
                }}
            """)
    
    def update_elapsed_time(self):
        """Update elapsed time display"""
        from time import time
        if self.start_time > 0:
            elapsed = int(time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.timer_label.setText(time_str)
    
    def get_input_data(self):
        """Get all input data"""
        return {
            'file1': self.file_selector1.get_file_paths(),
            'file2': self.file_selector2.get_file_paths()
        }
    
    def validate_inputs(self):
        """Validate inputs"""
        data = self.get_input_data()
        
        if not data['file1']['image']:
            return False, "Please select Pre image file"
        
        if not data['file2']['image']:
            return False, "Please select Post image file"
        
        # Make mask files required
        if not data['file1']['mask']:
            return False, "Please select Pre mask file (Required)"
        
        if not data['file2']['mask']:
            return False, "Please select Post mask file (Required)"
        
        if not os.path.exists(data['file1']['image']):
            return False, f"Pre image file does not exist: {data['file1']['image']}"
        
        if not os.path.exists(data['file2']['image']):
            return False, f"Post image file does not exist: {data['file2']['image']}"
        
        # Validate mask files exist
        if not os.path.exists(data['file1']['mask']):
            return False, f"Pre mask file does not exist: {data['file1']['mask']}"
        
        if not os.path.exists(data['file2']['mask']):
            return False, f"Post mask file does not exist: {data['file2']['mask']}"
        
        return True, "Validation passed"

    def update_dataframe_display(self, dataframe):
        """Update DataFrame display"""
        try:
            if dataframe is None or dataframe.empty:
                self.dataframe_table.setVisible(False)
                self.dataframe_placeholder.setVisible(True)
                self.dataframe_placeholder.setText("No data to display")
                return

            # Set up table dimensions
            self.dataframe_table.setRowCount(len(dataframe))
            self.dataframe_table.setColumnCount(len(dataframe.columns))

            # Hide row numbers (vertical header)
            self.dataframe_table.verticalHeader().setVisible(False)

            # Set headers
            self.dataframe_table.setHorizontalHeaderLabels(dataframe.columns.tolist())

            # Populate table
            for i in range(len(dataframe)):
                for j, col in enumerate(dataframe.columns):
                    value = dataframe.iloc[i, j]
                    # Format numeric values
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            item_text = f"{value:.4f}"
                        else:
                            item_text = str(value)
                    else:
                        item_text = str(value)

                    item = QTableWidgetItem(item_text)
                    # Center align all items
                    item.setTextAlignment(Qt.AlignCenter)
                    self.dataframe_table.setItem(i, j, item)

            # Optimize column widths for better visibility
            header = self.dataframe_table.horizontalHeader()

            # Calculate available width (considering the left panel width)
            available_width = 360  # Adjusted for 400px left panel minus margins
            num_columns = len(dataframe.columns)

            if num_columns > 0:
                # Set uniform column width to fit all columns without scrolling
                uniform_width = available_width // num_columns

                # Set minimum width to ensure readability, but allow smaller for more columns
                min_width = 50  # Reduced from 60 to 50 for more columns
                column_width = max(uniform_width, min_width)

                # If we have many columns, use content-based sizing
                if num_columns > 6:
                    header.setSectionResizeMode(QHeaderView.ResizeToContents)
                    # Set maximum width to prevent overly wide columns
                    for i in range(num_columns):
                        if self.dataframe_table.columnWidth(i) > 80:
                            self.dataframe_table.setColumnWidth(i, 80)
                else:
                    # For fewer columns, use uniform distribution
                    for i in range(num_columns):
                        self.dataframe_table.setColumnWidth(i, column_width)

                    # If total width is less than available, stretch to fill
                    total_width = column_width * num_columns
                    if total_width < available_width:
                        header.setSectionResizeMode(QHeaderView.Stretch)

            # Disable horizontal scrollbar if possible
            self.dataframe_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            # Set table height to fit content exactly
            row_count = len(dataframe)
            header_height = self.dataframe_table.horizontalHeader().height()
            row_height = self.dataframe_table.verticalHeader().defaultSectionSize()
            total_height = header_height + (row_count * row_height) + 4  # +4 for borders
            self.dataframe_table.setFixedHeight(total_height)

            # Show table and hide placeholder
            self.dataframe_table.setVisible(True)
            self.dataframe_placeholder.setVisible(False)

        except Exception as e:
            import traceback
            error_msg = f"Error updating DataFrame display: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            self.dataframe_table.setVisible(False)
            self.dataframe_placeholder.setVisible(True)
            self.dataframe_placeholder.setText(f"Error displaying data: {str(e)}")

    def update_result_display(self, y_value):
        """Update result display in left panel"""
        if hasattr(self, 'result_display') and self.result_display:
            self.result_display.setVisible(True)
            self.result_display.update_y_value(y_value)


class LabelsDisplayWidget(QGroupBox):
    """Labels display component for showing subregion colors"""

    def __init__(self, parent=None):
        super().__init__("Labels", parent)
        # 设置一致的字体大小，与其他面板保持一致
        self.setStyleSheet("QGroupBox { font-size: 12px; }")
        self.init_ui()

    def get_theme_colors(self):
        """Get colors based on current theme"""
        palette = self.palette()
        is_dark_theme = palette.color(QPalette.Window).lightness() < 128

        if is_dark_theme:
            return {
                'text_color': palette.color(QPalette.Text).name(),
                'border_color': palette.color(QPalette.Mid).name()
            }
        else:
            return {
                'text_color': '#555',
                'border_color': '#666'
            }

    def init_ui(self):
        """Initialize labels display"""
        # Set fixed height to fill the space above viewer
        self.setFixedHeight(80)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)  # 与其他面板保持一致的边距
        main_layout.setSpacing(4)  # 减少间距与其他面板保持一致

        # Labels container
        labels_container = QWidget()
        labels_layout = QHBoxLayout(labels_container)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_layout.setSpacing(15)

        # Define subregion colors (matching skimage label2rgb default colors)
        # These are the default colors from matplotlib's tab10 colormap used by label2rgb
        # subregion_colors = [
        #     ("#1f77b4", "SubRegion 1"),  # Blue
        #     ("#ff7f0e", "SubRegion 2"),  # Orange
        #     ("#2ca02c", "SubRegion 3"),  # Green
        #     ("#d62728", "SubRegion 4"),  # Red
        #     ("#9467bd", "SubRegion 5"),  # Purple
        # ]
        # Generate label colors using skimage label2rgb
        label_colors_array = label2rgb(np.arange(0, 6), bg_label=0)
        subregion_colors = []
        for i in range(5):
            # Convert RGB values from 0-1 range to 0-255 range and format as hex
            rgb_values = label_colors_array[i + 1]
            # Convert to 0-255 range and then to integers
            r = int(rgb_values[0] * 255)
            g = int(rgb_values[1] * 255)
            b = int(rgb_values[2] * 255)
            # Format as hex color string
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            subregion_colors.append((hex_color, f"SubRegion {i+1}"))

        # Get theme colors
        theme_colors = self.get_theme_colors()

        # Create color indicators for each subregion
        for color, label in subregion_colors:
            # Container for each label item
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(6)

            # Color indicator
            color_indicator = QLabel()
            color_indicator.setFixedSize(16, 16)
            color_indicator.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border: 1px solid {theme_colors['border_color']};
                    border-radius: 3px;
                }}
            """)
            item_layout.addWidget(color_indicator)

            # Label text
            label_text = QLabel(label)
            label_text.setStyleSheet(f"""
                QLabel {{
                    font-size: 11px;
                    color: {theme_colors['text_color']};
                    font-weight: 500;
                }}
            """)
            item_layout.addWidget(label_text)

            labels_layout.addWidget(item_widget)

        # Add stretch to center the labels
        labels_layout.addStretch()

        main_layout.addWidget(labels_container)


class RightPanel(QWidget):
    """Right display panel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_widget = None
        self.labels_widget = None
        self.init_ui()

    def init_ui(self):
        """Initialize interface"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)  # Reduce spacing from 10 to 5

        # Labels display widget (initially hidden)
        self.labels_widget = LabelsDisplayWidget()
        self.labels_widget.setVisible(False)
        self.layout.addWidget(self.labels_widget)

        # Default display placeholder
        self.placeholder_label = QLabel("Medical image viewer will be displayed here")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.placeholder_label)

        # Apply initial theme colors
        self.update_placeholder_colors()

    def get_theme_colors(self):
        """Get colors based on current theme"""
        palette = self.palette()
        is_dark_theme = palette.color(QPalette.Window).lightness() < 128

        if is_dark_theme:
            return {
                'text_color': palette.color(QPalette.Text).name(),
                'border_color': palette.color(QPalette.Mid).name()
            }
        else:
            return {
                'text_color': '#999',
                'border_color': '#ddd'
            }

    def update_placeholder_colors(self):
        """Update placeholder colors based on theme"""
        colors = self.get_theme_colors()
        self.placeholder_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['text_color']};
                font-size: 16px;
                font-style: italic;
                border: 2px dashed {colors['border_color']};
                border-radius: 8px;
                padding: 50px;
            }}
        """)
    
    def set_widget(self, widget):
        """Set widget to display"""
        # Clear current widget
        if self.current_widget:
            self.layout.removeWidget(self.current_widget)
            self.current_widget.setParent(None)

        # Hide placeholder
        self.placeholder_label.setVisible(False)

        # Show labels widget when MultiViewer is displayed
        try:
            from .multi_viewer import MultiViewer
        except ImportError:
            # Handle case when running as script directly
            try:
                from multi_viewer import MultiViewer
            except ImportError:
                MultiViewer = None

        if MultiViewer and isinstance(widget, MultiViewer):
            self.labels_widget.setVisible(True)
        else:
            self.labels_widget.setVisible(False)

        # Add new widget with proper alignment
        self.current_widget = widget
        self.layout.addWidget(widget, 0, Qt.AlignTop)  # Align to top to prevent stretching
        widget.show()
    
    def clear_widget(self):
        """Clear current widget, show placeholder"""
        if self.current_widget:
            self.layout.removeWidget(self.current_widget)
            self.current_widget.setParent(None)
            self.current_widget = None

        # Hide labels widget when clearing
        self.labels_widget.setVisible(False)
        self.placeholder_label.setVisible(True)


class MedicalImageApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.process_runner = None  # Changed to process_runner
        self.console_window = None  # Console window
        self.viewer = None  # Store viewer reference
        self.image_paths = {}  # Store image and mask paths for different regions
        self.init_ui()
    
    def init_ui(self):
        """Initialize interface"""
        self.setWindowTitle("MediAI Hub: LC-NICER")
        self.setMinimumSize(1400, 620)  # Reduced height from 900 to 700 since red space is minimized

        # Set window icon based on theme
        self.update_window_icon()
        
        # Create console window
        self.console_window = ConsoleWindow(self)
        
        # Central component
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - horizontal splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel
        self.left_panel = LeftPanel()
        self.left_panel.setMaximumWidth(420)  # Optimized for compact layout
        self.left_panel.setMinimumWidth(380)  # Minimum for functionality
        splitter.addWidget(self.left_panel)
        
        # Right panel
        self.right_panel = RightPanel()
        splitter.addWidget(self.right_panel)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 0)  # Left side does not stretch
        splitter.setStretchFactor(1, 1)  # Right side can stretch

        # Set initial splitter sizes - optimize for compact layout
        splitter.setSizes([400, 980])  # Left: 400px, Right: 980px (total 1380px)
        
        # Connect signals
        self.left_panel.run_btn.clicked.connect(self.start_processing)
        self.left_panel.console_btn.clicked.connect(self.show_console)

        # Apply initial theme colors
        self.update_all_theme_colors()

    def get_theme_colors(self):
        """Get colors based on current theme"""
        palette = self.palette()
        is_dark_theme = palette.color(QPalette.Window).lightness() < 128
        return is_dark_theme

    def update_window_icon(self):
        """Update window icon based on current theme"""
        is_dark_theme = self.get_theme_colors()

        if is_dark_theme:
            logo_path = os.path.join(os.path.dirname(__file__), "logo", "white.png")
        else:
            logo_path = os.path.join(os.path.dirname(__file__), "logo", "blue.png")

        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        else:
            # Fallback to original logo.png if theme-specific logo doesn't exist
            fallback_path = os.path.join(os.path.dirname(__file__), "logo.png")
            if os.path.exists(fallback_path):
                self.setWindowIcon(QIcon(fallback_path))

    def update_all_theme_colors(self):
        """Update all theme colors across the application"""
        try:
            # Update window icon based on theme
            self.update_window_icon()

            # Update left panel colors
            if hasattr(self.left_panel, 'update_placeholder_colors'):
                self.left_panel.update_placeholder_colors()
            if hasattr(self.left_panel, 'update_status_timer_colors'):
                self.left_panel.update_status_timer_colors()
            if hasattr(self.left_panel, 'update_dataframe_table_colors'):
                self.left_panel.update_dataframe_table_colors()
            if hasattr(self.left_panel, 'result_display') and self.left_panel.result_display:
                if hasattr(self.left_panel.result_display, 'update_theme_colors'):
                    self.left_panel.result_display.update_theme_colors()

            # Update right panel colors
            if hasattr(self.right_panel, 'update_placeholder_colors'):
                self.right_panel.update_placeholder_colors()
            if hasattr(self.right_panel, 'labels_widget') and self.right_panel.labels_widget:
                # Labels widget colors are applied during creation, no update method needed
                pass

        except Exception as e:
            print(f"Error updating theme colors: {e}")

    def changeEvent(self, event):
        """Handle application change events including theme changes"""
        super().changeEvent(event)
        if event.type() == event.Type.PaletteChange:
            # Theme changed, update all colors
            QTimer.singleShot(100, self.update_all_theme_colors)
    
    def start_processing(self):
        """Start processing"""
        # Validate input
        is_valid, message = self.left_panel.validate_inputs()
        if not is_valid:
            QMessageBox.warning(self, "Input Error", message)
            return
        
        # Get input data
        data = self.left_panel.get_input_data()
        
        # Disable run button and start timer
        self.left_panel.run_btn.setEnabled(False)
        self.left_panel.progress_bar.setVisible(True)
        self.left_panel.progress_bar.setValue(0)
        self.left_panel.start_timer()  # Start timing
        
        # Clear previous viewer and paths
        self.right_panel.clear_widget()
        self.image_paths.clear()  # Clear previous image paths

        # Log the files being used
        self.console_window.append_text(f"Pre image: {data['file1']['image']}")
        self.console_window.append_text(f"Pre mask: {data['file1']['mask']}")
        self.console_window.append_text(f"Post image: {data['file2']['image']}")
        self.console_window.append_text(f"Post mask: {data['file2']['mask']}")
        
        # Create processing thread
        self.process_runner = ProcessRunner(
            data['file1']['image'],  # pre_image_path
            data['file1']['mask'],   # pre_mask_path
            data['file2']['image'],  # post_image_path
            data['file2']['mask']    # post_mask_path
        )
        
        # Connect signals
        self.process_runner.progress_updated.connect(
            self.left_panel.progress_bar.setValue
        )
        self.process_runner.status_updated.connect(
            self.left_panel.status_label.setText
        )
        self.process_runner.console_output.connect(
            self.console_window.append_text
        )
        self.process_runner.finished_signal.connect(
            self.on_processing_finished
        )
        self.process_runner.result_path_generated.connect(
            self.on_result_path_generated
        )
        self.process_runner.y_value_generated.connect(
            self.on_y_value_generated
        )
        self.process_runner.dataframe_generated.connect(
            self.on_dataframe_generated
        )
        
        # Initialize viewer
        self.init_viewer_component()
        
        # Start thread
        self.process_runner.start_processing()
    
    def on_result_path_generated(self, path_type, path_value):
        """Callback when result path is generated"""
        self.console_window.append_text(f"Received callback: {path_type} -> {path_value}")

        if self.viewer is None:
            self.console_window.append_text("Viewer is None, initializing viewer first")
            self.init_viewer_component()
            if self.viewer is None:
                self.console_window.append_text("Failed to initialize viewer")
                return

        # Check if the file really exists
        if not os.path.exists(path_value):
            self.console_window.append_text(f"File does not exist: {path_value}")
            return

        try:
            # Parse the new callback format: pre_peritumor#3_image_path, post_tumor_mask_path, etc.
            self.console_window.append_text(f"Processing callback: {path_type} -> {path_value}")

            # Store the path
            self.image_paths[path_type] = path_value
            self.console_window.append_text(f"Stored path. Current paths: {list(self.image_paths.keys())}")

            # Parse path_type to extract phase, region, and type
            # Expected format: pre_peritumor#3_image_path, post_tumor_mask_path, etc.
            if path_type.endswith('_image_path'):
                # Remove '_image_path' suffix
                base_path = path_type[:-11]  # Remove '_image_path'
                file_type = 'image'
            elif path_type.endswith('_mask_path'):
                # Remove '_mask_path' suffix
                base_path = path_type[:-10]  # Remove '_mask_path'
                file_type = 'mask'
            else:
                self.console_window.append_text(f"Unknown path type format: {path_type}")
                return

            # Split base_path to get phase and region
            parts = base_path.split('_', 1)  # Split only on first underscore
            if len(parts) == 2:
                phase = parts[0]  # pre or post
                region_type = parts[1]  # tumor, peritumor#3, peritumor#5, peritumor#7

                self.console_window.append_text(f"Parsed: phase={phase}, region={region_type}, type={file_type}")

                # Check if we have both image and mask for this phase and region
                image_key = f"{phase}_{region_type}_image_path"
                mask_key = f"{phase}_{region_type}_mask_path"

                self.console_window.append_text(f"Looking for: {image_key} and {mask_key}")

                if image_key in self.image_paths and mask_key in self.image_paths:
                    # We have both image and mask, update the viewer
                    image_path = self.image_paths[image_key]
                    mask_path = self.image_paths[mask_key]

                    self.console_window.append_text(f"Found both files: image={image_path}, mask={mask_path}")

                    # Determine viewer position based on phase and region
                    position = self.get_viewer_position(phase, region_type)
                    self.console_window.append_text(f"Calculated position: {position}")

                    if position is not None:
                        self.update_viewer_at_position(position, image_path, mask_path, phase, region_type)
                    else:
                        self.console_window.append_text(f"Invalid position for {phase} {region_type}")
                else:
                    self.console_window.append_text(f"Waiting for pair: have {image_key}={image_key in self.image_paths}, have {mask_key}={mask_key in self.image_paths}")
            else:
                self.console_window.append_text(f"Invalid base_path format: {base_path}")

        except Exception as e:
            import traceback
            self.console_window.append_text(f"Error processing callback: {str(e)}")
            self.console_window.append_text(traceback.format_exc())

    def get_viewer_position(self, phase, region_type):
        """Get viewer position based on phase and region type"""
        # Map regions to column positions
        region_map = {
            'tumor': 0,
            'peritumor#3': 1,
            'peritumor#5': 2,
            'peritumor#7': 3
        }

        # Map phase to row
        phase_map = {
            'pre': 0,
            'post': 1
        }

        if region_type in region_map and phase in phase_map:
            row = phase_map[phase]
            col = region_map[region_type]
            position = row * 4 + col  # 4 columns per row
            return position

        return None

    def update_viewer_at_position(self, position, image_path, mask_path, phase, region_type):
        """Update viewer at specific position"""
        try:
            # Create viewer config
            title = f"{phase.capitalize()} - {region_type.replace('#', ' #')}"

            viewer_config = ViewerConfig(
                image_path=image_path,
                mask_path=mask_path,
                direction="axial",  # Default to axial view
                title=title
            )

            # Update the viewer
            self.viewer.update_viewer(position, viewer_config)
            self.console_window.append_text(f"Updated viewer at position {position}: {title}")

        except Exception as e:
            self.console_window.append_text(f"Error updating viewer at position {position}: {str(e)}")
    
    def init_viewer_component(self):
        """Initialize viewer component"""
        if MultiViewer is None:
            placeholder = QLabel("MultiViewer component not available\nPlease check multi_viewer.py file path")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                QLabel {
                    color: #d32f2f;
                    font-size: 14px;
                    border: 2px solid #d32f2f;
                    border-radius: 8px;
                    padding: 20px;
                    background-color: #ffeaea;
                }
            """)
            self.right_panel.set_widget(placeholder)
            return

        try:
            # Create empty viewer configuration list (2 rows 4 columns = 8 positions)
            viewers = [None] * 8  # Initialize all to None, will be dynamically loaded later

            # Create MultiViewer configuration
            # Reduce height to better fit the right panel and minimize red space
            config = MultiviewerConfig(
                row=2,
                col=4,
                width=980,   # Slightly reduce width to fit better
                height=450,  # Reduce height from 600 to 500 to minimize red space
                viewers=viewers,
                mask_alpha=0.5
            )

            # Create MultiViewer component
            self.viewer = create_multiviewer(config)

            # Set viewer size to match actual MultiViewer size (height + control panel)
            # MultiViewer actual height = config.height + 60 (for control panel)
            self.viewer.setFixedSize(980, 500)  # Fixed size to prevent stretching
            
            # Set to right panel
            self.right_panel.set_widget(self.viewer)
            
            # Ensure correct initial layout
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            QTimer.singleShot(100, lambda: self._initial_viewer_setup())
            
        except Exception as e:
            error_label = QLabel(f"Error initializing viewer:\n{str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("""
                QLabel {
                    color: #d32f2f;
                    font-size: 12px;
                    border: 1px solid #d32f2f;
                    border-radius: 4px;
                    padding: 15px;
                    background-color: #ffeaea;
                }
            """)
            self.right_panel.set_widget(error_label)
    
    def _initial_viewer_setup(self):
        """Initial viewer setup"""
        if self.viewer is None:
            return
        
        try:
            # Ensure viewer layout is correct
            self.viewer.update()
            
        except Exception as e:
            self.console_window.append_text(f"Error in initial viewer setup: {str(e)}")
    
    def update_viewer_for_pre_bc(self, original_image_path, mask_path):
        """Update Pre mask result display - Legacy method, kept for compatibility"""
        # This method is now handled by the new callback system
        self.console_window.append_text("Legacy pre_bc update method called - using new callback system instead")

    def update_viewer_for_post_bc(self, original_image_path, mask_path):
        """Update Post mask result display - Legacy method, kept for compatibility"""
        # This method is now handled by the new callback system
        self.console_window.append_text("Legacy post_bc update method called - using new callback system instead")
    
    def _force_viewer_refresh(self):
        """Force refresh viewer display"""
        if self.viewer is None:
            return
        
        try:
            # Force update MultiViewer display
            self.viewer.update()
            
        except Exception as e:
            self.console_window.append_text(f"Error forcing viewer refresh: {str(e)}")

    def on_processing_finished(self, success, message):
        """Processing completion callback"""
        # Stop timer and restore interface state
        self.left_panel.stop_timer(success)  # Stop timing
        self.left_panel.run_btn.setEnabled(True)
        self.left_panel.progress_bar.setVisible(False)
        
        if success:
            self.left_panel.status_label.setText("Processing Complete")
            QMessageBox.information(self, "Success", message)
        else:
            self.left_panel.status_label.setText("Processing Failed")
            QMessageBox.critical(self, "Error", message)
    
    def load_viewer_component(self):
        """Load viewer component to right panel - Replaced by init_viewer_component, kept as backup"""
        # This method is now replaced by init_viewer_component, kept as backup
        pass

    def show_console(self):
        """Show console"""
        self.console_window.show()

    def on_y_value_generated(self, y_value):
        """Callback after y value generation"""
        try:
            self.console_window.append_text(f"Y value generated: {y_value}")

            # Update left panel result display
            self.left_panel.update_result_display(y_value)

        except Exception as e:
            self.console_window.append_text(f"Error updating Y value display: {str(e)}")

    def on_dataframe_generated(self, dataframe):
        """Callback when DataFrame is generated"""
        try:
            self.console_window.append_text(f"DataFrame generated with shape: {dataframe.shape}")

            # Update DataFrame display in left panel
            self.left_panel.update_dataframe_display(dataframe)

        except Exception as e:
            self.console_window.append_text(f"Error updating DataFrame display: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        print("Closing application...")
        
        # Stop any running processes
        if self.process_runner:
            try:
                self.process_runner.stop_processing()
            except Exception as e:
                print(f"Error stopping process: {e}")
        
        # Stop system monitoring timer if exists
        if hasattr(self.left_panel, 'system_monitor') and self.left_panel.system_monitor:
            try:
                self.left_panel.system_monitor.monitor_timer.stop()
            except Exception as e:
                print(f"Error stopping monitor timer: {e}")
        
        # Stop any other timers
        if hasattr(self.left_panel, 'timer'):
            try:
                self.left_panel.timer.stop()
            except Exception as e:
                print(f"Error stopping left panel timer: {e}")
        
        # Close console window
        if self.console_window:
            self.console_window.close()
        
        # Accept the close event
        event.accept()
        print("Application closed successfully.")


def main():
    """Main function"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Set application icon
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))
    
    # Set up signal handling for Ctrl+C
    def signal_handler(sig, frame):
        """Handle interrupt signals (Ctrl+C)"""
        print("\nReceived interrupt signal, closing application...")
        app.quit()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Enable processing of keyboard interrupts
    # Create a timer to allow the Python interpreter to process signals
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # Empty lambda to allow signal processing
    timer.start(100)  # Check every 100ms
    
    # Create main window
    window = MedicalImageApp()
    window.show()
    
    # Print usage information
    print("Application started. Press Ctrl+C to exit.")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
