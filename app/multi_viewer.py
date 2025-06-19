"""
Nifti multi-view component - Based on PySide6 implementation
Supports multi-directional slice viewing, mask overlay, slice synchronization and other functions
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.color import label2rgb

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QCheckBox, QDoubleSpinBox,
    QFrame, QSizePolicy, QScrollArea, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont, QPalette


@dataclass
class ViewerConfig:
    """Single viewer configuration parameters"""
    image_path: str
    mask_path: Optional[str] = None
    direction: str = "axial"  # "axial", "coronal", "sagittal"
    title: Optional[str] = None


@dataclass
class MultiviewerConfig:
    """Multi-view component configuration parameters"""
    row: int
    col: int
    width: int
    height: int
    viewers: List[ViewerConfig]
    mask_alpha: float = 0.3


class ImageCache:
    """Image cache manager"""
    
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}
    
    def get_image(self, image_path: str) -> np.ndarray:
        """Get image data, with cache"""
        if image_path not in self._cache:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            nii = sitk.ReadImage(image_path)
            self._cache[image_path] = sitk.GetArrayFromImage(nii)
        
        return self._cache[image_path]
    
    def get_mask(self, mask_path: str) -> np.ndarray:
        """Get mask data, with cache"""
        if mask_path not in self._mask_cache:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
            nii = sitk.ReadImage(mask_path)
            self._mask_cache[mask_path] = sitk.GetArrayFromImage(nii)
        
        return self._mask_cache[mask_path]
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
        self._mask_cache.clear()


class SliceViewer(QWidget):
    """Single slice viewer"""
    
    def __init__(self, config: ViewerConfig, image_cache: ImageCache, 
                 viewer_width: int, viewer_height: int, mask_alpha: float = 0.5):
        super().__init__()
        
        self.config = config
        self.image_cache = image_cache
        self.viewer_width = viewer_width
        self.viewer_height = viewer_height
        self.mask_alpha = mask_alpha
        
        # Load image data
        self.image_data = self.image_cache.get_image(config.image_path)
        self.mask_data = None
        if config.mask_path:
            self.mask_data = self.image_cache.get_mask(config.mask_path)
        
        # Calculate slice count and current slice index
        self.slice_count = self._get_slice_count()
        self.current_slice = self.slice_count // 2  # Start from middle slice
        
        self._setup_ui()
        self._update_display()
    
    def _get_slice_count(self) -> int:
        """Get slice count based on direction"""
        # SimpleITK array dimension order is (z, y, x)
        if self.config.direction == "axial":
            return self.image_data.shape[0]  # z-axis direction
        elif self.config.direction == "coronal":
            return self.image_data.shape[1]  # y-axis direction
        elif self.config.direction == "sagittal":
            return self.image_data.shape[2]  # x-axis direction
        else:
            return self.image_data.shape[0]  # Default axis
    
    def _get_slice_data(self, slice_idx: int) -> np.ndarray:
        """Get image data of specified slice"""
        slice_idx = max(0, min(slice_idx, self.slice_count - 1))
        
        # SimpleITK array dimension order is (z, y, x)
        if self.config.direction == "axial":
            slice_data = self.image_data[slice_idx, :, :]  # Axis: Take z-axis slice
        elif self.config.direction == "coronal":
            slice_data = self.image_data[:, slice_idx, :]  # Coronal: Take y-axis slice
        elif self.config.direction == "sagittal":
            slice_data = self.image_data[:, :, slice_idx]  # Sagittal: Take x-axis slice
        else:
            slice_data = self.image_data[slice_idx, :, :]
        
        # Adjust image transformation according to ITK-SNAP standard
        if self.config.direction == "axial":
            pass
        elif self.config.direction == "coronal":
            slice_data = np.flipud(slice_data)
        elif self.config.direction == "sagittal":
            slice_data = np.flipud(slice_data)
        else:
            # Default axis processing
            slice_data = np.flipud(slice_data)
        
        return slice_data
    
    def _get_mask_slice_data(self, slice_idx: int) -> Optional[np.ndarray]:
        """Get mask data of specified slice"""
        if self.mask_data is None:
            return None
        
        slice_idx = max(0, min(slice_idx, self.slice_count - 1))
        
        # SimpleITK array dimension order is (z, y, x)
        if self.config.direction == "axial":
            mask_slice = self.mask_data[slice_idx, :, :]  # Axis: Take z-axis slice
        elif self.config.direction == "coronal":
            mask_slice = self.mask_data[:, slice_idx, :]  # Coronal: Take y-axis slice
        elif self.config.direction == "sagittal":
            mask_slice = self.mask_data[:, :, slice_idx]  # Sagittal: Take x-axis slice
        else:
            mask_slice = self.mask_data[slice_idx, :, :]
        
        # Adjust mask transformation according to ITK-SNAP standard (consistent with image data)
        if self.config.direction == "axial":
            # Axis: Flip up and down
            # mask_slice = np.flipud(mask_slice)
            pass
        elif self.config.direction == "coronal":
            mask_slice = np.flipud(mask_slice)
        elif self.config.direction == "sagittal":
            mask_slice = np.flipud(mask_slice)
        else:
            # Default axis processing
            mask_slice = np.flipud(mask_slice)
        
        return mask_slice
    
    def _setup_ui(self):
        """Set up UI layout"""
        self.setFixedSize(self.viewer_width, self.viewer_height)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)  # Reduce margin
        layout.setSpacing(1)  # Reduce spacing
        
        # Get system palette for adaptive colors
        palette = QApplication.palette()
        base_color = palette.color(QPalette.Base).name()
        text_color = palette.color(QPalette.WindowText).name()
        button_color = palette.color(QPalette.Button).name()
        
        # A area: Title (height reduced) - now with adaptive colors
        self.title_label = QLabel(self.config.title or "")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFixedHeight(20)  # Reduced from 30 to 20
        self.title_label.setStyleSheet(f"""
            QLabel {{ 
                background-color: {button_color}; 
                border: 1px solid #ccc; 
                font-size: 11px; 
                color: {text_color};
            }}
        """)
        layout.addWidget(self.title_label)
        
        # B area: Image display and navigation (horizontal layout)
        b_widget = QWidget()
        b_layout = QHBoxLayout(b_widget)  # Changed to horizontal layout
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.setSpacing(2)
        
        # b1 area: Image display (square, left)
        available_height = self.viewer_height - 20 - 4  # Subtract title and margin
        available_width = self.viewer_width - 50 - 4  # Reserve 50 pixels for right navigation bar
        image_size = min(available_width, available_height)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(image_size, image_size)
        self.image_label.setStyleSheet("QLabel { border: 1px solid #999; }")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        
        b_layout.addWidget(self.image_label)
        
        # b2 area: Slice navigation (right, vertical layout)
        nav_widget = QWidget()
        nav_widget.setFixedWidth(45)  # Fixed width
        nav_widget.setFixedHeight(image_size)  # Set height same as image
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(2, 2, 2, 2)
        nav_layout.setSpacing(2)
        
        # Vertical slider
        self.slice_slider = QSlider(Qt.Vertical)  # Changed to vertical
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.slice_count - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.setFixedWidth(20)
        self.slice_slider.setMinimumHeight(image_size - 30)  # Set minimum height to almost full image height
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        
        # Slice information label
        self.slice_info_label = QLabel(f"{self.current_slice + 1}/{self.slice_count}")
        self.slice_info_label.setAlignment(Qt.AlignCenter)
        self.slice_info_label.setFixedHeight(15)
        self.slice_info_label.setStyleSheet("QLabel { font-size: 10px; }")
        
        # Center slider and information label horizontally
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addStretch()
        slider_layout.addWidget(self.slice_slider)
        slider_layout.addStretch()
        
        nav_layout.addWidget(self.slice_info_label)
        nav_layout.addWidget(slider_container, 1)  # Give slider container stretch factor of 1
        nav_layout.addStretch(0)  # Remove bottom stretch or give it 0 weight
        
        b_layout.addWidget(nav_widget)
        
        layout.addWidget(b_widget)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        if image.size == 0:
            return image
        
        # Handle NaN and infinity values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min) * 255
        else:
            image = np.zeros_like(image)
        
        return image.astype(np.uint8)
    
    def _create_overlay_image(self, image_slice: np.ndarray, mask_slice: Optional[np.ndarray]) -> np.ndarray:
        """Create image with mask overlay"""
        # Normalize image
        normalized_image = self._normalize_image(image_slice)
        
        if mask_slice is None:
            # Convert to RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1)
        else:
            # Normalize grayscale image to [0,1] range for label2rgb
            image_01 = normalized_image / 255.0
            
            # Use label2rgb to create colored mask overlay
            rgb_image = label2rgb(mask_slice, image=image_01, alpha=self.mask_alpha, bg_label=0)
            
            # Convert back to 0-255 range
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image
    
    def _update_display(self):
        """Update image display"""
        # Get current slice data
        image_slice = self._get_slice_data(self.current_slice)
        mask_slice = self._get_mask_slice_data(self.current_slice)
        
        # Create RGB image
        rgb_image = self._create_overlay_image(image_slice, mask_slice)
        
        # Convert to QImage
        height, width = rgb_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        
        # Update slice information
        self.slice_info_label.setText(f"{self.current_slice + 1}/{self.slice_count}")
    
    def _on_slice_changed(self, value: int):
        """Slice slider change processing"""
        self.current_slice = value
        self._update_display()
    
    def set_slice(self, slice_idx: int):
        """Set current slice"""
        # Ensure slice index is within valid range
        slice_idx = max(0, min(slice_idx, self.slice_count - 1))
        
        if slice_idx != self.current_slice:
            self.current_slice = slice_idx
            self.slice_slider.blockSignals(True)  # Block signals to avoid loop
            self.slice_slider.setValue(slice_idx)
            self.slice_slider.blockSignals(False)
            self._update_display()
    
    def set_mask_alpha(self, alpha: float):
        """Set mask transparency"""
        self.mask_alpha = alpha
        self._update_display()
    
    def keyPressEvent(self, event):
        """Keyboard event processing"""
        if event.key() == Qt.Key_Up or event.key() == Qt.Key_Right:
            new_slice = min(self.current_slice + 1, self.slice_count - 1)
            self.slice_slider.setValue(new_slice)
        elif event.key() == Qt.Key_Down or event.key() == Qt.Key_Left:
            new_slice = max(self.current_slice - 1, 0)
            self.slice_slider.setValue(new_slice)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Mouse wheel event processing for slice navigation"""
        # Get wheel scroll direction
        angle_delta = event.angleDelta().y()
        
        if angle_delta > 0:
            # Wheel up: next slice
            new_slice = min(self.current_slice + 1, self.slice_count - 1)
            self.slice_slider.setValue(new_slice)
        elif angle_delta < 0:
            # Wheel down: previous slice
            new_slice = max(self.current_slice - 1, 0)
            self.slice_slider.setValue(new_slice)
        
        # Accept the event to prevent propagation
        event.accept()


class MultiViewer(QWidget):
    """Multi-view main component"""
    
    def __init__(self, config: MultiviewerConfig):
        super().__init__()
        
        self.config = config
        self.image_cache = ImageCache()
        self.viewers: List[SliceViewer] = []
        
        self._setup_ui()
        self._create_viewers()
    
    def _setup_ui(self):
        """Set up main UI layout"""
        self.setFixedSize(self.config.width, self.config.height + 60)  # Reduce extra height, from 100 to 60
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)  # Reduce margin, from 5 to 2
        main_layout.setSpacing(2)  # Reduce spacing, from 5 to 2
        
        # Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # View grid container
        self.grid_widget = QWidget()
        self.grid_widget.setFixedSize(self.config.width, self.config.height)
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(1)  # Reduce grid spacing
        
        main_layout.addWidget(self.grid_widget)
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel"""
        # Get system palette for adaptive colors
        palette = QApplication.palette()
        text_color = palette.color(QPalette.WindowText).name()
        
        panel = QGroupBox("Visualization")
        panel.setMaximumHeight(50)  # Limit control panel height
        panel.setStyleSheet(f"""
            QGroupBox {{ 
                font-size: 12px; 
                color: {text_color};
            }}
        """)  # Use adaptive text color
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Mask transparency control
        alpha_label = QLabel("Mask transparency:")
        alpha_label.setStyleSheet(f"""
            QLabel {{ 
                font-size: 10px; 
                color: {text_color};
            }}
        """)
        layout.addWidget(alpha_label)
        
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setRange(0.0, 1.0)
        self.alpha_spinbox.setSingleStep(0.1)
        self.alpha_spinbox.setValue(self.config.mask_alpha)
        self.alpha_spinbox.valueChanged.connect(self._on_alpha_changed)
        self.alpha_spinbox.setMaximumWidth(80)  # Limit width
        self.alpha_spinbox.setStyleSheet(f"""
            QDoubleSpinBox {{ 
                font-size: 10px; 
                color: {text_color};
            }}
        """)
        layout.addWidget(self.alpha_spinbox)
        
        # Clear cache button
        clear_cache_btn = QPushButton("Clear cache")
        clear_cache_btn.clicked.connect(self._clear_cache)
        clear_cache_btn.setMaximumWidth(80)  # Limit width
        clear_cache_btn.setStyleSheet(f"""
            QPushButton {{ 
                font-size: 10px; 
                color: {text_color};
            }}
        """)
        # layout.addWidget(clear_cache_btn)
        
        layout.addStretch()
        
        return panel
    
    def _create_viewers(self):
        """Create all viewers"""
        viewer_width = self.config.width // self.config.col
        viewer_height = self.config.height // self.config.row
        
        # Initialize viewers list, pre-fill None
        self.viewers = [None] * (self.config.row * self.config.col)
        
        for i in range(self.config.row * self.config.col):
            row = i // self.config.col
            col = i % self.config.col
            
            # Check if there is corresponding viewer configuration
            if i < len(self.config.viewers) and self.config.viewers[i] is not None:
                viewer_config = self.config.viewers[i]
                try:
                    viewer = SliceViewer(
                        viewer_config, 
                        self.image_cache, 
                        viewer_width, 
                        viewer_height,
                        self.config.mask_alpha
                    )
                    
                    self.grid_layout.addWidget(viewer, row, col)
                    self.viewers[i] = viewer
                    
                except Exception as e:
                    print(f"Error creating viewer {i}: {e}")
                    # Create placeholder for error display
                    error_label = QLabel(f"Error loading viewer {i}:\n{str(e)}")
                    error_label.setAlignment(Qt.AlignCenter)
                    error_label.setStyleSheet("QLabel { color: red; border: 1px solid red; }")
                    error_label.setFixedSize(viewer_width, viewer_height)
                    
                    self.grid_layout.addWidget(error_label, row, col)
            else:
                # Create blank placeholder
                placeholder = QLabel("Blank position")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("QLabel { background-color: #f5f5f5; border: 1px dashed #ccc; color: #999; font-size: 11px; }")
                placeholder.setFixedSize(viewer_width, viewer_height)
                
                self.grid_layout.addWidget(placeholder, row, col)
    
    def _on_alpha_changed(self, alpha: float):
        """Mask transparency change"""
        for viewer in self.viewers:
            if viewer is not None:
                viewer.set_mask_alpha(alpha)
    
    def _clear_cache(self):
        """Clear image cache"""
        self.image_cache.clear_cache()
        print("Image cache cleared")
    
    def update_viewer(self, position: int, viewer_config: ViewerConfig):
        """Dynamic update specified position viewer"""
        if position < 0 or position >= len(self.viewers):
            print(f"Invalid position: {position}")
            return False
        
        try:
            viewer_width = self.config.width // self.config.col
            viewer_height = self.config.height // self.config.row
            row = position // self.config.col
            col = position % self.config.col
            
            # Remove old widget
            old_widget = self.grid_layout.itemAtPosition(row, col)
            if old_widget:
                widget = old_widget.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
            
            # Create new viewer
            new_viewer = SliceViewer(
                viewer_config, 
                self.image_cache, 
                viewer_width, 
                viewer_height,
                self.config.mask_alpha
            )
            
            # Add to grid layout
            self.grid_layout.addWidget(new_viewer, row, col)
            
            # Update viewers list
            self.viewers[position] = new_viewer
            
            print(f"Successfully updated viewer at position {position}")
            return True
            
        except Exception as e:
            print(f"Error updating viewer at position {position}: {e}")
            return False
    
    def get_viewer_count(self) -> int:
        """Get total viewer count"""
        return len(self.viewers)
    
    def get_active_viewers(self) -> List[SliceViewer]:
        """Get all active viewers"""
        return [viewer for viewer in self.viewers if viewer is not None]


def create_multiviewer(config: MultiviewerConfig) -> MultiViewer:
    """Factory function: Create multi-view component"""
    return MultiViewer(config)


# Example usage code
def main():
    """Example main function"""
    app = QApplication(sys.argv)
    
    # Example configuration (set some viewers to empty)
    sample_viewers = [
        ViewerConfig(
            image_path="/mnt/tmp/nii/ZHJ_1_0000.nii.gz",
            mask_path="/mnt/tmp/nii_seg/ZHJ_1.nii.gz",
            direction="axial",
            title="Axial View 1"
        ),
        ViewerConfig(
            image_path="/mnt/tmp/nii/ZHJ_1_0000.nii.gz",
            mask_path="/mnt/tmp/nii_seg/ZHJ_1.nii.gz",
            direction="sagittal",
            title="Coronal View"
        ),
        ViewerConfig(
            image_path="/mnt/tmp/nii/ZHJ_1_0000.nii.gz",
            mask_path="/mnt/tmp/nii_seg/ZHJ_1.nii.gz",
            direction="coronal",
            title="Sagittal View"
        ),
        None  # Second row second column initially empty
    ]
    
    config = MultiviewerConfig(
        row=2,
        col=2,
        width=800,
        height=600,
        viewers=sample_viewers,
        mask_alpha=0.5
    )
    
    try:
        multiviewer = create_multiviewer(config)
        multiviewer.show()
        
        # Create 5 second timer for delayed update of second row second column viewer
        def update_bottom_right_viewer():
            """Update second row second column viewer after 5 seconds"""
            new_config = ViewerConfig(
                image_path="/mnt/tmp/nii_seg/ZHJ_1.nii.gz",
                mask_path="/mnt/tmp/nii_seg/ZHJ_1.nii.gz",
                direction="axial",
                title="Delayed Axial View"
            )
            # Position 3 = second row second column (row=1, col=1 → 1*2+1=3)
            success = multiviewer.update_viewer(3, new_config)
            if success:
                print("Successfully updated second row second column viewer!")
            else:
                print("Failed to update viewer!")
        
        # Set 5 second timer
        timer = QTimer()
        timer.timeout.connect(update_bottom_right_viewer)
        timer.setSingleShot(True)  # Execute only once
        timer.start(5000)  # Execute after 5 seconds
        
        print("Program started, will update second row second column image in 5 seconds...")
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error creating multiviewer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
