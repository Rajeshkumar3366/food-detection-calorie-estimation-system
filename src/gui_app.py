# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""PyQt5 desktop app for the food detection + calorie estimation pipeline.

Pick an image, run the fine-tuned YOLOv8 detector and the per-class
calorie regressor on it, and view the annotated result, a per-item
breakdown, and the total estimated calories.

Usage:
    python src/gui_app.py
"""

import sys
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from predict import annotate_image, load_models, run_inference

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "runs" / "yolo_food" / "weights" / "best.pt"
DEFAULT_CALORIE_MODEL = ROOT / "models" / "calorie_model.joblib"

APP_TITLE = "Food Detection & Calorie Estimation System"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp)"


def cv2_to_qpixmap(img_bgr) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class InferenceWorker(QThread):
    """Runs detection + calorie scoring off the GUI thread so the window
    doesn't freeze while the model runs."""

    succeeded = pyqtSignal(list, float, int, object)
    failed = pyqtSignal(str)

    def __init__(self, detector, calorie_models, image_path):
        super().__init__()
        self.detector = detector
        self.calorie_models = calorie_models
        self.image_path = image_path

    def run(self):
        try:
            items, total_calories, n_merged = run_inference(
                self.detector, self.calorie_models, self.image_path
            )
            annotated = annotate_image(self.image_path, items)
            self.succeeded.emit(items, total_calories, n_merged, annotated)
        except Exception as exc:  # surface any failure to the GUI thread
            self.failed.emit(str(exc))


class ImagePane(QFrame):
    """One titled, bordered panel that displays an image scaled to fit."""

    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self._pixmap = None

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: 600; padding: 4px;")

        self.image_label = QLabel(f"({title.lower()} will appear here)")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet(
            "background: #fafafa; border: 1px dashed #bbb; color: #999;"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(title_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._rescale()

    def clear(self, placeholder: str):
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(placeholder)

    def _rescale(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1150, 820)

        self.image_path = None
        self.detector = None
        self.calorie_models = None
        self.worker = None

        self._build_ui()
        self._load_models()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)

        heading = QLabel(APP_TITLE)
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("font-size: 20px; font-weight: 700; padding: 6px;")
        root.addWidget(heading)

        # --- controls row ---
        controls = QHBoxLayout()
        self.choose_btn = QPushButton("Choose Image...")
        self.choose_btn.clicked.connect(self.choose_image)
        self.path_label = QLabel("No image selected")
        self.path_label.setStyleSheet("color: #666;")
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_inference_clicked)
        controls.addWidget(self.choose_btn)
        controls.addWidget(self.path_label, stretch=1)
        controls.addWidget(self.run_btn)
        root.addLayout(controls)

        # --- two image panes ---
        panes = QHBoxLayout()
        self.source_pane = ImagePane("Source Image")
        self.result_pane = ImagePane("Detection Result")
        panes.addWidget(self.source_pane)
        panes.addWidget(self.result_pane)
        root.addLayout(panes, stretch=3)

        # --- detected items table ---
        table_label = QLabel("Detected Food Items")
        table_label.setStyleSheet("font-weight: 600;")
        root.addWidget(table_label)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Food Item", "Confidence", "Calories (kcal)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        root.addWidget(self.table, stretch=2)

        # --- total calories readout ---
        total_row = QHBoxLayout()
        total_caption = QLabel("Total Estimated Calories:")
        total_caption.setStyleSheet("font-weight: 600;")
        self.total_value = QLabel("-- kcal")
        self.total_value.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #1a7a3c;"
        )
        total_row.addWidget(total_caption)
        total_row.addWidget(self.total_value)
        total_row.addStretch(1)
        root.addLayout(total_row)

        # --- status line ---
        self.status_label = QLabel("Loading models...")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        root.addWidget(self.status_label)

    def _load_models(self):
        try:
            self.detector, self.calorie_models = load_models(
                str(DEFAULT_WEIGHTS), str(DEFAULT_CALORIE_MODEL)
            )
            self.status_label.setText("Models loaded. Choose an image to begin.")
        except Exception as exc:
            self.status_label.setText("Failed to load models.")
            QMessageBox.critical(
                self,
                "Model load error",
                f"Could not load the detector/calorie models:\n{exc}",
            )

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose an image", "", IMAGE_FILTER)
        if not path:
            return

        self.image_path = path
        self.path_label.setText(path)

        self.source_pane.set_pixmap(QPixmap(path))
        self.result_pane.clear("(run inference to see the result)")
        self.table.setRowCount(0)
        self.total_value.setText("-- kcal")
        self.status_label.setText("Image loaded. Click “Run Inference”.")
        self.run_btn.setEnabled(self.detector is not None)

    def run_inference_clicked(self):
        if not self.image_path or self.detector is None:
            return
        self.choose_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running inference...")

        self.worker = InferenceWorker(self.detector, self.calorie_models, self.image_path)
        self.worker.succeeded.connect(self.on_inference_succeeded)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.start()

    def on_inference_succeeded(self, items, total_calories, n_merged, annotated_bgr):
        self.result_pane.set_pixmap(cv2_to_qpixmap(annotated_bgr))

        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(item["class_name"]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{item['confidence']:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{item['calories']:.0f}"))

        self.total_value.setText(f"{total_calories:.0f} kcal")

        if items:
            status = f"Detected {len(items)} item(s)."
            if n_merged:
                status += f" ({n_merged} overlapping duplicate(s) merged away.)"
        else:
            status = "No food items detected in this image."
        self.status_label.setText(status)

        self.choose_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

    def on_inference_failed(self, message):
        self.choose_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Inference failed.")
        QMessageBox.critical(self, "Inference error", message)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
