"""
SuperGlue Result Viewer
-----------------------
簡單的 PyQt5 UI：
  - 「開啟資料夾 / 圖片」按鈕：選擇包含結果 PNG 的資料夾，或直接選一張結果圖
  - 「顯示結果」按鈕：將選取的圖片渲染到畫面上
  - 上一張 / 下一張：瀏覽資料夾內多張結果
"""

import sys
import os
from pathlib import Path

# 確保 PyQt5 使用本虛擬環境的 Qt plugins，避免 Anaconda 環境干擾
_qt_plugin_path = str(Path(__file__).parent / '.venv' / 'lib' / 'site-packages' / 'PyQt5' / 'Qt5' / 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = _qt_plugin_path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


class SuperGlueViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SuperGlue Result Viewer')
        self.resize(1200, 700)

        self.image_paths = []   # 目前資料夾內所有圖片路徑
        self.current_idx = 0    # 目前顯示的索引

        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI 建構                                                             #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        # ── 頂部工具列 ──────────────────────────────────────────────── #
        toolbar = QHBoxLayout()

        self.btn_open_folder = QPushButton('📂  開啟資料夾')
        self.btn_open_folder.setFixedHeight(40)
        self.btn_open_folder.setFont(QFont('Arial', 10))
        self.btn_open_folder.clicked.connect(self._on_open_folder)

        self.btn_open_image = QPushButton('🖼  開啟圖片')
        self.btn_open_image.setFixedHeight(40)
        self.btn_open_image.setFont(QFont('Arial', 10))
        self.btn_open_image.clicked.connect(self._on_open_image)

        self.lbl_path = QLabel('尚未選取任何資料夾或圖片')
        self.lbl_path.setFont(QFont('Arial', 9))
        self.lbl_path.setStyleSheet('color: #555;')
        self.lbl_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        toolbar.addWidget(self.btn_open_folder)
        toolbar.addWidget(self.btn_open_image)
        toolbar.addWidget(self.lbl_path)
        root_layout.addLayout(toolbar)

        # ── 圖片顯示區 ──────────────────────────────────────────────── #
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet('background: #1e1e1e;')

        self.img_label = QLabel('請先選取資料夾或圖片，再按「顯示結果」')
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFont(QFont('Arial', 12))
        self.img_label.setStyleSheet('color: #888;')
        self.scroll.setWidget(self.img_label)
        root_layout.addWidget(self.scroll, stretch=1)

        # ── 底部導航列 ──────────────────────────────────────────────── #
        nav = QHBoxLayout()

        self.btn_prev = QPushButton('◀  上一張')
        self.btn_prev.setFixedHeight(36)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._on_prev)

        self.lbl_counter = QLabel('')
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        self.lbl_counter.setFont(QFont('Arial', 10))
        self.lbl_counter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_next = QPushButton('下一張  ▶')
        self.btn_next.setFixedHeight(36)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_counter)
        nav.addWidget(self.btn_next)
        root_layout.addLayout(nav)

    # ------------------------------------------------------------------ #
    #  事件處理                                                            #
    # ------------------------------------------------------------------ #
    def _on_open_folder(self):
        """選擇資料夾，讀入全部圖片並直接顯示第一張"""
        folder = QFileDialog.getExistingDirectory(
            self, '選擇結果資料夾',
            str(Path(__file__).parent / 'dump_bullpen_matches')
        )
        if not folder:
            return
        paths = sorted(
            [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: (len(p.stem), p.stem)
        )
        if not paths:
            QMessageBox.warning(self, '警告', '此資料夾內沒有找到圖片檔案！')
            return
        self.image_paths = paths
        self.current_idx = 0
        self.lbl_path.setText(f'資料夾：{folder}  （共 {len(paths)} 張）')
        self._render(self.current_idx)
        self._update_nav()

    def _on_open_image(self):
        """選擇單張圖片並直接顯示"""
        file, _ = QFileDialog.getOpenFileName(
            self, '選擇圖片',
            str(Path(__file__).parent / 'dump_bullpen_matches'),
            'Images (*.png *.jpg *.jpeg *.bmp)'
        )
        if not file:
            return
        self.image_paths = [Path(file)]
        self.current_idx = 0
        self.lbl_path.setText(f'圖片：{file}')
        self._render(self.current_idx)
        self._update_nav()

    def _on_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._render(self.current_idx)
            self._update_nav()

    def _on_next(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self._render(self.current_idx)
            self._update_nav()

    # ------------------------------------------------------------------ #
    #  輔助方法                                                            #
    # ------------------------------------------------------------------ #
    def _render(self, idx: int):
        path = self.image_paths[idx]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.img_label.setText(f'無法讀取圖片：{path}')
            return

        # 等比例縮放，適應視窗大小
        available = self.scroll.size()
        scaled = pixmap.scaled(
            available.width() - 4,
            available.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.img_label.setPixmap(scaled)
        self.img_label.resize(scaled.size())
        self.setWindowTitle(f'SuperGlue Result Viewer  —  {path.name}')

    def _update_nav(self):
        n = len(self.image_paths)
        self.lbl_counter.setText(f'{self.current_idx + 1} / {n}  ·  {self.image_paths[self.current_idx].name}')
        self.btn_prev.setEnabled(self.current_idx > 0)
        self.btn_next.setEnabled(self.current_idx < n - 1)

    def resizeEvent(self, event):
        """視窗縮放時重新渲染圖片，保持適當大小"""
        super().resizeEvent(event)
        if self.image_paths and self.img_label.pixmap():
            self._render(self.current_idx)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = SuperGlueViewer()
    win.show()
    sys.exit(app.exec_())
