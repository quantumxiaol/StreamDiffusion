import os
import sys
import time
import threading
from typing import Optional, Dict, Literal
from queue import Queue

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


class CameraThread(QThread):
    """摄像头捕获线程"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            return
            
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
            else:
                break
            time.sleep(0.01)  # 控制帧率
            
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class StreamDiffusionThread(QThread):
    """StreamDiffusion 处理线程"""
    image_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, 
                 model_id: str,
                 prompt: str,
                 width: int = 512,
                 height: int = 512,
                 lora_dict: Optional[Dict[str, float]] = None,
                 acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
                 device: Optional[Literal["cpu", "cuda", "mps"]] = None):
        super().__init__()
        self.model_id = model_id
        self.prompt = prompt
        self.width = width
        self.height = height
        self.lora_dict = lora_dict
        self.acceleration = acceleration
        self.device = device
        self.running = False
        self.frame_queue = Queue(maxsize=2)  # 限制队列大小，避免内存堆积
        self.stream = None
        
    def run(self):
        # 初始化 StreamDiffusion
        print("正在初始化 StreamDiffusion...")
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=self.model_id,
            lora_dict=self.lora_dict,
            t_index_list=[32, 45],
            frame_buffer_size=1,
            width=self.width,
            height=self.height,
            warmup=10,
            acceleration=self.acceleration,
            do_add_noise=False,
            mode="img2img",
            output_type="pil",
            enable_similar_image_filter=True,
            similar_image_filter_threshold=0.98,
            use_denoising_batch=True,
            seed=2,
            device=self.device,
        )
        
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt="low quality, bad quality, blurry, low resolution",
            num_inference_steps=50,
            guidance_scale=1.2,
        )
        
        # Warmup
        dummy_image = Image.new("RGB", (self.width, self.height), (128, 128, 128))
        for _ in range(self.stream.batch_size):
            self.stream(dummy_image)
        
        print("StreamDiffusion 初始化完成")
        self.running = True
        
        while self.running:
            try:
                # 从队列获取帧
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    
                    # 转换为 PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # 处理图像
                    output_image = self.stream(pil_image)
                    
                    # 转换为 numpy array
                    if isinstance(output_image, Image.Image):
                        output_array = np.array(output_image)
                    else:
                        output_array = output_image
                    
                    self.image_ready.emit(output_array)
                else:
                    time.sleep(0.01)  # 避免 CPU 占用过高
            except Exception as e:
                print(f"处理图像时出错: {e}")
                time.sleep(0.1)
    
    def add_frame(self, frame: np.ndarray):
        """添加帧到处理队列"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            # 队列满时，丢弃最旧的帧
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
            except:
                pass
    
    def update_prompt(self, prompt: str):
        """更新提示词"""
        self.prompt = prompt
        if self.stream:
            self.stream.update_prompt(prompt)
    
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时摄像头重绘 - StreamDiffusion")
        self.setGeometry(100, 100, 1200, 700)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("输入提示词，例如: 1girl with brown dog ears, thick frame glasses")
        self.prompt_input.setText("1girl with brown dog ears, thick frame glasses")
        
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("模型ID或路径")
        self.model_input.setText("KBlueLeaf/kohaku-v2.1")
        
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_processing)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(QLabel("提示词:"))
        control_layout.addWidget(self.prompt_input)
        control_layout.addWidget(QLabel("模型:"))
        control_layout.addWidget(self.model_input)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(control_layout)
        
        # 视频显示区域
        video_layout = QHBoxLayout()
        
        # 原视频显示
        self.original_label = QLabel()
        self.original_label.setMinimumSize(512, 512)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setText("原视频")
        self.original_label.setStyleSheet("border: 2px solid gray; background-color: black; color: white;")
        
        # 重绘视频显示
        self.redrawn_label = QLabel()
        self.redrawn_label.setMinimumSize(512, 512)
        self.redrawn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.redrawn_label.setText("重绘视频")
        self.redrawn_label.setStyleSheet("border: 2px solid gray; background-color: black; color: white;")
        
        video_layout.addWidget(self.original_label)
        video_layout.addWidget(self.redrawn_label)
        
        main_layout.addLayout(video_layout)
        
        # 状态信息
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        # 线程
        self.camera_thread = None
        self.stream_thread = None
        
        # 定时器用于更新提示词
        self.prompt_timer = QTimer()
        self.prompt_timer.timeout.connect(self.update_prompt_if_changed)
        self.prompt_timer.start(1000)  # 每秒检查一次
        self.last_prompt = ""
    
    def update_prompt_if_changed(self):
        """检查并更新提示词"""
        current_prompt = self.prompt_input.text()
        if current_prompt != self.last_prompt and self.stream_thread and self.stream_thread.running:
            self.stream_thread.update_prompt(current_prompt)
            self.last_prompt = current_prompt
            self.status_label.setText(f"提示词已更新: {current_prompt}")
    
    def start_processing(self):
        """开始处理"""
        model_id = self.model_input.text()
        prompt = self.prompt_input.text()
        
        if not model_id or not prompt:
            self.status_label.setText("请填写模型ID和提示词")
            return
        
        self.status_label.setText("正在初始化...")
        
        # 启动摄像头线程
        self.camera_thread = CameraThread(camera_index=0)
        self.camera_thread.frame_ready.connect(self.on_camera_frame)
        self.camera_thread.start()
        
        # 启动 StreamDiffusion 线程
        self.stream_thread = StreamDiffusionThread(
            model_id=model_id,
            prompt=prompt,
            width=512,
            height=512,
            acceleration="xformers",
        )
        self.stream_thread.image_ready.connect(self.on_processed_image)
        self.stream_thread.start()
        
        self.last_prompt = prompt
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("运行中...")
    
    def stop_processing(self):
        """停止处理"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread.wait()
            self.stream_thread = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("已停止")
        
        # 清空显示
        self.original_label.setText("原视频")
        self.redrawn_label.setText("重绘视频")
    
    def on_camera_frame(self, frame: np.ndarray):
        """处理摄像头帧"""
        # 显示原视频
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.original_label.setPixmap(scaled_pixmap)
        
        # 发送到处理线程
        if self.stream_thread and self.stream_thread.running:
            # 调整大小以匹配模型输入
            resized_frame = cv2.resize(frame, (512, 512))
            self.stream_thread.add_frame(resized_frame)
    
    def on_processed_image(self, image: np.ndarray):
        """显示处理后的图像"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 确保是 RGB 格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.redrawn_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_processing()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

