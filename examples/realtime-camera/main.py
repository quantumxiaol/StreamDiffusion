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
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

# Demo 的默认配置
base_model = "stabilityai/sd-turbo"
default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"


class CameraThread(QThread):
    """摄像头捕获线程（使用 cv2）"""
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
            try:
                self.cap.release()
            except:
                pass
            self.cap = None


class StreamDiffusionThread(QThread):
    """StreamDiffusion 处理线程（完全按照 demo 的配置）"""
    image_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, 
                 model_id: str,
                 prompt: str,
                 width: int = 512,
                 height: int = 512,
                 device: Optional[torch.device] = None,
                 torch_dtype: torch.dtype = torch.float16,
                 acceleration: Literal["none", "xformers", "tensorrt"] = "none",
                 use_tiny_vae: bool = True):
        super().__init__()
        self.model_id = model_id
        self.prompt = prompt
        self.width = width
        self.height = height
        self.device = device
        self.torch_dtype = torch_dtype
        self.acceleration = acceleration
        self.use_tiny_vae = use_tiny_vae
        self.running = False
        self.frame_queue = Queue(maxsize=2)
        self.stream = None
        
    def run(self):
        # 初始化 StreamDiffusion（完全按照 demo 的配置）
        print("正在初始化 StreamDiffusion...")
        
        # 按照 demo 的方式初始化
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=self.model_id,
            t_index_list=[35, 45],  # demo 使用 [35, 45]
            frame_buffer_size=1,
            width=self.width,
            height=self.height,
            use_lcm_lora=False,  # demo 使用 False
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=self.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",  # demo 使用 "none"
            use_safety_checker=False,
            # enable_similar_image_filter 在 demo 中被注释掉了
            device=self.device,
            dtype=self.torch_dtype,
            use_tiny_vae=self.use_tiny_vae,
        )
        
        self.stream.prepare(
            prompt=self.prompt,
            negative_prompt=default_negative_prompt,  # 使用 demo 的 negative_prompt
            num_inference_steps=50,
            guidance_scale=1.2,
        )
        
        print("StreamDiffusion 初始化完成")
        self.running = True
        
        while self.running:
            try:
                # 从队列获取帧
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    
                    # 转换为 PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # 使用 demo 的方式：preprocess_image + stream(image=..., prompt=...)
                    image_tensor = self.stream.preprocess_image(pil_image)
                    output_image = self.stream(image=image_tensor, prompt=self.prompt)
                    
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
                import traceback
                traceback.print_exc()
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
        """更新提示词（demo 方式：prompt 在每次 stream() 调用时传入）"""
        self.prompt = prompt
    
    def stop(self):
        self.running = False
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break


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
        self.prompt_input.setPlaceholderText("输入提示词")
        self.prompt_input.setText(default_prompt)  # 使用 demo 的默认 prompt
        
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("模型ID或路径")
        self.model_input.setText(base_model)  # 使用 demo 的默认模型
        
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
        """检查并更新提示词（demo 方式：prompt 在每次 stream() 调用时传入）"""
        current_prompt = self.prompt_input.text()
        if current_prompt != self.last_prompt and self.stream_thread and self.stream_thread.running:
            # 只需要更新 self.prompt，下次 stream() 调用时会自动使用新的 prompt
            self.stream_thread.update_prompt(current_prompt)
            self.last_prompt = current_prompt
            self.status_label.setText("提示词已更新")
    
    def start_processing(self):
        """开始处理"""
        model_id = self.model_input.text()
        prompt = self.prompt_input.text()
        
        if not model_id or not prompt:
            self.status_label.setText("请填写模型ID和提示词")
            return
        
        self.status_label.setText("正在初始化...")
        
        # 确定设备（按照 demo 的方式）
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        torch_dtype = torch.float16
        
        # 启动摄像头线程
        self.camera_thread = CameraThread(camera_index=0)
        self.camera_thread.frame_ready.connect(self.on_camera_frame)
        self.camera_thread.start()
        
        # 启动 StreamDiffusion 线程（完全按照 demo 的配置）
        self.stream_thread = StreamDiffusionThread(
            model_id=model_id,
            prompt=prompt,
            width=512,
            height=512,
            device=device,
            torch_dtype=torch_dtype,
            acceleration="none",  # 对于 MPS，使用 "none" 更安全
            use_tiny_vae=True,
        )
        self.stream_thread.image_ready.connect(self.on_processed_image)
        self.stream_thread.start()
        
        self.last_prompt = prompt
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("运行中...")
    
    def stop_processing(self):
        """停止处理"""
        # 先断开信号连接，避免在清理过程中触发回调
        if self.camera_thread:
            try:
                self.camera_thread.frame_ready.disconnect()
            except:
                pass
            self.camera_thread.stop()
            if self.camera_thread.isRunning():
                self.camera_thread.wait(3000)  # 等待最多3秒
                if self.camera_thread.isRunning():
                    self.camera_thread.terminate()  # 强制终止
                    self.camera_thread.wait()
            self.camera_thread = None
        
        if self.stream_thread:
            try:
                self.stream_thread.image_ready.disconnect()
            except:
                pass
            self.stream_thread.stop()
            if self.stream_thread.isRunning():
                self.stream_thread.wait(3000)  # 等待最多3秒
                if self.stream_thread.isRunning():
                    self.stream_thread.terminate()  # 强制终止
                    self.stream_thread.wait()
            # 清理 StreamDiffusion 资源
            if hasattr(self.stream_thread, 'stream') and self.stream_thread.stream:
                del self.stream_thread.stream
            self.stream_thread = None
        
        # 清空显示（在清理线程后）
        self.original_label.clear()
        self.original_label.setText("原视频")
        self.redrawn_label.clear()
        self.redrawn_label.setText("重绘视频")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("已停止")
    
    def on_camera_frame(self, frame: np.ndarray):
        """处理摄像头帧（使用 PyQt6 显示）"""
        # 显示原视频 - 复制数据以避免内存问题
        frame_copy = frame.copy()
        h, w, ch = frame_copy.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.original_label.setPixmap(scaled_pixmap)
        
        # 发送到处理线程 - 保持比例进行缩放
        if self.stream_thread and self.stream_thread.running:
            # 保持比例缩放，然后填充到 512x512
            target_size = 512
            frame_h, frame_w = frame_copy.shape[:2]
            
            # 计算缩放比例，保持宽高比
            scale = min(target_size / frame_w, target_size / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            
            # 缩放图像
            resized = cv2.resize(frame_copy, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建 512x512 的黑色背景
            padded_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # 计算居中位置
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            
            # 将缩放后的图像放到中心
            padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            self.stream_thread.add_frame(padded_frame)
    
    def on_processed_image(self, image: np.ndarray):
        """显示处理后的图像（使用 PyQt6 显示）"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 确保是 RGB 格式并复制数据以避免内存问题
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_copy = image.copy()
            h, w, ch = image_copy.shape
            bytes_per_line = ch * w
            q_image = QImage(image_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.redrawn_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有处理
        self.stop_processing()
        
        # 清理定时器
        if hasattr(self, 'prompt_timer'):
            self.prompt_timer.stop()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
