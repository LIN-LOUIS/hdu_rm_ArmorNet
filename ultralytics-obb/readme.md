.
├── main.py                        # 主入口脚本（未来统一运行入口）
├── readme.md                      # 项目介绍文档（可记录说明与使用命令）
├── requirements.core.txt          # 依赖列表（torch, opencv-python, ultralytics, openvino 等）
└── ultralytics-obb/               # YOLO11-OBB 源码包根目录
    ├── docs/                      # 框架使用文档（一般是开发者文档）
    │   └── README.md
    ├── examples/                  # 各种部署与可视化示例
    │   ├── README.md
    │   ├── YOLO11-Triton-CPP/     # YOLO11 + Triton C++ 部署示例
    │   └── YOLO-Interactive-Tracking-UI/  # 交互式追踪界面（Streamlit/Gradio）
    ├── pyproject.toml             # 打包与依赖声明（等价 setup.py）
    ├── readme.md                  # 官方 readme
    ├── yolo11n.pt                 # YOLO11 AABB 检测模型权重
    ├── yolo11n-obb.pt             # YOLO11 旋转框（OBB）模型权重
    └── ultralytics/               # YOLO11 核心代码（主要工作区）
        ├── assets/                # 示例图片（bus.jpg / zidane.jpg）
        ├── cfg/                   # 各种配置文件（模型结构 / 数据集 / 默认参数）
        │   ├── default.yaml       # 默认推理/训练参数
        │   ├── datasets/          # 数据集配置文件（COCO / DOTA / VisDrone 等）
        │   └── models/11/         # YOLO11 结构定义
        │       ├── yolo11.yaml           # AABB 检测结构
        │       ├── yolo11-obb.yaml       # 旋转框检测结构（你当前主模型）
        │       ├── yolo11-seg.yaml       # 分割结构
        │       ├── yolo11-pose.yaml      # 姿态结构
        │       ├── yolo11-cls.yaml       # 分类结构
        │       ├── yolo11-cls-resnet18.yaml
        │       └── yoloe-11.yaml         # YOLOE 扩展结构
        ├── data/                  # 数据加载 & 训练时增强逻辑
        │   ├── dataset.py         # 数据集类定义（图片加载、标注解析）
        │   ├── augment.py         # 图像增强（mosaic / affine / flip 等）
        │   ├── split_dota.py      # DOTA 数据集切片脚本
        │   └── loaders.py         # 训练 / 推理 数据加载器
        ├── engine/                # 核心引擎：训练 / 验证 / 推理 / 导出
        │   ├── model.py           # YOLO() 封装类（train/val/predict/export 入口）
        │   ├── predictor.py       # 推理流程定义（含 preprocess/postprocess）
        │   ├── trainer.py         # 训练逻辑
        │   ├── validator.py       # 验证逻辑
        │   ├── exporter.py        # 模型导出（ONNX / TensorRT / OpenVINO）
        │   ├── results.py         # 推理结果封装类（boxes / masks / 绘图）
        │   └── tuner.py           # 超参调优模块
        ├── hub/                   # Ultralytics Hub 连接模块（可忽略）
        ├── models/                # 按任务拆分的模型逻辑
        │   ├── yolo/              # YOLO 系列任务主模块
        │   │   ├── detect/        # 常规AABB检测任务
        │   │   ├── obb/           # 旋转目标检测任务（重点目录）
        │   │   │   ├── train.py   # OBB 训练逻辑
        │   │   │   ├── predict.py # OBB 推理逻辑（待扩展）
        │   │   │   └── val.py     # OBB 验证逻辑
        │   │   ├── classify/      # 图像分类任务
        │   │   ├── segment/       # 分割任务
        │   │   ├── pose/          # 姿态估计任务
        │   │   ├── yoloe/         # YOLOE 变体
        │   │   └── world/         # 特殊训练世界场景
        │   ├── rtdetr/            # RT-DETR Transformer 检测器
        │   ├── sam/               # Segment Anything 模型
        │   ├── nas/               # Neural Architecture Search 模块
        │   ├── fastsam/           # FastSAM 快速分割
        │   └── utils/             # 任务通用工具
        ├── nn/                    # 网络结构组件（卷积层 / Head / Transformer）
        │   ├── modules/           # 结构层（Conv / C2f / SPPF / Head）
        │   ├── tasks.py           # 各任务模型构建器
        │   ├── autobackend.py     # 自动加载后端（ONNX / OpenVINO / TRT）
        │   └── text_model.py      # 文本模型支持（部分扩展）
        ├── solutions/             # 应用案例（人数计数 / 安防告警等）
        ├── trackers/              # 跟踪算法（ByteTrack / BoT-SORT）
        └── utils/                 # 工具库
            ├── nms.py             # 非极大值抑制 (NMS) & 旋转框NMS
            ├── plotting.py        # 绘制检测结果
            ├── loss.py / metrics.py  # 损失函数 / 评价指标
            ├── export/            # 导出工具
            ├── logger.py / tqdm.py   # 日志与进度条
            └── torch_utils.py     # Torch相关工具（自动设备检测等）
| 模块                     | 目标                        | 具体操作                                                                                                           |
| ---------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **1. ROI导出修改**       | 从 YOLO11-OBB 检测结果中提取装甲板区域 | 修改 `ultralytics/engine/predictor.py` → 在 `postprocess()` 或 `write_results()` 中新增 ROI 提取（保存为 `self.armor_rois`） |
| **2. 数字分类模型新增**      | 识别 ROI 区域中的数字             | 新建 `ultralytics/models/digit_classifier.py`：轻量 CNN 模型 or OCR 调用                                                |
| **3. 并行推理脚本新增**      | YOLO + 数字识别两路同时运行         | 新建 `ultralytics/models/yolo/obb/predict_dual.py`，用 `ThreadPoolExecutor` 实现并行推理                                 |
| **4. 主入口统一化**        | 提供统一运行脚本                  | 修改 `main.py`，选择模式（检测 / 导出 / 推理），加载上述模块                                                                         |
| **5. OpenVINO 导出整合** | 导出 .xml / .bin 格式模型       | 直接调用 `yolo export model=yolo11n-obb.pt format=openvino`                                                        |
| **6. 文档与依赖**         | 方便部署                      | 在 `readme.md` 补充运行命令；在 `requirements.core.txt` 加入 `openvino`, `opencv-python`, `torch`, `ultralytics`          |

## 修改1
    | 修改点                              | 操作                     | 作用              |
    | -------------------------------- | ---------------------- | --------------- |
    | ① 在 `__init__()` 中新增属性           | `self.armor_rois = []` | 存储每次推理后的 ROI 区域 |
    | ② 在 `postprocess()` 中新增 ROI 提取逻辑 | 检测结果过滤 & ROI 裁剪        | 提取每帧的装甲板区域      |

