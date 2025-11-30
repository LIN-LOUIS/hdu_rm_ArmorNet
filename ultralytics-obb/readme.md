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
    | 3 在 `BasePredictor` 中添加 ROI 缓存属性 | 定义了数字分类的大小和角点顺序        | 添加 ROI 缓存属性     |
    | 4 在 `postprocess()` 返回results | 修改了逻辑        | 使网络可以读懂视频和图片     |

## 修改2
    | 修改点                              | 操作                     | 作用              |
    | -------------------------------- | ---------------------- | --------------- |
    | ① 在 digit_classifier.py中新增属性           | 增添cnn代码 | 生成pt |
## 修改3
    | 修改点                              | 操作                     | 作用              |
    | -------------------------------- | ---------------------- | --------------- |
    | ①  新建 `ultralytics/models/yolo/obb/predict_dual.py`，用 `ThreadPoolExecutor` 实现并行推理| 每一帧的 armor_rois 交给 ThreadPoolExecutor 中的 classify_rois 并行跑 | 跑完后把 digits、scores 写回 result，并画到图上保存/显示 |

    | 修改点                              | 操作                     | 作用              |
    | -------------------------------- | ---------------------- | --------------- |
    | ①  修改 `main.py`，选择模式（检测 / 导出 / 推理），加载上述模| detect：普通 YOLO / YOLO-OBB 推理（只检测装甲板）;dual：你现在做的 OBB + 数字识别并行推理;export：导出 OpenVINO（顺便把 Step5 也接上）
    | 修改点                              | 操作                     | 作用              |
    | -------------------------------- | ---------------------- | --------------- |
    | ①  修改 `predictor.py`的write_result中 === 两行对接：将 ROI 丢给数字分类器 ===| 增加了置信度|增强网络的准确率
    | 2 修改 `digit_classifier.py`的classify_rois| 增加了置信度,若未识别数字返回-1|增强网络的准确率


# 先训练数字识别
    /home/lin/Desktop/deep_learning/digits_dataset/
    ├─ 0/
    │   ├─ xxx_1.jpg
    │   ├─ xxx_2.jpg
    ├─ 1/
    ├─ 2/
    ├─ ...
    └─ 9/
    图片随便命名，但必须放在对应数字的文件夹里
    图片是你从 armor_rois 裁下来的装甲板数字区域

   ## 训练 
    bush：
    cd ~/Desktop/deep_learning/ultralytics-obb
    conda activate yolo11
    
    python ultralytics/models/digit_classifier.py \
    --mode train \
    --data /home/lin/Desktop/deep_learning/digits_dataset \
    --weights digit_classifier.pt

    用 digits_dataset 读数据（0–9 文件夹）
    训练 DigitClassifier
    最好的模型保存在当前目录：digit_classifier.pt

   ## 预测
    然后跑一个预测试试看：

    python ultralytics-obb/main.py detect \
    --model yolo11n-obb.pt \
    --source ultralytics-obb/assets/frame_0354.jpg \
    --task obb \
    --conf 0.25 \
    --show
    
    
# 训练obb模型
    ~/Desktop/deep_learning/armor_obb_dataset/
    ├─ images/
    │    ├─ train/
    │    │    ├─ img001.jpg
    │    │    ├─ img002.jpg
    │    └─ val/
    │         ├─ img101.jpg
    │         ├─ img102.jpg
    └─ labels/
        ├─ train/
        │    ├─ img001.txt
        │    ├─ img002.txt
        └─ val/
                ├─ img101.txt
                ├─ img102.txt
   ## 训练 
    bush：
    cd ~/Desktop/deep_learning/ultralytics-obb
    conda activate yolo11

    yolo task=obb mode=train \
    model=yolo11n-obb.pt \
    data=cfg/armor_obb.yaml \
    imgsz=1024 \
    epochs=100 \
    batch=16 \
    device=0

    训练完后，权重在：runs/obb/train*/weights/best.pt

# 装甲板本身很小会怎样？
分两层看：
检测端（YOLO-OBB）
    输入图固定比如 1024×768，如果装甲板在原图里只有 10×10 像素，那对于 YOLO 来说已经接近“极小目标”了；
    检测 head 的下采样 stride（8 / 16 / 32）会让这种小目标的特征非常弱，检测就会先崩。

数字端（Digit CNN）
    如果 YOLO 给你的 bbox 就只有 10×10 像素，那 _preprocess_roi_bgr 会把这 10×10 放到一个 10×10 的 canvas，再 resize 成 64×64；
    数字本身其实被放大了，对 Digit CNN 来说不是坏事；

真正的风险是：小装甲板本身非常模糊、噪声重、被压缩后信息就少。

所以：
    微小装甲板识别难的第一凶手其实是检测端，不是 Digit CNN；
    Digit CNN 反而因为你有“放大 + 居中”的预处理，对小 ROI 并不一定吃亏。

因此准备现行训练，如果效果不好再考虑改进检测端。