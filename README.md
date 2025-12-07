Plastic Detection ML Model
Overview
This project develops a machine learning model for detecting plastic objects in real-time camera feeds. The goal is to enable applications such as waste sorting, environmental monitoring, and recycling automation. Two primary approaches are implemented: fine-tuning a pre-trained object detection model and building a custom classification model from scratch using curated datasets.
Approache:
Fine-Tuning an Object Detection Model

Utilizes a pre-trained model fine-tuned on domain-specific data.
Focuses on bounding box detection for plastics in video streams or camera inputs.
Advantages: Leverages transfer learning for faster convergence and higher accuracy with limited data.


Dataset

Data consists of collected images of plastics (e.g., bottles, bags) and non-plastics (e.g., metal, paper, organic waste).
Sources: Public datasets augmented with custom captures.
Preprocessing includes resizing, normalization, and augmentation techniques like rotation and flipping.

Requirements

Python 3.11+
Libraries: TensorFlow, OpenCV, NumPy, Pandas, Scikit-learn

Installation

Clone the repository: git clone https://github.com/yourusername/plastic-detection-ml.git
Install dependencies: pip install -r requirements.txt

Usage

Training the Fine-Tuned Model: Run python train_finetune.py --data_path /path/to/dataset --epochs 50
Training the Custom Model: Run python train_custom.py --data_path /path/to/dataset --epochs 100
Inference: Use python detect.py --model_path /path/to/model --camera_id 0 for real-time detection from webcam.

Evaluation

Metrics: Accuracy, Precision, Recall, F1-Score, IoU (for detection).
Performance is evaluated on a hold-out test set to ensure generalization.

Future Work

Integrate with edge devices for on-device inference.
Expand the dataset for diverse plastic types and environments.
Explore ensemble methods combining both approaches.

Contributors

Maxwell Tebi - Lead Developer

For questions or contributions, please open an issue or submit a pull request.
