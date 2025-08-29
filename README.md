
# FATA-WACV-Implementation: Interactive Test-Time Adaptation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a from-scratch implementation of the concepts presented in the WACV 2025 paper, **"Feature Augmentation based Test-Time Adaptation"** by Younggeol Cho et al. It includes the core logic in PyTorch and an interactive web application built with Streamlit to demonstrate the algorithm in real-time.

---

## 🚀 Live Demo

You can try the interactive application here:

**[\[Link to  Deployed Streamlit App\]](https://maxprogrammer007-fata.streamlit.app/)**

---

## ✨ Key Features

* **Core FATA Logic:** A Python implementation of the feature augmentation and adaptation loop as described in the paper.
* **Interactive Demo:** A Streamlit web app to visually demonstrate the effect of Test-Time Adaptation on a pre-trained ResNet-50 model.
* **Real-time Corruption:** Apply corruptions like Gaussian noise and blur to simulate the "domain shift" problem.
* **Live Adaptation:** Run a real adaptation step on the corrupted image and see the model's prediction and confidence change instantly.

## 🧠 Core Concept Explained

Test-Time Adaptation (TTA) aims to adapt a pre-trained model to a new, unseen data distribution (e.g., noisy or blurry images) without retraining the entire model. The FATA paper proposes a novel way to do this by augmenting the model's *internal features* instead of the input image.

The process can be summarized as follows:



1.  **Input:** A corrupted image is fed into a pre-trained model.
2.  **Feature Extraction:** The model processes the image up to an intermediate layer, producing a feature map `z`.
3.  **Dual Branch:**
    * **Original Branch:** The original feature `z` is passed through the rest of the model to get an initial prediction. If the model is confident (low entropy), this prediction is used as a reliable "pseudo-label".
    * **Augmented Branch:** The FATA Augmenter perturbs the feature `z` to create a new, diverse feature `z'`. This is also passed through the rest of the model.
4.  **Loss Calculation:** A loss is calculated to make the prediction from the *augmented feature* consistent with the *pseudo-label* from the original feature.
5.  **Adaptation:** The model's parameters (specifically, the BatchNorm layers) are updated using this loss, making the model more robust to the corruption.

## 📂 Project Structure

```

fata-project/
|
|-- 📁 fata/               \# Core FATA logic
|   |-- augmenter.py     \# The FATA\_Augmenter class
|   |-- model.py         \# Model setup and hooks
|   |-- adapt.py         \# The adaptation loop
|
|-- 📁 utils/              \# Utility functions
|   |-- corruptions.py   \# Image corruption functions
|
|-- 📜 main\_cli.py        \# Command-line script for testing
|-- 📜 app.py              \# The Streamlit application
|-- 📜 requirements.txt   \# Project dependencies
|-- 📜 README.md           \# You are here\!

````

## 🛠️ Setup and Installation

Follow these steps to run the project locally.

**1. Clone the repository:**

```bash
git clone https://github.com/maxprogrammer007/FATA-WACV-Implementation
cd fata-project
````

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run

### Command-Line Interface (for testing)

To test the core adaptation logic on a sample image, run the CLI script:

```bash
python main_cli.py
```

This will download a sample image, corrupt it, run one adaptation step, and print the model's predictions before and after.

### Streamlit Web Application

To launch the interactive demo:

```bash
streamlit run app.py
```

Your browser will open with the application running. You can then upload an image, apply corruptions, and click the "Adapt" button.

### Screenshot

-----

## 📜 Citation

This project is an implementation based on the following paper. Please cite the original authors if you use their work.

```
@inproceedings{cho2025fata,
  title={Feature Augmentation based Test-Time Adaptation},
  author={Cho, Younggeol and Kim, Youngrae and Hong, Seunghoon and Lee, Dongman and Yoon, Junho},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

A special thank you to the authors of the FATA paper for their excellent research and for making their work publicly available.

