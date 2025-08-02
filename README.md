# IITISoC2025-PS9 PolyOCR
## PolyOCR: Multilingual OCR 

A versatile OCR platform that automatically detects, recognizes, and processes text in 9 languages — **English, Hindi, Russian, French, Spanish, Korean, German, Italian and Turkish** — enhanced with translation, audio playback, spell checking,  and summarization.

🌐 Visit the live demo: [polyocr.vercel.app](https://polyocr.vercel.app)

---

### 📸 Example Usage



### ⚙️ Local Setup Guide

> *Instructions for setting up and running PolyOCR locally.*

To run PolyOCR locally, you need to create **two separate virtual environments**:

#### 1. CRAFT Text Detection Environment (Python 3.7)

```bash
# Create and activate virtual environment for CRAFT
py -3.7 -m venv craftenv
craftenv\Scripts\activate  # On Windows
source craftenv/bin/activate  # On macOS/Linux

# Navigate to CRAFT directory and install requirements
cd 'CRAFT Detection Model'  
pip install https://mirrors.aliyun.com/pytorch-wheels/cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl
pip install -r requirements.txt
```

#### 2. Create a Main Virtual Environment (Python 3.12)

```bash
# Create and activate main application environment
python -m venv polyocrenv
polyocrenv\Scripts\activate  # On Windows
source polyocrenv/bin/activate  # On macOS/Linux

# Install main app dependencies
cd easyOCR  
pip install -r requirements.txt
```
```bash
#Run 
python craftservice.py #in craftenv
python app.py #in polyocrenv
```

---

### 🛠️ Model Details

**Text Detection**

> *CRAFT (Character-Region Awareness For Text detection) is a convolutional neural network that predicts character region and affinity scores, enabling precise text localization even in curved or rotated layouts. It achieves state-of-the-art performance on scene text benchmarks. Learn more: [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) | [Original Paper](https://arxiv.org/abs/1904.01941)*

>*For handwritten English text detection, we leverage the EasyOCR implementation of CRAFT, which offers superior performance on handwritten samples than CRAFT. Details: [EasyOCR CRAFT](https://github.com/JaidedAI/EasyOCR/blob/master/trainer/craft/README.md)*

**Text Recognition**

> *Details about your fine-tuned EasyOCR models and TrOCR for handwritten English. (To be added)*

---

### ✨ Additional Features

* **Auto Language Detection**: Region-wise language identification using the `langdetect` library.
* **Spell Check**: Post-recognition correction powered by the Grok API.
* **Translation**: Neural translation to your preferred language.
* **Audio Playback**: Text-to-speech for any extracted text.
* **Summarization**: Condensed summaries of long text passages.

