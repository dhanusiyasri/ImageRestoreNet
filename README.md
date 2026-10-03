# SemiRestoreNet

### AI-Based Image Restoration and Inspection using RRDB

SemiRestoreNet is an **AI-powered image processing and restoration system** developed to restore degraded images and improve their visual quality for downstream inspection and analysis.

The project integrates an **RRDB-based deep learning restoration model** with a backend processing layer and an interactive frontend, providing an end-to-end workflow from image input to restoration and inspection.

---

## 📌 Project Overview

Image degradation caused by noise, blur, compression, low resolution, and other distortions can significantly affect the quality of visual inspection and computer vision tasks.

**SemiRestoreNet** addresses this problem by applying deep-learning-based image restoration to degraded input images.

The system combines:

* **RRDB-based image restoration**
* **WAF-RRDB-LITE restoration approach**
* Image processing and enhancement
* Backend-based model inference
* Web-based frontend
* Image inspection and result visualization
* Inspection report generation

The project demonstrates the integration of a deep-learning image restoration model into a complete application workflow.

---

## 🎯 Objectives

The main objectives of SemiRestoreNet are to:

* Restore degraded images using deep learning.
* Improve image quality while preserving important visual details.
* Develop an efficient image restoration pipeline.
* Integrate the trained restoration model into an application.
* Provide an interactive interface for image processing.
* Support image inspection after restoration.
* Generate structured inspection results.
* Demonstrate an end-to-end AI-based image processing system.

---

## ✨ Key Features

### 🧠 AI-Based Image Restoration

SemiRestoreNet uses an **RRDB (Residual-in-Residual Dense Block)** based deep-learning architecture for image restoration.

The model learns to reconstruct useful visual information from degraded images.

### ⚡ Lightweight Restoration

The project incorporates the **WAF-RRDB-LITE** restoration approach to support efficient image restoration while maintaining useful reconstruction quality.

### 🖼️ Image Processing

The system processes input images and produces restored/enhanced outputs suitable for visual inspection and further analysis.

### 🔍 Image Inspection

Restored images can be examined through the integrated inspection workflow, allowing users to compare and assess the processed results.

### 🌐 Web Application

A frontend interface provides an accessible way to interact with the image restoration system without directly running the underlying model.

### 🔗 Backend Integration

The backend connects the frontend application with the image-processing and AI inference components.

### 📊 Inspection Reporting

Inspection-related results are stored in a structured format and can be used for analysis and evaluation.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │    Input Image       │
                    │  Degraded / Noisy    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Frontend        │
                    │   Image Upload/UI    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Backend        │
                    │   Request Handling   │
                    │   Image Processing   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  SemiRestoreNet      │
                    │   RRDB / WAF-RRDB    │
                    │      Restoration     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Restored Image     │
                    │ Enhanced Visual      │
                    │ Information          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Image Inspection   │
                    │    & Evaluation      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Inspection Report /  │
                    │      Results         │
                    └──────────────────────┘
```

---

# 🔄 Processing Workflow

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Feature Extraction
     │
     ▼
RRDB-Based Restoration
     │
     ▼
Image Reconstruction
     │
     ▼
Restored Image
     │
     ▼
Inspection / Analysis
     │
     ▼
Result & Report
```

---

# 🧠 RRDB-Based Restoration

The core restoration component is based on **Residual-in-Residual Dense Blocks (RRDB)**.

RRDB combines:

* Dense feature connections
* Residual learning
* Multi-level feature propagation
* Deep feature representation

This architecture is well suited for image restoration because it enables the network to learn complex mappings between degraded and high-quality images.

### Conceptual Pipeline

```text
Degraded Image
      │
      ▼
Feature Extraction
      │
      ▼
Dense Feature Learning
      │
      ▼
Residual-in-Residual Blocks
      │
      ▼
Feature Reconstruction
      │
      ▼
Restored Image
```

---

# ⚡ WAF-RRDB-LITE

SemiRestoreNet also incorporates the **WAF-RRDB-LITE** restoration approach for efficient image restoration.

The lightweight architecture focuses on reducing unnecessary computational complexity while retaining the important restoration capability of the RRDB-based approach.

This makes the restoration pipeline more suitable for practical application scenarios where computational efficiency is important.

---

# 📂 Project Structure

```text
SemiRestoreNet/
│
├── RRDB_modal/
│   ├── Restoration model
│   ├── Model-related files
│   └── Image restoration components
│
├── backend/
│   ├── Backend application
│   ├── Image processing logic
│   └── Model integration
│
├── frontend/
│   ├── User interface
│   ├── Image upload
│   └── Result visualization
│
├── inspection_report_WAF-RRDB-LITE.csv
│   └── Inspection / evaluation results
│
├── demo_video.mp4
│   └── Project demonstration
│
└── README.md
```

---

# 🛠️ Technology Stack

## Artificial Intelligence & Image Processing

* Python
* Deep Learning
* RRDB
* WAF-RRDB-LITE
* Image Processing
* Computer Vision

## Backend

* Python
* Backend API
* AI Model Integration
* Image Processing Pipeline

## Frontend

* Web Technologies
* Interactive User Interface
* Image Upload
* Result Visualization

## Data & Evaluation

* CSV-based inspection reports
* Image restoration evaluation
* Inspection result analysis

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/dhanusiyasri/SemiRestoreNet.git
cd SemiRestoreNet
```

---

## 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the backend using the project's configured entry point.

---

## 3. Frontend Setup

Open a new terminal and navigate to:

```bash
cd frontend
```

Install the frontend dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed by the development server.

---

# 🔄 Application Flow

The completed application follows this workflow:

```text
        User
         │
         ▼
   Upload Image
         │
         ▼
     Frontend
         │
         ▼
      Backend
         │
         ▼
  AI Restoration
         │
         ▼
  RRDB / WAF-RRDB
         │
         ▼
 Restored Image
         │
         ▼
 Image Inspection
         │
         ▼
   Results / Report
```

---

# 📊 Inspection & Evaluation

The repository includes an inspection report:

```text
inspection_report_WAF-RRDB-LITE.csv
```

The report can be used to analyze the performance of the restoration workflow and inspect the generated results.

Image restoration quality can be evaluated using metrics such as:

* **PSNR** — Peak Signal-to-Noise Ratio
* **SSIM** — Structural Similarity Index
* **LPIPS** — Learned Perceptual Image Patch Similarity

These metrics provide different perspectives on reconstruction quality, including pixel-level similarity, structural preservation, and perceptual similarity.

---

# 🎥 Demonstration

A demonstration video of the implemented system is included in the repository:

```text
demo_video.mp4
```

The demonstration showcases the image restoration and inspection workflow.

---

# 💡 Applications

The techniques implemented in SemiRestoreNet can be applied to:

* Image restoration
* Low-quality image enhancement
* Computer vision preprocessing
* Industrial image inspection
* Automated visual inspection
* Image quality improvement
* Defect analysis
* Digital image processing
* AI-assisted inspection systems

---

# 🔬 Project Highlights

### Deep Learning

Implemented an RRDB-based deep-learning approach for image restoration.

### Image Processing

Developed an end-to-end workflow for processing degraded images and generating restored outputs.

### Full-Stack AI Integration

Integrated the AI model with a backend and frontend instead of limiting the project to a standalone model.

### Lightweight Restoration

Explored the WAF-RRDB-LITE approach for efficient image restoration.

### Inspection Pipeline

Integrated image restoration with inspection and structured result generation.

---

# 📈 Future Enhancements

The current implementation provides the foundation for further improvements such as:

* Real-time image restoration
* GPU-optimized inference
* Batch image restoration
* Advanced image-quality assessment
* Additional restoration architectures
* Automated defect detection
* Model benchmarking
* Explainable AI for inspection
* Cloud deployment
* REST API documentation

---

## ⭐ Project Summary

**SemiRestoreNet** is a completed AI-based image processing project that combines **RRDB-based deep-learning image restoration, lightweight restoration, backend integration, frontend interaction, and image inspection** into a single application.

```text
              SemiRestoreNet
                    │
        ┌───────────┴───────────┐
        │                       │
  AI Restoration          Web Application
        │                       │
   RRDB / WAF-RRDB          Frontend
        │                       │
        └───────────┬───────────┘
                    │
                 Backend
                    │
                    ▼
          Restored Image
                    │
                    ▼
          Image Inspection
                    │
                    ▼
             Final Results
```

**SemiRestoreNet — Deep Learning for Practical Image Restoration and Inspection.**
