Here is the updated `README.md` with the Hugging Face links included in the **Model Architecture** section, keeping everything else exactly as you requested.

-----

# Precisely (Requirement Bot) 🤖

**Precisely** is an intelligent, AI-powered system designed to automate and guide the software requirement gathering process. It combines a Python-based AI core for deep language analysis with a Node.js/Express backend and a web dashboard to manage projects.

The system uses Natural Language Processing (NLP) and Machine Learning (ML) to validate requirements against IEEE 830 standards, detect ambiguity, and classify requirements into functional and non-functional categories.

## 📂 Project Structure

The project is organized into modular components for the AI engine, web backend, frontend interface, and training scripts.

```text
Precisely (Requirement Bot)/
├── backend Express/          # Node.js/Express Web Backend
│   ├── controllers/          # Authentication & Logic Controllers
│   ├── db/                   # Database Connection Logic
│   ├── errors/               # Custom Error Handling Classes
│   ├── middlewares/          # Auth & Error Middlewares
│   ├── models/               # Mongoose Data Models (User, etc.)
│   ├── routes/               # API & View Routes
│   ├── app.js                # Backend Entry Point
│   └── package.json          # Node Dependencies
│
├── front end/                # User Interface (Web Dashboard)
│   ├── assets/               # Images and Icons
│   ├── dashboard.html        # Main Dashboard UI
│   ├── login.html            # Login/Register UI
│   ├── *.css                 # Stylesheets (dashboard.css, login.css, etc.)
│   └── *.js                  # Frontend Logic (dashboard.js, login.js)
│
├── src/                      # Python AI Application Core
│   ├── app.py                # Main CLI Application Entry Point
│   ├── backend.py            # Backend Python Logic
│   └── extensions.py         # Helper Extensions
│
├── Scripts/                  # ML Training & Testing Utilities
│   ├── Fine tune Models/     # Scripts to Train/Fine-tune Models
│   └── test models/          # Scripts to Validate Model Performance
│
├── data/                     # Project Data & Datasets
│   ├── datasets/             # CSVs for Intent, Ambiguity, & Attributes
│   └── projects.json         # Local project storage
│
└── models/                   # Directory for ML Models (Download Separately)
```

-----

## 🧠 Model Architecture & Performance

Precisely utilizes a suite of fine-tuned Transformer models to perform its analysis. The following specific architectures are used for each task:

  * **Intent Classification:** `DistilDeBERTa` ([Download Model](https://huggingface.co/AhmadMilo/intent_classifier))
      * Identifies the user's underlying goal or intent when typing a requirement.
  * **Ambiguity Detection:** `T5-Base` ([Download Model](https://huggingface.co/AhmadMilo/Ambiguity_Detector))
      * A sequence-to-sequence model that rewrites vague requirements into clearer alternatives.
  * **FR/NFR Classification:** `DeBERTa-v3-Base` ([Download Model](https://huggingface.co/AhmadMilo/Functional_Non-Functional_Classifier))
      * Distinguishes between Functional Requirements (FR) and Non-Functional Requirements (NFR).
  * **Quality Attribute Prediction:** `DeBERTa-v3-Base` ([Download Model](https://huggingface.co/AhmadMilo/quality_attribute_classifier))
      * Categorizes NFRs into specific quality attributes (e.g., Security, Performance, Usability).

> **⚠️ Note on Accuracy:** The accuracy of these models may not be optimal in all scenarios due to the limited size and variety of the current training datasets. They are intended for demonstration and educational purposes, and performance may vary on complex real-world data.

-----

## 🚀 Key Features

  * **Intelligent CLI:** A guided, interactive command-line interface that helps users define project scope and gather requirements step-by-step.
  * **Real-Time IEEE 830 Validation:** Automatically checks requirements for the SMART criteria (Specific, Measurable, Achievable, Relevant, Testable).
  * **Ambiguity Correction:** Flags vague language (e.g., "fast", "easy") and provides AI-generated suggestions for improvement.
  * **Web Dashboard:** A secure platform built with Express.js to manage user authentication and view project details.

-----

## 🛠️ Installation & Setup

### 1\. Prerequisites

Ensure you have the following installed on your machine:

  * **Python 3.8+**
  * **Node.js 14+** & **npm**
  * **MongoDB** (Local instance or MongoDB Atlas URI)

### 2\. Setting up the Python AI Core (`src/`)

The Python core handles the intelligence and requirement processing.

1.  **Navigate to the project root.**
2.  **Install Python Dependencies:**
    You will need libraries for PyTorch, Transformers, SpaCy, and others.
    ```bash
    pip install torch transformers sentence-transformers spacy pandas
    ```
3.  **Download Language Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```
4.  **Setup ML Models:**
      * Create a folder named `models` in the root directory.
      * Download the pre-trained models from the provided source (links above).
      * Place them inside `models/` so the structure matches:
          * `models/intent_classifier/`
          * `models/ambiguity_detector/`
          * `models/quality_attribute_classifier/`
          * `models/fr_nfr_classifier/`

### 3\. Setting up the Backend (`backend Express/`)

The backend manages user authentication and API requests. Since there is no default config file, you must configure the environment variables manually.

1.  **Navigate to the backend folder:**

    ```bash
    cd "backend Express"
    ```

2.  **Install Node Modules:**

    ```bash
    npm install
    ```

3.  **Configure Environment Variables:**
    Create a file named `.env` inside the `backend Express` folder and add the following configuration keys (you will need to provide your own values):

    ```env
    MONGO_URI=mongodb://localhost:27017/precisely_db
    JWT_SECRET_ACCESS=your_access_token_secret
    JWT_SECRET_REFRESH=your_refresh_token_secret
    EMAIL_SERVICE=gmail
    EMAIL_USER=your_email@gmail.com
    EMAIL_PASS=your_email_password
    ```

4.  **Start the Server:**

    ```bash
    npm start
    ```

    The server will start on port 5000 (default) or as specified in `app.js`.

### 4\. Setting up the Frontend (`front end/`)

The frontend provides the visual interface.

  * The files in the `front end/` directory (e.g., `login.html`, `dashboard.html`) are the UI entry points.
  * Open these files directly in a browser or serve them using a static server (e.g., Live Server).
  * Ensure your **Backend Express** server is running so the frontend can successfully communicate with the API.

-----

## 🖥️ Usage Guide

### Mode 1: Interactive AI CLI

To start the guided requirement gathering session:

1.  Open your terminal in the root directory.
2.  Run the Python app:
    ```bash
    python src/app.py
    ```
3.  Follow the on-screen prompts to:
      * **Define Scope:** Set project boundaries, actors, and constraints.
      * **Add Requirements:** Type requirements in natural language. The AI will validate and classify them instantly.

### Mode 2: Quick Test Mode

If you want to test the ML models without defining a full scope:

1.  Run `python src/app.py`.
2.  Select **Option 2** (Quick Test Mode).
3.  Enter individual sentences to see how the AI classifies and corrects them.

### Training the Models

To fine-tune the models on your own data:

1.  Navigate to `Scripts/Fine tune Models/`.
2.  Open `modelTrainer.py` and update the `CSV_PATH` to point to your new dataset.
3.  Run the training script:
    ```bash
    python "Scripts/Fine tune Models/modelTrainer.py"
    ```

-----

## ⚠️ Disclaimer

**This project is for EDUCATIONAL PURPOSES ONLY.**

It is designed as a learning tool for software engineering. It is not intended for commercial use or production environments.

## 📄 Project Report

View the detailed report of the project here for better understanding:
[View Project Report (PDF)](docs/PROJECT%20REPORT%20FOR%20PRECISELY.pdf)
