# Heart Disease Prediction Project

### Project Overview
This project uses machine learning to predict the likelihood of heart disease based on a set of health-related features.

### File Structure
- `data/`: Contains the raw dataset.
- `models/`: Stores the trained machine learning model (`final_model.pkl`).
- `notebooks/`: Includes the Jupyter notebooks for each step of the analysis.
- `ui/`: Contains the source code for the Streamlit web application.
- `utils/`: Contains utility scripts, such as `data_prep.py` for data preprocessing.
- `requirements.txt`: Lists all required Python libraries.

### How to Run the App
1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd Heart_Disease_Project
    ```
2.  **Create and activate a Python environment:**
    ```bash
    conda create -n heart_disease_env python=3.9
    conda activate heart_disease_env
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit application:**
    ```bash
    streamlit run ui/app.py
    ```