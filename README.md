# UK Road Safety: Collision Severity Prediction (2020-2024)

## 📌 Project Overview
This project aims to predict the severity of road accidents in the United Kingdom using official data from the Department for Transport (DfT). The core of the work involves a complex data integration process and the development of a Machine Learning pipeline using **XGBoost** to classify accidents based on their severity.

The target variable, originally three-classed, was re-engineered into a binary classification problem (**Slight** vs. **Serious/Fatal**) to handle the extreme scarcity of fatal cases and improve model robustness.

## 📊 Data Source
The analysis uses the "Latest 5 years" (2020-2024) of the **Road Safety Open Data** provided by the UK Government.
*   **Source:** [UK Gov Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)
*   **Datasets Used:** 
    *   `Collisions`: General accident information (location, weather, etc.).
    *   `Vehicles`: Details on vehicles involved (joined via `collision_index`).
    *   `Casualties`: Details on people injured (joined via `collision_index`).

## 🛠️ Data Science Pipeline
The entire workflow is documented in `codice.ipynb`:

1.  **Data Merging & Relational Handling:** Integrated the `Vehicles` and `Casualties` data (one-to-many) by creating specific **flag variables** at the collision level.
2.  **Target Re-engineering:** Merged "Serious" and "Fatal" classes into a single "Serious" category to mitigate extreme class imbalance.
3.  **Preprocessing:** Cleaned missing values, handled categorical encoding, and performed feature engineering on environmental factors.
4.  **Modeling:** Trained an **XGBoost** classifier, specifically tuned to handle the remaining class imbalance (Slight accidents are still the majority).
5.  **Evaluation:** Focused on metrics beyond simple accuracy, such as Precision-Recall and F1-Score, to ensure the model effectively identifies high-severity accidents.

## 🚀 Repository Structure
*   `codice.ipynb`: The complete notebook from data ingestion to model evaluation.
*   `xgboost_pipeline_2024.pickle`: The final trained model, saved as a production-ready pipeline.
*   `dashboard.py`: An interactive dashboard to visualize results and test the model.
*   `requirements.txt`: List of Python libraries needed to run the project.

## 💻 Installation and Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
