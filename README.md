# Student Placement Prediction using Machine Learning

A machine learning project built for predicting whether a college student is likely to get placed in campus placements based on their academic scores, technical skills, coding practice, internships, and interview preparation.

---

## Project Overview

Campus placement is a critical milestone for every college student. This project aims to analyze various factors—such as CGPA, coding platform scores, hackathons, mock interviews, and backlogs—and build classification models to predict placement outcomes.

We explore the data, clean it up, handle missing values and outliers, scale the features, and compare four different machine learning classifiers to determine which model performs best.

---

## 5 Core Objectives

1. **Identify Key Placement Factors:** Explore the dataset using charts and correlations to see how academic and technical performance affects placement status.
2. **Ensure Data Quality:** Clean the dataset by filling in missing values using median imputation and removing extreme outliers using the Interquartile Range (IQR) method.
3. **Prepare ML-Ready Features:** Convert categorical text columns (like gender and preferred domain) into numbers and normalize all numerical scores between 0 and 1 using MinMaxScaler.
4. **Predict Placement Outcomes:** Train four classification models—Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, and Decision Tree.
5. **Benchmark Model Performance:** Compare all four models using standard evaluation metrics (Accuracy, Precision, Recall, and F1-Score) to find the best model for practical use.

---

## Dataset Features

The dataset contains records of 8,000 students with 28 attributes:

* **Academic Information:** `cgpa`, `backlogs`, `attendance_percentage`
* **Technical & Coding Skills:** `coding_score`, `technical_score`, `coding_platform_score`, `github_projects`
* **Practical Experience:** `projects_count`, `major_projects`, `internships_count`, `internship_months`
* **Soft Skills & Preparation:** `communication_score`, `soft_skills_score`, `mock_interview_score`, `training_hours`, `resume_score`
* **Achievements & Profile:** `hackathons_participated`, `hackathons_won`, `certifications_count`, `linkedin_score`
* **Demographics & Preferences:** `age`, `gender`, `preferred_domain`
* **Target Label:** `placed` (1 = Placed, 0 = Not Placed)

---

## Step-by-Step Workflow

1. **Exploratory Data Analysis (EDA):**
   * Checked summary statistics, null counts, duplicate records, and class balance.
   * Visualized distributions for CGPA, coding scores, mock interview scores, and attendance.
   * Plotted **boxplots** and **scatterplots** (CGPA vs Coding Score / Technical Score) across placement status.
   * Created a correlation heatmap to understand relationships between features.

2. **Data Cleaning & Handling Outliers:**
   * Handled missing values in 11 numerical columns using median imputation.
   * Visualized outliers using **boxplots** and applied the IQR method ($Q_1 - 1.5 \times IQR$ to $Q_3 + 1.5 \times IQR$) to filter out unrealistic extreme values.

3. **Feature Preprocessing:**
   * Used **Label Encoding** for `gender`.
   * Used **One-Hot Encoding** for `preferred_domain`.
   * Dropped non-predictive identifier (`student_id`).
   * Split data into **80% training** and **20% testing** sets using stratified sampling.
   * Scaled all numerical features to $[0, 1]$ using **MinMaxScaler**.

4. **Model Training & Comparison:**
   * Trained 4 classifiers on the processed data:
     * **Logistic Regression**
     * **K-Nearest Neighbors (KNN)** (tested $K=1$ to $20$)
     * **Gaussian Naive Bayes**
     * **Decision Tree**
   * Plotted **confusion matrix heatmaps** for each model alongside accuracy and F1-score comparative bar charts.

---

## Model Comparison Results

Here is the final performance comparison across all four models on the test set:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.0%** | **82.7%** | **83.3%** | **0.830** |
| **Naive Bayes** | **84.3%** | **79.3%** | **83.6%** | **0.814** |
| **Decision Tree** | **79.4%** | **74.9%** | **74.9%** | **0.749** |
| **KNN (K=2)** | **68.7%** | **70.5%** | **40.5%** | **0.515** |

---

## Final Conclusion

* **Best Performer:** **Logistic Regression** achieved the highest overall performance, leading with **86.0% accuracy** and an **F1-score of 0.830**, offering an ideal balance between precision and recall.
* **Runner-Up:** **Naive Bayes** also performed remarkably well with **84.3% accuracy** and **0.814 F1-score**, proving to be a strong probabilistic baseline.
* **Key Placement Drivers:** Students with higher CGPA, solid coding/technical test scores, mock interview practice, and zero backlogs have significantly higher placement chances.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/shivam-kr11/INT234_CA2.git
cd INT234_CA2
```

### 2. Install required dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Python script
```bash
python student_placement_prediction.py
```

