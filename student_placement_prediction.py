#================ IMPORTING LIBRARIES ================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#================ LOADING DATASET ================
data = pd.read_csv("data/student_placement_train.csv")

#================ BASIC DATASET INFORMATION ================
print(data.head())
print(data.shape)
print(data.columns)

#================ DATASET INFORMATION ================
print(data.info())

#================ STATISTICAL SUMMARY ================
print(data.describe())

#================ CHECKING MISSING VALUES ================
print(data.isnull().sum())

#================ CHECKING DUPLICATE VALUES ================
print(data.duplicated().sum())

#================ CHECKING UNIQUE VALUES ================
print(data.nunique())

print(data.columns)

#================ TARGET VARIABLE DISTRIBUTION ================
print(data["placed"].value_counts())
print(data["placed"].value_counts(normalize=True) * 100)

#================ NUMERICAL AND CATEGORICAL COLUMNS ================
print("Numerical Columns:")
print(data.select_dtypes(include=["int64", "float64"]).columns)

print("Categorical Columns:")
print(data.select_dtypes(include=["object", "category"]).columns)

#================ CATEGORICAL COLUMN DISTRIBUTION ================
print(data["gender"].value_counts())
print(data["preferred_domain"].value_counts())
print(data["placed"].value_counts())

#================ TARGET VARIABLE VISUALIZATION ================
sns.countplot(x="placed", data=data)
plt.title("Placement Distribution")
plt.xlabel("Placed")
plt.ylabel("Number of Students")
plt.show()

#================ NUMERICAL FEATURE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["cgpa"], bins=20, kde=True)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.show()

#================ ATTENDANCE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["attendance_percentage"], bins=20, kde=True)
plt.title("Attendance Percentage Distribution")
plt.xlabel("Attendance Percentage")
plt.ylabel("Number of Students")
plt.show()

#================ CODING SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["coding_score"], bins=20, kde=True)
plt.title("Coding Score Distribution")
plt.xlabel("Coding Score")
plt.ylabel("Number of Students")
plt.show()

#================ APTITUDE SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["aptitude_score"], bins=20, kde=True)
plt.title("Aptitude Score Distribution")
plt.xlabel("Aptitude Score")
plt.ylabel("Number of Students")
plt.show()

#================ COMMUNICATION SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["communication_score"], bins=20, kde=True)
plt.title("Communication Score Distribution")
plt.xlabel("Communication Score")
plt.ylabel("Number of Students")
plt.show()

#================ TECHNICAL SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["technical_score"], bins=20, kde=True)
plt.title("Technical Score Distribution")
plt.xlabel("Technical Score")
plt.ylabel("Number of Students")
plt.show()

#================ AGE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.show()

#================ BACKLOGS DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.countplot(x="backlogs", data=data)
plt.title("Backlogs Distribution")
plt.xlabel("Number of Backlogs")
plt.ylabel("Number of Students")
plt.show()

#================ GENDER DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="gender", data=data)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.show()

#================ PREFERRED DOMAIN DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.countplot(x="preferred_domain", data=data)
plt.title("Preferred Domain Distribution")
plt.xlabel("Preferred Domain")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.show()

#================ PROJECTS COUNT DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="projects_count", data=data)
plt.title("Projects Count Distribution")
plt.xlabel("Number of Projects")
plt.ylabel("Number of Students")
plt.show()

#================ INTERNSHIP COUNT DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="internships_count", data=data)
plt.title("Internship Count Distribution")
plt.xlabel("Number of Internships")
plt.ylabel("Number of Students")
plt.show()

#================ MAJOR PROJECTS DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="major_projects", data=data)
plt.title("Major Projects Distribution")
plt.xlabel("Number of Major Projects")
plt.ylabel("Number of Students")
plt.show()

#================ CERTIFICATIONS COUNT DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="certifications_count", data=data)
plt.title("Certifications Count Distribution")
plt.xlabel("Number of Certifications")
plt.ylabel("Number of Students")
plt.show()

#================ HACKATHONS PARTICIPATED DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="hackathons_participated", data=data)
plt.title("Hackathons Participated Distribution")
plt.xlabel("Number of Hackathons")
plt.ylabel("Number of Students")
plt.show()

#================ HACKATHONS WON DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="hackathons_won", data=data)
plt.title("Hackathons Won Distribution")
plt.xlabel("Number of Hackathons Won")
plt.ylabel("Number of Students")
plt.show()

#================ CODING PLATFORM SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["coding_platform_score"], bins=20, kde=True)
plt.title("Coding Platform Score Distribution")
plt.xlabel("Coding Platform Score")
plt.ylabel("Number of Students")
plt.show()

#================ GITHUB PROJECTS DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="github_projects", data=data)
plt.title("GitHub Projects Distribution")
plt.xlabel("Number of GitHub Projects")
plt.ylabel("Number of Students")
plt.show()

#================ LINKEDIN SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["linkedin_score"], bins=20, kde=True)
plt.title("LinkedIn Score Distribution")
plt.xlabel("LinkedIn Score")
plt.ylabel("Number of Students")
plt.show()

#================ RESUME SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["resume_score"], bins=20, kde=True)
plt.title("Resume Score Distribution")
plt.xlabel("Resume Score")
plt.ylabel("Number of Students")
plt.show()

#================ SOFT SKILLS SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["soft_skills_score"], bins=20, kde=True)
plt.title("Soft Skills Score Distribution")
plt.xlabel("Soft Skills Score")
plt.ylabel("Number of Students")
plt.show()

#================ LEADERSHIP SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["leadership_score"], bins=20, kde=True)
plt.title("Leadership Score Distribution")
plt.xlabel("Leadership Score")
plt.ylabel("Number of Students")
plt.show()

#================ EXTRACURRICULAR SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["extracurricular_score"], bins=20, kde=True)
plt.title("Extracurricular Score Distribution")
plt.xlabel("Extracurricular Score")
plt.ylabel("Number of Students")
plt.show()

#================ TRAINING HOURS DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["training_hours"], bins=20, kde=True)
plt.title("Training Hours Distribution")
plt.xlabel("Training Hours")
plt.ylabel("Number of Students")
plt.show()

#================ MOCK INTERVIEW SCORE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["mock_interview_score"], bins=20, kde=True)
plt.title("Mock Interview Score Distribution")
plt.xlabel("Mock Interview Score")
plt.ylabel("Number of Students")
plt.show()

#================ GENDER VS PLACEMENT ================

plt.figure(figsize=(8, 5))
sns.countplot(x="gender", hue="placed", data=data)

plt.title("Gender vs Placement")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.show()

#================ PREFERRED DOMAIN VS PLACEMENT ================

plt.figure(figsize=(12, 6))
sns.countplot(x="preferred_domain", hue="placed", data=data)
plt.title("Preferred Domain vs Placement")
plt.xlabel("Preferred Domain")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.show()

#================ CORRELATION ANALYSIS ================

corr = data.select_dtypes(include="number").drop(columns=["student_id"], errors="ignore").corr()
print(corr)
plt.figure(figsize=(16, 10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

#================ CHECKING MISSING VALUES ================

print(data.isnull().sum())
print("Total missing values:", data.isnull().sum().sum())

#================ CHECKING DUPLICATE ROWS ================

print("Total duplicate rows:", data.duplicated().sum())

#================ HANDLING MISSING VALUES ================

missing_columns = {
    "attendance_percentage": data["attendance_percentage"].median(),
    "coding_score": data["coding_score"].median(),
    "aptitude_score": data["aptitude_score"].median(),
    "communication_score": data["communication_score"].median(),
    "technical_score": data["technical_score"].median(),
    "internship_months": data["internship_months"].median(),
    "linkedin_score": data["linkedin_score"].median(),
    "resume_score": data["resume_score"].median(),
    "soft_skills_score": data["soft_skills_score"].median(),
    "training_hours": data["training_hours"].median(),
    "mock_interview_score": data["mock_interview_score"].median()
}

data.fillna(missing_columns, inplace=True)

#================ VERIFYING MISSING VALUES ================

print(data.isnull().sum())

#================ CHECKING OUTLIERS ================

numerical_columns = data.select_dtypes(include="number").columns.drop(["student_id", "placed"], errors="ignore")

for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_limit) | (data[column] > upper_limit)]

    print(column, ":", len(outliers))


    
#================ REMOVING OUTLIERS ================

for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    data = data[
        (data[column] >= lower_limit) &
        (data[column] <= upper_limit)
    ]

print("Shape after removing outliers:", data.shape)

#================ ENCODING GENDER LABEL ENCODING ================

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])

#================ ENCODING PREFERRED DOMAIN ONE HOT ENCODING================
data = pd.get_dummies(data, columns=["preferred_domain"], dtype=int)

#================ SEPARATING FEATURES AND TARGET ================
X = data.drop(["placed"], axis=1)
y = data["placed"]

#================ TRAIN TEST SPLIT ================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

#================ SCALING NUMERICAL COLUMNS ================
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = X_train.copy()
X_test = X_test.copy()
numerical_columns = X_train.select_dtypes(include="number").columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

#================ TRAINING LOGISTIC REGRESSION ================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

model = LogisticRegression(class_weight="balanced", max_iter=1000,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#================ EVALUATING LOGISTIC REGRESSION ================
print("================Logistice Regression ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("=====================================================")

#================ TRAINING KNN ================
from sklearn.neighbors import KNeighborsClassifier

for k in range(1, 21):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    y_pred_k = model_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_k)
    print("K =", k, "Accuracy =", accuracy)
    
#================Suitable K  ================
for k in range(1,21):
    model_knn = KNeighborsClassifier(n_neighbors=k,weights="distance",metric="minkowski")
    model_knn.fit(X_train, y_train)

    y_pred = model_knn.predict(X_test)

    print(
    "K =", k,
    "| Accuracy =", round(accuracy_score(y_test, y_pred), 3),
    "| Precision =", round(precision_score(y_test, y_pred), 3),
    "| Recall =", round(recall_score(y_test, y_pred), 3),
    "| F1 Score =", round(f1_score(y_test, y_pred), 3)
)

#Conclusion-> As K increases, accuracy generally rises, but recall and F1-score fall sharply. At K = 14 and above, the model fails to identify any placed students.
#choose k as 1 to 5

model_knn = KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

#================ EVALUATING KNN ================
print("================ KNN ================")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall:", recall_score(y_test, y_pred_knn))
print("F1 Score:", f1_score(y_test, y_pred_knn))
print("=====================================================")


