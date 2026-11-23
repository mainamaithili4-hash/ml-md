#STEP 1: Import librabies
import pandas as pd
import joblib
from sklearn.datasets import load_breat_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocesssing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearnensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import(
 f1_score,
 accuracy_score,
 classification_report,
 confusion_matrix
)

#STEP 2: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(
  data.data,
  columns=data.feature_names
)
y = pd.Series(data.target,name="target")
print("Dataset shape:",X.shape)
print("\nFrist 5 rows:")
print(X.head())
print("\nTarget values:")
print(y.value_counts())

#STEP 3: Split dataset
X_train,X_test,y_train,y_test=train_test_split(
  X,
  y,
  test_size = 0.20,
  random_state = 42,
  stratify = y
)
print("\nTraining samples:",X_train.shape[0])
print("\nTesting samples:",X_test.shape[0])

#STEP 4: Define classsification algorithms
models = {
  "Logistic Regression":LogisticRegression(
    max_iter=5000
  ),
  "KNN": KNeighborsclassifier(),
  "Decision Tree": DecisionTreeClassifier(
    random_state=42
  ),
  "Random Forest": RandomForestClassifier(
    random_state=42
  ),
