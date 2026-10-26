<<<<<<< HEAD
# Student Performance Prediction

A small machine-learning project that trains a scikit-learn regression model from student data.

## Project structure

```text
student-performance-prediction/
├── data/
│   └── student_data.csv
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Add the student records to `data/student_data.csv`. The file must contain a numeric target column, `final_score` by default, plus one or more feature columns.

## Train the model

```bash
python train_model.py --target final_score
```

Use a different dataset or target column when needed:

```bash
python train_model.py --data data/student_data.csv --target score --output student_performance_model.joblib
```

The trained pipeline is saved as a `.joblib` file in the project root.
=======
# student-performance-prediction
A beginner-friendly machine learning project that predicts student exam scores based on study hours using Linear Regression in Python.
>>>>>>> e36c5f8ac7df0887f550cea61044755b988daa68
