
# Objective
Predict whether a person is **creditworthy** (good credit) or **not creditworthy**
(bad credit) based on their financial history — a simple classification problem.

A realistic **sample dataset of 1,000 people** is generated inside the script itself
(no external file needed), using simple, transparent rules based on real-world
credit factors. Features include:



**Top predictors of creditworthiness:** payment history score, debt-to-income ratio,
and years of employment — matching what real-world credit scoring relies on.

CodeAlpha_CreditScoring/
├── credit_scoring.py       
├── requirements.txt
├── README.md
└── outputs/
    ├── credit_dataset.csv
    ├── model_comparison.csv
    ├── results_summary.txt
    ├── 01_class_distribution.png
    ├── 02_model_comparison_bar.png
    ├── 03_roc_curves.png
    ├── 04_feature_importance.png
    ├── 05_decision_tree_diagram.png   
    ├── cm_logistic_regression.png
    └── cm_decision_tree.png

