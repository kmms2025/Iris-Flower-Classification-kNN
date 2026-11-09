# Iris Flower Species Classification Using kNN

Machine Learning Classification Project (CLA-3 Case Study): predicts the species of an
iris flower (setosa, versicolor, virginica) from its sepal and petal measurements
using k-Nearest Neighbours (kNN) with scikit-learn.

## Dataset
Fisher's Iris dataset (150 flowers, 4 features, 3 classes), loaded with `sklearn.datasets.load_iris()`.

## Method
- Stratified 80/20 split (120 train / 30 test), `random_state = 42`
- Features standardised with `StandardScaler` (inside a Pipeline, fitted on training data only)
- k chosen with 5-fold cross-validation on the training set (k = 1 to 15)

## Results
| Model (k = 5) | Test accuracy |
|---|---|
| kNN, standardised features (main model) | 93.3% (28/30) |
| kNN, raw features (comparison only) | 100% (30/30) |

Best k from cross-validation: 5 (CV accuracy 96.7%).
The only errors were 2 virginica flowers predicted as versicolor.

## How to run
```
pip install -r requirements.txt
python iris_knn.py
```
Or upload `Iris_kNN_Colab.ipynb` to Google Colab and choose Runtime > Run all.
The script prints all results and saves `fig_k.png`, `fig_cm.png` and `results.json`.

## Files
| File | Description |
|---|---|
| `iris_knn.py` | Complete code |
| `Iris_kNN_Colab.ipynb` | Same code as a Colab notebook |
| `requirements.txt` | Python libraries needed |
| `fig_k.png` | Cross-validation accuracy for k = 1 to 15 |
| `fig_cm.png` | Confusion matrix on the test set |
| `results.json` | Saved results |
| `run_output.txt` | Printed output of one run |
| `CLA3_Case_Study_Iris_kNN.pdf` | Case study report |

## Reference
T. Cover and P. Hart, "Nearest neighbor pattern classification," IEEE Trans. Inf. Theory, vol. 13, no. 1, pp. 21-27, 1967.
