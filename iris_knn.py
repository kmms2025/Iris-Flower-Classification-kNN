"""CLA-3 Case Study: Iris Flower Species Classification Using kNN"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names
print("Samples:", X.shape, "Classes:", dict(zip(names, np.bincount(y))))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", len(X_tr), "Test:", len(X_te))

# Choose k using 5-fold CV on the training set only
ks = list(range(1, 16))
cv = []
for k in ks:
    p = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv.append(cross_val_score(p, X_tr, y_tr, cv=5).mean())
best_k = ks[int(np.argmax(cv))]
print("CV acc:", dict(zip(ks, np.round(cv, 4))), "best k:", best_k)

# Baseline comparisons with k=5 (the textbook default) and without scaling
res = {}
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_tr, y_tr)
pred = model.predict(X_te)
res["best_k"] = best_k
res["test_acc_best_k_scaled"] = accuracy_score(y_te, pred)
m5 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_tr, y_tr)
res["test_acc_k5_scaled"] = accuracy_score(y_te, m5.predict(X_te))
mns = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr)
res["test_acc_best_k_unscaled"] = accuracy_score(y_te, mns.predict(X_te))
res["cv"] = dict(zip(map(str, ks), cv))
cm = confusion_matrix(y_te, pred)
res["cm"] = cm.tolist()
print(res)
print(classification_report(y_te, pred, target_names=names))
print(cm)

# Correct / incorrect examples
wrong = np.where(pred != y_te)[0]
right = np.where(pred == y_te)[0]
print("wrong idx:", wrong)
for i in list(right[:1]) + list(wrong[:2]):
    print(i, X_te[i], "true", names[y_te[i]], "pred", names[pred[i]])
    sc = model.named_steps["standardscaler"]; knn = model.named_steps["kneighborsclassifier"]
    d, idx = knn.kneighbors(sc.transform(X_te[i:i+1]))
    print("  neighbour labels:", [names[y_tr[j]] for j in idx[0]], "dist", np.round(d[0], 3))

# Worked example: first correct test sample, distances to 5 nearest (manual)
i = right[0]
sc = model.named_steps["standardscaler"]
print("scaler mean", sc.mean_, "scale", sc.scale_)

# Figures
plt.figure(figsize=(5,3.2)); plt.plot(ks, cv, marker="o"); plt.xlabel("k"); plt.ylabel("5-fold CV accuracy")
plt.title("Choosing k"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig_k.png", dpi=200)
fig, ax = plt.subplots(figsize=(4,3.4))
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, colorbar=False, cmap="Blues")
plt.title("Confusion matrix (test set)"); plt.tight_layout(); plt.savefig("fig_cm.png", dpi=200)
json.dump(res, open("results.json","w"), indent=1)

# ---- Worked example used in Section 3 of the report (raw cm, no scaling, k=5) ----
q = X_te[0]
d = np.sqrt(((X_tr - q) ** 2).sum(axis=1))
print("\nWorked example, test flower:", q, "true:", names[y_te[0]])
for rank, j in enumerate(np.argsort(d)[:5], 1):
    print(rank, X_tr[j], names[y_tr[j]], round(d[j], 3))
