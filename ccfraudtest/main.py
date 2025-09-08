import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

random_seed = 1


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(
        os.path.join(
            script_dir,
            "data/credit-card-fraud-detection-dataset-2023/creditcard_2023.csv",
        )
    )

    X = df.drop(
        [
            "Class",
            "id",
        ],
        axis=1,
    )
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    clf = DecisionTreeClassifier(random_state=random_seed, max_depth=4)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Not Fraud", "Actual Fraud"],
        columns=["Predicted Not Fraud", "Predicted Fraud"]
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    n_features = X.shape[1]
    max_depth = clf.get_depth()

    plt.figure(figsize=(max_depth * 12, n_features))
    plot_tree(
        clf,
        feature_names=X.columns,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        fontsize=10,
    )

    plt.savefig("decision_tree.svg")


if __name__ == "__main__":
    main()
