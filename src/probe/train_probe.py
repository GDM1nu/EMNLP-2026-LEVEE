from sklearn.linear_model import LogisticRegression

class LinearProbe:
    def __init__(self, seed=42):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            multi_class="auto",
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)
