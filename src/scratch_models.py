"""From-scratch model implementations used across Tasks 1 and 3.

Nothing in this file uses a pre-built model from sklearn or any other library.
NumPy does the arithmetic and SciPy is only used so sparse matrices can be
passed in; the learning rules themselves are written out here.

The final notebook embeds this file directly, so this is the single place the
implementations live.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Task 1: Logistic Regression
#
# The five functions the task brief asks for are sigmoid, loss, gradients,
# train and predict.
# ----------------------------------------------------------------------------
def sigmoid(z):
    """Squash any real number into the range (0, 1)."""
    z = np.clip(z, -500, 500)          # prevents exp() overflow
    return 1.0 / (1.0 + np.exp(-z))


def loss(y, y_hat, sample_weight=None):
    """Log loss (binary cross-entropy) between true and predicted labels."""
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)   # prevents log(0)
    per_row = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    if sample_weight is None:
        return np.mean(per_row)
    return np.sum(sample_weight * per_row) / np.sum(sample_weight)


def gradients(X, y, y_hat, sample_weight=None):
    """Partial derivatives of the log loss w.r.t. the weights and the bias."""
    m = X.shape[0]
    error = y_hat - y
    if sample_weight is not None:
        error = error * sample_weight
    return (X.T @ error) / m, np.sum(error) / m


def train(X, y, bs, epochs, lr, l2=0.0, class_weight=None, shuffle=True,
          seed=42, verbose=False):
    """Mini-batch gradient descent for logistic regression.

    bs, epochs and lr keep the positional order required by the task brief.
    The keyword arguments are the additions we tuned:

        l2           weight of the L2 penalty (0.0 reproduces the first version)
        class_weight "balanced" to reweight the classes to equal total mass
        shuffle      reshuffle the row order before every epoch
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    losses = []
    n_batches = (m - 1) // bs + 1
    rng = np.random.default_rng(seed)

    if class_weight == "balanced":
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: m / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weight = np.array([weight_map[label] for label in y])
    else:
        sample_weight = None

    for epoch in range(epochs):
        # Reshuffling each epoch stops the model from seeing the same fixed
        # batches in the same order every time
        order = rng.permutation(m) if shuffle else np.arange(m)

        for i in range(n_batches):
            batch_idx = order[i * bs:(i + 1) * bs]
            xb = X[batch_idx]
            yb = y[batch_idx]
            swb = None if sample_weight is None else sample_weight[batch_idx]

            y_hat = sigmoid(xb @ w + b)
            dw, db = gradients(xb, yb, y_hat, swb)
            dw = dw + l2 * w                     # L2 penalty, bias left out
            w -= lr * dw
            b -= lr * db

        losses.append(loss(y, sigmoid(X @ w + b), sample_weight))
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  loss {losses[-1]:.6f}")

    return w, b, losses


def predict(X, w, b, threshold=0.5):
    """Label each row 1 when its predicted probability clears the threshold."""
    return (sigmoid(X @ w + b) >= threshold).astype(int)


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# ----------------------------------------------------------------------------
# Task 3: Multinomial Naive Bayes
# ----------------------------------------------------------------------------
def nb_fit(X, y, alpha=1.0):
    """Estimate the class priors and the smoothed per-feature likelihoods."""
    classes = np.unique(y)
    class_log_prior = np.zeros(len(classes))
    feature_log_prob = np.zeros((len(classes), X.shape[1]))

    for idx, c in enumerate(classes):
        X_c = X[y == c]
        class_log_prior[idx] = np.log(X_c.shape[0] / X.shape[0])

        # alpha is Laplace/Lidstone smoothing, so unseen terms are not zeroed
        feature_mass = np.asarray(X_c.sum(axis=0)).ravel() + alpha
        feature_log_prob[idx] = np.log(feature_mass / feature_mass.sum())

    return classes, class_log_prior, feature_log_prob


def nb_predict_log_proba(X, class_log_prior, feature_log_prob):
    return X @ feature_log_prob.T + class_log_prior


def nb_predict(X, classes, class_log_prior, feature_log_prob):
    log_proba = nb_predict_log_proba(X, class_log_prior, feature_log_prob)
    return classes[np.argmax(log_proba, axis=1)]


def nb_predict_proba(X, class_log_prior, feature_log_prob):
    log_proba = nb_predict_log_proba(X, class_log_prior, feature_log_prob)
    log_proba -= log_proba.max(axis=1, keepdims=True)   # stabilise before exp
    proba = np.exp(log_proba)
    return proba / proba.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Task 3: Complement Naive Bayes
#
# Instead of asking "how typical is this document of class c", it asks "how
# untypical is it of everything that is not c", which copes better with an
# uneven class split.
# ----------------------------------------------------------------------------
def cnb_fit(X, y, alpha=1.0, normalize=True):
    classes = np.unique(y)
    complement_weights = np.zeros((len(classes), X.shape[1]))
    total_feature_mass = np.asarray(X.sum(axis=0)).ravel()

    for idx, c in enumerate(classes):
        class_feature_mass = np.asarray(X[y == c].sum(axis=0)).ravel()
        complement_mass = total_feature_mass - class_feature_mass + alpha

        weights = np.log(complement_mass / complement_mass.sum())
        if normalize:
            # keeps long documents from dominating the score
            weights = weights / np.sum(np.abs(weights))

        complement_weights[idx] = weights

    return classes, complement_weights


def cnb_predict_scores(X, complement_weights):
    return X @ complement_weights.T


def cnb_predict(X, classes, complement_weights):
    # low complement score means the document looks unlike the other class
    return classes[np.argmin(cnb_predict_scores(X, complement_weights), axis=1)]


# ----------------------------------------------------------------------------
# Task 3: linear classifier trained by mini-batch SGD
#
# With loss="hinge" this is a linear SVM; with loss="log_loss" it is logistic
# regression again. Writing one trainer for all three losses made comparing
# them a one-line change.
# ----------------------------------------------------------------------------
def compute_grad_f(f, y, loss_name):
    """Derivative of the chosen loss w.r.t. the decision value f."""
    if loss_name == "log_loss":
        return sigmoid(f) - y

    y_signed = 2 * y - 1                 # map {0,1} labels to {-1,+1}
    margin = y_signed * f
    if loss_name == "hinge":
        grad_margin = np.where(margin < 1, -1.0, 0.0)
    elif loss_name == "modified_huber":
        grad_margin = np.where(
            margin <= -1, -4.0,
            np.where(margin >= 1, 0.0, -2.0 * (1 - margin))
        )
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    return grad_margin * y_signed


def reg_gradient(w, penalty, alpha, l1_ratio=0.15):
    if penalty == "l2":
        return alpha * w
    if penalty == "l1":
        return alpha * np.sign(w)
    if penalty == "elasticnet":
        return alpha * (l1_ratio * np.sign(w) + (1 - l1_ratio) * w)
    raise ValueError(f"unsupported penalty: {penalty}")


def hinge_loss_value(f, y):
    margin = (2 * y - 1) * f
    return np.mean(np.maximum(0, 1 - margin))


def sgd_fit(X, y, loss="hinge", penalty="l2", alpha=0.0001, l1_ratio=0.15,
            lr=0.5, epochs=100, bs=256, class_weight=None, random_state=42,
            X_val=None, y_val=None, eval_fn=None):
    """Mini-batch SGD for a linear classifier.

    Passing X_val/y_val together with eval_fn records the validation score
    after every epoch, which is what the convergence plot uses.
    """
    rng = np.random.default_rng(random_state)
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    if class_weight == "balanced":
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: m / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_map[label] for label in y])
    else:
        sample_weight = np.ones(m)

    n_batches = (m - 1) // bs + 1
    losses = []
    val_history = []

    for epoch in range(epochs):
        order = rng.permutation(m)
        for i in range(n_batches):
            batch_idx = order[i * bs:(i + 1) * bs]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            swb = sample_weight[batch_idx]

            f = np.asarray(Xb @ w).ravel() + b
            grad_f = compute_grad_f(f, yb, loss) * swb

            dw = (np.asarray(Xb.T @ grad_f).ravel() / len(batch_idx)
                  + reg_gradient(w, penalty, alpha, l1_ratio))
            db = grad_f.mean()

            w -= lr * dw
            b -= lr * db

        f_full = np.asarray(X @ w).ravel() + b
        if loss == "hinge":
            losses.append(hinge_loss_value(f_full, y))
        elif loss == "log_loss":
            p = np.clip(sigmoid(f_full), 1e-15, 1 - 1e-15)
            losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            losses.append(np.mean(np.abs(compute_grad_f(f_full, y, loss))))

        if X_val is not None and y_val is not None and eval_fn is not None:
            val_history.append(eval_fn(y_val, sgd_predict(X_val, w, b)))

    return w, b, losses, val_history


def sgd_decision_function(X, w, b):
    return np.asarray(X @ w).ravel() + b


def sgd_predict(X, w, b, threshold=0.0):
    return (sgd_decision_function(X, w, b) >= threshold).astype(int)


# ----------------------------------------------------------------------------
# Shared helper
# ----------------------------------------------------------------------------
def best_threshold(y_true, scores, metric, low=0.05, high=0.95, steps=91):
    """Pick the cut-off on `scores` with the best metric value.

    The grid is deliberately coarse: a finer one just fits the noise in a
    single 4000-row validation split.
    """
    candidates = np.quantile(scores, np.linspace(low, high, steps))
    values = [metric(y_true, (scores >= t).astype(int)) for t in candidates]
    best = int(np.argmax(values))
    return float(candidates[best]), float(values[best])
