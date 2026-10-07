"""Target classifier: a fully connected ReLU network written in NumPy.

The network exposes exact gradients of the cross-entropy loss with respect to
its input (needed by FGSM) and its penultimate-layer activations (needed by the
feature-inconsistency descriptor).
"""

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes=(64, 64, 32, 10), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        # He initialisation for ReLU layers.
        self.W = [rng.normal(0.0, np.sqrt(2.0 / m), size=(m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]

    # ------------------------------------------------------------------ forward
    def _forward(self, X):
        """Return pre-activations and activations; acts[-1] are the logits."""
        pre, acts = [], [X]
        h = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pre.append(z)
            h = z if i == last else np.maximum(z, 0.0)
            acts.append(h)
        return pre, acts

    def logits(self, X):
        return self._forward(X)[1][-1]

    def penultimate(self, X):
        """Activation h(x) of the last hidden layer (32 units by default)."""
        return self._forward(X)[1][-2]

    def predict_proba(self, X):
        return softmax(self.logits(X))

    def predict(self, X):
        return self.logits(X).argmax(axis=1)

    # ----------------------------------------------------------------- backward
    def _backward(self, X, y):
        """Gradients of the summed cross-entropy loss w.r.t. weights, biases and input."""
        pre, acts = self._forward(X)
        delta = softmax(acts[-1])
        delta[np.arange(len(y)), y] -= 1.0
        grad_W, grad_b = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            grad_W[i] = acts[i].T @ delta
            grad_b[i] = delta.sum(axis=0)
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (pre[i - 1] > 0)
        return grad_W, grad_b, delta

    def input_gradient(self, X, y):
        """Per-sample gradient of the cross-entropy loss with respect to the input, nabla_x L(f(x), y)."""
        return self._backward(X, y)[2]

    def loss(self, X, y):
        p = self.predict_proba(X)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    # ----------------------------------------------------------------- training
    def fit(self, X, y, epochs=200, lr=1e-3, batch_size=32, seed=0, beta1=0.9, beta2=0.999, eps=1e-8):
        """Train with Adam on the mean cross-entropy loss."""
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                grad_W, grad_b, _ = self._backward(X[idx], y[idx])
                grads = [g / len(idx) for g in grad_W + grad_b]
                t += 1
                for k, (p, g) in enumerate(zip(params, grads)):
                    m[k] = beta1 * m[k] + (1 - beta1) * g
                    v[k] = beta2 * v[k] + (1 - beta2) * g * g
                    m_hat = m[k] / (1 - beta1 ** t)
                    v_hat = v[k] / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return self

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        state = {"sizes": np.array(self.sizes)}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            state[f"W{i}"] = W
            state[f"b{i}"] = b
        return state

    @classmethod
    def from_state_dict(cls, state):
        model = cls(tuple(int(s) for s in state["sizes"]))
        model.W = [np.asarray(state[f"W{i}"]) for i in range(len(model.W))]
        model.b = [np.asarray(state[f"b{i}"]) for i in range(len(model.b))]
        return model
