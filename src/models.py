import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import numpy as np

from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        ### NEW MODELS BELOW ###
        # TODO: Set all properly
        "fourier_sine_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "radial_sine_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "linear_sine_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "linear_modulo_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "saw_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "square_wave_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "triangle_wave_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

class MLPModel:
    def __init__(self):
        self.name = "mlp"
    
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cuda(), ys.cuda() #I'm basing this off of the GD model
        
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    input_size = train_xs.shape[1] if train_xs.ndim == 2 else 1
                    train_xs = train_xs.view(-1, input_size).float()
                    train_ys = train_ys.view(-1, 1).float()

                    model = MLP(input_size)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    # Simple training loop
                    for epoch in range(50):  # or use early stopping
                        model.train()
                        optimizer.zero_grad()
                        output = model(train_xs)
                        loss = criterion(output, train_ys)
                        loss.backward()
                        optimizer.step()

                    # Predict the next step
                    test_x = xs[j, i:i+1].view(1, -1).float()
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(test_x)
                        pred[j] = y_pred[0, 0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)

class FourierRegressionModel:
    def __init__(self, num_terms=3):
        self.name = "fourier"
        self.num_terms = num_terms  # Number of sine/cosine pairs

    def _design_matrix(self, t):
        """
        Build Fourier design matrix for input t.
        t: (N,) tensor
        returns: (N, 2K+1) tensor
        """
        t = t.view(-1, 1)
        X = [torch.ones_like(t)]  # bias term
        for k in range(1, self.num_terms + 1):
            X.append(torch.cos(2 * np.pi * k * t))
            X.append(torch.sin(2 * np.pi * k * t))
        return torch.cat(X, dim=1)

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs = xs[j, :i]            # shape (i,)
                    train_ys = ys[j, :i]            # shape (i,)

                    # # Use xs or time indices as Fourier input
                    # X = self._design_matrix(train_xs)   # (i, 2K+1)
                    # y = train_ys.view(-1, 1)            # (i, 1)

                    X = self._design_matrix(train_xs)   # (i, 2K+1)
                    y = train_ys.view(-1, 1)            # (i, 1)

                    if X.shape[0] != y.shape[0]:
                        print(f"Shape mismatch: X.shape = {X.shape}, y.shape = {y.shape}")
                        continue  # skip this step for now instead of crashing

                    beta = torch.linalg.lstsq(X, y).solution

                    # Fit via least squares: beta = (X^T X)^(-1) X^T y
                    beta = torch.linalg.lstsq(X, y).solution  # shape (2K+1, 1)

                    # Predict at the next step (use next x or time)
                    next_x = xs[j, i].view(1)          # scalar input for prediction
                    X_test = self._design_matrix(next_x)      # (1, 2K+1)
                    y_pred = X_test @ beta             # (1, 1)
                    pred[j] = y_pred.item()

            preds.append(pred)

        return torch.stack(preds, dim=1)

class SeparableFourierRegressor:
    def __init__(self, num_terms=3):
        self.name = f"separable_fourier_{num_terms}"  # Custom name for tracking
        self.num_terms = num_terms  # Number of sine terms per dimension

    def _design_matrix(self, x):
        """
        x: (N,) tensor for a single dimension
        returns: (N, 3 * num_terms + 1) tensor with [bias, sin(kx), cos(kx), kx]
        """
        x = x.view(-1, 1)  # Ensure column vector
        features = [torch.ones_like(x)]  # Bias term
        for k in range(1, self.num_terms + 1):
            features.append(torch.sin(k * x))
            features.append(torch.cos(k * x))
        return torch.cat(features, dim=1)  # shape: (N, 2*num_terms + 1)

    def __call__(self, xs, ys, inds=None):
        # xs, ys = xs.cpu(), ys.cpu()
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs = xs[j, :i]       # shape (i, d)
                    train_ys = ys[j, :i]       # shape (i,)
                    
                    # Build a separable design matrix
                    X_parts = [self._design_matrix(train_xs[:, d]) for d in range(train_xs.shape[1])]
                    X_full = torch.cat(X_parts, dim=1)  # shape: (i, d * (2*num_terms+1))
                    y = train_ys.view(-1, 1)            # shape (i, 1)

                    try:
                        beta = torch.linalg.lstsq(X_full, y).solution
                    except RuntimeError:
                        continue  # skip if not solvable

                    # Build test feature for next point
                    next_x = xs[j, i]  # shape (d,)
                    test_parts = [self._design_matrix(next_x[d].view(1)) for d in range(next_x.shape[0])]
                    X_test = torch.cat(test_parts, dim=1)  # shape: (1, d * (2*num_terms+1))

                    y_pred = X_test @ beta               # shape (1, 1)
                    pred[j] = y_pred.item()

            preds.append(pred)

        return torch.stack(preds, dim=1)

from symfit import variables, parameters, Fit, sin
import numpy as np

class SymFitModel:
    def __init__(self):
        self.name = "symfit_nd"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])

        preds = []

        for i in inds:
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                for j in range(ys.shape[0]):
                    X = xs[j, :i].numpy()  # shape (i, d)
                    Y = ys[j, :i].numpy()

                    d = X.shape[1]
                    # Create symbolic inputs: x0, x1, ..., xd-1
                    x_vars = variables(','.join([f'x{k}' for k in range(d)]))
                    amps = parameters(','.join([f'amp{k}' for k in range(d)]))
                    freqs = parameters(','.join([f'freq{k}' for k in range(d)]))
                    phases = parameters(','.join([f'phase{k}' for k in range(d)]))
                    offset = parameters('offset')[0]

                    # Build model: sum of amp_k * sin(freq_k * x_k + phase_k)
                    model = sum(
                        amps[k] * sin(freqs[k] * x_vars[k] + phases[k])
                        for k in range(d)
                    ) + offset

                    data_dict = {x_vars[k]: X[:, k] for k in range(d)}
                    data_dict['y'] = Y

                    try:
                        fit = Fit(model, **data_dict)
                        result = fit.execute()

                        x_input = xs[j, i].numpy()
                        x_eval = {x_vars[k]: x_input[k] for k in range(d)}
                        y_pred = model(**x_eval, **result.params).value

                        pred[j] = torch.tensor(y_pred)
                    except Exception:
                        pred[j] = torch.tensor(0.0)

            preds.append(pred)

        return torch.stack(preds, dim=1)
