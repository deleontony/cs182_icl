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
import math

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
        "sum_sine_regression": [
            (NNModel, {"n_neighbors": 3}),
            # (TorchSumSineModel, {}), # Fix this, so slow
            (MLPModel,{}), #This will be slow.
            (SIRENModel,{}), #THis will also be slow
        ],
        "radial_sine_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "linear_sine_regression": [
            (NNModel, {"n_neighbors": 3}),
        ],
        "linear_modulo_regression": [
            (NNModel, {"n_neighbors": 3}),
            (PiecewiseLinearModel,{"num_relus": 1}),
        ],
        "saw_regression": [
            (NNModel, {"n_neighbors": 3}),
            (PiecewiseLinearModel,{"num_relus": 1}),
        ],
        "square_wave_regression": [
            (NNModel, {"n_neighbors": 3}),
            (PiecewiseLinearModel,{"num_relus": 3}),
        ],
        "triangle_wave_regression": [
            (NNModel, {"n_neighbors": 3}),
            (PiecewiseLinearModel,{"num_relus": 2}),
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
                    model = model.to(xs.device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    # Simple training loop
                    for epoch in range(10):  # or use early stopping
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

# Sine activation with frequency scaling
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

# Initialization function for SIREN layers
def siren_init(layer, w0=1.0):
    with torch.no_grad():
        num_input = layer.in_features
        layer.weight.uniform_(-math.sqrt(6 / num_input) / w0, math.sqrt(6 / num_input) / w0)

# Full SIREN network
class SIREN(nn.Module):
    def __init__(self, input_size, hidden_size=64, hidden_layers=2, w0=1.0, w0_initial=30.0):
        super().__init__()

        layers = []

        # First layer with special frequency w0_initial
        first_layer = nn.Linear(input_size, hidden_size)
        siren_init(first_layer, w0_initial)
        layers.append(first_layer)
        layers.append(Sine(w0_initial))

        # Hidden layers with standard w0
        for _ in range(hidden_layers):
            hidden_layer = nn.Linear(hidden_size, hidden_size)
            siren_init(hidden_layer, w0)
            layers.append(hidden_layer)
            layers.append(Sine(w0))

        # Final output layer (no activation)
        final_layer = nn.Linear(hidden_size, 1)
        siren_init(final_layer, w0)
        layers.append(final_layer)

        # Wrap in nn.Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SIRENModel:
    def __init__(self):
        self.name = "siren"
    
    def __call__(self, xs, ys, inds=None):
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
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    input_size = train_xs.shape[1] if train_xs.ndim == 2 else 1
                    train_xs = train_xs.view(-1, input_size).float()
                    train_ys = train_ys.view(-1, 1).float()

                    model = SIREN(input_size).cuda()
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    for epoch in range(10):
                        model.train()
                        optimizer.zero_grad()
                        output = model(train_xs)
                        loss = criterion(output, train_ys)
                        loss.backward()
                        optimizer.step()

                    test_x = xs[j, i:i+1].view(1, -1).float()
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(test_x)
                        pred[j] = y_pred[0, 0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)

from scipy.optimize import curve_fit
# from multiprocessing import Pool
from joblib import Parallel, delayed

import torch.optim as optim
class TorchSumSineModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.name = "torch_sine_sum"

    def fit(self, x_train, y_train, epochs=10, lr=0.01): #num_epochs might be too low
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        _, n_dims = x_train.shape

        # Create parameters for all dimensions
        amps = nn.Parameter(torch.ones(n_dims, device=self.device))
        freqs = nn.Parameter(torch.ones(n_dims, device=self.device))
        phases = nn.Parameter(torch.zeros(n_dims, device=self.device))
        offsets = nn.Parameter(torch.zeros(n_dims, device=self.device))

        optimizer = optim.Adam([amps, freqs, phases, offsets], lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            # Compute predictions: shape (n_points, n_dims)
            sine_outputs = amps * torch.sin(x_train * freqs + phases) + offsets
            # Sum across dimensions to get final prediction
            y_pred = torch.sum(sine_outputs, dim=1)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

        return amps.detach(), freqs.detach(), phases.detach(), offsets.detach()

    def predict(self, x, params):
        x = x.to(self.device)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        amps, freqs, phases, offsets = params
        y_pred = torch.sum(amps * torch.sin(x * freqs + phases) + offsets, dim=1)
        return y_pred

    def _call_single(self, xs, ys, inds, b):
        n_points = xs.shape[0]
        if inds is None:
            inds = range(n_points)

        preds = torch.zeros(n_points, device=self.device)
        for i in inds:
            if i == 0:
                preds[i] = 0.0
                continue
            x_train, y_train = xs[:i], ys[:i]
            x_test = xs[i]

            params = self.fit(x_train, y_train)
            preds[i] = self.predict(x_test, params).item()

        return i, preds.unsqueeze(0)

    def __call__(self, xs, ys, inds=None):
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        print(f"xs.shape = {xs.shape}\nys.shape = {ys.shape}")

        if xs.ndim == 3:
            results = []
            for i in range(xs.shape[0]):
                _, pred = self._call_single(xs[i], ys[i], inds, i)
                results.append(pred)
            return torch.cat(results, dim=0)
        elif xs.ndim == 2:
            _, pred = self._call_single(xs, ys, inds, 0)
            return pred
        else:
            raise ValueError("Input xs must be 2D or 3D tensor.")

class PiecewiseLinearModel:
    def __init__(self, num_relus=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_relus = num_relus
        self.name = "piecewise_linear"

    def _piecewise_linear(self, x, weights, biases):
        """
        x: shape (n_points, n_dims)
        weights: shape (n_dims, num_relus)
        biases: shape (n_dims, num_relus)
        returns: shape (n_points,)
        """
        n_points, n_dims = x.shape
        out = biases[0] + weights[0] @ x  # Linear term for the first unit

        # Add ReLU components
        for i in range(1, self.num_relus):  # Additional ReLU terms
            out += torch.relu(biases[i] + weights[i] @ x)
        return out

    def fit(self, x_train, y_train, epochs=10, lr=0.01):
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        n_points, n_dims = x_train.shape

        weights = nn.Parameter(torch.randn(n_dims, self.num_relus, device=self.device) * 0.1)
        biases = nn.Parameter(torch.randn(n_dims, self.num_relus, device=self.device) * 0.1)

        optimizer = optim.Adam([weights, biases], lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            y_pred = self._piecewise_linear(x_train, weights, biases)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

        return weights.detach(), biases.detach()

    def predict(self, x, params):
        x = x.to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        weights, biases = params
        return self._piecewise_linear(x, weights, biases)

    def _call_single(self, xs, ys, inds, b):
        n_points = xs.shape[0]
        if inds is None:
            inds = range(n_points)

        preds = torch.zeros(n_points, device=self.device)
        last_i = -1

        for i in inds:
            last_i = i
            if i == 0:
                preds[i] = 0.0
                continue
            x_train, y_train = xs[:i], ys[:i]
            x_test = xs[i].unsqueeze(0) if xs[i].ndim == 1 else xs[i]

            params = self.fit(x_train, y_train)
            preds[i] = self.predict(x_test, params).item()

        return last_i, preds.unsqueeze(0)

    def __call__(self, xs, ys, inds=None):
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        print(f"xs.shape = {xs.shape}\nys.shape = {ys.shape}")

        if xs.ndim == 3:
            results = []
            for i in range(xs.shape[0]):
                _, pred = self._call_single(xs[i], ys[i], inds, i)
                results.append(pred)
            return torch.cat(results, dim=0)
        elif xs.ndim == 2:
            _, pred = self._call_single(xs, ys, inds, 0)
            return pred
        else:
            raise ValueError("Input xs must be 2D or 3D tensor.")



class ScipySumSineModel:
    def __init__(self):
        self.name = f"scipy_sine_sum"

    def _single_sine(self, x, amp, freq, phase, offset):
        return amp * np.sin((freq * x + phase) % (2 * np.pi)) + offset

    def fit(self, x_train, y_train):
        x_train = x_train.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()

        n_points, n_dims = x_train.shape
        amps = np.zeros(n_dims)
        freqs = np.zeros(n_dims)
        phases = np.zeros(n_dims)
        offsets = np.zeros(n_dims)

        preds = np.zeros((n_points, n_dims))
        for direction in ["forward", "backward"]:
            if direction == "forward":
                residual = y_train.copy()
                dim_order = range(n_dims)
            else:
                dim_order = range(n_dims - 2, -1, -1) # skip last sine
            
            for dim in dim_order:
                residual += preds[:, dim] # zero on forward pass, prev pred on backward pass -> undo contribution
                x = x_train[:, dim]
                try:
                    params, _ = curve_fit(
                        self._single_sine,
                        x,
                        residual,
                        p0=[1.0, 1.0, 0.0, 0.0],
                        bounds=([0, 0, -2 * np.pi, -np.inf], [np.inf, 20, 2 * np.pi, np.inf]),
                        maxfev=1000,
                    )
                    amp, freq, phase, offset = params
                except Exception:
                    amp, freq, phase, offset = 0.0, 0.0, 0.0, 0.0

                # Update stored params
                amps[dim] = amp
                freqs[dim] = freq
                phases[dim] = phase
                offsets[dim] = offset

                # Subtract this dimension's contribution
                pred = self._single_sine(x, amp, freq, phase, offset)
                residual -= pred
                preds[:, dim] = pred
        
        return amps, freqs, phases, offsets

    def predict(self, x, params):
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        amps, freqs, phases, offsets = params
        y_pred = np.sum(amps * np.sin(x * freqs + phases) + offsets, axis = 1)
        return torch.tensor(y_pred, dtype=torch.float32)

    def _call_single(self, xs, ys, inds, b):
        print(f"_call_single {b} called")
        n_points = xs.shape[0]
        if inds is None:
            inds = range(n_points)

        preds = np.zeros(n_points)
        for i in inds:
            if i == 0:
                preds[i] = 0.0
                continue
            x_train, y_train = xs[:i], ys[:i]
            x_test = xs[i]
            print(f"_call_single {b} fit call {i}")
            params = self.fit(x_train, y_train)
            print(f"_call_single {b} predict call {i}")
            preds[i] = self.predict(x_test, params).item()

        print(f"_call_single {b} returns")
        return i, torch.tensor(preds).unsqueeze(0)

    def __call__(self, xs, ys, inds=None):
        if isinstance(xs, torch.Tensor): xs = xs.cpu()
        if isinstance(ys, torch.Tensor): ys = ys.cpu()
        print(f"xs.shape = {xs.shape}\nys.shape = {ys.shape}")

        if xs.ndim == 3:
            i_s, results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._call_single)(xs[i], ys[i], inds, i)
                for i in range(xs.shape[0])
            )
            print(i_s)
            return torch.cat(results, dim=0)
        elif xs.ndim == 2:
            return self._call_single(self, xs, ys, inds)
        else:
            raise ValueError("Input xs must be 2D or 3D tensor.")


