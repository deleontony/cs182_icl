import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "sum_sine_regression": SumSineRegression,
        "radial_sine_regression": RadialSineRegression,
        "linear_sine_regression": LinearSineRegression,
        "linear_modulo_regression": LinearModuloRegression,
        "saw_regression": SawtoothRegression,
        "triangle_wave_regression":TriangleWaveRegression,
        "square_wave_regression":SquareWaveRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class SumSineRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1): 
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SumSineRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        #f(x[i]) = amp * sin(freq * x[i] + phase) + offset   for each i in n_dim
        if pool_dict is None and seeds is None:
            self.amp = torch.randn(self.b_size, 1, self.n_dims)
            self.freq = torch.randn(self.b_size, 1, self.n_dims)
            self.phase = torch.randn(self.b_size, 1, self.n_dims)
            self.offset = torch.randn(self.b_size, 1, self.n_dims)
        elif seeds is not None:
            self.amp = torch.zeros(self.b_size, 1, self.n_dims, 1)
            self.freq = torch.zeros(self.b_size, 1, self.n_dims, 1)
            self.phase = torch.zeros(self.b_size, 1, self.n_dims, 1)
            self.offset = torch.zeros(self.b_size, 1, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.amp[i] = torch.randn(1, self.n_dims, generator=generator)
                self.freq[i] = torch.randn(1, self.n_dims, generator=generator)
                self.phase[i] = torch.randn(1, self.n_dims, generator=generator)
                self.offset[i] = torch.randn(1, self.n_dims, generator=generator)
        else:
            assert all(k in pool_dict for k in ["amp", "freq", "phase", "offset"])
            indices = torch.randperm(len(pool_dict["amp"]))[:batch_size]
            self.amp = pool_dict["amp"][indices]
            self.freq = pool_dict["freq"][indices]
            self.phase = pool_dict["phase"][indices]
            self.offset = pool_dict["offset"][indices]
    
    def evaluate(self, xs_b):
        # xs_b: shape (B, T, D)
        amp = self.amp.to(xs_b.device)        # (B, 1, D)
        freq = self.freq.to(xs_b.device)      
        phase = self.phase.to(xs_b.device)
        offset = self.offset.to(xs_b.device)

        # print(f"xs_b.shape = {xs_b.shape}")
        # print(f"amp.shape = {amp.shape}")
        # print(f"freq.shape = {freq.shape}")
        # print(f"phase.shape = {phase.shape}")
        # print(f"offset.shape = {offset.shape}")

        fxs = amp * torch.sin(freq * xs_b + phase) + offset  # (B, T, D)
        # Sum across D (dim=2) to get scalar output per x
        ys = fxs.sum(dim=2).squeeze(-1)  # (B, T)
        return ys * self.scale
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {
            "amp": torch.randn(num_tasks, 1, n_dims),
            "freq": torch.randn(num_tasks, 1, n_dims),
            "phase": torch.randn(num_tasks, 1, n_dims),
            "offset": torch.randn(num_tasks, 1, n_dims),
        }
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class RadialSineRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.amp = torch.randn(batch_size, 1)
            self.freq = torch.randn(batch_size, 1)
            self.phase = torch.randn(batch_size, 1)
            self.offset = torch.randn(batch_size, 1)
        elif seeds is not None:
            self.amp = torch.zeros(batch_size, 1)
            self.freq = torch.zeros(batch_size, 1)
            self.phase = torch.zeros(batch_size, 1)
            self.offset = torch.zeros(batch_size, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.amp[i] = torch.randn(1, generator=generator)
                self.freq[i] = torch.randn(1, generator=generator)
                self.phase[i] = torch.randn(1, generator=generator)
                self.offset[i] = torch.randn(1, generator=generator)
        else:
            for k in ["amp", "freq", "phase", "offset"]:
                assert k in pool_dict
            idx = torch.randperm(len(pool_dict["amp"]))[:batch_size]
            self.amp = pool_dict["amp"][idx]
            self.freq = pool_dict["freq"][idx]
            self.phase = pool_dict["phase"][idx]
            self.offset = pool_dict["offset"][idx]

    def evaluate(self, xs_b):  # xs_b: (B, T, D)
        norms = xs_b.norm(dim=-1)  # (B, T)
        amp = self.amp.to(xs_b.device)
        freq = self.freq.to(xs_b.device)
        phase = self.phase.to(xs_b.device)
        offset = self.offset.to(xs_b.device)
        ys = amp * torch.sin(freq * norms + phase) + offset  # (B, T)
        return ys * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "amp": torch.randn(num_tasks, 1),
            "freq": torch.randn(num_tasks, 1),
            "phase": torch.randn(num_tasks, 1),
            "offset": torch.randn(num_tasks, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearSineRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w = torch.randn(batch_size, n_dims, 1)
            self.amp = torch.randn(batch_size, 1)
            self.phase = torch.randn(batch_size, 1)
            self.offset = torch.randn(batch_size, 1)
        elif seeds is not None:
            self.w = torch.zeros(batch_size, n_dims, 1)
            self.amp = torch.zeros(batch_size, 1)
            self.phase = torch.zeros(batch_size, 1)
            self.offset = torch.zeros(batch_size, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(n_dims, 1, generator=generator)
                self.amp[i] = torch.randn(1, generator=generator)
                self.phase[i] = torch.randn(1, generator=generator)
                self.offset[i] = torch.randn(1, generator=generator)
        else:
            idx = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][idx]
            self.amp = pool_dict["amp"][idx]
            self.phase = pool_dict["phase"][idx]
            self.offset = pool_dict["offset"][idx]

    def evaluate(self, xs_b):  # xs_b: (B, T, D)
        w = self.w.to(xs_b.device)           # (B, D, 1)
        amp = self.amp.to(xs_b.device)       # (B, 1)
        phase = self.phase.to(xs_b.device)   # (B, 1)
        offset = self.offset.to(xs_b.device) # (B, 1)

        dot = (xs_b @ w).squeeze(-1)         # (B, T)
        ys = amp * torch.sin(dot + phase) + offset  # (B, T)
        return ys * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "w": torch.randn(num_tasks, n_dims, 1),
            "amp": torch.randn(num_tasks, 1),
            "phase": torch.randn(num_tasks, 1),
            "offset": torch.randn(num_tasks, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearModuloRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w = torch.randn(batch_size, n_dims, 1)
            self.amp = torch.randn(batch_size, 1)
            self.phase = torch.randn(batch_size, 1)
            self.offset = torch.randn(batch_size, 1)
            self.divisor = torch.abs(torch.randn(batch_size, 1)) + 0.1  # ensure > 0
        elif seeds is not None:
            self.w = torch.zeros(batch_size, n_dims, 1)
            self.amp = torch.zeros(batch_size, 1)
            self.phase = torch.zeros(batch_size, 1)
            self.offset = torch.zeros(batch_size, 1)
            self.divisor = torch.zeros(batch_size, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(n_dims, 1, generator=generator)
                self.amp[i] = torch.randn(1, generator=generator)
                self.phase[i] = torch.randn(1, generator=generator)
                self.offset[i] = torch.randn(1, generator=generator)
                self.divisor[i] = torch.abs(torch.randn(1, generator=generator)) + 0.1
        else:
            for k in ["w", "amp", "phase", "offset", "divisor"]:
                assert k in pool_dict
            idx = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][idx]
            self.amp = pool_dict["amp"][idx]
            self.phase = pool_dict["phase"][idx]
            self.offset = pool_dict["offset"][idx]
            self.divisor = pool_dict["divisor"][idx]

    def evaluate(self, xs_b):  # xs_b: (B, T, D)
        w = self.w.to(xs_b.device)             # (B, D, 1)
        amp = self.amp.to(xs_b.device)         # (B, 1)
        phase = self.phase.to(xs_b.device)     # (B, 1)
        offset = self.offset.to(xs_b.device)   # (B, 1)
        divisor = self.divisor.to(xs_b.device) # (B, 1)

        dot = (xs_b @ w).squeeze(-1)  # (B, T)
        mod_result = (dot + phase) % divisor  # (B, T)

        ys = amp * mod_result + offset  # (B, T)
        return ys * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "w": torch.randn(num_tasks, n_dims, 1),
            "amp": torch.randn(num_tasks, 1),
            "phase": torch.randn(num_tasks, 1),
            "offset": torch.randn(num_tasks, 1),
            "divisor": torch.abs(torch.randn(num_tasks, 1)) + 0.1,  # ensure > 0
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SawtoothRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w = torch.randn(batch_size, n_dims, 1)
            self.amp = torch.randn(batch_size, 1)
            self.phase = torch.randn(batch_size, 1)
            self.offset = torch.randn(batch_size, 1)
            self.divisor = torch.abs(torch.randn(batch_size, 1)) + 0.1
        elif seeds is not None:
            self.w = torch.zeros(batch_size, n_dims, 1)
            self.amp = torch.zeros(batch_size, 1)
            self.phase = torch.zeros(batch_size, 1)
            self.offset = torch.zeros(batch_size, 1)
            self.divisor = torch.zeros(batch_size, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(n_dims, 1, generator=generator)
                self.amp[i] = torch.randn(1, generator=generator)
                self.phase[i] = torch.randn(1, generator=generator)
                self.offset[i] = torch.randn(1, generator=generator)
                self.divisor[i] = torch.abs(torch.randn(1, generator=generator)) + 0.1
        else:
            idx = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][idx]
            self.amp = pool_dict["amp"][idx]
            self.phase = pool_dict["phase"][idx]
            self.offset = pool_dict["offset"][idx]
            self.divisor = pool_dict["divisor"][idx]

    def evaluate(self, xs_b):  # (B, T, D)
        w = self.w.to(xs_b.device)
        amp = self.amp.to(xs_b.device)
        phase = self.phase.to(xs_b.device)
        offset = self.offset.to(xs_b.device)
        divisor = self.divisor.to(xs_b.device)

        dot = (xs_b @ w).squeeze(-1)  # (B, T)
        saw = ((dot + phase) % divisor) / divisor

        ys = amp * saw + offset
        return ys * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "w": torch.randn(num_tasks, n_dims, 1),
            "amp": torch.randn(num_tasks, 1),
            "phase": torch.randn(num_tasks, 1),
            "offset": torch.randn(num_tasks, 1),
            "divisor": torch.abs(torch.randn(num_tasks, 1)) + 0.1,
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    
class TriangleWaveRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w = torch.randn(batch_size, n_dims, 1)
            self.amp = torch.randn(batch_size, 1)
            self.phase = torch.randn(batch_size, 1)
            self.offset = torch.randn(batch_size, 1)
            self.divisor = torch.abs(torch.randn(batch_size, 1)) + 0.1
        elif seeds is not None:
            self.w = torch.zeros(batch_size, n_dims, 1)
            self.amp = torch.zeros(batch_size, 1)
            self.phase = torch.zeros(batch_size, 1)
            self.offset = torch.zeros(batch_size, 1)
            self.divisor = torch.zeros(batch_size, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(n_dims, 1, generator=generator)
                self.amp[i] = torch.randn(1, generator=generator)
                self.phase[i] = torch.randn(1, generator=generator)
                self.offset[i] = torch.randn(1, generator=generator)
                self.divisor[i] = torch.abs(torch.randn(1, generator=generator)) + 0.1
        else:
            idx = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][idx]
            self.amp = pool_dict["amp"][idx]
            self.phase = pool_dict["phase"][idx]
            self.offset = pool_dict["offset"][idx]
            self.divisor = pool_dict["divisor"][idx]

    def evaluate(self, xs_b):  # xs_b: (B, T, D)
        w = self.w.to(xs_b.device)             # (B, D, 1)
        amp = self.amp.to(xs_b.device)         # (B, 1)
        phase = self.phase.to(xs_b.device)     # (B, 1)
        offset = self.offset.to(xs_b.device)   # (B, 1)
        divisor = self.divisor.to(xs_b.device) # (B, 1)

        dot = (xs_b @ w).squeeze(-1) + phase   # (B, T)
        divisor_exp = divisor.expand_as(dot)  # (B, T)

        triangle = (2 / divisor_exp) * torch.abs((dot - divisor_exp / 2) % divisor_exp - divisor_exp / 2)
        ys = amp * triangle + offset           # (B, T)

        return ys * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "w": torch.randn(num_tasks, n_dims, 1),
            "amp": torch.randn(num_tasks, 1),
            "phase": torch.randn(num_tasks, 1),
            "offset": torch.randn(num_tasks, 1),
            "divisor": torch.abs(torch.randn(num_tasks, 1)) + 0.1,
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class SquareWaveRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w = torch.randn(batch_size, n_dims, 1)
            self.amp = torch.randn(batch_size, 1)
            self.phase = torch.randn(batch_size, 1)
            self.offset = torch.randn(batch_size, 1)
            self.divisor = torch.abs(torch.randn(batch_size, 1)) + 0.1
        elif seeds is not None:
            self.w = torch.zeros(batch_size, n_dims, 1)
            self.amp = torch.zeros(batch_size, 1)
            self.phase = torch.zeros(batch_size, 1)
            self.offset = torch.zeros(batch_size, 1)
            self.divisor = torch.zeros(batch_size, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w[i] = torch.randn(n_dims, 1, generator=generator)
                self.amp[i] = torch.randn(1, generator=generator)
                self.phase[i] = torch.randn(1, generator=generator)
                self.offset[i] = torch.randn(1, generator=generator)
                self.divisor[i] = torch.abs(torch.randn(1, generator=generator)) + 0.1
        else:
            idx = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w = pool_dict["w"][idx]
            self.amp = pool_dict["amp"][idx]
            self.phase = pool_dict["phase"][idx]
            self.offset = pool_dict["offset"][idx]
            self.divisor = pool_dict["divisor"][idx]

    def evaluate(self, xs_b):  # (B, T, D)
        w = self.w.to(xs_b.device)
        amp = self.amp.to(xs_b.device)
        phase = self.phase.to(xs_b.device)
        offset = self.offset.to(xs_b.device)
        divisor = self.divisor.to(xs_b.device)

        dot = (xs_b @ w).squeeze(-1)
        square = torch.sign(torch.sin((2 * math.pi * (dot + phase)) / divisor))
        ys = amp * square + offset
        return ys * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "w": torch.randn(num_tasks, n_dims, 1),
            "amp": torch.randn(num_tasks, 1),
            "phase": torch.randn(num_tasks, 1),
            "offset": torch.randn(num_tasks, 1),
            "divisor": torch.abs(torch.randn(num_tasks, 1)) + 0.1,
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    