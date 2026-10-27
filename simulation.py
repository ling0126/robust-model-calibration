# %% [markdown]
# Robust model calibration: Chapter 9.2 example and simulation extension
#
# Requires NumPy and Matplotlib. Run the cells in order.
# Calibration model, Eq. (9.12):
#     h(x; eta) = eta_0 + exp(-eta_1*x_1) + exp(-eta_2*x_2)

# %% Step 0: Setup
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

SEED = 20261013
rng = np.random.default_rng(SEED)

OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)


# %% Step 1: Calibration model and local design
def h(x, eta):
    """Equation (9.12). x shape=(n,2), eta=(eta0,eta1,eta2)."""
    x = np.asarray(x, dtype=float)
    return eta[0] + np.exp(-eta[1] * x[:, 0]) + np.exp(-eta[2] * x[:, 1])


def sensitivity(x, eta):
    """Gradient of h with respect to eta."""
    x = np.asarray(x, dtype=float)
    return np.column_stack(
        [
            np.ones(len(x)),
            -x[:, 0] * np.exp(-eta[1] * x[:, 0]),
            -x[:, 1] * np.exp(-eta[2] * x[:, 1]),
        ]
    )


def local_d_design(eta):
    """Four equal-weight points; boundary clipping is a practical approximation."""
    a = np.clip(1.0 / eta[1], 0.0, 1.0)
    b = np.clip(1.0 / eta[2], 0.0, 1.0)
    return np.array([[0, 0], [a, 0], [0, b], [a, b]], dtype=float)


eta_nominal = np.array([0.0, 2.0, 2.0])
local_four = local_d_design(eta_nominal)
design_local = np.repeat(local_four, 2, axis=0)  # 8 runs

print("Local D-optimal design (4 locations, each replicated twice):")
print(design_local)


# %% Step 2: Sample parameter uncertainty
# Parameter distributions from Section 9.2.1:
# eta_1 ~ Uniform(1,3), eta_2 ~ Normal(2, 0.5^2)
N_PARAMETER_SAMPLES = 500
eta1_samples = rng.uniform(1.0, 3.0, N_PARAMETER_SAMPLES)
eta2_samples = rng.normal(2.0, 0.5, N_PARAMETER_SAMPLES)

# Resample nonpositive decay parameters.
while np.any(eta2_samples <= 0):
    bad = eta2_samples <= 0
    eta2_samples[bad] = rng.normal(2.0, 0.5, bad.sum())

# Pool four local design points for each parameter draw.
local_pool = np.vstack(
    [local_d_design([0.0, e1, e2]) for e1, e2 in zip(eta1_samples, eta2_samples)]
)
pool_weights = np.full(len(local_pool), 1.0 / len(local_pool))

print("Number of parameter samples:", N_PARAMETER_SAMPLES)
print("Number of pooled local-design points:", len(local_pool))


# %% Step 3: Importance support points, Eq. (5.17)
def isp_objective_and_gradient(design, pool, weights):
    """Objective and gradient of equation (5.17)."""
    n = len(design)
    eps = 1e-10

    # Attraction: representative of the weighted target distribution.
    diff_pool = design[:, None, :] - pool[None, :, :]
    dist_pool = np.sqrt(np.sum(diff_pool**2, axis=2) + eps)
    attraction = (2.0 / n) * np.sum(dist_pool * weights[None, :])
    grad_attraction = (2.0 / n) * np.sum(
        weights[None, :, None] * diff_pool / dist_pool[:, :, None], axis=1
    )

    # Repulsion: prevent all representative points from collapsing together.
    diff_design = design[:, None, :] - design[None, :, :]
    dist_design = np.sqrt(np.sum(diff_design**2, axis=2) + eps)
    np.fill_diagonal(dist_design, np.inf)
    repulsion = np.sum(np.where(np.isfinite(dist_design), dist_design, 0.0)) / n**2
    grad_repulsion = -(2.0 / n**2) * np.sum(
        np.where(
            np.isfinite(dist_design)[:, :, None],
            diff_design / dist_design[:, :, None],
            0.0,
        ),
        axis=1,
    )

    objective = attraction - repulsion
    gradient = grad_attraction + grad_repulsion
    return objective, gradient


def importance_support_points(
    pool, weights, n_points, n_starts=6, steps=1800, learning_rate=0.025
):
    """Minimize (5.17) with projected Adam and several starting designs."""
    best_design, best_value = None, np.inf

    for _ in range(n_starts):
        ids = rng.choice(len(pool), n_points, replace=False, p=weights)
        design = pool[ids].copy()
        first_moment = np.zeros_like(design)
        second_moment = np.zeros_like(design)

        for step in range(1, steps + 1):
            value, gradient = isp_objective_and_gradient(design, pool, weights)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            m_hat = first_moment / (1.0 - 0.9**step)
            v_hat = second_moment / (1.0 - 0.999**step)
            rate = learning_rate / np.sqrt(1.0 + step / 500.0)
            design -= rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            design = np.clip(design, 0.0, 1.0)

        value, _ = isp_objective_and_gradient(design, pool, weights)
        if value < best_value:
            best_design, best_value = design.copy(), value

    order = np.lexsort((best_design[:, 1], best_design[:, 0]))
    return best_design[order]


# Figure 9.2 analogue: all 8 runs protect against parameter uncertainty.
design_parameter = importance_support_points(
    local_pool, pool_weights, n_points=8
)

# Figure 9.4 analogue: use 4 runs for parameter uncertainty.
design_parameter_four = importance_support_points(
    local_pool, pool_weights, n_points=4
)

print("Eight parameter-robust points:")
print(np.round(design_parameter, 4))


# %% Step 4: Sequential MaxPro augmentation, Eq. (4.27)
def logsumexp(values, axis=1):
    vmax = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(vmax, axis=axis) + np.log(
        np.sum(np.exp(values - vmax), axis=axis)
    )


def sequential_maxpro(existing, n_add, n_candidates=50_000):
    """Greedy candidate-set implementation of equation (4.27)."""
    design = existing.copy()
    candidates = rng.uniform(0.0, 1.0, size=(n_candidates, 2))

    for _ in range(n_add):
        delta = np.abs(candidates[:, None, :] - design[None, :, :])
        log_terms = -2.0 * np.sum(np.log(np.maximum(delta, 1e-12)), axis=2)
        scores = logsumexp(log_terms, axis=1)
        chosen = np.argmin(scores)
        design = np.vstack([design, candidates[chosen]])
        candidates = np.delete(candidates, chosen, axis=0)
    return design


design_combined = sequential_maxpro(design_parameter_four, n_add=4)
roles_combined = np.array(["parameter"] * 4 + ["space-filling"] * 4)

print("Combined robust design:")
print(np.round(design_combined, 4))


# %% Step 5: Plot the designs
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

axes[0].scatter(design_local[:, 0], design_local[:, 1], s=55)
axes[0].set_title("Local D-optimal\n(single nominal value)")

axes[1].scatter(design_parameter[:, 0], design_parameter[:, 1], s=55)
axes[1].set_title("Parameter-robust\n(8 ISP points)")

is_parameter = roles_combined == "parameter"
axes[2].scatter(
    design_combined[is_parameter, 0],
    design_combined[is_parameter, 1],
    s=55,
    label="parameter",
)
axes[2].scatter(
    design_combined[~is_parameter, 0],
    design_combined[~is_parameter, 1],
    s=65,
    marker="x",
    linewidths=2,
    label="space-filling",
)
axes[2].set_title("Parameter + model robust\n(4 ISP + 4 MaxPro)")
axes[2].legend(frameon=False)

for ax in axes:
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="$x_1$", ylabel="$x_2$")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")

fig.savefig(OUT / "design_comparison.png", dpi=180)
if plt.get_backend().lower() != "agg":
    plt.show()
else:
    plt.close(fig)


# %% Step 6: Simulated physical response
eta_true = np.array([0.30, 2.70, 1.40])
sigma_noise = 0.03


def discrepancy(x):
    """A smooth missing-physics effect near the upper-right region."""
    x = np.asarray(x)
    return 0.35 * np.exp(
        -((x[:, 0] - 0.82) ** 2 + (x[:, 1] - 0.82) ** 2) / (2 * 0.18**2)
    )


def physical_mean(x):
    return h(x, eta_true) + discrepancy(x)


# %% Step 7: Estimate calibration parameters
def fit_eta_gauss_newton(x, y, initial=eta_nominal, max_iter=100):
    eta = np.asarray(initial, dtype=float).copy()
    lower = np.array([-1.0, 0.2, 0.2])
    upper = np.array([1.0, 5.0, 5.0])

    for _ in range(max_iter):
        prediction = h(x, eta)
        residual = y - prediction
        s = sensitivity(x, eta)
        direction = np.linalg.solve(s.T @ s + 1e-6 * np.eye(3), s.T @ residual)

        # Backtracking prevents an overly large nonlinear update.
        old_sse = np.sum(residual**2)
        step_size = 1.0
        while step_size > 1e-5:
            candidate = np.clip(eta + step_size * direction, lower, upper)
            if np.sum((y - h(x, candidate)) ** 2) <= old_sse:
                break
            step_size *= 0.5

        if np.sum((y - h(x, candidate)) ** 2) > old_sse:
            break

        if np.linalg.norm(candidate - eta) < 1e-8:
            eta = candidate
            break
        eta = candidate
    return eta


# %% Step 8: Fit a GP to the residuals
def fit_residual_gp(x, residual, noise_sd):
    """Small grid search for GP length scales and process standard deviation."""
    ell_grid = np.geomspace(0.15, 2.0, 12)
    tau_grid = np.geomspace(0.02, 0.60, 10)
    best = None

    for ell1 in ell_grid:
        for ell2 in ell_grid:
            d1 = (x[:, None, 0] - x[None, :, 0]) / ell1
            d2 = (x[:, None, 1] - x[None, :, 1]) / ell2
            correlation = np.exp(-0.5 * (d1**2 + d2**2))

            for tau in tau_grid:
                k = tau**2 * correlation + (noise_sd**2 + 1e-8) * np.eye(len(x))
                sign, logdet = np.linalg.slogdet(k)
                if sign <= 0:
                    continue
                alpha = np.linalg.solve(k, residual)
                nll = 0.5 * (residual @ alpha + logdet)
                if best is None or nll < best[0]:
                    best = (nll, ell1, ell2, tau)

    _, ell1, ell2, tau = best
    return {"x": x, "residual": residual, "ell": (ell1, ell2),
            "tau": tau, "noise_sd": noise_sd}


def predict_residual_gp(gp, new_x):
    x = gp["x"]
    ell1, ell2 = gp["ell"]
    tau = gp["tau"]
    d1 = (x[:, None, 0] - x[None, :, 0]) / ell1
    d2 = (x[:, None, 1] - x[None, :, 1]) / ell2
    k = tau**2 * np.exp(-0.5 * (d1**2 + d2**2))
    k += (gp["noise_sd"] ** 2 + 1e-8) * np.eye(len(x))

    sd1 = (new_x[:, None, 0] - x[None, :, 0]) / ell1
    sd2 = (new_x[:, None, 1] - x[None, :, 1]) / ell2
    k_star = tau**2 * np.exp(-0.5 * (sd1**2 + sd2**2))
    alpha = np.linalg.solve(k, gp["residual"])
    return np.sum(k_star * alpha[None, :], axis=1)


def analyze_design(design, seed):
    local_rng = np.random.default_rng(seed)
    y = physical_mean(design) + local_rng.normal(0.0, sigma_noise, len(design))
    eta_hat = fit_eta_gauss_newton(design, y)
    residual = y - h(design, eta_hat)
    gp = fit_residual_gp(design, residual, sigma_noise)
    return {"y": y, "eta_hat": eta_hat, "residual": residual, "gp": gp}


designs = {
    "Local": design_local,
    "Parameter robust": design_parameter,
    "Combined robust": design_combined,
}
fits = {
    name: analyze_design(design, 101 + i)
    for i, (name, design) in enumerate(designs.items())
}


# %% Step 9: Evaluate predictions
axis = np.linspace(0.0, 1.0, 61)
x1_grid, x2_grid = np.meshgrid(axis, axis, indexing="xy")
test_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
truth = physical_mean(test_grid)

predictions = {}
rows = []
for name, design in designs.items():
    fit = fits[name]
    prediction = h(test_grid, fit["eta_hat"]) + predict_residual_gp(fit["gp"], test_grid)
    predictions[name] = prediction
    rows.append(
        [
            name,
            *fit["eta_hat"],
            np.linalg.norm(fit["eta_hat"] - eta_true),
            np.sqrt(np.mean((prediction - truth) ** 2)),
        ]
    )

print("\nTrue eta:", eta_true)
print("\nDesign                  eta0     eta1     eta2   parameter error   RMSE")
for row in rows:
    print(f"{row[0]:22s} {row[1]:7.3f} {row[2]:8.3f} {row[3]:8.3f}"
          f" {row[4]:16.3f} {row[5]:8.3f}")


# %% Step 10: Compare response surfaces
all_values = np.concatenate([truth, *predictions.values()])
vmin, vmax = all_values.min(), all_values.max()
fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)

image = axes[0, 0].imshow(
    truth.reshape(61, 61), origin="lower", extent=(0, 1, 0, 1),
    vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="equal"
)
axes[0, 0].set_title("True physical mean")

for ax, (name, prediction), row in zip(axes.flat[1:], predictions.items(), rows):
    ax.imshow(
        prediction.reshape(61, 61), origin="lower", extent=(0, 1, 0, 1),
        vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="equal"
    )
    ax.scatter(designs[name][:, 0], designs[name][:, 1], s=14, color="black")
    ax.set_title(f"{name}\nRMSE = {row[5]:.3f}")

for ax in axes.flat:
    ax.set(xlabel="$x_1$", ylabel="$x_2$")

fig.colorbar(image, ax=axes, shrink=0.75, label="response")
fig.savefig(OUT / "prediction_comparison.png", dpi=180)
if plt.get_backend().lower() != "agg":
    plt.show()
else:
    plt.close(fig)


# %% Step 11: Save results
with (OUT / "design_points.csv").open("w", encoding="utf-8") as file:
    file.write("design,role,run,x1,x2\n")
    for name, design in designs.items():
        for i, point in enumerate(design):
            if name == "Combined robust":
                role = "parameter" if i < 4 else "space_filling"
            elif name == "Parameter robust":
                role = "parameter"
            else:
                role = "local"
            file.write(f"{name},{role},{i + 1},{point[0]:.10f},{point[1]:.10f}\n")

with (OUT / "summary.csv").open("w", encoding="utf-8") as file:
    file.write("design,eta0_hat,eta1_hat,eta2_hat,parameter_error,prediction_rmse\n")
    for row in rows:
        file.write(",".join([row[0], *[f"{value:.10f}" for value in row[1:]]]) + "\n")

print("\nResults saved in:", OUT.resolve())

