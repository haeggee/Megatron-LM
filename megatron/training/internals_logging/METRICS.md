# Model Internals Logging - Metric Definitions

This document provides precise mathematical definitions for all metrics captured by the internals logging system.

**Notation.** We write $G := \nabla_\theta \mathcal{L}$ for the gradient of the loss with respect to a parameter $\theta$.

---

## 1. Activation Statistics

**Logged as:** `activations/layer_{N}/<metric>`

| Metric | Definition |
|--------|------------|
| `mean` | $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ |
| `std` | $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$ |
| `min` | $\min(X)$ |
| `max` | $\max(X)$ |
| `kurtosis` | $\kappa = \frac{E[(X-\mu)^4]}{\sigma^4} - 3$ &nbsp;&nbsp; *(excess kurtosis; 0 for normal distribution)* |

Where $X$ is the flattened activation tensor from the layer output.

---

## 2. Gradient Statistics

### Per-Parameter Norms

**Logged as:** `gradients/<param_name>/norm`

$$\|G\|_2 = \sqrt{\sum_i G_i^2}$$

### Per-Layer Aggregates

**Logged as:** `gradients_per_layer/layer_{N}/<metric>`

| Metric | Definition |
|--------|------------|
| `total_norm` | $\sqrt{\sum_{j \in \text{layer}_N} \|G_j\|_2^2}$ |
| `avg_norm` | $\frac{1}{m}\sum_{j \in \text{layer}_N} \|G_j\|_2$ |
| `max_norm` | $\max_{j \in \text{layer}_N} \|G_j\|_2$ |

### Gradient Flow (Between Layers)

**Logged as:** `gradient_flow/layer_{N}_to_{N+1}`

$$\text{flow}_{N \to N+1} = \frac{\text{total\_norm}_{N+1}}{\text{total\_norm}_N}$$

Values $< 1$ indicate vanishing gradients; values $> 1$ indicate exploding gradients.

---

## 3. Relative Weight Updates (δW)

### Per-Parameter

**Logged as:** `delta_W/<param_name>`

$$\delta_W = \frac{\|W_t - W_{t-1}\|_2}{\|W_{t-1}\|_2}$$

### Per-Neuron Statistics

**Logged as:** `delta_W_per_neuron/<param_name>/<metric>`

For a weight matrix $W \in \mathbb{R}^{m \times n}$, compute per-row (per-neuron) relative changes:

$$\delta_{W,j} = \frac{\|W_{t,j} - W_{t-1,j}\|_2}{\|W_{t-1,j}\|_2} \quad \text{for row } j$$

| Metric | Definition |
|--------|------------|
| `mean` | $\frac{1}{m}\sum_{j=1}^{m} \delta_{W,j}$ |
| `std` | $\sqrt{\frac{1}{m}\sum_{j=1}^{m}(\delta_{W,j} - \bar{\delta}_W)^2}$ |
| `max` | $\max_j(\delta_{W,j})$ |
| `min` | $\min_j(\delta_{W,j})$ |

### Per-Layer Aggregates

**Logged as:** `delta_W_avg/layer_{N}`, `delta_W_max/layer_{N}`

---

## 4. Angular Metrics

These metrics measure directional changes in weight space, particularly useful for spherical/normalized training methods (e.g., nGPT, EDM2).

### Per-Parameter

**Logged as:** `angular/<param_name>/<metric>`

| Metric | Definition |
|--------|------------|
| `cos_similarity` | $\cos(\theta) = \frac{W_t \cdot W_{t-1}}{\|W_t\|_2 \cdot \|W_{t-1}\|_2}$ |
| `degrees` | $\theta = \arccos(\cos(\theta)) \times \frac{180}{\pi}$ |

**Interpretation:**
- $\cos(\theta) \approx 1$: weight direction unchanged
- $\cos(\theta) \approx 0$: orthogonal (90° change)
- $\cos(\theta) \approx -1$: opposite direction (180° change)

### Per-Layer Aggregates

**Logged as:** `angular_avg/layer_{N}/<metric>`

| Metric | Definition |
|--------|------------|
| `degrees` | Average angular change across all parameters in layer |
| `max_degrees` | Maximum angular change in layer |
| `cos_similarity` | Average cosine similarity across layer |

---

## 5. Gradient-Weight Alignment

These metrics decompose the gradient into components parallel and perpendicular to the current weight vector.

### Per-Parameter

**Logged as:** `grad_weight_align/<param_name>/<metric>`

| Metric | Definition |
|--------|------------|
| `cos` | $\cos(G, W) = \frac{G \cdot W}{\|G\|_2 \cdot \|W\|_2}$ |
| `radial` | $r = \|G\|_2 \cdot \cos(G, W)$ |
| `tangential` | $t = \|G\|_2 \cdot \sqrt{1 - \cos^2(G, W)}$ |

### Geometric Interpretation

```
                    G (gradient)
                   /|
                  / |
                 /  | tangential (t)
                /   |
               /    |
              /_____|___________ W (weight direction)
                radial (r)
```

**For spherical/normalized training:**
- **Tangential component**: Updates that change weight *direction* while preserving norm
- **Radial component**: Updates that change weight *magnitude*
- High tangential + low radial → norm-preserving directional updates

### Per-Layer Aggregates

**Logged as:** `grad_weight_align_avg/layer_{N}/<metric>`

| Metric | Definition |
|--------|------------|
| `cos` | Average cosine alignment across layer parameters |
| `radial` | Average radial component across layer |
| `tangential` | Average tangential component across layer |

---

## Summary

| Category | Metrics per Layer | W&B Prefix |
|----------|-------------------|------------|
| Activation Statistics | 5 | `activations/layer_{N}/` |
| Gradient Statistics (per-param) | per-param | `gradients/` |
| Gradient Statistics (per-layer) | 3 | `gradients_per_layer/` |
| Gradient Flow | 1 (per layer pair) | `gradient_flow/` |
| Weight Updates (per-param) | per-param | `delta_W/`, `delta_W_per_neuron/` |
| Weight Updates (per-layer) | 5 | `delta_W_avg/`, `delta_W_max/`, `delta_W_per_neuron_avg/` |
| Angular Metrics (per-param) | per-param | `angular/` |
| Angular Metrics (per-layer) | 3 | `angular_avg/` |
| Gradient-Weight Alignment (per-param) | per-param | `grad_weight_align/` |
| Gradient-Weight Alignment (per-layer) | 3 | `grad_weight_align_avg/` |

---

## Performance Notes

### GPU Memory vs Transfer Overhead

By default, previous weights are stored on CPU to conserve GPU memory. This causes GPU↔CPU transfers on logging iterations, which can introduce throughput variance.

Use `--internals-weights-on-gpu` to keep previous weights on GPU:
- **Pros**: Eliminates PCIe transfer overhead and throughput variance
- **Cons**: Uses additional GPU memory (~1x model parameter size)

Recommended when:
- You have sufficient GPU memory headroom
- Consistent iteration timing is important for profiling
- Using `--log-relative-updates` or `--log-angular-metrics`
