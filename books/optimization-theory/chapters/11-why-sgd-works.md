# Chapter 11: Why SGD Works

We've established: the loss landscape is non-convex, full of saddle points, and high-dimensional. Classical optimization theory says this should be impossible. Yet SGD works remarkably well. This chapter explains why.

## The Puzzle

### What Theory Predicts

Classical optimization tells us:
- Non-convex → can get stuck in bad local minima
- Stochastic gradients → noisy, imprecise updates
- High dimension → exponentially many critical points
- No second-order info → slow convergence

**Prediction**: SGD should fail catastrophically.

### What Actually Happens

- SGD trains billion-parameter models
- Different random seeds give similar results
- Solutions generalize well
- Training is surprisingly fast

**Reality**: SGD works better than it has any right to.

## The Components of Success

### 1. The Noise Is a Feature, Not a Bug

#### Escaping Saddle Points

SGD's stochastic noise helps escape saddle points (Chapter 8).

At a saddle with gradient ~0:
- True gradient gives no escape direction
- Stochastic gradient has components along escape directions
- Noise "kicks" the optimizer toward descent

```python
import torch
import torch.nn as nn
from typing import List, Tuple

def sgd_escapes_saddles_demo():
    """Compare GD and SGD near a saddle point."""
    torch.manual_seed(42)

    # High-dimensional saddle: half eigenvalues positive, half negative
    n = 100
    eigenvalues = torch.linspace(-1, 1, n)  # Symmetric around 0

    def loss(x):
        return 0.5 * (eigenvalues * x**2).sum()

    def true_gradient(x):
        return eigenvalues * x

    # Start exactly at saddle
    x_gd = torch.zeros(n)
    x_sgd = torch.zeros(n)

    lr = 0.1
    noise_std = 0.01

    gd_losses = []
    sgd_losses = []

    for step in range(500):
        # GD: uses true gradient
        g_gd = true_gradient(x_gd)
        x_gd = x_gd - lr * g_gd
        gd_losses.append(loss(x_gd).item())

        # SGD: adds noise (simulating minibatch variance)
        g_sgd = true_gradient(x_sgd) + noise_std * torch.randn(n)
        x_sgd = x_sgd - lr * g_sgd
        sgd_losses.append(loss(x_sgd).item())

    print(f"GD final loss: {gd_losses[-1]:.6f} (stuck at saddle)")
    print(f"SGD final loss: {sgd_losses[-1]:.6f} (escaped!)")
```

#### Implicit Regularization

SGD noise biases toward "flat" minima—solutions where the loss doesn't change much with small parameter perturbations.

Flat minima tend to generalize better because:
- Test data is slightly different from training data
- Flat solutions are robust to this distribution shift

```python
def sharpness_comparison():
    """Compare sharpness of minima found by different methods."""
    torch.manual_seed(42)

    X = torch.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    def measure_sharpness(model, X, y, epsilon=0.01):
        """Measure average loss increase under random perturbation."""
        criterion = nn.MSELoss()
        base_loss = criterion(model(X), y).item()

        perturbed_losses = []
        for _ in range(10):
            # Perturb parameters
            with torch.no_grad():
                original_params = {name: p.clone() for name, p in model.named_parameters()}
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * epsilon)

                perturbed_loss = criterion(model(X), y).item()
                perturbed_losses.append(perturbed_loss)

                # Restore
                for name, p in model.named_parameters():
                    p.data = original_params[name]

        return sum(perturbed_losses) / len(perturbed_losses) - base_loss

    # Train with large batch (low noise) vs small batch (high noise)
    results = []

    for batch_size, name in [(800, "Large batch (low noise)"), (32, "Small batch (high noise)")]:
        model = nn.Sequential(nn.Linear(20, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        for epoch in range(100):
            perm = torch.randperm(800)
            for i in range(0, 800, batch_size):
                batch_X = X_train[perm[i:i+batch_size]]
                batch_y = y_train[perm[i:i+batch_size]]

                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

        train_loss = criterion(model(X_train), y_train).item()
        test_loss = criterion(model(X_test), y_test).item()
        sharpness = measure_sharpness(model, X_train, y_train)

        results.append((name, train_loss, test_loss, sharpness))

    print("Method               | Train Loss | Test Loss | Sharpness")
    print("-" * 60)
    for name, train, test, sharp in results:
        print(f"{name:20} | {train:.6f}   | {test:.6f}  | {sharp:.6f}")
```

### 2. Overparameterization Helps

#### More Parameters = Easier Optimization

Counterintuitively, more parameters make optimization easier:

```python
def overparameterization_landscape():
    """Show how overparameterization smooths the loss landscape."""
    torch.manual_seed(42)

    X = torch.randn(50, 10)
    y = torch.randn(50, 1)

    results = []

    for width in [10, 50, 200, 1000]:
        model = nn.Sequential(
            nn.Linear(10, width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )

        # Measure initial loss variance across random initializations
        init_losses = []
        for seed in range(20):
            torch.manual_seed(seed)
            for p in model.parameters():
                nn.init.normal_(p, 0, 0.1)
            loss = nn.MSELoss()(model(X), y).item()
            init_losses.append(loss)

        init_var = torch.tensor(init_losses).var().item()

        # Train and measure success rate
        successes = 0
        for seed in range(20):
            torch.manual_seed(seed)
            for p in model.parameters():
                nn.init.normal_(p, 0, 0.1)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(500):
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(X), y)
                loss.backward()
                optimizer.step()

            if nn.MSELoss()(model(X), y).item() < 0.01:
                successes += 1

        results.append((width, init_var, successes/20))

    print("Width | Init Loss Variance | Training Success Rate")
    print("-" * 50)
    for width, var, success in results:
        print(f"{width:5d} | {var:.4f}             | {success:.0%}")
```

#### The Interpolation Regime

With more parameters than data points, the network can perfectly fit the training data.

In this regime:
- Many global minima exist (all achieving zero training loss)
- SGD can find one relatively easily
- The question becomes: which minimum generalizes best?

### 3. The Loss Landscape Has Structure

#### Saddle Points Are Easy to Escape

As shown in Chapter 8:
- High-loss critical points have high index (many escape directions)
- Low-loss critical points have low index (nearly minima)
- SGD naturally flows toward low-loss regions

#### Mode Connectivity

As shown in Chapter 10:
- Good solutions are connected
- There aren't isolated bad local minima
- SGD stays in the good basin once it finds it

### 4. Initialization Matters

#### Breaking Symmetry

Proper initialization:
- Breaks symmetry between neurons
- Avoids vanishing/exploding gradients
- Starts in a "good" basin

```python
def initialization_importance():
    """Show how initialization affects training."""
    torch.manual_seed(42)

    X = torch.randn(500, 20)
    y = (X[:, 0] + X[:, 1] > 0).long()

    results = []

    for init_name, init_scale in [("Too small", 0.001), ("Good", 0.1), ("Too large", 10.0)]:
        model = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

        with torch.no_grad():
            for p in model.parameters():
                p.data = torch.randn_like(p) * init_scale

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        results.append((init_name, losses[0], losses[-1]))

    print("Initialization | Initial Loss | Final Loss")
    print("-" * 45)
    for name, init, final in results:
        print(f"{name:14} | {init:.4f}       | {final:.4f}")
```

### 5. Architecture Helps Optimization

#### Skip Connections

ResNets with skip connections have smoother loss landscapes:
- Gradients flow directly through skip connections
- Avoids vanishing gradient problem
- The loss landscape has fewer sharp regions

```python
def skip_connection_effect():
    """Compare optimization with and without skip connections."""
    torch.manual_seed(42)

    X = torch.randn(500, 20)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

    # Plain network (deep)
    class PlainNet(nn.Module):
        def __init__(self, depth=10):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(20, 20) for _ in range(depth)])
            self.output = nn.Linear(20, 1)

        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return torch.sigmoid(self.output(x))

    # ResNet (same depth)
    class ResNet(nn.Module):
        def __init__(self, depth=10):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(20, 20) for _ in range(depth)])
            self.output = nn.Linear(20, 1)

        def forward(self, x):
            for layer in self.layers:
                x = x + torch.relu(layer(x))  # Skip connection!
            return torch.sigmoid(self.output(x))

    results = []
    for NetClass, name in [(PlainNet, "Plain"), (ResNet, "ResNet")]:
        model = NetClass(depth=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        losses = []
        for epoch in range(200):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        results.append((name, losses[0], losses[-1], min(losses)))

    print("Architecture | Initial Loss | Final Loss | Best Loss")
    print("-" * 55)
    for name, init, final, best in results:
        print(f"{name:12} | {init:.4f}       | {final:.4f}     | {best:.4f}")
```

## The Complete Picture

### SGD's Implicit Bias

SGD doesn't just find any minimum—it finds specific kinds of minima:

1. **Low sharpness**: Noise pushes away from sharp regions
2. **Low complexity**: Simple solutions are reached faster
3. **Good generalization**: These properties correlate with generalization

```python
def implicit_bias_demo():
    """Demonstrate SGD's implicit bias toward simple solutions."""
    torch.manual_seed(42)

    # Data that can be fit by simple or complex solutions
    X = torch.randn(100, 2)
    y = (X[:, 0] > 0).float().unsqueeze(1)  # Simple linear boundary

    # Network that could implement complex solutions
    model = nn.Sequential(
        nn.Linear(2, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()

    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    # Check: does SGD find the simple solution?
    # Sample predictions across a grid
    grid_x = torch.linspace(-3, 3, 50)
    grid_y = torch.linspace(-3, 3, 50)
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        preds = model(grid).squeeze()

    # The decision boundary should be approximately x > 0
    # Check how many predictions match the simple rule
    simple_preds = (grid[:, 0] > 0).float()
    agreement = (preds.round() == simple_preds).float().mean()

    print(f"Agreement with simple linear rule: {agreement:.1%}")
```

### The Training Trajectory

SGD's path through parameter space:

1. **Early training**: Rapid descent, escape saddles
2. **Middle training**: Approach basin of good solutions
3. **Late training**: Settle into specific minimum

Each phase has different dynamics and benefits from different learning rates.

## Summary: Why It All Works

| Factor | Effect |
|--------|--------|
| Stochastic noise | Escapes saddles, regularizes |
| Overparameterization | Smoother landscape, easy interpolation |
| High-dimensional geometry | Saddles escape easily |
| Mode connectivity | No isolated bad minima |
| Careful initialization | Start in good basin |
| Architecture design | Skip connections smooth landscape |
| Implicit bias | SGD prefers simple solutions |

## The Deep Learning Optimization Recipe

1. **Use SGD with momentum** (or Adam)—noise + acceleration
2. **Initialize carefully** (Xavier/He)
3. **Use skip connections** for deep networks
4. **Overparameterize**—more parameters than data points
5. **Train long enough**—early stopping is optional
6. **Use learning rate schedules**—high early, low late

## What's Next

Part III applies these insights to practical optimizers:
- **Chapter 12**: Momentum and acceleration
- **Chapter 13**: Adaptive learning rates (Adam)
- **Chapters 14-17**: Second-order approximations

## Key Takeaways

1. **SGD works because of noise**, not despite it

2. **The loss landscape is friendlier than theory suggests**

3. **Overparameterization helps** optimization

4. **Architecture matters** for optimization, not just expressiveness

5. **Implicit bias** guides SGD toward good solutions

6. **The training trajectory matters**—different phases, different learning rates

## Exercises

1. **Noise level study**: How does SGD success rate depend on minibatch size?

2. **Width vs depth**: Which is better for optimization—wider or deeper networks?

3. **Initialization grid search**: What range of initializations leads to successful training?

4. **Decision boundary complexity**: Measure the complexity of the learned decision boundary for different training configurations.
