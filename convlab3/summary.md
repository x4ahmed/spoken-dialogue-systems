# PPO Dialogue Policy — Summary & Explanations

## Background Concepts

### What is a Dialogue Policy?

In a task-oriented dialogue system, the **policy** is the component that decides what the system should say next. It takes the current dialogue state (what the user wants, what information has been collected so far) and outputs a system action (e.g., "ask for cuisine", "recommend a restaurant", "book a table"). A good policy leads to successful, efficient dialogues; a bad policy wastes turns or fails to complete the task.

### What is MLE?

**Maximum Likelihood Estimation (MLE)** is a supervised learning approach. The policy network is trained on human dialogue transcripts — it learns to predict the same actions that humans took in similar situations. The model is trained by minimizing the difference between its predicted actions and the ground-truth human actions (binary cross-entropy loss). MLE learns to **imitate** human behavior but doesn't directly optimize for task success.

### What is PPO?

**Proximal Policy Optimization (PPO)** is a reinforcement learning algorithm. Instead of imitating humans, the policy learns by **trial and error**: it interacts with a simulated user, receives rewards (success = +80, failure = -40, minus turn penalty), and updates its strategy to maximize long-term reward. PPO uses a clipped objective to ensure the policy doesn't change too drastically in a single update, keeping training stable.

### Model Structure

Both MLE and PPO use the same network architecture:

- **Policy network** (`MultiDiscretePolicy`): A 3-layer MLP (input → 100 hidden units → 84 output units). Each output unit represents a dialogue action (e.g., "inform food", "request area"). The sigmoid output allows multiple actions per turn (multi-discrete).
- **Value network** (`Value`, PPO only): A 3-layer MLP (input → 50 hidden units → 1 output). Estimates the expected future reward from a given state. Used to compute advantages in GAE.
- **Input**: A 152-dimensional vector representing the dialogue state (user actions, system actions, belief state, database results).
- **Vectorizer** (`VectorBinary`): Converts between semantic dialogue state (dicts) and vectors.

### How MLE is Trained

1. Load human dialogue transcripts from the simplemultiwoz21 dataset
2. For each dialogue turn, extract (state, human action) pairs
3. Train the policy network with binary cross-entropy loss to predict human actions
4. Evaluate with precision, recall, F1 (action prediction accuracy)
5. No user simulator, no rewards — pure imitation

### How PPO is Trained

1. Initialize the policy network from the MLE-pretrained weights
2. For each epoch:
   - Collect dialogues by interacting with a rule-based user simulator
   - Record (state, action, reward, mask) for each turn
   - Compute GAE advantages (Task 1) using the value network
   - Update the policy using the PPO clip objective (Task 2)
   - Update the value network using MSE loss
3. Evaluate every 5 epochs against the same simulator (success rate, complete rate, return)

### Key Metrics

| Metric | Description | Better Performance |
|--------|-------------|-------------------|
| **Success Rate** | Fraction of dialogues where all task goals were fully achieved | ↑ Higher is better |
| **Complete Rate** | Fraction of dialogues where the user simulator considered the task complete | ↑ Higher is better |
| **Avg Return** | Average cumulative reward per dialogue (+80 success, -40 failure, minus turns) | ↑ Higher is better |
| **Avg Turns** | Average number of dialogue turns per dialogue | ↓ Lower is better |
| **Avg Actions** | Average number of system actions per turn | Moderate is better |

---

## GAE (Generalized Advantage Estimation) — Summary

### Why we need GAE

In RL, the **advantage** A(s,a) tells us: "was taking action a in state s better or worse than expected?" If positive, the action was good; if negative, it was bad. PPO uses advantages to scale policy updates — good actions get reinforced, bad ones discouraged.

The problem: we don't know the true advantage. We estimate it. **GAE** balances two extremes:
- **TD(0)**: uses only 1-step lookahead — low variance but biased
- **Monte Carlo**: uses full episode return — unbiased but high variance

GAE uses a parameter λ (called `tau` in the code) to interpolate between them.

### Line-by-line explanation

```python
for t in reversed(range(batchsz)):
```
Iterate **backwards** — from the last step to the first. This is because each step's advantage depends on the **next** step's advantage, so we compute the future first.

```python
v_target[t] = r[t] + self.gamma * mask[t] * prev_v_target
```
**Value target** — what the value network should predict. It's the Bellman equation: current reward + discounted future value. `prev_v_target` holds V(s_{t+1}) (already computed since we go backwards).

```python
delta[t] = r[t] + self.gamma * mask[t] * prev_v - v[t]
```
**TD error** — the difference between the bootstrapped value estimate (r_t + γ·V(s_{t+1})) and the critic's current prediction (V(s_t)). This is the 1-step advantage. `prev_v` = V(s_{t+1}), `v[t]` = V(s_t).

```python
A_sa[t] = delta[t] + self.gamma * self.tau * mask[t] * prev_A_sa
```
**GAE advantage** — the TD error plus a discounted, lambda-weighted sum of future TD errors. `prev_A_sa` = A_{t+1} (already computed). `tau` (λ) controls how far into the future we look: λ=1 → Monte Carlo, λ=0 → pure TD(0).

```python
prev_v_target = v_target[t]
prev_v = v[t]
prev_A_sa = A_sa[t]
```
Store current values to use as "next step" values when processing t-1.

### What is `mask`?

`mask[t] = 0` means step t is the **last step of a dialogue** (trajectory boundary). When mask=0, all future terms are zeroed out — there is no "next state" after the dialogue ends. This prevents rewards from one dialogue leaking into another.

### What is `v_target`?

The **target** the critic (value network) tries to predict. It's the discounted sum of future rewards. The critic is trained with MSE loss against this target — better predictions → better advantage estimates → better policy updates.

### What is `delta`?

The **TD error** — a 1-step measure of "how much better was reality than expected?" If the critic predicted V(s_t)=5 but the actual reward + next value was 7, then delta=2 (the action was better than expected).

### How it all fits together

```
delta (1-step error) → A_sa (GAE, multi-step) → PPO clip update
     ↑                        ↑                        ↑
  uses V(s) and V(s+1)   uses delta + future A   scales policy gradient
```

---

## Critic, Actual Returns, and Ground Truth — Simple Explanation

### The Setup

During training, the PPO policy talks to a **rule-based user simulator** (not a human). The simulator:
1. Has a fixed goal (e.g., "find a cheap Italian restaurant in the south")
2. Responds to the system's actions following simple rules
3. Gives a **reward** at the end of the dialogue: +80 if the goal was achieved, -40 if not, minus 1 per turn

### What is the "Actual Return"?

The **actual return** is the real reward the policy got from the simulator. For a 10-turn successful dialogue:

```
Turn 1: reward = 0
Turn 2: reward = 0
...
Turn 10: reward = +80 - 10 = +70  (success bonus minus turn penalty)
```

The actual return is the ground truth — it's what actually happened.

### What is the "Critic"?

The **critic** (value network) is a neural network that tries to **predict** the return *before* it happens. Given a state, it guesses: "from here, I expect to get ~15 reward."

The critic is like a coach watching a chess game and predicting "you're going to win" before the game is over.

### Why Do We Need Both?

- **Actual returns** are accurate but only available **after** the dialogue ends — too late to make decisions during the dialogue
- **Critic predictions** are available immediately but might be **wrong** (it's just a guess)

GAE combines them: use the critic for early steps (where we can't see the future), and correct it with actual rewards as they come in. The **TD error (delta)** measures how wrong the critic was:

```
delta = actual_reward + critic's_prediction_of_next_state - critic's_prediction_of_current_state
```

If delta is positive → the situation was better than the critic expected → the action was good.

### What is the Ground Truth?

There is **no labeled ground truth** like in supervised learning. The "ground truth" is the **reward from the simulator**:

- Did the dialogue achieve the user's goal? → +80 or -40
- How many turns did it take? → -1 per turn

The simulator checks this by comparing what the system did against the user's hidden goal (e.g., did it inform the right restaurant? did it book correctly?).

### Summary

| Concept | What it means | Analogy |
|---------|--------------|---------|
| **Actual return** | Real reward from the simulator | The final exam score |
| **Critic** | Neural net predicting the return before it happens | A teacher predicting your score mid-exam |
| **TD error (delta)** | How wrong the critic was | "You did better than I expected!" |
| **Ground truth** | The simulator's reward (+80/-40) | The answer key |
| **Advantage** | Was this action better than the critic expected? | "That move was better than average" |

The policy learns by trial and error — no human labels needed, just the simulator's reward signal.

---

## Task 2 — PPO Policy Update: Summary

### TODO 2a: Old Policy Log-Probabilities

```python
log_pi_old_sa = self.policy.get_log_prob(s, a, action_mask).squeeze(-1).detach()
```

Before updating the policy, we record what the **old policy** thought of each action it took. This is the probability of the action under the policy that collected the data. We `.detach()` it because this is a fixed reference — we don't want gradients flowing through it during the update.

**Why:** PPO compares the new policy to the old policy. The ratio `new/old` tells us how much the policy changed. We need the old probability as a baseline.

### TODO 2b: Critic Loss (MSE)

```python
loss = torch.nn.functional.mse_loss(self.value(s_b).squeeze(-1), v_target_b)
```

The critic (value network) predicts "how much reward will I get from this state?" We train it by comparing its prediction against `v_target` — the actual discounted return computed by GAE. MSE penalizes large errors more than small ones.

**Why:** A better critic → better advantage estimates → better policy updates. The critic is the "coach" that tells the policy whether its actions were good or bad.

### TODO 2c: Policy Ratio

```python
log_pi_sa = self.policy.get_log_prob(s_b, a_b, action_mask_b).squeeze(-1)
ratio = torch.exp(log_pi_sa - log_pi_old_sa_b)
```

The ratio measures **how much the policy changed** since collecting the data:
- `ratio = 1.0` → no change
- `ratio > 1.0` → new policy likes this action more
- `ratio < 1.0` → new policy likes this action less

We compute it in log-space (`exp(log_new - log_old)`) because multiplying many small probabilities causes numerical underflow. Subtracting logs is numerically stable.

**Why:** PPO uses this ratio to scale the advantage. If the policy now likes a good action more, that's reinforced. But if it changed *too much*, we clip it (see 2d).

### TODO 2d: PPO-Clip Surrogate

```python
surr1 = ratio * A_sa_b
surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
surrogate = -torch.min(surr1, surr2).mean()
```

This is the core of PPO. Two objectives are computed:

- **surr1** (unclipped): `ratio × advantage` — if the action was good (positive advantage) and the policy now likes it more (ratio > 1), this grows. No limit.
- **surr2** (clipped): same but with the ratio clamped to `[1-ε, 1+ε]` (e.g., `[0.8, 1.2]`). This limits how much the policy can change per update.

We take `min(surr1, surr2)` — the **pessimistic** bound. This means:
- If the action was good and ratio > 1.2 → use surr2 (capped at 1.2)
- If the action was bad and ratio < 0.8 → use surr2 (floored at 0.8)

The negative sign (`-`) is because PyTorch minimizes loss, but we want to **maximize** the PPO objective.

**Why the min?** Without clipping, the policy could make huge jumps — e.g., ratio = 10 on a good action would massively reinforce it, potentially destroying the policy. The clip prevents destructive updates while still allowing learning.

---

## What is "Network Clipping" and Why Do We Do It?

There are **two types of clipping** in this code:

### 1. PPO clip (epsilon clip) — in TODO 2d

```python
torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
```

This limits the **policy ratio** to `[0.8, 1.2]`. It prevents the policy from changing too drastically in a single update. Without it, the policy could exploit a single good experience and overfit to it, destroying previously learned behavior.

### 2. Gradient clipping — after backprop

```python
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
```

This limits the **gradient norm** to 10. It prevents exploding gradients — if one mini-batch produces a huge gradient, it gets scaled down. This keeps training stable.

### 3. Ratio clamping — before the clip

```python
ratio = torch.clamp(ratio, 0, 10)
```

This prevents `inf` values from extremely small probabilities (the multi-discrete action space multiplies many probabilities, which can underflow to near-zero, making the ratio explode).

### Summary of why clipping matters

| Clip | What it limits | Without it |
|------|---------------|-----------|
| PPO clip (ε) | Policy change per update | Policy makes huge jumps, destroys learning |
| Gradient clip (norm 10) | Gradient magnitude | Exploding gradients, NaN weights |
| Ratio clamp (0-10) | Numerical stability | Inf ratios → NaN gradients |

---

## Gamma vs Lambda (Tau) — Bias-Variance Tradeoff

There are **two** parameters that control bias-variance in GAE:

| Parameter | Code name | What it controls |
|-----------|-----------|-----------------|
| **gamma (γ)** | `self.gamma` | Discount factor — how far into the future rewards matter |
| **lambda (λ)** | `self.tau` | GAE bias-variance tradeoff — how much to trust the critic vs actual returns |

### What Each Does

**Gamma (γ)** — "How far ahead do I look?"
- Lower γ (0.90): Only cares about near-future rewards → less to estimate → lower variance, but ignores long-term consequences (bias)
- Higher γ (0.99): Cares about far-future rewards → more to estimate → higher variance, but captures long-term effects (less bias)

**Lambda (λ/tau)** — "How much do I trust my critic?"
- Lower λ (e.g., 0.5): Relies more on the critic's value estimates → more biased (if critic is wrong, advantages are wrong) but lower variance
- Higher λ (e.g., 0.95): Relies more on actual observed returns → less biased but higher variance (returns are noisy)

### What We Did

We tuned **gamma** (0.99 vs 0.90) because it has a clearer conceptual meaning (far-sighted vs myopic) and directly affects dialogue behavior. The `tau` parameter was kept fixed at 0.95.