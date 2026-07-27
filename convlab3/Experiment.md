# Experiment: PPO Dialogue Policy Training & Pipeline

> **Course**: Spoken Dialogue Systems — Heinrich Heine University Düsseldorf  
> **Framework**: ConvLab-3  
> **Date**: June 2026

---

## Task 1: GAE Advantage Estimation

Implemented Generalized Advantage Estimation (GAE) in `convlab/policy/ppo/ppo.py` → `est_adv()`.

The method iterates backwards through the trajectory and computes three values at each step:

- **v_target**: The value target used to train the critic. Computed as `r[t] + gamma * mask[t] * prev_v_target`.
- **delta**: The TD error. Computed as `r[t] + gamma * mask[t] * prev_v - v[t]`.
- **A_sa**: The advantage estimate. Computed as `delta[t] + gamma * tau * mask[t] * prev_A_sa`.

`mask[t]=0` marks the end of a trajectory, zeroing out future contributions. Advantages are normalized at the end for training stability.

---

## Task 2: PPO Clip Policy Update

Implemented the PPO clipped surrogate objective in `convlab/policy/ppo/ppo.py` → `update()`.

Four steps inside the update loop:

- **2a**: Compute old policy log-probabilities using `self.policy.get_log_prob(s, a, action_mask)`, detached from the graph.
- **2b**: Train the critic with MSE loss between predicted values and GAE value targets.
- **2c**: Compute the policy ratio as `exp(log_pi_new - log_pi_old)` in log-space for numerical stability.
- **2d**: Compute the PPO-clip surrogate as `-min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A).mean()`. Negative because we minimize loss but want to maximize the PPO objective.

The update runs for 5 rounds over the collected data, split into mini-batches of size 32, with gradient clipping at norm 10.

---

## Exercise 2: Dialogue Pipeline (Prerequisite)

Completed all 5 TODOs in `examples/agent_examples/interact.py`:

1. **NLU**: `BERTNLU(mode='sys', config_file='simplemultiwoz21_user_context3.json', model_file='pretrained_models/nlu/bert-mini_...zip')`
2. **DST**: `RuleDST()` (was already correct)
3. **Policy**: MLE policy loaded manually from `pretrained_models/policy/experiment_2025-02-19-13-02-06/` with correct `manually_add_entity_names=False`
4. **Pipeline**: `PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')`
5. **Interaction loop**: 40-turn loop with `input()`, breaks on "bye"

### How to Run

```bat
cd examples\agent_examples
python interact.py
```

This starts a chat with the MLE-based dialogue system. Type a request (e.g. `I want to find a restaurant`) and the system will respond. Type `bye` to end the conversation.

---

## Task 3: RL Training with Hyperparameter Tuning

We tune the **discount factor (gamma)**: 0.99 (far-sighted) vs 0.90 (myopic). Gamma controls how much future rewards matter in GAE and value target computation.

### Training Results

| Experiment | Gamma | Initial Success | Final Success | Initial Complete | Final Complete | Initial Return | Final Return | Final Turns |
|------------|-------|-----------------|---------------|-----------------|---------------|----------------|-------------|-------------|
| gamma099   | 0.99  | 0.478           | 0.584         | 0.61            | 0.682         | 6.69           | 20.14       | 9.94        |
| gamma090   | 0.90  | 0.478           | 0.58          | 0.61            | 0.654         | 6.69           | 19.35       | 10.25       |

> Replace `_TBD_` with actual values after training completes.

### Metric Explanations

- **Success**: Fraction of dialogues where all task goals were fully achieved (strict).
- **Complete**: Fraction of dialogues where the user simulator considered the task complete.
- **Return**: Average cumulative reward per dialogue (higher is better; includes success bonus and turn penalty).
- **Turns**: Average number of dialogue turns (lower means more efficient conversations).

### How to Run

```bat
cd convlab\policy\ppo

:: Train gamma=0.99
python train.py --config_name=RuleUser-Semantic-RuleDST-gamma099 --seed=42

:: Train gamma=0.90
python train.py --config_name=RuleUser-Semantic-RuleDST-gamma090 --seed=42
```

### Monitor with TensorBoard

```bat
cd convlab\policy\ppo
python -m tensorboard.main --logdir=experiments
```

Open http://localhost:6006 in your browser.

### Config Files

| File | Purpose |
|------|---------|
| `configs/ppo_config_gamma099.json` | PPO hyperparameters with gamma=0.99 |
| `configs/ppo_config_gamma090.json` | PPO hyperparameters with gamma=0.90 |
| `configs/RuleUser-Semantic-RuleDST-gamma099.json` | Environment config for gamma=0.99 |
| `configs/RuleUser-Semantic-RuleDST-gamma090.json` | Environment config for gamma=0.90 |
| `train_gamma099.bat` | Batch script for gamma=0.99 |
| `train_gamma090.bat` | Batch script for gamma=0.90 |

---

## Task 4: Plug PPO Policy into Pipeline

Created `examples/agent_examples/interact_ppo.py` — same pipeline as Exercise 2 but with the RL-trained PPO policy instead of MLE.

```bat
cd examples\agent_examples

:: Auto-discover latest experiment
python interact_ppo.py

:: Or specify a model path
python interact_ppo.py --model_path ../../convlab/policy/ppo/finished_experiments/experiment_XXX/save/best_ppo.pol.mdl
```

The only difference from Exercise 2 is the policy module: `PPO` (RL-trained) instead of `MLEPolicy` (supervised). Everything else (NLU, DST, NLG, pipeline) is identical.

### Example Conversations

The system supports **restaurant** and **hotel** domains. Type `bye` to end any conversation.

**Example 1 — Restaurant booking:**
```
You: I want to find a restaurant
You: It should be in the centre of town
You: I want italian food
You: What is the price range?
You: Book a table for 4 people at 19:00
You: bye
```

**Example 2 — Hotel search:**
```
You: I need a hotel
You: In the north area
You: I need free wifi
You: What type of hotel is it?
You: bye
```

**Example 3 — Multi-domain (restaurant + hotel):**
```
You: I am looking for a restaurant in the south part of town
You: I want cheap food
You: Can you also find me a hotel in the south?
You: Does it have free parking?
You: bye
```

**Tips:**
- The system may ask follow-up questions (cuisine, area, price range) — answer naturally
- If the system gives a recommendation, say `yes` or `that sounds good` to accept
- Keep sentences short — the BERT NLU works best with simple inputs

---

## File Index

| File | Changes |
|------|---------|
| `convlab/policy/ppo/ppo.py` | Task 1 (GAE), Task 2 (PPO clip), `config_file` param |
| `convlab/policy/ppo/train.py` | Read `ppo_config` from env config, Database pre-import |
| `convlab/util/custom_util.py` | Fixed `move_finished_training` to use copytree |
| `examples/agent_examples/interact.py` | Exercise 2: all 5 TODOs |
| `examples/agent_examples/interact_ppo.py` | Task 4: PPO pipeline (new) |
| `convlab/policy/ppo/configs/ppo_config_gamma099.json` | PPO config, gamma=0.99 (new) |
| `convlab/policy/ppo/configs/ppo_config_gamma090.json` | PPO config, gamma=0.90 (new) |
| `convlab/policy/ppo/configs/RuleUser-Semantic-RuleDST-gamma099.json` | Env config, gamma=0.99 (new) |
| `convlab/policy/ppo/configs/RuleUser-Semantic-RuleDST-gamma090.json` | Env config, gamma=0.90 (new) |
| `train_gamma099.bat` | Batch script (new) |
| `train_gamma090.bat` | Batch script (new) |

---

## Report & Analysis

### 1. Comparison of Trained Policies

Both PPO policies were initialized from the same pre-trained MLE model and trained for 10 epochs (10,000 dialogues each) with identical hyperparameters except for the discount factor gamma.

#### Overall Metrics

| Metric | MLE Baseline | gamma=0.99 (final) | gamma=0.90 (final) |
|--------|-------------|--------------------|--------------------|
| Success Rate | 0.478 | 0.584 | 0.58 |
| Complete Rate | 0.61 | 0.682 | 0.654 |
| Avg Return | 6.69 | 20.14 | 19.35 |
| Avg Turns | 10.67 | 9.94 | 10.25 |
| Avg Actions | 4.64 | 4.95 | 4.49 |

#### Per-Domain Success Rates

| Domain | MLE Baseline | gamma=0.99 | gamma=0.90 |
|--------|-------------|-----------|-----------|
| Restaurant | 0.431 | 0.537 | 0.559 |
| Hotel | 0.433 | 0.549 | 0.493 |

#### Analysis

**gamma=0.99 vs gamma=0.90:** The higher discount factor (0.99) achieved better overall results — higher complete rate (0.682 vs 0.654), higher return (20.14 vs 19.35), and fewer turns (9.94 vs 10.25). This makes sense: with gamma=0.99, the policy weighs future rewards almost as much as immediate ones, leading to better long-term planning in multi-turn dialogues. The policy learns to pursue task completion even when it requires several turns of information gathering.

**Domain differences:** Interestingly, gamma=0.90 performed better on restaurants (0.559 vs 0.537) but worse on hotels (0.493 vs 0.549). This suggests that the hotel domain may require more long-term planning (e.g., gathering more constraints before booking), which benefits from a higher discount factor. Restaurants may have simpler goals where a more myopic policy suffices.

**Convergence:** Both policies plateaued in success rate after epoch 5, but continued improving in complete rate and turn efficiency. This indicates the policy found a stable strategy early but kept refining its efficiency. The avg actions metric diverged: gamma=0.99 increased actions (4.64→4.95) while gamma=0.90 decreased them (4.64→4.49), suggesting gamma=0.99 learned to take more diverse actions to ensure task completion.

**Select actions:** gamma=0.99 developed a notable increase in select actions (0.0→0.078) by epoch 10, while gamma=0.90 stayed low (0.023). Select actions help the user choose between options, which may contribute to the higher complete rate of gamma=0.99.

### 2. Interaction Examples

#### MLE Policy (Exercise 2)

```
System: Hello, how can I help you?
You:   I want to find a restaurant
System: There are 10 available restaurants. Is there a certain kind of food you would like?
You:   In the centre of town
System: How about pizza hut city centre? It is in the centre area. There are 10 restaurants
        in that area that fit that criteria. That is a italian restaurant. They are in the cheap
        price range. Excellent. pizza hut city centre is just your thing. What type of food
        would you like?
You:   I want italian food
System: There is a restaurant called pizza hut city centre that meets your criteria. It is at
        Regent Street City Centre. It is in the centre area. There are 9 restaurants in that
        area that fit that criteria. They serve italian. This restaurant is in the cheap price
        range. You are welcome. Is there anything else I can help you with today?
You:   What is the price range?
System: Would you like to try pizza hut city centre? It is at Regent Street City Centre. It is
        located in the centre. They serve italian food. It's in the cheap price range. You are
        welcome. Is there anything else I can help you with today?
You:   bye
System: Goodbye!
```

#### PPO Policy (gamma=0.99, Task 4)

```
System: Hello, how can I help you today?
You:   I want to find a restaurant
System: I have 10 different restaurants I can give you some information for. They are all
        pretty good. They are in the cheap price range. Do you have any specific type of food
        you would like?
You:   In the centre of town
System: How about pizza hut city centre? It is in the centre area. There are 10 available
        restaurants. That is a italian restaurant. They serve indian food. This restaurant is
        in the cheap price range. I would suggest pizza hut city centre. What type of food
        would you like? Do you prefer indian food? Did you need any further assistance today?
You:   I want italian food
System: Would you like to try pizza hut city centre? It is located in the centre. I have 9
        options for you. It is italian food. They are cheap. Did you need any further
        assistance today?
You:   What is the price range?
System: How does the pizza hut city centre sound? It is at Regent Street City Centre. It is in
        the centre area. They serve italian. They are cheap. Did you need any further
        assistance today?
You:   bye
System: Goodbye! Have a nice day.
```

#### Comparison

Both policies successfully handle the restaurant search and recommend the same venue (pizza hut city centre). Key differences:

- **PPO takes more actions per turn** — it packs more information into each response (price range, suggestions, follow-up questions in a single turn), which aligns with the higher avg actions metric (4.95 vs 4.64).
- **PPO is more proactive** — it volunteers price range information and makes suggestions without being asked, while MLE tends to wait for explicit user requests.
- **PPO has some inconsistencies** — it mentions "They serve indian food" while also saying "italian restaurant," suggesting the RL-optimized policy sometimes generates conflicting actions. This is a known issue with multi-discrete action spaces where each action dimension is sampled independently.
- **MLE is more conservative** — it provides cleaner, more focused responses but may miss opportunities to proactively help the user.

### 3. Alignment Between RL Metrics and Interaction Quality

The RL evaluation metrics (success rate 58.4%, complete rate 68.2%) are measured against a **rule-based user simulator** that behaves predictably — it follows a fixed goal, provides clear inputs, and terminates when the goal is met. In contrast, human interaction (as simulated by our typed inputs) introduces several differences:

**Where metrics align with interaction:**
- The PPO policy **does** successfully complete the task — it finds a matching restaurant and provides relevant details (location, cuisine, price). This matches the 68.2% complete rate.
- The PPO policy is **more informative** per turn (higher avg actions), which the metrics capture and which we observe in the interaction.

**Where metrics diverge from interaction:**
- **Response coherence:** The RL metrics don't measure whether the system's utterances are coherent. In our interaction, the PPO policy said "They serve indian food" alongside "italian restaurant" — a contradiction. The reward function only measures task success, not linguistic quality, so the policy learned to maximize information output even when actions conflict.
- **User experience:** The rule-based simulator never gets confused by contradictory information, but a real user would. The 58.4% success rate may overestimate real-world performance.
- **Turn efficiency vs informativeness:** The PPO policy reduced turns (9.94 vs 10.67 baseline) by packing more actions per turn. But this makes responses longer and harder to parse, potentially degrading user experience despite improving the metric.

**Why this happens:** PPO optimizes a **task-completion reward** (success = +80, failure = −40, minus turn penalty). There is no penalty for contradictory actions or overly long responses. The policy learns that taking more actions per turn increases the chance of hitting the right information, which improves task success but can produce noisy outputs. The MLE policy, trained on human dialogues, produces more natural responses but is less aggressive about task completion.

**Conclusion:** The RL metrics accurately reflect task-completion ability but do not capture interaction quality. The PPO policy is better at completing tasks (higher success/complete rates) but produces less coherent responses than the MLE baseline. A hybrid approach — using RL to optimize task success while constraining action diversity — could potentially combine the strengths of both approaches.
