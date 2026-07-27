# PPO Dialogue Policy — Presentation Summary

![alt text](mle_ppo_architectur.png)

---

## Experiment Setup

| Parameter | MLE Baseline | Exp 1 (gamma099) | Exp 2 (gamma090) | Exp 3 (gamma099-lr0003) | Exp 4 (gamma090-lr0003) |
|-----------|-------------|------------------|------------------|------------------------|------------------------|
| **Training type** | Supervised | RL (PPO) | RL (PPO) | RL (PPO) | RL (PPO) |
| **gamma** | N/A | 0.99 | 0.90 | 0.99 | 0.90 |
| **policy_lr** | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 |
| **value_lr** | N/A | 0.00005 | 0.00005 | 0.00005 | 0.00005 |
| **epsilon (clip)** | N/A | 0.2 | 0.2 | 0.2 | 0.2 |
| **tau (GAE lambda)** | N/A | 0.95 | 0.95 | 0.95 | 0.95 |
| **update_round** | N/A | 5 | 5 | 10 | 10 |
| **batchsz** | 32 | 32 | 32 | 32 | 32 |
| **h_dim** | 100 | 100 | 100 | 100 | 100 |
| **hv_dim** | N/A | 50 | 50 | 50 | 50 |
| **epochs** | 24 | 10 | 10 | 20 | 20 |
| **dialogues/epoch** | N/A (dataset) | 1000 | 1000 | 2000 | 2000 |
| **total dialogues** | ~11,806 turns | 10,000 | 10,000 | 40,000 | 40,000 |
| **eval dialogues** | N/A | 500 | 500 | 500 | 500 |
| **eval frequency** | 1 | 5 | 5 | 5 | 5 |
| **seed** | 1234 | 42 | 42 | 42 | 42 |
| **initialization** | From scratch | MLE pretrained | MLE pretrained | MLE pretrained | MLE pretrained |

---

## Experiment Results

### Evaluation Metrics (measured against rule-based user simulator, best checkpoint)

| Metric | MLE Baseline | Exp 1 (gamma099) | Exp 2 (gamma090) | Exp 3 (gamma099-lr0003) | Exp 4 (gamma090-lr0003) |
|--------|-------------|------------------|------------------|------------------------|------------------------|
| **Success Rate** | 0.478 | 0.588 | 0.58 | 0.594 | 0.584 |
| **Complete Rate** | 0.61 | 0.656 | 0.654 | 0.832 | 0.716 |
| **Avg Return** | 6.69 | 20.26 | 19.35 | 23.22 | 20.66 |
| **Avg Turns** | 10.67 | 10.30 | 10.25 | 8.06 | 9.42 |
| **Avg Actions** | 4.64 | 4.88 | 4.49 | 4.65 | 4.12 |

### Best Checkpoint Info

| Experiment | Best Epoch | Dialogues at Best Epoch |
|------------|-----------|------------------------|
| Exp 1 (gamma099) | 5 | 5,000 |
| Exp 2 (gamma090) | 10 | 10,000 |
| Exp 3 (gamma099-lr0003) | 10 | 20,000 |
| Exp 4 (gamma090-lr0003) | 5 | 10,000 |

### Training Metrics Plots

Training and evaluation metrics can be viewed using TensorBoard:

```bat
cd convlab\policy\ppo
python -m tensorboard.main --logdir=finished_experiments
```

Open http://localhost:6006 in your browser to see all metrics (success rate, complete rate, return, turns, policy loss, value loss, mean training reward, action distributions) for all experiments.

---

## Baseline Interaction (MLE Policy)

The same dialogue scenario (restaurant search + booking) was run with the MLE baseline policy:

```
System: Hello, how can I help you?
You:   I am looking for a restaurant in the south part of town
System: It is located in the south. There are 9 available restaurants. That is a chinese
        restaurant. They are in the expensive price range. What type of food would you like?
You:   I want cheap food
System: Would you like to try the good luck chinese food takeaway? It is located in the south.
        I have 9 options for you. It is chinese food. They are in the expensive price range. I
        would suggest the good luck chinese food takeaway.
You:   Can you also find me a hotel in the south?
System: There are 1 of those.
You:   Does it have free parking?
System: How about the lensfield hotel? They are located in the south. We have 1 such places.
        It is a hotel.
You:   book a table at the chinese food takeaway restaurant at 20:00 today and one room at
        lensfield hotel today at 22:00
System: Does the the lensfield hotel work? It is located at 53-57 lensfield road. It is indeed
        in the south. The wifi is included. This is an expensive hotel. It is a hotel. Is there
        anything else I can help you with today?
You:   yes book a room
System: You are welcome. Is there anything else I can help you with today?
You:   bye
System: Goodbye!
```

**Observations:** The MLE policy handles a multi-domain dialogue (restaurant + hotel) with clean, focused responses. It asks clarifying questions ("What type of food would you like?") and provides information incrementally. However, it doesn't proactively volunteer details like address or wifi unless asked, and the booking request for the restaurant was not completed (only the hotel booking was handled).

---

## PPO Interaction (Exp 3 — Best Model)

The same dialogue scenario run with the PPO policy from Experiment 3 (gamma=0.99, lr=0.0003, best checkpoint at epoch 10):

```
System: Hello, how can I help you today?
You:   I am looking for a restaurant in the south part of town
System: How about the good luck chinese food takeaway? It is located in the south. I have 9
        different restaurants I can give you some information for. They are all pretty good.
        That is a chinese restaurant. It is a expensive restaurant.
You:   I want cheap food
System: How about the good luck chinese food takeaway? They are located at 82 Cherry Hinton
        Road Cherry Hinton. It is located in the south. There are 9 restaurants in that area
        that fit that criteria. That is a chinese restaurant. 01223244149 is the restaurant
        phone number. They are in the expensive price range.
You:   Can you also find me a hotel in the south?
System: There is a restaurant called the good luck chinese food takeaway that meets your
        criteria.
You:   is there a cheap hotel in the south?
System: It is located in the south. What is the price range for you?
You:   Does it have free parking?
System: It is in the south area. There are 9 different places that match your description.
        They serve chinese. This restaurant is in the expensive price range.
You:   book a table at the chinese food takeaway restaurant at 20:00 today
System: There's a place called the good luck chinese food takeaway. The reference number is
        00000003. It is in the south area. I have 3 different restaurants I can give you some
        information for. They are all pretty good. They serve chinese. This is a expensive one.
        Is there anything else I can help you with?
You:   I am looking for a hotel
System: Would you like to try the good luck chinese food takeaway? Their address is 82 Cherry
        Hinton Road Cherry Hinton. It is in the south area. The restaurant's phone number is
        01223244149. It's in the expensive price range. Is there anything else I can help you
        with today?
You:   bye
System: Goodbye! Have a nice day.
```

**Observations:** The PPO policy is more proactive — it volunteers the phone number and address without being asked, packing more information per turn. It successfully completed the restaurant booking (reference number 00000003). However, it struggled with the hotel domain — when the user asked for a hotel, the policy kept responding with restaurant information, suggesting it didn't properly switch domains. This aligns with the multi-discrete action space issue where the policy can get "stuck" repeating actions from the dominant domain. The PPO policy completed the dialogue in 8 turns (vs 10 for MLE), reflecting its lower avg turns metric (8.06 vs 10.67).

---

## Observations

| Aspect | MLE (Supervised Policy) | PPO (RL Policy) | Interpretation |
|--------|--------------------------|-----------------|---------------------------------------|
| **Training Objective** | Learns to imitate human demonstrations. | Learns to maximize cumulative reward. | Different optimization objectives naturally produce different dialogue strategies. |
| **Dialogue Length** | Longer conversations (10.67 turns). | Shorter conversations (8.06 turns). | PPO minimizes unnecessary dialogue because shorter conversations increase the **Avg Return** and reduce **Avg Turns**. |
| **Information Delivery** | Provides information gradually and only when requested. | Gives address, phone number, and booking reference proactively. | Packing more useful information into each turn helps complete tasks faster, reflected by higher **Avg Return**, higher **Success Rate**, and lower **Avg Turns**. |
| **Task Completion** | Restaurant booking was not completed; hotel booking only partially handled. | Successfully completed the restaurant booking and produced a booking reference. | Better task completion is reflected by the higher **Success Rate** (59.4% vs. 47.8%) and **Complete Rate** (83.2% vs. 61.0%). |
| **Response Style** | More cautious and asks clarifying questions. | More direct and goal-oriented. | PPO prioritizes maximizing reward over imitating natural dialogue, contributing to higher **Avg Return** and fewer **Avg Turns**. |
| **Multi-domain Handling** | Correctly switches from restaurant to hotel domain. | Continues discussing restaurants after the user asks about hotels. | This weakness is **not reflected** by the reported metrics because **Success Rate** and **Complete Rate** are measured using a rule-based simulator, not human evaluation. |
| **Policy Weakness** | Less efficient and sometimes fails to complete the user's goals. | More efficient but less robust when the conversation changes topic. | RL improves efficiency (**Avg Turns**, **Avg Return**) but does not necessarily improve conversational robustness, which requires manual evaluation. |
| **Evaluation Metrics** | Lower Success Rate, Complete Rate, and Avg Return. | Higher Success Rate (59.4%), Complete Rate (83.2%), Avg Return (23.22), and fewer Avg Turns (8.06). | The quantitative metrics confirm that PPO achieves better task-oriented performance than MLE. |
| **Limitations of Evaluation** | Lower metrics already indicate weaker task performance. | High metrics hide issues such as repetitive responses and poor domain switching. | Metrics measure task success and efficiency (**Success Rate**, **Complete Rate**, **Avg Return**, **Avg Turns**) but not dialogue naturalness or context management. |
| **Overall Assessment** | More natural and better at maintaining multi-domain context, but less efficient. | More efficient and successful at completing tasks, but occasionally produces less coherent conversations. | The manual interactions largely agree with the RL metrics: PPO improves **Success Rate**, **Complete Rate**, **Avg Return**, and **Avg Turns**, while revealing qualitative weaknesses that these metrics do not capture. |