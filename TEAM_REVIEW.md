## Instructions
- [x] 1. choose any robot learning paper (FlashSAC)
- [x] 2. for the first part of the project, they reproduce the results for the paper (Done, see project/16831FinalProject/results/nish/plots/paper_vs_baseline.png)
- [ ] 3. for the second part, they suggest an improvement for the paper and implement it. The improvement may or may not work, but they should explain the intuition behind it. (review explanations below please)
- [ ] 4. for the third part, analyze the approach by showing ablations beyond just the accuracy and explain why the intuition was right/wrong with the help of those ablations. (runs are nearly done, see RUNS.md)

## Final Presentation
- 20% of our final grade
    - 5%: basic participation
    - 10% presentation skills (based on thhe presentation on Decemeber 3rd)
    - 5% technical (based on the PDF version of our slides)
- 4 minutes to present, followed by a 1-minute Q&A session (cut-off at 4 minutes)
- "We also encourage everyone to participate in the project presentation. Knowing what other people are working on is an extremely important part of robot learning research." 
- Important note: don't forget to grant read permissions to everyone to play videos during the presentation. This is a very common mistake, and we won't be able to grant you extra time if you forget to do so.

## Final Report
- PDF version of the final report and slieds are due on Dec 11, no late days
**- 8 page limit, including references and figures!!!!!!**
- Corl Template
- Parts
    1. Motivation: Why is it important and relevant? Why should we care?
    2. Problem statement & prior work: Define the research problem. Briefly mention related works that are relevant to your idea instead of covering all related works.
    3. Your idea: What is the (key) idea?
    4. Your idea: Why does your idea make sense intuitively?
    5. Your idea: How does it relate to prior work in the area?
    6. Your results: If it worked, how much did you improve and why?
    7. Your results: If it did not work, why did you expect it to work? What are the next steps to try?
    8. Your results: Any negative results? Maybe you figured after trying that it did not make sense.
    9. Summary
    10. Future directions

## Theoretical Reasoning For Changes

### 1. Why removing the target network should be safe

**Why target networks exist.**
- DQN introduced the target network to stabilise bootstrapped Q-learning [2].
- The underlying problem is the *deadly triad*: bootstrapping, off-policy data and function approximation used together can diverge [3, 4].
- Updating Q(s,a) also moves Q(s′,a′) through shared features. The target therefore moves with every update, and errors can feed back on themselves.

**Normalization can do the target network's job.**
- **PQN (Gallici et al., ICLR 2025)** [5] has the key theoretical result. Regularization such as LayerNorm (plus ℓ2) makes off-policy TD *provably convergent without a target network or replay buffer*. Normalized features bound how much an update at (s,a) can move the prediction at (s′,a′). Empirically, parallel vectorized sampling stabilises training further [5].
- **Yue et al.** [6] trace Q-value divergence to "self-excitation", which they analyse through the neural tangent kernel, and show that LayerNorm prevents it.
- **Ball et al. (RLPD)** [7] show that LayerNorm bounds how far Q can extrapolate beyond the data.
- **Kumar et al. (DR3)** [8] identify feature co-adaptation between (s,a) and (s′,a′) as a cause of instability in TD learning, and show that explicitly regularizing it helps.
- **CrossQ (Bhatt et al., ICLR 2024)** [9] removes the target network from SAC and adds BatchNorm. It runs (s,a) and (s′,a′) through the critic as one concatenated batch, so both halves are normalized with the same statistics. CrossQ matches or beats REDQ [10] and DroQ [11] at an update-to-data (UTD) ratio of 1 instead of 20, with no "advanced bias-reduction schemes".
- **Palenicek et al. (CrossQ + WN, 2025)** [12] add weight normalization to CrossQ. It keeps the effective learning rate constant and lets CrossQ scale to higher UTD.

**FlashSAC's critic already has these stabilizers** [1]:
- BatchNorm with the CrossQ joint (s, s′) forward pass
- RMSNorm before the value head
- every weight vector projected onto the unit sphere after each step

**Targets are bounded by construction.**
- The critic is categorical and trained with cross-entropy, as in C51 [14]. Classification losses for value functions are more stable and scale better than MSE regression [15].
- Its fixed support of [−5, 5] means predicted values cannot run away in scale however the targets move.

**This regime replays each transition very few times (ours).**
- Our runs make 97,658 updates × 2048 samples over 50M transitions, so each transition is sampled about **4 times** on average.
- Single-env SAC or CrossQ at UTD 1 (batch 256, 1M steps) samples each transition about **256 times**. REDQ and DroQ at UTD 20 sample each one about **5,000 times** [10, 11].
- Target networks matter most when the critic can overfit its own bootstrapped targets on reused data. We reuse data about 60× less.

### 2. Why removing the second critic should be safe, and might help

**Clipped double-Q exists to fix overestimation.**
- Maximizing over noisy value estimates is biased upward: E[max] ≥ max E [16, 17].
- In actor-critic methods, the actor's gradient ascent on Q plays the role of the max. It seeks out actions whose error happens to be positive, and bootstrapping compounds the error.
- TD3 demonstrated this in continuous control and introduced the clipped double-Q min over two critics [18].
- REDQ [10], TQC [19] and Maxmin Q-learning [20] are ways to dial this bias up or down.

**The min has its own cost: systematic pessimism.**
- If two critics have independent Gaussian errors with spread σ, then E[min(Q₁, Q₂)] = Q − σ/√π. That bias applies everywhere, whether or not overestimation is happening.
- **Maxmin Q-learning** [20]: how much overestimation hurts learning depends on the environment.
- **TOP (Moskovitz et al.)** [21]: the best degree of optimism or pessimism varies across tasks *and over the course of training*.
- **OAC (Ciosek et al.)** [22]: a lower-bound critic causes *pessimistic underexploration*. The actor avoids uncertain actions, which are exactly the ones it needs to try.

**In massively parallel simulation, the trade-off can flip.**
- **WarpSAC (Wu et al., 2026)** [23], a direct FlashSAC follow-up, argues that massively parallel simulation changes the data regime. It finds stabilizers to be *data-regime-dependent*:
  - "clipped double-Q can be relaxed in high-throughput manipulation"
  - parameter normalization "restricts value fitting when data are abundant"
  - It proposes **WarpSAC-A** (normalization off, single Q) for data-abundant GPU training, and reports +23.1% normalized score–step AUC over FlashSAC across 14 GPU-parallel environments.
- **FastSAC / FastTD3 for humanoids (Seo et al., 2025)** [24]: on G1 rough-terrain locomotion, averaging the two Q values beats the clipped double-Q min.
- **BRO (Nauman et al., NeurIPS 2024)** [25] ranks clipped double-Q among the most harmful components in its ablation, and scales up the critic instead.
- **Counter-evidence: FastTD3** [26] found the min "generally performs better" without LayerNorm. FlashSAC's critic uses BatchNorm, RMSNorm and weight projection rather than LayerNorm, so the sign isn't settled in advance.

**Why overestimation should be weaker here (ours).**
- The actor can only exploit errors in regions the data covers poorly. With 1024 parallel environments [27, 28] and a 10M buffer, coverage is broad.
- The [−5, 5] support caps how far overestimation can go.
- SAC's entropy bonus already regularizes the policy.

### 3. Wider Critic

**Why widen the critic.**
- Dropping a critic halves the critic's parameters. Widening the hidden layers from 256 to 360 restores about (360/256)² ≈ 1.98× of the hidden-layer parameters.
- The comparison then separates "no min" from "smaller network".
- Our widening is modest (1.4× the width) and only *restores* the original parameter budget. CrossQ widens far more (8×), *adding* capacity beyond SAC's.

**What CrossQ says about width** [9]:
- **A wider critic is CrossQ's third design choice**, alongside BatchNorm and removing the target network.
  - Its critic layers are 2048 units wide versus SAC's 256: 8× the width, about 64× the hidden-to-hidden parameters.
- **The stated motivation is optimization speed, not stability.** CrossQ widens the critic "motivated by prior research on the ease of optimization of wider networks" (Ota et al. [29]), and finds wider layers "lead to even faster learning".
- **Width is secondary but still helps.**
  - "Most performance gains originate from the first two design choices; however, wider critic networks further boost the performance, helping to match or outperform REDQ and DroQ sample efficiency."
  - Shrinking the critic back to 256 "reduces performance and yet still significantly outperforms SAC". The smaller version is still competitive with full CrossQ in four of six environments.
- **Width only helps on top of normalization.**
  - "SAC does not benefit from simply widening critic layers."
  - Ota et al. [29] report the same: "naively increasing network capacity does not improve performance", which they attribute to instability when training larger networks.
- **Width costs compute.** CrossQ notes that wider critics "increase the computation cost". In its wall-clock table, Ant takes 1.5 h with the wide critic vs 1.2 h at width 256.
- **CrossQ's own ablation found that a single critic hurts:** "Training without double Q networks (single critic) harms performance." That result is at width 2048, single-environment, with UTD 1, so extra width did not replace the second critic in their setting. This is the strongest counter-evidence to §2 and should be acknowledged in the report.

**The same pattern elsewhere: extra capacity pays off only with normalization.**
- **BRO** [25]: "strong regularization allows for effective scaling of the critic networks."
- **SimBa** [30]: adding parameters consistently improves sample efficiency once observation normalization, residual blocks and LayerNorm "inject a simplicity bias". **SimbaV2** [13] adds hyperspherical normalization to scale further.
- **Palenicek et al.** [12]: wide BatchNorm critics (CrossQ) need weight normalization to keep improving at higher UTD. Weight normalization "keeps the effective learning rate constant".

**Why this applies to our change (ours).**
- FlashSAC's critic already has the normalization these papers require [1]:
  - BatchNorm
  - RMSNorm before the value head
  - weight projection onto the unit sphere
- So extra width should add usable capacity rather than instability.
- **Speed:** update time −8.4% (1 critic) and −9.8% (1 wide critic) vs the baseline, so the wider network gives back almost none of the savings.


### References

1. D. Kim et al. "FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control." arXiv:2604.04539, 2026. https://arxiv.org/abs/2604.04539
2. V. Mnih et al. "Human-level control through deep reinforcement learning." *Nature* 518, 2015.
3. R. Sutton and A. Barto. *Reinforcement Learning: An Introduction*, 2nd ed., §11.3 (the deadly triad), 2018.
4. H. van Hasselt et al. "Deep Reinforcement Learning and the Deadly Triad." arXiv:1812.02648, 2018. https://arxiv.org/abs/1812.02648
5. M. Gallici et al. "Simplifying Deep Temporal Difference Learning" (PQN). ICLR 2025. arXiv:2407.04811. https://arxiv.org/abs/2407.04811
6. Y. Yue et al. "Understanding, Predicting and Better Resolving Q-Value Divergence in Offline-RL." NeurIPS 2023. arXiv:2310.04411. https://arxiv.org/abs/2310.04411
7. P. Ball et al. "Efficient Online Reinforcement Learning with Offline Data" (RLPD). ICML 2023. arXiv:2302.02948. https://arxiv.org/abs/2302.02948
8. A. Kumar et al. "DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization." ICLR 2022. arXiv:2112.04716. https://arxiv.org/abs/2112.04716
9. A. Bhatt et al. "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity." ICLR 2024. arXiv:1902.05605. https://arxiv.org/abs/1902.05605
10. X. Chen et al. "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model" (REDQ). ICLR 2021. arXiv:2101.05982. https://arxiv.org/abs/2101.05982
11. T. Hiraoka et al. "Dropout Q-Functions for Doubly Efficient Reinforcement Learning" (DroQ). ICLR 2022. arXiv:2110.02034. https://arxiv.org/abs/2110.02034
12. D. Palenicek et al. "Scaling Off-Policy Reinforcement Learning with Batch and Weight Normalization." NeurIPS 2025. arXiv:2502.07523. https://arxiv.org/abs/2502.07523
13. H. Lee et al. "Hyperspherical Normalization for Scalable Deep Reinforcement Learning" (SimbaV2). ICML 2025. arXiv:2502.15280. https://arxiv.org/abs/2502.15280
14. M. Bellemare, W. Dabney, R. Munos. "A Distributional Perspective on Reinforcement Learning" (C51). ICML 2017. arXiv:1707.06887. https://arxiv.org/abs/1707.06887
15. J. Farebrother et al. "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL." ICML 2024. arXiv:2403.03950. https://arxiv.org/abs/2403.03950
16. S. Thrun and A. Schwartz. "Issues in Using Function Approximation for Reinforcement Learning." Connectionist Models Summer School, 1993.
17. H. van Hasselt. "Double Q-learning." NeurIPS 2010. See also H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double Q-learning," AAAI 2016, arXiv:1509.06461. https://arxiv.org/abs/1509.06461
18. S. Fujimoto, H. van Hoof, D. Meger. "Addressing Function Approximation Error in Actor-Critic Methods" (TD3). ICML 2018. arXiv:1802.09477. https://arxiv.org/abs/1802.09477
19. A. Kuznetsov et al. "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (TQC). ICML 2020. arXiv:2005.04269. https://arxiv.org/abs/2005.04269
20. Q. Lan et al. "Maxmin Q-learning: Controlling the Estimation Bias of Q-learning." ICLR 2020. arXiv:2002.06487. https://arxiv.org/abs/2002.06487
21. T. Moskovitz et al. "Tactical Optimism and Pessimism for Deep Reinforcement Learning" (TOP). NeurIPS 2021. arXiv:2102.03765. https://arxiv.org/abs/2102.03765
22. K. Ciosek et al. "Better Exploration with Optimistic Actor-Critic" (OAC). NeurIPS 2019. arXiv:1910.12807. https://arxiv.org/abs/1910.12807
23. Z. Wu et al. "WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation." arXiv:2608.24479, 2026 (not peer-reviewed). https://arxiv.org/abs/2608.24479
24. Y. Seo et al. "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes" (FastSAC). arXiv:2512.01996, 2025. https://arxiv.org/abs/2512.01996
25. M. Nauman et al. "Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control" (BRO). NeurIPS 2024. arXiv:2405.16158. https://arxiv.org/abs/2405.16158
26. Y. Seo et al. "FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control." arXiv:2505.22642, 2025. https://arxiv.org/abs/2505.22642
27. N. Rudin et al. "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." CoRL 2021. arXiv:2109.11978. https://arxiv.org/abs/2109.11978
28. Z. Li et al. "Parallel Q-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation" (PQL). ICML 2023. arXiv:2307.12983. https://arxiv.org/abs/2307.12983
29. K. Ota, D. K. Jha, A. Kanezaki. "Training Larger Networks for Deep Reinforcement Learning." arXiv:2102.07920, 2021. https://arxiv.org/abs/2102.07920
30. H. Lee et al. "SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning." ICLR 2025. arXiv:2410.09754. https://arxiv.org/abs/2410.09754

## Limitations 
1. We are limited in the number of ablations + ideas we can explore due to a lack of compute
2. We fail to examine sim-to-real transfer + real world performance due to a lack of a physical robot