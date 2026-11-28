## Changes
1. Online (no target network)
2. 1 critic (remove 2nd critic) or 1 wide critic (remove 2nd critic, increase width to double trainable params of the first critic)
3. Asymmetric critic
Note: architectural changes did not yield impressive results (see rafay folder)

Trainable params (from the saved checkpoints; the target network is not trained, so it is not counted):

| Critic params | Flat-G1 | Rough-G1 | Flat-H1 | Rough-H1 |
|---|---|---|---|---|
| Online (2 critics, hidden 256) | 2,242,378 | 2,338,870 | 2,205,226 | 2,301,718 |
| 1 critic (hidden 256) | 1,121,189 | 1,169,435 | 1,102,613 | 1,150,859 |
| 1 wide critic (hidden 360) | 2,175,541 | 2,243,235 | 2,149,477 | 2,217,171 |
| Wide critic vs. Online | −3.0% | −4.1% | −2.5% | −3.7% |
| Actor (same for all three) | 290,368 | 314,678 | 278,704 | 303,014 |

## Main Runs
- [x] Baseline
- [x] Online
- [x] 1 critic
- [x] 1 wide critic (nish is running this right now)
- [ ] asymmetric critic
- [x] Online + 1 critic (nish is running this right now)
- [x] Online + 1 wide critic (nish is running this right now)
- [ ] Online + 1 critic + asymmetric critic
- [ ] Online + 1 wide critic + asymmetric critic

## Additional Ablations
- We make 9 main runs to understand the importance of each of our proposed 3 changes
- The project description recommends "showing ablations beyond just the accuracy and explain why the intuition was right/wrong with the help of those ablations"
- We can make 2 additional runs to understand just how relevant critic width is 
    - WarpSAC does not ablate critic width
    - FlashSAC does ablate critic width, but it does not make our proposed changes
- We already look at 2 widths for the critic (1 critic, 1 wide critic)
- We can make 2 final runs to study the effect of critic width by looking at critics wider than the wide critic that we use
- Define the 2xWide critic and 4xWide critic as follows:
    1. 2xWide critic: a critic with the same depth and architecture of the wide critic but hidden dim = 512 (roughly 2x the trainable params of the wide critic)
    1. 4xWide critic: a critic with the same depth and architecture of the wide critic but hidden dim = 736 (roughly 4x the trainable params of the wide critic)
- We can make 2 final runs to study the effect of critic width
- [ ] Online + 1 2xWide critic + asymmetric critic
- [ ] Online + 1 4xWide critic + asymmetric critic

## Note on Timing
- We graph multiple time-based statistics. These stats are influeneced by the hardware that we run on.
- We have 2 issues
    1. Running on Shiyu's local lab setup does not give an exactly fair time comparison to running on AWS.
    2. The asymmetric critic network has a higher input dimension (as the model receives both the clean and noisy observations). As such, the replay buffer size increases. Since FlashSAC has such a massive buffer, we can no longer fit this on GPU. I found that, when running on a g6e.4xlarge instance, in which I moved the replay buffer to CPU to allow us to still use the same L40s GPU, moving the replay buffer to CPU incurs an additional time penalty. 
- We can do time based statistics for all symmetric runs and look at step based statistics for all assymetric runs. I (Nish) have ran out of AWS credits, so I can not run anything else. THe g6e.4xlarge requires a quota of 16 vCPUs (which I don't believe that anyone else has), so Shiyu will likely have to run all assymetric runs on his lab setup.

## Extras (Not 100% necessary, can be cut if we're short on compute or time)
- These runs are not absolutely necessary, less important than additional ablations, and can be cut if need be
    - [ ] Online + asymmetric critic 
    - [ ] 1 critic + asymmetric critic
    - [ ] 1 wide critic + asymmetric critic 
- I think that we should cut these runs