## Changes
1. Online (no target network)
2. 1 critic (remove 2nd critic) or 1 wide critic (remove 2nd critic, double width of first critic)
3. Asymmetric critic
Note: architectural changes were not impressive enough

## Main Runs
- [x] Baseline
- [x] Online
- [ ] 1 critic
- [ ] 1 wide critic
- [ ] asymmetric critic
- [x] Online + 1 critic
- [x] Online + 1 wide critic
- [x] Online + 1 critic + asymmetric critic (note: g6e.4xlarge runs (rough tasks) are currently running, will be done ~2 am EST 11/28. g6e.2x large runs (flat tasks) are partially done (3 of 6), the rest still need to be run)
- [x] Online + 1 wide critic + asymmetric critic (note: g6e.4xlarge runs (rough tasks) are currently running, will be done ~2 am EST 11/28. g6e.2x large runs (flat tasks) have not been run yet)

## Additional Ablations
- We make 12 main runs to understand the importance of each of the features
- We can make 2 additional runs to understand just how relevant critic depth is 
- We already look at 2 widths for the critic (1 critic, 1 wide critic)
- We can make 2 final runs to study the affect of critic width
- [ ] Online + 1 wide 2 critic (2x the trainable params of the wide critic, same depth) + asymmetric critic
- [ ] Online + 1 wide 4 critic (4x the trainable params of the wide critic, same depth) + asymmetric critic

## Note on Timing
- We graph multiple time-based statistics. These stats are influeneced by the hardware that we run on.
- Running on Shiyu's local lab setup does not give an exactly fair time comparison to running on AWS.
- For the main runs, we probably should graph time-based stats. For the additional ablations, we can just graph step-based states (ex: return vs steps, critic performance vs steps), so we can use the results from the local lab setup without issue.

## Extras (Not 100% necessary, can be cut if we're short on compute)
- These runs are not absolutely necessary, less important than additional ablations, and can be cut if need be
- [ ] Online + asymmetric critic 
- [ ] 1 critic + asymmetric critic
- [ ] 1 wide critic + asymmetric critic
