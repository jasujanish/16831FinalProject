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
- [ ] asymmetric critic (done by Shiyu, I believe)
- [x] Online + 1 critic
- [x] Online + 1 wide critic
- [x] Online + 1 critic + asymmetric critic (note: g6e.2x large runs are currently running, g6e.4xlarge runs are not done yet, will be done soon)
- [x] Online + 1 wide critic + asymmetric critic (note: g6e.2x large runs are currently running, g6e.4xlarge runs are not done yet, will be done soon)

## Additional Ablations
- We make 12 main runs to understand the importance of each of the features
- We can make 2 additional runs to understand just how relevant critic depth is 
- We already look at 2 widths for the critic (1 critic, 1 wide critic)
- We can make 2 final runs to study the affect of critic width
- [ ] Online + 1 wide 2 critic (2x the trainable params of the wide critic, same depth) + asymmetric critic
- [ ] Online + 1 wide 4 critic (4x the trainable params of the wide critic, same depth) + asymmetric critic

## Note on Timing
- Running on Shiyu's local lab setup does not give an exactly fair time comparison to running on AWS
- For the additional ablations, we can just make steps vs td error or steps vs performance, so we can use the results from the local lab setup without issue

## Extras (Not 100% necessary, can be cut if we're short on compute)
- [ ] Online + asymmetric critic (not absolutely necessary, less important than additional ablations, can be cut if need be)
- [ ] 1 critic + asymmetric critic (not absolutely necessary, less important than additional ablations, can be cut if need be)
- [ ] 1 wide critic + asymmetric critic (not absolutely necessary, less important than additional ablations, can be cut if need be)
