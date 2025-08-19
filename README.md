# Efficient AoE Implementation

This is an implementation of AoE based on [megablocks](https://github.com/databricks/megablocks). This implementation achieves up to 80% throughput even in 3-of-64 MoE configurations (the specific speed depends on the block size set in megablocks).

Ensure megablocks is installed, clone this repository, then you can import and use the ```VanillaAoE``` class in your project.

## Important Notes
This implementation does not automatically adjust the FFN intermediate size based on d_low to match MoE parameter counts. You must manually ensure:
- d_low and d_wide (see the paper for more details) are all multiples of the block size.
- Parameter counts are comparable to your target MoE setup (adjust dimensions manually to achieve this).
Running the code without verifying the above constraints may result in dimension mismatches, runtime errors, or inconsistent parameter counts.


If you find this code useful, please cite our paper.

```
@inproceedings{
lv2025autonomyofexperts,
title={Autonomy-of-Experts Models},
author={Ang Lv and Ruobing Xie and Yining Qian and Songhao Wu and Xingwu Sun and Zhanhui Kang and Di Wang and Rui Yan},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=8BIDrYWCeg}
}
```
