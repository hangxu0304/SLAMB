# SLAMB [[ICML 2023]](https://openreview.net/pdf?id=cMmjBH5LqW)
Large Batch Optimizer with Sparse Communication

```
@inproceedings{
  xu2023slamb,
  title={SLAMB: Accelerated Large Batch Training with Sparse Communication},
  author={Xu, Hang and Zhang, Wenxuan and Fei, Jiawei and Wu, Yuzhe and Xie, TingWen and Huang, Jun and Xie, Yuchen and Elhoseiny, Mohamed and Kalnis, Panos},
  booktitle={The International Conference on Machine Learning},
  year={2023}
}
```


## Content
- [Prerequisites](#prerequisites)
- [Code](#code)
- [Training](#training)
- [Todos](#todos)

## Prerequisites

The code is built with following libraries:
- Python >= 3.9
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.8
- [Apex](https://github.com/NVIDIA/apex) >= 20.0 (Optional)

## Code

The core code of SLAMB is in [Cifar10/main.py](Cifar10/main.py) and [Swin-Transformer/optimizer.py](Swin-Transformer/optimizer.py), where `SLAMB` is the naive implementation and `SLAMB_v2` is the optimized implementation.


## Training
For detailed experiment setup, please see the `README` in each subfolder. The launch scripts are in [Cifar10/scripts](Cifar10/scripts) and [Swin-Transformer/scripts](Swin-Transformer/scripts).

Try the Cifar10 example:
```
git clone https://github.com/hangxu0304/SLAMB.git
cd SLAMB/Cifar10
python3 main.py --data-dir=./data -a=resnet110 --batch-size=256 --lr=0.01 --optimizer=SLAMB --compress_ratio=0.1 --beta3=0.99
```

Cifar10 results (256 batch size * 4 GPUs):

![result-fig.png](Cifar10%2Fresults%2Fresult-fig.png)

|           | **global batch size** | **learning rate** | **acc@top1** |
|-----------|-----------------------|-------------------|--------------|
| **SGD-M** | 1024                  | 0.03              | 91.70%       |
| **LAMB**  | 1024                  | 0.01              | 93.15%       |
| **SLAMB** | 1024                  | 0.01              | 93.21%       |


## Todos

- **Hyper-parameters**: We recommend to fix the compression ratio (e.g. k=0.1) and then fine-tune $\beta_3$. `Local step (H)` requires less effort to optimize. Usually we set H=50 for small datasets like Cifar10 and H=100 for large datasets like ImageNet and wikipedia. 
- **Top-K**: We currently only support `Random-k` sparsified communication in SLAMB. `Top-K` is promising, but it requires further study on how to estimate the layer-wise scaling coefficients. 
- **Quantization**: Combine `Gradient Quantization` with SLAMB can further reduce the communication overhead. We have tested it with QSGD (8 bits) and Natural Compression (8 bits) and the results are promising.
- **Local reduction**: To enable extreme large scale training, performing local reduction of gradients in each node can improve the convergence. 
- **Overhead**: Implement fused cuda kernels may further reduce the computation overhead and memory footprint.

## License

See [LICENSE](LICENSE) for additional details.


## Acknowledgement
- Our implementation is based on [LAMB](https://github.com/cybertronai/pytorch-lamb), a large batch optimizer for deep learning.
- Experiment scripts are adapted from [pytorch_resnet_cifar10 (Yerlan Idelbayev)](https://github.com/akamaster/pytorch_resnet_cifar10) and [Swin-Transformer (Microsoft)](https://github.com/microsoft/Swin-Transformer).
