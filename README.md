# pytorch-distributed-cpu

## Motivation:

Due to the global shortage of graphic cards (and probably booming in crypto-currency), it is extremely difficult to get graphic cards.

This induces a huge impact especially on to whom are planning to buy the GPU for their machine learning research or study.

Fortunately, as an ETH Zurich student (and I guess the same for most university students in the world), I have an access to the supercomputer cluster, [the Euler cluster](https://scicomp.ethz.ch/wiki/Euler).

This cluster allows us to utilize up to 48 CPUs, therefore, I came up with an idea to use the cluster for machine learning.

If you already have a dedicated GPU, this might not be an option. However, if you are eager to learn machine learning but have no chance to get a graphic card, this can be your option.

## How is it possible?

In [PyTorch](https://pytorch.org/), there is a module called, ```torch.nn.parallel.DistributedDataParallel```. In combination with ```torch.utils.data.DistributedSampler```, you can utilize distributed training for your machine learning project.

It is primarily developed for distributed GPU training (multiple GPUs), but recently distributed CPU training becomes possible. To enable multi-CPU training, you need to keep in mind several things.

**The example of distributed training can be found in ```distributed_test.py``` in this repository.**

- Run ```torch.multiprocess``` module for distributed training

```python
    torch.multiprocessing.spawn(run_training, args=(world_size,),
	nprocs=world_size, join=True)
```


(Here, ```run_training``` is the function where your actual training is implemented. In this example, the function has inputs of ```rank``` and ```world_size```.)

- Initialize multiprocessing environment with ```gloo``` back-end. This backend is not optimized for distributed GPU training but, if you want to use CPU as distributed environment, use ```gloo```. And don't forget to setting ```MASTER_ADDR``` and ```MASTER_PORT``` to your environment variable.

```python
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12254'

    torch.distributed.init_process_group(backend,
	rank=rank, world_size=size)

    # Disabling randomness (recommended)
    torch.manual_seed(0)
    numpy.random.seed(0)
```

- Load data using ```DistributedSampler```. If you are using this sampler, ```shuffle``` option in the ```DataLoader``` has to be False. Also, since we are not using CUDA, ```pin_memory``` option has to be False as well.

```python
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, num_workers=0, pin_memory=False, sampler=sampler
    )
```

- Finally, load the model to ```DistributedDataParallel``` module. The important thing is, you need to set ```device_ids``` as ```None``` or empty list ```[]```. And don't forget to set the device as cpu not cuda.

```python
    device = torch.device('cpu')

    model.to(device)
    model = DDP(model, device_ids=None)
```

- Now, you are ready for your training.


## Test results:
resnet50 test with Fakedata size 1000

| System             | Elapsed time [s]   |
| :----------------: | :----------------: |
| GTX2080 Ti         | 20                 |
| GTX1060 (laptop)   | 39                 |
| Euler - 4 cpus     | 239                |
| Euler - 8 cpus     | 160                |
| Euler - 12 cpus    | 144                |
| Euler - 24 cpus    | 82                 |

## Conclusion:

It is obvious that GPU training is way faster than CPU training. Even slightly outdated GTX 1060 shows 4 times better performance than 24 CPUs.

However, if you are a student, who only has a laptop without a dedicated GPU on it, and no way to get a GPU in near future, this might be a good solution. (in our case, we can use up to 48 cores -haven't tested yet-. I guess this will give more-or-less same performance with GTX 1060 for the current test case. Not bad. Right?)

Also, based on my experience, if your network is shallow (for example, two hidden layers with 256 x 256 neurons), the training bottleneck comes from CPU (data loading mostly), not GPU. In this case, distributed CPU training can outperform single GPU training.

Moreover, one other good thing about using cluster is that you can submit a job and can forget about that. Thus, your system will not scream due to GPU utilization, and your workplace will be cool as charm.

Hope this can help your study or research.
