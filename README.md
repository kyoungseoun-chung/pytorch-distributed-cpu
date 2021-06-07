# pytorch-distributed-cpu

## Motivation:

Due to global shortage of the graphic cards (and probably booming in crypto-currency), it is extremely difficult to get the graphic cards.

This induces huge impact especially to whom are planning to buy the GPU for their machine learning research or study.

Fortunately, as a ETH Zurich student (and I guess same for most of university students in the world), I have an access to the supercomputer cluster, [the Euler cluster](https://scicomp.ethz.ch/wiki/Euler).

This cluster allow us to utilize up-to 48 CPUs, therefore, I came up with an idea to use the cluster for the machine learning.

If you already have dedicated GPU, this might not be an option. However, if you are eager to learn machine learning but have no chance to get a graphic card, this can be your option.

## How is possible?

In [PyTorch](https://pytorch.org/), there is a module called, *DistributedDataParallel*. In combination with *DistributedSampler*, you can utilize distributed training for your machine learning project.

It is primarily developed for distributed GPU training (multiple-GPUs), but recently distributed CPU training becomes possible. To enable multi-CPU training, you need to keep in mind several things.

**The example of distributed training can be found in ```distributed_test.py``` in this repository.**

- Run ```torch.multiprocess``` module for distributed training

```python
    torch.multiprocessing.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)
```

- Initialize multiprocessing environment with ```gloo``` back-end. This backend is not optimized for distributed GPU training but, if you want to use CPU as distributed environment, use ```gloo```. And don't forget to setting ```MASTER_ADDR``` and ```MASTER_PORT``` to your environment variable.

```python
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12254'

    torch.distributed.init_process_group(backend, rank=rank, world_size=size)

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

- Finally, load the model to ```DistributedDataParallel``` module. The important thing is, you need to set ```device_ids``` as ```None`` or empty list ```[]```.

```python
    device = torch.device('cpu')

    model.to(device)
    model = DDP(model, device_ids=None)
```

- Now, you are ready for your training.


## Test results:
resnet50 test with Fakedata size 1000

|      System      	| Elapsed time [s] 	|
|:----------------:	|:----------------:	|
| GTX1060 (laptop) 	|        39        	|
|  Euler - 4 cpus  	|        239       	|
|   Euler - 8 cpus  	 |         160       	 |
|   Euler - 12 cpus    |         144          |
|   Euler - 24 cpus    |          82          |
