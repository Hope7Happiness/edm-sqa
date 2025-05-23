# EDM on CIFAR

1. Download CIFAR-10 dataset

```
mkdir -p downloads/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O downloads/cifar10/cifar-10-python.tar.gz
```

2. process the CIFAR dataset and calculate FID reference:

```
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

3. FID sanity check:

```
torchrun --standalone --nproc_per_node=1 fid.py sqa --edm_path=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz \
    --sqa_ref=fid-refs/cifar10-32x32.npz
```

On my machine, this corrsponds to a FID of `0.00113984`. This number may depend on the specific environment, but it should be no larger thanthe order of `1e-3`.

4. Run training script:

```
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp
```

5. Eval FID

```
rm -rf fid-tmp
mkdir fid-tmp

torchrun --standalone --nproc_per_node=8 generate.py --steps=18 --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=training-runs/<name_of_exp>/network-snapshot-*.pkl

torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=fid-refs/cifar10-32x32.npz
```

This should give the result of around `1.97`. Notice that you should replace `<name_of_exp>` with the corresponding folder name in the directory `training-runs`. 


