## EDM repo
documentation by sqa

## Preparing datasets

**AFHQv2:** 

1. Open [this dropbox url](https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0) and download the `afhq_v2.zip` file.

2. put the `afhq_v2.zip` file in the `downloads` directory under `edm-sqa`.

3. Run

```.bash
unzip ./downloads/afhq_v2.zip -d ./downloads/afhqv2/
```

Then hopefully you will see `/train` and `/test` under `./downloads/afhqv2/`.

4. preprocess into 64x64 and calculate FID ref:

```.bash
python dataset_tool.py --source=downloads/afhqv2 \
    --dest=datasets/afhqv2-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/afhqv2-64x64.zip --dest=fid-refs/afhqv2-64x64.npz
```

5. sanity: (which will take 16 hours on one 4090, not recommended)

```.bash
# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=1 generate.py --steps=40 --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl

# Calculate FID
torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=fid-refs/afhqv2-64x64.npz
```

6. Another option: eval the FID for two different refs (our calculated and edm provided)

```.bash
torchrun --standalone --nproc_per_node=1 fid.py sqa --edm_path=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz \
    --sqa_ref=fid-refs/afhqv2-64x64.npz
```

the result should be about 5e-6.

**FFHQ:** 

1. follow the instructions in `https://github.com/qiaosungithub/ffhq-sqa.git`, and we get a folder `\images1024x1024`.

2. Move the folder to `downloads/ffhq/images1024x1024`.

3. preprocess into 64x64 and calculate FID ref:

```.bash
python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
    --dest=datasets/ffhq-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz
```

4. eval the FID for two different refs (our calculated and edm provided)

```.bash
torchrun --standalone --nproc_per_node=1 fid.py sqa --edm_path=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz \
    --sqa_ref=fid-refs/ffhq-64x64.npz
```

the result should be about 3e-6.

## Env

```.bash
conda env create -f environment-H100.yml

conda activate edm-H100
```

__sanity__: try run

```.bash
torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs \
    --data=datasets/afhqv2-64x64.zip --cond=0 --arch=ddpmpp --batch=4 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15 --tick=1
```

其中有可能要更改
1. torch version: 需要支持 H100. 现有的脚本是符合4090的.
2. numpy version: 需要和torch兼容, 反正就反复调一下重新搞个环境.

直到出现
```.bash
model.dec.64x64_block1    541952      -        [2, 128, 64, 64]  float32 
model.dec.64x64_block2    541952      -        [2, 128, 64, 64]  float32 
model.dec.64x64_block3    541952      -        [2, 128, 64, 64]  float32 
model.dec.64x64_block4    541952      -        [2, 128, 64, 64]  float32 
model.dec.64x64_aux_norm  256         -        [2, 128, 64, 64]  float32 
model.dec.64x64_aux_conv  3459        -        [2, 3, 64, 64]    float32 
model                     1152        -        [2, 3, 64, 64]    float32 
<top-level>               -           -        [2, 3, 64, 64]    float32 
---                       ---         ---      ---               ---     
Total                     61805571    48       -                 -       

Setting up optimizer...
Training for 200000 kimg...

tick 0     kimg 0.0       time 12s          sec/tick 4.5     sec/kimg 2272.16 maintenance 7.4    cpumem 3.05   gpumem 9.34   reserved 12.68
```
就说明成功了!

## Training new models

### AFHQv2 64x64:

run **VP**:

```.bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/afhqv2-64x64.zip --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15
```

run **VE**:

```.bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/afhqv2-64x64.zip --cond=0 --arch=ncsnpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15
```

__no time condition experiment__: add argument `--notime=True`.

__For sanity check__: in the command below, change `--network` to the path of the latest network snapshot in the training directory.

```.bash
rm -rf fid-tmp
mkdir fid-tmp

torchrun --standalone --nproc_per_node=1 generate.py --steps=40 --outdir=fid-tmp --seeds=0-9 --subdirs \
    --network=training-runs/<name_of_exp>/network-snapshot-*.pkl
```

__For evaluating FID__: in the command below, change `--network` to the path of the latest network snapshot in the training directory.

```.bash
rm -rf fid-tmp
mkdir fid-tmp

# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=8 generate.py --steps=40 --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=training-runs/<name_of_exp>/network-snapshot-*.pkl

# Calculate FID
torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid-tmp \
    --ref=fid-refs/afhqv2-64x64.npz
```

### FFHQ 64x64:

run **VP**:

```.bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
```

run **VE**:

```.bash
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/ffhq-64x64.zip --cond=0 --arch=ncsnpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
```

__For sanity check__: in the command below, change `--network` to the path of the latest network snapshot in the training directory.

```.bash
rm -rf fid-tmp
mkdir fid-tmp

torchrun --standalone --nproc_per_node=1 generate.py --steps=40 --outdir=fid-tmp --seeds=0-9 --subdirs \
    --network=training-runs/<name_of_exp>/network-snapshot-*.pkl
```

__For evaluating FID__: in the command below, change `--network` to the path of the latest network snapshot in the training directory.

```.bash
rm -rf fid-tmp
mkdir fid-tmp

# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=8 generate.py --steps=40 --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=training-runs/<name_of_exp>/network-snapshot-*.pkl

# Calculate FID
torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid-tmp \
    --ref=fid-refs/ffhq-64x64.npz
```

Here are documentations by edm repo:

Default batch size: 512 (controlled by `--batch`) that is divided evenly among 8 GPUs (controlled by `--nproc_per_node`) to yield 64 images per GPU. If GPU OOM, use `--batch-gpu=32`. This will lead to exactly the same result.

The results of each training run are saved to a newly created directory, for example `training-runs/00000-cifar10-cond-ddpmpp-edm-gpus8-batch64-fp32`. The training loop exports network snapshots (`network-snapshot-*.pkl`) and training states (`training-state-*.pt`) at regular intervals (controlled by `--snap` and `--dump`). The network snapshots can be used to generate images with `generate.py`, and the training states can be used to resume the training later on (`--resume`). Other useful information is recorded in `log.txt` and `stats.jsonl`. To monitor training convergence, we recommend looking at the training loss (`"Loss/loss"` in `stats.jsonl`) as well as periodically evaluating FID for `network-snapshot-*.pkl` using `generate.py` and `fid.py`.
