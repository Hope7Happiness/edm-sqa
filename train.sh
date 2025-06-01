source env.sh
date > cur.log
echo >> cur.log
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp \
    2>&1 | tee -a cur.log