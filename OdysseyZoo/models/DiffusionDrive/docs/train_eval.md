# DiffusionDrive Training and Evaluation

> **OdysseyZoo note.** `cache_diffusiondrive.sh`, `cache_metric_diffusiondrive.sh`, `train_diffusiondrive.sh` and `eval_diffusiondrive.sh` (in this model's folder, `models/DiffusionDrive/`) wrap the commands below for both arms (`baseline`, `sdroute`). Differences from upstream:
> - `cache_diffusiondrive.sh` caches with `diffusiondrive_sdroute_agent`, so one cache serves both arms. A cache built with the baseline agent, as below, lacks the route and the `sdroute` arm refuses it.
> - `diffusiondrive_baseline_agent` is camera-only (`latent: True`). Upstream's released checkpoint was trained with LiDAR, so evaluating it needs `agent.config.latent=False`.
> - `plan_anchor_path` defaults to the vendored `assets/kmeans_navsim_traj_20.npy`. `bkb_path` is needed only when timm cannot reach the Hugging Face hub. The downloads and config edits in section 2 are therefore optional.

## 1. Cache dataset for faster training and evaluation
```bash
# cache dataset for training
python navsim/planning/script/run_dataset_caching.py agent=diffusiondrive_baseline_agent experiment_name=training_diffusiondrive_agent train_test_split=navtrain

# cache dataset for evaluation
python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

## 2. Training
If your training machine does not have network access, you should download the pretrained ResNet-34 model from [huggingface](https://huggingface.co/timm/resnet34.a1_in1k) and upload it to your training machine. You should also download the [clustered anchors](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy)

Before starting training, ensure that you correctly set the `bkb_path` to the path of the downloaded pretrained ResNet-34 model. Additionally, set the `plan_anchor_path` to the path of the downloaded clustered anchors in the file located at `/path/to/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_config.py`.

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=diffusiondrive_baseline_agent \
        experiment_name=training_diffusiondrive_agent  \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 
```

## 3. Evaluation
You can use the following command to evaluate the trained model `export CKPT=/path/to/your/checkpoint.pth`, for example, you can download the our provided checkpoint from [huggingface](https://huggingface.co/hustvl/DiffusionDrive), and set `CKPT=/path/to/downloaded/huggingface_diffusiondrive_agent_ckpt/diffusiondrive_navsim_88p1_PDMS.pth`:
```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=diffusiondrive_baseline_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        agent.config.latent=False \
        experiment_name=diffusiondrive_agent_eval
```
`agent.config.latent=False` matches upstream's LiDAR checkpoint; drop it for a checkpoint trained with the command above.

