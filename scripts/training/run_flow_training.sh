TRAIN_TEST_SPLIT=navtrain
config="default_training" # this config uses the entire navtrain dataset for training
experiment_name=train_flow
agent=flow_agent
cache_path=$NAVSIM_EXP_ROOT/flow_agent_navtrain_dataset_caching/
metric_cache_path=$NAVSIM_EXP_ROOT/navtrain_metric_cache/

# training hyper-parameters
lr=0.0002
bs=16 #96GB
max_epochs=2

torchrun --nproc_per_node=gpu $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
    --config-name ${config} \
    agent=${agent} \
    experiment_name=${experiment_name} \
    train_test_split=$TRAIN_TEST_SPLIT \
    dataloader.params.batch_size=${bs} \
    trainer.params.max_epochs=${max_epochs} \
    trainer.params.precision=32 \
    agent.lr=${lr} \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    cache_path=$cache_path \
    metric_cache_path=$metric_cache_path \
    #+resume_ckpt_path=/root/ckpt/last.ckpt
    # trainer.params.accelerator=cpu
    # trainer.params.num_nodes=${NUM_NODES} \
    # agent.config.ckpt_path=${experiment_name} \


