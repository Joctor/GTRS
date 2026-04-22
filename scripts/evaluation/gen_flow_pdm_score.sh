TRAIN_TEST_SPLIT=navtrain
CHECKPOINT=$NAVSIM_EXP_ROOT/dp_ckpt/last.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/navtrain_metric_cache
experiment_name=gen_flow_pdm_score
agent=flow_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=$agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=$experiment_name \
metric_cache_path=$CACHE_PATH \
traffic_agents=reactive 

