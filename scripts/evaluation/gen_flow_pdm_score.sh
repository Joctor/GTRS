TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/navtrain_metric_cache
experiment_name=gen_flow_pdm_score
agent=human_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/gen_flow_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=$agent \
    experiment_name=$experiment_name \
    metric_cache_path=$CACHE_PATH \
    traffic_agents=reactive 

