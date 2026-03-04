TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=$NAVSIM_EXP_ROOT/dp_ckpt/epoch-02-step-0009.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/navhard_two_stage_metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
experiment_name=inference_flow
agent=flow_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_v2.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=$agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=$experiment_name \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
