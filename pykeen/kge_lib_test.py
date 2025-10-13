from pykeen.pipeline import pipeline


model_name='ComplEx'
dataset_name='FB15k_237'


pipeline_result = pipeline(
    random_seed=42,
    model=model_name,
    dataset=dataset_name,
    training_loop='sLCWA',
    negative_sampler='basic',
    evaluator='RankBasedEvaluator',
    stopper='early',
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs=dict(
        gamma=0.99,
    ),
    # model_kwargs=dict(
    #     scoring_fct_norm=2,
    # ),
)
# Handle complex-valued embeddings for plotting
try:
    pipeline_result.plot()
except ValueError as e:
    if "Complex data not supported" in str(e):
        print("Warning: Complex-valued embeddings detected. Skipping embedding visualization.")
        print("Training completed successfully. Loss plots are still available.")
    else:
        raise e

pipeline_result.plot_losses()

import os
import time
import uuid

pykeen_result_folder='pykeen_result'
model_name=f'{model_name}_{dataset_name}'
base_save_dir=f'{pykeen_result_folder}/{model_name}'

# 檢查資料夾是否存在，如果存在就加上時間戳和隨機ID
save_dir = base_save_dir
counter = 1
while os.path.exists(save_dir):
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]  # 取前8位作為簡短ID
    save_dir = f'{base_save_dir}_{timestamp}_{unique_id}'
    counter += 1
    if counter > 100:  # 防止無限循環
        break

print(f'Saving model to: {save_dir}')
pipeline_result.save_to_directory(save_dir)