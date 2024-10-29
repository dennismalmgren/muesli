import wandb
import matplotlib.pyplot as plt
import numpy as np

wandb.login()

api = wandb.Api()
alpha = 0.6
def exponential_moving_average(data, alpha=0.6):
    ema = []
    ema.append(data[0])  # The first value of EMA is just the first value of the data
    for t in range(1, len(data)):
        ema_value = alpha * data[t] + (1 - alpha) * ema[-1]
        ema.append(ema_value)
    return np.array(ema)

############################
# PPO training w/ smoothing (worst)

run_path_ppo_smoothing = "dennisathome/opus_smoothing/hkezjtt8"
#run = api.run(f"{entity}/{project}/{run_id}")
run = api.run(run_path_ppo_smoothing)
metric_id = "train/OpusSmoothingReward_alt"
history = run.history()

steps = []
metric = []
last_step_id = -1
history = run.scan_history(keys=["trainer/step", metric_id])
for row in history:
    if row.get(metric_id) is not None:
        steps.append(row["trainer/step"])
        metric.append(row[metric_id])
        if steps[-1] > 15000000 and last_step_id == -1:
            last_step_id = len(steps) - 1

plt.figure(figsize=(10, 6))
metric = np.asarray(metric[:last_step_id])
metric = exponential_moving_average(metric)
plt.plot(steps[:last_step_id], metric, label='PPO + Smoothing')
############################
# PPO training

run_path_ppo = "dennisathome/opus_smoothing/yasbi7sl"

run = api.run(run_path_ppo)
metric_id = "train/OpusSmoothingReward_alt"
metric = []
steps = []
last_step_id = -1
history = run.scan_history(keys=["trainer/step", metric_id])
for row in history:
    if row.get(metric_id) is not None:
        steps.append(row["trainer/step"])
        metric.append(row[metric_id])
        if steps[-1] > 15000000 and last_step_id == -1:
            last_step_id = len(steps) - 1

metric = np.asarray(metric[:last_step_id])
metric = exponential_moving_average(metric)

plt.plot(steps[:last_step_id], metric, label='PPO')
############################

# Cyclic training (best)
run_path_ppo_cyclic = "dennisathome/opus_smoothing/nba4bkl2"
run = api.run(run_path_ppo_cyclic)

metric_id = "train/OpusSmoothingReward_alt"
metric = []
steps = []
last_step_id = -1
history = run.scan_history(keys=["trainer/step", metric_id])
for row in history:
    if row.get(metric_id) is not None:
        steps.append(row["trainer/step"])
        metric.append(row[metric_id])
        if steps[-1] > 15000000 and last_step_id == -1:
            last_step_id = len(steps) - 1

metric = np.asarray(metric[:last_step_id])

metric = exponential_moving_average(metric)

plt.plot(steps[:last_step_id], metric, label='PPO+Cyclic')

plt.xlabel('Training Steps')
plt.ylabel('Averaged Reward')
plt.title('Task reward (average)')
plt.legend()

plt.show()

print('ok')