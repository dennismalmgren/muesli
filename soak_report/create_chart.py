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

#entity = "dennisathome"
#project = "opus_smoothing"
#run_id = "OPUS_opus_smoothing_b957736a_24_05_20-11_33_26"
run_path = "dennisathome/opus_smoothing/hkezjtt8"
#run = api.run(f"{entity}/{project}/{run_id}")
run = api.run(run_path)
metric_id = "eval/episode_OpusSmoothingReward_heading"
history = run.history()

steps = []
metric = []

history = run.scan_history(keys=["trainer/step", metric_id])
for row in history:
    if row.get(metric_id) is not None:
        steps.append(row["trainer/step"])
        metric.append(row[metric_id])

plt.figure(figsize=(10, 6))
metric = np.asarray(metric)
metric = exponential_moving_average(metric)
plt.plot(steps, metric, label='Search')

metric_id = "eval/episode_OpusSmoothingReward_speed"
metric = []
steps = []

history = run.scan_history(keys=["trainer/step", metric_id])
for row in history:
    if row.get(metric_id) is not None:
        steps.append(row["trainer/step"])
        metric.append(row[metric_id])
metric = np.asarray(metric)
metric = exponential_moving_average(metric)

plt.plot(steps, metric, label='Trajectory')

metric_id = "eval/episode_OpusSmoothingReward_alt"
metric = []
steps = []

history = run.scan_history(keys=["trainer/step", metric_id])
for row in history:
    if row.get(metric_id) is not None:
        steps.append(row["trainer/step"])
        metric.append(row[metric_id])
metric = np.asarray(metric)
metric = exponential_moving_average(metric)

plt.plot(steps, metric, label='Pursuit')

plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.title('Task reward')
plt.legend()

plt.show()

print('ok')