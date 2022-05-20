import numpy as np
import matplotlib.pyplot as plt

data = []
for i in range(1, 5):
    data.append(np.load(f"eval/no_question_trigram_repeat_prophetnet_two_qg_{2*i}_samples.npy"))

fig, ax = plt.subplots()
ax.set_title("Average Question PINC Scores for Number of Sampled Questions")
ax.set_xlabel("Number of samples")
ax.set_ylabel("Average PINC score")
ax.boxplot(data, labels=[2, 4, 6, 8], showfliers=False)
plt.savefig("boxplot.png")
