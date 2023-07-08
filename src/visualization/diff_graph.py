import dill
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("..") / ".." / "data"
FIGURE_PATH = Path("..") / ".." / "figures"

with open(DATA_PATH / 'fine_tune_data.pkl', 'br') as f:
    error = dill.load(f)

grid = sns.JointGrid(x=error[0], y=error[1], space=0, xlim=(0.005, 1), ylim=(0.005, 1))
grid.plot_joint(plt.scatter, s=15, alpha=0.5)
grid.ax_joint.plot([0.001, 1], [0.001, 1], 'k-', linewidth=1)
grid.fig.suptitle("Ground Truth Error Before and After Fine-Tuning")
grid.set_axis_labels('MSE Error Before Fine-Tuning', 'MSE Error After Fine-Tuning')

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.savefig(FIGURE_PATH / "Fine-Tuning")
plt.show()

