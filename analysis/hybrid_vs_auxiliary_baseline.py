import matplotlib.pyplot as plt

data = {
    "OpenML 1563": -0.0572,
    "OpenML 851": -0.0517,
    "clean1 / OpenML 40665": -0.0457,
    "OpenML 828": -0.0277,
    "OpenML 718": -0.0232,
    "OpenML 1017": -0.0180,
    "PC-GITA AWPE3": -0.0120,
    "PC-GITA F0WPE3": -0.0038,
    "Breast Cancer": -0.0002,
    "PC-GITA F0WPE4": 0.0005,
    "Sonar": 0.0052,
    "PC-GITA AWPE4": 0.0060,
    "Banknote": 0.0085,
    "UCI174 Parkinson": 0.0108,
    "Ionosphere": 0.0117,
    "OpenML 55": 0.0132,
    "Spambase": 0.0217,
}

datasets = list(data.keys())
values = list(data.values())

fig, ax = plt.subplots(figsize=(8, 7))

y = range(len(datasets))
ax.barh(y, values)

ax.set_yticks(y)
ax.set_yticklabels(datasets)
ax.invert_yaxis()

ax.axvline(0, linewidth=1)

ax.set_xlabel("Mean performance difference (HYBRID − baseline)")

ax.set_xlim(-0.07, 0.05)

for i, v in enumerate(values):
    ha = "left" if v >= 0 else "right"
    offset = 0.001 if v >= 0 else -0.001
    ax.text(
        v + offset,
        i,
        f"{v:+.4f}",
        va="center",
        ha=ha
    )

fig.tight_layout()
fig.savefig("delta_mitja_per_dataset.pdf", bbox_inches="tight")
plt.close(fig)
