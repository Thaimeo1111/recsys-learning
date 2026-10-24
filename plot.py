import matplotlib.pyplot as plt
import numpy as np

def plot_metric_by_param(results, xlabel, baseline=None, title=None):
 """
 results:  {giá_trị_tham_số: kết quả evaluate}, ví dụ {1: {...}, 5: {...}}
 xlabel:   tên tham số trục x, ví dụ "số lượt chấm tối thiểu N"
 baseline: (tùy chọn) {"popularity": kết quả evaluate} để vẽ đường nằm ngang
 """
 xs = list(results)
 fig, ax = plt.subplots(figsize=(7, 4))

 ax.plot(xs, [results[x]["ndcg@k"] for x in xs],
         marker="o", label="NDCG")
 ax.plot(xs, [results[x]["recall@k"] for x in xs],
         marker="o", label="Recall")

 if baseline:
  for name, res in baseline.items():
   ax.axhline(res["ndcg@k"], linestyle="--", color="C0", alpha=0.5,
              label=f"NDCG {name}")
   ax.axhline(res["recall@k"], linestyle="--", color="C1", alpha=0.5,
              label=f"Recall {name}")

 ax.set_xlabel(xlabel)
 ax.set_ylabel("điểm")
 if title:
  ax.set_title(title)
 ax.legend()
 plt.tight_layout()
 plt.show()

def plot_distribution(results: dict, title: str):
 """results = {"random": kết quả evaluate, "popularity": kết quả evaluate}"""
 fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 for ax, key, name in [(axes[0], "recalls", "Recall@k"),
                       (axes[1], "ndcgs", "NDCG@k")]:
  for label, res in results.items():
   ax.hist(res[key], bins=20, range=(0, 1), alpha=0.5, label=label)
   ax.axvline(np.mean(res[key]), linestyle="--",
              color=ax.patches[-1].get_facecolor())
  ax.set_xlabel(name)
  ax.set_ylabel("số người dùng")
  ax.set_title(f"{title}: {name}")
  ax.legend()
 plt.tight_layout()
 plt.show()