import numpy as np
import pandas as pd

POSITIVE = 4.0  # rating >= 4 được coi là "thích"

def load(path="ml-latest-small/"):
 return pd.read_csv(path + "ratings.csv"), pd.read_csv(path + "movies.csv")

def time_split(ratings, test_frac=0.2):
 """Chia theo thời gian: train là quá khứ, test là tương lai."""
 cutoff = ratings.timestamp.quantile(1 - test_frac)
 return ratings[ratings.timestamp < cutoff], ratings[ratings.timestamp >= cutoff]

def evaluate(recommend_fn, train, test, k=10):
 """recommend_fn(user_id, seen_items:set, k) -> list[movieId]"""
 seen = train.groupby("userId").movieId.apply(set).to_dict()
 truth = (test[test.rating >= POSITIVE]
          .groupby("userId").movieId.apply(set).to_dict())
 recalls, ndcgs, n_rel, skipped = [], [], [], 0
 for u, relevant in truth.items():
  if u not in seen:          # người dùng chưa có lịch sử: cold start
   skipped += 1
   continue
  recs = recommend_fn(u, seen[u], k)
  hits = [1 if m in relevant else 0 for m in recs]
  recalls.append(sum(hits) / min(len(relevant), k))
  dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
  idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
  ndcgs.append(dcg / idcg)
  n_rel.append(len(relevant))
 return {"recall@k": round(float(np.mean(recalls)), 4),
         "ndcg@k": round(float(np.mean(ndcgs)), 4),
         "users_evaluated": len(recalls),
         "users_skipped_cold": skipped,
         "recalls" : recalls, "ndcgs" : ndcgs, "n_relevant" : n_rel}