import random
from common import load, time_split, evaluate, POSITIVE
from plot import plot_distribution,plot_metric_by_param
ratings, movies = load()
train, test = time_split(ratings)

# --- Hệ thống 1: phổ biến nhất (đếm lượt thích) ---
pos = train[train.rating >= POSITIVE]
ranked_by_count = pos.movieId.value_counts().index.tolist()
#E1
stats = train.group_by("movieId").rating.agg(["mean","count"])

def make_ranking(min_count):
 s = stats[stats["count"] >= min_count]
 return s.sort_values("mean", ascending= False).index.to_list()

def make_recommender(ranking):
 def recommend(u,seen,k):
  return[m for m in ranking if m not in seen][:k]
 return recommend

results = {}
for n in [1,5,10,20,50,100]:
 rec = make_recommender(make_ranking(n))
 results[n] = evaluate(rec,train,test)
 r = results[n]
 print("with n = ", n, r["recall@k"],r["ndcg@k"])

def popularity(u, seen, k):
 return [m for m in ranked_by_count if m not in seen][:k]

# --- Hệ thống 2: ngẫu nhiên (để biết "đáy" nằm ở đâu) ---
all_items = train.movieId.unique().tolist()

def rand(u, seen, k):
 return random.sample([m for m in all_items if m not in seen], k)

res_rand = evaluate(rand, train, test)
res_pop = evaluate(popularity, train, test)

#print("random    ", evaluate(rand, train, test))
#print("popularity", evaluate(popularity, train, test))

plot_metric_by_param(
 results,
 xlabel="số lượt chấm tối thiểu N",
 baseline={"popularity (lượt thích)": res_pop},
 title="Xếp theo rating trung bình",
)

