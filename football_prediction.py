"""
AIT 312 - Python for Artificial Intelligence | Group Assignment (Group: SPORT)
Katsina State Institute of Technology and Management
School of Technology, Department of Artificial Intelligence

Project : Predicting the Outcome of International Football Matches
          (Home Win / Draw / Away Win) using Pandas and a Neural Network
Author  : Hussaina Akilu (HND/AIT/25/0057)

Pipeline: raw data -> inspection -> cleaning -> transformation/feature
          engineering -> analysis -> visualisation -> neural-network
          prediction -> evaluation -> interpretation

Dataset : International football results, 1872-2026 (Mart Jurisoo),
          https://github.com/martj42/international_results  (results.csv)

Run     : python football_prediction.py
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import json
import os
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "results.csv"
FIG_DIR = "figures"
OUT_DIR = "outputs"
MODEL_START = "2000-01-01"   # modelling window starts here
TEST_START = "2022-01-01"    # chronological hold-out test set
RANDOM_STATE = 42
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

GREEN, GOLD, GREY, RED = "#0F5132", "#C9A227", "#8A938D", "#B23A48"
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
LABELS = ["Away Win", "Draw", "Home Win"]
report = {}                   # collects numbers used in the written report


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1-2. Read the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
print("Loaded:", df.shape)

# ---------------------------------------------------------------------------
# 3. Inspect and describe
# ---------------------------------------------------------------------------
print(df.head())
print(df.dtypes)
print(df.describe())
report["raw_records"] = int(df.shape[0])
report["raw_attributes"] = int(df.shape[1])
report["raw_dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
report["describe"] = df[["home_score", "away_score"]].describe().round(3).to_dict()
report["n_teams"] = int(pd.concat([df.home_team, df.away_team]).nunique())
report["n_tournaments"] = int(df.tournament.nunique())
report["n_countries"] = int(df.country.nunique())
report["date_range"] = [df.date.min(), df.date.max()]
report["neutral_counts"] = df.neutral.value_counts().to_dict()

# ---------------------------------------------------------------------------
# 4. Data-quality checks and cleaning
# ---------------------------------------------------------------------------
quality = {}
quality["missing_per_column"] = df.isna().sum().to_dict()
quality["exact_duplicates"] = int(df.duplicated().sum())
key_dups = df[df.duplicated(["date", "home_team", "away_team"], keep=False)]
quality["same_fixture_same_day"] = int(len(key_dups))
quality["same_fixture_rows"] = key_dups.to_dict("records")
quality["date_dtype_before"] = str(df["date"].dtype)
quality["self_matches"] = int((df.home_team == df.away_team).sum())
quality["negative_scores"] = int(((df.home_score < 0) | (df.away_score < 0)).sum())

# (a) text cleaning: strip whitespace in categorical columns
for col in ["home_team", "away_team", "tournament", "city", "country"]:
    df[col] = df[col].astype(str).str.strip()

# (b) incorrect data type: date is stored as text -> convert to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")
quality["unparseable_dates"] = int(df["date"].isna().sum())
df = df.dropna(subset=["date"])

# (c) remove exact duplicate rows (none expected, but enforced)
df = df.drop_duplicates()

# (d) matches without a recorded score (future fixtures) cannot be labelled
before = len(df)
df = df.dropna(subset=["home_score", "away_score"])
quality["rows_without_score_removed"] = int(before - len(df))
df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(int)

# (e) inconsistent / high-cardinality categorical values:
#     202 tournament names are grouped into 6 consistent categories
def tournament_category(name):
    n = name.lower()
    if n == "friendly":
        return "Friendly"
    if n == "fifa world cup":
        return "World Cup"
    if "qualification" in n:
        return "Qualifier"
    if "nations league" in n:
        return "Nations League"
    continental = ["uefa euro", "copa américa", "african cup of nations",
                   "afc asian cup", "gold cup", "ofc nations cup",
                   "confederations cup"]
    if any(n.startswith(c) for c in continental):
        return "Continental Final"
    return "Other Tournament"


df["tournament_cat"] = df["tournament"].apply(tournament_category)

# (f) extreme scores (e.g. 31-0) are genuine results between very unequal
#     teams, so they are kept; they are flagged for the analysis.
df["goal_diff"] = df.home_score - df.away_score
quality["extreme_margin_matches_(>=10)"] = int((df.goal_diff.abs() >= 10).sum())
df = df.sort_values("date").reset_index(drop=True)
quality["records_after_cleaning"] = int(len(df))
report["quality"] = quality
print("Data quality:", {k: v for k, v in quality.items() if k != "same_fixture_rows"})

# ---------------------------------------------------------------------------
# 5. Transformation and feature creation
#    Features are computed strictly from matches played BEFORE each game,
#    so no information from the match itself leaks into its prediction.
# ---------------------------------------------------------------------------
df["result"] = np.select([df.goal_diff > 0, df.goal_diff == 0], ["Home Win", "Draw"],
                         default="Away Win")
df["total_goals"] = df.home_score + df.away_score
df["year"] = df.date.dt.year
df["decade"] = (df.year // 10) * 10
df["neutral"] = df["neutral"].astype(int)
df["home_in_own_country"] = (df.home_team == df.country).astype(int)

K_BY_CAT = {"World Cup": 60, "Continental Final": 50, "Qualifier": 40,
            "Nations League": 40, "Other Tournament": 30, "Friendly": 20}
HOME_ADV = 100
elo = defaultdict(lambda: 1500.0)
history = defaultdict(lambda: deque(maxlen=5))   # (goals_for, goals_against, points)
played = defaultdict(int)

rows = []
for r in df.itertuples(index=False):
    h, a = r.home_team, r.away_team
    eh, ea = elo[h], elo[a]

    def form(team):
        hist = history[team]
        if not hist:
            return np.nan, np.nan, np.nan
        g = np.array(hist)
        return g[:, 0].mean(), g[:, 1].mean(), g[:, 2].mean()

    hgf, hga, hpts = form(h)
    agf, aga, apts = form(a)
    rows.append((eh, ea, hgf, hga, hpts, agf, aga, apts, played[h], played[a]))

    # --- update Elo (World Football Elo style) -----------------------------
    adv = 0 if r.neutral else HOME_ADV
    exp_h = 1 / (1 + 10 ** ((ea - (eh + adv)) / 400))
    s_h = 1.0 if r.goal_diff > 0 else 0.5 if r.goal_diff == 0 else 0.0
    gd = abs(r.goal_diff)
    mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
    change = K_BY_CAT[r.tournament_cat] * mult * (s_h - exp_h)
    elo[h] += change
    elo[a] -= change

    # --- update recent form and experience ---------------------------------
    pts_h = 3 if s_h == 1 else 1 if s_h == 0.5 else 0
    pts_a = 3 if s_h == 0 else 1 if s_h == 0.5 else 0
    history[h].append((r.home_score, r.away_score, pts_h))
    history[a].append((r.away_score, r.home_score, pts_a))
    played[h] += 1
    played[a] += 1

feat_cols = ["home_elo", "away_elo", "home_gf5", "home_ga5", "home_pts5",
             "away_gf5", "away_ga5", "away_pts5", "home_played", "away_played"]
df[feat_cols] = pd.DataFrame(rows, index=df.index, columns=feat_cols)
df["elo_diff"] = df.home_elo - df.away_elo + np.where(df.neutral == 1, 0, HOME_ADV)
df["form_diff"] = df.home_pts5 - df.away_pts5
df["exp_diff"] = np.log1p(df.home_played) - np.log1p(df.away_played)

# Missing values created by feature engineering (a team's first match has
# no previous form). Handled by imputing the neutral value for a new team.
missing_form = int(df[["home_gf5", "away_gf5"]].isna().any(axis=1).sum())
report["quality"]["missing_form_values_imputed"] = missing_form
impute = {"home_gf5": 1.0, "home_ga5": 1.0, "home_pts5": 1.0,
          "away_gf5": 1.0, "away_ga5": 1.0, "away_pts5": 1.0}
df = df.fillna(impute)
df["form_diff"] = df.home_pts5 - df.away_pts5

df.to_csv(os.path.join(OUT_DIR, "cleaned_featured_dataset.csv"), index=False)

# ---------------------------------------------------------------------------
# 6. Analysis with Pandas and statistics
# ---------------------------------------------------------------------------
res_share = (df.result.value_counts(normalize=True) * 100).round(2)
report["result_share_all"] = res_share.to_dict()

decade_goals = df.groupby("decade")["total_goals"].mean().round(3)
report["goals_by_decade"] = decade_goals.to_dict()

home_adv = (pd.crosstab(df.neutral.map({0: "Home venue", 1: "Neutral venue"}),
                        df.result, normalize="index") * 100).round(2)
report["home_advantage"] = home_adv.to_dict("index")

tour = (df.groupby("tournament_cat")
          .agg(matches=("result", "size"), avg_goals=("total_goals", "mean"),
               home_win_pct=("result", lambda s: (s == "Home Win").mean() * 100))
          .round(2).sort_values("matches", ascending=False))
report["tournament_summary"] = tour.to_dict("index")

modern = df[df.date >= MODEL_START].copy()
modern["elo_bin"] = pd.cut(modern.elo_diff, bins=[-800, -300, -200, -100, 0, 100,
                                                  200, 300, 400, 1200])
elo_bins = (pd.crosstab(modern.elo_bin, modern.result, normalize="index") * 100).round(2)
report["elo_bins"] = {str(k): v for k, v in elo_bins.to_dict("index").items()}

corr_cols = ["elo_diff", "form_diff", "exp_diff", "home_gf5", "home_ga5",
             "away_gf5", "away_ga5", "neutral", "home_in_own_country"]
modern["target_num"] = modern.result.map({"Away Win": -1, "Draw": 0, "Home Win": 1})
corr = modern[corr_cols + ["target_num"]].corr().round(3)
report["corr_with_target"] = corr["target_num"].drop("target_num").to_dict()

top_teams = (pd.Series(elo).sort_values(ascending=False).head(10).round(1))
report["top_elo_teams"] = top_teams.to_dict()

# ---------------------------------------------------------------------------
# 7. Visualisation
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
order = ["Home Win", "Draw", "Away Win"]
vals = [res_share[o] for o in order]
bars = ax.bar(order, vals, color=[GREEN, GREY, GOLD])
ax.bar_label(bars, fmt="%.1f%%")
ax.set_ylim(0, max(vals) * 1.15)
ax.set_ylabel("Share of matches (%)")
ax.set_title("Distribution of match outcomes (1872–2026)")
save(fig, "fig01_outcome_distribution.png")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(decade_goals.index, decade_goals.values, marker="o", color=GREEN, lw=2)
ax.set_xlabel("Decade"); ax.set_ylabel("Average goals per match")
ax.set_title("Average goals per international match by decade")
save(fig, "fig02_goals_by_decade.png")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
home_adv[order].plot(kind="bar", ax=ax, color=[GREEN, GREY, GOLD], rot=0)
ax.set_ylabel("Share of matches (%)"); ax.set_xlabel("")
ax.set_title("Home advantage: outcomes at home vs neutral venues")
ax.legend(title="")
save(fig, "fig03_home_advantage.png")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
clipped = df.goal_diff.clip(-8, 8)
ax.hist(clipped, bins=np.arange(-8.5, 9.5, 1), color=GREEN, edgecolor="white")
ax.set_xlabel("Goal difference (home − away), clipped at ±8")
ax.set_ylabel("Number of matches")
ax.set_title("Distribution of goal difference")
save(fig, "fig04_goal_difference.png")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
x = [str(i) for i in elo_bins.index]
for lab, col in zip(order, [GREEN, GREY, GOLD]):
    ax.plot(x, elo_bins[lab], marker="o", label=lab, color=col, lw=2)
ax.set_xlabel("Elo difference (home − away, incl. home advantage)")
ax.set_ylabel("Outcome share (%)")
ax.set_title("Outcome probability rises with Elo rating difference (2000–2026)")
ax.tick_params(axis="x", rotation=35)
ax.legend()
save(fig, "fig05_elo_vs_outcome.png")

fig, ax = plt.subplots(figsize=(6.8, 5.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax,
            cbar_kws={"shrink": .75}, annot_kws={"size": 7})
ax.set_title("Correlation matrix of engineered features and target")
save(fig, "fig06_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.barh(top_teams.index[::-1], top_teams.values[::-1], color=GREEN)
ax.set_xlim(top_teams.min() - 80, top_teams.max() + 30)
ax.set_xlabel("Elo rating (end of dataset)")
ax.set_title("Top 10 national teams by computed Elo rating")
save(fig, "fig07_top_elo_teams.png")

# ---------------------------------------------------------------------------
# 8. Input features and target
# ---------------------------------------------------------------------------
num_features = ["elo_diff", "home_elo", "away_elo", "form_diff",
                "home_gf5", "home_ga5", "home_pts5",
                "away_gf5", "away_ga5", "away_pts5",
                "exp_diff", "neutral", "home_in_own_country"]
data = df[df.date >= MODEL_START].copy()
X = pd.get_dummies(data[num_features + ["tournament_cat"]],
                   columns=["tournament_cat"], prefix="tour", dtype=int)
# label encoding of the target: 0 = Away Win, 1 = Draw, 2 = Home Win
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
y = data["result"].map(LABEL_TO_ID)
feature_names = list(X.columns)
report["features"] = feature_names
report["model_records"] = int(len(data))

# ---------------------------------------------------------------------------
# 9. Train / test split (chronological, to mimic real forecasting)
# ---------------------------------------------------------------------------
train_mask = data.date < TEST_START
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
report["split"] = {"train": int(len(X_train)), "test": int(len(X_test)),
                   "train_period": f"{MODEL_START[:4]}–2021",
                   "test_period": f"2022–{data.date.max():%Y}"}

scaler = StandardScaler()
scale_cols = [c for c in feature_names if not c.startswith("tour_")
              and c not in ("neutral", "home_in_own_country")]
X_train_s, X_test_s = X_train.copy().astype(float), X_test.copy().astype(float)
X_train_s[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

# ---------------------------------------------------------------------------
# 10. Neural network: architecture selection on a time-based validation set
# ---------------------------------------------------------------------------
val_mask = data.loc[train_mask, "date"] >= "2018-01-01"
Xtr_in, Xval = X_train_s[~val_mask.values], X_train_s[val_mask.values]
ytr_in, yval = y_train[~val_mask.values], y_train[val_mask.values]
candidates = {"MLP (32)": (32,), "MLP (64, 32)": (64, 32), "MLP (128, 64, 32)": (128, 64, 32)}
cand_scores = {}
for name, layers in candidates.items():
    m = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam",
                      alpha=1e-3, learning_rate_init=1e-3, batch_size=128,
                      max_iter=300, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=15, random_state=RANDOM_STATE)
    m.fit(Xtr_in, ytr_in)
    p = m.predict_proba(Xval)
    cand_scores[name] = {"val_accuracy": round(accuracy_score(yval, m.predict(Xval)) * 100, 2),
                         "val_log_loss": round(log_loss(yval, p, labels=m.classes_), 4),
                         "epochs": int(m.n_iter_)}
report["architecture_search"] = cand_scores
best_name = min(cand_scores, key=lambda k: cand_scores[k]["val_log_loss"])
report["best_architecture"] = best_name
print("Architecture search:", cand_scores, "-> best:", best_name)

# ---------------------------------------------------------------------------
# 11. Train final network and predict the test set
# ---------------------------------------------------------------------------
model = MLPClassifier(hidden_layer_sizes=candidates[best_name], activation="relu",
                      solver="adam", alpha=1e-3, learning_rate_init=1e-3,
                      batch_size=128, max_iter=300, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=15,
                      random_state=RANDOM_STATE)
model.fit(X_train_s, y_train)
y_pred = np.array(LABELS)[model.predict(X_test_s)]
y_prob = model.predict_proba(X_test_s)          # columns: Away, Draw, Home
y_test = y_test.map(dict(enumerate(LABELS)))
report["epochs_trained"] = int(model.n_iter_)
report["n_parameters"] = int(sum(w.size for w in model.coefs_) +
                             sum(b.size for b in model.intercepts_))

fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
ax1.plot(model.loss_curve_, color=GREEN, lw=2, label="Training loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
ax2 = ax1.twinx()
ax2.plot(model.validation_scores_, color=GOLD, lw=2, label="Validation accuracy")
ax2.set_ylabel("Validation accuracy"); ax2.grid(False)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
ax1.set_title("Neural network training history")
save(fig, "fig08_training_curve.png")

# ---------------------------------------------------------------------------
# 12. Evaluation
# ---------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
report["test_accuracy"] = round(acc * 100, 2)
report["test_macro_f1"] = round(f1_score(y_test, y_pred, average="macro"), 4)
report["test_weighted_f1"] = round(f1_score(y_test, y_pred, average="weighted"), 4)
report["test_log_loss"] = round(log_loss(y_test, y_prob, labels=LABELS), 4)
report["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

# baselines for context
majority = LABELS[y_train.value_counts().idxmax()]
report["baseline_majority_acc"] = round((y_test == majority).mean() * 100, 2)
elo_rule = np.where(data.loc[~train_mask, "elo_diff"] > 0, "Home Win", "Away Win")
report["baseline_elo_rule_acc"] = round((y_test.values == elo_rule).mean() * 100, 2)
report["baseline_uniform_log_loss"] = round(np.log(3), 4)

cm = confusion_matrix(y_test, y_pred, labels=order)
report["confusion_matrix"] = cm.tolist()
fig, ax = plt.subplots(figsize=(5.2, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=order,
            yticklabels=order, ax=ax, cbar=False)
ax.set_xlabel("Predicted outcome"); ax.set_ylabel("Actual outcome")
ax.set_title("Confusion matrix – test set (2022–2026)")
save(fig, "fig09_confusion_matrix.png")

# ---------------------------------------------------------------------------
# 13. Actual vs predicted
# ---------------------------------------------------------------------------
test = data.loc[~train_mask, ["date", "home_team", "away_team", "home_score",
                              "away_score", "tournament", "result"]].copy()
test["predicted"] = y_pred
cls = LABELS
for lab in order:
    test[f"p_{lab}"] = y_prob[:, cls.index(lab)]
test["correct"] = test.result == test.predicted
test.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

act = y_test.value_counts().reindex(order)
pred = pd.Series(y_pred).value_counts().reindex(order).fillna(0)
report["actual_vs_pred_counts"] = {"actual": act.to_dict(), "predicted": pred.astype(int).to_dict()}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
w = 0.38; xi = np.arange(3)
b1 = ax.bar(xi - w / 2, act.values, w, color=GREEN, label="Actual")
b2 = ax.bar(xi + w / 2, pred.values, w, color=GOLD, label="Predicted")
ax.bar_label(b1); ax.bar_label(b2)
ax.set_xticks(xi, order); ax.set_ylabel("Number of matches")
ax.set_title("Actual vs predicted outcomes – test set"); ax.legend()
save(fig, "fig10_actual_vs_predicted.png")

# calibration: predicted home-win probability vs observed home-win frequency
test["p_bin"] = pd.cut(test["p_Home Win"], bins=np.linspace(0, 1, 11))
calib = test.groupby("p_bin", observed=True).agg(
    mean_pred=("p_Home Win", "mean"),
    observed=("result", lambda s: (s == "Home Win").mean()),
    n=("result", "size"))
calib = calib[calib.n >= 20]
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="Perfect calibration")
ax.plot(calib.mean_pred, calib.observed, marker="o", color=GREEN, lw=2, label="Neural network")
ax.set_xlabel("Predicted probability of home win")
ax.set_ylabel("Observed frequency of home win")
ax.set_title("Probability calibration – test set"); ax.legend()
save(fig, "fig11_calibration.png")
report["calibration"] = calib.round(3).reset_index(drop=True).to_dict("records")

# accuracy by confidence
test["confidence"] = y_prob.max(axis=1)
conf = test.groupby(pd.cut(test.confidence, [0, .5, .6, .7, .8, 1.0]), observed=True).agg(
    matches=("correct", "size"), accuracy=("correct", "mean"))
conf["accuracy"] = (conf.accuracy * 100).round(2)
report["accuracy_by_confidence"] = {str(k): v for k, v in conf.to_dict("index").items()}

# showcase: FIFA World Cup 2026
wc = test[(test.tournament == "FIFA World Cup") & (test.date.dt.year == 2026)]
report["wc2026"] = {"matches": int(len(wc)),
                    "accuracy": round(wc.correct.mean() * 100, 2) if len(wc) else None}
sample = wc.tail(8) if len(wc) else test.tail(8)
report["sample_predictions"] = [
    {"date": f"{r.date:%d %b %Y}", "match": f"{r.home_team} vs {r.away_team}",
     "score": f"{r.home_score}–{r.away_score}", "actual": r.result,
     "predicted": r.predicted,
     "probs": f"{r['p_Home Win']:.2f} / {r['p_Draw']:.2f} / {r['p_Away Win']:.2f}"}
    for _, r in sample.iterrows()]

# permutation importance (drop in accuracy when each feature group is shuffled)
rng = np.random.default_rng(RANDOM_STATE)
groups = {"Elo ratings": ["elo_diff", "home_elo", "away_elo"],
          "Recent form": ["form_diff", "home_gf5", "home_ga5", "home_pts5",
                          "away_gf5", "away_ga5", "away_pts5"],
          "Experience": ["exp_diff"],
          "Venue": ["neutral", "home_in_own_country"],
          "Tournament type": [c for c in feature_names if c.startswith("tour_")]}
imp = {}
for g, cols in groups.items():
    drops = []
    for _ in range(5):
        Xp = X_test_s.copy()
        idx = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].values[idx]
        drops.append(acc - accuracy_score(y_test, np.array(LABELS)[model.predict(Xp)]))
    imp[g] = round(float(np.mean(drops)) * 100, 2)
imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
report["feature_group_importance"] = imp
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.barh(list(imp)[::-1], list(imp.values())[::-1], color=GREEN)
ax.set_xlabel("Drop in test accuracy when shuffled (percentage points)")
ax.set_title("Permutation importance of feature groups")
save(fig, "fig12_feature_importance.png")

with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"Test accuracy: {report['test_accuracy']}%  | macro-F1: {report['test_macro_f1']}"
      f"  | log-loss: {report['test_log_loss']}")
print("Done. Figures saved in", FIG_DIR, "and outputs in", OUT_DIR)
