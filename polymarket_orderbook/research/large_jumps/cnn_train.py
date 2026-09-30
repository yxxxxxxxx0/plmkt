# Learning curves: trees (tabular) vs CNN (book sequence) vs hybrid CNN (sequence + tabular), by share of training games.
import os, time, json
import numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn, warnings
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore'); torch.set_num_threads(16); torch.manual_seed(0)
S = SP = 'research/large_jumps/cache'   # generated, git-ignored
IN = 'results/large_jumps/inputs'          # small committed inputs
A = pd.read_pickle(os.path.join(S, 'cnn_meta.pkl'))
X = np.load(os.path.join(S, 'cnn_X.npy'), mmap_mode='r')
keep = A.seq_ok.to_numpy(); A = A[keep].reset_index(drop=False)
Xs = np.asarray(X[A['index'].to_numpy()])                               # float16, ~2 GB
TAB = [c for c in A.columns if c.startswith(('f_', 'g_'))] + ['yes_bats']
Tb = A[TAB].to_numpy(np.float32); Tb_nan = np.isnan(Tb).astype(np.float32)
test = (A.sess >= 'books_2026-09-21').to_numpy()
rng = np.random.default_rng(0)
train_games = np.array(sorted(A[~test].slug.unique())); rng.shuffle(train_games)
val_games = set(train_games[:len(train_games) // 10]); pool = train_games[len(train_games) // 10:]
isval = A.slug.isin(val_games).to_numpy() & ~test

# channel normalisation from the training pool
tr_all = ~test & ~isval
mu = Xs[tr_all][::20].astype(np.float32).reshape(-1, Xs.shape[2]).mean(0)
sd = Xs[tr_all][::20].astype(np.float32).reshape(-1, Xs.shape[2]).std(0) + 1e-3
tmu = np.nanmean(Tb[tr_all], 0); tsd = np.nanstd(Tb[tr_all], 0) + 1e-6
TbN = np.nan_to_num((Tb - tmu) / tsd); TbF = np.concatenate([TbN, Tb_nan], 1).astype(np.float32)


class Net(nn.Module):
    def __init__(self, tab):
        super().__init__()
        c = Xs.shape[2]
        self.conv = nn.Sequential(nn.Conv1d(c, 64, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.ReLU(),
                                  nn.Conv1d(64, 64, 5, padding=8, dilation=4), nn.ReLU())
        self.tab = tab
        self.head = nn.Sequential(nn.Linear(128 + (TbF.shape[1] if tab else 0), 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, t):
        h = self.conv(x.transpose(1, 2)); z = torch.cat([h.mean(2), h[:, :, -10:].mean(2)], 1)
        return self.head(torch.cat([z, t], 1) if self.tab else z).squeeze(1)


def batches(idx, bs, shuffle):
    idx = rng.permutation(idx) if shuffle else idx
    for k in range(0, len(idx), bs):
        j = np.sort(idx[k:k + bs])
        yield j, torch.from_numpy((Xs[j].astype(np.float32) - mu) / sd), torch.from_numpy(TbF[j])


def predict(net, idx):
    net.eval(); out = []
    with torch.no_grad():
        for j, x, t in batches(idx, 2048, False): out.append(torch.sigmoid(net(x, t)).numpy())
    return np.concatenate(out)


def fit_cnn(tr, y, tab, epochs=6):
    net = Net(tab); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    pos = y[tr].mean(); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(50.0, np.sqrt((1 - pos) / pos))))
    va = np.flatnonzero(isval); best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        for j, x, t in batches(np.flatnonzero(tr), 512, True):
            opt.zero_grad(); loss = lossf(net(x, t), torch.from_numpy(y[j].astype(np.float32))); loss.backward(); opt.step()
        v = average_precision_score(y[va], predict(net, va))
        if v > best: best, best_state, bad = v, {k: w.clone() for k, w in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2: break
    net.load_state_dict(best_state); return net


def fit_tree(tr, y):
    Xt = pd.DataFrame(Tb, columns=TAB).assign(mtype=A.mtype.astype('category'))
    va = isval
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(Xt[tr], y[tr], eval_set=[(Xt[va], y[va])], eval_metric='average_precision', callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(Xt[test])[:, 1]


res = []
te_idx = np.flatnonzero(test)
for target in ('y_real', 'y_large'):
    y = A[target].to_numpy().astype(bool)
    for frac in (0.1, 0.25, 0.5, 1.0):
        g = set(pool[:max(3, int(round(frac * len(pool))))])
        tr = A.slug.isin(g).to_numpy() & ~test & ~isval
        for model in ('trees', 'cnn_book', 'cnn_book+tab'):
            t0 = time.time()
            if model == 'trees': p = fit_tree(tr, y)
            else: p = predict(fit_cnn(tr, y, tab=(model == 'cnn_book+tab')), te_idx)
            yt = y[test]; r = dict(target=target, frac=frac, games=len(g), n_train=int(tr.sum()), pos_train=int(y[tr].sum()),
                                   model=model, auc=roc_auc_score(yt, p), ap=average_precision_score(yt, p),
                                   top1=yt[p >= np.percentile(p, 99)].mean(), base=yt.mean(), secs=round(time.time() - t0))
            res.append(r); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            pd.DataFrame(res).to_csv("results/large_jumps/learning_curve/learning_curve.csv", index=False)
