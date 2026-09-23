"""Post hoc, not pre-specified: recording- and patient-level disease-group models.

Features are aggregates of the nested-CV outer-test event probabilities (each child's
probabilities come only from the outer fold in which the child was tested) plus age and sex,
evaluated with a new patient-grouped 5-fold CV repeated over five seeds. Also writes the
breakdown of recorded diagnoses within each disease group. Output: results_clean/posthoc_disease/."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import lightgbm as lgb
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results_clean', 'nested_cv') + '/'
OUT = os.path.join(ROOT, 'results_clean', 'posthoc_disease') + '/'
os.makedirs(OUT, exist_ok=True)
ev = pd.concat([pd.read_csv(f'{R}fold{k}/probabilities/meta_test.csv') for k in range(1, 6)],
               ignore_index=True)
fa = pd.read_csv(R + 'patient_fold_assignment.csv').set_index('_group_key').outer_fold
assert (ev._group_key.map(fa) == ev.outer_fold).all()
assert ev._group_key.nunique() == 736 == len(fa)
# ground-truth event annotations (event_type, model1_label, model2_label) are NOT used as features.
P = ['model1_label_p0', 'model1_label_p1', 'model1_label_p2', 'model2_label_p1',
     'model3_label_p0', 'model3_label_p1', 'model3_label_p2']
ev['pred1'] = ev[['model1_label_p0', 'model1_label_p1', 'model1_label_p2']].values.argmax(1)
ev['pred3'] = ev[['model3_label_p0', 'model3_label_p1', 'model3_label_p2']].values.argmax(1)
ev['f_crackle'] = (ev.pred1 == 1).astype(float)
ev['f_wheeze'] = (ev.pred1 == 2).astype(float)
ev['f_advent1'] = (ev.pred1 != 0).astype(float)
ev['f_abn2'] = (ev.model2_label_p1 > 0.5).astype(float)
for c in range(3):
    ev[f'f_pred3_{c}'] = (ev.pred3 == c).astype(float)
ev['dur'] = ev.event_duration_ms / 1000.0
FR = ['f_crackle', 'f_wheeze', 'f_advent1', 'f_abn2', 'f_pred3_0', 'f_pred3_1', 'f_pred3_2']


def q(p):
    f = lambda s: s.quantile(p); f.__name__ = f'q{int(p*100)}'; return f


def aggregate(g, key):
    a = g.groupby(key)[P].agg(['mean', 'max', 'min', 'std', q(.25), q(.5), q(.75)])
    a.columns = [f'{c}_{s}' for c, s in a.columns]
    a = a.join(g.groupby(key)[FR].mean())
    a['n_events'] = g.groupby(key).size()
    a['dur_mean'] = g.groupby(key).dur.mean()
    a['dur_total'] = g.groupby(key).dur.sum()
    a['dur_max'] = g.groupby(key).dur.max()
    a = a.fillna(0.0)  # std of single-event units
    return a


rec = aggregate(ev, 'filename')
rmeta = ev.groupby('filename').agg(pid=('_group_key', 'first'), y=('model3_label', 'first'),
                                   age=('age', 'first'), sex=('gender_code', 'first'),
                                   dx=('disease', 'first'))
rec = rec.join(rmeta)
pat = aggregate(ev, '_group_key')
pat['n_recordings'] = ev.groupby('_group_key').filename.nunique()
pmeta = ev.groupby('_group_key').agg(y=('model3_label', 'first'), age=('age', 'mean'),
                                     sex=('gender_code', 'first'), dx=('disease', 'first'))
pmeta['pid'] = pmeta.index
pat = pat.join(pmeta)
assert ev.groupby('_group_key').model3_label.nunique().max() == 1  # no conflicting dx
ACOU_R = [c for c in rec.columns if c not in ('pid', 'y', 'age', 'sex', 'dx')]
ACOU_P = [c for c in pat.columns if c not in ('pid', 'y', 'age', 'sex', 'dx')]
DEMO = ['age', 'sex']


def make(learner, ncls):
    if learner == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=0)


def oof(df, feats, learner, y, seeds=(42, 43, 44, 45, 46)):
    ncls = len(np.unique(y))
    reps = []
    for s in seeds:
        pr = np.zeros((len(df), ncls))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=s)
        for tr, te in cv.split(df, y, groups=df.pid.values):
            m = make(learner, ncls).fit(df[feats].values[tr], y[tr])
            pr[te] = m.predict_proba(df[feats].values[te])
        reps.append(pr)
    return reps


def metrics(y, pr):
    ncls = pr.shape[1]
    if ncls == 2:
        auc = roc_auc_score(y, pr[:, 1]); per = [auc]
    else:
        auc = roc_auc_score(y, pr, multi_class='ovr', average='macro')
        per = [roc_auc_score(y == c, pr[:, c]) for c in range(ncls)]
    yh = pr.argmax(1)
    return [auc, accuracy_score(y, yh), balanced_accuracy_score(y, yh)] + per


def boot(y, pr, groups, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_of = [np.flatnonzero(inv == i) for i in range(len(ug))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        ix = np.concatenate([idx_of[i] for i in pick])
        try:
            out.append(metrics(y[ix], pr[ix]))
        except ValueError:
            pass
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


# recorded diagnoses within each disease group
dx = pd.concat([pd.read_csv(os.path.join(ROOT, 'results_clean', f'split_{p}.csv'),
                            usecols=['disease', 'model3_label', '_group_key', 'filename'])
                for p in ('train', 'val', 'test')])
dx = (dx.groupby(['model3_label', 'disease'])
        .agg(patients=('_group_key', 'nunique'), recordings=('filename', 'nunique'),
             events=('filename', 'size'))
        .reset_index().sort_values(['model3_label', 'patients'], ascending=[True, False]))
dx.to_csv(OUT + 'diagnoses_by_group.csv', index=False)

rows = []
for level, df_all, acou in [('recording', rec, ACOU_R), ('patient', pat, ACOU_P)]:
    for target in ['3class', 'pneumonia_vs_rest', 'pneumonia_vs_control']:
        # pneumonia versus healthy control: children (recordings) of those two diagnoses only
        df = (df_all[(df_all.y == 0) | (df_all.dx == 'Control Group')]
              if target == 'pneumonia_vs_control' else df_all)
        y = df.y.values.astype(int)
        if target != '3class':
            y = (y == 0).astype(int)
        maj = np.bincount(y).max() / len(y)
        configs = [('demo', DEMO), ('acoustic', acou), ('acoustic+demo', acou + DEMO)]
        # no-learning reference: mean of base model3 probabilities (as in the primary aggregation)
        m3 = df[['model3_label_p0_mean', 'model3_label_p1_mean', 'model3_label_p2_mean']].values
        m3 = m3 / m3.sum(1, keepdims=True)
        if target != '3class':
            m3 = np.c_[1 - m3[:, 0], m3[:, 0]]
        todo = [('ref: mean base-model3 prob (no fit)', None, None, [m3])]
        for cname, feats in configs:
            for learner in ['logreg', 'lgbm']:
                todo.append((cname, learner, feats, None))
        for cname, learner, feats, reps in todo:
            if reps is None:
                reps = oof(df, feats, learner, y)
            per_rep = np.array([metrics(y, r) for r in reps])
            avg = np.mean(reps, 0)
            pt = metrics(y, avg)
            ci = boot(y, avg, df.pid.values)
            names = ['auc', 'acc', 'bacc'] + (['auc_pneu', 'auc_bronch', 'auc_normoth']
                                              if target == '3class' else [])
            row = dict(level=level, target=target, features=cname, learner=learner or '-',
                       n=len(y), majority=round(maj, 3))
            for i, nm in enumerate(names):
                row[nm] = round(pt[i], 3)
                row[nm + '_ci'] = f'{ci[0, i]:.3f}-{ci[1, i]:.3f}'
            row['auc_rep_sd'] = round(per_rep[:, 0].std(), 3)
            row['acc_rep_mean'] = round(per_rep[:, 1].mean(), 3)
            rows.append(row)
            print(row, flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT + 'results.csv', index=False)
print(res.to_string())

# age by disease group (healthy controls shown separately), patient level
grp = pat.y.map({0: 'Pneumonia', 1: 'Bronchial disease', 2: 'Normal/other'})
grp = grp.where(pat.dx != 'Control Group', 'Healthy control (within normal/other)')
age = (pat.groupby(grp).age.describe(percentiles=[.25, .5, .75])[['count', '25%', '50%', '75%']]
       .round(2).rename(columns={'count': 'children', '25%': 'q1', '50%': 'median', '75%': 'q3'}))
age.to_csv(OUT + 'age_by_group.csv')
print(age)

# severe versus non-severe pneumonia (descriptive): acoustic summaries of each child
pn = pat[pat.y == 0].copy()
pn['severe'] = (pn.dx == 'Pneumonia (severe)').astype(int)
sev_rows = []
for feat, label in [('f_abn2', 'fraction of events predicted adventitious (screening)'),
                    ('f_crackle', 'fraction of events predicted crackles'),
                    ('model3_label_p0_mean', 'mean predicted probability of pneumonia')]:
    yy, xx = pn.severe.values, pn[feat].values
    auc = roc_auc_score(yy, xx)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        ix = rng.integers(0, len(yy), len(yy))
        if yy[ix].min() != yy[ix].max():
            bs.append(roc_auc_score(yy[ix], xx[ix]))
    sev_rows.append(dict(feature=label, n_severe=int(yy.sum()), n_non_severe=int((1 - yy).sum()),
                         median_severe=round(float(np.median(xx[yy == 1])), 3),
                         median_non_severe=round(float(np.median(xx[yy == 0])), 3),
                         auc_severe_vs_non=round(auc, 3),
                         auc_ci=f'{np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}'))
pd.DataFrame(sev_rows).to_csv(OUT + 'severity.csv', index=False)
print(pd.DataFrame(sev_rows).to_string())


# paired patient-bootstrap contrasts: accuracy minus majority rate, AUC gain over demographics
def auc_(y, p):
    return roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class='ovr', average='macro')


out=[]
for level, df, acou in [('recording', rec, ACOU_R), ('patient', pat, ACOU_P)]:
    for target in ['3class','pneumonia_vs_rest']:
        y=df.y.values.astype(int)
        if target!='3class': y=(y==0).astype(int)
        mc=np.bincount(y).argmax()
        A={}
        for nm,f,l in [('demo_lgbm',DEMO,'lgbm'),('demo_lr',DEMO,'logreg'),('ad_lr',acou+DEMO,'logreg'),('ad_lgbm',acou+DEMO,'lgbm')]:
            A[nm]=np.mean(oof(df,f,l,y),0)
        rng=np.random.default_rng(1); ug,inv=np.unique(df.pid.values,return_inverse=True)
        idx=[np.flatnonzero(inv==i) for i in range(len(ug))]
        D={k:[] for k in ['acc-maj ad_lr','acc-maj ad_lgbm','dAUC ad_lr-demo_lgbm','dAUC ad_lgbm-demo_lgbm']}
        for _ in range(1000):
            ix=np.concatenate([idx[i] for i in rng.integers(0,len(ug),len(ug))]); yy=y[ix]
            maj=(yy==mc).mean()
            D['acc-maj ad_lr'].append((A['ad_lr'][ix].argmax(1)==yy).mean()-maj)
            D['acc-maj ad_lgbm'].append((A['ad_lgbm'][ix].argmax(1)==yy).mean()-maj)
            dl=auc_(yy,A['demo_lgbm'][ix])
            D['dAUC ad_lr-demo_lgbm'].append(auc_(yy,A['ad_lr'][ix])-dl)
            D['dAUC ad_lgbm-demo_lgbm'].append(auc_(yy,A['ad_lgbm'][ix])-dl)
        for k,v in D.items():
            v=np.array(v); r=dict(level=level,target=target,contrast=k,mean=round(v.mean(),3),lo=round(np.percentile(v,2.5),3),hi=round(np.percentile(v,97.5),3),p_le0=round((v<=0).mean(),3)); out.append(r); print(r,flush=True)
pd.DataFrame(out).to_csv(OUT+'paired_contrasts.csv',index=False)
