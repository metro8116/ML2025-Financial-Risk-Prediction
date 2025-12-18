import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 1. 读取数据
df_train = pd.read_csv('train.csv')
df_testA  = pd.read_csv('testA.csv')

# 2. 分离 ID、标签、特征
y        = df_train['isDefault']
X        = df_train.drop(['id', 'isDefault'], axis=1)
test_ids = df_testA['id']
X_test   = df_testA.drop(['id'], axis=1)

# 3. 数值/类别列识别
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 4. 缺失值插补函数
def impute(df, num_cols, cat_cols, num_imp=None, cat_imp=None):
    if num_imp is None:
        num_imp = SimpleImputer(strategy='median')
        df[num_cols] = num_imp.fit_transform(df[num_cols])
    else:
        df[num_cols] = num_imp.transform(df[num_cols])
    if cat_imp is None:
        cat_imp = SimpleImputer(strategy='constant', fill_value='missing')
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
    else:
        df[cat_cols] = cat_imp.transform(df[cat_cols])
    return df, num_imp, cat_imp

X, num_imp, cat_imp = impute(X, num_cols, cat_cols)
X_test, _, _        = impute(X_test, num_cols, cat_cols, num_imp, cat_imp)

# 5. 目标编码
te = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)
te.fit(X, y)
X      = te.transform(X)
X_test = te.transform(X_test)

# 6. 标准化数值特征
def scale(df, cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler

X, scaler     = scale(X, num_cols)
X_test, _     = scale(X_test, num_cols, scaler)

# 7. 预先构造测试集 DMatrix
dtest = xgb.DMatrix(X_test)

# 8. 手写 5 折 CV，并启用 histogram + 多线程
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores    = []
preds_test    = np.zeros(len(X_test))

# 用于累加各折的重要性
feature_names = X.columns.tolist()
imp_accum     = np.zeros(len(feature_names), dtype=float)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': (y_tr == 0).sum() / (y_tr == 1).sum(),
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'nthread': -1,
        'seed': 42
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # 验证 AUC
    pred_val = bst.predict(dval)
    auc = roc_auc_score(y_val, pred_val)
    print(f"Fold {fold} AUC: {auc:.4f}")
    auc_scores.append(auc)

    # 累加测试集预测
    preds_test += bst.predict(dtest, iteration_range=(0, bst.best_iteration))

    # 累加本折特征重要性（使用 'weight'，也可选 'gain' 或 'cover'）
    fold_imp_dict = bst.get_score(importance_type='weight')
    fold_imp = np.array([ fold_imp_dict.get(f, 0.0) for f in feature_names ], dtype=float)
    imp_accum += fold_imp

# 输出平均 CV AUC
print(f"Mean CV AUC: {np.mean(auc_scores):.4f}")

# 输出结果文件
preds_test /= skf.n_splits
df_submit = pd.DataFrame({'id': test_ids, 'isDefault': preds_test})
df_submit.to_csv('Result.csv', index=False)
print("Result.csv 已生成。")

# 9. 计算并保存平均特征重要性图（只保存Top 20，不显示）
imp_mean     = imp_accum / skf.n_splits
idx_sorted   = np.argsort(imp_mean)[::-1][:20]  # 取前20
sorted_feats = [feature_names[i] for i in idx_sorted]
sorted_imp   = imp_mean[idx_sorted]

plt.figure(figsize=(8, 10))
plt.barh(range(len(sorted_feats)), sorted_imp, color='skyblue')
plt.yticks(range(len(sorted_feats)), sorted_feats)
plt.xlabel('Average Feature Importance (weight)')
plt.title('Top 20 Features by Importance (XGBoost)')
plt.gca().invert_yaxis()  # 最重要的在上
plt.tight_layout()
plt.savefig('feature_importance.png')  # 只保存，不显示
