import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import japanize_matplotlib

class ModelTrainer:
    """モデル訓練を担当するクラス"""
    def __init__(self):
        self.models = {}

    def train_binary_lgbm(self, X, y, df, categorical_indices):
        """LightGBMを使用した二値分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )
        
        # ======== LightGBMモデル構築 ========
        # パラメータ設定
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=7,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction = 0.8,
            bagging_freq = 5,
            verbose=-1,
            random_seed=777
        )

        # モデル訓練
        model.fit(X_train, y_train,
            categorical_feature=categorical_indices,
            eval_set=[(X_test, y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True),
                       lgb.log_evaluation(100)]
        )

        # ======== 評価 ========
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # クラス1の確率を取得
        y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"\nモデル評価 (LightGBM):")
        print(f"- 精度: {accuracy:.4f}")
        print(f"- AUC: {auc:.4f}")
        print(f"- 適合率: {precision:.4f}")
        print(f"- 再現率: {recall:.4f}")

        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model, figsize=(20, 12),max_num_features=20)
        #plt.show()
        
        # ======== 予測分布の可視化 ========
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(confusion_matrix(y_test, y_pred))

        return model

    def train_multiclass_lgbm(self, X, y, df):
        """LightGBMを使用した多クラス分類モデルの訓練（内部メソッド）"""
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y-1, df.index, test_size=0.2, random_state=42
        )

        # ======== モデル構築（LightGBM） ========
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=6,
            boosting_type='gbdt',#'gbdt
            learning_rate=0.05,#0.05
            n_estimators=1000,#100
            max_depth=8,
            num_leaves=31,#31
            min_data_in_leaf=8,
            feature_fraction=0.8,
            class_weight='balanced',
            verbose=-1
        )
                
        verbose_eval=50
        # モデル学習
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True),
                            lgb.log_evaluation(verbose_eval)])

        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model,figsize=(6,30))
        #plt.show()
        
        # ======== 評価 ========
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(classification_report(y_test, y_pred, digits=3))
        print("クラスごとの精度:")
        print(confusion_matrix(y_test, y_pred))
        
        return model

    def train_multiclass_lgbm_exclusion_one(self, X, y, df,categorical_indices):
        """1号艇が1着でないデータのみを使って多クラス分類モデルを訓練"""
        # 1号艇を除外する場合
        mask = (y != 1)  # 2-6号艇のみ
        X_filtered = X[mask]
        y_filtered = y[mask]
        df_filtered = df.iloc[mask]
        
        # データ分割（元のコードと同様）
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_filtered, y_filtered, df_filtered.index, test_size=0.2, random_state=42
        )
        # ======== モデル構築（LightGBM） ========
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=5,
            boosting_type='gbdt',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=-1,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction = 0.75,
            bagging_freq = 5,
            verbose=-1,
            class_weight='balanced'
        )
        
        # 学習と予測
        model.fit(X_train, y_train-2, eval_set=[(X_test, y_test-2)],
                    categorical_feature=categorical_indices,
                    callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True),
                            lgb.log_evaluation(100)]) 
               
        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model,figsize=(20, 12) ,max_num_features=20)
        #plt.show()               

        # ======== 評価 ========
        y_pred = model.predict(X_test) + 2
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        
        print("\n=== 予測結果の詳細分析 ===")
        print("混同行列:")
        print(confusion_matrix(y_test, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test, y_pred))
        
        return model

    def train_multiclass_lgbm_target_1st(self, X, y, df, target_1st_num=1):
        """
        1着が指定した艇のデータのみを使って多クラス分類モデルを訓練
        """
        # 1着情報を取得（dfから1着の艇番号が入っている列を指定）
        first_place = df['1着艇'].values

        # 1着が指定艇番号のもののみマスク
        mask = (first_place == target_1st_num)
        X_filtered = X[mask]
        y_filtered = y[mask]  # このyは2着の艇番号
        df_filtered = df.iloc[mask]

        # 2着候補の艇番号リスト（1着艇は除く）
        candidate_boats = sorted([x for x in set(y_filtered) if x != target_1st_num])
        num_classes = len(candidate_boats)

        # 艇番号を 0,1,2,... にマッピング
        label_mapping = {boat: idx for idx, boat in enumerate(candidate_boats)}
        y_mapped = np.array([label_mapping[boat] for boat in y_filtered])

        # 以降は元の処理と同じ（データ分割・モデル訓練・評価）
        # ================================================
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_filtered, y_mapped, df_filtered.index, test_size=0.2, random_state=42
        )

        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            boosting_type='gbdt',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=-1,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction = 0.75,
            bagging_freq = 5,
            verbose=-1,
            class_weight='balanced'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(-1)
            ]
        )

        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model,figsize=(20, 12),max_num_features=20)
        #plt.show()
        
        # 評価（元の艇番号に戻して表示）
        y_pred_mapped = model.predict(X_test)
        inverse_mapping = {idx: boat for idx, boat in enumerate(candidate_boats)}
        y_pred = np.array([inverse_mapping[idx] for idx in y_pred_mapped])
        y_test_original = np.array([inverse_mapping[idx] for idx in y_test])

        print(f"1着が {target_1st_num} 号艇のレースのみ使用")
        print(f"2着候補: {candidate_boats}")
        print(f"データ数: {len(X_filtered)}")
        print(f"分類精度: {accuracy_score(y_test_original, y_pred):.4f}")
        print("混同行列:")
        print(confusion_matrix(y_test_original, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test_original, y_pred))
        return model

    def train_multiclass_lgbm_target_1st_2nd(self, X, y, df, target_1st_num=1, target_2nd_num=2):
        """
        1着と2着が指定した艇のデータのみを使って多クラス分類モデルを訓練
        （3着を予測するモデル）
        """
        # 1着と2着情報を取得
        first_place = df['1着艇'].values
        second_place = df['2着艇'].values

        # 1着がtarget_1st_numかつ2着がtarget_2nd_numのもののみマスク
        mask = (first_place == target_1st_num) & (second_place == target_2nd_num)
        X_filtered = X[mask]
        y_filtered = y[mask]  # このyは3着の艇番号
        df_filtered = df.iloc[mask]

        # 3着候補の艇番号リスト（1着艇と2着艇は除く）
        candidate_boats = sorted([x for x in set(y_filtered) if x not in [target_1st_num, target_2nd_num]])
        num_classes = len(candidate_boats)

        # 艇番号を 0,1,2,... にマッピング
        label_mapping = {boat: idx for idx, boat in enumerate(candidate_boats)}
        y_mapped = np.array([label_mapping[boat] for boat in y_filtered])

        # 以降は元の処理と同じ（データ分割・モデル訓練・評価）
        # ================================================
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_filtered, y_mapped, df_filtered.index, test_size=0.2, random_state=42
        )

        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            boosting_type='gbdt',
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=-1,
            num_leaves=63,
            min_data_in_leaf=20,
            bagging_fraction=0.75,
            bagging_freq=5,
            verbose=-1,
            class_weight='balanced'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(-1)
            ]
        )
        
        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model,figsize=(20, 12),max_num_features=20)
        #plt.show()
        
        # 評価（元の艇番号に戻して表示）
        y_pred_mapped = model.predict(X_test)
        candidate_boats = [boat for boat in label_mapping.keys()]
        inverse_mapping = {idx: boat for boat, idx in label_mapping.items()}
        
        y_pred = np.array([inverse_mapping[idx] for idx in y_pred_mapped])
        y_test_original = np.array([inverse_mapping[idx] for idx in y_test])

        print(f"1着が {target_1st_num} 号艇、2着が {target_2nd_num} 号艇のレースのみ使用")
        print(f"3着候補: {sorted(candidate_boats)}")
        print(f"データ数: {len(X_filtered)}")
        print(f"分類精度: {accuracy_score(y_test_original, y_pred):.4f}")
        print("混同行列:")
        print(confusion_matrix(y_test_original, y_pred))
        print("\nクラスごとの精度:")
        print(classification_report(y_test_original, y_pred))
        
        return model
    