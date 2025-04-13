import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from collections import Counter
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from keras import models, layers, regularizers, callbacks
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
import japanize_matplotlib

from analyzer import BoatraceAnalyzer

class BoatraceML(BoatraceAnalyzer):
    def __init__(self, folder):
        super().__init__(folder)
        self.folder = folder
        self.model = None
      
    def compiling_race_data(self):
        merged_csv_folder = os.path.join(self.folder, "merged_csv")
        all_files = [os.path.join(merged_csv_folder, f) for f in os.listdir(merged_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            # ファイル名から日付を抽出
            filename = os.path.basename(filepath)
            file_date = "20" + filename[:2] + "-" + filename[2:4] + "-" + filename[4:6]

            # CSV読み込み
            df = pd.read_csv(filepath,encoding="shift-jis")

            # 出力用リスト
            records = []

            # レースごとに処理
            for (place, race_no), group in df.groupby(['レース場', 'レース番号']):
                if len(group) != 6:
                    continue  # 6艇そろってないレースは除外

                # 1-3着をとった艇番
                winner_row = group[group['着順'] == 1]
                second_row = group[group['着順'] == 2]
                third_row = group[group['着順'] == 3]

                winner_teiban = int(winner_row['艇番'].values[0])
                second_teiban = int(second_row['艇番'].values[0])
                third_teiban = int(third_row['艇番'].values[0])

                # 舟券学習用データ構造
                record = {
                    '日付': file_date,
                    'レース場': place,
                    'レース番号': race_no,
                    '1着艇': winner_teiban,
                    '2着艇': second_teiban,
                    '3着艇': third_teiban
                }

                # 各艇のデータを展開
                for i in range(1, 7):
                    boat = group[group['艇番'] == i]
                    if boat.empty:
                        continue

                    b = boat.iloc[0]
                    prefix = f"{i}号艇"
                    record.update({
                        f'{prefix}_登録番号': b['登録番号'],
                        f'{prefix}_年齢': b['年齢'],
                        f'{prefix}_支部': b['支部'],
                        f'{prefix}_体重': b['体重'],
                        f'{prefix}_級別': b['級別'],
                        f'{prefix}_全国勝率': b['全国勝率'],
                        f'{prefix}_全国2連対率': b['全国2連対率'],
                        f'{prefix}_当地勝率': b['当地勝率'],
                        f'{prefix}_当地2連対率': b['当地2連対率'],
                        f'{prefix}_モーター番号': b['モーター番号'],
                        f'{prefix}_モーター2連対率': b['モーター2連対率'],
                        f'{prefix}_ボート番号': b['ボート番号'],
                        f'{prefix}_ボート2連対率': b['ボート2連対率'],
                        f'{prefix}_展示タイム': b['展示タイム'],
                        f'{prefix}_平均ST': b['平均ST'],
                        f'{prefix}_全体平均ST': b['全体平均ST'],
                        f'{prefix}_1着率': b['1着率'],
                        f'{prefix}_2-3着率': b['2-3着率'],
                        f'{prefix}_逃し率': b['逃し率'],
                    })

                records.append(record)
            
            all_dataframes.append(pd.DataFrame(records))

        return pd.concat(all_dataframes, ignore_index=True)

    def Compiling_odds_data(self):
        o_csv_folder = os.path.join(self.folder, "o_csv")
        all_files = [os.path.join(o_csv_folder, f) for f in os.listdir(o_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            try:
                df = pd.read_csv(filepath, encoding='shift-jis')
            except Exception as e:
                print(f"読み込みエラー: {filepath} → {e}")
                continue

            all_dataframes.append(df)

        # すべてのDataFrameをまとめて返す
        if all_dataframes:
            return pd.concat(all_dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

    def preprocess_multiclass(self,df):
        """1着が何号艇になるかどうかの前処理"""
        # ======== ラベル作成 ========
        # 目的変数（1〜6の整数）→ 0〜5 にして one-hot化
        y = df['1着艇'].values - 1
        
        # ======== 特徴量前処理 ========
        # 不要列の削除
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])

        # カテゴリ変数のダミー変換
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 欠損値処理・型変換
        df = df.fillna(0).astype('float32')
        X = df

        # ======== 正規化 ========
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        return X,y,mean,std

    def preprocess_binaly(self,df):
        # ======== ラベル作成：1号艇が勝ったか？ ========
        df['target'] = df['1着艇'].apply(lambda x: 1 if x == 1 else 0)

        # ======== 特徴量前処理 ========
        df = df.drop(columns=['日付', 'レース番号', '1着艇', '2着艇', '3着艇'])

        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        df = df.fillna(0).astype('float32')

        X = df.drop(columns=['target'])
        y = df['target'].values

        # ======== 正規化 ========
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / (std + 1e-7)
        
        return X,y,mean,std
          
    def train_multiclass_Keras(self, X, y, df):    
        # ======== データ分割 ========
        # 元のインデックスを保持
        y_onehot = to_categorical(y, num_classes=6)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y_onehot, df.index, test_size=0.3, random_state=42
        )

        # ======== モデル構築 ========
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(6, activation='softmax')  # 6クラス分類
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )

        # ======== コールバック ========
        callbacks_list = [callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                     callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
        ]

        # ======== 学習 ========
        history = model.fit(X_train, y_train,
                            epochs=200,
                            batch_size=128,
                            validation_split=0.2,
                            callbacks=callbacks_list,
                            verbose=1)

        # ======== 評価 ========
        results = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nKerasモデルの多クラス分類精度:")
        print(f"- 精度: {results[1]:.4f}")
        print(f"- AUC: {results[2]:.4f}")
        print(f"- 適合率: {results[3]:.4f}")
        print(f"- 再現率: {results[4]:.4f}")

        # ======== 可視化 ========
        import matplotlib.pyplot as plt
        import japanize_matplotlib
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.show()
        
        # ======== 予測結果のDataFrame作成 ========
        y_pred_prob = model.predict(X_test)  # 各艇番の予測確率 (shape: [n_samples, 6])
        
        # 元のDataFrameから対応する行を取得
        original_data = df.loc[idx_test]
        
        # 結果DataFrame作成
        result_df = pd.DataFrame({
            'original_index': idx_test,  # 元のDataFrameのインデックス
            '日付': original_data['日付'].values,
            'レース場': original_data['レース場'].values,
            'レース番号': original_data['レース番号'].values,
            '正解1着艇': original_data['1着艇'].values,  # 元のdfから取得
        })
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            result_df[f'{boat_num}号艇勝利確率'] = y_pred_prob[:, boat_num-1]
        
        # 予測情報追加
        result_df['予測1着艇'] = np.argmax(y_pred_prob, axis=1) + 1
        result_df['予測確信度'] = np.max(y_pred_prob, axis=1)
        result_df['正解'] = (result_df['予測1着艇'] == result_df['正解1着艇']).astype(int)
        
        return model, X_test, y_test, result_df

    def train_multiclass_LGBM(self, X, y, df):
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.3, random_state=42
        )

        # ======== モデル構築（LightGBM） ========
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=6,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_data_in_leaf=8,
            feature_fraction=0.8,
            class_weight='balanced',
            verbose=-1
        )
                
        verbose_eval=10
        # モデル学習
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True),
                            lgb.log_evaluation(verbose_eval)])

        # ======== 評価 ========
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"LightGBMモデルの多クラス分類精度: {acc:.4f}")
        
        from sklearn.metrics import classification_report, confusion_matrix

        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred))
        

        # ======== 予測結果のDataFrame作成 ========
        y_pred_prob = model.predict_proba(X_test)

        # 元のDataFrameから対応する行を取得
        original_data = df.loc[idx_test]

        # 結果DataFrame作成
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': original_data['日付'].values,
            'レース場': original_data['レース場'].values,
            'レース番号': original_data['レース番号'].values,
            '正解1着艇': original_data['1着艇'].values,
        })

        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            result_df[f'{boat_num}号艇勝利確率'] = y_pred_prob[:, boat_num-1]

        # 予測情報追加
        result_df['予測1着艇'] = np.argmax(y_pred_prob, axis=1) + 1
        result_df['予測確信度'] = np.max(y_pred_prob, axis=1)
        result_df['正解'] = (result_df['予測1着艇'] == result_df['正解1着艇']).astype(int)
        
        # 各艇番ごとの予測評価
        for boat_num in range(1, 7):
            predicted_as_boat = (result_df['予測1着艇'] == boat_num)
            
            if predicted_as_boat.sum() > 0:
                accuracy = result_df.loc[predicted_as_boat, '正解'].mean()
                
                # 実際の正解クラスの分布
                actual_classes = result_df.loc[predicted_as_boat, '正解1着艇']
                class_distribution = actual_classes.value_counts().sort_index()
                class_distribution_pct = actual_classes.value_counts(normalize=True).sort_index()
                
                print(f"\n{boat_num}号艇が1着と予測したデータの評価")
                print(f"- 予測件数: {predicted_as_boat.sum()} レース/{len(y_test)}レース")
                print(f"- 正答率: {accuracy:.2%}")
                
                print("\n実際の正解クラス分布:")
                for cls, count in class_distribution.items():
                    pct = class_distribution_pct[cls]
                    print(f"  {cls}号艇: {count}回 ({pct:.2%})")
            else:
                print(f"\n{boat_num}号艇が1着と予測したデータはありません")
        
        lgb.plot_importance(model)
        plt.show()

        return model, X_test, y_test, result_df
        
    def train_binaly_Keras(self, X,y,df):
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )

        # ======== モデル構築 ========
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X.shape[1],),
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        # ======== コールバック ========
        callbacks = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
        ]

        # ======== 学習 ========
        history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=64,
                            validation_split=0.2,
                            callbacks=callbacks,
                            verbose=1)

        # ======== 評価 ========
        results = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n1号艇モデル評価:")
        print(f"- 精度: {results[1]:.4f}")
        print(f"- AUC: {results[2]:.4f}")
        print(f"- 適合率: {results[3]:.4f}")
        print(f"- 再現率: {results[4]:.4f}")

        # ======== 可視化 ========
        import matplotlib.pyplot as plt
        import japanize_matplotlib
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.show()
        
        # ======== 確率出力 ========
        y_pred_prob = model.predict(X_test).flatten()  # 各サンプルの1号艇勝利確率（0〜1）

        original_data = df.loc[idx_test]
        # DataFrame化して確認
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': original_data['日付'].values,
            'レース場': original_data['レース場'].values,
            'レース番号': original_data['レース番号'].values,
            f'1号艇勝利確率': y_pred_prob,
            '正解ラベル（1=勝ち）': y_test
        })
        
        return model,X_test,y_test,result_df

    def trian_binaly_LGBM(self,X,y,df):
        # ======== データ分割 ========
        # 元のインデックスを保持
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.2, random_state=42
        )

        # ======== LightGBMモデル構築 ========
        # データセットの作成
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # パラメータ設定
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        verbose_eval=50
        # モデル訓練
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=True), lgb.log_evaluation(verbose_eval)]
        )

        # ======== 評価 ========
        y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"\n1号艇モデル評価 (LightGBM):")
        print(f"- 精度: {accuracy:.4f}")
        print(f"- AUC: {auc:.4f}")
        print(f"- 適合率: {precision:.4f}")
        print(f"- 再現率: {recall:.4f}")

        # ======== 特徴量重要度の可視化 ========
        lgb.plot_importance(model, max_num_features=20, figsize=(10, 6))
        plt.title('Feature Importance')
        plt.show()

        # ======== 確率出力 ========
        original_data = original_data.loc[idx_test]
        # DataFrame化して確認
        result_df = pd.DataFrame({
            'original_index': idx_test,
            '日付': original_data['日付'].values,
            'レース場': original_data['レース場'].values,
            'レース番号': original_data['レース番号'].values,
            '1号艇勝利確率': y_pred_prob,
            '正解ラベル（1=勝ち）': y_test
        })

        return model, X_test, y_test, result_df

    def calculate_return_rate_binaly(self, result_df, df_odds, threshold=0.6, bet_type='単勝', bet_amount=100):
        """
        回収率を計算する関数
        
        Parameters:
        ----------
        original_df : pd.DataFrame
            元のデータフレーム（日付、レース番号などの情報を含む）
        result_df : pd.DataFrame
            モデルの予測結果（original_indexを含む）
        df_odds : pd.DataFrame
            オッズデータ
        threshold : float
            ベットする確率の閾値
        bet_type : str
            賭ける舟券の種類（'単勝', '複勝'など）
        bet_amount : int
            1レースあたりの賭け金
        """        
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == bet_type]
        
        # 1号艇のオッズのみを取得（単勝の場合）
        if bet_type == '単勝':
            odds_filtered = odds_filtered[odds_filtered['組合せ'] == '1']
        elif bet_type == '複勝':
            odds_filtered = odds_filtered[odds_filtered['組合せ'] == '1']
        # 他の賭け方にも対応可能
        
        merged_df = pd.merge(result_df,odds_filtered,
            on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 閾値以上の予測のみを選択
        high_conf_bets = merged_df[merged_df['1号艇勝利確率'] >= threshold]
        
        # 回収率計算
        total_bet = len(high_conf_bets) * bet_amount
        total_return = high_conf_bets[high_conf_bets['正解ラベル（1=勝ち）'] == 1]['払戻金'].sum()
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
        
        print(f"\nベットタイプ: {bet_type}")
        print(f"閾値: {threshold:.0%} 以上")
        print(f"ベット件数: {len(high_conf_bets)} レース/{len(merged_df)}レース")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        return roi, high_conf_bets

    def calculate_return_rate_multiclass(self, race_df, result_df, df_odds, prob_threshold=0.5, margin_threshold=0.1, bet_type='単勝',bet_amount=100):
        """
        多クラス分類モデルを使用して舟券予測と回収率計算を行う
        確率に応じてベット金額を変更:
        - 0.7 <= prob < 0.8 → 100円
        - 0.8 <= prob < 0.9 → 200円
        - prob >= 0.9 → 300円
        """
        # 元のデータと結果を結合
        merged_df = pd.merge(result_df, race_df[['1着艇', '2着艇', '3着艇']].reset_index(),
                            left_on='original_index', right_on='index', how='left')
            
        # オッズデータをフィルタリング
        odds_filtered = df_odds[df_odds['舟券種'] == bet_type].copy()
                            
        final_df = pd.merge(merged_df, odds_filtered,
                            on=['日付', 'レース場', 'レース番号'], how='left')
                
        # 各艇の確率を取得
        boat_probs = []
        for i in range(1, 7):
            boat_probs.append(final_df[f'{i}号艇勝利確率'].values)
        boat_probs = np.column_stack(boat_probs)
            
        # トップの艇番と確率、2番手の確率を計算
        final_df['top_boat'] = np.argmax(boat_probs, axis=1) + 1
        final_df['top_boat'] = final_df['top_boat'].astype(str)
        final_df['top_prob'] = np.max(boat_probs, axis=1)
        final_df['second_prob'] = np.sort(boat_probs, axis=1)[:, -2]
        final_df['prob_diff'] = final_df['top_prob'] - final_df['second_prob']
                    
        # 賭ける条件を満たすレースをフィルタリング
        condition = (
            (final_df['top_prob'] >= prob_threshold) & 
            (final_df['prob_diff'] >= margin_threshold))
        bets_df = final_df[condition].copy()

        if bets_df.empty:
            print("条件を満たすレースがありませんでした")
            return 0, pd.DataFrame()
                
        # 的中判定
        bets_df['1着艇'] = bets_df['1着艇'].astype(str)
        bets_df['2着艇'] = bets_df['2着艇'].astype(str)
        bets_df['3着艇'] = bets_df['3着艇'].astype(str)

        if bet_type == '単勝':
            bets_df['won'] = (bets_df['top_boat'] == bets_df['1着艇'])
        elif bet_type == '複勝':
            bets_df['won'] = (
                (bets_df['top_boat'] == bets_df['1着艇']) | 
                (bets_df['top_boat'] == bets_df['2着艇']) | 
                (bets_df['top_boat'] == bets_df['3着艇']))

        # 一律100円ベット
        bets_df['bet_amount'] = bet_amount
            
        # 回収率計算
        total_bet = bets_df['bet_amount'].sum()
        total_return = bets_df[bets_df['won']]['払戻金'].sum()*(bet_amount/100)
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
            
        # 結果表示
        print(f"\nベットタイプ: {bet_type}")
        print(f"確率閾値: {prob_threshold:.0%} 以上")
        print(f"トップ確率差閾値: {margin_threshold:.0%} 以上")
        print(f"ベット件数: {len(bets_df)} レース/{len(result_df)}レース")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
            
        return roi,bets_df

    def compiling_and_preprocess_and_train_lane_data(self):
        """1見すると、正答率85％あるが、1号艇を1それ以外は0と予測するだけで達成できるから意味なし"""
        merged_csv_folder = os.path.join(self.folder, "merged_csv")
        all_files = [os.path.join(merged_csv_folder, f) for f in os.listdir(merged_csv_folder) if f.endswith('.csv')]

        all_dataframes = []
        for filepath in all_files:
            filename = os.path.basename(filepath)
            file_date = "20" + filename[:2] + "-" + filename[2:4] + "-" + filename[4:6]

            try:
                df = pd.read_csv(filepath, encoding='shift-jis')
            except Exception as e:
                print(f"読み込みエラー: {filepath} → {e}")
                continue

            # 必須列チェック
            if '着順' not in df.columns:
                print(f"'着順' 列が見つかりません → {filepath}")
                continue
        
            df['日付'] = file_date
            df = df.rename(columns={'着順': 'Rank'})

            cols = [
                '日付', 'レース場', 'レース番号', '艇番', '年齢', '体重', '級別',
                '全国勝率', '全国2連対率', '当地勝率', '当地2連対率',
                'モーター番号', 'モーター2連対率', 'ボート番号', 'ボート2連対率',
                '展示タイム','平均ST',
                'Rank'
            ]

            # 必要な列がすべて揃ってるか確認してから処理
            if all(col in df.columns for col in cols):
                df_out = df[cols].copy()
                all_dataframes.append(df_out)
            else:
                print(f"列が足りない → {filepath}")

        # すべてのDataFrameをまとめて返す
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # ========================
        #  学習用前処理
        # ========================
        # ラベル変換：Rank==1 → 1、それ以外 → 0
        df['Rank'] = (df['Rank'] == 1).astype(int)

        # 不要な列を除外
        df = df.drop(['日付', 'レース番号'], axis=1)

        # カテゴリ列のエンコード（級別など）
        df = pd.get_dummies(df, columns=["レース場","級別"], drop_first=True)

        # 残りの欠損値をすべて0で埋める（例外的に）
        df = df.fillna(0)

        # 型を揃える
        df = df.astype("float32")
        
        # 特徴量・目的変数に分ける
        X = df.drop(columns=["Rank"]).values
        y = df["Rank"].values

        # ========================
        # 学習用データ分割
        # ========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 正規化
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # ========================
        # シンプルなモデル構築
        # ========================
        model = models.Sequential()
        model.add(layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))  # バイナリ出力

        model.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=256,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            verbose=1)

        # ========================
        # モデル評価
        # ========================hb
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # ========================
        # 可視化（lossとaccuracy）
        # ========================
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='Training acc')
        plt.plot(history.history['val_accuracy'], label='Validation acc')
        plt.title('Accuracy')
        plt.legend()
        plt.show()        

if __name__=="__main__":
    BoatraceML = BoatraceML(folder = "C:\\Users\\msy-t\\boatrace-ai\\data")
    # レースデータとオッズデータのコンパイル
    #df_race = BoatraceML.compiling_race_data()
    #df_odds = BoatraceML.Compiling_odds_data()
    #df_race.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv", index=False, encoding="shift_jis")
    #df_odds.to_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv", index=False, encoding="shift_jis")
    df_race = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_race.csv", encoding='shift-jis')
    df_odds = pd.read_csv("C:\\Users\\msy-t\\boatrace-ai\\data\\agg_results\\df_odds.csv", encoding='shift-jis')
    
    # モデル訓練（マルチクラス）
    X,y,mean,std = BoatraceML.preprocess_multiclass(df_race)
    model,X_test,y_test,result_df = BoatraceML.train_multiclass_LGBM(X,y,df_race)
    
    # 単勝予測と回収率計算
    roi = BoatraceML.calculate_return_rate_multiclass(race_df=df_race,result_df=result_df,df_odds=df_odds,prob_threshold=0.7,margin_threshold=0.0,bet_type='単勝',bet_amount=100)
    
    # モデル訓練（バイナリクラス）
    #X,y,mean,std = BoatraceML.preprocess_binaly(df_race)
    #model,X_test,y_test,result_df = BoatraceML.train_binaly_Keras(X,y,df_race)
        
    # 回収率計算（単勝の場合）
    #roi, bet_results = BoatraceML.calculate_return_rate_1_win(result_df=result_df,df_odds=df_odds,threshold=0.70,bet_type='単勝',bet_amount=100)
    # 複勝の回収率計算（閾値は調整が必要）
    #roi_fukusho, fukusho_bets = BoatraceML.calculate_return_rate_1_win(result_df=result_df,df_odds=df_odds,threshold=0.55, bet_type='複勝',bet_amount=100)