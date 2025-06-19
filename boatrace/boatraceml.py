import pandas as pd
import numpy as np

from boatrace.machinelearning.bettingstrategyevaluator import BettingStrategyEvaluator
from boatrace.machinelearning.modeltrainer import ModelTrainer
from boatrace.machinelearning.datacompiler import DataCompiler
from .base.base import BoatraceBase
class BoatraceML(BoatraceBase):
    """メインクラス - 各コンポーネントを統合"""
    def __init__(self):
        super().__init__()
        self.data_compiler = DataCompiler()
        self.model_trainer = ModelTrainer()
        self.evaluator = BettingStrategyEvaluator()
    
    def run_pipeline(self,
                     train_start_date,train_end_date,test_start_date,test_end_date,compile=True):
        if compile==True:
            train_df = self.data_compiler.compile_race_data(start_date=train_start_date,end_date=train_end_date,target_places=None)
            test_df = self.data_compiler.compile_race_data(start_date=test_start_date,end_date=test_end_date,target_places=None)
            train_df.to_csv(self.data_folder+"/agg_results/train_df.csv",index=False, encoding='shift-jis')
            test_df.to_csv(self.data_folder+"/agg_results/test_df.csv",index=False, encoding='shift-jis')

        # データ読み込み
        train_df= pd.read_csv(self.data_folder+"/agg_results/train_df.csv", encoding='shift-jis')
        test_df= pd.read_csv(self.data_folder+"/agg_results/test_df.csv", encoding='shift-jis')
                
        # === 1着予想 ===
        # 1号艇は勝てるか？
        # 前処理器の初期化
        preprocessor = DataCompiler()
        
        # 特徴量前処理（全モデル共通）
        train_X = preprocessor.preprocess_features(train_df)
        test_X = preprocessor.preprocess_features(test_df)
        
        # バイナリ分類（1号艇が1着かどうか）
        y_binary = preprocessor.get_binary_target(train_df, boat_num=1, top_num=1)
                
        # 前処理情報の保存
        preprocessor.save_preprocessor()
        
        model_one = self.model_trainer.train_binary_lgbm(train_X, y_binary, train_df, preprocessor.categorical_indices)
        
        # 1号艇が負けたとき勝つのは？
        y_multi = self.data_compiler.get_multiclass_target(train_df, top_num=1) 
        model_defeat_one = self.model_trainer.train_multiclass_lgbm_exclusion_one(train_X, y_multi, train_df, preprocessor.categorical_indices)
        
        # 確率計算
        p1 = model_one.predict_proba(test_X)[:, 1]  # 1号艇の1着確率
        p2to6 = model_defeat_one.predict_proba(test_X)  # 1号艇以外の1着確率
        
        # 最終確率計算
        final_probs = np.zeros((len(test_X), 6))
        final_probs[:, 0] = p1 * 1
        final_probs[:, 1:6] = (1 - p1 * 1).reshape(-1, 1) * p2to6
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)  # 正規化
        top6_indices = np.argsort(-final_probs, axis=1) + 1  # +1で1-basedの艇番号に
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            test_df.loc[:, f'{boat_num}号艇勝利確率'] = final_probs[:, boat_num-1]
            test_df[f'{boat_num}着艇予想'] = top6_indices[:, boat_num-1]
        
        # === 2着予想 ===
        model_twos = {}
        for num in range(1, 7):
            y_mullti = self.data_compiler.get_multiclass_target(train_df, top_num=2)
            model_two = self.model_trainer.train_multiclass_lgbm_target_1st(train_X, y_mullti, train_df, target_1st_num=num)
            probs_two = model_two.predict_proba(test_X)
            
            candidate_boats = [i for i in range(1, 7) if i != num]
            predicted_boats = [candidate_boats[idx] for idx in np.argmax(probs_two, axis=1)]
            
            test_df[f'{num}号艇が1着のとき2着艇予想'] = predicted_boats
            for idx, boat_num in enumerate(candidate_boats):
                test_df[f'{num}号艇が1着のとき{boat_num}号艇の2着確率'] = probs_two[:, idx]
            model_twos[num] = model_two
        
        # === 3着予想 ===
        model_threes = {}
        results = {'predictions': {},'probabilities': {}}

        for first_num in range(1, 7):
            for second_num in range(1, 7):
                if first_num == second_num:
                    continue
                    
                y_multi = self.data_compiler.get_multiclass_target(train_df, top_num=3)
                model_three = self.model_trainer.train_multiclass_lgbm_target_1st_2nd(train_X, y_multi, train_df, 
                    target_1st_num=first_num, target_2nd_num=second_num)
                probs_three = model_three.predict_proba(test_X)
                candidate_boats = [i for i in range(1, 7) if i not in [first_num, second_num]]
                
                key = f'{first_num}号艇1着&{second_num}号艇2着'
                results['predictions'][f'{key}のとき3着艇予想'] = [candidate_boats[idx] for idx in np.argmax(probs_three, axis=1)]
                
                for idx, boat_num in enumerate(candidate_boats):
                    results['probabilities'][f'{key}のとき{boat_num}号艇の3着確率'] = probs_three[:, idx]
                
                model_threes[(first_num, second_num)] = model_three
                
        # DataFrameに結果を追加
        predictions_df = pd.DataFrame(results['predictions'])
        probabilities_df = pd.DataFrame(results['probabilities'])

        # 元のtest_dfと結合（inplace=Falseで新しいDataFrameを作成）
        final_test_df = pd.concat([test_df.reset_index(drop=True),
            predictions_df.reset_index(drop=True),
            probabilities_df.reset_index(drop=True)], axis=1)
        
        return final_test_df,model_one,model_defeat_one,model_twos,model_threes
        
    def run_pipeline_validation(self,result_df,compile=True):
        if compile:
            O_df = self.data_compiler.compile_odds_data(start_date="2025-03-01",end_date="2025-06-18")
            all_odds_df = self.data_compiler.compile_all_odds_data(start_date="2025-03-01",end_date="2025-06-18")
            O_df.to_csv(self.data_folder+"/agg_results/O_df.csv",index=False, encoding='shift-jis')
            all_odds_df.to_csv(self.data_folder+"/agg_results/odds_df.csv",index=False, encoding='shift-jis')
        O_df = pd.read_csv(self.data_folder+"/agg_results/O_df.csv", encoding='shift-jis')
        all_odds_df = pd.read_csv(self.data_folder+"/agg_results/odds_df.csv", encoding='shift-jis')
        
        # ベット
        #self.evaluator.Win_calculate_return_rate(result_df, O_df, bet_amount=100)
        #self.evaluator.Duble_calculate_return_rate(result_df, O_df, bet_amount=100)
        self.evaluator.Trifecta_calculate_return_rate(result_df, O_df, all_odds_df,bet_amount=100)

    def run_pipeline_practice(self,model_one,model_defeat_one,model_twos,model_threes,all_odds,target_date="2024-04-01", place='大村', race_no=1,weather='晴',wind_dir='東',wind_spd=0,wave_hgt=1, Exhibition_time={1:7.00, 2:7.00, 3:7.00, 4:7.00, 5:7.00, 6:7.00}):
        new_df = self.data_compiler.compile_race_data_B(target_date,place,race_no,weather,wind_dir,wind_spd,wave_hgt,Exhibition_time)
        loaded_preprocessor = DataCompiler.load_preprocessor()
        X_new = loaded_preprocessor.preprocess_features(new_df)

        # ベットタイム
        p1 = model_one.predict_proba(X_new)[:, 1]  # 1号艇の1着確率
        p2to6 = model_defeat_one.predict_proba(X_new)  # 1号艇以外の1着確率
        
        # 最終確率計算
        final_probs = np.zeros((len(X_new), 6))
        final_probs[:, 0] = p1
        final_probs[:, 1:6] = (1 - p1).reshape(-1, 1) * p2to6
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)  # 正規化
        top6_indices = np.argsort(-final_probs, axis=1) + 1  # +1で1-basedの艇番号に
        
        # 各艇番の予測確率を追加
        for boat_num in range(1, 7):
            new_df.loc[:, f'{boat_num}号艇勝利確率'] = final_probs[:, boat_num-1]
            new_df[f'{boat_num}着艇予想'] = top6_indices[:, boat_num-1]
                
        # === 2着予想 ===
        for num in range(1, 7):
            probs_two = model_twos[num].predict_proba(X_new)
            
            candidate_boats = [i for i in range(1, 7) if i != num]
            predicted_boats = [candidate_boats[idx] for idx in np.argmax(probs_two, axis=1)]
            
            new_df[f'{num}号艇が1着のとき2着艇予想'] = predicted_boats
            for idx, boat_num in enumerate(candidate_boats):
                new_df[f'{num}号艇が1着のとき{boat_num}号艇の2着確率'] = probs_two[:, idx]

        # === 3着予想 ===
        results = {'predictions': {},'probabilities': {}}

        for first_num in range(1, 7):
            for second_num in range(1, 7):
                if first_num == second_num:
                    continue
                    
                probs_three = model_threes[(first_num, second_num)].predict_proba(X_new)
                candidate_boats = [i for i in range(1, 7) if i not in [first_num, second_num]]
                
                key = f'{first_num}号艇1着&{second_num}号艇2着'
                results['predictions'][f'{key}のとき3着艇予想'] = [candidate_boats[idx] for idx in np.argmax(probs_three, axis=1)]
                
                for idx, boat_num in enumerate(candidate_boats):
                    results['probabilities'][f'{key}のとき{boat_num}号艇の3着確率'] = probs_three[:, idx]
        
                # DataFrameに結果を追加
        predictions_df = pd.DataFrame(results['predictions'])
        probabilities_df = pd.DataFrame(results['probabilities'])

        # 元のnew_dfと結合（inplace=Falseで新しいDataFrameを作成）
        final_new_df = pd.concat([new_df.reset_index(drop=True),
            predictions_df.reset_index(drop=True),
            probabilities_df.reset_index(drop=True)], axis=1)
                
        self.evaluator.Trifecta_practice(final_new_df,all_odds)