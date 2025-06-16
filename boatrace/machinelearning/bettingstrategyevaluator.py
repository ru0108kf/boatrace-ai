import os
import pandas as pd
import numpy as np

class BettingStrategyEvaluator:
    """ベッティング戦略の評価を担当するクラス"""
    def Win_calculate_return_rate(self, df, df_odds, threshold={1: 0.7,2: 0.4,3: 0.3,4: 0.2,5: 0.1,6: 0.1}, bet_amount=100):
        """単勝回収率算出"""
        # レース場フィルタリングを追加
        #target_stadiums = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
        #filtered_df = df[df['レース場'].isin(target_stadiums)].copy()
        filtered_df = df
        
        # オッズデータと結合（単勝のみをフィルタリング）
        odds_filtered = df_odds[df_odds['舟券種'] == '単勝'].copy() 
        bets_df = pd.merge(filtered_df,odds_filtered,on=['日付', 'レース場', 'レース番号'],how='left')
        
        # 高信頼度のベットをフィルタリング
        high_conf_bets = self._create_bet_strategy2(bets_df,threshold)
        
        if not high_conf_bets:
            print("ベット対象となるレースがありませんでした")
            return 0.0
        
        high_conf_bets_df = pd.DataFrame(high_conf_bets)
        
        # クロス集計表を表示
        print(pd.crosstab(
            high_conf_bets_df['1着艇予想'], 
            high_conf_bets_df['1着艇'], 
            margins=True, 
            margins_name="合計"
        ))
        
        # 回収率計算
        total_bet = len(high_conf_bets_df) * bet_amount
        correct_bets = high_conf_bets_df[high_conf_bets_df['won']]
        total_return = correct_bets['払戻金'].sum() * (bet_amount / 100)
        roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0.0
        
        print(f"\nベットタイプ: 単勝")
        print(f"閾値: {threshold}")
        print(f"ベット件数: {len(high_conf_bets_df)} レース/{len(bets_df)}レース")
        print(f"的中数: {len(correct_bets)} 回")
        print(f"的中率: {len(correct_bets)/len(high_conf_bets_df):.2%}")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        """
        # 的中レース詳細
        if len(correct_bets) > 0:
            print("\n【的中レース詳細】")
            for _, row in correct_bets.iterrows():
                print(
                    f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 艇番{row['1着艇予想']} → 結果 艇番{row['1着艇']} "
                    f"配当 {row['払戻金']}円 "
                    f"予測確率: {row[f'{row['1着艇予想']}号艇勝利確率']:.2f}"
                )
        else:
            print("\n的中したレースはありませんでした")
        """
        return roi  

    def _create_bet_strategy1(self,bets_df,threshold):
        # 高信頼度のベットをフィルタリング
        high_conf_bets = []
        for _, row in bets_df.iterrows():
            for boat_num in range(1, 7):
                win_rate = row.get(f'{boat_num}号艇勝利確率', 0)
                boat_threshold = threshold.get(boat_num, 0.5)
                
                if win_rate >= boat_threshold:
                    # ベット対象として行をコピーし、予想艇番号を設定
                    bet_row = row.copy()
                    bet_row['1着艇予想'] = boat_num
                    bet_row['predicted_patterns'] = f"艇番{boat_num}"
                    bet_row['won'] = (bet_row['1着艇'] == boat_num)  # 的中フラグを追加
                    high_conf_bets.append(bet_row)
        return high_conf_bets
        
    def _create_bet_strategy2(self,bets_df,threshold):
        # 高信頼度のベットをフィルタリング（ロジック変更）
        high_conf_bets = []
        for _, row in bets_df.iterrows():
            best_bet_found = False
            # 各艇の勝利確率とその艇番号をリストに格納
            probabilities = []
            for boat_num in range(1, 7):
                win_rate = row.get(f'{boat_num}号艇勝利確率', 0)
                probabilities.append({'boat_num': boat_num, 'win_rate': win_rate})
            
            # 勝利確率が高い順にソート
            probabilities.sort(key=lambda x: x['win_rate'], reverse=True)
            
            # 確率が高い順に、閾値を超えているかチェック
            for prob_info in probabilities:
                boat_num = prob_info['boat_num']
                win_rate = prob_info['win_rate']
                boat_threshold = threshold.get(boat_num, 0.0) # 閾値が設定されていない艇番の場合のデフォルト値を0.0に修正
                
                if win_rate >= boat_threshold:
                    # ベット対象として行をコピーし、予想艇番号を設定
                    bet_row = row.copy()
                    bet_row['1着艇予想'] = boat_num
                    bet_row['predicted_patterns'] = f"艇番{boat_num}"
                    bet_row['won'] = (bet_row['1着艇'] == boat_num)  # 的中フラグを追加
                    high_conf_bets.append(bet_row)
                    best_bet_found = True
                    break # このレースではこれ以上ベットしない
        return high_conf_bets

    def Duble_calculate_return_rate(self, df, df_odds, bet_amount=100):
        #target_stadiums = ['大村', '徳山', '芦屋', '下関', '尼崎', '住之江']
        #filtered_df = df[df['レース場'].isin(target_stadiums)].copy()
        filtered_df = df

        # オッズデータとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "２連単"].copy()
        bets_df = pd.merge(filtered_df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
        # 初期化
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['bet_count'] = 0
        bets_df['bet_combinations'] = ""
        bets = 0

        # メイン処理
        for idx, row in bets_df.iterrows():
            P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
            P_2nd = {
                i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                for j in range(1,7) if j != i]
                for i in range(1,7)
            }
            # 2連単の全組み合わせの確率計算
            pattern_probs = []
            for i in range(1, 7):
                for j in range(1, 7):
                    if i == j:
                        continue
                    prob_1st = P_1st[i-1]
                    # 2着確率はP_2nd[i]の中でjに対応するインデックスは j-1 ただしj > iの場合はインデックスが変わるため調整
                    idx_2nd = j-1 if j < i else j-2
                    prob_2nd = P_2nd[i][idx_2nd]
                    pattern_probs.append(((i, j), prob_1st * prob_2nd))

            # 確率の高い順にソート
            pattern_probs.sort(key=lambda x: x[1], reverse=True)

            # ベット候補の抽出（例: 確率上位3つにベット）
            top_n = 1
            bet_patterns = pattern_probs[:top_n]

            # ベット数をカウント
            bets_df.at[idx, 'bet_count'] = len(bet_patterns)
            # ベットパターンを文字列で格納
            combinations = ", ".join([f"{p[0][0]}-{p[0][1]}" for p in bet_patterns])
            bets_df.at[idx, 'predicted_patterns'] = ", ".join([f"{p[0][0]}-{p[0][1]}" for p in bet_patterns])
            bets_df.at[idx, 'bet_combinations'] = bet_patterns
            
            actual = row['組合せ']
            bets_df.at[idx, 'won'] = actual in combinations

            # 合計ベット数を加算
            if len(bet_patterns) != 0:
                bets += 1
            
        # 回収率計算
        total_bet = bets_df['bet_count'].sum() * bet_amount
        total_return = bets_df[bets_df['won']]['払戻金'].sum()
        roi = total_return / total_bet if total_bet > 0 else 0
        
        # 結果表示
        print(f"\nベットタイプ: 2連単")
        print(f"ベット件数: {bets} レース/{len(bets_df)}レース")
        print(f"総ベット数: {bets_df['bet_count'].sum()} 点")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}") 
        """
        # 的中レース詳細
        if bets_df['won'].sum() > 0:
            print("\n【的中レース詳細】")
            won_races = bets_df[bets_df['won']].copy()
            
            print("\n【的中した3連単パターン】")
            for _, row in won_races.iterrows():
                print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']} → 結果 {row['組合せ']} "
                    f"配当 {row['払戻金']}円 "
                    f"予測確率:{[float(f'{row[f"{i}号艇勝利確率"]:.2f}') for i in range(1,7)]}")
        else:
            print("\n的中したレースはありませんでした") 
        """
        
    def Trifecta_calculate_return_rate(self, df, df_odds, all_odds, bet_amount=100, venues = None):     
        # レース場フィルタリング
        if venues is not None:
            # venues = ['宮島']
            df = df[df['レース場'].isin(venues)].copy()

        # オッズデータとマージ
        odds_filtered = df_odds[df_odds['舟券種'] == "３連単"].copy()
        bets_df = pd.merge(df, odds_filtered, on=['日付', 'レース場', 'レース番号'], how='left').copy()
        bets_df = pd.merge(bets_df, all_odds.copy(), on=['日付', 'レース場', 'レース番号'], how='left').copy()
        
        # 初期化
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['bet_count'] = 0
        bets_df['strategy_used'] = ""
        bets_df['bet_combinations'] = ""
        bets = 0

        # メイン処理
        for idx, row in bets_df.iterrows():
            P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
            P_2nd = {
                i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                for j in range(1,7) if j != i]
                for i in range(1,7)
            }
            P_3rd = {
                (i,j): [row[f'{i}号艇1着&{j}号艇2着のとき{k}号艇の3着確率'] 
                    for k in range(1,7) if k not in [i,j]]
                for i in range(1,7)
                for j in range(1,7)
                if i != j
            }
            
            combinations, strategy_name = self._create_formation_biased_odds_strategy2(P_1st, P_2nd, P_3rd,row)
            
            if len(combinations) != 0:
                bets += 1
            
            # 記録
            bets_df.at[idx, 'strategy_used'] = strategy_name
            bets_df.at[idx, 'predicted_patterns'] = ", ".join(combinations)
            bets_df.at[idx, 'bet_count'] = len(combinations)
            bets_df.at[idx, 'bet_combinations'] = combinations
            
            # 的中チェック
            actual = row['組合せ']
            bets_df.at[idx, 'won'] = actual in combinations
 
        # 回収率計算
        total_bet = bets_df['bet_count'].sum() * bet_amount
        total_return = bets_df[bets_df['won']]['払戻金'].sum()
        roi = total_return / total_bet if total_bet > 0 else 0
        
        
        # 戦術別の出現数と割合を計算
        strategy_counts = bets_df['strategy_used'].value_counts().reset_index()
        strategy_counts.columns = ['戦術名', '出現回数']
        strategy_counts['割合'] = (strategy_counts['出現回数'] / len(bets_df) * 100).round(1)

        # 結果表示
        print("\n【戦術使用頻度】")
        print(strategy_counts.to_string(index=False))

        # あるいはシンプルに
        print("\n【戦術別出現回数】")
        print(bets_df['strategy_used'].value_counts())
        
        # 結果表示
        print(f"\nベットタイプ: 3連単")
        print(f"ベット件数: {bets} レース/{len(bets_df)}レース")
        print(f"総ベット数: {bets_df['bet_count'].sum()} 点")
        print(f"的中数: {bets_df['won'].sum()} 回")
        print(f"総賭け金: {total_bet:,} 円")
        print(f"総払戻金: {total_return:,} 円")
        print(f"回収率: {roi:.2%}")
        
        # 的中レース詳細
        if bets_df['won'].sum() > 0:
            print("\n【的中レース詳細】")
            won_races = bets_df[bets_df['won']].copy()
            
            print("\n【的中した3連単パターン】")
            for _, row in won_races.iterrows():
                print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']} → 結果 {row['組合せ']} "
                    f"配当 {row['払戻金']}円 " f"戦術 {row['strategy_used']} ")
        else:
            print("\n的中したレースはありませんでした")
        
        return bets_df, roi

    def Trifecta_jissen(self, df, all_odds):
        bets_df = df.copy()  # 元のデータフレームをコピー
        
        # 初期化（全レコードに対して行う）
        bets_df['won'] = False
        bets_df['predicted_patterns'] = ""
        bets_df['strategy_used'] = ""
        bets_df['bet_combinations'] = ""
        
        # 各行に対して処理
        for idx, row in bets_df.iterrows():
            try:
                P_1st = [row[f'{i}号艇勝利確率'] for i in range(1,7)]
                P_2nd = {
                    i: [row[f'{i}号艇が1着のとき{j}号艇の2着確率'] 
                    for j in range(1,7) if j != i]
                    for i in range(1,7)
                }
                P_3rd = {
                    (i,j): [row[f'{i}号艇1着&{j}号艇2着のとき{k}号艇の3着確率'] 
                        for k in range(1,7) if k not in [i,j]]
                    for i in range(1,7)
                    for j in range(1,7)
                    if i != j
                }

                combinations, strategy_name = self._create_formation_biased_odds_strategy2(P_1st, P_2nd, P_3rd, all_odds)

                # 記録
                bets_df.at[idx, 'strategy_used'] = strategy_name
                bets_df.at[idx, 'predicted_patterns'] = ", ".join(combinations)
                bets_df.at[idx, 'bet_combinations'] = combinations
                
            except Exception as e:
                print(f"行 {idx} の処理中にエラーが発生しました: {str(e)}")
                continue
            
        for _, row in bets_df.iterrows():
            print(f"{row['日付']} {row['レース場']} {row['レース番号']}R: "
                    f"予測 {row['predicted_patterns']}" f" 戦術 {row['strategy_used']} "
                    f"予測1着確率:{row[f'1号艇勝利確率']:.2%} {row[f'2号艇勝利確率']:.2%} {row[f'3号艇勝利確率']:.2%} {row[f'4号艇勝利確率']:.2%} {row[f'5号艇勝利確率']:.2%} {row[f'6号艇勝利確率']:.2%} -"
                    f"\n1が1着のときの2着確率[2,3,4,5,6]: {row[f"1号艇が1着のとき2号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき3号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき4号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき5号艇の2着確率"]:.2%} {row[f"1号艇が1着のとき6号艇の2着確率"]:.2%}"
                    f"\n2が1着のときの2着確率[1,3,4,5,6]: {row[f'2号艇が1着のとき1号艇の2着確率']:.2%} {row[f'2号艇が1着のとき3号艇の2着確率']:.2%} {row[f'2号艇が1着のとき4号艇の2着確率']:.2%} {row[f'2号艇が1着のとき5号艇の2着確率']:.2%} {row[f'2号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n3が1着のときの2着確率[1,2,4,5,6]: {row[f'3号艇が1着のとき1号艇の2着確率']:.2%} {row[f'3号艇が1着のとき2号艇の2着確率']:.2%} {row[f'3号艇が1着のとき4号艇の2着確率']:.2%} {row[f'3号艇が1着のとき5号艇の2着確率']:.2%} {row[f'3号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n4が1着のときの2着確率[1,2,3,5,6]: {row[f'4号艇が1着のとき1号艇の2着確率']:.2%} {row[f'4号艇が1着のとき2号艇の2着確率']:.2%} {row[f'4号艇が1着のとき3号艇の2着確率']:.2%} {row[f'4号艇が1着のとき5号艇の2着確率']:.2%} {row[f'4号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n5が1着のときの2着確率[1,2,3,4,6]: {row[f'5号艇が1着のとき1号艇の2着確率']:.2%} {row[f'5号艇が1着のとき2号艇の2着確率']:.2%} {row[f'5号艇が1着のとき3号艇の2着確率']:.2%} {row[f'5号艇が1着のとき4号艇の2着確率']:.2%} {row[f'5号艇が1着のとき6号艇の2着確率']:.2%}"
                    f"\n6が1着のときの2着確率[1,2,3,4,5]: {row[f'6号艇が1着のとき1号艇の2着確率']:.2%} {row[f'6号艇が1着のとき2号艇の2着確率']:.2%} {row[f'6号艇が1着のとき3号艇の2着確率']:.2%} {row[f'6号艇が1着のとき4号艇の2着確率']:.2%} {row[f'6号艇が1着のとき5号艇の2着確率']:.2%}"
            )
        return bets_df

    def _create_formation1(self, P_1st, P_2nd, P_3rd, threshold=0.2, hole_threshold=0.12):
        formations = []
        strategy_names = []  # 使用された戦略名を記録
        
        entropy = sum([-p * (p and (p > 0) and np.log(p)) for p in P_1st])
        
        sorted_1st = sorted([(p, i+1) for i, p in enumerate(P_1st)], reverse=True)
        top1_p, top1 = sorted_1st[0]
        top2_p, top2 = sorted_1st[1]
        top3_p, top3 = sorted_1st[2]
        
        top1_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != top1],P_2nd[top1])]
        sorted_top1_2nd = sorted(top1_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
        top2_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != top2],P_2nd[top2])]
        sorted_top2_2nd = sorted(top2_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
        top3_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != top3],P_2nd[top3])]
        sorted_top3_2nd = sorted(top3_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
        
        top1_2nd_top1_p, top1_2nd_top1 = sorted_top1_2nd[0]
        top1_2nd_top2_p, top1_2nd_top2 = sorted_top1_2nd[1]
        top1_2nd_top3_p, top1_2nd_top3 = sorted_top1_2nd[2]
        
        top2_2nd_top1_p, top2_2nd_top1 = sorted_top2_2nd[0]
        top2_2nd_top2_p, top2_2nd_top2 = sorted_top2_2nd[1]
        top2_2nd_top3_p, top2_2nd_top3 = sorted_top2_2nd[2]
        
        top3_2nd_top1_p, top3_2nd_top1 = sorted_top3_2nd[0]
        top3_2nd_top2_p, top3_2nd_top2 = sorted_top3_2nd[1]
        top3_2nd_top3_p, top3_2nd_top3 = sorted_top3_2nd[2]
        
        if entropy > 1.6: 
            # エントロピーが高すぎるから、賭けない　→　予測不能
            return [], strategy_names
        if (top1_p - top2_p) > threshold:
            # 1号艇と2号艇の差が顕著
            if (top1==1 and top1_p < 0.6) or (top1==2 and top1_p < 0.5):
                if (top1_2nd_top1_p - top1_2nd_top2_p)>threshold:
                    strategy_names.append("1-1-2")
                    formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
                    formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top3}")
                else:
                    strategy_names.append("1-1=1")
                    formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
                    formations.append(f"{top1}-{top1_2nd_top2}-{top1_2nd_top1}")
            else:
                strategy_names.append("1-1-1")
                formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
            # 穴狙い 456のうちどれかの勝率が12％以上ある場合
            top4s = [t+1 for t in sorted(range(6), key=lambda x: -P_1st[x])[:4]]
            for dark in range(5, 7):
                if P_1st[dark-1] > hole_threshold and (dark in top4s):
                    strategy_names.append("Dark Horse Pursuit (Win Prob)")
                    dark_2nd_prob_boat_pairs = [(prob, boat) for boat, prob in zip([x for x in range(1, 7) if x != dark],P_2nd[dark-1])]
                    sorted_dark_2nd = sorted(dark_2nd_prob_boat_pairs, key=lambda x: x[0], reverse=True)
                    dark_2nd_top1_p, dark_2nd_top1 = sorted_dark_2nd[0]
                    dark_2nd_top2_p, dark_2nd_top2 = sorted_dark_2nd[1]
                    dark_2nd_top3_p, dark_2nd_top3 = sorted_dark_2nd[2]
                    formations.append(f"{dark}-{dark_2nd_top1}-{dark_2nd_top2}")
                    formations.append(f"{dark}-{dark_2nd_top2}-{dark_2nd_top1}")
                    formations.append(f"{dark}-{dark_2nd_top1}-{dark_2nd_top3}")
                    formations.append(f"{dark}-{dark_2nd_top2}-{dark_2nd_top3}")
        elif top2_p > 0.2:
            # 2つの艇が拮抗
            strategy_names.append("Twin Peaks Strategy")
            formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
            formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top3}")
            formations.append(f"{top2}-{top2_2nd_top1}-{top2_2nd_top2}")
            formations.append(f"{top2}-{top2_2nd_top1}-{top2_2nd_top3}")
                    
        elif ((top2_p - top3_p) <= threshold and top3_p > 0.12):
            # 3つの艇が拮抗
            strategy_names.append("Triple Threat Strategy")
            formations.append(f"{top1}-{top1_2nd_top1}-{top1_2nd_top2}")
            formations.append(f"{top1}-{top1_2nd_top2}-{top1_2nd_top1}")
            formations.append(f"{top2}-{top2_2nd_top1}-{top2_2nd_top2}")
            formations.append(f"{top2}-{top2_2nd_top2}-{top2_2nd_top1}")
            formations.append(f"{top3}-{top3_2nd_top1}-{top3_2nd_top2}")
            formations.append(f"{top3}-{top3_2nd_top2}-{top3_2nd_top1}")

        # Remove duplicates and sort
        formations = list(set(formations))
        
        return formations, strategy_names

    def _create_formation2(self, P_1st, P_2nd, P_3rd):
        strategies = []
        combinations = []
        
        # 1. 基本データ準備
        boat_numbers = list(range(1, 7))  # 1-6号艇
        sorted_boats = sorted(zip(P_1st, boat_numbers), key=lambda x: -x[0])
        
        # 2. 各艇のインデックスマッピング（1号艇→0ではなく1のまま）
        top1_p, top1 = sorted_boats[0]
        top2_p, top2 = sorted_boats[1]
        top3_p, top3 = sorted_boats[2]
        
        # 3. 安全な確率取得関数
        def get_second_prob(first, second):
            """ P_2ndから安全に2着確率を取得 """
            if first not in P_2nd:
                return 0
            second_list = [b for b in boat_numbers if b != first]
            try:
                idx = second_list.index(second)
                return P_2nd[first][idx]
            except (ValueError, IndexError):
                return 0
        
        # 4. 戦略1: 1番人気を外した組み合わせ
        if top1_p > 0.5:  # 1番人気が50%以上
            # 2-3番人気を1着に
            for first in [top2, top3]:
                # 2着候補（1番人気優先）
                seconds = sorted(
                    [(get_second_prob(first, b), b) for b in boat_numbers if b != first],
                    reverse=True
                )
                
                # 1番人気を2着に配置
                if top1 != first:
                    second = top1
                    # 3着候補（2着確率10%以上）
                    thirds = [
                        b for b in boat_numbers 
                        if b not in [first, second] 
                        and get_second_prob(second, b) > 0.1
                    ][:2]
                    
                    for third in thirds:
                        combo = f"{first}-{second}-{third}"
                        combinations.append(combo)
                        strategies.append("Top1 Exclusion")
        
        # 5. 戦略2: ダークホース活用（4-6番人気）
        dark_horses = [b for p, b in sorted_boats[3:] if p > 0.07]  # 7%以上
        for first in dark_horses[:2]:  # 最大2艇まで
            seconds = sorted(
                [(get_second_prob(first, b), b) for b in boat_numbers if b != first],
                reverse=True
            )[:2]  # 2着候補トップ2
            
            for sec_p, second in seconds:
                if sec_p < 0.1:
                    continue
                    
                # 3着は1番人気or2番人気
                thirds = [b for b in [top1, top2] if b not in [first, second]][:1]
                for third in thirds:
                    combo = f"{first}-{second}-{third}"
                    combinations.append(combo)
                    strategies.append("Dark Horse")
        
        # 6. デフォルト戦略（最低1組み合わせ）
        if not combinations:
            combo = f"{top1}-{top2}-{top3}"
            combinations.append(combo)
            strategies.append("Default")
        
        return list(set(combinations)), list(set(strategies))

    def _create_formation3(self, P_1st, P_2nd, P_3rd):
        combinations = []
        strategy_name = "3着確率考慮"
        
        threshold_1st=0.15
        threshold_2nd=0.2
        threshold_3rd=0.25
                
        try:
            # 1着候補を選定
            first_candidates = [i for i in range(1,7) if P_1st[i-1] >= threshold_1st]
            
            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                # 確率リストの長さを確認
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]
                
                second_candidates = [j for j in range(1,7) 
                                if j != first and 
                                len(valid_second_probs) >= j and 
                                valid_second_probs[j-1] >= threshold_2nd]
                
                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]
                    
                    third_candidates = [k for k in range(1,7) 
                                    if k not in [first, second] and 
                                    len(valid_third_probs) >= k and 
                                    valid_third_probs[k-1] >= threshold_3rd]
                    
                    for third in third_candidates:
                        # 三連単組み合わせを生成 (1着-2着-3着)
                        combo = f"{first}-{second}-{third}"
                        combinations.append(combo)
            
            def filter_top_combinations(combinations, P_1st, P_2nd, P_3rd, top_n=10):
                combo_scores = []
                for combo in combinations:
                    first, second, third = map(int, combo.split('-'))
                    # 組み合わせの総合確率を計算
                    score = (P_1st[first-1] * 
                            P_2nd[first][second-1] * 
                            P_3rd[(first, second)][third-1])
                    combo_scores.append((combo, score))
                
                # 確率の高い順にソート
                combo_scores.sort(key=lambda x: x[1], reverse=True)
                
                # トップNを返す
                return [combo for combo, score in combo_scores[:top_n]]
            
            # 組み合わせが多すぎる場合、確率の高いもののみにフィルタリング
            if len(combinations) > 10:
                combinations = filter_top_combinations(combinations, P_1st, P_2nd, P_3rd, top_n=10)
                strategy_name = "トップ10組み合わせ"
        except Exception as e:
            print(f"Error in _create_formation3: {str(e)}")
            print(f"P_1st: {P_1st}")
            print(f"P_2nd: {P_2nd}")
            print(f"P_3rd: {P_3rd}")
            combinations = []
            strategy_name = "エラー発生"
        
        return combinations, strategy_name

    def _create_formation4(self, P_1st, P_2nd, P_3rd):
        """
        1着、2着、3着の確率を考慮し、確率の高い順に上位3つの三連単組み合わせを返す
        """
        combo_scores = []  # (組み合わせ, スコア) を保存するリスト
        n = 1
        strategy_name = f"上位{n}組み合わせ"
        
        try:
            # 1着候補を選定 (確率>0のもの)
            first_candidates = [i for i in range(1,7) if P_1st[i-1] > 0]
            
            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]
                
                second_candidates = [j for j in range(1,7) 
                                if j != first and 
                                len(valid_second_probs) > j-1 and 
                                valid_second_probs[j-1] > 0]
                
                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]
                    
                    third_candidates = [k for k in range(1,7) 
                                    if k not in [first, second] and 
                                    len(valid_third_probs) > k-1 and 
                                    valid_third_probs[k-1] > 0]
                    
                    for third in third_candidates:
                        # 組み合わせの総合確率を計算
                        current_score = (P_1st[first-1] * 
                                    P_2nd[first][second-1] * 
                                    P_3rd[(first, second)][third-1])
                        
                        # 組み合わせとスコアを保存
                        combo_scores.append((f"{first}-{second}-{third}", current_score))
            
            # スコアの高い順にソート
            combo_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 上位3つの組み合わせを取得
            top_combos = [combo for combo, score in combo_scores[:n]]
            
            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"
                
        except Exception as e:
            print(f"Error in _create_formation4: {str(e)}")
            print(f"P_1st: {P_1st}")
            print(f"P_2nd: {P_2nd}")
            print(f"P_3rd: {P_3rd}")
            return [], "エラー発生"
        
    def _create_formation5(self, P_1st, P_2nd, P_3rd, score_threshold=0.05): 
        """
        1着、2着、3着の確率を考慮し、確率の高い順に上位n個の三連単組み合わせを返す
        score_threshold: この閾値未満の総合確率の組み合わせは除外する
        """
        combo_scores = []  # (組み合わせ, スコア) を保存するリスト
        n = 1
        strategy_name = f"上位{n}組み合わせ"
        
        try:
            # 1着候補を選定 (確率>0のもの)
            first_candidates = [i for i in range(1, 7) if P_1st[i-1] > 0]
            
            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]
                
                second_candidates = [j for j in range(1, 7) 
                                    if j != first and 
                                    len(valid_second_probs) > j-1 and 
                                    valid_second_probs[j-1] > 0]
                
                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]
                    
                    third_candidates = [k for k in range(1, 7) 
                                        if k not in [first, second] and 
                                        len(valid_third_probs) > k-1 and 
                                        valid_third_probs[k-1] > 0]
                    
                    for third in third_candidates:
                        # 組み合わせの総合確率を計算
                        current_score = (P_1st[first-1] * P_2nd[first][second-1] * P_3rd[(first, second)][third-1])
                        
                        ### 変更点 2: 計算したスコアが閾値以上かチェック ###
                        if current_score >= score_threshold:
                            # 組み合わせとスコアを保存
                            combo_scores.append((f"{first}-{second}-{third}", current_score))
            
            # スコアの高い順にソート
            combo_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 上位n個の組み合わせを取得
            top_combos = [combo for combo, score in combo_scores[:n]]
            
            if top_combos:
                return top_combos, strategy_name
            else:
                # 閾値を超え、かつ上位n個に残る組み合わせがなかった場合
                return [], "有効な組み合わせなし"
                
        except Exception as e:
            print(f"Error in _create_formation4: {str(e)}")
            # (エラー処理の詳細は省略)
            return [], "エラー発生"
        
    def _create_formation_expected_value1(self, P_1st, P_2nd, P_3rd, all_odds_for_race, expected_value_threshold=10, n_top_combos=3):
        """
        1着、2着、3着の確率と各組み合わせのオッズを考慮し、期待値の高い順に上位n個の三連単組み合わせを返す
        expected_value_threshold: この閾値未満の期待値の組み合わせは除外する
        n_top_combos: 返す上位の組み合わせの数
        """
        combo_expected_values = []  # (組み合わせ, 期待値) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（期待値順）"

        try:
            # 1着候補を選定 (確率>0のもの)
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0]

                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        # 組み合わせの総合確率を計算 (元のコードの `current_score` に相当)
                        current_probability = (P_1st[first - 1] *
                                               P_2nd[first][second - 1] *
                                               P_3rd[(first, second)][third - 1])
                    for third in third_candidates:
                        # 組み合わせの総合確率を計算
                        current_score = (P_1st[first-1] * P_2nd[first][second-1] * P_3rd[(first, second)][third-1])
                        
                        # オッズを取得
                        combo_str = f"{first}-{second}-{third}"
                        # ミスでスペース入れないとだめ
                        odds = all_odds_for_race.get(f" {combo_str}", 0) # オッズがない場合は0とする

                        # 期待値を計算: 総合確率 * オッズ
                        current_expected_value = current_probability * float(odds)

                        if current_expected_value >= expected_value_threshold:
                            combo_expected_values.append((combo_str, current_expected_value))

            # 期待値の高い順にソート
            combo_expected_values.sort(key=lambda x: x[1], reverse=True)

            # 上位n個の組み合わせを取得
            top_combos = [combo for combo, ev in combo_expected_values[:n_top_combos]]

            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_expected_value: {str(e)}")
            return [], "エラー発生"

    def _create_formation_expected_value2(self, P_1st, P_2nd, P_3rd, all_odds_for_race, probability_threshold=0.01, n_top_combos=3):
        """
        1着、2着、3着の確率と各組み合わせのオッズを考慮し、確率の高い順に上位n個の三連単組み合わせを返す
        probability_threshold: この閾値未満の総合確率の組み合わせは除外する
        n_top_combos: 返す上位の組み合わせの数
        """
        combo_expected_values = []  # (組み合わせ, 期待値) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（確率順、期待値計算）" # 戦術名を更新

        try:
            # 1着候補を選定 (確率>0のもの)
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0]

                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        # 組み合わせの総合確率を計算
                        current_probability = (P_1st[first - 1] *
                                               P_2nd[first][second - 1] *
                                               P_3rd[(first, second)][third - 1])
                        
                        # ここで確率の閾値でフィルター
                        if current_probability >= probability_threshold:
                            # オッズを取得
                            combo_str = f"{first}-{second}-{third}"
                            # 列名に空白はないはずなので、そのままアクセス
                            odds = all_odds_for_race.get(f" {combo_str}", 0) 

                            # 期待値を計算: 総合確率 * オッズ
                            # オッズが数値型であることを確認するために float() でキャスト
                            current_expected_value = current_probability * float(odds)
                            
                            # 期待値ではなく、ここでは「確率」に基づいてフィルター
                            # ここは、期待値計算結果をリストに入れる部分なので、期待値を使う
                            combo_expected_values.append((combo_str, current_expected_value))

            # 期待値の高い順にソート (上位n個を取得するためには期待値でソートが必要)
            combo_expected_values.sort(key=lambda x: x[1], reverse=True)

            # 上位n個の組み合わせを取得
            top_combos = [combo for combo, ev in combo_expected_values[:n_top_combos]]

            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_expected_value: {str(e)}")
            return [], "エラー発生"

    def _create_formation_biased_odds_strategy1(self, P_1st, P_2nd, P_3rd, all_odds_for_race, expected_value_threshold=0.3, n_top_combos=1):
        """
        1着、2着、3着の確率と各組み合わせのオッズを考慮し、
        「モデル確率が市場確率より高く、かつ期待値が閾値を超える」組み合わせを、
        期待値の高い順に上位n個の三連単組み合わせを返す戦略。
        
        expected_value_threshold: この閾値未満の期待値の組み合わせは除外する（過小評価されていると判断しても、期待値が低すぎる場合は除外）
        n_top_combos: 返す上位の組み合わせの数
        """
        combo_results = []  # (組み合わせ, 期待値, モデル確率, 市場確率) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（市場歪み狙い）"

        try:
            # 1着候補を選定 (確率>0のもの)
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                # 2着候補を選定 (1着艇を除く)
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0]

                for second in second_candidates:
                    # 3着候補を選定 (1着艇と2着艇を除く)
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        # 組み合わせ文字列
                        combo_str = f"{first}-{second}-{third}"

                        # オッズを取得
                        odds = all_odds_for_race.get(f" {combo_str}", 0) # オッズがない場合は0とする

                        # オッズが0や非常に小さい場合（計算不能な場合）はスキップ
                        if float(odds) <= 1.0: # オッズが1以下の場合は期待値が計算できないためスキップ
                            continue

                        # 組み合わせの総合確率（モデル確率）を計算
                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])

                        # 市場の確率 (オッズの逆数。控除率は考慮せず単純に1/Odds)
                        market_probability = 1.0 / float(odds)

                        # 期待値を計算: モデル確率 * オッズ
                        current_expected_value = model_probability * float(odds)

                        # 「過小評価されている」条件と「期待値閾値」を満たすかチェック
                        # モデル確率が市場確率より高い (市場が過小評価している)
                        # かつ、その期待値が設定した閾値以上である
                        if model_probability > market_probability and current_expected_value >= expected_value_threshold:
                            combo_results.append((combo_str, current_expected_value, model_probability, market_probability))

            # 期待値の高い順にソート（この戦略では期待値でソート）
            combo_results.sort(key=lambda x: x[1], reverse=True)

            # 上位n個の組み合わせを取得
            top_combos = [combo for combo, ev, mp, mkp in combo_results[:n_top_combos]]

            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_biased_odds_strategy: {str(e)}")
            return [], "エラー発生"
        
    def _create_formation_biased_odds_strategy2(self, P_1st, P_2nd, P_3rd, all_odds_for_race, expected_value_threshold=1.0, n_top_combos=3, min_prob_advantage=0.003, min_odds=10.0):
        # min_prob_advantage: モデル確率が市場確率より優位であると判断する最小の差（例: 0.01 = 1%の差）
        # min_odds: オッズの最低値（低オッズの組み合わせは除外）
        
        combo_results = []  # (組み合わせ, 期待値, モデル確率, 市場確率) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（市場歪み狙い、差{min_prob_advantage*100:.1f}%以上）"

        try:
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]

            for first in first_candidates:
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]

                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0] 

                for second in second_candidates:
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]

                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]

                    for third in third_candidates:
                        combo_str = f"{first}-{second}-{third}"

                        odds = all_odds_for_race.get(f" {combo_str}", 0) 
                        
                        # オッズが有効な数値で、かつ最低オッズ以上であることを確認
                        if not isinstance(float(odds), (int, float)) or float(odds) <= min_odds:
                            continue

                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])

                        # 市場の確率 (控除率を考慮しない単純な逆数)
                        market_probability = 1.0 / float(odds)

                        current_expected_value = model_probability * float(odds)

                        # 新しい条件: モデル確率が市場確率より指定した差以上高い
                        if (model_probability - market_probability >= min_prob_advantage and
                            current_expected_value >= expected_value_threshold): # かつ期待値も閾値以上
                            
                            combo_results.append((combo_str, current_expected_value, model_probability, market_probability))

            combo_results.sort(key=lambda x: x[1], reverse=True) # 期待値の高い順

            top_combos = [combo for combo, ev, mp, mkp in combo_results[:n_top_combos]]

            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_biased_odds_strategy: {str(e)}")
            return [], "エラー発生"

    def _create_formation_adjusted_expected_value(self, P_1st, P_2nd, P_3rd, all_odds_for_race, 
                                                expected_value_threshold=0.5, n_top_combos=3, 
                                                alpha=0.3, min_odds=1.0):
        """
        調整済み期待値 (期待値 * (モデル確率^alpha)) に基づいて組み合わせをソート・選択する戦略。
        alpha: 調整係数 (0 < alpha < 1)。alphaが小さいほど確率の高い組み合わせを重視。
        """
        combo_results = []  # (組み合わせ, 調整済み期待値, モデル確率, 期待値) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（調整期待値 α={alpha:.1f}）"

        try:
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]
            for first in first_candidates:
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]
                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0]
                for second in second_candidates:
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]
                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]
                    for third in third_candidates:
                        combo_str = f"{first}-{second}-{third}"
                        odds = all_odds_for_race.get(f" {combo_str}", 0)

                        if not isinstance(float(odds), (int, float)) or float(odds) <= min_odds:
                            continue
                        if P_1st[first - 1] == 0 or P_2nd[first][second - 1] == 0 or P_3rd[(first, second)][third - 1] == 0:
                            continue

                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])
                        
                        # モデル確率が0の場合は、log(0)などでエラーになる可能性があるのでスキップ
                        if model_probability <= 0:
                            continue

                        current_expected_value = model_probability * float(odds)
                        
                        # 調整済み期待値を計算
                        # model_probabilityが非常に小さい場合、model_probability**alphaが丸め誤差で0になる可能性も考慮
                        adjusted_expected_value = current_expected_value * (model_probability ** alpha)

                        if adjusted_expected_value >= expected_value_threshold:
                            combo_results.append((combo_str, adjusted_expected_value, model_probability, current_expected_value))

            combo_results.sort(key=lambda x: x[1], reverse=True) # 調整済み期待値の高い順にソート

            top_combos_with_ev = combo_results[:n_top_combos]
            top_combos = [combo for combo, adj_ev, mp, ev in top_combos_with_ev] # 実際の組み合わせ文字列

            if top_combos:
                return top_combos, strategy_name#, top_combos_with_ev
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_adjusted_expected_value: {str(e)}")
            return [], "エラー発生"

    def _create_formation_tiered_filter(self, P_1st, P_2nd, P_3rd, all_odds_for_race, 
                                        probability_filter_threshold=0.05, # 確率フィルターの閾値
                                        n_top_combos=3, 
                                        expected_value_sort_threshold=1, # 抽出後の期待値ソート時の閾値
                                        min_odds=10.0):
        """
        まず確率でフィルターし、その後期待値の高い順にソート・選択する戦略。
        probability_filter_threshold: 総合確率の最低閾値。
        expected_value_sort_threshold: 確率フィルターを通過した後に、期待値でさらにフィルターする閾値。
        """
        filtered_combos_by_prob = [] # 確率フィルターを通過した組み合わせ
        strategy_name = f"上位{n_top_combos}組み合わせ（段階フィルター 確率>{probability_filter_threshold*100:.1f}%, 期待値>{expected_value_sort_threshold*100:.1f}%）"

        try:
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]
            for first in first_candidates:
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]
                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0]
                for second in second_candidates:
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]
                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]
                    for third in third_candidates:
                        combo_str = f"{first}-{second}-{third}"
                        odds = all_odds_for_race.get(f" {combo_str}", 0)

                        if not isinstance(float(odds), (int, float)) or float(odds) <= min_odds:
                            continue
                        if P_1st[first - 1] == 0 or P_2nd[first][second - 1] == 0 or P_3rd[(first, second)][third - 1] == 0:
                            continue

                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])
                        
                        current_expected_value = model_probability * float(odds)

                        # 段階1: 確率フィルター
                        if model_probability >= probability_filter_threshold:
                            # 段階2: 期待値フィルター（期待値閾値も超えるもののみ）
                            if current_expected_value >= expected_value_sort_threshold:
                                filtered_combos_by_prob.append((combo_str, current_expected_value, model_probability))

            filtered_combos_by_prob.sort(key=lambda x: x[1], reverse=True) # 期待値の高い順にソート

            top_combos_with_ev = filtered_combos_by_prob[:n_top_combos]
            top_combos = [combo for combo, ev, mp in top_combos_with_ev]

            if top_combos:
                return top_combos, strategy_name#, top_combos_with_ev
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_tiered_filter: {str(e)}")
            return [], "エラー発生"
    
    def _create_formation_medium_odds_strategy(self, P_1st, P_2nd, P_3rd, all_odds_for_race, 
                                                min_odds=10.0, max_odds=100.0, # 20倍〜50倍のオッズ範囲
                                                expected_value_threshold=0.8, # オッズ範囲内で高期待値
                                                n_top_combos=2,               # ベット点数
                                                min_prob_advantage=0.01,      # モデル確率の優位性 (市場歪み)
                                                deduction_rate=0.25):
        """
        特定のオッズ範囲 (中穴) に絞り込み、モデル優位性があり、期待値が閾値を超える組み合わせを狙う戦略。
        min_odds, max_odds: 対象とするオッズの下限と上限。
        """
        combo_results = []  # (組み合わせ, 期待値, モデル確率, 市場確率) を保存するリスト
        strategy_name = f"上位{n_top_combos}組み合わせ（中穴狙い {min_odds:.1f}～{max_odds:.1f}倍）"

        try:
            first_candidates = [i for i in range(1, 7) if P_1st[i - 1] > 0]
            for first in first_candidates:
                second_probs = P_2nd.get(first, [])
                valid_second_probs = [p for p in second_probs if isinstance(p, (int, float))]
                second_candidates = [j for j in range(1, 7)
                                     if j != first and
                                     len(valid_second_probs) > j - 1 and
                                     valid_second_probs[j - 1] > 0] 
                for second in second_candidates:
                    third_probs = P_3rd.get((first, second), [])
                    valid_third_probs = [p for p in third_probs if isinstance(p, (int, float))]
                    third_candidates = [k for k in range(1, 7)
                                        if k not in [first, second] and
                                        len(valid_third_probs) > k - 1 and
                                        valid_third_probs[k - 1] > 0]
                    for third in third_candidates:
                        combo_str = f"{first}-{second}-{third}"
                        odds = all_odds_for_race.get(f" {combo_str}", 0) 

                        # オッズが有効な数値で、かつ指定された中穴範囲内であることを確認
                        if not isinstance(float(odds), (int, float)) or float(odds) < min_odds or float(odds) > max_odds:
                            continue
                        
                        if P_1st[first - 1] == 0 or P_2nd[first][second - 1] == 0 or P_3rd[(first, second)][third - 1] == 0:
                            continue

                        model_probability = (P_1st[first - 1] *
                                             P_2nd[first][second - 1] *
                                             P_3rd[(first, second)][third - 1])
                        
                        # モデル確率が0の場合はスキップ
                        if model_probability <= 0:
                            continue

                        market_probability = (1.0 / float(odds)) * (1.0 - deduction_rate)
                        current_expected_value = model_probability * float(odds)

                        # 「過小評価されている」条件と「期待値閾値」を満たすかチェック
                        if (model_probability - market_probability >= min_prob_advantage and
                            current_expected_value >= expected_value_threshold):
                            
                            combo_results.append((combo_str, current_expected_value, model_probability, market_probability))

            combo_results.sort(key=lambda x: x[1], reverse=True) # 期待値の高い順にソート

            top_combos_with_ev = combo_results[:n_top_combos]
            top_combos = [combo for combo, ev, mp, mkp in top_combos_with_ev]

            if top_combos:
                return top_combos, strategy_name
            else:
                return [], "有効な組み合わせなし"

        except Exception as e:
            print(f"Error in _create_formation_medium_odds_strategy: {str(e)}")
            return [], "エラー発生"