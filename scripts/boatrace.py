import re
import pandas as pd

def parse_boatrace_txt(file_path):
    # Shift_JIS でファイルを開く
    with open(file_path, "r", encoding="shift_jis") as f:
        lines = f.readlines()
    
    race_data = []
    race_info = None
    race_place = None
    race_player = None
    current_race = []
    
    for line in lines:
        line = line.strip()
        if not line:  # 空行はスキップ
            continue
    
        # レース場情報を取得（ボートレースの後に3文字を固定取得）
        match_place = re.search(r"ボートレース(.{3})", line)
        if match_place:
            race_place = match_place.group(1)
        
        # レース情報の取得
        match_race = re.match(r"([０-９])Ｒ\s+(\S+)", line)
        if match_race:
            if current_race:
                race_data.extend(current_race)
                current_race = []
    
            race_number = match_race.group(1)
            race_type = match_race.group(2)
            race_info = (race_number, race_type, race_place)
    
        # 選手情報の取得
        match_player = re.match(r"^\d\s\d{4}", line)
        if match_player:
            parts = line.split(maxsplit=2)  # 先頭2つ（艇番, 登録番号）は固定なので分割
            
            boat_number = parts[0]  # 艇番
            raw_data = parts[1] + " " + parts[2]  # 登録番号以降のデータを結合して処理

            # 登録番号 (4桁固定)
            register_number = raw_data[:4].strip()
            # 選手名 (4文字固定)
            name = raw_data[4:8].strip()
            # 年齢 (2桁固定)
            age = raw_data[8:10].strip()
            # 支部 (2文字固定)
            region = raw_data[10:12].strip()
            # 体重 (2桁固定)
            weight = raw_data[12:14].strip()
            # 級別 (A1, A2, B1, B2)
            rank = raw_data[14:16].strip()
            
            # 残りのデータを適切に分割
            remaining_data = raw_data[16:].split()

            # None を空文字に変換
            remaining_data = [str(x) if x is not None else "" for x in remaining_data]

            win_rate_national = remaining_data[0]
            win_rate_national_2 = remaining_data[1]
            win_rate_local = remaining_data[2]
            win_rate_local_2 = remaining_data[3]
            motor_no = remaining_data[4]
            motor_rate = remaining_data[5]
            boat_no = remaining_data[6]
            boat_rate = remaining_data[7]

            current_race.append([
                race_info[0], race_info[1], race_info[2],
                boat_number, register_number, name, age, region, weight, rank,
                win_rate_national, win_rate_national_2, win_rate_local, win_rate_local_2,
                motor_no, motor_rate, boat_no, boat_rate
            ])
    
    # 最後のレースデータを追加
    if current_race:
        race_data.extend(current_race)
    
    # DataFrame に変換
    columns = [
        "レース番号", "レース種別", "レース場", 
        "艇番", "登録番号", "選手名", "年齢", "所属", "体重", "級別",
        "全国勝率", "全国2連対率", "当地勝率", "当地2連対率",
        "モーター番号", "モーター2連対率", "ボート番号", "ボート2連対率"
    ]

    df = pd.DataFrame(race_data, columns=columns)
    return df

def parse_boatrace_results(file_path):
    with open(file_path, "r", encoding="shift_jis") as f:
        lines = f.readlines()
    
    race_data = []
    race_info = None
    race_place = None
    race_player = None
    
    for line in lines:
        line = line.strip()
    
        # レース場の取得
        match_place = re.search(r"(.{1}　.{1})［", line)
        if match_place:
            race_place = match_place.group(1)
    
        # レース情報の取得
        match_race = re.match(r"([0-9]+R)\s+(\S+).*?H\d+m\s+(\S+)\s+風\s+(\S+)\s+(\d+m)\s+波\s+(\d+cm)", line)
        if match_race:
            race_info = {
                "レース場": race_place,
                "レース番号": match_race.group(1),
                "レース種別": match_race.group(2),
                "天候": match_race.group(3),
                "風向": match_race.group(4),
                "風速(m)": match_race.group(5),
                "波高(cm)": match_race.group(6),
                "決まり手": "",
            }
    
        # 決まり手の取得
        match_kimarite = re.search(r"ﾚｰｽﾀｲﾑ\s+(.+)", line)
        if match_kimarite and race_info:
            race_info["決まり手"] = match_kimarite.group(1).replace("　", "").strip()
    
        # 選手データの取得
        match_player = re.match(r"^\d{2}\s+\d\s+\d{4}", line)
        if match_player and race_info:
            race_player = {
                "着順": line[0:4].strip(),
                "艇番": line[4:6].strip(),
                "登録番号": line[6:11].strip(),
                "選手名": line[11:19].replace("　", "").strip(),
                "モーター": line[19:22].strip(),
                "ボート": line[22:27].strip(),
                "展示タイム": line[27:33].strip(),
                "進入": line[33:37].strip(),
                "スタートタイミング": line[37:45].strip(),
                "レースタイム": line[45:56].strip(),
            }
    
            # レース情報と結合して1行分のレースデータを作成
            full_data = {**race_info, **race_player}
            race_data.append(full_data)
    
    # DataFrame 変換
    columns = ["レース場", "レース番号", "レース種別", "天候", "風向", "風速(m)", "波高(cm)", "決まり手",
               "着順", "艇番", "登録番号", "選手名", "モーター", "ボート", "展示タイム", 
               "進入", "スタートタイミング", "レースタイム"]
    
    df = pd.DataFrame(race_data, columns=columns)
    return df
file_path = r"C:\Users\archi\boatrace\kaitou_B\B250201.TXT"  # テキストファイルのパス
df = parse_boatrace_txt(file_path)

# CSVとして保存（Shift_JISで書き出し）
csv_path = "boatrace_data_with_place1.csv"
df.to_csv(csv_path, index=False, encoding="shift_jis")

# データの最初の5行を表示
print("データの最初の5行:")
print(df.head())

print(f"\nCSVファイルが保存されました: {csv_path}")