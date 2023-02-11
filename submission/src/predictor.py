import pandas as pd
import numpy as np
import json
import joblib


class ScoringService(object):
    # ラベルエンコーディングの辞書をロード
    # 学習時と予測時でインデックスを揃えるために共通の辞書を使う
    # 観測所
    staion_json = open("./water_station_dict.json", "r")
    WATER_STATION_DICT = json.load(staion_json)
    WATER_STATION_DICT_REVERSE = {v: k for k, v in WATER_STATION_DICT.items()}
    # 河川
    river_json = open("./river_dict.json", "r")
    RIVER_DICT = json.load(river_json)
    # 水系
    system_json = open("./system_dict.json", "r")
    SYSTEM_DICT = json.load(system_json)
    # 緯度・経度
    loc_json = open("./loc_dict.json", "r")
    LOC_DICT = json.load(loc_json)

    @classmethod
    def get_model(cls, model_path):
        # 単数モデル
        # with open(model_path + '/model.pkl', mode='rb') as f:
        #     cls.model = joblib.load(f)
        # 複数モデル
        with open(model_path + "/periodic_model.pkl", mode="rb") as f:
            cls.periodic_model = joblib.load(f)
        with open(model_path + "/non_periodic_model.pkl", mode="rb") as f:
            cls.non_periodic_model = joblib.load(f)
        return True

    @classmethod
    def predict(cls, input):
        # 入力データを取得
        # 日付
        today = input["date"]
        # 観測所
        stations = input["stations"]
        # 水位
        waterlevel = input["waterlevel"]
        water_data = pd.DataFrame(waterlevel)
        water_data["date"] = today
        # 雨量
        rainfall = input["rainfall"]
        rainfall_data = pd.DataFrame(rainfall)
        rainfall_data["date"] = today

        # 水位の前処理
        # 観測対象の station を指定
        water_data = water_data[water_data["station"].isin(stations)]
        # date が先頭に来るようにする
        water_data = water_data.reindex(
            columns=["date", "station", "river", "hour", "value"]
        )
        water_data = cls.water_data_preprocess(water_data)

        # 雨量の前処理
        rainfall_data["value"] = pd.to_numeric(
            rainfall_data["value"], errors="coerce")
        rainfall_data["sum"] = rainfall_data["value"].sum()

        # データセットの作成
        dataset = cls.make_dataset(input, water_data, rainfall_data)
        # 翌日のぶんの DataFrame のみ使う
        dataset = dataset[dataset["date"] > today]

        # 予測
        # 特徴量の選定
        features = [
            "station_id",
            "river_id",
            "system_id",
            "latitude",
            "longitude",
            "hour",
            "rainfall_sum",
            "value_last23",
        ]
        # 水位が周期的な観測所とそうでない観測所に分割
        # WATER_STATION_DICT に依存
        periodic_list = [6, 7, 37, 113, 114, 129, 130, 131]
        dataset_periodic = dataset[dataset["station_id"].isin(
            periodic_list)].copy()
        dataset_non_periodic = dataset[
            ~dataset["station_id"].isin(periodic_list)
        ].copy()

        # 周期的な観測所
        if dataset_periodic.size > 0:
            X = dataset_periodic[features]
            y = cls.periodic_model.predict(X)
            dataset_periodic["value"] = y

        # 非周期的な観測所
        if dataset_non_periodic.size > 0:
            X = dataset_non_periodic[features]
            y = cls.non_periodic_model.predict(X)
            dataset_non_periodic["value"] = y

        # 連結
        df_submission = pd.concat([dataset_periodic, dataset_non_periodic])

        # 単数モデルによる予測
        # df_submission = dataset.copy()
        # X = dataset[features]
        # ypred = cls.model.predict(X)
        # df_submission['value'] = ypred

        # station のデコード
        groups = df_submission.groupby("station_id")
        df_submisison_new = pd.DataFrame()
        for id, group in groups:
            df_tmp = pd.DataFrame(
                columns=["date", "station", "river", "hour", "value"])
            df_tmp[["date", "hour", "value"]
                   ] = group[["date", "hour", "value"]]
            df_tmp["station"] = cls.WATER_STATION_DICT_REVERSE[id]
            df_submisison_new = pd.concat([df_submisison_new, df_tmp])

        # 出力ファイルを作成
        submission_list = []
        for row in df_submisison_new.itertuples():
            submission_list.append(
                {"hour": row.hour, "station": row.station, "value": row.value}
            )

        return submission_list

    @classmethod
    def water_data_preprocess(cls, water_data):
        # 特徴量の追加
        # 観測所ごとに追加して連結する
        groups = water_data.groupby("station")
        df_new = pd.DataFrame()
        for name, group in groups:
            df_tmp = group.copy()
            # 観測所名称
            df_tmp["station_id"] = cls.WATER_STATION_DICT[name]
            # 河川名
            df_tmp["river_id"] = cls.RIVER_DICT[name]
            # 水系名
            df_tmp["system_id"] = cls.SYSTEM_DICT[name]
            # 緯度
            df_tmp["latitude"] = cls.LOC_DICT[name][0]
            # 経度
            df_tmp["longitude"] = cls.LOC_DICT[name][1]
            df_new = pd.concat([df_new, df_tmp])
        water_data_new = df_new

        # 欠損値をすべて float 型に変換
        water_data_new["value"] = pd.to_numeric(
            water_data_new["value"], errors="coerce"
        )

        # 欠損値を時系列順 に埋める
        # station, date, hour の順にソート
        water_data_new.sort_values(
            ["station_id", "date", "hour"], inplace=True)
        groups = water_data_new.groupby("station_id")
        df_new = pd.DataFrame()
        for _, group in groups:
            df_tmp = group.copy()
            # 時間に対して後ろ向きに埋めた後、前向きに埋める
            df_tmp["value"].fillna(method="bfill", inplace=True)
            df_tmp["value"].fillna(method="ffill", inplace=True)
            df_new = pd.concat([df_new, df_tmp])
        water_data_new = df_new

        return water_data_new

    @classmethod
    def make_dataset(cls, input, water_data, rainfall_data):
        # 水位データを基にデータセットを作る
        dataset = water_data
        # 翌日のデータを表す DataFrame
        df_next = dataset.copy()
        # 翌日の value は nan にしておく
        df_next["value"] = float("nan")
        # 翌日なので日付を一日進める
        df_next["date"] = input["date"] + 1
        # 当日と翌日の DataFrame を連結
        dataset = pd.concat([dataset.copy(), df_next.copy()]
                            ).reset_index(drop=True)
        # インデックスを整える｀
        dataset = dataset[
            [
                "date",
                "hour",
                "station",
                "station_id",
                "river",
                "river_id",
                "system_id",
                "latitude",
                "longitude",
                "value",
            ]
        ]
        # 前日県内で降った雨の総量を追加
        dataset["rainfall_sum"] = rainfall_data["sum"]
        # 前日の23時の水位と前日を特徴量として追加
        # 日付が欠損していないのを前提としている
        # データセットをソートしておく
        dataset.sort_values(["station_id", "date", "hour"], inplace=True)
        # 23時，0時のデータのみ取り出す
        df_tmp1 = dataset[dataset["hour"] == 23]
        df_tmp2 = dataset[dataset["hour"] == 0]
        # それぞれの station，date について前日23時，0時のデータを24時間分リピート
        df_last23 = np.repeat(df_tmp1["value"].values, 24, axis=0)
        df_last0 = np.repeat(df_tmp2["value"].values, 24, axis=0)
        # 差分を取る
        diff = df_last0 - df_last23
        # 1日分ずらして挿入
        dataset["value_last23"] = float("nan")
        dataset.iloc[24:, -1] = df_last23[:-24]
        dataset["value_diff"] = float("nan")
        dataset.iloc[24:, -1] = diff[:-24]
        # 初日は他の station のデータが入っているのでクリア
        row_indexer = dataset[dataset["date"] == input["date"]].index
        dataset.loc[row_indexer, "value_last23"] = float("nan")
        dataset.loc[row_indexer, "value_diff"] = float("nan")

        return dataset
