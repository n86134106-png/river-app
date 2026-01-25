import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, GroupKFold
import sys

# -------------------------------------------------------
# 解決 Matplotlib 中文亂碼問題
# -------------------------------------------------------
if sys.platform.startswith('darwin'):  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
elif sys.platform.startswith('win'):  # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False
# -------------------------------------------------------

# -------------------------
# 1. 基礎設定與參數
# -------------------------
st.set_page_config(page_title="無測站流量推估系統", page_icon="logo.png", layout="wide")

# 欄位名稱設定 (需與 Excel 一致)
COL_REGION = "region"  # 必須要有這個欄位才能分區
COL_STA_ID = "station_id"
COL_STA_NAME = "station_name"
COL_AREA = "area_km2"
COL_RAIN = "rain_mean"
COL_PCT = "percentile"
COL_Y = "obs_Q"

X_COLS = [COL_AREA, COL_RAIN]
PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# SVM 參數搜尋網格
PARAM_GRID = {
    "svr__C": [1, 10, 100, 300],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__gamma": ["scale", 0.1, 0.05, 0.01],
}


# -------------------------
# 2. 核心函數
# -------------------------

def _make_estimator():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])


def _safe_log1p(y):
    return np.log1p(np.clip(y, a_min=0, a_max=None))


def _inv_log1p(ylog):
    y = np.expm1(ylog)
    return np.clip(y, a_min=0, a_max=None)


@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file, engine="openpyxl")


@st.cache_resource
def train_models(df, target_region="不分區 (全流域)", do_grid_search=True, exclude_name="無 (全量訓練)"):
    """
    訓練模型
    target_region: 指定要使用的資料區域 (不分區 或 第一區、第二區...)
    exclude_name: 指定要剔除的測站
    """
    models = {}
    best_params_record = {}

    # 1. 先進行區域篩選
    if target_region == "不分區 (全流域)":
        df_region = df.copy()
        region_msg = "全流域資料"
    else:
        # 只保留該區域的資料
        df_region = df[df[COL_REGION] == target_region].copy()
        region_msg = f"{target_region}資料"

    # 2. 再進行測站剔除 (模擬無測站)
    if exclude_name != "無 (全量訓練)":
        train_df = df_region[df_region[COL_STA_NAME] != exclude_name].copy()
        msg = f"正在訓練 ({region_msg})... 已剔除: {exclude_name}"
    else:
        train_df = df_region.copy()
        msg = f"正在訓練 ({region_msg})... 全量模式"

    # 如果篩選完沒資料了 (防呆)
    if len(train_df) == 0:
        return {}, {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(msg)

    total_steps = len(PCTS)

    for idx, pct in enumerate(PCTS):
        d = train_df[train_df[COL_PCT] == pct].copy()
        if len(d) == 0:
            continue

        X = d[X_COLS].to_numpy()
        y0 = d[COL_Y].to_numpy()
        y = _safe_log1p(y0)

        est = _make_estimator()

        groups = d[COL_STA_ID].to_numpy()
        n_groups = len(np.unique(groups))

        # 只有當樣本數夠多時才做 Grid Search
        if do_grid_search and n_groups >= 3:
            cv_splits = min(5, n_groups)
            gs = GridSearchCV(
                est,
                param_grid=PARAM_GRID,
                cv=GroupKFold(n_splits=cv_splits),
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )
            gs.fit(X, y, groups=groups)
            final_model = gs.best_estimator_
            best_params_record[pct] = gs.best_params_
        else:
            est.set_params(svr__C=100, svr__gamma=0.1, svr__epsilon=0.1)
            est.fit(X, y)
            final_model = est
            best_params_record[pct] = "Fixed/Default"

        models[pct] = final_model
        progress_bar.progress((idx + 1) / total_steps)

    status_text.text(f"訓練完成！範圍：{target_region} | 模式：{exclude_name}")
    progress_bar.empty()
    return models, best_params_record


# -------------------------
# 3. Streamlit 介面佈局
# -------------------------

st.title("🌊 臺灣南部無測站流域流量推估系統")
st.markdown("""
本系統展示 **支撐向量機 (SVR)** 於水文區域化之應用。
透過左側設定，可選擇 **「推估區域範圍」** 並切換 **「全量應用/模擬驗證」** 模式。
""")

# --- [左側側邊欄] ---
st.sidebar.header("1. 資料來源")
uploaded_file = st.sidebar.file_uploader("上傳訓練資料 (Excel)", type=["xlsx"])
DEFAULT_PATH = Path("data.xlsx")

df = None
if uploaded_file:
    df = load_data(uploaded_file)
elif DEFAULT_PATH.exists():
    df = load_data(DEFAULT_PATH)
else:
    st.error("請上傳資料檔")

if df is not None:
    st.sidebar.success(f"資料載入成功 (共 {len(df)} 筆)")
    st.sidebar.divider()
    st.sidebar.header("2. 模式設定")

    do_grid_search = st.sidebar.checkbox("啟用最佳參數搜尋 (較慢)", value=False)

    # -------------------------------------------------------
    # [功能] 區域選擇 (Region Selection) - 強制排序
    # -------------------------------------------------------
    if COL_REGION in df.columns:
        unique_regions = list(df[COL_REGION].dropna().unique())
        desired_order = ["第一區", "第二區", "第三區"]
        available_regions = sorted(
            unique_regions,
            key=lambda x: desired_order.index(x) if x in desired_order else 999
        )
    else:
        available_regions = []

    region_options = ["不分區 (全流域)"] + available_regions

    target_region = st.sidebar.selectbox(
        "選擇推估區域範圍",
        options=region_options,
        index=0,
        help="選擇「不分區」將使用所有資料建立 Global Model；選擇特定區域將建立 Local Model。"
    )

    # -------------------------------------------------------
    # [新增功能] 顯示分區地圖
    # -------------------------------------------------------
    with st.sidebar.expander("🗺️ 查看流域分區地圖"):
        # 這裡設定圖片檔名為 map.png
        # 請確保您已經將圖片存為 map.png 並放在程式同一資料夾
        map_path = Path("map.png")
        if map_path.exists():
            st.image(str(map_path), caption="南部流域分區示意圖", use_container_width=True)
        else:
            st.info("請將圖片命名為 map.png 並放入資料夾中。")
    # -------------------------------------------------------

    # -------------------------------------------------------
    # [連動功能] 測站選擇 (根據區域過濾)
    # -------------------------------------------------------
    if target_region == "不分區 (全流域)":
        filtered_stations = sorted(list(df[COL_STA_NAME].unique()))
    else:
        filtered_stations = sorted(list(df[df[COL_REGION] == target_region][COL_STA_NAME].unique()))

    station_options = ["無 (全量訓練)"] + filtered_stations

    exclude_station = st.sidebar.selectbox(
        "模擬無測站情境 (剔除指定測站)",
        options=station_options,
        index=0,
        help="選擇一個測站將其從訓練資料中移除。注意：若上方選擇了特定區域，此處只會顯示該區域內的測站。"
    )

    # -------------------------------------------------------
    # 開始訓練與推估
    # -------------------------------------------------------
    try:
        models, params_record = train_models(df, target_region, do_grid_search, exclude_station)

        # --- [右側主畫面] ---
        default_area = 100.0
        default_rain = 2500.0
        actual_values = None

        if exclude_station != "無 (全量訓練)":
            st.info(f"🔍 目前模式：**{target_region}** 模型 | 已剔除 **{exclude_station}**")

            target_station_rows = df[df[COL_STA_NAME] == exclude_station]
            if not target_station_rows.empty:
                target_station_data = target_station_rows.iloc[0]
                default_area = float(target_station_data[COL_AREA])
                default_rain = float(target_station_data[COL_RAIN])

                subset = target_station_rows.set_index(COL_PCT)
                actual_values = subset[COL_Y].to_dict()
        else:
            st.success(f"✅ 目前模式：**{target_region}** 模型 | 全量訓練 (可用於預測新地點)")

        st.subheader("參數輸入")
        col_in1, col_in2, col_btn = st.columns([2, 2, 1])
        with col_in1:
            input_area = st.number_input("流域集水面積 (km²)", min_value=0.1, value=default_area, step=10.0)
        with col_in2:
            input_rain = st.number_input("流域年均雨量 (mm)", min_value=0.0, value=default_rain, step=100.0)
        with col_btn:
            st.write("")
            st.write("")
            calc_clicked = st.button("開始推估", type="primary", use_container_width=True)

        if calc_clicked:
            if not models:
                st.warning("⚠️ 無法建立模型，可能是該區域資料不足。")
            else:
                results = []
                input_data = np.array([[input_area, input_rain]])

                temp_preds = []
                valid_pcts = []
                for pct in PCTS:
                    model = models.get(pct)
                    if model:
                        pred_log = model.predict(input_data)[0]
                        pred_val = _inv_log1p(pred_log)
                        temp_preds.append(pred_val)
                        valid_pcts.append(pct)

                if len(temp_preds) > 0:
                    temp_preds = np.sort(temp_preds)[::-1]

                for i, pct in enumerate(valid_pcts):
                    final_val = temp_preds[i]
                    row = {"Percentile": pct, "Predicted Flow (cms)": final_val}

                    if actual_values and pct in actual_values:
                        row["Observed Flow (cms)"] = actual_values[pct]
                        row["AE (cms)"] = np.abs(final_val - actual_values[pct])

                    results.append(row)

                res_df = pd.DataFrame(results)

                st.divider()
                col_chart, col_table = st.columns([1.5, 1])

                with col_chart:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(res_df["Percentile"], res_df["Predicted Flow (cms)"],
                            marker='o', linestyle='-', color='blue', label=f'Predicted ({target_region})', linewidth=2)

                    if "Observed Flow (cms)" in res_df.columns:
                        ax.plot(res_df["Percentile"], res_df["Observed Flow (cms)"],
                                marker='x', linestyle='--', color='red', label='Observed (Actual)', alpha=0.7)
                        ax.fill_between(res_df["Percentile"],
                                        res_df["Predicted Flow (cms)"],
                                        res_df["Observed Flow (cms)"],
                                        color='gray', alpha=0.1, label='Difference')

                    ax.set_title(f"Flow Duration Curve\nRegion: {target_region} | Target: {exclude_station}")
                    ax.set_xlabel("Exceedance Probability (%)")
                    ax.set_ylabel("Flow (cms)")
                    ax.legend()
                    ax.grid(True, linestyle=':', alpha=0.6)
                    st.pyplot(fig)

                with col_table:
                    st.subheader("詳細數據")
                    format_dict = {"Predicted Flow (cms)": "{:.2f}"}
                    if "Observed Flow (cms)" in res_df.columns:
                        format_dict["Observed Flow (cms)"] = "{:.2f}"
                        format_dict["AE (cms)"] = "{:.2f}"
                    st.dataframe(res_df.style.format(format_dict), hide_index=True, use_container_width=True)

                if "AE (cms)" in res_df.columns:
                    mean_ae = res_df["AE (cms)"].mean()
                    st.info(f"ℹ️ [{target_region}] 模擬驗證結果：平均絕對誤差 (MAE) 為 {mean_ae:.2f} cms。")

    except Exception as e:
        st.error(f"執行錯誤: {e}")
