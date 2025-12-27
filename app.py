import streamlit as st
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from prophet import Prophet
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import time
from datetime import timedelta, datetime
import warnings
import uuid
import os
import json
import pytz # THƯ VIỆN MỚI ĐỂ XỬ LÝ GIỜ VIỆT NAM

# --- CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="DỰ BÁO NHU CẦU SỬ DỤNG NƯỚC - MHS", layout="wide", page_icon="🌊")
warnings.filterwarnings('ignore')

# --- THIẾT LẬP MÚI GIỜ VIỆT NAM ---
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# --- BẢO MẬT  ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if not st.session_state.password_correct:
        st.markdown("## 🔐 Cổng Bảo Mật")
        password = st.text_input("Mời Hoàng thượng nhập mật mã:", type="password")
        if st.button("Xác nhận"):
            try:
                secret_pass = st.secrets["APP_PASSWORD"]
            except:
                secret_pass = "NMCN2960" 

            if password == secret_pass: 
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Mật mã sai rồi ạ!")
        return False
    return True

if check_password():
    SHEET_NAME = "Data_Nuoc_MinhHung"
    SHEET_TAB = "Dulieu_Tho"
    
    # --- LOGIC THỜI GIAN CẬP NHẬT (QUAN TRỌNG) ---
    # Lấy giờ hiện tại ở VN
    now_vn = datetime.now(VN_TZ)
    
    # Logic: Nếu hiện tại chưa qua 00:10 sáng, thì vẫn tính là dữ liệu của ngày hôm qua.
    # Nếu đã qua 00:10, thì tính là dữ liệu ngày hôm nay.
    # Biến data_version này sẽ dùng để làm "Key" cho bộ nhớ Cache. Khi Key thay đổi, Cache tự xóa và tải mới.
    if now_vn.hour == 0 and now_vn.minute < 10:
        data_version = (now_vn - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        data_version = now_vn.strftime('%Y-%m-%d')

    # --- HÀM 1: LẤY DỮ LIỆU (CÓ CACHE THÔNG MINH) ---
    # ttl=3600: Cache sống tối đa 1 tiếng phòng hờ.
    # Key thay đổi theo data_version sẽ ép buộc load lại đúng 00:10
    @st.cache_data(ttl=3600, show_spinner="Đang thỉnh dữ liệu từ Google Sheets về...")
    def get_data(version_key):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = None
            
            if os.path.exists("key.json"):
                creds = ServiceAccountCredentials.from_json_keyfile_name("key.json", scope)
            elif "gcp_service_account" in dict(st.secrets): 
                key_dict = dict(st.secrets["gcp_service_account"])
                if "private_key" in key_dict:
                      key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
                creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
            elif "GCP_SERVICE_ACCOUNT" in os.environ:
                key_dict = json.loads(os.environ['GCP_SERVICE_ACCOUNT'])
                creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)

            if creds is None:
                return None

            client = gspread.authorize(creds)
            sheet = client.open(SHEET_NAME).worksheet(SHEET_TAB)
            
            data = sheet.get_all_records()
            df = pd.DataFrame(data)
            
            df['FM_DATE'] = df['FM_DATE'].astype(str).str.split(':').str[0]
            df['Date'] = pd.to_datetime(df['FM_DATE'], format='%d/%m/%Y', errors='coerce')
            
            cols_map = {'FM0301.DAY': 'Flow_1', 'FMDN630.DAY': 'Flow_2'}
            available_cols = [c for c in cols_map.keys() if c in df.columns]
            for col in available_cols:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                
            df['Total_Flow'] = 0
            if 'FM0301.DAY' in df.columns: df['Total_Flow'] += df['FM0301.DAY']
            if 'FMDN630.DAY' in df.columns: df['Total_Flow'] += df['FMDN630.DAY']
            
            df = df[(df['Total_Flow'] > 0) & (df['Total_Flow'] < 200000)]
            result = df.groupby('Date')['Total_Flow'].sum().reset_index().sort_values('Date')
            return result
        except Exception as e:
            return str(e) # Trả về lỗi dạng string để xử lý bên ngoài

    # --- MODELING ---
    TRAIN_WINDOW = 365 

    def run_prophet(df, days=7):
        try:
            df_train = df.tail(TRAIN_WINDOW).copy()
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.5)
            m.add_country_holidays(country_name='VN') 
            m.fit(df_train.rename(columns={'Date': 'ds', 'Total_Flow': 'y'}))
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            if 'yhat' in forecast.columns:
                return forecast[['ds', 'yhat']].tail(days).rename(columns={'ds': 'Date', 'yhat': 'Prophet'})
            else: return pd.DataFrame()
        except: return pd.DataFrame()

    def run_arima(df, days=7):
        try:
            df_train = df.tail(60).copy()
            model = ARIMA(df_train['Total_Flow'].values, order=(5,1,0)).fit()
            preds = model.forecast(steps=days)
            dates = [df['Date'].iloc[-1] + timedelta(days=i+1) for i in range(days)]
            return pd.DataFrame({'Date': dates, 'ARIMA': preds})
        except: return pd.DataFrame()

    def run_xgboost(df, days=7):
        try:
            df_ml = df.tail(90).copy()
            for i in range(1, 4): df_ml[f'Lag{i}'] = df_ml['Total_Flow'].shift(i)
            df_ml = df_ml.dropna()
            if len(df_ml) < 5: return pd.DataFrame()

            model = XGBRegressor(n_estimators=200, learning_rate=0.2, max_depth=5).fit(df_ml[['Lag1', 'Lag2', 'Lag3']], df_ml['Total_Flow'])
            
            curr_lags = list(df_ml.iloc[-1][['Total_Flow', 'Lag1', 'Lag2']])
            preds = []
            curr_date = df['Date'].iloc[-1]
            for _ in range(days):
                curr_date += timedelta(days=1)
                p = model.predict(pd.DataFrame([curr_lags], columns=['Lag1', 'Lag2', 'Lag3']))[0]
                preds.append({'Date': curr_date, 'XGBoost': p})
                curr_lags = [p] + curr_lags[:2]
            return pd.DataFrame(preds)
        except: return pd.DataFrame()

    def run_linear(df, days=7):
        try:
            df_train = df.tail(60).copy()
            df_train['D'] = df_train['Date'].map(datetime.toordinal)
            model = LinearRegression().fit(df_train[['D']], df_train['Total_Flow'])
            dates = [df['Date'].iloc[-1] + timedelta(days=i+1) for i in range(days)]
            preds = model.predict([[d.toordinal()] for d in dates])
            return pd.DataFrame({'Date': dates, 'LinearReg': preds})
        except: return pd.DataFrame()

    # --- TÍNH TOÁN CORE ---
    def calculate_ratio_correction(df, days_check=7):
        if len(df) < days_check + 5: return 1.0, 1.0, 1.0, 1.0, 0.0
        
        train = df.iloc[:-days_check]
        actual = df.iloc[-days_check:]['Total_Flow'].values
        
        try:
            p = run_prophet(train, days_check)['Prophet'].values
            a = run_arima(train, days_check)['ARIMA'].values
            x = run_xgboost(train, days_check)['XGBoost'].values
            l = run_linear(train, days_check)['LinearReg'].values
        except: return 1.0, 1.0, 1.0, 1.0, 0.0

        def get_ratio(act, pred):
            if len(pred) != len(act) or pred.sum() == 0: return 1.0
            return act.sum() / (pred.sum() + 1e-9)

        r_p = get_ratio(actual, p)
        r_a = get_ratio(actual, a)
        r_x = get_ratio(actual, x)
        r_l = get_ratio(actual, l)
        
        best_mae = 500
        try:
            maes = []
            for pred in [p, a, x, l]:
                if len(pred) == len(actual): maes.append(mean_absolute_error(actual, pred))
            if maes: best_mae = min(maes)
        except: pass
        
        def clip(r): return max(0.5, min(r, 2.0))
        return clip(r_p), clip(r_a), clip(r_x), clip(r_l), best_mae

    # --- GIAO DIỆN CHÍNH ---
    # Truyền data_version vào để cache biết khi nào cần thay mới
    df = get_data(version_key=data_version)
    
    # Kiểm tra xem df có phải là lỗi (string) hay không
    if isinstance(df, str):
        if "secrets" in df:
            st.error("Lỗi cấu hình: Đang chạy local nhưng thiếu file 'key.json'.")
        else:
            st.error(f"Lỗi đọc dữ liệu: {df}")
        df = None # Reset về None để không chạy code bên dưới

    if df is not None:
        # UI Makeover Style
        st.markdown("""<style>div[data-testid="stMetricValue"] { font-size: 24px; }</style>""", unsafe_allow_html=True)
        
        c_title, c_clock = st.columns([3, 1])
        with c_title:
            st.title("🌊 DỰ BÁO NHU CẦU NƯỚC SỬ DỤNG NƯỚC - MHS")
        with c_clock:
            st.caption(f"🕒 Giờ VN: {now_vn.strftime('%H:%M %d/%m')}")
            st.caption(f"🔄 Cập nhật lần tới: 00:10")

        # Sidebar
        st.sidebar.header("⚙️ Cấu hình hiển thị")
        show_optimized = st.sidebar.checkbox("✅ DỰ BÁO TỐI ƯU", value=True)
        show_range = st.sidebar.checkbox("🚧 Hiển thị Max/Min", value=True)
        show_actual = st.sidebar.checkbox("🌊 Thực tế (Cyan)", value=True)
        st.sidebar.markdown("---")
        history_days = st.sidebar.selectbox("🔍 Kiểm tra lịch sử:", [7, 14, 28], index=0)
        
        # --- BỎ VÒNG LẶP WHILE TRUE --- 
        # Code chạy thẳng từ trên xuống dưới 1 lần rồi dừng.
        # Nhưng ở cuối sẽ có lệnh ngủ và rerun.

        container = st.container()
        with container:
            # TÍNH TOÁN
            r_p, r_a, r_x, r_l, safety_margin = calculate_ratio_correction(df, days_check=7)
            
            # Trọng số
            train_test = df.iloc[:-7]
            actual_test = df.iloc[-7:]['Total_Flow'].values
            t_p = run_prophet(train_test, 7)['Prophet'].values * r_p
            t_a = run_arima(train_test, 7)['ARIMA'].values * r_a
            t_x = run_xgboost(train_test, 7)['XGBoost'].values * r_x
            try: 
                if len(t_x) != 7: t_x = t_p
            except: t_x = t_p
            t_l = run_linear(train_test, 7)['LinearReg'].values * r_l
            
            maes = []
            for pred in [t_p, t_a, t_x, t_l]:
                maes.append(mean_absolute_error(actual_test, pred) if len(pred)==7 else 1e9)
            weights = 1 / (np.array(maes)**2 + 1e-9)
            weights /= weights.sum()

            # Dự báo tương lai
            f_p = run_prophet(df); f_a = run_arima(df); f_x = run_xgboost(df); f_l = run_linear(df)
            future = f_p.merge(f_a, on='Date', how='outer').merge(f_x, on='Date', how='outer').merge(f_l, on='Date', how='outer')
            
            if not future.empty:
                future = future.fillna(0)
                if 'Prophet' in future.columns: future['Prophet'] *= r_p
                if 'ARIMA' in future.columns: future['ARIMA'] *= r_a
                if 'XGBoost' in future.columns: future['XGBoost'] *= r_x
                if 'LinearReg' in future.columns: future['LinearReg'] *= r_l
                
                future['AI_Optimized'] = (future['Prophet'] * weights[0]) + (future['ARIMA'] * weights[1]) + \
                                            (future['XGBoost'] * weights[2]) + (future['LinearReg'] * weights[3])
                
                buffer = safety_margin * 1.5 
                future['AI_Max'] = future['AI_Optimized'] + buffer
                future['AI_Min'] = future['AI_Optimized'] - buffer

                # --- UI SECTION: HERO METRICS ---
                tomorrow_forecast = future.iloc[-7]['AI_Optimized']
                tomorrow_max = future.iloc[-7]['AI_Max']
                tomorrow_min = future.iloc[-7]['AI_Min']
                last_actual = df['Total_Flow'].iloc[-1]
                delta_val = tomorrow_forecast - last_actual
                
                best_model_name = "Prophet"
                if weights[1] == max(weights): best_model_name = "ARIMA"
                if weights[2] == max(weights): best_model_name = "XGBoost"
                if weights[3] == max(weights): best_model_name = "Linear"

                st.subheader("🚀 TỔNG QUAN VẬN HÀNH NGÀY MAI")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📅 Dự báo Ngày mai", f"{tomorrow_forecast:,.0f} m³", f"{delta_val:,.0f} m³ vs Hôm nay", delta_color="inverse")
                c2.metric("🛡️ Vùng biến động", f"{tomorrow_min:,.0f} - {tomorrow_max:,.0f}", "Max/Min")
                c3.metric("🤖 Độ tin cậy (Biên độ)", f"±{safety_margin:.0f} m³", f"Dựa trên {history_days} ngày qua")
                c4.metric("🏆 Model tốt nhất", best_model_name, f"Trọng số: {max(weights):.1%}")

                st.markdown("---")
                
                # BIỂU ĐỒ TƯƠNG LAI
                st.write("### 📈 Chi tiết Dự báo 7 ngày tới")
                c1_chart, c1_table = st.columns([2.5, 1])
                with c1_chart:
                    fig = go.Figure()
                    if show_range and show_optimized:
                        fig.add_trace(go.Scatter(x=future['Date'], y=future['AI_Max'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=future['Date'], y=future['AI_Min'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='Vùng biến động'))
                    if show_actual:
                        fig.add_trace(go.Scatter(x=df['Date'], y=df['Total_Flow'], name='Thực tế', line=dict(color='#00FFFF', width=3))) 
                    if show_optimized:
                        fig.add_trace(go.Scatter(x=future['Date'], y=future['AI_Optimized'], name='DỰ BÁO TỐI ƯU', line=dict(color='red', width=4, dash='solid'), mode='lines+markers'))
                    fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
                
                with c1_table:
                    display_future = future[['Date', 'AI_Optimized', 'AI_Max', 'AI_Min']].copy()
                    display_future['Date'] = display_future['Date'].dt.strftime('%d/%m/%Y')
                    display_future.columns = ['Ngày', 'Dự báo', 'Max', 'Min']
                    st.dataframe(display_future.style.format("{:.0f}", subset=['Dự báo', 'Max', 'Min']), hide_index=True, use_container_width=True, height=400)

                # BIỂU ĐỒ QUÁ KHỨ VÀ ĐÁNH GIÁ)
                st.write(f"### 2. Đánh giá Quá khứ ({history_days} ngày)")
                past_start = len(df) - history_days
                past_actual = df.iloc[past_start:]['Total_Flow'].values
                past_dates = df.iloc[past_start:]['Date']
                mean_actual = np.mean(past_actual) if len(past_actual) > 0 else 1
                
                h_p = run_prophet(df.iloc[:-history_days], history_days)['Prophet'].values * r_p
                h_a = run_arima(df.iloc[:-history_days], history_days)['ARIMA'].values * r_a
                h_x = run_xgboost(df.iloc[:-history_days], history_days)['XGBoost'].values * r_x
                h_l = run_linear(df.iloc[:-history_days], history_days)['LinearReg'].values * r_l
                
                min_len = min(len(past_actual), len(h_p), len(h_a), len(h_x), len(h_l))
                h_optimized = (h_p[:min_len] * weights[0]) + (h_a[:min_len] * weights[1]) + (h_x[:min_len] * weights[2]) + (h_l[:min_len] * weights[3])
                h_max = h_optimized + buffer
                h_min = h_optimized - buffer

                ranking_data = []
                for name, pred in zip(['Prophet', 'ARIMA', 'XGBoost', 'Linear'], [h_p, h_a, h_x, h_l]):
                    try:
                        if len(pred) >= min_len:
                            mae = mean_absolute_error(past_actual[:min_len], pred[:min_len])
                            mae_pct = (mae / mean_actual)
                            ranking_data.append({"Mô hình": name, "Sai số (MAE)": mae, "Sai số (%)": mae_pct})
                    except: pass
                ranking_df = pd.DataFrame(ranking_data).sort_values("Sai số (MAE)")

                c2_rank, c2_chart = st.columns([1, 1.5])
                with c2_rank:
                    st.dataframe(ranking_df.style.format({
                        "Sai số (MAE)": "{:.0f}", 
                        "Sai số (%)": "{:.1%}"
                    }).background_gradient(subset=["Sai số (MAE)"], cmap="RdYlGn_r"), use_container_width=True)

                with c2_chart:
                    fig_err = go.Figure()
                    if show_range:
                        fig_err.add_trace(go.Scatter(x=past_dates[:min_len], y=h_max, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig_err.add_trace(go.Scatter(x=past_dates[:min_len], y=h_min, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='Vùng biến động'))
                    fig_err.add_trace(go.Scatter(x=past_dates[:min_len], y=past_actual[:min_len], name='THỰC TẾ', line=dict(color='black', width=3)))
                    fig_err.add_trace(go.Scatter(x=past_dates[:min_len], y=h_optimized, name='DỰ BÁO TỐI ƯU (Past)', line=dict(color='red', width=3, dash='solid')))
                    fig_err.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_err, use_container_width=True, key=str(uuid.uuid4()))

                st.success(f"Trạng thái: 🟢 Online | Dữ liệu phiên bản: {data_version} (Cập nhật 00:10 hằng ngày)")

    # --- TỰ ĐỘNG REFRESH NHẸ NHÀNG ---
    # Thay vì loop 60s, ta cho ngủ 10 phút (600s) rồi mới Rerun để check giờ.
    # Như vậy server đỡ tốn sức, mà vẫn đảm bảo qua 00:10 nó sẽ tự cập nhật.
    time.sleep(600)
    st.rerun()
