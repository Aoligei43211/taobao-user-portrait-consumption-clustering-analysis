import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

st.set_page_config(page_title="KMeans 用户画像分析", page_icon="", layout="wide")

st.title("基于 KMeans 的淘宝用户消费聚类分析")
st.markdown("技术路线：数值特征 + 独热编码文本特征 | 标准化处理")

@st.cache_data
def load_data():
    df = pd.read_csv("data/淘宝.csv")
    df["age"] = pd.cut(df["用户年龄"], bins=[17, 35, 50, 65], labels=["青年", "中年", "老年"])
    return df

df = load_data()

with st.sidebar:
    st.header("数据概览")
    st.metric("总记录数", len(df))
    st.metric("用户年龄范围", f"{df['用户年龄'].min()} - {df['用户年龄'].max()}")
    st.metric("商品类别数", df["商品类别"].nunique())
    
    st.divider()
    st.header("聚类参数")
    k_min = st.slider("最小簇数", 2, 10, 2)
    k_max = st.slider("最大簇数", 3, 15, 12)

st.header("1. 数据预处理")
st.markdown("""
- **数值特征**：消费金额（log变换 + 标准化）
- **类别特征**：商品类别、用户性别、age（独热编码）
- **控制变量**：与 KPrototypes 保持一致的 4 个原始字段
""")

features = ["商品类别", "消费金额", "用户性别", "age"]
data = df.loc[:, ["商品类别", "消费金额", "用户性别", "age"]].copy()

data["消费金额"] = np.log1p(data["消费金额"])

data_encoded = pd.get_dummies(data, columns=["商品类别", "用户性别", "age"], dtype=float)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)
data_scaled_df = pd.DataFrame(data_scaled, columns=data_encoded.columns)

with st.expander("查看预处理后数据"):
    st.dataframe(data_scaled_df.head(10), use_container_width=True)

st.header("2. 确定最佳簇数")
st.markdown("使用肘部法则、轮廓系数、CH指数综合判断")

costs = []
silhouettes = []
ch_scores = []
k_range = range(k_min, k_max + 1)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data_scaled)
    costs.append(km.inertia_)
    silhouettes.append(silhouette_score(data_scaled, labels))
    ch_scores.append(calinski_harabasz_score(data_scaled, labels))

col1, col2, col3 = st.columns(3)

with col1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=costs, mode="lines+markers", name="Cost", line=dict(color="#636EFA")))
    fig_elbow.update_layout(title="肘部法则", xaxis_title="簇数 K", yaxis_title="Cost (Inertia)", height=350)
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=list(k_range), y=silhouettes, mode="lines+markers", name="Silhouette", line=dict(color="#EF553B")))
    fig_sil.update_layout(title="轮廓系数", xaxis_title="簇数 K", yaxis_title="Silhouette Score", height=350)
    st.plotly_chart(fig_sil, use_container_width=True)

with col3:
    fig_ch = go.Figure()
    fig_ch.add_trace(go.Scatter(x=list(k_range), y=ch_scores, mode="lines+markers", name="CH Index", line=dict(color="#00CC96")))
    fig_ch.update_layout(title="CH 指数", xaxis_title="簇数 K", yaxis_title="Calinski-Harabasz Score", height=350)
    st.plotly_chart(fig_ch, use_container_width=True)

best_k_sil = k_range[np.argmax(silhouettes)]
best_k_ch = k_range[np.argmax(ch_scores)]

col1, col2 = st.columns(2)
col1.metric("轮廓系数最佳 K", best_k_sil, f"得分: {max(silhouettes):.4f}")
col2.metric("CH指数最佳 K", best_k_ch, f"得分: {max(ch_scores):.4f}")

st.header("3. 聚类结果分析")

with st.sidebar:
    st.divider()
    st.header("选择聚类数")
    selected_k = st.radio("选择 K 值", [best_k_sil, best_k_ch] + [k for k in k_range if k not in [best_k_sil, best_k_ch]], index=0)

km_final = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(data_scaled)

cluster_stats = df.groupby("cluster").agg({
    "用户ID": "count",
    "消费金额": ["mean", "median", "min", "max"],
    "商品类别": lambda x: x.mode().iloc[0],
    "用户性别": lambda x: x.mode().iloc[0],
    "age": lambda x: x.mode().iloc[0]
}).reset_index()
cluster_stats.columns = ["簇", "人数", "平均消费", "中位消费", "最低消费", "最高消费", "主导类别", "主导性别", "主导年龄段"]
cluster_stats["人数占比"] = (cluster_stats["人数"] / len(df) * 100).round(1)

st.subheader("3.1 各簇画像摘要")
st.dataframe(cluster_stats, use_container_width=True)

st.subheader("3.2 簇人数分布")
fig_pie = px.pie(cluster_stats, values="人数", names="簇", title=f"K={selected_k} 簇人数占比", hole=0.3)
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("3.3 消费金额分布（按簇）")
fig_box = px.box(df, x="cluster", y="消费金额", color="cluster", title="各簇消费金额箱线图", points="outliers")
fig_box.update_layout(xaxis_title="簇编号", yaxis_title="消费金额")
st.plotly_chart(fig_box, use_container_width=True)

st.subheader("3.4 商品类别偏好（按簇）")
cat_cluster = pd.crosstab(df["cluster"], df["商品类别"], normalize="index") * 100
fig_cat = go.Figure()
for col_name in cat_cluster.columns:
    fig_cat.add_trace(go.Bar(x=cat_cluster.index, y=cat_cluster[col_name], name=col_name))
fig_cat.update_layout(barmode="group", title="各簇商品类别占比", xaxis_title="簇编号", yaxis_title="占比 (%)", height=500)
st.plotly_chart(fig_cat, use_container_width=True)

st.subheader("3.5 性别与年龄段分布（按簇）")
tab1, tab2 = st.tabs(["性别分布", "年龄段分布"])
with tab1:
    gender_cluster = pd.crosstab(df["cluster"], df["用户性别"], normalize="index") * 100
    fig_gender = go.Figure()
    for col_name in gender_cluster.columns:
        fig_gender.add_trace(go.Bar(x=gender_cluster.index, y=gender_cluster[col_name], name=col_name))
    fig_gender.update_layout(barmode="group", title="各簇性别占比", xaxis_title="簇编号", yaxis_title="占比 (%)", height=400)
    st.plotly_chart(fig_gender, use_container_width=True)

with tab2:
    age_cluster = pd.crosstab(df["cluster"], df["age"], normalize="index") * 100
    fig_age = go.Figure()
    for col_name in age_cluster.columns:
        fig_age.add_trace(go.Bar(x=age_cluster.index, y=age_cluster[col_name], name=col_name))
    fig_age.update_layout(barmode="group", title="各簇年龄段占比", xaxis_title="簇编号", yaxis_title="占比 (%)", height=400)
    st.plotly_chart(fig_age, use_container_width=True)

st.header("4. 交互式用户画像看板")
selected_cluster = st.selectbox("选择簇编号查看详细画像", sorted(df["cluster"].unique()))

cluster_data = df[df["cluster"] == selected_cluster]

col1, col2, col3, col4 = st.columns(4)
col1.metric("簇人数", len(cluster_data))
col2.metric("平均消费", f"{cluster_data['消费金额'].mean():.2f}")
col3.metric("主导商品", cluster_data["商品类别"].mode().iloc[0])
col4.metric("主导年龄段", cluster_data["age"].mode().iloc[0])

st.subheader("消费金额分布")
fig_hist = px.histogram(cluster_data, x="消费金额", nbins=20, title=f"簇 {selected_cluster} 消费金额直方图")
st.plotly_chart(fig_hist, use_container_width=True)
