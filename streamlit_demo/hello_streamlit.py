import streamlit as st
import pandas as pd

st.title("这是一个标题")  # 显示标题
st.write("这是一段文本。")  # 显示文本
st.code('''print("这是一段代码")''', language='python')  # 显示代码段

# 显示表格
df = pd.DataFrame({"column1": (1, 2, 3), "column2": ("a", "b", "c")})
st.write(df)