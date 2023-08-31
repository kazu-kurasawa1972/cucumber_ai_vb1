
import streamlit as st
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("きゅうりのスライス評価アプリ")
st.sidebar.write("このアプリは、きゅうりのスライスの写真をもとに、どれだけ上手にスライスされているかを0から10のスコアで評価します。")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        score = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        st.write(f"このきゅうりのスライスのスコアは: {score:.2f}/10 です。")

st.sidebar.write("")
st.sidebar.write("")

st.sidebar.caption("""
このアプリは、きゅうりのスライスの評価のためにカスタムトレーニングされたモデルを使用しています。
""")
