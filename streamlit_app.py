import streamlit as st
from PIL import Image
import torch
import pandas as pd
import os
import time

from transformers import CLIPProcessor, CLIPModel
from streamlit_js_eval import streamlit_js_eval

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

classes = ["a photo of a driver club",
        "a photo of a 3-wood club", "a photo of a 5-wood club",
        "a photo of a 2-hybrid club", "a photo of a 3-hybrid club", "a photo of a 4-hybrid club",
        "a photo of a 2-iron club", "a photo of a 3-iron club", "a photo of a 4-iron club",
        "a photo of a 5-iron club", "a photo of a 6-iron club", "a photo of a 7-iron club",
        "a photo of a 8-iron club", "a photo of a 9-iron club",
        "a photo of a pitching wedge club", "a photo of a sand wedge club", "a photo of a lob wedge club",
        "a photo of a gap wedge club", "a photo of a putter club", "a photo of a something that is not a golf club"]

CLASSES = classes

threshold = 0.06

def send_req(img):
    inputs = processor(text=classes, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()
    return probs

def save_img(path, image):
    if not os.path.exists(path):
        os.makedirs(path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(f"{path}/{len(os.listdir(path)) + 1}.jpg")

if 'show_selectbox' not in st.session_state:
    st.session_state.show_selectbox = False

# Streamlit app
st.markdown(f"<h1 style='text-align: center'><b>โปรอาร์มทำนายไม้กอล์ฟ</b></h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("แปะรูปเลยยย", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    probs = send_req(image)
    prob, txt = max(zip(probs[0], classes))
    if (probs[0][-1] >= threshold):
        txt = "not_golf_club"
        st.markdown(f"<h2 style='text-align: center'><b>หาไม่เจอหรือมันไม่มี</b> </h2>", unsafe_allow_html=True)
    else:
        funny_text = ["มั้ง", "แหละ", "ชัว ๆ", "ชัว ๆ ยิ่งกว่าโดนหวยแดก"]
        funny_text_prob = [0.05, 0.10, 0.15, 0.2]
        txt = txt.split("photo of a")[1][: -5]
        idx = 0
        for i in range(len(funny_text)):
            if (prob < funny_text_prob[i]):
                break
            idx = i
        st.markdown(f"<h2 style='text-align: center'><b>{txt}</b> {funny_text[idx]}</h2>", unsafe_allow_html=True)

        with st.expander("ลองคลิกดู"):
            data = sorted(zip(classes[:-1], probs[0][:-1]), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(data, columns=["Class", "Probability"])
            df["Probability"] = df["Probability"].map("{:.2f}".format)
            df["Probability"] = df["Probability"].astype(float) * 100
            df["Probability"] = df["Probability"].apply(lambda x: f"{x}%")
            df["Class"] = df["Class"].apply(lambda x: x.split("photo of a ")[1])
            st.table(df)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f"<h5 style='text-align: left; color: grey'>ถูกไหมบอกฉันหน่อย 🥲</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
        }
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        btn1 = st.button("ถูก ๆ เก่ง ๆ")

    with col2:
        btn2 = st.button("ผิด อ่อน...")

    txt = txt.strip().replace(" ", "_").strip().lower()
    if btn1:
        path = './' + txt
        save_img(path, image)
        st.success("สวยยยยย 🤩")
        time.sleep(0.5)
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
    if btn2:
        st.session_state.show_selectbox = True

    if st.session_state.show_selectbox:
        classes = list(map(lambda x: x.split("photo of a ")[1][: -5].strip().lower(), classes))[:-1]
        option = st.selectbox(
            "เฉลยหน่อยสิ",
            classes + ["บ่ใช่ไม้กอล์ฟ"],
            index=None,
            placeholder="เลือกเลยย"
        )

        if option == "บ่ใช่ไม้กอล์ฟ":
            option = "not_golf_club"
        if option == txt:
            path = './' + option
            save_img(path, image)
            st.error("มึงเล่นไรนิ... 🧐")
            time.sleep(0.5)
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        elif option:
            path = './' + option
            save_img(path, image)
            st.success("ใจเด้อ 😍")
            time.sleep(0.5)
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
