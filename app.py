import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from mtcnn import MTCNN
from deepface import DeepFace

st.set_page_config(layout="wide")
st.title("👥 Gender Participant Counter — SDG5 AI")

st.info("Upload image → AI detects faces → predicts gender → counts participants")

detector = MTCNN()

file = st.file_uploader("Upload Group Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    faces = detector.detect_faces(img_np)

    male = female = unknown = 0
    rows = []

    for face in faces:
        x,y,w,h = face["box"]
        x,y = max(0,x), max(0,y)
        crop = img_np[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                crop,
                actions=['gender'],
                enforce_detection=False
            )[0]

            g = result["gender"]
            m = g["Man"]
            f = g["Woman"]

            if abs(m-f) < 15:
                label = "Unknown"
                unknown += 1
                color=(150,150,150)
            elif m > f:
                label = "Male"
                male += 1
                color=(0,0,255)
            else:
                label = "Female"
                female += 1
                color=(255,0,255)

        except:
            label = "Unknown"
            unknown += 1
            color=(150,150,150)

        cv2.rectangle(img_np,(x,y),(x+w,y+h),color,2)
        cv2.putText(img_np,label,(x,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        rows.append([label])

    st.image(img_np)

    total = male + female + unknown

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Male", male)
    c3.metric("Female", female)
    c4.metric("Unknown", unknown)

    chart = pd.DataFrame({
        "Gender":["Male","Female","Unknown"],
        "Count":[male,female,unknown]
    })
    st.bar_chart(chart.set_index("Gender"))

    df = pd.DataFrame(rows, columns=["Gender"])
    st.download_button("Download Report",
                       df.to_csv(index=False),
                       "gender_report.csv")
