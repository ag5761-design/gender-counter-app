import streamlit as st
from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np
from PIL import Image
import pandas as pd

st.title("👥 Gender Participant Counter — SDG5 AI")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)

    detections = RetinaFace.detect_faces(img_np)

    male = female = unknown = 0
    rows = []

    if isinstance(detections, dict):
        for key in detections:
            area = detections[key]["facial_area"]
            x1,y1,x2,y2 = area["x1"],area["y1"],area["x2"],area["y2"]
            crop = img_np[y1:y2, x1:x2]

            try:
                res = DeepFace.analyze(
                    crop,
                    actions=['gender'],
                    enforce_detection=False
                )[0]

                g = res["gender"]
                m = g["Man"]
                f = g["Woman"]

                if abs(m-f) < 15:
                    label="Unknown"; unknown+=1
                elif m>f:
                    label="Male"; male+=1
                else:
                    label="Female"; female+=1

            except:
                label="Unknown"; unknown+=1

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
    st.download_button("Download CSV",
                       df.to_csv(index=False),
                       "report.csv")
