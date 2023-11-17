import pandas as pd
import shutil

image_data = pd.read_csv("./images/OASIS.csv")
for index, row in image_data.iterrows():
    image_name = row['Theme'].strip()
    valence = float(row['Valence_mean'])
    try:
        if valence > 6:
            shutil.move("./images/" + image_name + ".jpg", "./images/positive/" + image_name + ".jpg")
        elif valence < 4:
            shutil.move("./images/" + image_name + ".jpg", "./images/negative/" + image_name + ".jpg")
        else:
            shutil.move("./images/" + image_name + ".jpg", "./images/neutral/" + image_name + ".jpg")
    except Exception as e:
        print(e)
