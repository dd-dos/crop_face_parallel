import os
import glob
import tqdm
import shutil

shutil.rmtree("./new_cropped_face", ignore_error=True)
os.makedirs("./new_cropped_face")
os.makedirs("./new_cropped_face/live")
os.makedirs("./new_cropped_face/spoof")


for file in tqdm.tqdm(glob.glob("./cropped_face/*/*.jpg")):
    name = file.split("/")[-1]
    new_name = "test." + name
    if "live" in file:
        shutil.copy(file, os.path.join("./new_cropped_face/live", new_name))
    elif "spoof" in file:
        shutil.copy(file, os.path.join("./new_cropped_face/spoof", new_name))
