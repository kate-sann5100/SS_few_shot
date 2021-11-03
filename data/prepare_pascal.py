import os
import shutil

# list all images required for training and validation
img_list = []
for f in ["val.txt", "voc_sbd_merge_noduplicate.txt"]:
    img_list += [l[-16: -5] for l in open(f"./lists/pascal/{f}").readlines()]

for f in os.listdir(f"./data/pascal/SegmentationClassAug"):
    if f[:-4] not in img_list:
        os.remove(f"./data/pascal/SegmentationClassAug/{f}")

# move VOC image to target folder
for f in img_list:
    shutil.move(
        f"./data/pascal/VOCdevkit/VOC2012/JPEGImages/{f}.jpg",
        f"./data/pascal/JPEGImages/{f}.jpg"
    )

# remove unused files to save space
shutil.rmtree(f"./data/pascal/VOCdevkit")