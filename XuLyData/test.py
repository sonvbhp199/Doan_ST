import os
import json
# import skimage
files=[]
files_json=[]
# json_file = os.path.join(os.path.join("via_project_22Mar2022_18h40m (1).json"))
# with open(json_file) as f:
#     annotations = json.load(f)

# data = annotations["_via_img_metadata"]
# for image in data:
#     print(data[image]["filename"])
#     for region in data[image]["regions"]:
#         l = len(region["shape_attributes"]["all_points_x"])
#         for i in range(l):
#             print(region["shape_attributes"]["all_points_x"][i])
# for foder in os.scandir("smoke_jpg"):
#     for file in os.scandir(foder):
#         print(file)
# for file in os.listdir("images_2000"):
#     files.append(file.split(".")[0])
# for file in os.listdir("json_jpg"):
#     files_json.append(file.split(".")[0])
# print(set(files)^set(files_json))
# print(len(set(files)))

json_file = 'C:/Users/SON/Downloads/2112.json'
with open(json_file) as f:
    annotations = json.load(f)
data = annotations["_via_img_metadata"]
for image in data:
    im ={}
    im["flag"]={}
    im["shape"]=[]
    im["lineColor"]= [0,255,0,128]
    im["fillColor"]=[255,0,0,128]
    im["imagePath"]= data[image]["filename"]
    im["imageData"]=""
    shapes=[]
    for region in data[image]["regions"]:
        shape={}
        shape["label"]="smoke"
        shape["line_color"]=None
        shape["fill_color"]=None
        points=[]
        l = len(region["shape_attributes"]["all_points_x"])
        for i in range(l):
            points.append([region["shape_attributes"]["all_points_x"][i],region["shape_attributes"]["all_points_y"][i]])
        shape["points"] = points
        shapes.append(shape)
    im["shapes"]=shapes
    with open('jsonthieu/'+str(data[image]["filename"]).split('.')[0]+'.json', 'w') as outfile:
        json.dump(im, outfile) 