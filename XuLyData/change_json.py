import json
import os
# import skimage

train = []
for foder in os.scandir("json_7000"):
    for file in os.scandir(foder):
        json_file = os.path.join(os.path.join(file))
        with open(json_file) as f:
            annotations = json.load(f)
        data = annotations["_via_img_metadata"]
        # data = annotations
        for image in data:
            im = {}
            im["flag"] = {}
            im["shape"] = []
            im["lineColor"] = [0, 255, 0, 128]
            im["fillColor"] = [255, 0, 0, 128]
            im["imagePath"] = data[image]["filename"]
            im["imageData"] = ""
            shapes = []
            print(data[image]["filename"])
            for region in data[image]["regions"]:
                shape = {}
                shape["label"] = "smoke"
                shape["line_color"] = None
                shape["fill_color"] = None
                points = []
                if(data[image]["regions"]!=[]):
                    if region["shape_attributes"]["name"]=="polyline":
                        l = len(region["shape_attributes"]["all_points_x"])
                        print(l)
                        for i in range(l):
                            points.append([region["shape_attributes"]["all_points_x"]
                                        [i], region["shape_attributes"]["all_points_y"][i]])
                shape["points"] = points
                shapes.append(shape)
            im["shapes"] = shapes
            with open('json/'+str(data[image]["filename"]).split('.')[0]+'.json', 'w') as outfile:
                json.dump(im, outfile)
