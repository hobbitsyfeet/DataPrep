import h5py
import numpy as np
from plyfile import PlyData, PlyElement
import os
import random
import pcl
import argparse



#this is what's used for cross validation
from sklearn.model_selection import train_test_split

def collect_files(top_dir, file_type="ply"):
    """
    Collects files ending in 'file_type' anywhere within a directory.
    """
    filenames =[]
    dir_list = []
    #walk down the root, 
    for root, dirs, files in os.walk(top_dir, topdown=False):
        
        for name in files:
            #print(os.path.join(root, name))
            if name[-3:] == file_type:
                filenames.append(os.path.join(root, name))
                
        for name in dirs:
            dir_list.append(os.path.join(root, name))
            print(os.path.join(root, name))
        #print("Exploring " + str(dirs))
    return filenames, dir_list

def get_label(filename, labels=["head", "body", "arm","tail", "leg", "ear"]):
    """
    If the filename contains the word in the list called labels, it returns a pair of name and id number.

    example: 
            get_labels(./raccoon/labels/head.ply, ["head, leg"]): It will return ("head",0)
            get_labels(./raccoon/somefilename.ply, ["goose","raccoon"]): It will return ("raccoon,1")
    """
    for label in labels:
        if label.lower() in filename.lower():
            return (label, labels.index(label))

    return -1
    #raise Exception("There exists no label with "+ filename +". Provide the label of the contained file through the folder name or filename.")


def extract_ply_data(filenames, min_points=1024):
    """
    Collects all the point data from ply files, and places it into a list.
    If the file does not contain enough points, ignore the file.


    Param:
        filenames: A list of filenames to extract data from. Use collect_files to get this list.

        min_points: The minimum number of points needed to allow the file to be extracted.
    """
    data_list = [] #(x,y,z,dtype=np.float32)
    normal_list = [] #(normal,dtype=np.float32)
    label_list = [] #(dtype=uint8)
    #a_data = np.zeros((len(filenames), 2048, 3))
    for file_count in range(0, len(filenames)):
        # try:
        plydata = PlyData.read(filenames[file_count])
        
        #collect all of the data from the ply [[x,y,z,r,g,b][...]]
        data = plydata['vertex'][:][:]
        print(len(data))
        #do not allow files that have less than min_points since this is our sample size.
        if len(data) >= min_points:

            normal = calculate_normals(filenames[file_count])

            label, label_index = get_label(filenames[file_count], labels=["Head", "Hips", "left_knee","Left_Ankle", "Left_Bicep", "Left_Elbow", "Left_Femur", "Left_Foot", "Left_Forearm", "Left_Hand", "Left_Hip", "Left_Shin", "Left_Sholder", "Left_Wrist",
                                                            "Right_Ankle", "right_knee" , "Right_Ankle", "Right_Bicep", "Right_Elbow", "Right_Femur", "Right_Foot", "Right_Forearm", "Right_Hand", "Right_Hip", "Right_Shin", "Right_Sholder", "Right_Wrist", 
                                                            "Neck", "Torso", "Waist"])
            #add data and label to their respective lists.
            if label != -1:
                print("Adding " + str(filenames[file_count]) + ": " + label[0].capitalize())
                data_list.append(data)
                label_list.append(label[1])
                normal_list.append(normal)
            else:
                print("Skipping " + str(filenames[file_count] + ": Label Does Not Exist"))
        else: 
            print("Skipping " + str(filenames[file_count] + ": Not Enough Data"))

    print("Number of Elements in Data: "+str(len(data_list)))
    return data_list, normal_list, label_list

def calculate_normals(pcl_cloud):
    ne = pcl_cloud.make_NormalEstimation()
    tree = pcl_cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.5)
    cloud_normals = ne.compute()
    #print(ne)
    #print(cloud_normals)

    #print(cloud_normals)

    return cloud_normals

def get_samples(data_list, sample_size=2048):
    """
    Takes data from a list and samples until a specific number. This (does not yet) perfoms on the data and the id's
    """
    #ensure sample size before creating a data object
    # for file_count in range(len(data_list)):
    #     try:
    #         if len(data_list[file_count]) < sample_size:
    #             print("Removing data from index: " + str(file_count))
    #             del data_list[file_count]
    #             del normal_list[file_count]
    #     #I'm too tired to fix the loop running over...
    #     except:
    #         pass
    
    a_data = np.zeros((len(data_list), sample_size, 3))

    #a_pid = np.zeros((len(filenames), 2048), dtype = np.uint8)	

    for file_count in range(0, len(data_list)):
        #select the set of points from what was an entire file
        data = data_list[file_count]
        #normal = normal_list[file_count]

        #data_pair = (data,normal)

        #randomly sample a number of points until sample size is reached. Samples not duplicated.
        subsampled = random.sample(data, sample_size)
        
        #for the data sampled, use only the x,y,z values
        for j in range(0, len(subsampled)):
            data_sampled = subsampled
            #normal_sampled = subsampled[1]
            a_data[file_count, j] = [data_sampled[j][0], data_sampled[j][1], data_sampled[j][2]]
            #n_data[file_count,j] = [data_sampled[j][0], data_sampled[j][1], data_sampled[j][2]]

        print("Working on file number: " + str(file_count+1) ,end='\r')
    print('\n')
    return a_data

def export_seg():
    """
    """
    pass

def export_classify():
    pass

def export_sem_seg():
    pass


if __name__ == "__main__":
    
    # DATA_DIR = "E:/Data/Kinect/StickMan/Segmented/Forward"
    # OUT_DIR = "E:/Data/Kinect/StickMan/Labelled"

    # DATA_DIR = "D:/Data/ALL_DATA/Seg_Raccoon/Left_Right_Perspective_Seg/Both_Seg/Segmented_L"
    DATA_DIR = "D:/Data/ALL_DATA/Seg_Raccoon/Left_Right_Perspective_Seg/Both_Seg/OneSample"
    OUT_DIR = "D:/Data/ALL_DATA/Seg_Raccoon/Left_Right_Perspective_Seg/Both_Seg"
    sample_size = 2048

    plydata = None
    All_Files, dirs = collect_files(DATA_DIR,file_type="ply")
    #print(len(dirs))
    data_list = []
    normal_list = []
    label_list = []

        # #write test_data, test_label, train_data, train_label val_data, val_data
    folder_name = 'raccoon'
    file_name = "raccoon"
    
    # try:
    #     os.mkdir(OUT_DIR+"/Exported")
    # except:
    #     pass
    # OUT_DIR = OUT_DIR + "/Exported"
    # print(OUT_DIR)
    try:
        os.mkdir(OUT_DIR+"/test_train")
    except:
        pass
    try:
        os.mkdir(OUT_DIR+"/data")
    except:
        pass
    try:
        os.mkdir(OUT_DIR + "/test_ply")
    except:
        pass
    try:
        os.mkdir(OUT_DIR + "train_ply")
    except:
        pass

    
    file_count = 0
    folder_name = 'raccoon'
    file_name = "raccoon"


    #filelists for .h5
    train_files = open(OUT_DIR +"/test_train/" + "train_files.txt", 'a')
    test_files = open(OUT_DIR +"/test_train/" + "test_files.txt", 'a')

    #print(All_Files)
    #print(len(All_Files))
    data_h5 = np.zeros((len(dirs), sample_size, 3))
    normals_h5 = np.zeros((len(dirs), sample_size, 3))
    labels_h5 = np.zeros((len(dirs), sample_size))

    data_num_h5 = np.zeros((len(dirs)))
    model_id_h5 = np.zeros((len(dirs)))

    #file_count = 0
    #set a random seed once for all datasets for sampling
    random.seed(random.random())
    for directory in dirs:
        print(directory)
        filenames, dirs = collect_files(directory,file_type="ply")
        txt_file = open(OUT_DIR + "/data/" + file_name + str(file_count) + ".txt", 'w')
        # train_data_file = open(OUT_DIR+"/train_data/" + file_name + str(file_count) + ".pts", 'w')
        # train_label_file = open(OUT_DIR+"/train_label/" + file_name + str(file_count) + ".seg", 'w')
        # test_data_file = open(OUT_DIR+"/test_data/" + file_name + str(file_count) + ".pts", 'w')
        # test_label_file = open(OUT_DIR+"/test_label/" + file_name + str(file_count) + ".seg", 'w')
        print("================Writing File " + file_name + str(file_count) + ".txt================")
        temp_data_list = []
        temp_normal_list = []
        temp_label_list = []
        for file in filenames:
            print(file)
            pcl_cloud = pcl.load(file,format="ply")


            size = len(pcl_cloud.to_list())
            if size > 64:
                normal_cloud = (calculate_normals(pcl_cloud))

                # label = get_label(file, labels=["Head", "Hips", "left_knee","Left_Ankle", "Left_Bicep", "Left_Elbow", "Left_Femur", "Left_Foot", "Left_Forearm", "Left_Hand", "Left_Hip", "Left_Shin", "Left_Sholder", "Left_Wrist",
                                                            # "Right_Ankle", "right_knee" , "Right_Ankle", "Right_Bicep", "Right_Elbow", "Right_Femur", "Right_Foot", "Right_Forearm", "Right_Hand", "Right_Hip", "Right_Shin", "Right_Sholder", "Right_Wrist", 
                                                            # "Neck", "Torso", "Waist"])[1]
                label = get_label(file,labels=["Body", "Head", "Arm", "Tail"])


                for point in range(len(pcl_cloud.to_list())):
                    data_string = str(pcl_cloud[point][0]) + " " + str(pcl_cloud[point][1]) + " " + str(pcl_cloud[point][2]) + " " 
                    normal_string = str(normal_cloud[point][0]) + " " + str(normal_cloud[point][1]) + " " +str(normal_cloud[point][2]) + " "
                    label_string =  str(label)

                    #print(string)
                    temp_data_list.append((pcl_cloud[point][0],pcl_cloud[point][1],pcl_cloud[point][2]))
                    temp_normal_list.append((normal_cloud[point][0],normal_cloud[point][1],normal_cloud[point][2]))
                    temp_label_list.append(label)

                    txt_file.write(data_string + normal_string + label_string + '\n')
        data_list.append(temp_data_list)
        normal_list.append(temp_normal_list)
        label_list.append(temp_label_list)
        #print(temp_label_list)

        txt_file.close()
        # if data_list != []:

        file_count += 1
    # print(data_list)
    # print(len(data_list))
        #filelists for test_data, train_data ...
    try:
        os.mkdir(OUT_DIR+"/test_data")
    except:
        pass
    try:
        os.mkdir(OUT_DIR+"/test_label")
    except:
        pass
    try:
        os.mkdir(OUT_DIR+"/train_data")
    except:
        pass
    try:
        os.mkdir(OUT_DIR+"/train_label")
    except:
        pass
    try:
        os.mkdir(OUT_DIR+"/val_data")
    except:
        pass
    try:
        os.mkdir(OUT_DIR+"/val_label")
    except:
        pass
    try:
        os.mkdir(OUT_DIR +"/train")
    except:
        pass
    try: 
        os.mkdir(OUT_DIR+"/test")
    except:
        pass
    
    try:
        data_train, data_test, normal_train, normal_test, label_train, label_test = train_test_split(data_list, normal_list, label_list, test_size=0.33, random_state=42)
    except:
        print("Not enough for test_train split. Setting Testing and training to be the same.")
        data_train = data_list
        data_test = data_list

        normal_train = normal_list
        normal_test = normal_list
        
        label_train = label_list
        label_test = label_list

    file_count = 0
    #print(data_train)
    print("Exporting .pts and .seg")
    for model_index in range(len(data_train)):

        train_file = open(OUT_DIR+"/train/" + file_name + str(file_count) + ".txt", 'w')
        train_data_file = open(OUT_DIR+"/train_data/" + file_name + str(file_count) + ".pts", 'w')
        train_label_file = open(OUT_DIR+"/train_label/" + file_name + str(file_count) + ".seg", 'w')
        for data in range(len(data_train[model_index])):
            # print(label_train)
            # train_str = data_train[model_index][]

            data_str = str(data_train[model_index][data][0]) + " " + str(data_train[model_index][data][1]) + " " + str(data_train[model_index][data][2])
            normal_str = str(normal_train[model_index][data][0]) + " " + str(normal_train[model_index][data][1]) + " " + str(normal_train[model_index][data][2])
            label_str = str(int(label_train[model_index][data][1]))
            
            train_file.write(data_str + " " + normal_str + " " + label_str + '\n')

            train_data_file.write((data_str + '\n'))
            train_label_file.write((label_str + '\n'))
            
        file_count += 1

    for model_index in range(len(data_test)):

        test_file = open(OUT_DIR+"/test/" + file_name + str(file_count) + ".txt", 'w')
        test_data_file = open(OUT_DIR+"/test_data/" + file_name + str(file_count) + ".pts", 'w')
        test_label_file = open(OUT_DIR+"/test_label/" + file_name + str(file_count) + ".seg", 'w')

        try:
            test_cloud = pcl.PointCloud(data_train[model_index])
            print(test_cloud)
            pcl.save(test_cloud,(OUT_DIR + "/test_ply/" + file_name + str(file_count) + ".ply"), format="ply")
        except:
            print("No Data, skipping.")
            continue
        for data in range(len(data_test[model_index])):
            
            data_str = str(data_test[model_index][data][0]) + " " + str(data_test[model_index][data][1]) + " " + str(data_test[model_index][data][2])
            normal_str = str(normal_test[model_index][data][0]) + " " + str(normal_test[model_index][data][1]) + " " + str(normal_test[model_index][data][2])
            label_str = (str(int(label_test[model_index][data][1])))

            test_file.write(data_str + " " + normal_str + " " + label_str + '\n')
            # data_str = str(data_test[model_index][data][0]) + " " + str(data_test[model_index][data][1]) + " " + str(data_test[model_index][data][2])
            test_data_file.write((data_str + '\n'))
            test_label_file.write((label_str +'\n'))

        file_count += 1
        print(".pts .seg Success")
    # data_string = str(pcl_cloud[point][0]) + " " + str(pcl_cloud[point][1]) + " " + str(pcl_cloud[point][2]) + " " 
    # normal_string = str(normal_cloud[point][0]) + " " + str(normal_cloud[point][1]) + " " +str(normal_cloud[point][2]) + " "
    # label_string = str(label)
    
    # train_data_file.write(str(data_train[file_count]))
    # train_label_file.write(data_train[file_count])
    
    # test_data_file.write(data_test[f])
    # test_label_file.write(label_test)

        # print(data_num_h5.shape)
        print("Exporting .h5...")
        data_num_h5 = []
        model_id_h5 = []
        for model_num in range(len(data_list)): #There should be as many 'lists' as there are complete scans

            #Whole model prediction
            # data_num_h5[model_num] = len(data_list[model_num]) #data_num is the number of points in the model
            # model_id_h5[model_num] = 0

            print("Creating HDF5")
            sampled_data = random.sample(data_list[model_num], sample_size)
            sampled_normals = random.sample(normal_list[model_num], sample_size)
            sampled_labels = random.sample(label_list[model_num], sample_size)
            
            data_num_h5.append(len(sampled_data))

            # data_num_h5[model_num] = len(sampled_data) #data_num is the number of points in the model
            print("DATANUM", data_num_h5)
            # model_id_h5[model_num] = 0
            model_id_h5.append(0)
            print("MODEL_ID", model_id_h5)
            for point in range(len(sampled_data)):

                    data_h5[model_num, point] = [sampled_data[point][0], sampled_data[point][1], sampled_data[point][2]]
                    #print(data_h5[file_count,point])
                    normals_h5[model_num, point] = sampled_normals[point]
                    #print(normals_h5[file_count,point])

                    labels_h5[model_num, point] = sampled_labels[point][1]
                    # print(labels_h5[index,point])

            # print(data_h5[model_num])
            print(data_num_h5)
    
    # print(data_h5)
    # print(len(data_h5))


    print("CREATING HDF5 DATASET")
    
    try:
        # data_train, data_test, normal_train, normal_test, label_train, label_test = train_test_split(data_list, normal_list, label_list, test_size=0.33, random_state=42)
        data_train, data_test, normal_train, normal_test, label_train, label_test, data_num_train, data_num_test, model_id_train, model_id_test = train_test_split(data_h5, normals_h5, labels_h5, data_num_h5, model_id_h5, test_size=0.33, random_state=42)
    except:
        print("Not enough for test_train split. Setting Testing and training to be the same.")
        data_train = data_h5
        data_test = data_h5

        data_num_test = data_num_h5
        data_num_train = data_num_h5

        model_id_test = model_id_h5
        model_id_train = model_id_h5

        normal_train = normals_h5
        normal_test = normals_h5
        
        label_train = labels_h5
        label_test = labels_h5
        # data_train, data_test, normal_train, normal_test, label_train, label_test = train_test_split(data_list, normal_list, label_list, test_size=0.33, random_state=42)
        # data_train, data_test, normal_train, normal_test, label_train, label_test, data_num_train, data_num_test, model_id_train, model_id_test = train_test_split(data_h5, normals_h5, labels_h5, data_num_h5, model_id_h5, test_size=0.33, random_state=42)
    
    # print(data_train)

    print("DATA_TRAIN")


    # #write H5 files
    train_filename = OUT_DIR + "/data/"+ file_name +"_seg_train.h5"
    train_files.write(train_filename + '\n')
    
    hdf_train = h5py.File(train_filename, "w")
    #dataset = f.create_dataset("data", data = point_data)
    hdf_train.create_dataset("data", data = data_train)
    hdf_train.create_dataset("data_num", data = data_num_train)
    hdf_train.create_dataset("label", data = model_id_train) #Here we are just saying the labels belong to only one object (stick man, raccoon...)
    hdf_train.create_dataset("label_seg", data = label_train) #The labels for each point
    hdf_train.create_dataset("normal", data = normal_train) #surface normals

    hdf_train.flush()
    hdf_train.close()

    test_filename =OUT_DIR + "/data/"+ file_name +"_seg_test.h5"
    test_files.write(test_filename + '\n')
    hdf_test = h5py.File(test_filename, "w")
    #dataset = f.create_dataset("data", data = point_data)
    hdf_test.create_dataset("data", data = data_test)
    hdf_test.create_dataset("data_num", data = data_num_test)
    hdf_test.create_dataset("label", data = model_id_test) #Here we are just saying the labels belong to only one object (stick man, raccoon...)
    hdf_test.create_dataset("label_seg", data = label_test)
    hdf_test.create_dataset("normal", data = normal_test)

    hdf_test.flush()
    hdf_test.close()
    print("HDF5 DATASET COMPLETE")

    # # #write test_data, test_label, train_data, train_label val_data, val_data
    # folder_name = 'Raccoons'
    # file_name = "raccoon"
    
    # #filelists for test_data, train_data ...
    # try:
    #     os.mkdir(OUT_DIR+"/test_data")
    # except:
    #     pass
    # try:
    #     os.mkdir(OUT_DIR+"/test_label")
    # except:
    #     pass
    # try:
    #     os.mkdir(OUT_DIR+"/train_data")
    # except:
    #     pass
    # try:
    #     os.mkdir(OUT_DIR+"/train_label")
    # except:
    #     pass
    # try:
    #     os.mkdir(OUT_DIR+"/val_data")
    # except:
    #     pass
    # try:
    #     os.mkdir(OUT_DIR+"/val_label")
    # except:
    #     pass



    # #normal cloud and pointcloud are the same length SO we can do this
    # pair_list = []
    
    # for file in range(len(data_list)):
    #     pairs=[]
    #     #do this because pcl_cloud.size() doesn't work.
    #     for point in range(len(data_list[file].to_list())):
    #         #print(len(data_list[file].to_list()))
    #         #subsampled = random.sample(file, 1024)
    #         pairs.append((data_list[file][point],normal_list[file][point]))
    #     pair_list.append(pairs)
    #     pairs=[]

    # print(len(pair_list))
    
    #data_list, label_list = extract_ply_data(filenames,min_points=64)
    #print(label_list)

    
    # print( "Total files added: " + str(len(data_list)))
    # for file in pair_list:
    #     print(file)
        
    #data_list = get_samples(pair_list, sample_size=64)


    #Do a 33%Test/66%Train split
#     print(len(data_list[0]))
#     print(len(normal_list[0]))
#    # print(label_list)
#     File = open("Raccoon" + str(0) + ".txt" ,"w")
#     for file_num in range(len(data_list)):
#         #check if it's a new directory, if so, write into a new file
#         print(file_num)
#         print(dirs[file_num])
#         if dirs[file_num] != dirs[file_num-1] and file_num != 0:
#             File.close()
#             print("Opening a new File")
#             File = open("Raccoon" + str(file_num) + ".txt" ,"w")

#         data = data_list[file_num]
#         normal = data_list[file_num]
#         for point in range(len(data)):
#             data_tuple = (data[point][0],data[point][1],data[point][2])
#             normal_tuple = (normal[point][0],normal[point][1],normal[point][2])

#             string = str(data_tuple[0]) + " " + str(data_tuple[1]) + " " + str(data_tuple[2]) + " " + str(normal_tuple[0]) + " " + str(normal_tuple[1]) + " " + str(normal_tuple[2]) + " " + str(label_list[file_num]) + '\n'
#             File.write(string)
    # data_train, data_test, normal_train, normal_test, label_train, label_test = train_test_split(data_list,normal_list, id_list, test_size=0.33, random_state=42)
    

    #print("Total files with sufficient data " + str(len(data_list)))

    # hdf_train = h5py.File("D:/Data/Animal_seg_train.h5", "w")
    # #dataset = f.create_dataset("data", data = point_data)
    # hdf_train.create_dataset("data", data = data_train)
    # hdf_train.create_dataset("label", data = label_train)
    # hdf_train.create_dataset("normal", data = normal_train)

    # hdf_train.flush()
    # hdf_train.close()

    # hdf_test = h5py.File("D:/Data/Animal_seg_test.h5", "w")
    # #dataset = f.create_dataset("data", data = point_data)
    # hdf_test.create_dataset("data", data = data_test)
    # hdf_test.create_dataset("label", data = label_test)
    # hdf_train.create_dataset("normal", data = normal_train)

    # hdf_test.flush()
    # hdf_test.close()


    #Creates/writes a file, and write the data in with create_dataset
    # with  as f:
    #     dataset = f.create_dataset("data", data = point_data)
    # with h5py.File = =    
    # dataset = f.create_dataset("label", data = point_id)
    #     print(dataset)


    #pid = f.create_dataset("pid", data = a_pid)
