import numpy as np
import pcl
import os
import joblib
import pickle
import codecs


from plyfile import PlyData, PlyElement
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler

def hexdump(src, length=16, sep='.'):
  """
  >>> print(hexdump('\x01\x02\x03\x04AAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBB'))
  00000000:  01 02 03 04 41 41 41 41  41 41 41 41 41 41 41 41  |....AAAAAAAAAAAA|
  00000010:  41 41 41 41 41 41 41 41  41 41 41 41 41 41 42 42  |AAAAAAAAAAAAAABB|
  00000020:  42 42 42 42 42 42 42 42  42 42 42 42 42 42 42 42  |BBBBBBBBBBBBBBBB|
  00000030:  42 42 42 42 42 42 42 42                           |BBBBBBBB|
  >>>
  >>> print(hexdump(b'\x01\x02\x03\x04AAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBB'))
  00000000:  01 02 03 04 41 41 41 41  41 41 41 41 41 41 41 41  |....AAAAAAAAAAAA|
  00000010:  41 41 41 41 41 41 41 41  41 41 41 41 41 41 42 42  |AAAAAAAAAAAAAABB|
  00000020:  42 42 42 42 42 42 42 42  42 42 42 42 42 42 42 42  |BBBBBBBBBBBBBBBB|
  00000030:  42 42 42 42 42 42 42 42                           |BBBBBBBB|
  """
  FILTER = ''.join([(len(repr(chr(x))) == 3) and chr(x) or sep for x in range(256)])
  lines = []
  for c in range(0, len(src), length):
    chars = src[c:c+length]
    hexstr = ' '.join(["%02x" % ord(x) for x in chars]) if type(chars) is str else ' '.join(['{:02x}'.format(x) for x in chars])
    if len(hexstr) > 24:
      hexstr = "%s %s" % (hexstr[:24], hexstr[24:])
    printable = ''.join(["%s" % ((ord(x) <= 127 and FILTER[ord(x)]) or sep) for x in chars]) if type(chars) is str else ''.join(['{}'.format((x <= 127 and FILTER[x]) or sep) for x in chars])
    lines.append("%08x:  %-*s  |%s|" % (c, length*3, hexstr, printable))
  return '\n'.join(lines)



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
        if label in filename.lower():
            return (label, labels.index(label))

    return -1
    #raise Exception("There exists no label with "+ filename +". Provide the label of the contained file through the folder name or filename.")

def calculate_normals(pcl_cloud):
    ne = pcl_cloud.make_NormalEstimation()
    tree = pcl_cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.5)
    cloud_normals = ne.compute()
    #print(ne)
    #print(cloud_normals)

    print(cloud_normals)

    return cloud_normals


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
    point_labels = []
    face_list = []

    data_lengths = []

    data = None
    face_length = 0
    
    for file_count in range(0, len(filenames)):

        #get the filename (purely for print)
        file_name = filenames[file_count][filenames[file_count].rfind('\\') + 1 :]

        plydata = PlyData.read(filenames[file_count])
        
        #collect all of the data from the ply [[x,y,z,r,g,b][...]]
        if (data != None):
            face_length += len(data)

        data = plydata['vertex'][:][:]
        
        #print("Part number of points:" +str(len(data)))
        #do not allow files that have less than min_points since this is our sample size.
        if len(data) >= min_points:

            #normal = calculate_normals(filenames[file_count])
            label = get_label(filenames[file_count])
          
            face = plydata['face'][:][:]

            #add data and label to their respective lists.
            if label != -1:
                print( file_name + ": (" + label[0].capitalize() + ") " + str(len(data)) + " points | " + str(len(face)) + " faces")
                label_val = [label[1]]
                data_lengths.append(len(data))
                data_list.extend(data)
                label_list.append(label[1])
                point_labels.extend(label_val*len(data))

                #Offset Face indicies to the length of the last set of data
                face = np.array(face.tolist())
                if file_count != 0:
                    face += face_length

                face_list.extend(face)

            else:
                print("Skipping " + str(filenames[file_count] + ": Label Does Not Exist"))
        else: 
            print("Skipping " + str(filenames[file_count] + ": Not Enough Data"))

    print("Number of Elements in Data: "+str(len(data_list)) + " Total Faces: " + str(len(face_list)) + "\n")
    #print(len(data_list))

    return data_list ,data_lengths, normal_list, label_list, point_labels, face_list


def scale_to_unit(vertices):
    '''
    Scales an arbitrarily sized object and uniformly scales it to 1.0,1.0,1.0

    Because Each point is scaled individually, the magnitide of each scale must be recorded to reverse
    NOTE: WE NEED TO CALCULATE THE DIFFERENCE OF THE SCALE TO REVERT AND MEASURE
    '''    
    
    #NOTE this old code works for scaling and centering, but does not do so uniformly across all points.


    scaler = MinMaxScaler()
    points = []
    #print(vertices)
    for point in vertices:
        #print(point)
        points.append((point[0],point[1],point[2]))
    
    scaler.fit(points)
    scaled_points = scaler.transform(points)

    '''NOTE: This is how you 
    scaled_points = scaled_points.tolist()
    scaled_points.append((20,20,20))
    scaled_points = scaler.inverse_transform(scaled_points)

    # we use joblib.dump(scalar, out_file) to save the scalar for later
    # we then use scalar = joblib.load(in_file) for inverting the scale

    '''

    return scaled_points, scaler
    

def write_regions_indices(label_name, label_id, points, point_range, output_filename):
    '''
    regions.txt format is:
    NOTE:Points index or location??
    <region_name> <point_1> <point_2> ... <point_n> <region_id>

    '''

    print("Writing " + output_filename)
    #append to file, since multiple regions exist. NOTE: this means should remove file if run multiple times.
    regions_file = open(output_filename, 'a')

    #write region_name
    regions_file.write((label_name + " "))

    #NOTE: depending on how the points are required, maybe it's just point index and not point?
    #write region_points

    for point in range(point_range[0], point_range[1]):
        #regions_file.write(str(point[0]) +" "+ str(point[1]) +" "+ str(point[2]) + " ")
        regions_file.write(str(point)+ " ")
    
    regions_file.write(str(label_id) + '\n')
    
def write_regions_points(label_name, label_id, points, output_filename):
    '''
    regions.txt format is:
    NOTE:Points index or location??
    <region_name> <point_1> <point_2> ... <point_n> <region_id>

    '''

    print("Writing " + output_filename)
    #append to file, since multiple regions exist. NOTE: this means should remove file if run multiple times.
    regions_file = open(output_filename, 'a')

    #write region_name
    regions_file.write((label_name + " "))

    #NOTE: depending on how the points are required, maybe it's just point index and not point?
    #write region_points

    for point in points:
        regions_file.write(str(point[0]) +" "+ str(point[1]) +" "+ str(point[2]) + " ")
        #regions_file.write(str(point)+ " ")
    
    regions_file.write(str(label_id) + '\n')



def write_mesh_off(vertices, normals, labels, point_labels, faces, output_filename):
    '''
    Writes a mesh point cloud into .off (Object File Format)

    Format:

        OFF #File type (descriptor?)
        numvertices numfaces numedges #numedges may remain 0
        p1
        p2
        p3
        face1
        face2
        face3
    '''
    print("Writing " +  output_filename)
    colours = []

    if len(vertices) == 0:
        return

    off_file = open(output_filename, 'w')

    #write header information (Refer to documentation)
    off_file.write("OFF\n")
    off_file.write(str(len(vertices)) + " " + str(len(faces)) + " " + '0'+'\n') #numvert numface numedge=0
    
    #collect point colours for future face colours
    for point in vertices:
        try:
            colours.append((point[3],point[4],point[5])) # append point colours to a list
        except: 
            colours.append((255,255,255)) #set to white
    vertices, scaler = scale_to_unit(vertices)
    #NOTE this is to provide inverse scaling later on
    joblib.dump(scaler, (output_filename[:-3] + "gz"))

    #write points (x,y,z) to file
    for point in vertices:
            off_file.write(str(point[0]) +" "+ str(point[1]) +" "+ str(point[2]) )
            off_file.write('\n')
    
    #write faces
    R_avg = 0
    G_avg = 0
    B_avg = 0

    #iterate through faces and print each index of vertecies used for face
    print(len(faces))
    for face in faces:
        
        for vertex_indices in face:

            off_file.write(str(len(vertex_indices)) + " ")

            for index in vertex_indices:
                off_file.write(str(index) + " ")

                #sum indices for average
                R_avg+=colours[index][0]
                G_avg+=colours[index][1]
                B_avg+=colours[index][2]

            #write colours: numcolourparam r g b (not RGBA, that would make numcolourparam=4)
            off_file.write(str(int(R_avg/len(vertex_indices))) +" "+ str(int(G_avg/len(vertex_indices))) +" "+ str(int(B_avg/len(vertex_indices))))
            off_file.write('\n')
            
            #reset sums
            R_avg = 0
            G_avg = 0
            B_avg = 0
    


def write_labels_gt(filenames, output_filename):

    print("Writing " + output_filename)
    output_file = open(output_filename, 'w')

    data_list, data_lengths, normal_list, label_list, point_labels, face_list = extract_ply_data(filenames,min_points=1)

    label_string =""

    for label in point_labels:
        label_string += str(label)
        label_string += '\n'


    #print(label_string)

    for label in label_list:
        output_file.write(label_string)

# pickled = codecs.encode(pickle.dumps(labels), "base64").decode()
# print(pickled)
#pickle.dump(labels, open( output_filename, "wb" ))


if __name__ == "__main__":
    print("HELLO")
    #DATA_DIR = "D:/Data/ALL_DATA/Seg_Raccoon/Left_Right_Perspective_Seg/Both_Seg/Segmented_L"
    #OUT_DIR = "D:/Data/ALL_DATA/Seg_Raccoon/Left_Right_Perspective_Seg/Both_Seg"
    DATA_DIR = "D:/Data/ALL_DATA/Full_Raccoon"
    OUT_DIR = "D:/Data/ALL_DATA/Full_Raccoon/Full_Raccoon_Exported"
    All_Files, dirs = collect_files(DATA_DIR,file_type="ply")
    file_count = 0
    for directory in dirs:
        if file_count >= 1:
            break
        print (directory)
        
            #collects all files in the directory of the raccoon. All files contain one entire scan of a raccoon, split up into labelled categories.
        filenames, dirs = collect_files(directory,file_type="ply")
        if(filenames):

                #get the folder name (purely for print)
            first_file = filenames[0]
            folder_name = first_file[first_file.rfind('\\',0,first_file.rfind('\\')) + 1 :first_file.rfind('\\')]
            print("="*15+folder_name+"="*15 )

            data_list, data_lengths, normal_list, label_list, point_labels, face_list = extract_ply_data(filenames,min_points=1)

            first_file = filenames[0]
            new_filename = first_file[first_file.rfind('\\',0,first_file.rfind('\\')) + 1 :first_file.rfind('\\')] #grab the name of the last folder
            #for data in data_list:
            write_mesh_off(data_list, normal_list, label_list, point_labels, face_list, (OUT_DIR + "/mesh_labels/off/" + new_filename + ".off") )
            write_labels_gt(filenames,(OUT_DIR + "/mesh_labels/gt/" + new_filename + ".seg"))
            
            #print(data_list)
            #labels = []
            regions_i_filename = (OUT_DIR + "/mesh_labels/" + new_filename+ "_iregions.txt")
            regions_p_filename =(OUT_DIR + "/mesh_labels/" + new_filename+ "_pregions.txt") 
            if os.path.isfile(regions_i_filename) or os.path.isfile(regions_p_filename):
                try:
                    os.remove(regions_i_filename)
                    print("Removing "+ regions_i_filename)
                except:
                    pass
                try:
                    print("Removing" + regions_p_filename)
                    os.remove(regions_p_filename)
                except:
                    pass

            point_range = (0,data_lengths[0])
            current_index = 0
            total_points = 0
            for filename in filenames:
                print(point_range, end = "| ")
                label = get_label(filename)
                print(label, end=": ")

                write_regions_indices(label[0], label[1], data_list, point_range, regions_i_filename )
                write_regions_points(label[0], label[1], data_list, regions_p_filename)
            
                total_points += data_lengths[current_index]
                print(total_points)
                
                current_index += 1
                if current_index < len(filenames):
                    point_range = (total_points, total_points + data_lengths[current_index])
                
                

            
            
            file_count += 1

