'''
Split traning and testing data given a image folder
'''
import os

def Push(folder, filenames, file, class_label):
    for name in [os.path.join(folder, name) for name in filenames]:
        s = name + ' ' + str(class_label) + '\n'
        file.write(s)
        
def SubFolder2Files(root, sub_folder, file_tr, file_val, class_label):
    path = os.path.join(root, sub_folder)
    imgs = os.listdir(path)
    if len(imgs)>2:
        Push(sub_folder, imgs[:1], file_val, class_label)
        Push(sub_folder, imgs[1:], file_tr, class_label)
    else:
        Push(sub_folder, imgs, file_tr, class_label)

if __name__=='__main__':
    path_source = 'F:\\FV_TMP\\Data\\Raw'
    path_list_tr = 'F:\\FV_TMP\\Data\\list_tr.txt'
    path_list_val = 'F:\\FV_TMP\\Data\\list_val.txt'
    
    file_tr = open(path_list_tr, 'a')
    file_val = open(path_list_val, 'a')
    persons = os.listdir(path_source)
    for label, person in enumerate(persons):
        SubFolder2Files(path_source, person, file_tr, file_val, label)
    file_tr.close()
    file_val.close()
    print('finished')
    