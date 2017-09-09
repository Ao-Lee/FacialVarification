'''
Decode MS-Celeb-1M tsv data to subfolders containing images
'''

import base64
import os
    
def WriteImage(filename, img_encoded):
    img_binary=base64.b64decode(img_encoded)
    with open(filename,"wb") as file:
        file.write(img_binary)
        
def Unpack(path_csv, path_target):
    file = open(path_csv, 'r', encoding='utf8')
    counter = 0
    for line in file:
        contents = line.split('\t')
        # 0: Freebase MID (unique key for each entity)
        # 1: Image Search Rank
        # 2: Image URL
        # 3: Page URL
        # 4: FaceID
        # 5: Face Bounding Box
        # 6: Image Data
        MID = contents[0]
        image_search_rank = contents[1]
        faceID = contents[4]
        image_data = contents[6]               
        img_dir=os.path.join(path_target,MID)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_name="%s-%s"%(image_search_rank,faceID) + ".jpg"
        WriteImage(os.path.join(img_dir, img_name), image_data)
        counter += 1
        if counter % 5000 == 0:
            print('image {} is being processed'.format(counter))
    file.close()
    print("all finished")
    
if __name__=='__main__': 
    path_csv = 'Sample.tsv'
    path_csv = 'D:\\MS-Celeb-1M\\训练数据\\tsv\\FaceImageCroppedWithOutAlignment.tsv'
    path_target = 'E:\\MS-Celeb-1M\\imgs'
    Unpack(path_csv, path_target)
    
    
    
    
    