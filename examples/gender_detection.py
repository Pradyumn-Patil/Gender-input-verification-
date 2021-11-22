
import cv2
import cvlib as cv
import sys
import numpy as np

import pandas as pd 
import glob
# All files ending with .txt with depth of 2 folder

candidate_list = pd.read_excel("ViewEnrollmentData_CutOFFDate.xlsx")
cand_list = candidate_list[['ApplicationID', 'Name','Gender','InstituteCodeAdmitted']]

print(list(candidate_list.columns))

data = glob.glob(r"data/*/*.*") 
df = pd.DataFrame(data, columns= ['file_name'])
df =df[df['file_name'].str.contains("_P.jpg")]
df['Gender']= -1
df['confidence'] = -1 
df['ApplicationID']=''

for ind in df.index:
    path = df['file_name'][ind]
    print(path)
    sstring_end = path[10:]
    AppID= sstring_end[:11]
    df['ApplicationID'][ind] = AppID
    print(path, ' - ', AppID)

    img = cv2.imread(path)
    
    # apply face detection
    face, conf = cv.detect_face(img)
    
    padding = 20
    
    # loop through detected faces
    for f in face:
        try:    
            (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
            (endX,endY) = min(img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
            
            # draw rectangle over face
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
        
            face_crop = np.copy(img[startY:endY, startX:endX])
        
            # apply gender detection
            (label, confidence) = cv.detect_gender(face_crop)
        
           
            #print(confidence)
            #print(label)
        
            idx = np.argmax(confidence)
            label = label[idx]
        
            df['Gender'][ind] = label
            df['confidence'][ind] = confidence[idx] * 100
    
            label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
    
            #print(f' Application ID {path} {label[0]} ')
        
            Y = startY - 10 if startY - 10 > 10 else startY + 10
        
            cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        except:
            print("An exception occurred")
            

cand_data_matching =  cand_list.merge(df,how='left',on='ApplicationID')
detection_data_matching =  df.merge(cand_list,how='left',on='ApplicationID')

with pd.ExcelWriter('output.xlsx') as writer:  
    df.to_excel(writer, sheet_name='Detection')
    cand_list.to_excel(writer, sheet_name='Registtered')
    cand_data_matching.to_excel(writer, sheet_name='candidate matching')      
    detection_data_matching.to_excel(writer, sheet_name=' detection matching') 
# cv2.imshow("gender detection", img)
# cv2.waitKey()


# cv2.imwrite("gender_detection.jpg", img)


# cv2.destroyAllWindows()
