import pandas as pd 
import glob
# All files ending with .txt with depth of 2 folder
data = glob.glob(r"data/*/*/*.*") 

print(data)



df = pd.DataFrame(data, columns= ['file_name'])
df['file'] = df['file_name'].str[:-15]

df_files =df[df['file_name'].str.contains("_P.jpg")]

df1 = df.head(10)

for index, row in df1.iterrows():
   print (row['file'], '   -  ', row['file_name'])