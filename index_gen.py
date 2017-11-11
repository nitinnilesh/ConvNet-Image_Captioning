import sys
import os
import pandas as pd
import re



if __name__ == '__main__':

 	word_list = []
 	count  = 0
 	index = pd.DataFrame()


 	
	path = sys.argv[1]
	for root, dirs, files in os.walk(path):
		for f in files:
			
			if os.path.splitext(f)[0] == 'Flickr8k.token':
				count +=1
				print os.path.splitext(f)[0]
				with open(os.path.join(root, f), "r") as infile:
					f = infile.read()
					#f.replace('\t', ' ')
					f = f.replace(", ", "")
					f = f.replace('"', '')
					f = f.replace(";", "")
					f = f.replace(": ", "")
					f = f.replace("!", "")
					#f = re.sub("\S*\d\S*", "", f).strip()
					l = k = f.split()
					#l = [x for x in l if len(x)>5 and len(x)<11]
					l = set(l)
					print len(l)
					word_list = [w for w in l if w.isalpha()]
					print len(word_list)

					
					i=0
					for i in range(len(k)):
						print i
						if(i==2000):
							break
						if(not(k[i].isalpha())  and len(k[i])>19): 
							id = k[i]
							index.loc[k[i], 0] = ""
							
						else:
							index.loc[id,0] += k[i]+" " 

		
							


						


						

		with open('mt.csv', 'a') as f:
 			index.to_csv(f, sep='\t', encoding='utf-8')










	
	
