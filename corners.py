import numpy as np

def corners(corner_list):
	a = np.array(corner_list)
	
	ind = np.lexsort((a[:,1], a[:,0]))
	
	b = a[ind].tolist()
	
	result = [[],[],[],[]]
	
	if b[0][1] > b[1][1]:
		result[0] = b[0]
		result[3] = b[1]
	else:
		result[0] = b[1]
		result[3] = b[0]
		
	if b[2][1] > b[3][1]:
		result[1] = b[2]
		result[2] = b[3]
	else:
		result[1] = b[3]
		result[2] = b[2]
	
	return result
	
print corners([[465, 159], [443, 488], [65, 437], [102, 212]])
