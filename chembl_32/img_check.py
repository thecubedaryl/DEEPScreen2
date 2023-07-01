import os
import cv2


deepscreen_version = "2.2"
target_id = "CHEMBL4282"

training_dataset_path = "/home/hayriye/DEEPScreen{}/training_files/target_training_datasets/{}/{}/".format(deepscreen_version, target_id, "imgs")
print(training_dataset_path)

#print(os.listdir(training_dataset_path))

comp_id_list = [f for f in os.listdir(training_dataset_path)]
print(comp_id_list[:10])

#count = 0
for comp_id in comp_id_list:
	img_path = os.path.join(training_dataset_path, "{}".format(comp_id))
	img_arr = cv2.imread(img_path)
	try:
		shape = img_arr.shape
		#print(img_arr.shape)
		if shape == (300, 300, 3):
			pass
			#print("YES!")
		else:
			print(comp_id, "is having shape: {}".format(shape))
	except:
		print("shape error {}".format(comp_id))
	#if count >= 10:
	#	break
	#count+=1
