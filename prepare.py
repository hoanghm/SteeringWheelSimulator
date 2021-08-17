import os

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_dataset = os.path.join(current_dir, 'dataset')

def rename():
	id = 0
	for f in os.listdir(path_to_dataset):
		if id % 30 == 0:
				print('renaming id={}'.format(id))
		if f.endswith('.jpg'):
			name = str(id).zfill(3)
			os.rename(path_to_dataset + '/' + f, path_to_dataset + '/' + name + '.jpg')
			id += 1
			
def create_train_and_test_files():
	images_paths = []
	for f in os.listdir(path_to_dataset):
		if f.endswith('.jpg'):
			path = os.path.join(path_to_dataset, f)
			images_paths.append(path + '\n')

	train_end_idx = int(len(images_paths)*0.9)
	with open(path_to_dataset + '/train.txt', 'w') as train_file:
		for path in images_paths[:train_end_idx]:
			train_file.write(path)
	with open(path_to_dataset + '/test.txt', 'w') as test_file:
		for path in images_paths[train_end_idx:]:
			test_file.write(path)

def create_data_and_name_files():

	# create classes.names
	num_classes = 0
	with open(path_to_dataset + '/' + 'classes.names', 'w') as names, \
		open(path_to_dataset + '/' + 'classes.txt', 'r') as txt:
		for line in txt:
			names.write(line)  # Copying all info from file txt to names
			num_classes += 1

	with open(path_to_dataset + '/' + 'drivingWheel.data', 'w') as data:
		data.write('classes = ' + str(num_classes) + '\n')
		data.write('train = ' + path_to_dataset + '/' + 'train.txt' + '\n')
		data.write('valid = ' + path_to_dataset + '/' + 'test.txt' + '\n')
		data.write('names = ' + path_to_dataset + '/' + 'classes.names' + '\n')
		data.write('backup = backup')

create_data_and_name_files()
