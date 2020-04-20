from pyspark import SparkContext
import pydoop.hdfs as hdfs
import ALS
import os

def als_run_tuning(base_path):
    als_train_split = 0.8	# default train-test split
    max_iters = [i for i in range(1,21)]
    reg_params = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    log_iters = []

    for reg_param in reg_params:	
	    for iteration in max_iters:
	    	rmse = ALS.als_example(base_path, als_train_split, iteration, reg_param)
	    	log_iters.append([rmse, reg_param, iteration])


    output_path_iters = base_path + 'ALS_out.txt' 
    with open(output_path_iters, 'w') as writer:
    	for i in range(len(log_iters)):
    		writer.write("For regularization parameter: " + str(log_iters[i][1]) + " and No. of iteration: " + str(log_iters[i][2]) + ", RMSE: " + str(log_iters[i][0]) + "\n")
   
    writer.close()

    hdfs.put(output_path_iters, '/data_mm16b029/ALS_out.txt')
    os.remove(output_path_iters)
    return min(log_iters)



def als_run_train_split(base_path, iteration, reg_param):
	log_rmse = []
	trains_split_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	for train_split in trains_split_range:
		rmse = ALS.als_example(base_path, train_split, iteration, reg_param)
		log_rmse.append([rmse, train_split])

	output_path_train_split = base_path + 'ALS_out_train_split.txt'
	with open(output_path_train_split, 'w') as writer:
		for i in range(len(log_rmse)):
			writer.write("Train-Split: " + str(log_rmse[i][1]) + ", RMSE: " + str(log_rmse[i][0]) + "\n")
	writer.close()
	hdfs.put(output_path_train_split, '/data_mm16b029/ALS_out_train_split.txt')
	os.remove(output_path_train_split)
	

if __name__ == "__main__":
    sc = SparkContext(appName="Solution_MM16B029")

    base_path = 'solution_mm16b029/'

    min_params = als_run_tuning(base_path)
    min_iter = min_params[2]
    min_reg_param = min_params[1]

    als_run_train_split(base_path, min_iter, min_reg_param)



