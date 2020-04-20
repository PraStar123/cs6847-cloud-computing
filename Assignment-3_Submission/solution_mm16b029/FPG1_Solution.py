from pyspark.mllib.fpm import FPGrowth
from pyspark import SparkContext
import pydoop.hdfs as hdfs
import DataCleaner
import os

def fpgrowth_example(input_path, minSupport, numPartitions):
	sc = SparkContext(appName="FPGrowth")
	data = sc.textFile(input_path)
	transactions = data.map(lambda line: line.strip().split(' '))
	model = FPGrowth.train(transactions, minSupport, numPartitions)
	result = model.freqItemsets().collect()
	return result


		
if __name__ == "__main__":

	base_path = 'solution_mm16b029/'
	file_path = 'solution_mm16b029/FP_Part-1.csv'
	input_path = base_path + 'FP-1.txt'

	try: 
		hdfs.get('/data_mm16b029/FP_Part-1.csv', file_path)
	except IOError:
		pass

	DataCleaner.pre_processing(file_path, input_path)

	minSupport = 0.04
	numPartitions = 4	

	result = fpgrowth_example(input_path, minSupport, numPartitions)
	freq_pairs = []
	for row in result:
		if len(row[0]) == 2:
			freq_pairs.append([row[1], row[0]])

	freq_pairs.sort()
	top_five = freq_pairs[:5]


	output_path = base_path + "FP_out1.txt"
	with open(output_path, 'w') as writer:
		for pairs in top_five:
			writer.write("Frequency: " + str(pairs[0]) + ", Item Pairs = " + str(pairs[1]) + "\n")
	writer.close()

	hdfs.put(output_path, '/data_mm16b029/FP_out1.txt')
	os.remove(output_path)

	

