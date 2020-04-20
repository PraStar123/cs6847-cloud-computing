from csv import reader
import csv
import pandas
import pydoop.hdfs as hdfs
import os


# Input to Spark program 
def pre_processing(file_path, input_path):
	# File preprocessing
	input_fpgrowth = []
	with open(file_path, 'r') as read_obj:
		csv_reader = reader(read_obj)
		for row in csv_reader:
			data_row = []
			for element in row:
				element.strip()
				el = element.replace(" ", "_")
				if el in data_row:
					continue
				data_row.append(el)
			input_fpgrowth.append(data_row)

	os.remove(file_path)

	# Prepare input file to 'fpgrowth_example.py'
	with open(input_path, 'w') as file_writer:
		for row in input_fpgrowth:
			for el in row:
				file_writer.write('%s ' % el)
			file_writer.write('\n')
	file_writer.close()

def data_cleaning(file_path):
	cleaned_data = []
	with open(file_path, 'r') as read_obj:
		csv_reader = reader(read_obj)
		header = next(csv_reader)
		transaction = []
		invoice_no = []
		count = 0
		for row in csv_reader:
			if not count:
				invoice_no = row[0]
			if row[0] == invoice_no:
				transaction.append(row[2])
			else:
				cleaned_data.append(transaction)
				transaction = []
				transaction.append(row[2])
				invoice_no = row[0]
			count += 1

	output_path = 'solution_mm16b029/formatted.csv' 
	with open(output_path, 'w') as file_writer:
		csvWriter = csv.writer(file_writer, delimiter=',', lineterminator='\n')
		csvWriter.writerows(cleaned_data)
	file_writer.close()

	hdfs.put(output_path, '/data_mm16b029/formatted.csv')
	os.remove(output_path)


