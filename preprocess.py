
import numpy as np
import matplotlib.pyplot as plt

# filename = 'test_x.csv'
filename = 'train_labeled'
threshold = 190
with open(filename,'r') as file_read, open(filename+'_nbg', 'w+') as file_write:
        while 1:
            line = file_read.readline()
            if not line:
                break
            (label, line) = line.split(",")[0], line.split(",")[1]
            line_vector = map(int, line.strip().split(' '))
            for index, item in enumerate(line_vector):
                if item > threshold:
                    line_vector[index] = 255
                else:
                    line_vector[index] = 0

            newline = ' '.join(str(item) for item in line_vector)
            file_write.writelines(label + ',' + newline+'\n')

# with open(filename, 'r') as file_read, open(filename + '_nbg', 'w+') as file_write:
#     while 1:
#         line = file_read.readline()
#         if not line:
#             break
#         line = line.replace(',', '').replace('\n', '')
#         line_vector = map(int, line.strip().split(' '))
#         for index,item in enumerate(line_vector):
#             if item > threshold:
#                 line_vector[index] = 255
#             else:
#                 line_vector[index] = 0
#
#
#         newline = ' ,'.join(str(item) for item in line_vector)
#         file_write.writelines(newline+'\n')
