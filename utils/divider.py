# coding=utf-8
count = 0
thefile = open('../data/train_labeled_nbg', 'rb')
while True:
    buffer = thefile.read(8192 * 1024)
    if not buffer:
        break
    count += buffer.count('\n')
print count
thefile.close()

read = open('../data/train_labeled_nbg', 'r')
writefile_train = open('../data/train_0.8', 'w+')
writefile_test = open('../data/val_0.2', 'w+')
write_count = 0

while write_count < 0.8 * count:
    line = read.readline()
    writefile_train.writelines(line)
    write_count += 1
while write_count <= count:
    line = read.readline()
    writefile_test.writelines(line)
    write_count += 1

read.close()
writefile_test.close()
writefile_train.close()
