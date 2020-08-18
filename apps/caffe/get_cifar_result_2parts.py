import os, sys
import re

NUM_WORKER = 2
TIME_INTERVAL = 1000
RECORD_INTERVAL = 1000
TOTAL_ITERATION = 65000
NUM_RESULT = TOTAL_ITERATION / RECORD_INTERVAL
SKIP_TIME_NUM = ((RECORD_INTERVAL - TIME_INTERVAL) / TIME_INTERVAL) * NUM_WORKER
CONV_CHECK_NUM = 3
CONV_CHECK_THRESHOLD = 0.0
CONV_CHECK_VALUE = 0.0

result_read_time_arr = []
result_write_time_arr = []
result_compute_time_arr = []
result_total_time_arr = []
result_loss_arr = []
result_accu_arr = []
result_conv_accu = []
result_conv_time = []
result_conv_iter = []

cur_read_time = [0.0] * NUM_RESULT
cur_write_time = [0.0] * NUM_RESULT
cur_compute_time = [0.0] * NUM_RESULT
cur_loss = [0.0] * NUM_RESULT
cur_accu = [0.0] * NUM_RESULT

result_count = 0
worker_result_count = 0
time_result_count = 0

def add_result_and_reset():
    global cur_read_time, cur_write_time, cur_compute_time, cur_loss, cur_accu

    result_read_time_arr.append(cur_read_time)
    result_write_time_arr.append(cur_write_time)
    result_compute_time_arr.append(cur_compute_time)
    cur_total_time = [x + y + z for x, y, z in \
                      zip(cur_read_time, cur_write_time, cur_compute_time)]
    result_total_time_arr.append(cur_total_time)
    result_loss_arr.append(cur_loss)
    result_accu_arr.append(cur_accu)

    # Find out the converge accuracy and time    
    found_conv_point = False    
    for i in range(len(cur_accu) - CONV_CHECK_NUM):        
        if (CONV_CHECK_VALUE != 0.0):
            found_conv_point = False
            if (cur_accu[i] >= CONV_CHECK_VALUE):
                found_conv_point = True
                break
        else:
            found_conv_point = True
            for j in range(i + 1, i + CONV_CHECK_NUM + 1):
                if cur_accu[i] != 0:
                    if (((cur_accu[j] - cur_accu[i]) / cur_accu[i]) > CONV_CHECK_THRESHOLD):
                        found_conv_point = False
                        break
        if (found_conv_point):
            break

    if (found_conv_point):
        result_conv_accu.append(cur_accu[i + CONV_CHECK_NUM])
        result_conv_time.append(cur_total_time[i + CONV_CHECK_NUM])
        result_conv_iter.append((i + CONV_CHECK_NUM) * RECORD_INTERVAL)
    else:
        result_conv_accu.append(-1.0)
        result_conv_time.append(-1.0)
        result_conv_iter.append(-1.0)

    cur_read_time = [0.0] * NUM_RESULT
    cur_write_time = [0.0] * NUM_RESULT
    cur_compute_time = [0.0] * NUM_RESULT
    cur_loss = [0.0] * NUM_RESULT
    cur_accu = [0.0] * NUM_RESULT


if __name__ == '__main__': 
    result_file = dat_file = open(sys.argv[1])

    if (len(sys.argv) > 2):
        CONV_CHECK_VALUE = float(sys.argv[2]) * NUM_WORKER

    line = result_file.readline()

    read_time_re = re.compile("Read PS time:\s+([-+]?[\d\.\d]+[e]?[-+]?[\d]*)")
    write_time_re = re.compile("Write PS time:\s+([-+]?[\d\.\d]+[e]?[-+]?[\d]*)")
    compute_time_re = re.compile("Compute time:\s+([-+]?[\d\.\d]+[e]?[-+]?[\d]*)")

    accu_re = re.compile("#0: accuracy\s+=\s+([-+]?[\d\.\d]+[e]?[-+]?[\d]*)")

    loss_re = re.compile("Iteration " + str(RECORD_INTERVAL) + ", loss = ([-+]?[\d\.\d]+[e]?[-+]?[\d]*)")

    while line:
        time_match = read_time_re.search(line)
        if (time_match):
            if (time_result_count >= SKIP_TIME_NUM):
                #print "Read Time = " + time_match.group(1)
                cur_read_time[result_count % NUM_RESULT] += float(time_match.group(1))

        time_match = write_time_re.search(line)
        if (time_match):
            if (time_result_count >= SKIP_TIME_NUM):
                #print "Write Time = " + time_match.group(1)
                cur_write_time[result_count % NUM_RESULT] += float(time_match.group(1))

        time_match = compute_time_re.search(line)
        if (time_match):
            if (time_result_count >= SKIP_TIME_NUM):
                #print "Compute Time = " + time_match.group(1)
                cur_compute_time[result_count % NUM_RESULT] += float(time_match.group(1))
            time_result_count += 1

        accu_match = accu_re.search(line)
        if(accu_match):
            #print "Accuracy = " + accu_match.group(1)
            cur_accu[result_count % NUM_RESULT] += float(accu_match.group(1))

        loss_match = loss_re.search(line)
        if(loss_match):
            #print "Loss = " + loss_match.group(1)
            cur_loss[result_count % NUM_RESULT] += float(loss_match.group(1))
            worker_result_count += 1
            if (worker_result_count >= NUM_WORKER):
                worker_result_count = 0
                time_result_count = 0
                result_count += 1
                loss_re = re.compile("Iteration " + \
                              str(RECORD_INTERVAL * ((result_count % NUM_RESULT) + 1)) +\
                              ", loss = ([-+]?[\d\.\d]+[e]?[-+]?[\d]*)")
                if (result_count % NUM_RESULT == 0):
                    add_result_and_reset()
                    

        line = result_file.readline()

    if (result_count % NUM_RESULT != 0):
        add_result_and_reset()

    for i in range(len(result_read_time_arr)):
        out_str = ""
        for j in range(NUM_RESULT):
            out_str += str(result_read_time_arr[i][j] / NUM_WORKER) + " "
        #print out_str
        out_str = ""
        for j in range(NUM_RESULT):
            out_str += str(result_write_time_arr[i][j] / NUM_WORKER) + " "
        #print out_str
        out_str = ""
        for j in range(NUM_RESULT):
            out_str += str(result_compute_time_arr[i][j] / NUM_WORKER) + " "
        #print out_str
        out_str = ""
        for j in range(NUM_RESULT):
            out_str += str(result_total_time_arr[i][j] / NUM_WORKER) + " "
        #print out_str
        out_str = ""
        for j in range(NUM_RESULT):
            out_str += str(result_loss_arr[i][j] / NUM_WORKER) + " "
        #print out_str
        out_str = ""
        for j in range(NUM_RESULT):
            out_str += str(result_accu_arr[i][j] / NUM_WORKER) + " "
        #print out_str
        print(result_accu_arr[i][-2] / NUM_WORKER)
        #print result_conv_time[i] / NUM_WORKER
        #print result_conv_iter[i]
        #print result_conv_accu[i] / NUM_WORKER
 
