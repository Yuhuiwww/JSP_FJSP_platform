import numpy as np
import os
import re
from torch.distributions.categorical import Categorical
import sys
# 加载数据
def load_data(config):
    N_JOBS_P = config.Pn_j  # 工件的数量
    N_MACHINES_P = config.Pn_m  # 机床的数量
    TEST_DATA =config.test_datas+config.test_datas_type
    # 加载数据
    dataLoaded = np.load(str(TEST_DATA) +str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.npy')
    dataset = []
    #
    for i in range(dataLoaded.shape[0]):
        # # for i in range(1):
        dataset.append((dataLoaded[i][0], dataLoaded[i][1]))
    return dataset

def FJSP_load_gurobi(file):

    machine_allocations = {}
    operations_times = {}
    sdsts = {}
    numonJobs = []
    total_op_nr = 0
    # 加载数据
    with open(file, 'r') as f:
        # Extract header data
        number_operations, number_machines, _ = map(float, f.readline().split())
        number_jobs = int(number_operations)
        number_machines = int(number_machines)

        # Process job operations data
        for i in range(number_jobs):
            operation_data = list(map(int, f.readline().split()))
            operation_num = operation_data[0]
            numonJobs.append(operation_num)

            index, operation_id = 1, 0
            while index < len(operation_data):
                total_op_nr += 1

                # Extract machine and processing time data for each operation
                o_num = operation_data[index]
                job_machines = operation_data[index + 1:index + 1 + o_num * 2:2]
                job_processingtime = operation_data[index + 2:index + 2 + o_num * 2:2]
                machine_allocations[(i + 1, operation_id + 1)] = job_machines

                # Save processing times
                for l, machine in enumerate(job_machines):
                    operations_times[(i + 1, operation_id + 1, machine)] = job_processingtime[l]

                operation_id += 1
                index += o_num * 2 + 1

    # Calculate derived values
    jobs = list(range(1, number_jobs + 1))
    machines = list(range(1, number_machines + 1))
    operations_per_job = {j: list(range(1, numonJobs[j - 1] + 1)) for j in jobs}
    # calculate upper bound
    largeM = sum(
        max(operations_times[(job, op, l)] for l in machine_allocations[(job, op)]) for job in jobs for op in
        operations_per_job[job]
    )

    # Return parsed data
    return {
        'number_jobs': number_jobs,
        'number_machines': number_machines,
        'jobs': jobs,
        'machines': machines,
        'operations_per_job': operations_per_job,
        'machine_allocations': machine_allocations,
        'operations_times': operations_times,
        'largeM': largeM,  #
        "sdsts": sdsts
    }

def load_dataDAN(config):
    TEST_DATA = config.test_datas +'/'
    N_JOBS_P = config.Pn_j  # 工件的数量
    N_MACHINES_P = config.Pn_m  # 机床的数量

    # 路径problem/FJSP_test_datas/SD1/10x5
    filesPath = [str(TEST_DATA) +'/'+config.test_datas_type+'/'+ str(N_JOBS_P) + 'x' + str(N_MACHINES_P)]
    data_list = []
    for data_name in filesPath:
        data_path = f'./{data_name}'
        data_list.append((load_data_FJSPalgorithm_files(config,data_path), str(N_JOBS_P) + 'x' + str(N_MACHINES_P)))
    config.num_ins = len(data_list[0][0][0])
    return data_list,config

# 查找最近的文件
# def find_nearest_file(directory, target_file_type, target_file):
#     files = os.listdir(directory)
#     files.sort()
#     finded_file = []
#     for file in files:
#         filename = os.path.basename(file)
#         if filename.startswith(target_file_type):
#             match = re.search(r"(\d+)x(\d+)", file)
#             first_number = match.group(1)
#             second_number = match.group(2)
#             target_match = re.search(r"(\d+)x(\d+)", target_file)
#             target_first_number = target_match.group(1)
#             target_second_number = target_match.group(2)
#             num = (int(first_number)-int(target_first_number))+(int(second_number)-int(target_second_number))
#             if num==0:
#                 return filename
#             finded_file.append((filename, abs(num)))
#     if len(finded_file)==0:
#         print(directory+"路径下没有找到匹配"+target_file_type+'的文件！！！！！！！')
#         sys.exit()
#     min_distance_entry = min(finded_file, key=lambda x: x[1])
#     (min_filename, min_num) = min_distance_entry
#     return min_filename
def find_nearest_file(directory, target_file):
    files = os.listdir(directory)
    files.sort()
    finded_file = []
    for file in files:
        filename = os.path.basename(file)
        match = re.search(r"(\d+)x(\d+)", file)
        first_number = match.group(1)
        second_number = match.group(2)
        target_match = re.search(r"(\d+)x(\d+)", target_file)
        target_first_number = target_match.group(1)
        target_second_number = target_match.group(2)
        num = (int(first_number) - int(target_first_number)) + (int(second_number) - int(target_second_number))
        if num == 0:
            return filename
        finded_file.append((filename, abs(num)))
    # if len(finded_file)==0:
    #     print(directory+"路径下没有找到匹配"+target_file_type+'的文件！！！！！！！')
    #     sys.exit()
    min_distance_entry = min(finded_file, key=lambda x: x[1])
    (min_filename, min_num) = min_distance_entry
    return min_filename

def load_data_from_files(directory):
    """
        load all files within the specified directory
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        return [], []

    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
        for f in files:
            # print(f)
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return dataset_job_length, dataset_op_pt


def load_data_FJSPalgorithm_files(config,directory):
    """
        load all files within the specified directory
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        return [], []

    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
        config.File_GAN=files
        for f in files:
            # print(f)
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return dataset_job_length, dataset_op_pt

def text_to_matrix(text):
    """
            Convert text form of the data into matrix form
    :param text: the standard text form of the instance
    :return:  the matrix form of the instance
            job_length: the number of operations in each job (shape [J])
            op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    """
    n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
    n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

    job_length = np.zeros(n_j, dtype='int32')
    op_pt = []

    for i in range(n_j):
        content = np.array([int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
        job_length[i] = content[0]

        idx = 1
        for j in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype='int32')
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt

            idx = next_idx
            op_pt.append(op_pt_row)

    op_pt = np.array(op_pt)

    return job_length, op_pt

def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str
def sample_action(p):
    """
         sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def SD2_instance_generator(config):
        """
        :param config: a package of parameters
        :return: a fjsp instance generated by SD2, with
            job_length : the number of operations in each job (shape [J])
            op_pt: the processing time matrix with shape [N, M],
                    where op_pt[i,j] is the processing time of the ith operation
                    on the jth machine or 0 if $O_i$ can not process on $M_j$
            op_per_mch : the average number of compatible machines of each operation
        """
        n_j = config.Pn_j
        n_m = config.Pn_m
        if config.op_per_job == 0:
            op_per_job = n_m
        else:
            op_per_job = config.op_per_job

        low = config.low
        high = config.high
        data_suffix = config.data_suffix

        op_per_mch_min = 1
        if data_suffix == "nf":
            op_per_mch_max = 1
        elif data_suffix == "mix":
            op_per_mch_max = n_m
        else:
            op_per_mch_min = config.op_per_mch_min
            op_per_mch_max = config.op_per_mch_max
        if op_per_mch_min < 1 or op_per_mch_max > n_m:
            print(f'Error from Instance Generation: [{op_per_mch_min},{op_per_mch_max}] '
                  f'with num_mch : {n_m}')
            sys.exit()

        n_op = int(n_j * op_per_job)
        job_length = np.full(shape=(n_j,), fill_value=op_per_job, dtype=int)
        op_use_mch = np.random.randint(low=op_per_mch_min, high=op_per_mch_max + 1,
                                       size=n_op)

        op_per_mch = np.mean(op_use_mch)
        op_pt = np.random.randint(low=low, high=high + 1, size=(n_op, n_m))

        for row in range(op_pt.shape[0]):
            mch_num = int(op_use_mch[row])
            if mch_num < n_m:
                inf_pos = np.random.choice(np.arange(0, n_m), n_m - mch_num, replace=False)
                op_pt[row][inf_pos] = 0

        return job_length, op_pt, op_per_mch

# def FJSP_Basicalgorithm_parseFile(path):
#     problem = None
#     with open(os.path.join(os.getcwd(), path), "r") as data:
#         number_jobs, number_machines = [int(element) for element in re.findall("\S+", data.readline())[0:2]]
#
#         jobs = []
#         for job_id, current_job in enumerate(data):
#             if job_id >= number_jobs:
#                 break
#
#             # Every line is a job
#             job = Job(job_id+1)
#
#             # First number in line represent number of tasks
#             id_task = 1
#
#             # Successive numbers denote numbers of possible Operation
#             operation = 1
#
#             current_job = re.findall('\S+', current_job)
#
#             while operation < len(current_job):
#                 # Number of operations
#                 task = Task(id_task, job)
#                 number_operations = int(current_job[operation])
#                 for current_tuple in range(0, number_operations):
#                     id_machine = current_job[operation + 1 + current_tuple*2]
#                     processing_time = current_job[operation + 2 + current_tuple*2]
#                     task.add_operation(Operation(int(id_machine), int(processing_time)))
#                 # Next operation has a certain offset
#                 operation += number_operations*2 + 1
#
#                 # Next Task
#                 id_task += 1
#                 job.add_task(task)
#
#             jobs.append(job)
#
#         machines = []
#         for i in range(1, number_machines+1):
#             machines.append(Machine(i))
#
#         problem = Problem(jobs, machines)
#     return problem
