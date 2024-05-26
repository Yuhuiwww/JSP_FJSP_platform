from Problem.Basic_problem import Basic_Problem


class FJSP(Basic_Problem):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def cal_FJSP_objective(self,machine_operations):
        data = {}
        min_makespan = 0
        for idx, machine in enumerate(machine_operations):
            machine_name = "Machine-{}".format(idx + 1)
            operations = []
            for operation in machine:
                operations.append([operation[3], operation[3] + operation[1], operation[0]])
            data[machine_name] = operations
            for num in range(len(data[machine_name])):
                if (min_makespan < data[machine_name][num][1]):
                    min_makespan = data[machine_name][num][1]


        return min_makespan









