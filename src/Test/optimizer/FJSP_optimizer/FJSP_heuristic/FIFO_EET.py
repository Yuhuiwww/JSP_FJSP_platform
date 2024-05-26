class FIFO_EET:
    def __init__(self, config,simulationEnv):
        self.config = config
        self.simulationEnv = simulationEnv

    def get_op(self,machine,operationProcessing_times,operation):
        earliest_end_time_machines = self.simulationEnv.get_earliest_end_time_machines(operation)
        if machine.machine_id in earliest_end_time_machines:
            operationProcessing_times[operation] = operation.job_id


