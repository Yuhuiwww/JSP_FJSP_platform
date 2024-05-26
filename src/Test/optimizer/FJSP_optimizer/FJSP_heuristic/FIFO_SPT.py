class FIFO_SPT:
    def __init__(self, config, simulationEnv):
        self.config = config
        self.simulationEnv = simulationEnv
    def get_op(self,machine,operationProcessing_times,operation):
        min_processing_time = min(operation.processing_times.values())
        min_keys = [key for key, value in operation.processing_times.items() if
                    value == min_processing_time]
        if machine.machine_id in min_keys:
            operationProcessing_times[operation] = operation.job_id
