class MWR_SPT:
    def __init__(self, config, simulationEnv):
        self.config = config
        self.simulationEnv = simulationEnv

    def get_work_remaining(self,simulationEnv, operation):
        """get amount of work remaining of the job"""
        operations_remaining = [operation for operation in operation.job.operations if
                                operation not in simulationEnv.processed_operations]
        return sum([sum(operation.processing_times.values()) / len(operation.processing_times) for operation in
                    operations_remaining])
    def get_op(self,machine,operationProcessing_times,operation):
        min_processing_time = min(operation.processing_times.values())
        min_keys = [key for key, value in operation.processing_times.items() if
                    value == min_processing_time]
        if machine.machine_id in min_keys:
            operationProcessing_times[operation] = self.get_work_remaining(self.simulationEnv, operation)
