class MOR_EET:
    def __init__(self, config,simulationEnv):
        self.config = config
        self.simulationEnv = simulationEnv

    def get_earliest_end_time_machines(self,simulationEnv, operation):
        """get earliest end time of machines, when operation would be scheduled on it"""
        finish_times = {}
        machine_options = operation.processing_times.keys()
        for machine_option in machine_options:
            machine = simulationEnv.JobShop.get_machine(machine_option)
            if machine.scheduled_operations == []:
                finish_times[machine_option] = simulationEnv.simulator.now + operation.processing_times[machine_option]
            else:
                finish_times[machine_option] = operation.processing_times[machine_option] + max(
                    [operation['end_time'] for operation in machine._processed_operations])
        earliest_end_time = min(finish_times.values())  # Find the minimum value in the dictionary
        return [key for key, value in finish_times.items() if value == earliest_end_time]

    def get_operations_remaining(self,simulationEnv, operation):
        """get remaining operations of the job"""
        return len(
            [operation for operation in operation.job.operations if
             operation not in simulationEnv.processed_operations])
    def get_op(self,machine,operationProcessing_times,operation):
        earliest_end_time_machines = self.get_earliest_end_time_machines(self.simulationEnv, operation)
        if machine.machine_id in earliest_end_time_machines:
            operationProcessing_times[operation] = self.get_operations_remaining(self.simulationEnv, operation)
