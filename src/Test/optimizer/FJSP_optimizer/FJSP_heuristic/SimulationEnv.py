import simpy

from Test.optimizer.FJSP_optimizer.FJSP_heuristic.Job import JobShop
from typing import Optional, List, Dict

class SimulationEnv:
    """
    Main scheduling_environment class for the an online job shop
    """

    def __init__(self, online_arrivals: bool):
        self.simulator = simpy.Environment()
        self.JobShop = JobShop()
        self.online_arrivals = online_arrivals
        self.machine_resources = []
        self.processed_operations = set()

        # Parameters related to online job arrivals
        self.inter_arrival_time: Optional[int] = None
        self.min_nr_operations_per_job = None
        self.max_nr_operations_per_job = None
        self.min_duration_per_operation = None
        self.max_duration_per_operation = None

    def add_machine_resources(self) -> None:
        """Add a machine to the environment."""
        self.machine_resources.append(simpy.Resource(self.simulator, capacity=1))

    def perform_operation(self, operation, machine):
        """Perform operation on the machine (block resource for certain amount of time)"""
        if machine.machine_id in operation.processing_times:
            with self.machine_resources[machine.machine_id].request() as req:
                yield req
                start_time = self.simulator.now
                processing_time = operation.processing_times[machine.machine_id]
                # print('scheduled job:', operation.job_id, 'operation:', operation.operation_id, 'at', start_time, 'taking', processing_time)
                machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time=0)
                yield self.simulator.timeout(processing_time - 0.0001)
                # print('machine', machine.machine_id, 'released at time', simulationEnv.simulator.now)

                self.processed_operations.add(operation)

    def get_earliest_end_time_machines(simulationEnv, operation):
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
