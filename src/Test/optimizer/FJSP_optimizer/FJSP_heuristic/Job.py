from .operation import Operation
from collections import OrderedDict
from typing import List, Dict
class Operation:
    def __init__(self, job, job_id, operation_id):
        self._job = job
        self._job_id = job_id
        self._operation_id = operation_id
        self._processing_times = OrderedDict()
        self._predecessors: List = []
        self._scheduling_information = {}

    def reset(self):
        self._scheduling_information = {}

    def __str__(self):
        return f"Job {self.job_id}, Operation {self.operation_id}"

    @property
    def job(self):
        """Return the job object of the operation."""
        return self._job

    @property
    def job_id(self) -> int:
        """Return the job's id of the operation."""
        return self._job_id

    @property
    def operation_id(self) -> int:
        """Return the operation's id."""
        return self._operation_id

    @property
    def scheduling_information(self) -> Dict:
        """Return the scheduling information of the operation"""
        return self._scheduling_information

    @property
    def processing_times(self) -> dict:
        """Return a dictionary of machine ids and processing time durations."""
        return self._processing_times

    @property
    def scheduled_start_time(self) -> int:
        """Return the scheduled start time of the operation."""
        if 'start_time' in self._scheduling_information:
            return self._scheduling_information['start_time']
        return None

    @property
    def scheduled_end_time(self) -> int:
        """Return the scheduled end time of the operation."""
        if 'end_time' in self._scheduling_information:
            return self._scheduling_information['end_time']
        return None

    @property
    def scheduled_duration(self) -> int:
        """Return the scheduled duration of the operation."""
        return self._scheduled_duration

    @property
    def scheduled_machine(self) -> None:
        """Return the machine id that the operation is scheduled on."""
        if 'machine_id' in self._scheduling_information:
            return self._scheduling_information['machine_id']
        return None

    @property
    def predecessors(self) -> List:
        """Return the list of predecessor operations."""
        return self._predecessors

    @property
    def finishing_time_predecessors(self) -> int:
        """Return the finishing time of the latest predecessor."""
        if not self.predecessors:
            return 0
        end_times_predecessors = [operation.scheduled_end_time for operation in self.predecessors]
        return max(end_times_predecessors)

    def update_job_id(self, new_job_id: int) -> None:
        """Update the id of a job (used for assembly scheduling problems, with no pregiven job id)"""
        self._job_id = new_job_id

    def update_job(self, job) -> None:
        self._job = job

    def add_predecessors(self, predecessors: List) -> None:
        """Add a list of predecessor operations to the current operation."""
        self.predecessors.extend(predecessors)

    def add_operation_option(self, machine_id, duration) -> None:
        """Add an operation option to the current operation."""
        self._processing_times[machine_id] = duration

    def update_sequence_dependent_setup_times(self, start_time_setup, setup_duration):
        self._start_time_setup = start_time_setup
        self._setup_duration = setup_duration

    def add_operation_scheduling_information(self, machine_id: int, start_time: int, setup_time: int, duration) -> None:
        """Add scheduling information to the current operation."""
        self._scheduling_information = {
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': start_time + duration,
            'processing_time': duration,
            'start_setup': start_time - setup_time,
            'end_setup': start_time,
            'setup_time': setup_time}

class Job:
    def __init__(self, job_id: int):
        self._job_id: int = job_id
        self._operations: List[Operation] = []

    def add_operation(self, operation: Operation):
        """Add an operation to the job."""
        self._operations.append(operation)

    @property
    def nr_of_ops(self) -> int:
        """Return the number of jobs."""
        return len(self._operations)

    @property
    def operations(self) -> List[Operation]:
        """Return the list of operations."""
        return self._operations

    @property
    def job_id(self) -> int:
        """Return the job's id."""
        return self._job_id

    def get_operation(self, operation_id):
        """Return operation object with operation id."""
        for operation in self._operations:
            if operation.operation_id == operation_id:
                return operation

class Machine:
    def __init__(self, machine_id, machine_name=None):
        self._machine_id = machine_id
        self._machine_name = machine_name
        self._processed_operations = []

    def reset(self):
        self._processed_operations = []

    def __str__(self):
        return f"Machine {self._machine_id}, {len(self._processed_operations)} scheduled operations"

    @property
    def machine_id(self):
        """Return the machine's id."""
        return self._machine_id

    @property
    def machine_name(self):
        """Return the machine's name."""
        return self._machine_name

    @property
    def scheduled_operations(self) -> List[Operation]:
        """Return the list of scheduled operations on this machine."""
        sorted_operations = sorted(self._processed_operations, key=lambda op: op['start_time'])
        return [op['operation'] for op in sorted_operations]

    def add_operation_to_schedule_at_time(self, operation, start_time, processing_time, setup_time):
        """Scheduled an operations at a certain time."""

        operation.add_operation_scheduling_information(
            self.machine_id, start_time, setup_time, processing_time)

        self._processed_operations.append({
            'operation': operation,
            'start_time': start_time,
            'end_time': start_time + processing_time,
            'processing_time': processing_time,
            'start_setup': start_time - setup_time,
            'end_setup': start_time,
            'setup_time': setup_time
        })

    def add_operation_to_schedule_backfilling(self, operation: Operation, processing_time,
                                              sequence_dependent_setup_times):
        """Add an operation to the scheduled operations list of the machine using backfilling."""

        # find max finishing time predecessors
        finishing_time_predecessors = operation.finishing_time_predecessors
        finishing_time_machine = max([operation.scheduled_end_time for operation in self.scheduled_operations],
                                     default=0)

        setup_time = 0
        if len(self.scheduled_operations) != 0:
            setup_time = \
                sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[-1].operation_id][
                    operation.operation_id]

        # # find backfilling opportunity
        start_time_backfilling, setup_time_backfilling = self.find_backfilling_opportunity(
            operation, finishing_time_predecessors, processing_time, sequence_dependent_setup_times)

        if start_time_backfilling is not None:
            start_time = start_time_backfilling
            setup_time = setup_time_backfilling
        else:
            # find time when predecessors are finished and machine is available
            start_time = max(finishing_time_predecessors,
                             finishing_time_machine + setup_time)

        operation.add_operation_scheduling_information(
            self.machine_id, start_time, setup_time, processing_time)

        self._processed_operations.append({
            'operation': operation,
            'start_time': start_time,
            'end_time': start_time + processing_time,
            'processing_time': processing_time,
            'start_setup': start_time - setup_time,
            'end_setup': start_time,
            'setup_time': setup_time
        })

    def find_backfilling_opportunity(self, operation, finishing_time_predecessors, duration,
                                     sequence_dependent_setup_times):
        """Find the earliest time to start the operation on this machine."""

        if len(self.scheduled_operations) > 0:
            # check if backfilling is possible before the first scheduled operation:
            if duration <= self.scheduled_operations[0].scheduled_start_time - \
                    sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                        self.scheduled_operations[0].operation_id] \
                    and finishing_time_predecessors <= self.scheduled_operations[0].scheduled_start_time - duration - \
                    sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                        self.scheduled_operations[0].operation_id]:
                start_time_backfilling = min(
                    [finishing_time_predecessors, (self.scheduled_operations[0].scheduled_start_time - duration -
                                                   sequence_dependent_setup_times[self.machine_id][
                                                       operation.operation_id][
                                                       self.scheduled_operations[0].operation_id])])

                # update sequence dependent setup time for next operation on machine
                next_operation = self.scheduled_operations[0]
                next_operation.update_sequence_dependent_setup_times(next_operation.scheduled_start_time -
                                                                     sequence_dependent_setup_times[self.machine_id][
                                                                         operation.operation_id][
                                                                         self.scheduled_operations[0].operation_id],
                                                                     sequence_dependent_setup_times[self.machine_id][
                                                                         operation.operation_id][
                                                                         self.scheduled_operations[0].operation_id])
                # assumption that the first operation has no setup times!
                return start_time_backfilling, 0

            else:
                for i in range(1, len(self.scheduled_operations[1:])):
                    # if gap between two operations is large enough to fit the new operations (including setup times)
                    if (self.scheduled_operations[i].scheduled_start_time - self.scheduled_operations[
                        i - 1].scheduled_end_time) >= duration + sequence_dependent_setup_times[self.machine_id][
                        self.scheduled_operations[i - 1].operation_id][operation.operation_id] + \
                            sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                self.scheduled_operations[i].operation_id]:

                        # if predecessors finishes before a potential start time
                        if self.scheduled_operations[i - 1].scheduled_end_time + \
                                sequence_dependent_setup_times[self.machine_id][
                                    self.scheduled_operations[i - 1].operation_id][
                                    operation.operation_id] >= finishing_time_predecessors:
                            # update sequence dependent setup time for next operation on machine
                            self.scheduled_operations[i].update_sequence_dependent_setup_times(
                                self.scheduled_operations[i].scheduled_start_time -
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id],
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id])
                            return self.scheduled_operations[i - 1].scheduled_end_time + \
                                   sequence_dependent_setup_times[self.machine_id][
                                       self.scheduled_operations[i - 1].operation_id][operation.operation_id], \
                            sequence_dependent_setup_times[self.machine_id][
                                self.scheduled_operations[i - 1].operation_id][operation.operation_id]
                        elif finishing_time_predecessors + \
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id] + duration <= self.scheduled_operations[
                            i - 1].scheduled_end_time:
                            self.scheduled_operations[i].update_sequence_dependent_setup_times(
                                self.scheduled_operations[i].scheduled_start_time -
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id],
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id])

                            return finishing_time_predecessors + sequence_dependent_setup_times[self.machine_id][
                                self.scheduled_operations[i - 1].operation_id][operation.operation_id], \
                            sequence_dependent_setup_times[self.machine_id][
                                self.scheduled_operations[i - 1].operation_id][operation.operation_id]

        return None, None

    def unschedule_operation(self, operation: Operation):
        """Remove an operation from the scheduled operations list of the machine."""
        self._processed_operations.remove(operation)


class JobShop:
    def __init__(self) -> None:
        self._nr_of_jobs = 0
        self._nr_of_machines = 0
        self._jobs: List[Job] = []
        self._operations: List[Operation] = []
        self._machines: List[Machine] = []
        self._precedence_relations_jobs: Dict[int, List[int]] = {}
        self._precedence_relations_operations: Dict[int, List[int]] = {}
        self._sequence_dependent_setup_times: List = []
        self._operations_to_be_scheduled: List[Operation] = []
        self._operations_available_for_scheduling: List[Operation] = []
        self._scheduled_operations: List[Operation] = []
        self._name: str = ""

    def reset(self):
        self._scheduled_operations = []

        self._operations_to_be_scheduled = [
            operation for operation in self._operations]

        for machine in self._machines:
            machine.reset()

        for operation in self._operations:
            operation.reset()

    def __str__(self):
        return f"Instance {self._name}, {self.nr_of_jobs} jobs, {len(self.operations)} operations, {len(self.machines)} machines"

    def set_nr_of_jobs(self, nr_of_jobs: int) -> None:
        """Set the number of jobs."""
        self._nr_of_jobs = nr_of_jobs

    def set_nr_of_machines(self, nr_of_machines: int) -> None:
        """Set the number of jobs."""
        self._nr_of_machines = nr_of_machines

    def add_operation(self, operation) -> None:
        """Add an operation to the environment."""
        self._operations_to_be_scheduled.append(operation)
        self._operations.append(operation)

    def add_machine(self, machine) -> None:
        """Add a machine to the environment."""
        self._machines.append(machine)

    def add_job(self, job) -> None:
        """Add a job to the environment."""
        self._jobs.append(job)

    def add_precedence_relations_jobs(self, precedence_relations_jobs: Dict[int, List[int]]) -> None:
        """Add precedence relations between jobs --> applicable for assembly scheduling problems."""
        self._precedence_relations_jobs = precedence_relations_jobs

    def add_precedence_relations_operations(self, precedence_relations_operations: Dict[int, List[int]]) -> None:
        """Add precedence relations between operations."""
        self._precedence_relations_operations = precedence_relations_operations

    def add_sequence_dependent_setup_times(self, sequence_dependent_setup_times: List) -> None:
        """Add sequence dependent setup times."""
        self._sequence_dependent_setup_times = sequence_dependent_setup_times

    def set_operations_available_for_scheduling(self, operations_available_for_scheduling: List) -> None:
        """Set the operations that are available for scheduling."""
        self._operations_available_for_scheduling = operations_available_for_scheduling

    def get_job(self, job_id):
        """Return operation object with operation id."""
        return next((job for job in self.jobs if job.job_id == job_id), None)

    def get_operation(self, operation_id):
        """Return operation object with operation id."""
        return next((operation for operation in self.operations if operation.operation_id == operation_id), None)

    def get_machine(self, machine_id):
        """Return machine object with machine id."""
        for machine in self._machines:
            if machine.machine_id == machine_id:
                return machine

    @property
    def jobs(self) -> List[Job]:
        """Return all the jobs."""
        return self._jobs

    @property
    def nr_of_jobs(self) -> int:
        """Return the number of jobs."""
        return self._nr_of_jobs

    @property
    def operations(self) -> List[Operation]:
        """Return all the operations."""
        return self._operations

    @property
    def nr_of_operations(self) -> int:
        """Return the number of jobs."""
        return len(self._operations)

    @property
    def machines(self) -> List[Machine]:
        """Return all the machines"""
        return self._machines

    @property
    def nr_of_machines(self) -> int:
        """Return the number of jobs."""
        return self._nr_of_machines

    @property
    def operations_to_be_scheduled(self) -> List[Operation]:
        """Return all the operations to be schedule"""
        return self._operations_to_be_scheduled

    @property
    def operations_available_for_scheduling(self) -> List[Operation]:
        """Return all the operations that are available for scheduling"""
        return self._operations_available_for_scheduling

    @property
    def scheduled_operations(self) -> List[Operation]:
        """Return all the operations that are scheduled"""
        return self._scheduled_operations

    @property
    def precedence_relations_operations(self) -> Dict[int, List[int]]:
        """Return the precedence relations between operations."""
        return self._precedence_relations_operations

    @property
    def precedence_relations_jobs(self) -> Dict[int, List[int]]:
        """Return the precedence relations between operations."""
        return self._precedence_relations_jobs

    @property
    def instance_name(self) -> str:
        """Return the name of the instance."""
        return self._name

    @property
    def makespan(self) -> float:
        """Return the total makespan needed to complete all operations."""
        return max(
            [operation.scheduled_end_time for machine in self.machines for operation in machine.scheduled_operations])

    @property
    def total_workload(self) -> float:
        """Return the total workload (sum of processing times of all scheduled operations)"""
        return sum(
            [operation.scheduled_duration for machine in self.machines for operation in machine.scheduled_operations])

    @property
    def max_workload(self) -> float:
        """Return the max workload of machines (sum of processing times of all scheduled operations on a machine)"""
        return max(sum(op.scheduled_duration for op in machine.scheduled_operations) for machine in self.machines)

    def schedule_operation_with_backfilling(self, operation: Operation, machine_id, duration) -> None:
        """Schedule an operation"""
        if operation not in self.operations_available_for_scheduling:
            raise ValueError(
                f"Operation {operation.operation_id} is not available for scheduling")
        machine = self.get_machine(machine_id)
        if not machine:
            raise ValueError(
                f"Invalid machine ID {machine_id}")
        self.schedule_operation_on_machine_backfilling(operation, machine_id, duration)
        self.mark_operation_as_scheduled(operation)

    def unschedule_operation(self, operation: Operation) -> None:
        """Unschedule an operation"""
        machine = self.get_machine(operation.scheduled_machine)
        machine.unschedule_operation(operation)
        self.mark_operation_as_available(operation)

    def schedule_operation_on_machine_backfilling(self, operation: Operation, machine_id, duration) -> None:
        """Schedule an operation on a specific machine."""
        machine = self.get_machine(machine_id)
        if machine is None:
            raise ValueError(
                f"Invalid machine ID {machine_id}")

        machine.add_operation_to_schedule_backfilling(
            operation, duration, self._sequence_dependent_setup_times)

    def mark_operation_as_scheduled(self, operation: Operation) -> None:
        """Mark an operation as scheduled."""
        if operation not in self.operations_available_for_scheduling:
            raise ValueError(
                f"Operation {operation.operation_id} is not available for scheduling")
        self.operations_available_for_scheduling.remove(operation)
        self.scheduled_operations.append(operation)
        self.operations_to_be_scheduled.remove(operation)

    def mark_operation_as_available(self, operation: Operation) -> None:
        """Mark an operation as available for scheduling."""
        if operation not in self.scheduled_operations:
            raise ValueError(
                f"Operation {operation.operation_id} is not scheduled")
        self.scheduled_operations.remove(operation)
        self.operations_available_for_scheduling.append(operation)
        self.operations_to_be_scheduled.append(operation)