import logging
import tomli
import re
from pathlib import Path
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.FIFO_EET import FIFO_EET
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.FIFO_SPT import FIFO_SPT
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.Job import Job, Operation, JobShop, Machine
from FJSP_config import get_FJSPconfig
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.MOR_EET import MOR_EET
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.MOR_SPT import MOR_SPT
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.MWR_SPT import MWR_SPT
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.SimulationEnv import SimulationEnv

def load_parameters(config_toml):
    """Load parameters from a toml file"""
    with open(config_toml, "rb") as f:
        config_params = tomli.load(f)
    return config_params


def schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule):
    """Schedule operations on the machines based on the priority values"""
    machines_available = [machine for machine in simulationEnv.JobShop.machines if
                          simulationEnv.machine_resources[machine.machine_id].count == 0]
    machines_available.sort(key=lambda m: m.machine_id)

    for machine in machines_available:
        operation_to_schedule = select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule)
        if operation_to_schedule is not None:
            simulationEnv.JobShop._scheduled_operations.append(operation_to_schedule)
            # Check if all precedence relations are satisfied
            simulationEnv.simulator.process(simulationEnv.perform_operation(operation_to_schedule, machine))


def run_simulation(simulationEnv, dispatching_rule, machine_assignment_rule):
    """Schedule simulator and schedule operations with the dispatching rules"""

    if simulationEnv.online_arrivals:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling_environment until all operations are processed
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)

    else:
        # add machine resources to the environment
        for _ in simulationEnv.JobShop.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling_environment and schedule operations until all operations are processed from the data instance
        while len(simulationEnv.processed_operations) < sum(
                [len(job.operations) for job in simulationEnv.JobShop.jobs]):
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)


def check_precedence_relations(simulationEnv, operation):
    """Check if all precedence relations of an operation are satisfied"""
    for preceding_operation in operation.predecessors:
        if preceding_operation not in simulationEnv.processed_operations:
            return False
    return True



def check_precedence_relations(simulationEnv, operation):
    """Check if all precedence relations of an operation are satisfied"""
    for preceding_operation in operation.predecessors:
        if preceding_operation not in simulationEnv.processed_operations:
            return False
    return True


def select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule):
    """use dispatching rules to select the next operation to schedule"""
    operation_priorities = {}

    # Calculate the priority values for the operations based on the dispatching rule and machine assignment rule
    for job in simulationEnv.JobShop.jobs:
        for operation in job.operations:
            if operation not in simulationEnv.processed_operations and operation not in simulationEnv.JobShop.scheduled_operations and machine.machine_id in operation.processing_times:
                if check_precedence_relations(simulationEnv, operation):
                    if dispatching_rule == 'FIFO' and machine_assignment_rule == 'SPT':
                        getfilefifospt=FIFO_SPT(config=get_FJSPconfig(), simulationEnv=simulationEnv)
                        getfilefifospt.get_op(machine,operation_priorities,operation)

                    elif dispatching_rule in ['MOR', 'LOR'] and machine_assignment_rule == 'SPT':  # MOR_SPT
                        getfilemorspt=MOR_SPT(config=get_FJSPconfig(), simulationEnv=simulationEnv)
                        getfilemorspt.get_op(machine,operation_priorities,operation)


                    elif dispatching_rule in ['MWR', 'LWR'] and machine_assignment_rule == 'SPT':  # MWR_SPT
                        getfilemwrspt=MWR_SPT(config=get_FJSPconfig(),simulationEnv=simulationEnv)
                        getfilemwrspt.get_op(machine,operation_priorities,operation)

                    elif dispatching_rule == 'FIFO' and machine_assignment_rule == 'EET':  # FIFO_EET
                        getfilefifoeet=FIFO_EET(config=get_FJSPconfig(),simulationEnv=simulationEnv)
                        getfilefifoeet.get_op(machine, operation_priorities, operation)

                    elif dispatching_rule in ['MOR', 'LOR'] and machine_assignment_rule == 'EET':  # MOR_EET
                        getfilemoreet = MOR_EET(config=get_FJSPconfig(), simulationEnv=simulationEnv)
                        getfilemoreet.get_op(machine, operation_priorities,operation)



    if len(operation_priorities) == 0:
        return None
    else:
        if dispatching_rule == 'FIFO' or dispatching_rule == 'LOR' or dispatching_rule == 'LWR':
            return min(operation_priorities, key=operation_priorities.get)
        else:
            return max(operation_priorities, key=operation_priorities.get)


def get_operations_remaining(simulationEnv, operation):
    """get remaining operations of the job"""
    return len(
        [operation for operation in operation.job.operations if operation not in simulationEnv.processed_operations])


def parse(JobShop, instance, from_absolute_path=True):
    if not from_absolute_path:
        base_path = Path(__file__).parent.parent.absolute()
        data_path = base_path.joinpath(instance)
    else:
        data_path = instance

    with open(data_path, "r") as data:
        total_jobs, total_machines, max_operations = re.findall(
            '\S+', data.readline())
        number_total_jobs, number_total_machines, number_max_operations = int(
            total_jobs), int(total_machines), int(float(max_operations))

        JobShop.set_nr_of_jobs(number_total_jobs)
        JobShop.set_nr_of_machines(number_total_machines)

        precedence_relations = {}
        job_id = 0
        operation_id = 0

        for key, line in enumerate(data):
            if key >= number_total_jobs:
                break
            # Split data with multiple spaces as separator
            parsed_line = re.findall('\S+', line)

            # Current item of the parsed line
            i = 1

            job = Job(job_id)

            while i < len(parsed_line):
                # Total number of operation options for the operation
                operation_options = int(parsed_line[i])
                # Current activity
                operation = Operation(job, job_id, operation_id)

                for operation_options_id in range(operation_options):
                    machine_id = int(parsed_line[i + 1 + 2 *
                                                 operation_options_id]) - 1
                    duration = int(
                        parsed_line[i + 2 + 2 * operation_options_id])
                    operation.add_operation_option(machine_id, duration)
                job.add_operation(operation)
                JobShop.add_operation(operation)
                if i != 1:
                    precedence_relations[operation_id] = [
                        JobShop.get_operation(operation_id - 1)]

                i += 1 + 2 * operation_options
                operation_id += 1

            JobShop.add_job(job)
            job_id += 1

    # add also the operations without precedence operations to the precendence relations dictionary
    for operation in JobShop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    sequence_dependent_setup_times = [
        [[0 for r in range(len(JobShop.operations))] for t in range(len(JobShop.operations))] for
        m in range(number_total_machines)]

    # Precedence Relations & sequence dependent setup times
    JobShop.add_precedence_relations_operations(precedence_relations)
    JobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    # Machines
    for id_machine in range(0, number_total_machines):
        JobShop.add_machine((Machine(id_machine)))

    return JobShop


class Heuristic_Framework:
    def __init__(self, config):
        self.config = config

    def run_rule(self,path):
        simulationEnv = SimulationEnv(
            online_arrivals=self.config.online_arrivals
        )

        if not self.config.online_arrivals:
            try:
                if self.config.problem_name == 'FJSP':
                    simulationEnv.JobShop = parse(simulationEnv.JobShop,path)
            except Exception as e:
                print(f"able to schedule '/fjsp/: {e}")

            simulationEnv.simulator.process(run_simulation(simulationEnv, self.config.dispatching_rule,
                                                           self.config.machine_assignment_rule))
            simulationEnv.simulator.run()
            logging.info(f"Makespan: {simulationEnv.JobShop.makespan}")
            print('Makespan: ', simulationEnv.JobShop.makespan)
        return simulationEnv


