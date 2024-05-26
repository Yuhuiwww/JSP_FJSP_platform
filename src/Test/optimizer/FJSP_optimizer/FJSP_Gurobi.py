# Code based on the paper (Note: based on MILP-5):
# "Mathematical modelling and a meta-heuristic for flexible job shop scheduling"
# by V. Roshanaei, Ahmed Azab & H. ElMaraghy
# Presented in International Journal of Production Research, 2013.
# Paper URL: https://www.tandfonline.com/doi/full/10.1080/00207543.2013.827806
import time

from gurobipy import Model, GRB, quicksum

class FJSPModel:
    def __init__(self, config):
        self.config=config
    def FJSP_solveGurobi(self,instance_data):
        # Extracting the data
        jobs = instance_data['jobs']  # j,h
        operations_per_job = instance_data['operations_per_job']  # l,z
        machine_allocations = instance_data['machine_allocations']  # Rj,l
        operations_times = instance_data['operations_times']  # pj,l,i
        largeM = instance_data['largeM']  # M
        model = Model("MILP")
        start_time=time.time()
        # Decision Variables
        Y = {}
        S = {}
        X = {}

        for j in jobs:
            for l in operations_per_job[j]:
                for i in machine_allocations[j, l]:
                    S[j, l, i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"S_{j}_{l}_{i}")
                    Y[j, l, i] = model.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{l}_{i}")
                    for h in jobs:
                        if h > j:
                            for z in operations_per_job[h]:
                                X[j, l, h, z] = model.addVar(vtype=GRB.BINARY, name=f"X_{j}_{l}_{h}_{z}")

        # Objective function
        cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")
        model.setObjective(cmax, GRB.MINIMIZE)

        # Constraints

        # 3.2.2 Assignment constraint sets
        for j in jobs:
            for l in operations_per_job[j]:
                model.addConstr(quicksum(Y[j, l, i] for i in machine_allocations[j, l]) == 1)
                for i in machine_allocations[j, l]:
                    model.addConstr(S[j, l, i] <= largeM * Y[j, l, i])

        # 3.2.3.1 Logical precedence constraints among operations of a job
        for j in jobs:
            for l in operations_per_job[j][:-1]:
                lhs = quicksum(S[j, l + 1, i] for i in machine_allocations[j, l + 1])
                rhs = quicksum(S[j, l, i] for i in machine_allocations[j, l]) + quicksum(
                    operations_times[j, l, i] * Y[j, l, i] for i in machine_allocations[j, l])
                model.addConstr(lhs >= rhs)

        # 3.2.3.2 Machine non-interference constraints and precedence constraints among operations of different jobs
        for j in jobs:
            for l in operations_per_job[j]:
                for h in jobs:
                    if h > j:
                        for z in operations_per_job[h]:
                            common_machines = set(machine_allocations[j, l]) & set(machine_allocations[h, z])
                            for i in common_machines:
                                model.addConstr(
                                    S[j, l, i] >= S[h, z, i] + operations_times[h, z, i] - (largeM * (
                                            3 - X[j, l, h, z] - Y[j, l, i] - Y[h, z, i])))
                                model.addConstr(
                                    S[h, z, i] >= S[j, l, i] + operations_times[j, l, i] - (largeM * (
                                            X[j, l, h, z] + 2 - Y[j, l, i] - Y[h, z, i])))

        # 3.2.3.3 Constraints for capturing the value of objective function
        for j in jobs:
            last_op = max(operations_per_job[j])
            model.addConstr(
                cmax >= quicksum(S[j, last_op, k] for k in machine_allocations[j, last_op]) + quicksum(
                    operations_times[j, last_op, k] * (Y[j, last_op, k]) for k in machine_allocations[j, last_op])
            )
        model.params.TimeLimit = self.config.FJSP_gurobi_time_limit
        model.optimize()
        end_time = time.time()
        solve_time = end_time - start_time
        print("求解时间: {:.2f} 秒".format(solve_time))
        if model.status == GRB.OPTIMAL:
            obj_wal = model.objVal
            print('Optimal objective: %g' % model.objVal)
        return model.objVal, solve_time