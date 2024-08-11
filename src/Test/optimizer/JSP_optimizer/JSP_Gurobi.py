import time

from gurobipy import GRB, quicksum, GurobiError
import gurobipy as gp

class JSPModel:
    def __init__(self, num_macmhines, num_jobs, job_seqs_vec, prod_time_mat,limitedtime):
        self.m = num_macmhines
        self.n = num_jobs
        self.job_seqs = job_seqs_vec
        self.production_time = prod_time_mat
        self.limitedtime = limitedtime

    def solveGurobi(self, save_to_file=False, save_file_name=""):
        print("Starting Gurobi ...")
        print()
        try:

            model = gp.Model('JSP')

            # Add decision variable X
            # X[i][j] means the integer start time of job i on machine j
            start_time = time.time()
            X = model.addVars(self.n, self.m, vtype=GRB.CONTINUOUS, name="x")
            # Add decision variable Z
            # Z[k][i][j] is equal to 1 if job k precedes job i on machine j
            Z = model.addVars(self.n, self.n, self.m, vtype=GRB.BINARY, name='z')  # Variable x_{it}

            # Temporary variable Cmax
            Cmax = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_max")

            # Set objective function
            model.setObjective(Cmax, GRB.MINIMIZE)
            # Add constraints
            # Equation (3), means that each step can be taken only after the precedent step finished
            for i in range(self.n):  # Workpiece i
                for h in range(self.m):  # h is the operation self.job_seqs[i][h] represents the machine tool
                    model.addConstr(X[i, self.job_seqs[i][h]] >= 0)

            for j in range(self.m):  # Workpiece i
                for k in range(self.n):
                    for i in range(self.n):
                        if i != k:
                            model.addConstr(Z[k, i, j] + Z[i, k, j] <= 1)

            V = 100000
            # # for i in range(self.n):  # Workpiece i
            # #         model.addConstr(
            # #             X[i][self.job_seqs[i][0]] >= X[k][self.job_seqs[i][0]] + self.production_time[k][self.job_seqs[i][0]] - V * Z[k][i][self.job_seqs[i][0]])
            # # Equation (4) and (5), means that no two jobs can be scheduled on the same machine at the same time
            #  # Set to a very big number
            for k in range(self.n):  # i is the latter workpiece, K is the former workpiece
                for i in range(self.n):
                    for j in range(self.m):
                        for h in range(len(self.production_time[k])):
                            if self.job_seqs[i][h] == j and i != k:
                                desired_h = h
                                model.addConstr(
                                    X[k, j] >= X[i, j] + self.production_time[i][desired_h] - V * (Z[k, i, j]))

            for k in range(self.n):  # i is the latter workpiece, K is the former workpiece
                for i in range(self.n):
                    for j in range(self.m):
                        for h in range(len(self.production_time[k])):
                            if self.job_seqs[k][h] == j and i != k:
                                desired_h = h
                                model.addConstr(
                                    X[i, j] >= X[k, j] + self.production_time[k][desired_h] - V * (1 - Z[k, i, j]))

            # # Equation (6), ensure Cmax is the latest job finish time

            for i in range(self.n):  #  Workpiece i
                for h in range(1, self.m):  # h is the operation
                    model.addConstr(
                        X[i, self.job_seqs[i][h]] >= X[i, self.job_seqs[i][h - 1]] + self.production_time[i]
                        [h - 1])

            for i in range(self.n):
                model.addConstr(
                    Cmax >= X[i, self.job_seqs[i][self.m - 1]] + self.production_time[i][self.m - 1])
            model.params.TimeLimit = self.limitedtime
            model.optimize()

            end_time = time.time()
            solve_time = end_time - start_time
            print("solution time: {:.2f} s".format(solve_time))
            if model.status == GRB.OPTIMAL:
                obj_wal = model.objVal
                print('Optimal objective: %g' % model.objVal)
                # print("x", X)
                # print("Z", Z)

        except GurobiError as e:
            print("Error code =", str(e.errno))
            print(str(e.message))
        return  model.objVal, solve_time


