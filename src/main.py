
from Problem.JSP import JSP
from FJSP_config import get_FJSPconfig

from test import Tester
from JSP_config import get_config
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# conduct the JSP algorithms
if __name__ == '__main__':

    N = [10, 20, 6, 10, 20, 10, 10, 15, 15, 15, 20, 20, 30, 10, 20, 20, 50,20,20, 20, 30, 30, 40, 40, 50, 50,15, 20, 20, 30, 30, 50, 50, 100]
    M = [10, 15, 6, 10, 5, 5, 10, 5, 10, 15, 5, 10, 10, 10, 10, 15, 10,20,15, 20, 15, 20, 15, 20, 15, 20,15, 15, 20, 15, 20, 15, 20, 20]
    prefer = [('abz', 2), ('ft', 3), ('la', 8), ('orb', 1), ('swv', 3), ('yn', 1),('dmu', 8),('tai', 8)]
    count=0
    # # # 循环prefer
    for i in range(len(prefer)):
        name,times = prefer[i]
        for j in range(times):
            print('start---',name,N[j+count],'X',M[j+count])
            '''
            for the jsp, config = get_config()
            '''
            config = get_config()
            config.Pn_j = N[j+count]
            config.Pn_m = M[j+count]
            config.test_datas_type = name
            tester = Tester(config)
            ''' for the jsp:
                    if the algorithm is a learning-based algorithm select test_JSP();
                    if the algorithm is a heuristic rule select test_JSP_heuristic();
                    if the algorithm is Gurobi select JSP_Gurobi();
                    if the algorithm is an evolutionary algorithm select test_JSP_algorithm();
            '''
            tester.test_JSP_algorithm()
        count += times
# conduct the FJSP algorithms
# if __name__ == '__main__':
#     N = [10, 10, 15, 15, 20, 20, 20, 10, 10, 15, 15, 15, 20, 20, 30, 10, 10, 15, 15, 15, 20, 20, 30, 10, 10, 15, 15, 15,
#          20, 20, 30]
#     M = [6, 10, 4, 8, 5, 10, 15, 5, 10, 5, 10, 15, 5, 10, 10, 5, 10, 5, 10, 15, 5, 10, 10, 5, 10, 5, 10, 15, 5, 10, 10]
#     prefer = [('Brandimarte', 7), ('Hurink_edata', 8), ('Hurink_rdata', 8), ('Hurink_vdata', 8)]
#     count = 0
#     for i in range(len(prefer)):
#         name,times = prefer[i]
#         for j in range(times):
#             print('start---',name,N[j+count],'X',M[j+count])
#             config = get_FJSPconfig()
#             config.Pn_j = N[j+count]
#             config.Pn_m = M[j+count]
#             config.test_datas_type = name
#             ''' for the fjsp:
#                     if the algorithm is a learning-based algorithm select test_FJSP();
#                     if the algorithm is a heuristic rule select test_FJSP_Heuristic();
#                     if the algorithm is Gurobi select FJSP_Gurobi();
#                     if the algorithm is an evolutionary algorithm select test_FJSP_traditionalAlgorithm();'''
#             tester = Tester(config)
#             tester.FJSP_Gurobi()
#         count += times
