	This is a unified platform for solving job scheduling problem (JSP) and flexible JSP (FJSP). The platform provides a flexible algorithmic template where users can effortlessly implement their unique designs. In the platform, the benchmark contains a total of 300 instances. The baseline contains learning-based algorithms and learning-based algorithms and non-learning algorithms (heuristic rules, Gurobi, and evolutionary algorithms). For the JSP, there are 5 learning-based algorithms, 9 no-learning algorithms including 4 heuristic rules, 4 basic metaheuristics, and a Gurobi solver. The FJSP contains 2 learning-based algorithms, 9 no-learning algorithms including 5 heuristic rules, 3 basic metaheuristics, and a Gurobi solver. 



## **Citing**

```latex
@inproceedings{BPLS-JSS,
title={BPLS-JSS: A Benchmark Platform for Evaluating Learning-based job shop Scheduling Algorithms}
}
```



## **System module**

```python
src   
├── C++exe                                               // c++ exe
├── Problem                         			 // Problem module
│       └── Basic_problem.py                        	 // Problem base class interface
│       └── FJSP.py                        	         // FJSP decoding
│       └── JSP.py                        		 // JSP decoding
├── Result                         		         // Store results
│       └── FJSP                        		 // Results for FJSP baselines
│       └── JSP                        			 // Results for FJSP baselines
├── Test          					 // Test module
│       └── agent                       		 // Agent class file
│       └── data_test                    		 // Benchmarks
│       └── environment                   		 // Environment
│       └── optimizer                         		 // Optimizer class file
├── Train         					 // Train module
│       └── FJSP_train                      		 // Baselines to be trained in FJSP
│       └── JSP_train                      		 // Baselines to be trained in JSP│       
│       └── FJSP_generate_trainSet                       // Randomly generate FJSP training instances
│       └── JSP_generate_trainSet                     	 // Randomly generate JSP training instances
├── main.py         				         // Main function 
├── test.py         					 // Load the baseline type and benchmarks to be tested
├── FJSP_config.py         				 // Parameter configuration of FJSP
├── JSP_config.py         				 // Parameter configuration of JSP
├── LoadUtils.py         				 // Load datas of JSP and FJSP
├── main-C++.py                                          // Conduct C++ exe


```



## **Requirements**

To run the  **JSP** related code, you need to install the python package in requirements.txt:

Python=3.10.1

ale_py==0.7.5

gurobipy==10.0.1

gym==0.21.0

job_shop_cp_env==1.0.0

matplotlib==3.8.0

networkx==2.8rc1

numpy==1.26.4

ortools==9.7.2996

pandas==1.3.5

plotly==5.11.0

ray==2.6.1

simpy==4.0.1

stable_baselines3==1.6.0

sympy==1.12

tomli==2.0.1

torch==1.13.1

torch_geometric==2.5.2

tqdm==4.64.0

~orch==1.12.0

~orch==2.0.0

~orch==2.2.0

~orch==2.2.1

To run **FJSP** related code, you need to install the python package in FJSP-requirements.txt

gym==0.20.0

matplotlib==3.7.2

networkx==3.1

numpy==1.23.0

numpy==1.23.5

pandas==2.0.2

stable_baselines3==2.0.0

torch==2.0.1

torch_geometric==2.5.1

wandb==0.16.4

 

## **Quick Start**

​		We provide the following commands for fast execution of algorithms, including fast execution of training and testing on learning-based algorithms in JSP, testing of non-learning algorithms, and training and testing on learning-based algorithms in FJSP, testing of non-learning algorithms.

 

**First, get into the main code folder src.**

cd ../src

 

**To train L2D in JSP, we run the following command:**

python .\Train\JSP_train\trainL2D.py --test_datas Train/JSP_train/L2D_train/ --device cpu --problem_name JSP

**To train L2D in JSP, we run the following command:**

python main.py --optimizer L2D_optimizer --test_datas Test/data_test/JSP_benchmark/ --device cpu --problem_name JSP

**To run LPT in JSP heuristic rules, we execute the following command for training:**

python main.py --optimizer LPT --test_datas Test/data_test/JSP_benchmark/ --device cpu --problem_name JSP

Tips: To convert JSP to FJSP, you need to modify the main function

**To train FJSP_DAN in the FJSP, we execute the following command:** 

python .\Train\FJSP_train\FJSP_DAN.py --test_datas Train/FJSP_train/FJSP_DAN_train/ --device cpu --problem_name FJSP

**To run FJSP_DAN in the FJSP, we execute the following command:**

python main.py --optimizer FJSP_DAN_optimizer --test_datas Test\data_test\FJSP_test_datas --device cpu --problem_name FJSP

**To run Gurobi of FJSP, we execute the following command:**

python main.py --optimizer FJSP_Gurobi --test_datas Test\data_test\FJSP_test_datas --device cpu --problem_name FJSP --FJSP_gurobi_time_limit 3600

**To run MWR_SPT of FJSP, we execute the following command:**
python main.py --optimizer Heuristic_Framework --test_datas Test\data_test\FJSP_test_datas --device cpu --problem_name FJSP --dispatching_rule MWR --machine_assignment_rule SPT

**Datasets**

​		For the JSP, there are 170 instances. The ABZ, FT, LA, ORB , YN, SWV, and TAI benchmark, borrowed from [[GitHub - zcaicaros/L2S: Official implementation of paper "Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling"](https://github.com/zcaicaros/L2S)], and the DMU benchmark, borrowed from [[GitHub - zcaicaros/L2D: Official implementation of paper "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning"](https://github.com/zcaicaros/L2D)]

​		For the FJSP, there are 130 instances, contains Brandimarte, Hurink_edata, Hurink_rdata, and Hurink_vdata benchmark, borrowed from the benchmark of [https://github.com/wrqccc/fjsp-drl]

 

## **Baseline Library**

The learning-based algorithms in JSP and FJSP include:

| **JSP**     |                                                              |                                                              |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| L2D         | [2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html) | Learning to dispatch for job shop scheduling via deep reinforcement learning. |
| L2S         | [2022](https://arxiv.org/abs/2211.10936)                     | Learning to search for job shop scheduling via deep reinforcement learning. |
| RL-GNN      | [2021](https://www.tandfonline.com/doi/abs/10.1080/00207543.2020.1870013) | Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning. |
| ScheduleNet | [2021](https://arxiv.org/abs/2106.03051)                     | ScheduleNet: Learn to solve multi-agent scheduling problems with reinforcement learning. |
| JSSEnv      | [2021](https://arxiv.org/abs/2104.03760)                     | A reinforcement learning environment for job-shop scheduling. |
| **FJSP**    |                                                              |                                                              |
| FJSP_DAN    | [2022](https://ieeexplore.ieee.org/abstract/document/9826438) | Flexible job-shop scheduling via graph neural network and deep reinforcement learning. |
| FJSP_GNN    | [2023](https://ieeexplore.ieee.org/abstract/document/10246328) | Flexible job shop scheduling via dual attention network-based reinforcement learning. |

In addition to these mentioned learning-based algorithms, for the JSP, the non-learning algorithms include:

[Four heuristic rules: SPT, LPT, SRPT, and LRPT](https://ieeexplore.ieee.org/abstract/document/8676376)

[Exact algorithm: Gurobi](https://www.sciencedirect.com/science/article/pii/S0305054816300764)

Evolutionary algorithms: [GA](https://www.sciencedirect.com/science/article/pii/S221065022400018X), [ABC](https://ieeexplore.ieee.org/abstract/document/9953057), [PSO](https://www.sciencedirect.com/science/article/pii/S0377221722007184), and [Jaya](https://ieeexplore.ieee.org/abstract/document/8345746)

For the FJSP, the non-learning algorithms include:

[Five heuristic rules: FIFO_EET, FIFO_SPT, MOR_EET, MOR_SPT, and MWR_SPT.](https://arxiv.org/abs/2308.12794)

[Exact algorithm: Gurobi](https://www.sciencedirect.com/science/article/pii/S0305054816300764)

Evolutionary algorithms are [ABC](https://ieeexplore.ieee.org/abstract/document/9953057), [GA](https://www.sciencedirect.com/science/article/pii/S221065022400018X), and [PSO](https://www.sciencedirect.com/science/article/pii/S0377221722007184).

For the L2D and L2S in JSP and FJSP_DAN and FJSP_GNN in FJSP need to be retrained to obtain the training model.

 
