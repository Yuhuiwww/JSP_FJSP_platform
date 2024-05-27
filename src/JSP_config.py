import argparse
import os
import time

import multiprocessing as mp


def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def get_config(args=None):
    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--optimizer', type=str, default='L2D_optimizer')
    parser.add_argument('--Pn_j', type=int, default=50, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=20, help='Number of machines instances to test')
    parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
    parser.add_argument('--test_datas', type=str, default='Train/JSP_train/',
                        help='test_datas position')#'Test/data_test/JSP_benchmark/','Train/JSP_train/L2S_train/','Train/JSP_train/L2D_train/',
    parser.add_argument('--test_datas_type', type=str, default='dmu')
    parser.add_argument('--model_source', type=str, default='tai', help='Suffix of the data that model trained on')

    parser.add_argument('--problem_name', type=str, default='JSP', help='problem_name position')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    # GNN agent
    parser.add_argument('--NOT_START_NODE_SIG', type=int, default=-1)
    parser.add_argument('--PROCESSING_NODE_SIG', type=int, default=0)
    parser.add_argument('--DONE_NODE_SIG', type=int, default=1)
    parser.add_argument('--DELAYED_NODE_SIG', type=int, default=2)
    parser.add_argument('--DUMMY_NODE_SIG', type=int, default=3)
    parser.add_argument('--CONJUNCTIVE_TYPE', type=int, default=0)
    parser.add_argument('--DISJUNCTIVE_TYPE', type=int, default=-1)
    parser.add_argument('--FORWARD', type=int, default=0)
    parser.add_argument('--BACKWARD', type=int, default=1)
    parser.add_argument('--N_SEP', type=int, default=1)
    parser.add_argument('--SEP', type=str, default=' ')
    parser.add_argument('--NEW', type=str, default='\n')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')######
    parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')
    parser.add_argument('--decay_step_size', type=int, default=2000, help='decay_step_size')
    parser.add_argument('--decay_ratio', type=float, default=0.9, help='decay_ratio, e.g. 0.9, 0.95')
    parser.add_argument('--gamma', type=float, default=1, help='discount factor')
    parser.add_argument('--k_epochs', type=int, default=1, help='update policy for K epochs')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
    parser.add_argument('--ploss_coef', type=float, default=2, help='policy loss coefficient')
    parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='No. of layers of feature extraction GNN including input layer')
    parser.add_argument('--neighbor_pooling_type', type=str, default='sum', help='neighbour pooling type')
    parser.add_argument('--input_dim', type=int, default=2, help='number of dimension of raw node features')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim of MLP in fea extract GNN')#####
    parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=2,
                        help='No. of layers of MLP in fea extract GNN')
    parser.add_argument('--num_mlp_layers_actor', type=int, default=2, help='No. of layers in actor MLP')
    parser.add_argument('--hidden_dim_actor', type=int, default=32, help='hidden dim of MLP in actor')
    parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers in critic MLP')
    parser.add_argument('--hidden_dim_critic', type=int, default=32, help='hidden dim of MLP in critic')
    parser.add_argument('--graph_pool_type', type=str, default='average', help='graph pooling type')
    parser.add_argument('--init_quality_flag', type=bool, default=False,
                        help='Flag of whether init state quality is 0, True for 0')
    parser.add_argument('--et_normalize_coef', type=int, default=1000,
                        help='Normalizing constant for feature LBs (end time), normalization way: fea/constant')
    parser.add_argument('--rewardscale', type=float, default=0., help='Reward scale for positive rewards')
    parser.add_argument('--num_envs', type=int, default=4, help='No. of envs for training')
    parser.add_argument('--torch_seed', type=int, default=600, help='Seed for torch')
    parser.add_argument('--np_seed_train', type=int, default=200, help='Seed for numpy for training')
    parser.add_argument('--np_seed_validation', type=int, default=200, help='Seed for numpy for validation')
    parser.add_argument('--max_updates', type=int, default=10000, help='No. of episodes of each env for training')
    parser.add_argument('--embedding_layer', type=int, default=4)
    parser.add_argument('--policy_layer', type=int, default=4)
    parser.add_argument('--embedding_type', type=str, default='gin+dghan')  # 'gin', 'dghan', 'gin+dghan'
    parser.add_argument('--heads', type=int, default=1)  # dghan parameters
    parser.add_argument('--drop_out', type=float, default=0.)  # dghan parameters
    parser.add_argument('--episodes', type=int, default=128000)
    parser.add_argument('--step_validation', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--transit', type=int, default=500)
    parser.add_argument('--steps_learn', type=int, default=10)
    parser.add_argument('--init_type', type=str, default='fdd-divide-mwkr')
    parser.add_argument('--gym-id', type=str, default="compiled_env:jss-v4",
                        help='the id of the gym environment')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the number of parallel worker')  # mp.cpu_count()"End2End"
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--wandb-project-name', type=str, default="BenchmarkCPEnv",
                        help="the wandb's project name")
    parser.add_argument('--itration', type=int, default=0)
    parser.add_argument('--flag_sample', type=bool, default=False,
                        help='Flag ture of false')
    parser.add_argument('--op_per_job', type=float, default=2,
                        help='Number of operations per job, default 0, means the number equals m')
    parser.add_argument('--test_model', nargs='+', default=['10x6'], help='List of model for testing')
    parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')
    parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')
    parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
    parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')
    parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[32, 8],
                        help='Output dimension of the DAN layers')
    parser.add_argument('--fea_j_input_dim', type=int, default=10, help='Dimension of operation raw feature vectors')
    parser.add_argument('--fea_m_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')
    parser.add_argument('--num_heads_OAB', nargs='+', type=int, default=[4, 4],
                        help='Number of attention head of operation message attention block')
    parser.add_argument('--num_heads_MAB', nargs='+', type=int, default=[4, 4],
                        help='Number of attention head of machine message attention block')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--FJSP_hidden_dim_actor', type=int, default=64, help='hidden dim of MLP in actor')
    parser.add_argument('--FJSP_num_mlp_layers_critic', type=int, default=3, help='No. of layers in critic MLP')
    parser.add_argument('--FJSP_hidden_dim_critic', type=int, default=64, help='hidden dim of MLP in critic')
    parser.add_argument('--FJSP_num_mlp_layers_actor', type=int, default=3, help='No. of layers in actor MLP')
    parser.add_argument('--FJSP_k_epochs', type=int, default=4, help='Update frequency of each episode')
    parser.add_argument('--FJSP_ploss_coef', type=float, default=1, help='Policy loss coefficient')
    parser.add_argument('--FJSP_vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
    parser.add_argument('--reset_env_timestep', type=int, default=20, help='Interval for reseting the environment')
    parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
    parser.add_argument('--FJSP_num_envs', type=int, default=20, help='Batch size for training environments')
    parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')
    parser.add_argument('--model_suffix', type=str, default='', help='Suffix of the model')
    parser.add_argument('--seed_train', type=int, default=300, help='Seed for training')
    parser.add_argument('--runtime', type=int, default=1000, help='Seed for training')
    parser.add_argument('--JSP_gurobi_time_limit', type=int, default=3600)
    config = parser.parse_args()

    config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.problem_name}D'
    # config = parser.parse_args(args=[])
    return config
