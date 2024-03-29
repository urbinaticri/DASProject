from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

def read_file(filename, n_agents):
    PP = []
    Adj = []
    cntr = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            if cntr < n_agents: #PP
                PP.extend([float(p) for p in line.split()])
            else: #Adj
                Adj.append([float(p) for p in line.split()])
            cntr += 1
    return np.array(PP), np.asarray(Adj)


def generate_launch_description():
    MAXITERS = 2000
    COMM_TIME = 5e-2 # communication time period
    np.random.seed(5)

    filename = "formation_S"
    NN = 6 # number of agents
    n_leaders = 2 # number of leaders
    d = 2 # space dimension

    # Load positions and adjacency matrix of desired formation
    L = 1.0 # Scale
    PP, Adj = read_file(filename, NN)
    PP = L*PP

    # initial positions
    p = np.vstack((
        np.array(PP[:n_leaders*d]).reshape(d*n_leaders, 1),
        np.zeros((d*(NN-n_leaders),1)) + 5*np.random.rand(d*(NN-n_leaders),1)
    ))

    # initial velocities
    constant_v = 0 #Leaders constant velocity
    v = np.vstack((
        np.zeros((d*n_leaders,1)) + constant_v,
        np.zeros((d*(NN-n_leaders),1))
    ))

    # state vector initialization for all agents
    x_init = np.vstack((
        p,
        v
    ))

    # bearing unit vector g_{ij} function
    def g(pp,ii,jj):
        index_ii = ii*d + np.arange(d)
        index_jj = jj*d + np.arange(d)
        return (pp[index_jj] - pp[index_ii]) / (np.linalg.norm(pp[index_jj] - pp[index_ii]) + 1e-15)

    # orthogonal projection matrix P_{g_{ij}} function
    def P(g_ij):
        g_ij = g_ij.reshape((-1, 1)) #here reshape because from row array i.e. [1,0] we want col array i.e. [[1], [0]] 
        return np.identity(d) - g_ij@(g_ij.T)

    GG = np.zeros((NN, NN, d), dtype=np.float32) # matrix containing bearing vector of desired formation
    Pg_star = np.zeros((NN, NN, d, d), dtype=np.float32) # projection matrix of desired formation

    for ii in range(NN):
        for jj in range(NN):
            g_star = g(PP, ii, jj)
            GG[ii, jj, :] = g_star
            Pg_star[ii, jj, :] = P(g_star)

    # When writing report, use this to demonstrate antisymmetry: g_{ji} = - g_{ij}
    is_GG_antisym = not np.any(GG+np.transpose(GG, axes= (1, 0, 2)))
    print(is_GG_antisym)

    # Network generation
    p_ER = 0.9
    I_NN = np.identity(NN, dtype=int)

    while 1:
        Adj = np.random.binomial(1, p_ER, (NN, NN)) # Generates a NNxNN matrix drawing values from a binomial distribution
        Adj = np.logical_or(Adj, Adj.T) # Makes the matrix symmetric
        Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int) # Set 0 on main diagonal

        # test connectivity
        test = np.linalg.matrix_power((I_NN+Adj),NN)  #Strongly connected graph test
        
        if np.all(test>0):
            print("the graph is connected\n")
            break

    
    launch_description = [] # Append here your nodes
    ################################################################################
    # RVIZ
    ################################################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('my_formation_rviz')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    launch_description.append(
        Node(
            package='rviz2', 
            node_executable='rviz2', 
            arguments=['-d', rviz_config_file],
            # output='screen',
            # prefix='xterm -title "rviz2" -hold -e'
            ))

    ################################################################################
    for ii in range(NN):

        bearing_ii = Pg_star[:, ii, :, :].flatten().tolist() # bearaing vectors relative to neighbors of agent i
        N_ii = np.nonzero(Adj[:, ii])[0].tolist() # agent i neighbors indeces list
        ii_index = ii*d + np.arange(d)
        x_init_ii = np.concatenate((
            x_init[ii_index].flatten(),
            x_init[ii_index + NN*d].flatten()),
            axis=0
        ).tolist() # Initial position and velociti of the agent i    

        launch_description.append(
            Node(
                package='my_formation_rviz',
                node_namespace ='agent_{}'.format(ii),
                node_executable='agent_i',
                parameters=[{
                                'agent_id': ii, 
                                'max_iters': MAXITERS, 
                                'communication_time': COMM_TIME, 
                                'neigh': N_ii, 
                                'x_init': x_init_ii, #[p_x, p_y, v_x, v_y]
                                'dist' : bearing_ii
                                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

        ################################################################################

        launch_description.append(
            Node(
                package='my_formation_rviz', 
                node_namespace='agent_{}'.format(ii),
                node_executable='visualizer', 
                parameters=[{
                                'agent_id': ii,
                                'communication_time': COMM_TIME,
                                }],
            ))


    return LaunchDescription(launch_description)