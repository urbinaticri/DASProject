from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    MAXITERS = 2000
    COMM_TIME = 5e-2 # communication time period
    np.random.seed(5)

    NN = 4 # number of agents
    n_leaders = 2 # number of leaders
    d = 2 # dimension of positions and velocities

    # formation: square ex. in fig 2 -> agent 1 bottom-left, order counter-clockwise
    L = 1.0
    PP = np.array([L, 0, L, L, 0, L, 0, 0])

    # initial positions
    p = np.vstack((
        np.array([L, 0, L, L]).reshape(d*n_leaders, 1),
        np.zeros((d*(NN-n_leaders),1)) + 5*np.random.rand(d*(NN-n_leaders),1)
    ))

    # initial velocities
    constant_v = 0.5 #Leaders constant velocity
    v = np.vstack((
        np.zeros((d*n_leaders,1)) + constant_v,
        np.zeros((d*(NN-n_leaders),1))
    ))

    x_init = np.vstack((
        p,
        v
    ))

    # bearing unit vector g_{ij}
    def g(pp,ii,jj):
        index_ii = ii*d + np.arange(d)
        index_jj = jj*d + np.arange(d)
        return (pp[index_jj] - pp[index_ii]) / (np.linalg.norm(pp[index_jj] - pp[index_ii]) + 1e-15)

    # orthogonal projection matrix P_{g_{ij}}
    def P(g_ij):
        g_ij = g_ij.reshape((-1, 1)) #here reshape because from row array i.e. [1,0] we want col array i.e. [[1], [0]] 
        return np.identity(d) - g_ij@(g_ij.T)

    GG = np.zeros((NN, NN, d), dtype=np.float32)
    Pg_star = np.zeros((NN, NN, d, d), dtype=np.float32)

    for ii in range(NN):
        for jj in range(NN):
            g_star = g(PP, ii, jj)
            GG[ii, jj, :] = g_star
            Pg_star[ii, jj, :] = P(g_star)

    # TODO: when writing report, use this to demonstrate antisymmetry
    """ is_GG_antisym = GG+np.transpose(GG, axes= (1, 0, 2))
    print(is_GG_antisym)
    print(GG)
    exit() """

    # ER Network generation
    p_ER = 0.9

    I_NN = np.identity(NN, dtype=int)
    I_nx = np.identity(d, dtype=int)
    I_NN_nx = np.identity(d*NN, dtype=int)
    O_NN = np.ones((NN,1), dtype=int)

    while 1:
        Adj = np.random.binomial(1, p_ER, (NN, NN))
        Adj = np.logical_or(Adj, Adj.T)
        Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

        # test connectivity
        test = np.linalg.matrix_power((I_NN+Adj),NN)
        
        if np.all(test>0):
            print("the graph is connected\n")
            break

    # Adj = np.asarray([
    # 	[0,     1,      1,		1],
    # 	[1,     0,      1,    	0],
    # 	[1,     1,      0,    	1],     
    # 	[1,     0,      1,    	0]
    # ])
    DEGREE = np.sum(Adj, axis=0)
    D_IN = np.diag(DEGREE)
    # Laplacian Matrix
    L_IN = D_IN - Adj.T

    
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
    print(x_init)
    for ii in range(NN):

        bearing_ii = Pg_star[:, ii, :, :].flatten().tolist()

        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*d + np.arange(d)
        x_init_ii = np.concatenate((
            x_init[ii_index].flatten(),
            x_init[ii_index + NN*d].flatten()),
            axis=0
        ).tolist()    

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
        # RVIZ
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