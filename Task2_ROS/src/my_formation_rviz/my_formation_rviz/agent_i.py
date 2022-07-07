#
# 11/05/2022
# IN-LP
#
from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as msg_float

k_p = 10
k_v = 10

n_f = 2
NN = 4
d = 2

def writer(file_name, string):
    """
      inner function for logging
    """
    file = open(file_name, "a") # "a" is for append
    file.write(string)
    file.close()

class Agent(Node):
    def __init__(self):
        super().__init__('agent',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
            
        # Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.neigh = self.get_parameter('neigh').value
        
        dist = self.get_parameter('dist').value
        self.dist_ii = np.array(dist).reshape(4, 2, 2) # it returns an n_x by 1 array

        x_i = self.get_parameter('x_init').value
        self.n_x = len(x_i)
        self.x_i = np.array(x_i) # it returns an n_x by 1 array
        
        self.max_iters = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value

        self.tt = 0

        # create logging file
        self.file_name = "_csv_file/agent_{}.csv".format(self.agent_id)
        file = open(self.file_name, "w+") # 'w+' needs to create file and open in writing mode if doesn't exist
        file.close()

        # initialize subscription dict
        self.subscriptions_list = {}

        # create a subscription to each neighbor
        for j in self.neigh:
            topic_name = '/topic_{}'.format(j)
            self.subscriptions_list[j] = self.create_subscription(
                                                                msg_float, 
                                                                topic_name, 
                                                                lambda msg, node = j: self.listener_callback(msg, node), 
                                                                10)
        
        # create the publisher
        self.publisher_ = self.create_publisher(
                                                                msg_float, 
                                                                '/topic_{}'.format(self.agent_id),
                                                                10)

        self.timer = self.create_timer(self.communication_time, self.timer_callback)

        # initialize a dictionary with the list of received messages from each neighbor j [a queue]
        self.received_data = { j: [] for j in self.neigh }

        print("Setup of agent {} complete".format(self.agent_id))

    def listener_callback(self, msg, node):
        self.received_data[node].append(list(msg.data))

    def timer_callback(self):
        # Initialize a message of type float
        msg = msg_float()

        if self.tt == 0: # Let the publisher start at the first iteration
            msg.data = [float(self.tt)]

            # for element in self.x_i: 
            #     msg.data.append(float(element))
            [msg.data.append(float(element)) for element in self.x_i] # msg.data: [tt, p_x, p_y, v_x, v_y]
            
            self.publisher_.publish(msg)
            self.tt += 1

            # log files
            # 1) visualize on the terminal
            string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
            print("Iter = {} \t Value = {}".format(int(msg.data[0]), string_for_logger))

            # 2) save on file
            data_for_csv = msg.data.tolist().copy()
            data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]]
            data_for_csv = ','.join(data_for_csv)
            writer(self.file_name,data_for_csv+'\n')

        else: 
            # Check if lists are nonempty
            all_received = all(self.received_data[j] for j in self.neigh) # check if all neighbors' have been received

            sync = False
            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(self.tt-1 == self.received_data[j][0][0] for j in self.neigh) # True if all True

            if sync:
                DeltaT = self.communication_time/10
                self.formation_update(DeltaT)
                
                # publish the updated message
                msg.data = [float(self.tt)]
                [msg.data.append(float(element)) for element in self.x_i]
                self.publisher_.publish(msg)

                # save data on csv file
                data_for_csv = msg.data.tolist().copy()
                data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]]
                data_for_csv = ','.join(data_for_csv)
                writer(self.file_name,data_for_csv+'\n')

                string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
                print("Iter = {} \t Value = {}".format(int(msg.data[0]), string_for_logger))
                
                # Stop the node if tt exceeds MAXITERS
                if self.tt > self.max_iters:
                    print("\nMAXITERS reached")
                    sleep(3) # [seconds]
                    self.destroy_node()

                # update iteration counter
                self.tt += 1

    
    def formation_update(self, dt):
        """
        dt    = discretization step
        """
        dx = np.zeros(self.x_i.shape)

        u = np.zeros(d)
        for j in self.neigh:
            x_j = np.array(self.received_data[j].pop(0)[1:])
            pp = self.x_i[0:2] - x_j[0:2]
            vv = self.x_i[2:4] - x_j[2:4]
            pg_star = self.dist_ii[j]
            u -=  pg_star @ (k_p*pp + k_v*vv)

        dp = self.x_i[2:4]
        dv = [0,0] if (self.agent_id < 2) else u

        dx[0:2] = dp
        dx[2:4] = dv

        # Forward Euler
        self.x_i += dx*dt

def main(args=None):
    rclpy.init(args=args)

    agent = Agent()
    print("Agent {:d} -- Waiting for sync.".format(agent.agent_id))
    sleep(0.5)
    print("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()