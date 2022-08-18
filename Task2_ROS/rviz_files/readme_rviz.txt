- Preliminaries:

1. Copy visualizer.py in package_name/package_name
2. Add 'visualizer' in package_name/setup.py
3. Copy rviz_config.rviz in package_name/resource
4. Add '('share/' + package_name, glob('resource/*.rviz')),' below '(.../*.launch.py)),' in package_name/setup.py
4bis. Remember the comma
5. Add geometry_msgs and visualization_msgs as <exec_depend> in package_name/package.xml
6. Compile

- Modify the subscription in visualizer.py: select the topic you want to visualize

- Modify the launcher file:

1. import some libraries

    import os
    from ament_index_python.packages import get_package_share_directory

2. Open RVIZ

    ################################################################################
    # RVIZ
    ################################################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('package_name')
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

3 Launch the visualizer node

    ################################################################################
    # RVIZ
    ################################################################################

    launch_description.append(
        Node(
            package='package_name', 
            node_namespace='agent_{}'.format(ii),
            node_executable='visualizer', 
            parameters=[{
                            'agent_id': ii,
                            'communication_time': COMM_TIME,
                            }],
        ))

4. Compile and launch
