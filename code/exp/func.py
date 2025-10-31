def generate_circular_trajectories(arena_map, R_out, R_in,
                                    mean_vel, std_vel, # Angular veloity
                                    time_points=100, batch_size=1, visualize=True):
    import numpy as np
    from matplotlib import pyplot as plt
    
    dim = arena_map.shape[0]
    y_c, x_c = int(dim/2), int(dim/2)
    loop_width = R_out - R_in
    avg_R = (R_out + R_in) / 2
    edge = loop_width / 6

    # Initiate trajectories
    trajectories = np.zeros((batch_size, time_points, 2))
    displacements = np.zeros((batch_size, time_points, 2))
    hds = np.zeros((batch_size, time_points, 1))

    current_batch = 0
    while current_batch < batch_size:
        i = current_batch

        # Initial positions
        R_0 = np.random.uniform(R_in+edge, R_out-edge)
        theta_0 = np.random.uniform(0, 2*np.pi)

        # Angular velocity generation: omega(t)
        omega = np.random.normal(mean_vel, std_vel, time_points) / avg_R
        # Angular displacement: theta(t)
        theta = np.cumsum(omega) + theta_0

        # Insert variations
        delta_R = (edge/2) * np.sin(np.linspace(0, np.pi*4, time_points)) # Narrow the range of R to ensure not hitting the wall
        # Radial displacement: R(t)
        R = R_0 + delta_R

        # Simulate trajectory
        x_traj = x_c + R*np.cos(theta)
        y_traj = y_c + R*np.sin(theta)

        # Add noise
        x_traj += np.random.uniform(-1, 1, size=time_points)
        y_traj += np.random.uniform(-1, 1, size=time_points)

        # Ensure all sampled points are valid
        valid_mask = ( (0 <= x_traj) & (x_traj < dim) &
                       (0 <= y_traj) & (y_traj < dim) &
                       (arena_map[y_traj.astype(int), x_traj.astype(int)] == 0) )
        if np.all(valid_mask):
            # print(f"Trajectory {i+1} is valid.")

            trajectories[i, :, 0] = x_traj
            trajectories[i, :, 1] = y_traj
            # Generate displacements
            dx, dy = np.diff(x_traj), np.diff(y_traj)
            displacements[i, :len(x_traj)-1, 0] = dx
            displacements[i, :len(y_traj)-1, 1] = dy
            # Generate head directions
            hds[i, :len(x_traj)-1, 0] = np.arctan2(dy, dx)

            current_batch += 1
        else:
            pass

    # Visualize
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(arena_map, cmap="gray_r", origin="lower")  # 显示 arena
        for i in range(batch_size):
            plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], label=f'Trajectory {i+1}')
        plt.scatter([x_c], [y_c], c='blue', marker='o', label="Center")  # 圆心
        plt.legend()
        plt.title("Circular Trajectories in Arena")
        plt.show()

    # Return trajectories data
    Traj = {'coords': trajectories, 'hds': hds, 'disps': displacements}

    return Traj