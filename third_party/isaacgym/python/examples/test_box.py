import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments()

# Configure the simulation
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = False
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Create a viewer (optional)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Create the environment
env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
env = gym.create_env(sim, env_lower, env_upper, 1)

# Create a cube asset (box)
box_size = 0.5
box_asset_options = gymapi.AssetOptions()
box_asset_options.fix_base_link = False
box_asset_options.disable_gravity = False
box_asset = gym.create_box(sim, box_size, box_size, box_size, box_asset_options)

box_rigid_shape_props = gym.get_asset_rigid_shape_properties(box_asset)
box_rigid_shape_props[0].friction = 0.1
gym.set_asset_rigid_shape_properties(box_asset, box_rigid_shape_props)

# Define the initial pose for the cube
box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(0.0, 0.0, box_size / 2)  # Placing the cube just above the ground

# Add the box (cube) to the environment
box_handle = gym.create_actor(env, box_asset, box_pose, "box", 0, 1)

# Apply a force to the cube to make it move
# Define the force vector (in Newtons)
force = gymapi.Vec3(1000.0, 0.0, 1000.0)  # Apply force along the x-axis
torque  = gymapi.Vec3(0.0, 0.0, 0.0)

# Apply the force at the center of mass of the cube
gym.apply_body_forces(env, box_handle, force=force, torque=torque, space=gymapi.CoordinateSpace.ENV_SPACE)

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    gym.apply_body_forces(env, box_handle, force=force, torque=torque, space=gymapi.CoordinateSpace.ENV_SPACE)
    
    # Step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # Wait for dt to elapse in real time (synchronizes the physics simulation with the rendering rate)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
