try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym.torch_utils import *
except:
    print("Cannot import isaacgym, works only for deployment")

import numpy as np
import torch
import os
from termcolor import cprint

import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

from modules.feature_extractor import DepthFeatureExtractor
from modules.conv2d import Conv2dHeadModel
from modules.feature_extractor import DepthOnlyFCBackbone58x87, CNNFeatureExtractor, DepthOnlyFCBackbone54x96, GuidedTransformerBlock, SharedCNNBackbone, PositionalEncoding

from learning.dagger_trainer import DAggerTrainer
from learning.dagger_rnn import DAGGER_DEFAULT_CONFIG as DAGGER_RNN_DEFAULT_CONFIG, DAgger_RNN
from learning.dagger import DAGGER_DEFAULT_CONFIG, DAgger

# -------------------- Policy with Transformer --------------------
class Policy(DeterministicMixin, Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 img_height=54,
                 img_width=96,
                 frames=3,
                 modalities=2,
                 views=2,
                 vision_dim=64,
                 robot_state_dim=61):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.img_h = img_height
        self.img_w = img_width
        self.frames = frames
        self.modalities = modalities
        self.views = views
        self.vision_dim = vision_dim
        self.state_dim = robot_state_dim
        self.action_dim = action_space.shape[0]

        # Shared CNN for mask+depth per frame
        self.shared_cnn = SharedCNNBackbone(feature_dim=vision_dim)
        # Transformers for arm and base
        self.transformer_arm = GuidedTransformerBlock(feature_dim=vision_dim)
        self.transformer_base = GuidedTransformerBlock(feature_dim=vision_dim)
        # MLPs to unify dims before concat
        self.arm_proj = nn.Linear(vision_dim, vision_dim)
        self.base_proj = nn.Linear(vision_dim, vision_dim)
        # Final action head
        fusion_dim = vision_dim * 2 + robot_state_dim
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, self.action_dim)
        )

        self.residual_mlp = nn.Sequential(
            nn.Linear(fusion_dim + 9, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions)
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states_raw = inputs['states']  # [B, ...]
        # assume last state_dim dims are robot state
        robot_state = states_raw[..., -self.state_dim:]
        # the rest are images flattened
        img_flat = states_raw[..., :-self.state_dim]
        B = img_flat.shape[0]
        # reshape to [B, views*frames*modalities, H, W]
        c = self.views * self.frames * self.modalities  # c=12
        # print(f"states_raw.shape is {states_raw.shape}, B is {B}")
        images = img_flat.view(B, c, self.img_h, self.img_w)
        
        # base_imgs = images[:, [0, 2, 4, 6, 8, 10], :, :]  
        # arm_imgs = images[:, [1, 3, 5, 7, 9, 11], :, :]   
        if self.frames == 1:
            base_imgs = images[:, [0, 2], :, :]  
            arm_imgs = images[:, [1, 3], :, :] 
        if self.frames == 3:
            base_imgs = images[:, [0, 2, 4, 6, 8, 10], :, :]  
            arm_imgs = images[:, [1, 3, 5, 7, 9, 11], :, :] 
        if self.frames == 5:
            base_imgs = images[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], :, :]  
            arm_imgs = images[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], :, :] 
        if self.frames == 7:
            base_imgs = images[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26], :, :]  
            arm_imgs = images[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27], :, :] 


        def extract_feats(x):
            # x: [B, 6, H, W] -> split frames
            x = x.view(B * self.frames, self.modalities, self.img_h, self.img_w)
            feats = self.shared_cnn(x)            # [B*frames, D]
            feats = feats.view(B, self.frames, -1)  # [B, frames, D]
            return feats

        arm_seq = extract_feats(arm_imgs)
        base_seq = extract_feats(base_imgs)
        # transformer with guidance
        arm_feat = self.transformer_arm(arm_seq, robot_state)
        base_feat = self.transformer_base(base_seq, robot_state)
        # unify dims
        arm_feat = self.arm_proj(arm_feat)
        base_feat = self.base_proj(base_feat)
        # fuse and predict
        fusion = torch.cat([arm_feat, base_feat, robot_state], dim=1)
        actions = self.action_head(fusion)
        # residual actions
        residual_actions = self.residual_mlp(torch.cat([fusion, actions], dim=1))
        
        actions += residual_actions
        
        return actions, residual_actions, {}
 
def create_env(cfg, args, mode, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor ,intervel):
    from envs import B1Z1PickMulti, B1Z1Float
    import utils.wrapper as wrapper
    from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
    cfg_terrain = B1Z1RoughCfg()


    cfg["sensor"]["enableCamera"] = True
    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = mode
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)
    robot_start_pose = (-2.00, 0, 0.55)
    if args.eval:
        robot_start_pose = (-1.5, 0, 0.55)
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless, 
                         use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands, robot_start_pose=robot_start_pose,
                         no_feature=args.no_feature, mask_arm=args.mask_arm, depth_random=args.depth_random, stu_distill=True, pitch_control=args.pitch_control, pred_success=args.pred_success,
                         rand_control=args.rand_control, arm_delay=args.arm_delay, rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip, stop_pick=args.stop_pick, arm_kp=args.arm_kp, arm_kd=args.arm_kd, table_height=args.table_height, eval=args.eval,
                         camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor, intervel=intervel,  cfg_terrain = cfg_terrain)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

def get_predict_point(camera_6p_info=None, cube_predict_info_path=None, cube_root_states_info_path=None, num_env=None, intervel=None, delta_height=None):
    import json
    import os
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir,"PredictPiontTransform" , "transform_info")
    cube_root_states_info_path = os.path.join(data_dir, cube_root_states_info_path)
    save_dir = os.path.join(data_dir, "grasp_videos")
    os.makedirs(save_dir, exist_ok=True)

    # load cubes's states（position + quaternion + velocity...）
    cube_states = torch.load(cube_root_states_info_path)[0].cpu()
    cprint(f"cube_states.shape is {cube_states.shape}", "yellow")

    cube_pos = cube_states[:, :3]
    cube_quat = cube_states[:, 3:7]
    cube_rpy = torch.tensor(R.from_quat(cube_quat.numpy()).as_euler('xyz'))  # turn to RPY
    cube_init_tensor = torch.cat([cube_pos, cube_rpy], dim=1)
    cube_init_tensor = cube_init_tensor.to(torch.float32)

    base_n = cube_init_tensor.shape[0]  
    T_grasp_cv_all = []

    # read base_num sample's all transform
    for i in range(base_n):
        sample_dir = os.path.join(data_dir, cube_predict_info_path)
        json_path = os.path.join(sample_dir, f"predictions_image_{(i):02d}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        grasp_matrices = np.array(data['transform'])  
        T_grasp_cv_all.append(grasp_matrices)

    # turn to tensor，shape is [base_n, n, 4, 4]   # base_n=cube_onject_num=30, n=num_grasptransform_per_object 
    T_grasp_cv_all = np.stack(T_grasp_cv_all, axis=0)
    n_grasps = T_grasp_cv_all.shape[1]  # get n
    T_grasp_cv_all = torch.tensor(T_grasp_cv_all, dtype=torch.float32)

    # extent to num_env
    T_grasp_cv_list = [T_grasp_cv_all[i % base_n] for i in range(num_env)]
    grasp_cv_tensor = torch.stack(T_grasp_cv_list, dim=0)  # 形状 [num_env, n, 4, 4]

    camera_tensor = []
    cube_init_list = []
    cube_init_ori_list = []

    for i in range(num_env):
        base_idx = i % base_n
        height_offset = (delta_height / intervel) * ((i % intervel) + 1)

        # the initial pose of the camera, x,y is related to the initial pose of the float base 
        camera_x = -0.3
        camera_y = 2.0
        camera_z = 0.80 + height_offset
        camera_info = [camera_x, camera_y, camera_z, 0.0, 0.70, 0.0]
        camera_tensor.append([camera_info] * n_grasps)

        cube_init = cube_init_tensor[base_idx].clone()
        cube_init[2] += height_offset 
        cube_init_list.append([cube_init] * n_grasps)
        cube_init_ori_list.append(cube_init)

    camera_tensor = torch.tensor(camera_tensor, dtype=torch.float32)  # [num_env, n, 6]
    cube_init_tensor = torch.stack([torch.stack(cube_init, dim=0) for cube_init in cube_init_list], dim=0)  # [num_env, n, 6]
    return camera_tensor, grasp_cv_tensor, cube_init_tensor

def get_trainer(is_eval=False):
    from utils.config import load_cfg, get_params, copy_cfg
    
    args = get_params()
    set_seed(args.seed)
    args.eval = is_eval
    use_roboinfo = args.roboinfo
    if args.wrist_seg:
        mode = "wrist_seg"
    elif args.front_only:
        mode = "front_only"
    elif args.seperate:
        mode = "seperate"
    else:
        mode = "full"
    
    # assert use_roboinfo, "Are you sure not using roboinfo?" # TODO: temporarily for reminder
    args.wandb = args.wandb and (not args.eval) and (not args.debug)
    
    cfg_file = "DQ_stu.yaml"
    file_path = "data/cfg/" + cfg_file
    
    if args.resume:
        experiment_dir = os.path.join(args.experiment_dir, args.wandb_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        pt_files = os.listdir(checkpoint_dir)
        pt_files = [file for file in pt_files if file.endswith(".pt") and (not file.startswith("best"))]
        # Find the latest checkpoint
        checkpoint_steps = 0
        if len(pt_files) > 0:
            args.checkpoint = os.path.join(checkpoint_dir, sorted(pt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1])
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        cfg_files = os.listdir(experiment_dir)
        cfg_files = [file for file in cfg_files if file.endswith(".yaml")]
        if len(cfg_files) > 0:
            cfg_file = cfg_files[0]
            file_path = os.path.join(experiment_dir, cfg_file)
        
        print("Find the latest checkpoint: ", args.checkpoint)
        
    cfg = load_cfg(file_path)
    cprint(f"Using config file: {file_path}","red" )
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob
    if args.eval or args.debug:
        cfg['env']['numEnvs'] = 200
        if args.checkpoint:
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
            cfg["env"]["globalStepCounter"] = checkpoint_steps

    """
    To clarify, when capturing images of each object, the images were taken from a car. 
    The z-height of each environment's car was determined by the intervel. 
    To ensure correct camera pose for each cube,
    the intervel needed to be aligned here. This is a legacy issue and will not be used further.
    Since we no longer use the car as a moving base, this intervel has no effect in the subsequent training code; 
    it is only used when initializing the camera pose.
    """

    intervel = 23
    args.intervel = intervel
    cube_predict_info_path = "contact_grasp_info_mul"
    cube_root_states_info_path = "30all_nomove_cube_root_states5.pt"
    camera_6p_tensor, grasp_cv_tensor, cube_init_tensor = get_predict_point(cube_predict_info_path=cube_predict_info_path
                                                                            ,cube_root_states_info_path=cube_root_states_info_path,num_env=cfg['env']['numEnvs'],
                                                                            intervel=intervel,delta_height=0.1)
    env_num = cfg['env']['numEnvs']
    cprint(f"env_num is {env_num}", "yellow")
    cprint(f"camera_6p_tensor.shape is {camera_6p_tensor.shape}", "yellow")
    env = create_env(cfg=cfg, args=args, mode=mode,  camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor,intervel=intervel)
    device = env.rl_device
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device) # 原本是24

    student_action_space = env.action_space
    student_obs_space = env.observation_space
    if args.pred_success:
        student_action_space = (student_action_space.shape[0]+1,)
    
    model_dagger = {}
    # model_dagger["policy"] = Policy(student_obs_space, student_action_space, device, num_envs=env.num_envs, mode=mode, use_roboinfo=use_roboinfo, use_tanh=args.use_tanh, use_gru=not args.mlp_stu, pitch_control=args.pitch_control, floating_base=cfg["env"].get("floatingBase", False))
    model_dagger["policy"] = Policy(student_obs_space, student_action_space, device)
    
    dagger_config = DAGGER_DEFAULT_CONFIG if args.mlp_stu else DAGGER_RNN_DEFAULT_CONFIG
    
    cfg_dagger = dagger_config.copy()
    cfg_dagger["rollouts"] = 24  
    cfg_dagger["learning_epochs"] = 5
    cfg_dagger["mini_batches"] = 6  
    cfg_dagger["discount_factor"] = 0.99
    cfg_dagger["lambda"] = 0.95
    cfg_dagger["learning_rate"] = 5e-5
    cfg_dagger["learning_rate_scheduler"] = None
    cfg_dagger["random_timesteps"] = 0
    cfg_dagger["learning_starts"] = 0
    cfg_dagger["grad_norm_clip"] = 1.0
    cfg_dagger["ratio_clip"] = 0.2
    cfg_dagger["value_clip"] = 0.2
    cfg_dagger["clip_predicted_values"] = True
    cfg_dagger["value_loss_scale"] = 1.0
    cfg_dagger["kl_threshold"] = 0
    cfg_dagger["rewards_shaper"] = None
    cfg_dagger["experiment"]["write_interval"] = 24
    cfg_dagger["experiment"]["checkpoint_interval"] = 1000
    cfg_dagger["experiment"]["directory"] = args.experiment_dir
    cfg_dagger["experiment"]["experiment_name"] = args.wandb_name
    cfg_dagger["experiment"]["wandb"] = args.wandb
    # Train a fixed base policy
    cfg_dagger["fixed_base"] = args.fixed_base
    cfg_dagger["reach_only"] = args.reach_only
    cfg_dagger["pred_success"] = args.pred_success
    if args.wandb:
        cfg_dagger["experiment"]["wandb_kwargs"] = {"project": args.wandb_project, "tensorboard": False, "name": args.wandb_name}
    
    if args.mlp_stu:
        agent = DAgger(models=model_dagger,
                    memory=memory,
                    cfg=cfg_dagger,
                    observation_space=student_obs_space,
                    action_space=env.action_space,
                    state_space=env.state_space,
                    device=device)
    else:
        agent = DAgger_RNN(models=model_dagger,
                memory=memory,
                cfg=cfg_dagger,
                observation_space=student_obs_space,
                action_space=env.action_space,
                state_space=env.state_space,
                device=device)
    
    from train_multistate_DQ_teacher import Policy as PPOPolicy
    from train_multistate_DQ_teacher import Value as PPOValue
    from train_multistate_DQ_teacher import PPO_DEFAULT_CONFIG, PPO

    teacher_action_space = env.action_space
    teacher_obs_space = env.observation_space
    # ------------------- PPO CONFIG -------------------
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 24  # memory_size
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 6  # 24 * 8192 / 32768
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 4.2e-4 ## my change ##
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 1.0
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0
    cfg_ppo["rewards_shaper"] = None
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": teacher_obs_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    ppo_models = {}
    ppo_models["policy"] = PPOPolicy(teacher_obs_space, teacher_action_space, device, num_features=1024, encode_dim=128, camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor ,clip_actions=args.use_tanh, deterministic=True)
    ppo_models["value"] = PPOValue(teacher_obs_space, teacher_action_space, device, num_features=1024, encode_dim=128, camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor)
    ppo_agent = PPO(models=ppo_models,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=teacher_obs_space,
                action_space=teacher_action_space,
                device=device)
        
    if not args.eval:
        print("load teacher ckpt: ", args.teacher_ckpt_path)
        ppo_agent.load(args.teacher_ckpt_path)
    
    # cfg_trainer = {"timesteps": args.timesteps, "headless": True, "teacher_pretrain": True}
    cfg_trainer = {"timesteps": args.timesteps, "teacher_pretrain": True}
    cfg_trainer["pretrain_timesteps"] = 8000 if args.depth_random else 4000
    if args.checkpoint:
        print("Resuming from checkpoint: ", args.checkpoint)
        agent.load(args.checkpoint)
        checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        if args.record_video:
            experiment_dir = args.checkpoint.split("/")[0]
            wandb_name = args.checkpoint.split("/")[1]
            cfg_trainer["video_name"] = wandb_name +"-"+str(checkpoint_steps)
            cfg_trainer["log_dir"] = experiment_dir
            cfg_trainer["record_video"] = True
        if not args.eval:
            cfg_trainer["initial_timestep"] = checkpoint_steps
        
    trainer = DAggerTrainer(cfg=cfg_trainer, env=env, agents=agent, teacher_agents=ppo_agent)
    if args.wandb:
        import wandb
        wandb.save("data/cfg/" + cfg_file, policy="now")
        wandb.save("envs/b1z1_" + args.task[4:].lower() + ".py", policy="now")
        wandb.save("train_multi_bc_deter_full.py", policy="now")
    if not args.eval:
        if not os.path.exists(os.path.join(args.experiment_dir, args.wandb_name, cfg_file)):
            copy_cfg(file_path, os.path.join(args.experiment_dir, args.wandb_name))
    return trainer
    
    
if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train()
