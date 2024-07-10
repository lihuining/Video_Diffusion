import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)

# ### text2video
# prompt = "A horse galloping on a street"
# params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

# out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
# model.process_text2video(prompt, fps = fps, path = out_path, **params)

# prompt = 'an astronaut dancing in outer space'
# motion_path = '__assets__/poses_skeleton_gifs/dance1_corr.mp4'
# out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.gif"
# model.process_controlnet_pose(motion_path, prompt=prompt, save_path=out_path)

prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = '__assets__/canny_videos_mp4/deer.mp4'
out_path = f'./text2video_edge_guidance_{prompt}.mp4'
model.process_controlnet_canny(video_path, prompt=prompt, save_path=out_path)
'''
pip install diffusers==0.26.3
pip install transformers==4.38.1
pip install accelerate==0.27.2
'''