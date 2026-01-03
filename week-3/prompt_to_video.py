

import gradio as gr
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import imageio
import numpy as np
from pathlib import Path

@torch.no_grad()
def load_model():
    model_id = "cerspense/zeroscope_v2_576w"

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    return pipe

pipe = load_model()
print('model loaded')

def generate_video(prompt, negative_prompt="", num_frames=16, num_inference_steps=25):
    try:
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=256,
            width=256,
            guidance_scale=9.0
        ).frames[0]
        output_path = "output_video.mp4"
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
        imageio.mimsave(output_path, video_frames, fps=8)

        return output_path

    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Text to Video Generator") as demo:
    gr.Markdown("# Text to Video Generator")
    gr.Markdown("Generate short videos from text descriptions using AI.")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="A cat walking on the beach at sunset...",
                lines=3
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="blurry, low quality...",
                lines=2
            )

            with gr.Row():
                num_frames = gr.Slider(
                    minimum=8,
                    maximum=24,
                    value=16,
                    step=8,
                    label="Number of Frames"
                )
                num_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=25,
                    step=5,
                    label="Inference Steps"
                )

            generate_btn = gr.Button("Generate Video", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Video")

    gr.Examples(
        examples=[
            ["A golden retriever playing in the snow"],
            ["Astronaut riding a horse in space"],
            ["Fireworks exploding over a city at night"],
            ["A butterfly landing on a flower in slow motion"]
        ],
        inputs=prompt_input
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[prompt_input, negative_prompt_input, num_frames, num_steps],
        outputs=video_output
    )
demo.launch(share=True, debug=True)

 