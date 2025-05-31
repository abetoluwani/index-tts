import json
import os
import sys
import threading
import time

import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bigvgan_generator.pth",
    "bpe.model",    
    "gpt.pth",
    "config.yaml",
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="en_US")
MODE = 'local'
tts = IndexTTS(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),)


os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

def gen_single(prompt, text, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if infer_mode == "Standard Inference":
        output = tts.infer(prompt, text, output_path, verbose=cmd_args.verbose,
                           max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                           **kwargs)
    else:
        # Batch Inference
        output = tts.infer_fast(prompt, text, output_path, verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            sentences_bucket_max_size=(sentences_bucket_max_size),
            **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
</p>
    ''')
    with gr.Tab("Audio Generation"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label="Reference Audio",key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label="Text",key="input_text_single", placeholder="Please enter target text", info="Current model version: {}".format(tts.model_version or "1.0"))
                infer_mode = gr.Radio(choices=["Standard Inference", "Batch Inference"], label="Inference Mode",info="Batch Inference: Better for long sentences, doubles performance",value="Standard Inference")        
                gen_button = gr.Button("Generate Audio", key="gen_button",interactive=True)
            output_audio = gr.Audio(label="Generated Result", visible=True,key="output_audio")
        with gr.Accordion("Advanced Generation Parameters", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT2 Sampling Settings** _Parameters affect audio diversity and generation speed. See [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)_")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="Whether to perform sampling")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=600, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="Maximum number of tokens to generate. Too small may cause audio to be truncated.", key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown("**Sentence Settings** _Parameters affect audio quality and generation speed_")
                    with gr.Row():
                        max_text_tokens_per_sentence = gr.Slider(
                            label="Max Tokens per Sentence", value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_sentence",
                            info="Recommended range: 80-200. Higher values make longer sentences, lower values make shorter ones. Values too high or too low may reduce audio quality.",
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="Max Sentences in Bucket (Batch Inference)", value=4, minimum=1, maximum=16, step=1, key="sentences_bucket_max_size",
                            info="Recommended range: 2-8. Higher values process more sentences per batch. Too high may cause memory overflow.",
                        )
                    with gr.Accordion("Sentence Preview", open=True) as sentences_settings:
                        sentences_preview = gr.Dataframe(
                            headers=["Index", "Sentence Content", "Token Count"],
                            key="sentences_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence))
            data = []
            for i, s in enumerate(sentences):
                sentence_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, sentence_str, tokens_count])
            
            return {
                sentences_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=["Index", "Sentence Content", "Token Count"])
            return {
                sentences_preview: gr.update(value=df)
            }

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single, infer_mode,
                             max_text_tokens_per_sentence, sentences_bucket_max_size,
                             *advanced_params,
                     ],
                     outputs=[output_audio])


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=True)
