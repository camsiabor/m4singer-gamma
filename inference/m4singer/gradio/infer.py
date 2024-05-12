import importlib
import re

import gradio as gr
import yaml
from gradio.components import Textbox, Dropdown

from inference.m4singer.base_svs_infer import BaseSVSInfer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import numpy as np
from inference.m4singer.gradio.share_btn import community_icon_html, loading_icon_html, share_js

class GradioInfer:
    def __init__(self, exp_name, inference_cls, title, description, article, example_inputs):
        self.exp_name = exp_name
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs
        pkg = ".".join(inference_cls.split(".")[:-1])
        cls_name = inference_cls.split(".")[-1]
        self.inference_cls = getattr(importlib.import_module(pkg), cls_name)

    def greet(self, singer, text, notes, notes_duration):
        PUNCS = '。？；：'
        sents = re.split(rf'([{PUNCS}])', text.replace('\n', ','))
        sents_notes = re.split(rf'([{PUNCS}])', notes.replace('\n', ','))
        sents_notes_dur = re.split(rf'([{PUNCS}])', notes_duration.replace('\n', ','))

        if sents[-1] not in list(PUNCS):
            sents = sents + ['']
            sents_notes = sents_notes + ['']
            sents_notes_dur = sents_notes_dur + ['']

        audio_outs = []
        s, n, n_dur = "", "", ""
        for i in range(0, len(sents), 2):
            if len(sents[i]) > 0:
                s += sents[i] + sents[i + 1]
                n += sents_notes[i] + sents_notes[i+1]
                n_dur += sents_notes_dur[i] + sents_notes_dur[i+1]
            if len(s) >= 400 or (i >= len(sents) - 2 and len(s) > 0):
                audio_out = self.infer_ins.infer_once({
                    'spk_name': singer,
                    'text': s,
                    'notes': n,
                    'notes_duration': n_dur,
                })
                audio_out = audio_out * 32767
                audio_out = audio_out.astype(np.int16)
                audio_outs.append(audio_out)
                audio_outs.append(np.zeros(int(hp['audio_sample_rate'] * 0.3)).astype(np.int16))
                s = ""
                n = ""
        audio_outs = np.concatenate(audio_outs)
        return (hp['audio_sample_rate'], audio_outs), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    def run(self):
        set_hparams(config=f'checkpoints/{self.exp_name}/config.yaml', exp_name=self.exp_name, print_hparams=False)
        infer_cls = self.inference_cls
        self.infer_ins: BaseSVSInfer = infer_cls(hp)
        example_inputs = self.example_inputs
        for i in range(len(example_inputs)):
            singer, text, notes, notes_dur = example_inputs[i].split('<sep>')
            example_inputs[i] = [singer, text, notes, notes_dur]

        singerList = \
            [
            'Tenor-1', 'Tenor-2', 'Tenor-3', 'Tenor-4', 'Tenor-5', 'Tenor-6', 'Tenor-7',
            'Alto-1', 'Alto-2', 'Alto-3', 'Alto-4', 'Alto-5', 'Alto-6', 'Alto-7',
            'Soprano-1', 'Soprano-2', 'Soprano-3',
            'Bass-1',  'Bass-2',  'Bass-3',
            ]

        css = """
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        """
        with gr.Blocks(css=css) as demo:
            gr.HTML("""<div style="text-align: center; margin: 0 auto;">
                          <div
                          style="
                              display: inline-flex;
                              align-items: center;
                              gap: 0.8rem;
                              font-size: 1.75rem;
                          "
                          >
                          <h1 style="font-weight: 900; margin-bottom: 10px; margin-top: 14px;">
                              M4Singer
                          </h1>
                          </div>
                        </div>
                        """
                    )
            gr.Markdown(self.description)
            with gr.Row():
                with gr.Column():
                    singer_l = Dropdown(choices=singerList, value=example_inputs[0][0], label="SingerID", elem_id="inp_singer")
                    inp_text = Textbox(lines=2, placeholder=None, value=example_inputs[0][1], label="input text", elem_id="inp_text")
                    inp_note = Textbox(lines=2, placeholder=None, value=example_inputs[0][2], label="input note", elem_id="inp_note")
                    inp_duration = Textbox(lines=2, placeholder=None, value=example_inputs[0][3], label="input duration", elem_id="inp_duration")
                    generate = gr.Button("Generate Singing Voice from Musical Score")
                with gr.Column(lem_id="col-container"):
                    singing_output = gr.Audio(
                        label="Result",
                        type="numpy",
                        elem_id="music-output",
                        interactive=True,
                        show_download_button=True,
                    )

                    with gr.Group(elem_id="share-btn-container"):
                        community_icon = gr.HTML(community_icon_html, visible=False)
                        loading_icon = gr.HTML(loading_icon_html, visible=False)
                        share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
            gr.Examples(examples=self.example_inputs,
                        inputs=[singer_l, inp_text, inp_note, inp_duration],
                        outputs=[singing_output, share_button, community_icon, loading_icon],
                        fn=self.greet,
                        cache_examples=True)
            gr.Markdown(self.article)
            generate.click(self.greet,
                               inputs=[singer_l, inp_text, inp_note, inp_duration],
                               outputs=[singing_output, share_button, community_icon, loading_icon],)
            share_button.click(None, [], [], _js=share_js)
        demo.queue().launch(share=False)


if __name__ == '__main__':
    gradio_config = yaml.safe_load(open('inference/m4singer/gradio/gradio_settings.yaml'))
    g = GradioInfer(**gradio_config)
    g.run()

