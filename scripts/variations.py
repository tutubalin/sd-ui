import math

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images
from modules.shared import opts, state
import modules.sd_samplers

from itertools import product, chain
import re

def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_processed = None

    state.job_count = len(xs) * len(ys)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state.job = f"{ix + iy * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed = cell(x, y)
            if first_processed is None:
                first_processed = processed

            res.append(processed.images[0])

    grid = images.image_grid(res, rows=len(ys))
    grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    first_processed.images = [grid]

    return first_processed

def interlace(a, b):
    for x, y in zip(a,b):
        yield x
        yield y
    yield a[-1]

class Script(scripts.Script):
    def title(self):
        return "Variations"

    def ui(self, is_img2img):
        gr.HTML('<br />')
        with gr.Row():
            with gr.Column():
                margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))

        return [margin_size]

    def run(self, p, margin_size):
        modules.processing.fix_seed(p)

        positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        
        parts = re.split(r'\[(.*?)\]', positive_prompt)
        static_parts = parts[::2]
        dynamic_parts = [[word.strip()for word in part.split('|')]for part in parts[1::2]]

        # print('Static parts: ', static_parts)
        # print('Dynamic parts: ', dynamic_parts)

        all_prompts = []
        print('Generated prompts:')
        for combimation in product(*dynamic_parts):

            # print('zip: ', *zip(static_parts, combimation))
            # print('chain: ', *chain(*zip(static_parts, combimation)))

            selected_prompts = ''.join(interlace(static_parts, combimation))

            # selected_prompts = ' '.join(sum(*zip(static_parts, combimation),()))

            print(' * '+selected_prompts)
            all_prompts.append(selected_prompts)

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

        p.prompt = all_prompts
        p.seed = [p.seed] * len(all_prompts)
        p.prompt_for_display = positive_prompt
        processed = process_images(p)

        grid = images.image_grid(processed.images, p.batch_size, rows=len(dynamic_parts[0]))
        # grid = images.draw_prompt_matrix(grid, processed.images[0].width, processed.images[0].height, dynamic_parts[0], margin_size)
        processed.images.insert(0, grid)
        processed.index_of_first_image = 1
        processed.infotexts.insert(0, processed.infotexts[0])

        if opts.grid_save:
            images.save_image(processed.images[0], p.outpath_grids, "prompt_matrix", extension=opts.grid_format, prompt=positive_prompt, seed=processed.seed, grid=True, p=p)

        return processed
