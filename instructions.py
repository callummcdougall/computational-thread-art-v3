import math
import os
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfMerger, PdfReader
from reportlab.lib.colors import black, blue, gray
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from coordinates import get_distance
from image_color import ROOT_PATH, Img
from misc import get_range_of_lines


def format_node(node_idx: int, marked: bool = False) -> tuple[str, str]:
    """
    Takes a node (between 0 and n_nodes) and returns instructions (i.e. side, node identifier, and parity).

    If marked=True then we put a caret before the parity number (this indicates whether the lines we're being shown are
    actually from the prev group, not the current group).
    """
    node = int(node_idx / 2)
    str_node = f"{node // 10: 3} {node % 10}"

    parity = node_idx % 2
    str_parity = ("^" if marked else " ") + str(parity)

    return str_node, str_parity


def generate_instructions_pdf(
    img: Img,
    line_dict: list[tuple[int, int]],
    font_size: int = 32,
    num_cols: int = 3,
    num_rows: int = 20,
    true_x: float = 0.58,
    save_dir: Path | None = None,
    filename: str | None = None,
    show_stats: bool = True,
    true_thread_diameter=0.25,
    font: str = "source-code-pro.ttf",
):
    img_name = img.args.name.split("/")[-1]
    save_dir = save_dir or img.save_dir
    filename = (filename or str(save_dir / f"lines-{img_name}"),)

    try:
        font_file = ROOT_PATH / "lines" / font
        font_name = font.split(".")[0].replace("-", " ").title()
        assert font_file.exists()
        prime_font = TTFont(font_name, str(font_file))
        pdfmetrics.registerFont(prime_font)
    except:
        font_name = ""

    # A4 = (596, 870)
    width = A4[0]
    height = A4[1]

    # Store instructions, in printable form
    lines = []

    total_nlines_so_far = 0
    total_nlines = sum([len(value) for value in line_dict.values()])

    if img.args.group_orders[0].isdigit():
        group_orders = img.args.group_orders
    else:
        d = {color_name[0]: str(idx) for idx, (color_name, color_value) in enumerate(img.palette.items())}
        group_orders = "".join([d[char] for char in img.args.group_orders])

    idx = 0
    curr_color = "not_a_color"
    color_count = {color: [0, 0] for color in set(group_orders)}
    group_len_list = []
    while idx < len(group_orders):
        if group_orders[idx] == curr_color:
            group_len_list[-1][1] += 1
        else:
            group_len_list.append([group_orders[idx], 1])
            curr_color = group_orders[idx][0]
            color_count[group_orders[idx]][1] += 1
        idx += 1

    # group_len_list looks like [['0', 3], ['3', 1], ['1', 2], ... for ███ █ ██ ...
    for group_idx, (color_idx, group_len) in enumerate(group_len_list):
        # Figure out how many groups of this particular colour there are, and which group we're currently on
        n_groups = sum([j[1] for j in group_len_list if j[0] == color_idx])
        group_start = sum([j[1] for j in group_len_list[:group_idx] if j[0] == color_idx])
        group_end = group_start + group_len

        # We'll have some small group overlap, this helps keep track between different groups of the same color
        group_overlap = 5 if group_start != 0 else 0

        # Get the color and color description, from the first letter of the color description (img.e. color_idx="0" gets (0,0,0), "black")
        color_name = list(img.palette.keys())[int(color_idx)]

        # Get the line range we need to draw
        lines_colorgroup = line_dict[color_name]
        line_range = get_range_of_lines(
            n_lines=len(lines_colorgroup),
            n_groups=n_groups,
            group_number=(group_start, group_end),
            start_offset=group_overlap,
        )

        # Add the start titles to the instructions
        color_count[color_idx][0] += 1
        thiscolor_group = f"{color_count[color_idx][0]}/{color_count[color_idx][1]}"
        lines += [
            "=================",
            f"ByNow = {total_nlines_so_far}/{total_nlines}",
            f"ByEnd = {total_nlines_so_far + len(line_range)}/{total_nlines}",
            f"NOW   = {color_name} {thiscolor_group}",
            "=================",
        ]

        # If this is the first group of the color, then we should start by adding the very first one
        if group_start == 0:
            lines.append(format_node(lines_colorgroup[-1][-1]))

        # Get the points where we'll add in-group progress markers: min of (every 100 lines) and (5 times total)
        progress_gap = min(100, len(line_range) // 5)

        # Add all the instruction lines
        for raw_idx, line_idx in enumerate(line_range):
            # We mark them with a caret if it's in the first 5 nodes of a group which isn't the very first one
            marked = (raw_idx < group_overlap) and (group_start != 0)
            lines.append(format_node(lines_colorgroup[-line_idx][0], marked=marked))
            # We also add progress markers, every 100 lines (or more frequently)
            if (raw_idx % progress_gap == 0) and (0 < raw_idx < len(line_range) - 1):
                lines.append(f"LINES: {raw_idx}/{len(line_range)}")

        # Add the end titles to the instructions
        lines += ["=================", f"DONE  = {color_name} {thiscolor_group}"]
        total_nlines_so_far += len(line_range)

    group_page_list = []
    page_counter = 0

    while len(lines) > 0:
        next_lines, lines = lines[: num_rows * num_cols], lines[num_rows * num_cols :]

        filepath = save_dir / f"lines-{img.args.name.split('/')[-1]}-{page_counter}.pdf"
        canvas = Canvas(str(filepath), pagesize=A4)

        page_counter += 1

        canvas.setLineWidth(0.4)
        canvas.setStrokeGray(0.8)
        canvas.setStrokeColor(gray)

        for col_no, row_no in product(range(num_cols), range(num_rows)):
            x = 0.5 * cm + col_no * (width / num_cols)
            y = height * (1 - (1 + row_no) / (0.6 + num_rows))
            if next_lines:
                next_line = next_lines.pop(0)
            else:
                break

            if isinstance(next_line, str):
                # Case: next line is text that goes between instruction groups
                to = canvas.beginText()
                if font_name != "":
                    to.setFont(font_name, 14 if num_cols == 3 else 18)

                to.setTextOrigin(x, y)
                to.setFillColor(black)
                to.textLine(next_line)

                canvas.drawText(to)

                if next_line.startswith("NOW"):
                    group_page_list.append([page_counter, next_line[6:]])
            else:
                # Case 2: next line is an instruction
                tens, units = next_line

                to = canvas.beginText()
                if font_name != "":
                    to.setFont(font_name, font_size)  # "Symbola"

                to.setTextOrigin(x, y)
                to.setFillColor(black)
                to.textLine(tens)

                to.setTextOrigin(x + 2.45 * (font_size / 20) * cm, y)
                to.setFillColor(blue)
                to.textLine(units)

                canvas.drawText(to)

                canvas_path = canvas.beginPath()
                canvas_path.moveTo(x, y - 0.25 * (font_size / 20) * cm)
                canvas_path.lineTo(x + 3.65 * (font_size / 20) * cm, y - 0.25 * (font_size / 20) * cm)
                canvas.drawPath(canvas_path, fill=0, stroke=1)

        canvas.setLineWidth(3)
        canvas.setStrokeGray(0.5)
        canvas.setStrokeColor(black)
        canvas_path = canvas.beginPath()
        canvas_path.moveTo(width / num_cols, 0)
        canvas_path.lineTo(width / num_cols, height)
        canvas_path.moveTo(2 * width / num_cols, 0)
        canvas_path.lineTo(2 * width / num_cols, height)

        canvas.drawPath(canvas_path, fill=0, stroke=1)

        canvas.save()

    merger = PdfMerger()
    for g in range(page_counter):
        page_filename = save_dir / f"lines-{img.args.name.split('/')[-1]}-{g}.pdf"
        with open(page_filename, "rb") as f:
            merger.append(PdfReader(f))
        os.remove(page_filename)
    for idx, (pagenum, desc) in enumerate(group_page_list):
        merger.add_outline_item(title=f"{idx + 1}/{len(group_page_list)} {desc}", pagenum=pagenum - 1)

    pdf_filename = str(save_dir / f"{filename}.pdf")
    merger.write(pdf_filename)
    print(f"Wrote to {pdf_filename!r}")

    if show_stats:
        df_dicts = {}
        for color_name, lines in line_dict.items():
            nodes = [i[0] // 2 for i in lines]
            counter = Counter(nodes)
            node_frequencies = np.array([counter.get(i, 0) for i in range(max(img.args.d_coords) // 2)])
            node_frequencies_averaged = np.convolve(node_frequencies, np.ones(4), "valid") / 4
            df_dicts[color_name] = node_frequencies_averaged

        fig = px.line(
            pd.DataFrame(df_dicts),
            labels={"index": "node-pair", "value": "# lines", "variable": "Color"},
        )
        fig.update_layout(
            template="ggplot2",
            width=800,
            height=450,
            margin=dict(t=60, r=30, l=30, b=40),
            title_text="Frequencies of lines at nodes, by color",
        )
        fig.show()

        max_colorname_len = max(len(color_name) for color_name in line_dict.keys())
        image_x = max(coord[1] for coord in img.args.d_coords.values())
        image_y = max(coord[0] for coord in img.args.d_coords.values())
        sf = true_x / image_x
        true_y = image_y * sf
        for color_name, lines in line_dict.items():
            total_distance = sum(
                [
                    get_distance(
                        p0=img.args.d_coords[line[0]],
                        p1=img.args.d_coords[line[1]],
                    )
                    for line in lines
                ]
            )
            print(f"{color_name:{max_colorname_len}} | {total_distance * sf / 1000:.2f} km")

        n_buckets = 200

        max_len = 0
        for lines in line_dict.values():
            for line in lines:
                line_len = int(img.args.d_pixels[tuple(sorted(line))].shape[1])
                max_len = max(max_len, line_len)

        true_area_covered_by_thread = 0
        df_dicts = {}
        for color_name, lines in line_dict.items():
            df_dicts[color_name] = [0 for i in range(n_buckets + 1)]
            for line in lines:
                line_len = get_distance(
                    p0=img.args.d_coords[line[0]],
                    p1=img.args.d_coords[line[1]],
                )
                true_area_covered_by_thread += (line_len * (sf * 1000)) * true_thread_diameter
                line_len_bucketed = int(line_len * n_buckets / max_len)
                df_dicts[color_name][line_len_bucketed] += 1

        df_from_dict = pd.DataFrame(df_dicts, index=np.arange(n_buckets + 1) * 100 / n_buckets)

        fig = px.line(
            df_from_dict,
            labels={
                "index": "distance (as % of max)",
                "value": "# lines",
                "variable": "Color",
            },
        )
        fig.update_layout(
            template="ggplot2",
            width=800,
            height=450,
            margin=dict(t=60, r=30, l=30, b=40),
            title_text="Distance of lines, by color",
        )

        fig.add_vline(x=100 * min(image_x, image_y) / max_len, line_width=3)
        if img.args.shape == "Rectangle":
            fig.add_vline(x=100 * max(image_x, image_y) / max_len, line_width=3)
            fig.add_trace(
                go.Scatter(
                    x=[
                        100 * min(image_x, image_y) / max_len + 3,
                        100 * max(image_x, image_y) / max_len + 3,
                    ],
                    y=[
                        0.9 * df_from_dict.values.max(),
                        0.9 * df_from_dict.values.max(),
                    ],
                    text=["x", "y"] if image_x < image_y else ["y", "x"],
                    mode="text",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[100 * image_x / max_len + 3],
                    y=[0.9 * df_from_dict.values.max()],
                    text=["r"],
                    mode="text",
                )
            )
        fig["data"][-1]["showlegend"] = False  # type: ignore
        fig.show()

        true_area = 10e6 * true_x * true_y
        if img.args.shape == "Ellipse":
            true_area *= math.pi / 4
        print(f"""Total area covered (counting overlaps) = {true_area_covered_by_thread / true_area:.3f}

Baseline:
        
MDF: around 0.2 is enough. The stag_large sent to Ben was 0.209. This is with true diameter = 0.25, width 120cm, 10000 lines total. And I think down to 0.2 wouldn't have fucked it up, but smaller values might have.
        
Wheel: 0.264 is what I used for stag, probably 0.3 would have been better (it's got a transparent background!). This would mean about 7200 threads for a standard wheel.
""")
