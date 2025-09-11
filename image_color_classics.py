import math
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyperclip
from IPython.display import clear_output
from PIL import Image, ImageDraw


def generate_hooks(n_hooks: int, wheel_pixel_size: int):
    r = (wheel_pixel_size / 2) - 1

    theta = (np.arange(2 * n_hooks, dtype="float64") / (2 * n_hooks)) * (2 * np.pi)
    # theta = (np.arange(n_hooks, dtype="float64") / n_hooks) * (2 * np.pi)
    # epsilon = np.arcsin(hook_pixel_size / wheel_pixel_size)
    # theta_acw = theta.copy() + epsilon
    # theta_cw = theta.copy() - epsilon
    # theta = np.stack((theta_cw, theta_acw)).ravel("F")

    x = r * (1 + np.cos(theta)) + 0.5
    y = r * (1 + np.sin(theta)) + 0.5

    return np.array((x, y)).T


def through_pixels(p0, p1):
    d = max(int(((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5), 1)

    pixels = p0 + (p1 - p0) * np.array([np.arange(d + 1), np.arange(d + 1)]).T / d
    pixels = np.unique(np.round(pixels), axis=0).astype(int)

    return pixels


def build_through_pixels_dict(hooks, n_hooks):
    n_hook_sides = n_hooks * 2

    l = [(0, 1)]
    for j in range(n_hook_sides):
        for i in range(j):
            if j - i > 10 and j - i < (n_hook_sides - 10):
                l.append((i, j))

    random_order = np.random.choice(len(l), len(l), replace=False)

    d = {}
    t0 = time.time()

    for n in range(len(l)):
        (i, j) = l[random_order[n]]
        p0, p1 = hooks[i], hooks[j]
        d[(i, j)] = through_pixels(p0, p1)

        t = time.time() - t0
        t_left = t * (len(l) - n - 1) / (n + 1)
        print(f"time left = {time.strftime('%M:%S', time.gmtime(t_left))}", end="\r")

    clear_output()
    return d


def fitness(image, through_pixels_dict, line, darkness, lp, w, w_pos, w_neg, line_norm_mode):
    pixels = through_pixels_dict[tuple(sorted(line))]

    old_pixel_values = image[tuple(pixels.T)]
    new_pixel_values = old_pixel_values - darkness

    if isinstance(w, bool) and isinstance(w_pos, bool):
        new_penalty = new_pixel_values.sum() - (1 + lp) * new_pixel_values[new_pixel_values < 0].sum()
        old_penalty = old_pixel_values.sum() - (1 + lp) * old_pixel_values[old_pixel_values < 0].sum()
    elif isinstance(w_pos, bool):
        pixel_weightings = w[tuple(pixels.T)]
        new_w_pixel_values = new_pixel_values * pixel_weightings
        old_w_pixel_values = old_pixel_values * pixel_weightings
        new_penalty = new_w_pixel_values.sum() - (1 + lp) * new_w_pixel_values[new_pixel_values < 0].sum()
        old_penalty = old_w_pixel_values.sum() - (1 + lp) * old_w_pixel_values[old_pixel_values < 0].sum()
    elif isinstance(w, bool):
        pos_pixel_weightings = w_pos[tuple(pixels.T)]
        neg_pixel_weightings = w_neg[tuple(pixels.T)]
        new_wpos_pixel_values = new_pixel_values * pos_pixel_weightings
        new_wneg_pixel_values = new_pixel_values * neg_pixel_weightings
        old_wpos_pixel_values = old_pixel_values * pos_pixel_weightings
        old_wneg_pixel_values = old_pixel_values * neg_pixel_weightings
        new_penalty = (
            new_wpos_pixel_values[new_pixel_values > 0].sum() - lp * new_wneg_pixel_values[new_pixel_values < 0].sum()
        )
        old_penalty = (
            old_wpos_pixel_values[old_pixel_values > 0].sum() - lp * old_wneg_pixel_values[old_pixel_values < 0].sum()
        )

    if line_norm_mode == "length":
        line_norm = len(pixels)
    elif line_norm_mode == "weighted length":
        if isinstance(w_pos, bool):
            line_norm = pixel_weightings.sum()
        else:
            line_norm = pos_pixel_weightings.sum()
    elif line_norm_mode == "none":
        line_norm = 1

    if line_norm == 0:
        return 0
    else:
        return (old_penalty - new_penalty) / line_norm


def optimise_fitness(
    image,
    through_pixels_dict,
    previous_edge,
    darkness,
    lightness_penalty,
    w,
    w_pos,
    w_neg,
    line_norm_mode,
    time_saver,
    flip_parity,
    shortest_line,
    n_hooks,
):
    if not flip_parity:
        starting_edge = previous_edge
    elif previous_edge % 2 == 0:
        starting_edge = previous_edge + 1
    else:
        starting_edge = previous_edge - 1

    sides_A = np.ones(n_hooks * 2) * starting_edge
    sides_B = np.arange(n_hooks * 2)
    next_lines = np.stack((sides_A, sides_B)).ravel("F").reshape((n_hooks * 2, 2)).astype(int)
    mask = (np.abs(next_lines.T[1] - next_lines.T[0]) > shortest_line) & (
        np.abs(next_lines.T[1] - next_lines.T[0]) < n_hooks * 2 - shortest_line
    )
    next_lines = next_lines[mask]

    if time_saver == 1:
        next_lines = next_lines.tolist()
    else:
        next_lines = next_lines[
            np.random.choice(np.arange(len(next_lines)), int(len(next_lines) * time_saver))
        ].tolist()

    fitness_list = [
        fitness(image, through_pixels_dict, line, darkness, lightness_penalty, w, w_pos, w_neg, line_norm_mode)
        for line in next_lines
    ]
    best_line_idx = fitness_list.index(max(fitness_list))
    best_line = next_lines[best_line_idx]

    pixels = through_pixels_dict[tuple(sorted(best_line))]
    image[tuple(pixels.T)] -= darkness

    return image, best_line


def find_lines(
    image,
    through_pixels_dict,
    n_lines,
    darkness,
    lightness_penalty,
    line_norm_mode,
    n_hooks,
    wheel_pixel_size,
    w=False,
    w_pos=False,
    w_neg=False,
    time_saver=1,
    flip_parity=True,
    shortest_line: int = 40,
):
    list_of_lines = []
    previous_edge = np.random.choice(n_hooks * 2)

    image_copy = image.copy()

    for i in range(n_lines):
        if i == 0:
            t0 = time.time()
            initial_penalty = get_penalty(image_copy, lightness_penalty, w, w_pos, w_neg)
            initial_avg_penalty = f"{initial_penalty / (wheel_pixel_size**2):.2f}"
        elif i % 50 == 0:
            t_so_far = time.strftime("%M:%S", time.gmtime(time.time() - t0))
            t_left = time.strftime("%M:%S", time.gmtime((time.time() - t0) * (n_lines - i) / i))
            penalty = get_penalty(image, lightness_penalty, w, w_pos, w_neg)
            avg_penalty = f"{penalty / (wheel_pixel_size**2):.2f}"
            print(
                f"{i}/{n_lines}, average penalty = {avg_penalty}/{initial_avg_penalty}, \
time = {t_so_far}, time left = {t_left}    ",
                end="\r",
            )

        image, line = optimise_fitness(
            image_copy,
            through_pixels_dict,
            previous_edge,
            darkness,
            lightness_penalty,
            w,
            w_pos,
            w_neg,
            line_norm_mode,
            time_saver,
            flip_parity,
            shortest_line,
            n_hooks,
        )
        previous_edge = line[1]

        list_of_lines.append(line)

    penalty = get_penalty(image_copy, lightness_penalty, w, w_pos, w_neg)
    avg_penalty = f"{penalty / (wheel_pixel_size**2):.2f}"
    print(f"{len(list_of_lines)}/{n_lines}, average penalty = {avg_penalty}/{initial_avg_penalty}")
    print("time = " + time.strftime("%M:%S", time.gmtime(time.time() - t0)))

    return list_of_lines


def get_penalty(image, lightness_penalty, w, w_pos, w_neg):
    if isinstance(w, bool) and isinstance(w_pos, bool):
        return image.sum() - (1 + lightness_penalty) * image[image < 0].sum()
    elif isinstance(w_pos, bool):
        image_w = image * w
        return image_w.sum() - (1 + lightness_penalty) * image_w[image < 0].sum()
    elif isinstance(w, bool):
        image_wpos = image * w_pos
        image_wneg = image * w_neg
        return image_wpos[image > 0].sum() - lightness_penalty * image_wneg[image < 0].sum()


def prepare_image(file_name, wheel_pixel_size: int, colour=False, weighting=False):
    image = Image.open(file_name).resize((wheel_pixel_size, wheel_pixel_size))

    if colour:
        image = np.array(image.convert(mode="HSV").getdata()).reshape((wheel_pixel_size, wheel_pixel_size, 3))[:, :, 1]
    elif weighting:
        image = 1 - np.array(image.convert(mode="L").getdata()).reshape((wheel_pixel_size, wheel_pixel_size)) / 255
    else:
        image = 255 - np.array(image.convert(mode="L").getdata()).reshape((wheel_pixel_size, wheel_pixel_size))

    coords = np.array(list(product(range(wheel_pixel_size), range(wheel_pixel_size))))
    x_coords = coords.T[0]
    y_coords = coords.T[1]
    coords_distance_from_centre = np.sqrt(
        (x_coords - (wheel_pixel_size - 1) * 0.5) ** 2 + (y_coords - (wheel_pixel_size - 1) * 0.5) ** 2
    )
    mask = np.array(coords_distance_from_centre > wheel_pixel_size * 0.5)
    mask = np.reshape(mask, (-1, wheel_pixel_size))
    image[mask] = 0

    return image.T[:, ::-1]


def display_images(image_list):
    fig, axs = plt.subplots(1, len(image_list), figsize=(30, 30))
    for i, j in zip(range(len(image_list)), image_list):
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].imshow(j[:, ::-1].T, cmap=plt.get_cmap("Greys"))


def save_plot(list_coloured_lines, list_colours, file_name, size, n_hooks):
    new_hooks = generate_hooks(n_hooks, size)

    for i in range(len(new_hooks)):
        new_hooks[i] = [new_hooks[i][0], size - new_hooks[i][1]]

    # Create and save the JPEG image
    jpg_image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(jpg_image)
    svg_lines = []

    for lines, colour in zip(list_coloured_lines, list_colours):
        pixel_pairs = [(new_hooks[n[0]], new_hooks[n[1]]) for n in lines]
        for j in pixel_pairs:
            draw.line((tuple(j[0]), tuple(j[1])), fill=colour)
            # draw_svg.line((tuple(j[0]), tuple(j[1])), fill=colour)
            svg_lines.append(
                f'<line x1="{j[0][0]:.0f}" y1="{j[0][1]:.0f}" x2="{j[1][0]:.0f}" y2="{j[1][1]:.0f}" stroke="rgb({colour[0]}, {colour[1]}, {colour[2]})" stroke-width="1"/>'
            )

    jpg_image.save(file_name + ".jpg", format="JPEG")
    svg = (
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" style="background-color: transparent;">'
        + "".join(svg_lines)
        + "</svg>"
    )
    with open(file_name + ".svg", "w") as f:
        f.write(svg)


def save_plot_progress(list_coloured_lines, list_colours, file_name, size, n_hooks, proportion_list):
    for prop in proportion_list:
        file_name_temp = f"{file_name} {int(100 * prop)}%"
        lines_temp = list_coloured_lines[-1][: int(len(list_coloured_lines[-1]) * prop)]
        list_coloured_lines_temp = list_coloured_lines[:-1] + [lines_temp]
        save_plot(list_coloured_lines_temp, list_colours, file_name_temp, size=size, n_hooks=n_hooks)


def generate_gcode(line_list, n_hooks, x0, y0, x1, y1, speed=10_000, pen_height=450):
    # nodes = [line_list[0][0]] + [line_list[i][1] for i in range(1, len(line_list))]
    nodes = [line_list[i][1] for i in range(len(line_list))]

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = (x1 - x0) / 2

    def hook_pos(i):
        angle = 2 * math.pi * i / n_hooks
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    gcode = []
    gcode.append(f"M3S{pen_height} ; raise pen")
    gcode.append("G1 X0 Y0 F10000")

    x_start, y_start = hook_pos(nodes[0])
    gcode.append(f"G1 X{x_start:.3f} Y{y_start:.3f} F{speed}")
    gcode.append("M3S0 ; lower pen to start drawing")

    for idx in nodes[1:]:
        x, y = hook_pos(idx)
        gcode.append(f"G1 X{x:.3f} Y{y:.3f} F{speed}")

    gcode.append(f"M3S{pen_height} ; raise pen")
    gcode.append("G1 X0 Y0 F10000")

    return gcode
