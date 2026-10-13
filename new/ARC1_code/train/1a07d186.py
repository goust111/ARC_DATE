
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_h, bbox_w, contact, draw_line, find_components, paste, translate

# concepts:
# collision detection, sliding objects, horizontal/vertical bars

# description:
# In the input you will see horizontal/vertical bars and individual coloured pixels sprinkled on a black background
# Move each colored pixel to the bar that has the same colour until the pixel touches the bar.
# If a colored pixel doesn't have a corresponding bar, it should be deleted.

def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide each loose pixel until it touches the full bar with the same color, deleting unmatched pixels.

    Input structure:
    - The input contains full horizontal or vertical bars and single colored pixels on black.
    - Bars are monochromatic and span the full height or width.
    - Some loose pixels share colors with bars and some may not.

    Rule:
    - Copy all bars unchanged to the output.
    - For each loose pixel with a matching-color bar, slide it in a cardinal direction until it contacts that bar.
    - Drop any loose pixel whose color has no corresponding bar.

    Key insight:
    - Bar color is the destination label: matching pixels move until first contact with their same-color
      full-line anchor.
    """
    output_grid = np.zeros_like(input_grid)

    # each object is either a bar or pixel, all uniform color
    objects = find_components(input_grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="objects")

    # separate the bars from the pixels
    bars, pixels = [], []
    for obj in objects:
        b = bbox(obj != Color.BLACK)
        assert b is not None
        h = bbox_h(b)
        w = bbox_w(b)
        if h == input_grid.shape[0] or w == input_grid.shape[1]:
            bars.append(obj)
        else:
            pixels.append(obj)
    
    # copy the bars to the output grid
    for bar in bars:
        paste(output_grid, bar, 0, 0, transparent=Color.BLACK, clip=True)
    
    # slide each pixel until it just barely touches the bar with the matching color
    for pixel in pixels:
        color = np.unique(pixel)[1]
        matching_bars = [bar for bar in bars if np.unique(bar)[1] == color]

        # if there is no matching bar, delete the pixel
        if len(matching_bars) == 0:
            continue

        # consider sliding in the 4 cardinal directions, and consider sliding as far as possible
        possible_displacements = [ (slide_distance*dr, slide_distance*dc)
                                   for slide_distance in range(max(input_grid.shape))
                                   for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)] ]
        for dr, dc in possible_displacements:
            new_pixel = translate(pixel, dr, dc, background=Color.BLACK)
            if contact(object1=matching_bars[0], object2=new_pixel):
                paste(output_grid, new_pixel, 0, 0, transparent=Color.BLACK, clip=True)
                break
    
    return output_grid



def generate_input() -> np.ndarray:

    n, m = np.random.randint(10, 20), np.random.randint(10, 20)
    
    # separately generate canvas for bars and pixels, so that we can put the bars on top of the pixels
    just_the_bars = np.zeros((m, n), dtype=int)
    just_the_pixels = np.zeros((m, n), dtype=int)

    n_bars = np.random.randint(1, 5)

    # make sure that every bar has a different color
    bar_colors = random.sample(Color.NOT_BLACK, n_bars)

    horizontal_or_vertical = np.random.choice(["horizontal", "vertical"])
    for bar in range(n_bars):
        color = bar_colors[bar]

        if horizontal_or_vertical == "horizontal":
            r = np.random.randint(0, m)
            just_the_bars[r, :] = color
        else:
            c = np.random.randint(0, n)
            just_the_bars[:, c] = color

        n_pixels = np.random.randint(1, 5)
        for _ in range(n_pixels):
            r, c = np.random.randint(0, m), np.random.randint(0, n)
            just_the_pixels[r, c] = color

    # sample a few pixels that aren't the same color as any of the bars
    not_a_bar_color = set(Color.NOT_BLACK) - set(bar_colors)
    n_distracter_pixels = np.random.randint(1, 5)
    for _ in range(n_distracter_pixels):
        r, c = np.random.randint(0, m), np.random.randint(0, n)
        just_the_pixels[r, c] = np.random.choice(list(not_a_bar_color))

    # put the bars on top of the pixels by copying them on top
    grid = just_the_pixels.copy()
    paste(grid, just_the_bars, 0, 0, transparent=Color.BLACK, clip=True)
    # Equivalent to:
    # grid[just_the_bars != Color.BLACK] = just_the_bars[just_the_bars != Color.BLACK]
    

    return grid


        






    return grid
