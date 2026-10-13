import numpy as np
import random

from arc_utils import (
    Color,
    bbox,
    bbox_anchor,
    bbox_h,
    bbox_w,
    collision,
    colors_present,
    contact,
    crop_bbox,
    draw_line,
    find_components,
    get_bg,
    paste,
    random_free_location,
    random_sprite,
)

# concepts:
# repeated translation, indicator pixels, non-black background

# description:
# In the input you will see a non-black background with a black object and an indicator pixel touching it of a different color.
# To make the output, repeatedly translate the black object in the direction of the indicator pixel and make the color of these repeated translations match the indicator.

def main(input_grid):
    """
    Repeat a black template object outward in the color and direction of its touching indicator.

    Input structure:
    - The grid has a non-black background.
    - A black template object touches one or more single-pixel indicator objects.
    - Each indicator color is different from the background.

    Rule:
    - Detect the background, template objects, and one-pixel indicators.
    - For each touching template-indicator pair, infer the direction from template center to indicator.
    - Recolor the template to the indicator color and paste repeated translated copies outward.

    Key insight:
    - The indicator acts as both a color label and a direction handle for repeated
      translation of the same template shape.
    """
    # Plan:
    # 1. Parse the input into the black object and the indicator pixel(s)
    # 2. Determine the direction of translation
    # 3. Put down recolored versions of the black object

    # 1. Parse the input
    # Background is the most common color.
    bg = get_bg(input_grid)

    # Extract connected components (as full-grid objects).
    objects = find_components(input_grid, bg=bg, connectivity=8, monochromatic=True, return_format="objects")

    # Indicators are single-pixel objects.
    indicator_objects = [obj for obj in objects if int((obj != bg).sum()) == 1]

    # Templates are larger objects.
    template_objects = [obj for obj in objects if int((obj != bg).sum()) != 1]

    # We draw on top of the output, so copy it
    output_grid = input_grid.copy()

    # Iterate over every template and indicator which are in contact
    for template_obj in template_objects:
        b_template = bbox(template_obj != bg)
        assert b_template is not None
        template_sprite = crop_bbox(template_obj, b_template)
        for indicator_obj in indicator_objects:
            if not contact(object1=template_obj, object2=indicator_obj, background=bg, connectivity=8):
                continue

            # 2. Determine the direction of translation
            b_ind = bbox(indicator_obj != bg)
            assert b_ind is not None
            ind_r, ind_c = bbox_anchor(b_ind, anchor="center")
            tpl_r, tpl_c = bbox_anchor(b_template, anchor="center")

            dr = int(np.sign(ind_r - tpl_r))
            dc = int(np.sign(ind_c - tpl_c))
            # Figure out the stride the translation, which is as far as we can go while still covering the indicator pixel
            possible_strides = [
                stride
                for stride in range(1, max(output_grid.shape))
                if collision(
                    object1=indicator_obj,
                    object2=template_obj,
                    r2=stride * dr,
                    c2=stride * dc,
                    background=bg,
                )
            ]
            stride = max(possible_strides)

            # 3. Put down recolored versions of the black object as much as we can until we fall out of the canvas
            # Prepare a new version of the sprite
            new_color = colors_present(indicator_obj, exclude={bg})[0]
            recolored_template_sprite = template_sprite.copy()
            recolored_template_sprite[recolored_template_sprite != bg] = new_color

            # Put down the recolored sprite at every stride
            for i in range(1, 10):
                old_r, old_c = bbox_anchor(b_template, anchor="upper left")
                new_r = old_r + i * dr * stride
                new_c = old_c + i * dc * stride
                output_grid = paste(
                    output_grid,
                    recolored_template_sprite,
                    new_r,
                    new_c,
                    transparent=bg,
                    clip=True,
                )
    
    return output_grid

def generate_input():
    background_color = int(np.random.choice(Color.NOT_BLACK))
    indicator_color = int(np.random.choice([c for c in Color.NOT_BLACK if c != background_color]))
    template_color = Color.BLACK

    H, W = np.random.randint(10, 25), np.random.randint(10, 25)
    grid = np.full((H, W), background_color)

    template_sprite = random_sprite(range(3, 6), range(3, 6), color_palette=[template_color], background=background_color)
    r0, c0 = random_free_location(grid, template_sprite, transparent=background_color, border=0, padding=0, padding_connectivity=8)
    paste(grid, template_sprite, r0, c0, transparent=background_color, clip=True)

    template_mask = (grid == template_color)
    b_template = bbox(template_mask)
    assert b_template is not None
    center_r, center_c = bbox_anchor(b_template, anchor="center")

    # Choose an indicator position adjacent to the template such that `main`'s inferred (dr,dc)
    # also corresponds to an actual touching direction (so collision(..., stride=1) succeeds).
    template_pixels = np.argwhere(template_mask)
    candidates: list[tuple[int, int]] = []
    for pr, pc in template_pixels:
        pr = int(pr)
        pc = int(pc)
        for dr0 in (-1, 0, 1):
            for dc0 in (-1, 0, 1):
                if dr0 == 0 and dc0 == 0:
                    continue
                ir = pr + dr0
                ic = pc + dc0
                if 0 <= ir < H and 0 <= ic < W and int(grid[ir, ic]) == int(background_color):
                    candidates.append((ir, ic))
    random.shuffle(candidates)

    for ir, ic in candidates:
        dr_main = int(np.sign(ir - center_r))
        dc_main = int(np.sign(ic - center_c))
        if dr_main == 0 and dc_main == 0:
            continue
        pr = ir - dr_main
        pc = ic - dc_main
        if 0 <= pr < H and 0 <= pc < W and bool(template_mask[pr, pc]):
            grid[ir, ic] = indicator_color
            break
    else:
        raise ValueError("generate_input: could not place indicator")

    # random rotation to get a variety of orientations
    grid = np.rot90(grid, -(np.random.randint(0, 4)))
    return grid
# ============= remove below this point for prompting =============
