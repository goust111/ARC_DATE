
import numpy as np
from typing import *
import random
from arc_utils import Color, bbox, bbox_h, colors_present, crop_bbox, draw_line, find_components, object_neighbors, paste, random_free_location, random_sprite, scale_sprite

# concepts:
# scaling, puzzle pieces, indicator pixels

# description:
# In the input you will see objects with exactly 2 colors, each with one pixel/rectangle of a different color as an indicator. The indicator color is the same across objects.
# To make the output, one of those objects is a template shape that you are going to translate/recolor/rescale to match indicators with each other object.
# Place the rescaled template on top of the other shape so that the indicators are at the same position, and change color to match what you are placing on top of.

def main(input_grid):
    """
    Complete each partial object by scaling and recoloring the largest template so its indicator aligns with the partial indicator.

    Input structure:
    - The grid contains several two-color objects on a black background.
    - One indicator color appears in every object.
    - One object is the full template; the others are partial scaled objects with their own colors.

    Rule:
    - Find the common indicator color across all objects.
    - Choose the template by largest object size normalized by indicator size.
    - Scale and recolor the template for each other object, then paste it with indicators aligned.

    Key insight:
    - The indicator region supplies both correspondence and scale, while the template supplies
      the missing shape around each partial object.
    """
    # Plan:
    # 1. Parse the input into template object and other objects, and determine the indicator color
    # 2. For each other object, rescale+recolor the template to match indicators

    # 1. Parse the input

    # Extract all the objects from the input grid
    background = Color.BLACK
    objects = find_components(input_grid, bg=background, connectivity=8, monochromatic=False, return_format="objects")

    # The indicator pixel's color appears in all the objects
    possible_indicator_colors = [ color for color in Color.ALL_COLORS
                                 if all( color in colors_present(obj, exclude={ background }) for obj in objects)]
    assert len(possible_indicator_colors) == 1, "There should be exactly one indicator color"
    indicator_color = possible_indicator_colors[0]

    # Find the template object, which is the biggest object after you scale the indicator down to have size 1x1
    object_sizes = [ np.sum(obj != background) for obj in objects]
    indicator_sizes = [ np.sum(obj == indicator_color) for obj in objects]
    rescaled_sizes = [size // indicator_size for size, indicator_size in zip(object_sizes, indicator_sizes)]
    template_index = np.argmax(rescaled_sizes)
    template_object = objects[template_index]
    other_objects = [obj for i, obj in enumerate(objects) if i != template_index]

    _b = bbox(template_object != background)
    assert _b is not None
    template_sprite = crop_bbox(template_object, _b)

    # 2. For each other object, rescale+recolor the template to match indicators
    # Determine the scaling factor by the ratio of the size of the indicator pixel region
    # Determine the color according to the non-indicator color of the object
    # Determine the position so that indicator pixels are overlaid

    # To produce the output we draw on top of the input
    output_grid = input_grid.copy()

    for other_object in other_objects:

        # Find the new shape's color
        new_color = [ color for color in colors_present(other_object, exclude={ background }) if color != indicator_color][0]

        # find the new scale, which is the ratio of the size of the indicator pixel in the original shape to the size of the indicator pixel in the new shape
        b_other = bbox(other_object == indicator_color)
        b_template = bbox(template_object == indicator_color)
        assert b_other is not None and b_template is not None
        new_scale = bbox_h(b_other) // bbox_h(b_template)

        # Scale the original template to the same scale...
        template_sprite_scaled = scale_sprite(template_sprite, new_scale)
        # ...and change its color to the new shape's color
        template_sprite_scaled[(template_sprite_scaled != background) & (template_sprite_scaled != indicator_color)] = new_color

        # Overlay the indicator pixels from the scaled/recolored template sprite with the indicator pixels from the other object
        r = np.min(np.argwhere(other_object == indicator_color)[:,0]) - np.min(np.argwhere(template_sprite_scaled == indicator_color)[:,0])
        c = np.min(np.argwhere(other_object == indicator_color)[:,1]) - np.min(np.argwhere(template_sprite_scaled == indicator_color)[:,1])
        paste(output_grid, template_sprite_scaled, r, c, transparent=Color.BLACK, clip=True)
        
    return output_grid

def generate_input():
    # Generate the background grid
    background = Color.BLACK
    width, height = np.random.randint(15, 30), np.random.randint(15, 30)
    grid = np.full((width, height), background)

    # Randomly select the color of objects and indicator, which should all be distinct
    n_objects = 2
    colors = np.random.choice(Color.NOT_BLACK, n_objects + 1, replace=False)
    indicator_color, template_color, other_colors = colors[0], colors[1], colors[2:]

    # Ensure the shapes after completion do not overlap by having a canvas that shows what things will look like after producing the output
    output_grid = grid.copy()    

    # Generate the original shape
    w, h = np.random.randint(3, 5), np.random.randint(3, 5)
    template_sprite = random_sprite(w, h, color_palette=[template_color], connectivity=4)

    # Randomly turn one pixel into the indicator pixel
    indicator_pixel = random.choice(np.argwhere(template_sprite == template_color))
    template_sprite[indicator_pixel[0], indicator_pixel[1]] = indicator_color

    # Place the template on the grid, and on the predicted output
    r, c = random_free_location(output_grid, template_sprite, transparent=Color.BLACK, border=0, padding=0, padding_connectivity=8)
    paste(grid, template_sprite, r, c, transparent=Color.BLACK, clip=True)
    paste(output_grid, template_sprite, r, c, transparent=Color.BLACK, clip=True)

    # Check which pixels are neighbors of the indicator pixel, which will be included in the other objects
    neighbor_mask = (template_sprite == template_color) & object_neighbors(template_sprite == indicator_color, connectivity=8)
    
    # Place the other objects
    for other_color in other_colors:
        # there is the completed other object which will be in the output, and the uncompleted other object which will be in the input
        complete_other_object = template_sprite.copy()
        complete_other_object[complete_other_object == template_color] = other_color

        # Make the incomplete object, which just has the indicator pixel plus one of its neighbors
        incomplete_other_object = template_sprite.copy()
        incomplete_other_object[incomplete_other_object == template_color] = background        
        neighbor_r, neighbor_c = random.choice(np.argwhere(neighbor_mask))
        incomplete_other_object[neighbor_r, neighbor_c] = other_color

        # Rescale both
        scale = np.random.randint(1, 3)
        complete_other_object = scale_sprite(complete_other_object, scale)
        incomplete_other_object = scale_sprite(incomplete_other_object, scale)

        # Place the complete object on the predicted output
        try:
            r, c = random_free_location(output_grid, complete_other_object, transparent=Color.BLACK, border=0, padding=1, padding_connectivity=8)
        except ValueError:
            break
        paste(output_grid, complete_other_object, r, c, transparent=Color.BLACK, clip=True)

        # place the incomplete object such that its indicators overlap at the same location
        r -= np.min(np.argwhere(complete_other_object == indicator_color)[:,0]) - np.min(np.argwhere(incomplete_other_object == indicator_color)[:,0])
        c -= np.min(np.argwhere(complete_other_object == indicator_color)[:,1]) - np.min(np.argwhere(incomplete_other_object == indicator_color)[:,1])
        paste(grid, incomplete_other_object, r, c, transparent=Color.BLACK, clip=True)

    return grid
