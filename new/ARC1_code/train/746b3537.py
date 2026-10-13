import numpy as np
import random
from arc_utils import Color, colors_present, draw_line, find_components


# concepts:
# line detection, color extraction

# description:
# In the input you will see a grid consisting of stripes that are either horizontal or vertical.
# To make the output, make a grid with one pixel for each stripe whose color is the same color as that stripe.
# If the stripes are vertical, the output should be vertical, and if the stripes are horizontal, the output should be horizontal. The colors should be in the order they appear in the input.

def main(input_grid):
    """
    Compress uniform horizontal or vertical color stripes into a one-pixel-wide color sequence.

    Input structure:
    - The input is made of full-width horizontal stripes or full-height vertical stripes.
    - Stripes may have thickness greater than one cell.
    - Stripe colors appear in order across rows or columns.

    Rule:
    - Detect whether rows or columns are constant.
    - Collapse consecutive duplicate row or column colors into one entry per stripe.
    - Return a column for horizontal stripes or a row for vertical stripes.

    Key insight:
    - Stripe thickness is redundant; only orientation and ordered colors remain.
    """
    H, W = input_grid.shape

    rows_constant = all(np.all(input_grid[r, :] == input_grid[r, 0]) for r in range(H))
    cols_constant = all(np.all(input_grid[:, c] == input_grid[0, c]) for c in range(W))
    if rows_constant and not cols_constant:
        # Horizontal stripes -> output is a vertical list of row colors (top->bottom).
        raw = [int(input_grid[r, 0]) for r in range(H)]
        colors: list[int] = []
        for c in raw:
            if not colors or c != colors[-1]:
                colors.append(c)
        out = np.full((len(colors), 1), Color.BLACK, dtype=int)
        out[:, 0] = colors
        return out
    if cols_constant and not rows_constant:
        # Vertical stripes -> output is a horizontal list of column colors (left->right).
        raw = [int(input_grid[0, c]) for c in range(W)]
        colors: list[int] = []
        for c in raw:
            if not colors or c != colors[-1]:
                colors.append(c)
        out = np.full((1, len(colors)), Color.BLACK, dtype=int)
        out[0, :] = colors
        return out
    # Fallback: pick whichever holds.
    if rows_constant:
        raw = [int(input_grid[r, 0]) for r in range(H)]
        colors: list[int] = []
        for c in raw:
            if not colors or c != colors[-1]:
                colors.append(c)
        out = np.full((len(colors), 1), Color.BLACK, dtype=int)
        out[:, 0] = colors
        return out
    if cols_constant:
        raw = [int(input_grid[0, c]) for c in range(W)]
        colors: list[int] = []
        for c in raw:
            if not colors or c != colors[-1]:
                colors.append(c)
        out = np.full((1, len(colors)), Color.BLACK, dtype=int)
        out[0, :] = colors
        return out
    raise ValueError("The stripes are not aligned in a single axis")
    


def generate_input():
    orientation = random.choice(["horizontal", "vertical"])
    n_stripes = np.random.randint(2, 8)
    stripe_thicknesses = [int(np.random.randint(1, 3)) for _ in range(n_stripes)]
    stripe_colors = [int(c) for c in np.random.choice(list(Color.NOT_BLACK), n_stripes, replace=False)]

    if orientation == "horizontal":
        H = sum(stripe_thicknesses)
        W = int(np.random.randint(3, 12))
        grid = np.full((H, W), Color.BLACK, dtype=int)
        r0 = 0
        for t, col in zip(stripe_thicknesses, stripe_colors):
            grid[r0 : r0 + t, :] = col
            r0 += t
        return grid

    # vertical
    H = int(np.random.randint(3, 12))
    W = sum(stripe_thicknesses)
    grid = np.full((H, W), Color.BLACK, dtype=int)
    c0 = 0
    for t, col in zip(stripe_thicknesses, stripe_colors):
        grid[:, c0 : c0 + t] = col
        c0 += t
    return grid
# ============= remove below this point for prompting =============
