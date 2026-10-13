from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll concentric colored rectangles into centered horizontal bands whose indents accumulate by ring thickness.

    Input structure:
    - Black background with a centered nested set of colored rectangular rings and a solid core.
    - Each layer uses a distinct color and may have its own uniform thickness.

    Rule:
    - Order the layers from outermost to innermost by bbox size.
    - Output one row per layer using the outermost bbox width.
    - The first row is fully filled with the outermost color; each next row is indented on both sides by the sum of previous layer thicknesses.
    - Fill only the centered band span for each row.

    Key insight:
    - Horizontal band width is determined by cumulative ring thickness peeled from the outside inward.
    """
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    layers = []
    for comp in comps:
        r0,c0,r1,c1 = map(int, comp['bbox'])
        color = int(comp['color'])
        thick = min(np.sum(g[r0:r1+1,c0] == color), np.sum(g[r0,c0:c1+1] == color))
        layers.append((r1-r0+1, c1-c0+1, thick, color, (r0,c0,r1,c1)))
    layers.sort(reverse=True)
    width = layers[0][1]
    out = np.zeros((len(layers), width), dtype=int)
    indent = 0
    for i,(_,_,thick,color,_) in enumerate(layers):
        out[i, indent:width-indent] = color
        indent += thick
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((21,21), dtype=int)
    r0,c0,r1,c1 = 2,2,18,18
    for color,thick in zip(random.sample([1,2,3,4,6,7,8,9], random.randint(3,5)), [1,1,2,1,1]):
        if r1 <= r0 or c1 <= c0: break
        g[r0:r0+thick, c0:c1+1] = color; g[r1-thick+1:r1+1, c0:c1+1] = color
        g[r0:r1+1, c0:c0+thick] = color; g[r0:r1+1, c1-thick+1:c1+1] = color
        r0 += thick+1; c0 += thick+1; r1 -= thick+1; c1 -= thick+1
    g[r0:r1+1, c0:c1+1] = color
    return g
