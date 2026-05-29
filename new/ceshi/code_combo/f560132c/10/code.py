# Combo variant of f560132c (index 10)
# combo_spec: D5/1+D16/2
# Name: 嵌套框架图例+同心嵌套分割

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for this combo variant
# a) Concrete behavior:
#    The original task scattered four pieces that tiled one square, then used a 2x2 palette inside
#    one multicolor piece to recolor the assembled pieces by spatial quadrant. Here both ends of the
#    pipeline change together. The assembled target is no longer a cross-split square; it is four
#    concentric layers: outer square frame, second frame, third frame, and a solid center square.
#    The legend is no longer a 2x2 patch; it is a visible 7x7 nested-frame code whose outer ring,
#    middle rings, and center encode the colors for those four target layers from outside to inside.
#    Example: if the assembled layer sizes are 15/11/9/7, then after sorting pieces by bbox side
#    length we place the 15x15 frame on the canvas border, center the 11x11 frame inside it, center
#    the 9x9 frame inside that, and fill the 7x7 center block last. The legend's outermost code
#    color paints the 15x15 frame, the next code color paints the 11x11 frame, and so on inward.
#    Relative to the original, the changed step is not just "read colors differently"; the spatial
#    assembly itself is now nested rather than quadrant-based.
# b) Complexity calibration:
#    The generator always uses three ring pieces plus one solid block, so the visual difference from
#    the original is immediate. At the same time, the family stays small and learnable: there are
#    always exactly four layers and exactly four legend colors. To avoid overfitting to the D5-only
#    single variant, we reject samples where concentric depth order accidentally matches area order.
# c) Hidden assumptions in the original main():
#    `_extract_legend_map()` assumed a 2x2 spatial palette window; that is invalid and replaced by a
#    search for a 7x7 monochrome nested-frame code. `_fill_holes_in_component()` was previously safe
#    because holes were not semantically meaningful; here it would destroy the frame pieces, so the
#    solver never fills holes. `_assemble_exact_cover()` assumed arbitrary polyomino tiling plus
#    quadrant-based recoloring; the new solver instead assumes every piece has a square bbox and the
#    correct assembly is obtained by centering those square-bbox masks in descending size order.
# d) Placement legality:
#    The generator builds the output first, derives the four concentric masks exactly, crops each
#    piece to its bbox, then scatters those crops with `random_free_location`, so pieces do not
#    overlap and stay in-bounds. The legend is always the center block, guaranteeing enough room for
#    the nested code.
# e) Visual observability:
#    A human can see two repeated cues across examples: the legend shows nested color layers, and
#    the output shows nested geometric layers. Because the same outside-to-inside order appears in
#    both, the mapping is directly inferable from a few examples.
# f) Visual cleanliness:
#    Only one component is multicolor. The other three are plain square frames. No markers overlap
#    across components, and the output is a clean set of concentric layers rather than interleaved
#    decorations, so the task stays readable.
# g) Change inferability:
#    The color rule depends on visible layer order, not on numeric color ids or invisible metadata.
#    The generator keeps the functional color count at four output colors, matching the four target
#    layers, and ensures the same legend order is reused consistently within each sample.
# h) Variant verification metric:
#    Two metrics matter. First, all generated outputs should contain three frame pieces that span
#    multiple quadrant regions, proving the topology is concentric rather than the original cross
#    partition. Second, the output produced by concentric depth order must differ from the output
#    produced by the D5-only fallback of ranking pieces by area; otherwise D16/2 would be inert.
# i) New solver assumptions and generator guarantees:
#    The solver assumes there are exactly four components, exactly one legend component with the most
#    distinct colors, exactly one valid 7x7 nested-frame code inside that legend, and four unique
#    square bbox sizes whose centered placement yields the target. The generator enforces all of
#    these conditions by construction and rechecks them through `main(grid) == expected`.


NUM_PIECES = 4
LEGEND_IDX = 3
NESTED_SIZE = 7
NESTED_LAYERS = 4
SIDE_CHOICES = [13, 15, 17]
CORE_SIDE_CHOICES = [7, 9]
INPUT_SIZE_LO = 28
INPUT_SIZE_HI = 40
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400


def _ring_mask(size: int, depth: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    lo = int(depth)
    hi = int(size - depth - 1)
    mask[lo, lo : hi + 1] = True
    mask[hi, lo : hi + 1] = True
    mask[lo : hi + 1, lo] = True
    mask[lo : hi + 1, hi] = True
    return mask


def _nested_window_colors(window: np.ndarray) -> list[int] | None:
    if window.shape != (NESTED_SIZE, NESTED_SIZE):
        return None
    layers = []
    for depth in range(NESTED_LAYERS):
        cols = set(map(int, window[_ring_mask(NESTED_SIZE, depth)].ravel()))
        if len(cols) != 1:
            return None
        layers.append(int(next(iter(cols))))
    if len(set(layers)) != NESTED_LAYERS:
        return None
    return layers


def _extract_nested_legend_colors(g: np.ndarray, pts: np.ndarray) -> list[int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert hh >= NESTED_SIZE and ww >= NESTED_SIZE, "legend component must fit the nested-frame code"

    hits: list[tuple[int, int, list[int]]] = []
    for rr in range(hh - NESTED_SIZE + 1):
        for cc in range(ww - NESTED_SIZE + 1):
            colors = _nested_window_colors(sub[rr : rr + NESTED_SIZE, cc : cc + NESTED_SIZE])
            if colors is not None:
                hits.append((rr, cc, colors))
    assert len(hits) == 1, "legend component must contain exactly one valid nested-frame code"
    return list(hits[0][2])


def _component_info(mask: np.ndarray) -> dict:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = crop.shape
    assert h == w, "each piece bbox must be square in the concentric variant"
    coords = tuple(sorted(map(tuple, np.argwhere(crop))))
    return {"coords": coords, "side": int(h), "area": int(crop.sum())}


def _assemble_concentric(pieces: list[dict], legend_layers: list[int]) -> np.ndarray:
    assert len(pieces) == NUM_PIECES, "this combo expects exactly four pieces"
    assert len(legend_layers) == NUM_PIECES, "legend must expose four nested colors"

    ordered = sorted(enumerate(pieces), key=lambda x: -x[1]["side"])
    sides = [info["side"] for _, info in ordered]
    assert len(set(sides)) == NUM_PIECES, "piece bbox sizes must be distinct"

    side = int(sides[0])
    out = np.zeros((side, side), dtype=int)
    for depth, (_pi, info) in enumerate(ordered):
        offset = (side - info["side"]) // 2
        col = int(legend_layers[depth])
        for dr, dc in info["coords"]:
            out[offset + dr, offset + dc] = col
    return out


def _pts_to_mask(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered square-frame pieces into concentric layers, recoloring them by a legend's nested frames.

    Input structure:
    - bg = most common color.
    - Exactly four non-bg sprites are scattered across the grid.
    - Three sprites are hollow square frames and one sprite is a solid square block.
    - The solid block is the legend: inside it sits one 7x7 nested-frame code with four visible
      layers from outer ring to center.

    Rule:
    - Identify the legend by its multiple colors and read its four nested-frame colors from outside
      to inside.
    - Crop every sprite to its own square bounding box, sort those boxes from largest to smallest,
      and center them concentrically to reconstruct the target square.
    - Recolor the outermost assembled layer with the legend's outer color, the next layer with the
      next legend color, and continue inward until the center block.

    Key insight: Both the legend and the output use the same outside-to-inside hierarchy. The task
    is no longer about quadrant position or area ranking; it is about matching nested color layers
    to nested geometric layers.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, "grid must contain exactly four non-background components"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1 and max_distinct > 1, "legend component must be uniquely multicolor"

    legend_layers = _extract_nested_legend_colors(g, comps[legend_candidates[0]])
    pieces = [_component_info(_pts_to_mask(g.shape, pts)) for pts in comps]
    return _assemble_concentric(pieces, legend_layers)


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(int(total),)]
    out = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            out.append((int(first),) + tuple(rest))
    return out


def _build_masks(side: int, core_side: int, thicknesses: tuple[int, int, int]) -> list[np.ndarray]:
    t0, t1, t2 = map(int, thicknesses)
    sizes = [int(side), int(side - 2 * t0), int(side - 2 * (t0 + t1)), int(core_side)]
    assert sizes[2] == core_side + 2 * t2
    masks: list[np.ndarray] = []
    for i, size in enumerate(sizes):
        mask = np.zeros((side, side), dtype=bool)
        off = (side - size) // 2
        mask[off : off + size, off : off + size] = True
        if i < NUM_PIECES - 1:
            inner = sizes[i + 1]
            off2 = (side - inner) // 2
            mask[off2 : off2 + inner, off2 : off2 + inner] = False
        masks.append(mask)
    return masks


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _paint_nested_frames(patch: np.ndarray, r0: int, c0: int, colors: list[int]) -> None:
    for depth, col in enumerate(colors):
        lo = int(depth)
        hi = int(NESTED_SIZE - depth)
        patch[r0 + lo : r0 + hi, c0 + lo : c0 + hi] = int(col)


def _nested_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(0, h - NESTED_SIZE + 1):
        for c in range(0, w - NESTED_SIZE + 1):
            if mask[r : r + NESTED_SIZE, c : c + NESTED_SIZE].all():
                spots.append((r, c))
    return spots


def _scatter_pieces(
    masks: list[np.ndarray], body_colors: list[int], legend_layers: list[int], side: int
) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi])
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(body_colors[pi])
        if pi == LEGEND_IDX:
            spots = _nested_spots(oriented)
            if not spots:
                return None
            rr, cc = random.choice(spots)
            _paint_nested_frames(patch, rr, cc, legend_layers)
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _expected_from_depth(masks: list[np.ndarray], legend_layers: list[int]) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    for depth, mask in enumerate(masks):
        out[mask] = int(legend_layers[depth])
    return out


def _expected_from_area(masks: list[np.ndarray], legend_layers: list[int]) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    ranked = sorted(range(NUM_PIECES), key=lambda pi: (-int(masks[pi].sum()), pi))
    for rank, pi in enumerate(ranked):
        out[masks[pi]] = int(legend_layers[rank])
    return out


def _choose_layout() -> tuple[int, int, tuple[int, int, int]] | None:
    side = random.choice(SIDE_CHOICES)
    core_opts = [c for c in CORE_SIDE_CHOICES if c <= side - 6 and (side - c) % 2 == 0]
    if not core_opts:
        return None
    core_side = random.choice(core_opts)
    total_thickness = (side - core_side) // 2
    opts = _compositions(total_thickness, 3)
    random.shuffle(opts)
    for thicks in opts:
        masks = _build_masks(side, core_side, thicks)
        if not np.array_equal(_expected_from_depth(masks, [1, 2, 3, 4]), _expected_from_area(masks, [1, 2, 3, 4])):
            return side, core_side, tuple(map(int, thicks))
    return None


def _frame_quadrant_span(mask: np.ndarray) -> int:
    side = mask.shape[0]
    span = set()
    for r, c in np.argwhere(mask):
        rr = min(1, int(2 * r / side))
        cc = min(1, int(2 * c / side))
        span.add((rr, cc))
    return len(span)


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    side, core_side, thicknesses = layout
    masks = _build_masks(side, core_side, thicknesses)
    infos = [_component_info(mask) for mask in masks]
    if [info["side"] for info in infos] != sorted([info["side"] for info in infos], reverse=True):
        return None
    if any(_frame_quadrant_span(masks[i]) < 4 for i in range(NUM_PIECES - 1)):
        return None

    colors = random.sample(range(1, 10), 8)
    body_colors = colors[:NUM_PIECES]
    legend_layers = colors[NUM_PIECES:]
    random.shuffle(legend_layers)

    expected = _expected_from_depth(masks, legend_layers)
    if np.array_equal(expected, _expected_from_area(masks, legend_layers)):
        return None

    grid = _scatter_pieces(masks, body_colors, legend_layers, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
