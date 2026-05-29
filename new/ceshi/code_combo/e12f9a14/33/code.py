# Combo variant of e12f9a14 (index 33)
# combo_spec: D1/0+D4/3
# Name: 边缘像素源+正交射线

import itertools

import numpy as np

from arc_utils import get_bg


# Design analysis for this combo variant
# --------------------------------------
# concretization:
#   The framed 2x2 blocks disappear entirely. The input is now an otherwise empty square
#   field whose only non-background objects are isolated single pixels placed on the outer
#   border. Each border pixel is both a visible source marker and the color of the ray it
#   emits. The ray direction is no longer diagonal: top markers fire straight downward,
#   bottom markers upward, left markers rightward, and right markers leftward.
# behavior trace:
#   Original task:
#   - find a 2x2 colored block inside a frame;
#   - infer missing frame cells as exits;
#   - launch diagonal rays whose later collisions can bend into new trajectories.
#   Combo task:
#   - find every non-background border pixel directly;
#   - infer its direction only from the border side it touches;
#   - extend a straight row/column ray of the same color through background cells.
#   Example local change: an original top-left frame gap could create a NW diagonal. In the
#   combo task, a top-edge marker at (0, 7) instead paints the whole column downward. If a
#   left-edge marker on row 10 also exists, the meeting point (10, 7) is cleared back to
#   background and the two rays continue straight, producing a blank-centered plus shape
#   rather than a bend or zigzag.
# complexity calibration:
#   The source vocabulary is drastically simplified, so the generator compensates by
#   requiring repeated instances of each directional family. Every sample has at least two
#   vertical rays and two horizontal rays, yielding multiple orthogonal crossings. That
#   keeps the variant visibly distinct from both the framed original and trivial one-line
#   scenes, while avoiding too many source types or case splits.
# implicit assumption audit:
#   Original assumptions that are now invalid:
#   - searching for 2x2 monochrome blocks;
#   - inferring a shared frame color around each block;
#   - interpreting missing frame-border pixels as exits;
#   - supporting diagonal/zigzag beam state transitions after collisions.
#   New solver assumptions:
#   - every non-background input pixel is a border source;
#   - sources are not corners, because a corner would have two inward orthogonal choices;
#   - all vertical sources use distinct columns, and all horizontal sources use distinct
#     rows, so same-axis overlaps and head-on cases never need to be interpreted;
#   - crossing cells are ordinary background cells, never source cells.
#   The generator explicitly guarantees each of these conditions.
# placement legality:
#   Source rows/columns are sampled away from corners, with unique columns for top/bottom
#   sources and unique rows for left/right sources. This guarantees non-overlap, keeps all
#   crossings inside the canvas interior, and makes each ray run unobstructed until the far
#   border.
# visual observability:
#   Across a few examples, a human sees only border dots in the input and long inward
#   orthogonal beams in the output. Because top/bottom sources always create vertical lines
#   and left/right sources always create horizontal lines, the source-direction rule is
#   repeated several times per puzzle. The blank crossing cells make the collision rule
#   directly visible without relying on hidden timing or color ordering.
# visual cleanliness:
#   The canvas contains only 4-6 sparse border markers and otherwise plain background. No
#   frames, nested objects, or multi-stage redirections are present. Unique rows/columns
#   prevent same-axis overpainting, so the output remains a readable weave of straight
#   colored rows and columns with isolated holes at crossings.
# color inference:
#   The mapping is identity: the source marker color is the ray color. No palette ranking,
#   counting, or numeric color semantics are used.
# new_main_assumptions_and_generator_support:
#   main() assumes:
#   - border-only non-background input;
#   - non-corner sources;
#   - unique horizontal rows and unique vertical columns among sources.
#   generate_input() enforces:
#   - 1-2 sources on each side family, total 4-6;
#   - source positions at least two cells away from corners;
#   - at least 2 vertical and 2 horizontal rays;
#   - solver validation plus an exact crossing-count check on every generated sample.


GRID_SIZE_RANGE = (13, 31)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.3, 0.7]
PALETTE_FOR_BG = {
    8: [1, 2, 3, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 8, 9],
}
SOURCE_MARGIN = 2
MIN_REQUIRED_CROSSINGS = 4
MIN_BEAM_PIXELS_PER_SOURCE = 5
GEN_RETRIES = 300


_SIDE_COUNT_CHOICES = [
    counts
    for total in range(4, 7)
    for counts in itertools.product((1, 2), repeat=4)
    if sum(counts) == total
]


def _axis(direction: tuple[int, int]) -> str:
    return "h" if direction[0] == 0 else "v"


def _source_dir(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    on_tb = r in (0, h - 1)
    on_lr = c in (0, w - 1)
    assert on_tb or on_lr, "source must lie on the border"
    assert not (on_tb and on_lr), "corner source would make inward orthogonal direction ambiguous"
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    return (0, -1)


def _find_edge_sources(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    h, w = g.shape
    sources: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == bg:
                continue
            sources.append((v, (r, c), _source_dir(h, w, r, c)))
    sources.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    return sources


def _ray_cells(
    g: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = g.shape
    cells = [seed]
    r, c = seed
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(g[r, c]) != bg:
            break
        cells.append((r, c))
    return cells


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    sources = _find_edge_sources(g, bg)
    assert sources, "expected at least one border source"

    out = g.copy()
    rays: list[dict[str, object]] = []
    source_cells: set[tuple[int, int]] = set()
    used_rows_h: set[int] = set()
    used_cols_v: set[int] = set()

    for color, seed, direction in sources:
        assert seed not in source_cells, "expected unique source positions"
        source_cells.add(seed)
        axis = _axis(direction)
        if axis == "h":
            assert seed[0] not in used_rows_h, "horizontal sources must occupy distinct rows"
            used_rows_h.add(seed[0])
        else:
            assert seed[1] not in used_cols_v, "vertical sources must occupy distinct columns"
            used_cols_v.add(seed[1])
        cells = _ray_cells(g, bg, seed, direction)
        ray = {
            "color": int(color),
            "seed": seed,
            "axis": axis,
            "cells": cells,
            "set": set(cells),
        }
        rays.append(ray)
        for r, c in cells:
            out[r, c] = int(color)

    crossings: set[tuple[int, int]] = set()
    horizontal = [ray for ray in rays if ray["axis"] == "h"]
    vertical = [ray for ray in rays if ray["axis"] == "v"]
    for hray in horizontal:
        row = hray["seed"][0]
        for vray in vertical:
            cell = (row, vray["seed"][1])
            if cell in source_cells:
                continue
            if int(g[cell[0], cell[1]]) != bg:
                continue
            if cell not in hray["set"] or cell not in vray["set"]:
                continue
            crossings.add(cell)

    for r, c in crossings:
        out[r, c] = bg

    stats = {
        "source_count": len(sources),
        "horizontal_count": len(horizontal),
        "vertical_count": len(vertical),
        "orth_crossings": len(crossings),
        "changed_pixels": int(np.count_nonzero(out != g)),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit inward orthogonal rays from border markers and punch out blank cells where rows and columns cross.

    Input structure:
    - The background is the most common color, and every non-background pixel is a single-cell source located on the outer border.
    - There are no frames and no interior 2x2 blocks; the only information needed to read a source is its border side and its own color.

    Rule:
    - A source on the top edge fires straight downward, on the bottom edge upward, on the left edge rightward, and on the right edge leftward.
    - Each ray keeps the source color and paints consecutive background cells until it would leave the grid or hit another non-background input pixel.
    - When a horizontal ray and a vertical ray would occupy the same cell, that shared cell is cleared back to background while both straight rays remain visible on the cells before and after it.

    Key insight:
    - The fused change is not just "different sources" plus "different paths" independently: border markers directly encode inward row/column motion, and the resulting outputs are read as sparse edge-controlled crosshatch patterns with blank-centered intersections instead of frame-driven diagonal beam bends.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


def _pick_bg_and_colors(rng: np.random.Generator, count: int) -> tuple[int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    palette = PALETTE_FOR_BG[bg]
    colors = list(map(int, rng.choice(palette, size=count, replace=False)))
    return bg, colors


def _render_sources(n: int, bg: int, placements: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for color, r, c in placements:
        grid[r, c] = int(color)
    return grid


def _input_is_border_only(grid: np.ndarray, bg: int) -> bool:
    n, m = grid.shape
    cells = np.argwhere(grid != bg)
    for r, c in cells:
        if not (r in (0, n - 1) or c in (0, m - 1)):
            return False
        if r in (0, n - 1) and c in (0, m - 1):
            return False
    return True


def _sample_placements(
    n: int,
    counts: tuple[int, int, int, int],
    colors: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int, int]] | None:
    top_count, bottom_count, left_count, right_count = counts
    vertical_count = top_count + bottom_count
    horizontal_count = left_count + right_count
    slots = list(range(SOURCE_MARGIN, n - SOURCE_MARGIN))
    if len(slots) < max(vertical_count, horizontal_count):
        return None

    cols = list(map(int, rng.choice(slots, size=vertical_count, replace=False)))
    rows = list(map(int, rng.choice(slots, size=horizontal_count, replace=False)))
    cols = list(rng.permutation(cols))
    rows = list(rng.permutation(rows))

    placements: list[tuple[int, int, int]] = []
    idx = 0
    for c in cols[:top_count]:
        placements.append((int(colors[idx]), 0, int(c)))
        idx += 1
    for c in cols[top_count:]:
        placements.append((int(colors[idx]), n - 1, int(c)))
        idx += 1
    for r in rows[:left_count]:
        placements.append((int(colors[idx]), int(r), 0))
        idx += 1
    for r in rows[left_count:]:
        placements.append((int(colors[idx]), int(r), n - 1))
        idx += 1

    perm = list(rng.permutation(len(placements)))
    return [placements[i] for i in perm]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    counts = _SIDE_COUNT_CHOICES[int(rng.integers(len(_SIDE_COUNT_CHOICES)))]
    source_count = sum(counts)
    vertical_count = counts[0] + counts[1]
    horizontal_count = counts[2] + counts[3]

    bg, colors = _pick_bg_and_colors(rng, source_count)
    placements = _sample_placements(n, counts, colors, rng)
    if placements is None:
        return None

    grid = _render_sources(n, bg, placements)
    if not _input_is_border_only(grid, bg):
        return None
    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid)
    except Exception:
        return None

    if stats["source_count"] != source_count:
        return None
    if stats["vertical_count"] != vertical_count or stats["horizontal_count"] != horizontal_count:
        return None
    if stats["orth_crossings"] != vertical_count * horizontal_count:
        return None
    if stats["orth_crossings"] < MIN_REQUIRED_CROSSINGS:
        return None
    if stats["changed_pixels"] < source_count * MIN_BEAM_PIXELS_PER_SOURCE:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
