from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 3
LEGEND_COLORS = (1, 2, 4)
OUTPUT_BLOCK = 3
MAX_GENERATION_ATTEMPTS = 400

SHAPE_CATALOG = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    mask[rr, cc] = True
    return mask


def _find_divider_corner(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for r in range(1, h):
        for c in range(1, w):
            if int(g[r, c]) != DIVIDER:
                continue
            if np.all(g[r, : c + 1] == DIVIDER) and np.all(g[: r + 1, c] == DIVIDER):
                return int(r), int(c)
    raise AssertionError("expected one top-left L-shaped divider")


def _read_legend(grid: np.ndarray, divider_r: int, divider_c: int) -> dict[int, np.ndarray]:
    legend = np.asarray(grid[:divider_r, :divider_c], dtype=int)
    mapping: dict[int, np.ndarray] = {}
    for color in LEGEND_COLORS:
        comps = _components_of_color(legend, color)
        if not comps:
            continue
        assert len(comps) == 1, "each legend color must appear in one component"
        mapping[int(color)] = _mask_from_coords(comps[0])
    assert 2 <= len(mapping) <= 3, "legend must contain 2-3 template sprites"
    return mapping


def _control_bbox(grid: np.ndarray, divider_r: int, divider_c: int) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[divider_r:, divider_c:] = True
    mask &= g != BG
    mask &= g != DIVIDER
    coords = np.argwhere(mask)
    assert coords.size > 0, "missing control grid"
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "control grid must occupy a 3x3 box"
    return bbox


def main(grid: np.ndarray) -> np.ndarray:
    """Index legend shapes by color and stamp the matched sprite into each 3x3 output slot.

    Input structure
    - A black background contains a green L-shaped divider touching the top and left edges.
    - The divider encloses a top-left legend region holding 2-3 disconnected monochrome template sprites, each in a distinct non-black color.
    - A separate 3x3 control grid sits in the lower-right area, and each control cell is either black or one of the legend colors.

    Rule
    - Read each legend sprite as a mapping from its color to its tight-bbox binary mask.
    - Read the lower-right 3x3 control grid cell by cell.
    - Build a 9x9 black output split into 3x3 block slots; for every non-black control color, stamp that color's legend mask into the aligned output slot with top-left alignment.

    Key insight
    - The control grid does not directly show the output pixels; each color is only an index into the legend, so the solver must follow the color -> shape -> position chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_r, divider_c = _find_divider_corner(g)
    legend = _read_legend(g, divider_r, divider_c)
    cr0, cc0, cr1, cc1 = _control_bbox(g, divider_r, divider_c)
    control = g[cr0 : cr1 + 1, cc0 : cc1 + 1]

    out = np.zeros((9, 9), dtype=int)
    for br in range(3):
        for bc in range(3):
            color = int(control[br, bc])
            if color == BG:
                continue
            assert color in legend, "control color must exist in the legend"
            mask = legend[color]
            mh, mw = mask.shape
            slot = out[
                br * OUTPUT_BLOCK : br * OUTPUT_BLOCK + mh,
                bc * OUTPUT_BLOCK : bc * OUTPUT_BLOCK + mw,
            ]
            slot[mask] = color
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint_shape(grid: np.ndarray, top: tuple[int, int], color: int, shape: tuple[tuple[int, int], ...]) -> None:
    r0, c0 = map(int, top)
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = int(color)


def _legend_layout(
    rng: random.Random,
    shapes: list[tuple[tuple[int, int], ...]],
) -> tuple[int, int, list[tuple[int, int]]]:
    shape_dims = [_shape_dims(shape) for shape in shapes]
    max_h = max(h for h, _ in shape_dims)
    legend_h = rng.randint(max_h + 1, 6)
    gaps = [rng.randint(1, 2) for _ in range(len(shapes) - 1)]
    left_pad = rng.randint(0, 1)
    right_pad = rng.randint(0, 2)
    legend_w = left_pad + sum(w for _, w in shape_dims) + sum(gaps) + right_pad
    positions: list[tuple[int, int]] = []
    col = left_pad
    for idx, (h, w) in enumerate(shape_dims):
        row = rng.randint(0, legend_h - h)
        positions.append((row, col))
        col += w
        if idx < len(gaps):
            col += gaps[idx]
    return legend_h, legend_w, positions


def _valid_control_positions(coords: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    return len(rows) == 3 and len(cols) == 3 and (2, 2) in coords


def _make_control(
    rng: random.Random,
    colors: list[int],
) -> np.ndarray:
    while True:
        nonblack_count = rng.randint(max(5, len(colors)), 9)
        all_cells = [(r, c) for r in range(3) for c in range(3)]
        chosen = rng.sample(all_cells, nonblack_count)
        if not _valid_control_positions(chosen):
            continue

        values = colors.copy()
        while len(values) < nonblack_count:
            values.append(rng.choice(colors))
        rng.shuffle(values)

        control = np.zeros((3, 3), dtype=int)
        for (r, c), color in zip(chosen, values):
            control[r, c] = int(color)
        if len(set(values)) >= 2:
            return control


def _build_input(rng: random.Random) -> np.ndarray:
    template_count = rng.randint(2, 3)
    colors = list(rng.sample(LEGEND_COLORS, template_count))
    shapes = list(rng.sample(SHAPE_CATALOG, template_count))
    legend_h, legend_w, positions = _legend_layout(rng, shapes)

    top_gap = legend_h + 1
    left_gap = legend_w + 1
    control_r = top_gap + rng.randint(2, 6)
    control_c = left_gap + rng.randint(2, 6)
    bottom_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    height = control_r + 3 + bottom_margin
    width = control_c + 3 + right_margin
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)
    grid[legend_h, : legend_w + 1] = DIVIDER
    grid[: legend_h + 1, legend_w] = DIVIDER

    for color, shape, pos in zip(colors, shapes, positions):
        _paint_shape(grid, pos, color, shape)

    control = _make_control(rng, colors)
    grid[control_r : control_r + 3, control_c : control_c + 3] = control
    return grid


def _is_valid_example(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    divider_r, divider_c = _find_divider_corner(inp)
    legend = _read_legend(inp, divider_r, divider_c)
    colors_used = set(map(int, np.unique(inp))) - {BG, DIVIDER}
    if len(legend) < 2:
        return False
    if not set(legend).issubset(colors_used):
        return False
    if out.shape != (9, 9):
        return False
    control_bbox = _control_bbox(inp, divider_r, divider_c)
    cr0, cc0, cr1, cc1 = control_bbox
    control = inp[cr0 : cr1 + 1, cc0 : cc1 + 1]
    if len({int(v) for v in control.flat if int(v) != BG}) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        inp = _build_input(rng)
        if _is_valid_example(inp):
            return inp
    raise RuntimeError("generate_input: failed to build a valid instance")
