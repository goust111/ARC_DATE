from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MOTIF = 5
LEGEND_FRAME = 6
LEGEND_PALETTE = (1, 2, 3, 4, 7, 8, 9)
MOTIF_SIZE = 5
INNER_SLICE = slice(1, 4)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _valid_legend_colors(colors: list[int]) -> bool:
    return len(colors) == 5 and len(set(colors)) == 5 and all(color not in (BG, MOTIF, LEGEND_FRAME) for color in colors)


def _find_legend_mapping(grid: np.ndarray) -> list[int]:
    candidates: list[list[int]] = []
    for coords in _components_mask(grid == LEGEND_FRAME):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if (h, w) not in ((3, 7), (7, 3)):
            continue
        if not (np.all(patch[0, :] == LEGEND_FRAME) and np.all(patch[-1, :] == LEGEND_FRAME)):
            continue
        if not (np.all(patch[:, 0] == LEGEND_FRAME) and np.all(patch[:, -1] == LEGEND_FRAME)):
            continue
        inner = patch[1:-1, 1:-1]
        if np.any(inner == LEGEND_FRAME):
            continue
        if inner.shape == (1, 5):
            colors = [int(v) for v in inner[0, :]]
        elif inner.shape == (5, 1):
            colors = [int(v) for v in inner[:, 0]]
        else:
            continue
        if _valid_legend_colors(colors):
            candidates.append(colors)
    assert len(candidates) == 1, "expected exactly one valid legend frame"
    return candidates[0]


def _find_motif_grid(grid: np.ndarray) -> tuple[list[int], list[int], dict[tuple[int, int], np.ndarray]]:
    motifs: dict[tuple[int, int], np.ndarray] = {}
    for coords in _components_mask(grid == MOTIF):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape != (MOTIF_SIZE, MOTIF_SIZE):
            continue
        if not (np.all(patch[0, :] == MOTIF) and np.all(patch[-1, :] == MOTIF)):
            continue
        if not (np.all(patch[:, 0] == MOTIF) and np.all(patch[:, -1] == MOTIF)):
            continue
        if not np.all(np.isin(patch[INNER_SLICE, INNER_SLICE], [BG, MOTIF])):
            continue
        motifs[(r0, c0)] = patch.copy()

    assert motifs, "expected a motif array"
    rows = sorted({r for r, _c in motifs})
    cols = sorted({c for _r, c in motifs})
    assert len(motifs) == len(rows) * len(cols), "motifs must form a full rectangle"
    assert all(rows[i + 1] - rows[i] == 6 for i in range(len(rows) - 1)), "motif rows must be separated by one background row"
    assert all(cols[i + 1] - cols[i] == 6 for i in range(len(cols) - 1)), "motif cols must be separated by one background col"
    for r0 in rows:
        for c0 in cols:
            assert (r0, c0) in motifs, "missing motif from rectangular array"
    return rows, cols, motifs


def _hole_count(patch: np.ndarray) -> int:
    inner = patch[INNER_SLICE, INNER_SLICE] == BG
    count = len(_components_mask(inner))
    assert 0 <= count <= 4
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each 5x5 motif by its enclosed hole count, then compress the motif lattice using the external legend bar.

    Input structure:
    - The background is black.
    - A rectangular array of 5x5 motifs is drawn in gray, with one black row/column between neighboring motifs.
    - Each motif has a 1-pixel gray border and a 3x3 interior containing gray cells plus 0 to 4 black hole cells.
    - Outside the array there is one magenta rectangular frame around a 5-cell legend strip, laid out horizontally or vertically.

    Rule:
    - Read the legend strip in order: left-to-right for a horizontal strip, top-to-bottom for a vertical strip.
    - The legend position index 0..4 gives the output color for motifs with that many enclosed 4-connected black hole regions.
    - For every motif in the array, count the number of 4-connected black hole components in its 3x3 interior.
    - Return the compressed R x C grid whose cell color is the legend-mapped color of that motif.

    Key insight:
    - The motifs do not encode color directly; they encode a topological feature (hole-count), and the external framed legend translates that feature into the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _find_legend_mapping(g)
    rows, cols, motifs = _find_motif_grid(g)

    out = np.zeros((len(rows), len(cols)), dtype=int)
    for rr, r0 in enumerate(rows):
        for cc, c0 in enumerate(cols):
            holes = _hole_count(motifs[(r0, c0)])
            out[rr, cc] = legend[holes]
    return out


def _independent_inner_patterns() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    by_count: dict[int, list[tuple[tuple[int, int], ...]]] = {0: [tuple()]}
    for k in range(1, 5):
        valid: list[tuple[tuple[int, int], ...]] = []
        for combo in itertools.combinations(cells, k):
            ok = True
            for i in range(k):
                for j in range(i + 1, k):
                    if abs(combo[i][0] - combo[j][0]) + abs(combo[i][1] - combo[j][1]) == 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid.append(combo)
        by_count[k] = valid
    return by_count


PATTERNS_BY_HOLE_COUNT = _independent_inner_patterns()


def _build_motif(hole_count: int) -> np.ndarray:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), MOTIF, dtype=int)
    pattern = random.choice(PATTERNS_BY_HOLE_COUNT[hole_count])
    for r, c in pattern:
        patch[r + 1, c + 1] = BG
    return patch


def _render_array(count_grid: np.ndarray) -> np.ndarray:
    r_count, c_count = count_grid.shape
    h = r_count * MOTIF_SIZE + (r_count - 1)
    w = c_count * MOTIF_SIZE + (c_count - 1)
    canvas = np.full((h, w), BG, dtype=int)
    for rr in range(r_count):
        for cc in range(c_count):
            patch = _build_motif(int(count_grid[rr, cc]))
            r0 = rr * (MOTIF_SIZE + 1)
            c0 = cc * (MOTIF_SIZE + 1)
            canvas[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE] = patch
    return canvas


def _render_legend(colors: list[int], orientation: str) -> np.ndarray:
    if orientation == "h":
        patch = np.full((3, 7), LEGEND_FRAME, dtype=int)
        patch[1, 1:6] = np.array(colors, dtype=int)
        return patch
    patch = np.full((7, 3), LEGEND_FRAME, dtype=int)
    patch[1:6, 1] = np.array(colors, dtype=int)
    return patch


def _sample_count_grid(r_count: int, c_count: int) -> np.ndarray:
    total = r_count * c_count
    if total >= 5 and random.random() < 0.6:
        active = [0, 1, 2, 3, 4]
    else:
        min_distinct = min(3, total)
        active = random.sample(range(5), random.randint(min_distinct, min(5, total)))
        if total >= 4 and 0 not in active and random.random() < 0.5:
            active[random.randrange(len(active))] = 0
        if total >= 4 and 4 not in active and random.random() < 0.5:
            active[random.randrange(len(active))] = 4
        active = sorted(set(active))

    flat = list(active)
    while len(flat) < total:
        flat.append(random.choice(active))
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(r_count, c_count)


def _compose_scene(
    motif_array: np.ndarray,
    legend_patch: np.ndarray,
    side: str,
) -> np.ndarray | None:
    array_h, array_w = motif_array.shape
    legend_h, legend_w = legend_patch.shape

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    gap = random.randint(1, 3)
    slack = random.randint(0, 4)

    if side == "right":
        height = max(array_h, legend_h) + top_pad + bottom_pad + slack
        width = left_pad + array_w + gap + legend_w + right_pad
        if height > 30 or width > 30:
            return None
        grid = np.full((height, width), BG, dtype=int)
        array_top = random.randint(top_pad, height - bottom_pad - array_h)
        legend_top = random.randint(top_pad, height - bottom_pad - legend_h)
        array_left = left_pad
        legend_left = left_pad + array_w + gap
        grid[array_top : array_top + array_h, array_left : array_left + array_w] = motif_array
        grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = legend_patch
        return grid

    height = top_pad + array_h + gap + legend_h + bottom_pad
    width = max(array_w, legend_w) + left_pad + right_pad + slack
    if height > 30 or width > 30:
        return None
    grid = np.full((height, width), BG, dtype=int)
    array_left = random.randint(left_pad, width - right_pad - array_w)
    legend_left = random.randint(left_pad, width - right_pad - legend_w)
    array_top = top_pad
    legend_top = top_pad + array_h + gap
    grid[array_top : array_top + array_h, array_left : array_left + array_w] = motif_array
    grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = legend_patch
    return grid


def _expected_output(count_grid: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    return np.vectorize(lambda v: legend_colors[int(v)])(count_grid).astype(int)


def _is_semantic_example(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(np.unique(expected)) < 2:
        return False
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _generate_attempt() -> np.ndarray | None:
    orientation = random.choice(["h", "v"])
    side = random.choice(["right", "bottom"])
    r_count = random.randint(2, 4)
    c_count = random.randint(2, 4)

    count_grid = _sample_count_grid(r_count, c_count)
    motif_array = _render_array(count_grid)
    legend_colors = random.sample(list(LEGEND_PALETTE), 5)
    legend_patch = _render_legend(legend_colors, orientation)
    grid = _compose_scene(motif_array, legend_patch, side)
    if grid is None:
        return None

    expected = _expected_output(count_grid, legend_colors)
    if not _is_semantic_example(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
