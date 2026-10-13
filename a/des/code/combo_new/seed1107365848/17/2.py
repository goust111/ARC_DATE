from __future__ import annotations

import random
from collections import deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]
L_FILL_OFFSETS = {
    (0, 0): [(0, 1), (1, 0), (1, 1)],
    (0, 2): [(0, 1), (1, 1), (1, 2)],
    (2, 0): [(1, 0), (1, 1), (2, 1)],
    (2, 2): [(1, 1), (1, 2), (2, 1)],
}

GRID_SIZE_CHOICES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
TEMPLATE_N_CHOICES = [2, 3, 4, 5]
GEN_RETRIES = 300


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(np.asarray(values, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_template(grid: np.ndarray, bg: int) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict[str, int]] = []

    for color in (int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if (r1 - r0) != (c1 - c0):
            continue
        size = r1 - r0 + 1
        if size < 9 or size % 4 != 1:
            continue

        rows = [i for i in range(size) if np.all(g[r0 + i, c0 : c0 + size] == color)]
        cols = [i for i in range(size) if np.all(g[r0 : r0 + size, c0 + i] == color)]
        expected = list(range(0, size, 4))
        if rows != expected or cols != expected:
            continue

        lattice = np.zeros((size, size), dtype=bool)
        lattice[rows, :] = True
        lattice[:, cols] = True
        if not np.array_equal(g[r0 : r0 + size, c0 : c0 + size] == color, lattice):
            continue

        candidates.append(
            {
                "color": color,
                "top": r0,
                "left": c0,
                "size": size,
                "n": (size - 1) // 4,
            }
        )

    assert len(candidates) == 1, "expected exactly one grey lattice template"
    return candidates[0]


def _collect_clues(grid: np.ndarray, bg: int, grey: int, template: dict[str, int]) -> dict[int, dict[str, int | tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    clues: dict[int, dict[str, int | tuple[int, int]]] = {}
    top = int(template["top"])
    left = int(template["left"])
    n = int(template["n"])

    for cell_r in range(n):
        for cell_c in range(n):
            r0 = top + 1 + 4 * cell_r
            c0 = left + 1 + 4 * cell_c
            cell = g[r0 : r0 + 3, c0 : c0 + 3]
            hint_coords = np.argwhere((cell != bg) & (cell != grey))
            assert len(hint_coords) <= 1, "each template cell may contain at most one hint"
            if len(hint_coords) == 0:
                continue
            dr, dc = map(int, hint_coords[0])
            assert (dr, dc) in L_FILL_OFFSETS, "hint must lie in a cell corner"
            color = int(cell[dr, dc])
            assert color not in clues, "hint colors must be unique"
            clues[color] = {
                "cell_r": cell_r,
                "cell_c": cell_c,
                "cell_top": r0,
                "cell_left": c0,
                "missing": (dr, dc),
            }

    assert clues, "expected at least one template hint"
    return clues


def _collect_pieces(grid: np.ndarray, bg: int, grey: int, template: dict[str, int]) -> dict[int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    inside_template = np.zeros(g.shape, dtype=bool)
    top = int(template["top"])
    left = int(template["left"])
    size = int(template["size"])
    inside_template[top : top + size, left : left + size] = True

    mask = (~inside_template) & (g != bg) & (g != grey)
    pieces: dict[int, list[tuple[int, int]]] = {}
    for comp in _components(mask):
        color_set = {int(g[r, c]) for r, c in comp}
        assert len(color_set) == 1, "each source piece must be monochrome"
        color = color_set.pop()
        coords = np.asarray(comp, dtype=int)
        assert len(comp) == 3, "source pieces must be triominoes"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert (r1 - r0) == 1 and (c1 - c0) == 1, "source pieces must fit a 2x2 box"
        rel = {(int(r - r0), int(c - c0)) for r, c in comp}
        assert len(rel) == 3, "source pieces must occupy three cells of the 2x2 box"
        assert color not in pieces, "source piece colors must be unique"
        pieces[color] = comp

    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack each scattered colored L triomino into the template cell whose corner hint shares its color.

    Input structure:
    - The background is the dominant color.
    - A single grey square lattice in the middle divides an N×N template into 3×3 cells with 1-pixel grid lines.
    - Some template cells contain one non-background, non-grey hint pixel in one of the four cell corners.
    - Outside the template, each hint color appears exactly once as a separate monochrome L triomino occupying three cells of a 2×2 box.

    Rule:
    - Match every hinted template cell with the external source piece of the same color.
    - The hint corner chooses which corner of the 2×2 placement box stays empty inside that 3×3 cell.
    - Erase the external source pieces, erase the hint pixel itself, and draw the corresponding rotated L triomino into the hinted cell.
    - Keep the grey grid lines and all unhinted template cells unchanged.

    Key insight:
    - Color determines which fragment belongs to which cell, while the hint's corner position encodes the fragment's missing-corner orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    template = _find_template(g, bg)
    grey = int(template["color"])
    clues = _collect_clues(g, bg, grey, template)
    pieces = _collect_pieces(g, bg, grey, template)
    assert set(clues) == set(pieces), "hint colors and source-piece colors must match"

    out = g.copy()
    top = int(template["top"])
    left = int(template["left"])
    size = int(template["size"])
    inside_template = np.zeros(g.shape, dtype=bool)
    inside_template[top : top + size, left : left + size] = True
    out[(~inside_template) & (out != bg) & (out != grey)] = bg

    for color, clue in clues.items():
        cell_top = int(clue["cell_top"])
        cell_left = int(clue["cell_left"])
        missing = clue["missing"]
        assert isinstance(missing, tuple)
        miss_r, miss_c = missing
        out[cell_top + miss_r, cell_left + miss_c] = bg
        for dr, dc in L_FILL_OFFSETS[(miss_r, miss_c)]:
            out[cell_top + dr, cell_left + dc] = int(color)

    return out


def _draw_template(grid: np.ndarray, top: int, left: int, n: int, grey: int) -> None:
    size = 4 * n + 1
    for i in range(n + 1):
        r = top + 4 * i
        c = left + 4 * i
        grid[r, left : left + size] = grey
        grid[top : top + size, c] = grey


def _central_offset(extra: int) -> int:
    center = extra // 2
    jitter = random.randint(-2, 2)
    return min(max(center + jitter, 1), extra - 1)


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    blocked[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)] = True


def _place_source_pieces(
    grid: np.ndarray,
    blocked: np.ndarray,
    colors: list[int],
) -> bool:
    h, w = grid.shape
    slots = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    random.shuffle(slots)

    for color in colors:
        placed = False
        for r, c in slots:
            if blocked[r - 1 : r + 3, c - 1 : c + 3].any():
                continue
            missing = random.choice(CORNERS)
            for dr, dc in L_FILL_OFFSETS[missing]:
                grid[r + dr, c + dc] = color
            blocked[r - 1 : r + 3, c - 1 : c + 3] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(TEMPLATE_N_CHOICES)
    template_size = 4 * n + 1
    min_side = max(18, template_size + 8)
    if min_side > 30:
        return None

    piece_count_choices = [4]
    if n >= 3:
        piece_count_choices.append(8)
    piece_count = random.choice(piece_count_choices)

    palette = random.sample(range(10), piece_count + 2)
    bg = palette[0]
    grey = palette[1]
    piece_colors = palette[2:]

    h = random.choice([s for s in GRID_SIZE_CHOICES if s >= min_side])
    w = random.choice([s for s in GRID_SIZE_CHOICES if s >= min_side])
    extra_h = h - template_size
    extra_w = w - template_size
    if extra_h < 2 or extra_w < 2:
        return None

    top = _central_offset(extra_h)
    left = _central_offset(extra_w)

    grid = np.full((h, w), bg, dtype=int)
    _draw_template(grid, top, left, n, grey)

    chosen_cells = random.sample([(r, c) for r in range(n) for c in range(n)], piece_count)
    hint_corners = CORNERS * (piece_count // 4)
    random.shuffle(hint_corners)
    random.shuffle(piece_colors)
    for (cell_r, cell_c), color, corner in zip(chosen_cells, piece_colors, hint_corners):
        r0 = top + 1 + 4 * cell_r
        c0 = left + 1 + 4 * cell_c
        grid[r0 + corner[0], c0 + corner[1]] = color

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    _reserve(blocked, top - 1, left - 1, top + template_size + 1, left + template_size + 1)
    if not _place_source_pieces(grid, blocked, piece_colors):
        return None

    if _mode_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
