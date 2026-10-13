from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ["up", "down", "left", "right"]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MIN_PATH_LEN = 2
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 300


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)

    return comps


def _perpendiculars(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (dc, dr), (-dc, -dr)


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _arrow_spec_from_base(
    base: tuple[int, int],
    direction: tuple[int, int],
    sequence: list[int],
    h: int,
    w: int,
) -> dict[str, object] | None:
    br, bc = map(int, base)
    dr, dc = map(int, direction)
    perp_a, perp_b = _perpendiculars(dr, dc)

    grey_cells = [
        (br, bc),
        (br + dr, bc + dc),
        (br + perp_a[0], bc + perp_a[1]),
        (br + perp_b[0], bc + perp_b[1]),
    ]
    shaft_cells = [
        (br - dr * step, bc - dc * step)
        for step in range(len(sequence), 0, -1)
    ]

    arrow_cells = grey_cells + shaft_cells
    if not all(_in_bounds(h, w, r, c) for r, c in arrow_cells):
        return None

    path: list[tuple[int, int]] = []
    rr, cc = br + 2 * dr, bc + 2 * dc
    while _in_bounds(h, w, rr, cc):
        path.append((int(rr), int(cc)))
        rr += dr
        cc += dc

    if len(path) < MIN_PATH_LEN:
        return None

    return {
        "base": (br, bc),
        "direction": (dr, dc),
        "sequence": list(sequence),
        "grey_cells": grey_cells,
        "shaft_cells": shaft_cells,
        "arrow_cells": arrow_cells,
        "path": path,
    }


def _infer_arrow(comp: list[tuple[int, int]], grid: np.ndarray, bg: int) -> dict[str, object] | None:
    h, w = grid.shape
    comp_set = {(int(r), int(c)) for r, c in comp}
    grey_cells = [(int(r), int(c)) for r, c in comp if int(grid[r, c]) == GREY]
    grey_set = set(grey_cells)

    if len(grey_cells) != 4:
        return None

    base_candidates: list[tuple[int, int, tuple[int, int]]] = []
    for r, c in grey_cells:
        grey_neighbors = []
        for dr, dc in DIR4:
            if (r + dr, c + dc) in grey_set:
                grey_neighbors.append((dr, dc))
        if len(grey_neighbors) != 3:
            continue
        tail_dirs = [d for d in DIR4 if d not in grey_neighbors]
        if len(tail_dirs) != 1:
            continue
        base_candidates.append((r, c, tail_dirs[0]))

    if len(base_candidates) != 1:
        return None

    br, bc, tail_dir = base_candidates[0]
    tr, tc = br + tail_dir[0], bc + tail_dir[1]
    if not _in_bounds(h, w, tr, tc):
        return None
    if (tr, tc) not in comp_set or int(grid[tr, tc]) in (bg, GREY):
        return None

    shaft_near_head: list[tuple[int, int]] = []
    rr, cc = tr, tc
    while _in_bounds(h, w, rr, cc) and (rr, cc) in comp_set and int(grid[rr, cc]) != GREY:
        shaft_near_head.append((int(rr), int(cc)))
        rr += tail_dir[0]
        cc += tail_dir[1]

    if not shaft_near_head:
        return None

    shaft_set = set(shaft_near_head)
    if comp_set != grey_set | shaft_set:
        return None

    direction = (-tail_dir[0], -tail_dir[1])
    sequence = [int(grid[r, c]) for r, c in reversed(shaft_near_head)]
    spec = _arrow_spec_from_base((br, bc), direction, sequence, h, w)
    if spec is None:
        return None

    expected_arrow = set(spec["arrow_cells"])
    if expected_arrow != comp_set:
        return None
    return spec


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect arrow sprites, read the shaft stripe sequence from tail to head,
    and emit that sequence forward from the tip until the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)
    out = g.copy()

    for comp in _components(g != bg):
        spec = _infer_arrow(comp, g, bg)
        if spec is None:
            continue

        sequence = spec["sequence"]
        path = spec["path"]
        for idx, (r, c) in enumerate(path):
            out[int(r), int(c)] = int(sequence[idx % len(sequence)])

    return out


def _sample_sequence() -> list[int]:
    palette_size = random.randint(2, 4)
    palette = random.sample([c for c in range(1, 10) if c != GREY], palette_size)
    length = random.randint(palette_size, palette_size + 2)

    seq = palette[:]
    random.shuffle(seq)
    while len(seq) < length:
        choices = [c for c in palette if c != seq[-1]]
        seq.append(random.choice(choices))
    return [int(c) for c in seq]


def _dilate_chebyshev(cells: list[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(h, w, rr, cc):
                    out.add((int(rr), int(cc)))
    return out


def _render_input(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for spec in specs:
        for r, c in spec["grey_cells"]:
            grid[int(r), int(c)] = GREY
        for (r, c), color in zip(spec["shaft_cells"], spec["sequence"], strict=False):
            grid[int(r), int(c)] = int(color)
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in specs:
        seq = spec["sequence"]
        for idx, (r, c) in enumerate(spec["path"]):
            out[int(r), int(c)] = int(seq[idx % len(seq)])
    return out


def _place_arrows(h: int, w: int, arrow_count: int) -> list[dict[str, object]] | None:
    specs: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()

    directions = random.sample(DIR_NAMES, k=min(arrow_count, len(DIR_NAMES)))
    while len(directions) < arrow_count:
        directions.append(random.choice(DIR_NAMES))
    random.shuffle(directions)

    for direction_name in directions:
        placed = False
        direction = DIR_VECTORS[direction_name]
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            sequence = _sample_sequence()
            base = (random.randint(0, h - 1), random.randint(0, w - 1))
            spec = _arrow_spec_from_base(base, direction, sequence, h, w)
            if spec is None:
                continue

            exact_cells = list(spec["arrow_cells"]) + list(spec["path"])
            if any(cell in blocked for cell in exact_cells):
                continue

            padded_arrow = _dilate_chebyshev(list(spec["arrow_cells"]), h, w, radius=1)
            if any(cell in blocked for cell in padded_arrow):
                continue

            specs.append(spec)
            blocked.update(spec["path"])
            blocked.update(padded_arrow)
            placed = True
            break

        if not placed:
            return None

    return specs


def _generate_case() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    area = h * w
    max_arrows = 4 if area >= 220 else 3 if area >= 160 else 2
    arrow_count = random.randint(1, max_arrows)

    specs = _place_arrows(h, w, arrow_count)
    if specs is None:
        return None

    inp = _render_input(h, w, specs)
    expected = _render_expected(inp, specs)

    try:
        got = main(inp)
    except Exception:
        return None

    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-emission input")
