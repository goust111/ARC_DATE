from __future__ import annotations

import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))

_ARROW_SPECS = {
    "right": {
        "cells": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
        "dir": (0, 1),
        "shaft_mid": (1, 1),
        "tip_base": (1, 2),
    },
    "left": {
        "cells": {(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)},
        "dir": (0, -1),
        "shaft_mid": (1, 1),
        "tip_base": (1, 0),
    },
    "up": {
        "cells": {(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)},
        "dir": (-1, 0),
        "shaft_mid": (1, 1),
        "tip_base": (0, 1),
    },
    "down": {
        "cells": {(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)},
        "dir": (1, 0),
        "shaft_mid": (1, 1),
        "tip_base": (2, 1),
    },
}


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in _DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _match_arrow(coords: np.ndarray) -> dict[str, object] | None:
    if coords.shape != (5, 2):
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if int(r1 - r0) != 2 or int(c1 - c0) != 2:
        return None

    normalized = {(int(r - r0), int(c - c0)) for r, c in coords}
    for name, spec in _ARROW_SPECS.items():
        if normalized != spec["cells"]:
            continue
        mid_r, mid_c = spec["shaft_mid"]
        tip_r, tip_c = spec["tip_base"]
        return {
            "name": name,
            "coords": coords.copy(),
            "coord_set": {tuple(map(int, rc)) for rc in coords},
            "dir": spec["dir"],
            "shaft_mid": (int(r0 + mid_r), int(c0 + mid_c)),
            "tip_base": (int(r0 + tip_r), int(c0 + tip_c)),
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a ray from each colored arrow to its matching barrier, then mirror the barrier's hole state back into the arrow.

    Input structure:
    - The grid has a dominant black background and one or more foreground colors.
    - For each foreground color there is exactly one 5-pixel arrow: a 3-cell shaft plus
      two head pixels, fitting one of the four 3x3 directional arrow templates.
    - The same color also forms one perpendicular barrier ahead of the arrow; the barrier
      spans 3 to 5 cells along a single row or column and may be missing one interior cell.

    Rule:
    - For each color, identify the arrow and follow the arrow direction starting one cell
      beyond its tip until the first same-color barrier pixel is reached; paint that path.
    - Do not overwrite the barrier itself; the ray stops on the cell immediately before it.
    - If the barrier span contains a hole, erase the middle cell of the arrow shaft;
      otherwise leave the arrow unchanged.

    Key insight:
    - The ray connection is only half of the rule: the barrier's own structure encodes a
      boolean instruction that modifies the source arrow after the ray has been drawn.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        mask = g == color
        comps = _components(mask)
        arrows = [match for comp in comps if (match := _match_arrow(comp)) is not None]
        assert len(arrows) == 1, f"expected one arrow for color {color}"
        arrow = arrows[0]

        arrow_mask = np.zeros((h, w), dtype=bool)
        for r, c in arrow["coord_set"]:
            arrow_mask[r, c] = True

        barrier_coords = np.argwhere(mask & ~arrow_mask)
        assert barrier_coords.size > 0, f"expected a barrier for color {color}"
        dr, dc = arrow["dir"]

        if dr == 0:
            barrier_cols = sorted(set(int(c) for c in barrier_coords[:, 1]))
            assert len(barrier_cols) == 1, f"horizontal arrow color {color} needs one vertical barrier"
            span_min = int(barrier_coords[:, 0].min())
            span_max = int(barrier_coords[:, 0].max())
        else:
            barrier_rows = sorted(set(int(r) for r in barrier_coords[:, 0]))
            assert len(barrier_rows) == 1, f"vertical arrow color {color} needs one horizontal barrier"
            span_min = int(barrier_coords[:, 1].min())
            span_max = int(barrier_coords[:, 1].max())

        span = span_max - span_min + 1
        count = int(len(barrier_coords))
        assert 3 <= span <= 5, f"barrier span out of range for color {color}"
        assert count in {span, span - 1}, f"invalid barrier occupancy for color {color}"
        has_hole = count < span

        tip_r, tip_c = arrow["tip_base"]
        ray_r = tip_r + dr
        ray_c = tip_c + dc
        painted: list[tuple[int, int]] = []
        hit = False

        while 0 <= ray_r < h and 0 <= ray_c < w:
            if int(g[ray_r, ray_c]) == color and (ray_r, ray_c) not in arrow["coord_set"]:
                hit = True
                break
            assert int(g[ray_r, ray_c]) == bg, f"ray path must stay on background for color {color}"
            painted.append((ray_r, ray_c))
            ray_r += dr
            ray_c += dc

        assert hit, f"ray for color {color} must hit its barrier"
        for r, c in painted:
            out[r, c] = color

        if has_hole:
            mid_r, mid_c = arrow["shaft_mid"]
            out[mid_r, mid_c] = bg

    return out


_BG = 0
_PALETTE = tuple(range(1, 10))
_PAIR_COUNT_RANGE = (1, 4)
_GRID_MAX = 30
_GEN_ATTEMPTS = 400
_PAIR_ATTEMPTS = 250
_GAP_MIN = 1
_GAP_MAX = 9


def _hole_pattern(n_pairs: int) -> list[bool]:
    if n_pairs == 1:
        return [bool(random.getrandbits(1))]

    holes = [True] * (n_pairs // 2) + [False] * (n_pairs // 2)
    if len(holes) < n_pairs:
        holes.append(bool(random.getrandbits(1)))
    random.shuffle(holes)
    return holes


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in _DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _build_layout(
    h: int,
    w: int,
    direction: str,
    length: int,
    has_hole: bool,
    gap: int,
    hit_index: int,
    arrow_top: int,
    arrow_left: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    spec = _ARROW_SPECS[direction]
    arrow = [(arrow_top + rr, arrow_left + cc) for rr, cc in spec["cells"]]
    tip_r = arrow_top + spec["tip_base"][0]
    tip_c = arrow_left + spec["tip_base"][1]
    hole_index = length // 2 if has_hole else None

    if direction == "right":
        barrier_top = tip_r - hit_index
        barrier_col = tip_c + gap + 1
        barrier = [
            (barrier_top + i, barrier_col)
            for i in range(length)
            if i != hole_index
        ]
        path = [(tip_r, tip_c + step) for step in range(1, gap + 1)]
    elif direction == "left":
        barrier_top = tip_r - hit_index
        barrier_col = tip_c - gap - 1
        barrier = [
            (barrier_top + i, barrier_col)
            for i in range(length)
            if i != hole_index
        ]
        path = [(tip_r, tip_c - step) for step in range(1, gap + 1)]
    elif direction == "down":
        barrier_row = tip_r + gap + 1
        barrier_left = tip_c - hit_index
        barrier = [
            (barrier_row, barrier_left + i)
            for i in range(length)
            if i != hole_index
        ]
        path = [(tip_r + step, tip_c) for step in range(1, gap + 1)]
    else:
        barrier_row = tip_r - gap - 1
        barrier_left = tip_c - hit_index
        barrier = [
            (barrier_row, barrier_left + i)
            for i in range(length)
            if i != hole_index
        ]
        path = [(tip_r - step, tip_c) for step in range(1, gap + 1)]

    footprint = arrow + barrier + path
    if not footprint:
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
        return None
    return arrow, barrier, path


def _sample_pair_layout(
    h: int,
    w: int,
    direction: str,
    has_hole: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    lengths = [3, 5] if has_hole else [3, 4, 5]
    axis = w if direction in {"left", "right"} else h
    gap_hi = min(_GAP_MAX, axis - 6)
    if gap_hi < _GAP_MIN:
        return None

    for _ in range(_PAIR_ATTEMPTS):
        length = random.choice(lengths)
        hole_index = length // 2 if has_hole else None
        hit_choices = [i for i in range(length) if i != hole_index]
        hit_index = random.choice(hit_choices)
        gap = random.randint(_GAP_MIN, gap_hi)

        if direction == "right":
            row_lo = max(0, hit_index - 1)
            row_hi = min(h - 3, h - length + hit_index - 1)
            col_lo = 0
            col_hi = w - gap - 4
        elif direction == "left":
            row_lo = max(0, hit_index - 1)
            row_hi = min(h - 3, h - length + hit_index - 1)
            col_lo = gap + 1
            col_hi = w - 3
        elif direction == "down":
            row_lo = 0
            row_hi = h - gap - 4
            col_lo = max(0, hit_index - 1)
            col_hi = min(w - 3, w - length + hit_index - 1)
        else:
            row_lo = gap + 1
            row_hi = h - 3
            col_lo = max(0, hit_index - 1)
            col_hi = min(w - 3, w - length + hit_index - 1)

        if row_lo > row_hi or col_lo > col_hi:
            continue

        arrow_top = random.randint(row_lo, row_hi)
        arrow_left = random.randint(col_lo, col_hi)
        layout = _build_layout(
            h,
            w,
            direction,
            length,
            has_hole,
            gap,
            hit_index,
            arrow_top,
            arrow_left,
        )
        if layout is not None:
            return layout
    return None


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*_PAIR_COUNT_RANGE)
    base = min(_GRID_MAX, 10 + 4 * n_pairs)
    if base > _GRID_MAX:
        return None

    h = random.randint(base, _GRID_MAX)
    w = random.randint(base, _GRID_MAX)
    grid = np.full((h, w), _BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    colors = random.sample(_PALETTE, n_pairs)
    hole_flags = _hole_pattern(n_pairs)

    for color, has_hole in zip(colors, hole_flags):
        placed = False
        for _ in range(_PAIR_ATTEMPTS):
            direction = random.choice(tuple(_ARROW_SPECS))
            layout = _sample_pair_layout(h, w, direction, has_hole)
            if layout is None:
                continue
            arrow, barrier, path = layout
            footprint = set(arrow) | set(barrier) | set(path)
            if any(blocked[r, c] for r, c in footprint):
                continue

            for r, c in arrow:
                grid[r, c] = color
            for r, c in barrier:
                grid[r, c] = color

            for r, c in _expanded(footprint, (h, w)):
                blocked[r, c] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if out.shape != inp.shape:
            continue
        if inp.ndim != 2 or max(inp.shape) > 30:
            continue
        if out.ndim != 2 or max(out.shape) > 30:
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {_GEN_ATTEMPTS} attempts")
