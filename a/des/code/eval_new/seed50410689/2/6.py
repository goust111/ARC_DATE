from __future__ import annotations

import random

import numpy as np


DIVIDER_COLOR = 5
BACKGROUND_PALETTE = (3, 4, 7, 8)
TOWER_PALETTE = (1, 2, 6, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORTH_DIRS = tuple(DIRS.values())
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for dr, dc in ORTH_DIRS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((int(rr), int(cc)))
        comps.append(comp)
    return comps


def _majority_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_dividers(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if np.all(grid[r] == grid[r, 0])]
    col_candidates = [c for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    assert len(row_candidates) == 1, "expected exactly one constant divider row"
    assert len(col_candidates) == 1, "expected exactly one constant divider column"
    div_r = int(row_candidates[0])
    div_c = int(col_candidates[0])
    divider = int(grid[div_r, 0])
    assert int(grid[0, div_c]) == divider, "divider row and column must share one color"
    return div_r, div_c, divider


def _quadrant_colors(grid: np.ndarray, div_r: int, div_c: int) -> dict[tuple[int, int], int]:
    quads = {
        (-1, -1): grid[:div_r, :div_c],
        (-1, 1): grid[:div_r, div_c + 1 :],
        (1, -1): grid[div_r + 1 :, :div_c],
        (1, 1): grid[div_r + 1 :, div_c + 1 :],
    }
    colors = {key: _majority_color(block) for key, block in quads.items()}
    assert len(set(colors.values())) == 4, "quadrant background colors must be distinct"
    return colors


def _quadrant_key(r: int, c: int, div_r: int, div_c: int) -> tuple[int, int]:
    assert r != div_r and c != div_c, "pivot cell cannot lie on a divider"
    return (-1 if r < div_r else 1, -1 if c < div_c else 1)


def _tower_cells(center: tuple[int, int], missing: str) -> set[tuple[int, int]]:
    r, c = center
    cells = {(int(r), int(c))}
    for name, (dr, dc) in DIRS.items():
        if name == missing:
            continue
        cells.add((int(r + dr), int(c + dc)))
    return cells


def _parse_tower(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(comp) != 4:
        return None
    pts = set(comp)
    centers = []
    for r, c in comp:
        deg = sum((r + dr, c + dc) in pts for dr, dc in ORTH_DIRS)
        if deg == 3:
            centers.append((int(r), int(c)))
    if len(centers) != 1:
        return None
    center = centers[0]
    missing = [name for name, (dr, dc) in DIRS.items() if (center[0] + dr, center[1] + dc) not in pts]
    if len(missing) != 1:
        return None
    return {"center": center, "missing": missing[0], "cells": pts}


def _ray_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(int(r0), int(c)) for c in range(c0 + step, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(int(r), int(c0)) for r in range(r0 + step, r1 + step, step)]
    raise AssertionError("ray endpoints must share a row or column")


def _pair_path(
    tower_a: dict[str, object],
    tower_b: dict[str, object],
    div_r: int,
    div_c: int,
    quad_colors: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], int]:
    center_a = tower_a["center"]
    center_b = tower_b["center"]
    dir_a = tower_a["missing"]
    dir_b = tower_b["missing"]
    assert isinstance(center_a, tuple) and isinstance(center_b, tuple)
    assert isinstance(dir_a, str) and isinstance(dir_b, str)

    horiz = {"left", "right"}
    vert = {"up", "down"}

    if dir_a in horiz and dir_b in vert or dir_a in vert and dir_b in horiz:
        tower_h = tower_a if dir_a in horiz else tower_b
        tower_v = tower_b if tower_h is tower_a else tower_a
        center_h = tower_h["center"]
        center_v = tower_v["center"]
        dir_h = tower_h["missing"]
        dir_v = tower_v["missing"]
        assert isinstance(center_h, tuple) and isinstance(center_v, tuple)
        assert isinstance(dir_h, str) and isinstance(dir_v, str)

        corner = (int(center_h[0]), int(center_v[1]))
        if dir_h == "left":
            assert corner[1] < int(center_h[1])
        else:
            assert dir_h == "right"
            assert corner[1] > int(center_h[1])
        if dir_v == "up":
            assert corner[0] < int(center_v[0])
        else:
            assert dir_v == "down"
            assert corner[0] > int(center_v[0])

        cells = list(dict.fromkeys(_ray_cells(center_h, corner) + _ray_cells(center_v, corner)))
        color = quad_colors[_quadrant_key(corner[0], corner[1], div_r, div_c)]
        return cells, int(color)

    if dir_a in horiz and dir_b in horiz:
        assert int(center_a[0]) == int(center_b[0]), "horizontal straight pair must share a row"
        left = tower_a if int(center_a[1]) < int(center_b[1]) else tower_b
        right = tower_b if left is tower_a else tower_a
        assert left["missing"] == "right" and right["missing"] == "left"
        left_center = left["center"]
        right_center = right["center"]
        assert isinstance(left_center, tuple) and isinstance(right_center, tuple)
        total = int(left_center[1]) + int(right_center[1])
        assert total % 2 == 0, "straight horizontal path must have a cell midpoint"
        cells = _ray_cells(left_center, right_center)[:-1]
        midpoint = (int(left_center[0]), total // 2)
        color = quad_colors[_quadrant_key(midpoint[0], midpoint[1], div_r, div_c)]
        return cells, int(color)

    if dir_a in vert and dir_b in vert:
        assert int(center_a[1]) == int(center_b[1]), "vertical straight pair must share a column"
        top = tower_a if int(center_a[0]) < int(center_b[0]) else tower_b
        bottom = tower_b if top is tower_a else tower_a
        assert top["missing"] == "down" and bottom["missing"] == "up"
        top_center = top["center"]
        bottom_center = bottom["center"]
        assert isinstance(top_center, tuple) and isinstance(bottom_center, tuple)
        total = int(top_center[0]) + int(bottom_center[0])
        assert total % 2 == 0, "straight vertical path must have a cell midpoint"
        cells = _ray_cells(top_center, bottom_center)[:-1]
        midpoint = (total // 2, int(top_center[1]))
        color = quad_colors[_quadrant_key(midpoint[0], midpoint[1], div_r, div_c)]
        return cells, int(color)

    raise AssertionError("signal-tower directions must form an L or a straight connection")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored signal towers using direction-coded paths and quadrant-colored links.

    Input structure:
    - A grey horizontal divider row and a grey vertical divider column split the
      grid into four colored background quadrants.
    - Scattered on the quadrants are small monochrome signal towers. Each tower is
      a 3x3 cross missing exactly one arm, so it uses four pixels total.
    - Every tower color appears exactly twice, and the pair members are the two
      towers that must be connected.

    Rule:
    - For each tower, recover its center and the direction of the missing arm.
    - For each same-colored pair, extend a ray from each center along its missing
      arm direction. The two rays either meet as an L-shaped path or form one
      straight segment between opposite-facing towers.
    - Paint the whole connection path with the background color of the pivot cell:
      the L-corner for bent paths or the midpoint cell for straight paths. Keep
      the towers themselves unchanged.

    Key insight:
    Tower color defines the pairing, the missing arm encodes the path direction,
    and the connection color comes from the quadrant background at the path's
    geometric pivot rather than from the tower color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    div_r, div_c, divider = _find_dividers(g)
    quad_colors = _quadrant_colors(g, div_r, div_c)

    towers_by_color: dict[int, list[dict[str, object]]] = {}
    for color in sorted(int(v) for v in np.unique(g)):
        if color == divider:
            continue
        for comp in _components(g == color):
            info = _parse_tower(comp)
            if info is not None:
                towers_by_color.setdefault(int(color), []).append(info)

    assert towers_by_color, "expected at least one paired tower color"
    for color, towers in towers_by_color.items():
        assert len(towers) == 2, f"tower color {color} must appear exactly twice"

    tower_mask = np.zeros_like(g, dtype=bool)
    for towers in towers_by_color.values():
        for tower in towers:
            for r, c in tower["cells"]:
                tower_mask[int(r), int(c)] = True

    out = g.copy()
    for color in sorted(towers_by_color):
        path_cells, line_color = _pair_path(towers_by_color[color][0], towers_by_color[color][1], div_r, div_c, quad_colors)
        for r, c in path_cells:
            if not tower_mask[int(r), int(c)]:
                out[int(r), int(c)] = int(line_color)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (17, 29)
GRID_WIDTH_RANGE = (17, 29)
PAIR_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 240
MIN_RAY_LEN = 2


def _odd_rand(lo: int, hi: int) -> int:
    choices = [v for v in range(int(lo), int(hi) + 1) if v % 2 == 1]
    return int(random.choice(choices))


def _blank_grid(h: int, w: int, quad_order: tuple[int, int, int, int]) -> np.ndarray:
    div_r = h // 2
    div_c = w // 2
    tl, tr, bl, br = map(int, quad_order)
    grid = np.empty((h, w), dtype=int)
    grid[:div_r, :div_c] = tl
    grid[:div_r, div_c + 1 :] = tr
    grid[div_r + 1 :, :div_c] = bl
    grid[div_r + 1 :, div_c + 1 :] = br
    grid[div_r, :] = DIVIDER_COLOR
    grid[:, div_c] = DIVIDER_COLOR
    return grid


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _valid_tower_cells(
    tower_cells: set[tuple[int, int]],
    h: int,
    w: int,
    div_r: int,
    div_c: int,
) -> bool:
    for r, c in tower_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r == div_r or c == div_c:
            return False
    return True


def _l_candidate(
    h: int,
    w: int,
    div_r: int,
    div_c: int,
) -> dict[str, object] | None:
    corner_r_choices = [r for r in range(1, h - 1) if r != div_r]
    corner_c_choices = [c for c in range(1, w - 1) if c != div_c]
    if not corner_r_choices or not corner_c_choices:
        return None

    corner = (int(random.choice(corner_r_choices)), int(random.choice(corner_c_choices)))
    horiz_side = int(random.choice((-1, 1)))
    vert_side = int(random.choice((-1, 1)))

    max_dh = corner[1] - 1 if horiz_side < 0 else (w - 2) - corner[1]
    max_dv = corner[0] - 1 if vert_side < 0 else (h - 2) - corner[0]
    if max_dh < MIN_RAY_LEN or max_dv < MIN_RAY_LEN:
        return None

    dh = random.randint(MIN_RAY_LEN, max_dh)
    dv = random.randint(MIN_RAY_LEN, max_dv)

    tower_h_center = (int(corner[0]), int(corner[1] + horiz_side * dh))
    tower_v_center = (int(corner[0] + vert_side * dv), int(corner[1]))
    tower_h_missing = "right" if horiz_side < 0 else "left"
    tower_v_missing = "down" if vert_side < 0 else "up"

    tower_h_cells = _tower_cells(tower_h_center, tower_h_missing)
    tower_v_cells = _tower_cells(tower_v_center, tower_v_missing)
    if not _valid_tower_cells(tower_h_cells | tower_v_cells, h, w, div_r, div_c):
        return None

    path_cells = set(_ray_cells(tower_h_center, corner) + _ray_cells(tower_v_center, corner))
    if not path_cells:
        return None

    return {
        "kind": "L",
        "pivot": corner,
        "towers": [
            {"center": tower_h_center, "missing": tower_h_missing, "cells": tower_h_cells},
            {"center": tower_v_center, "missing": tower_v_missing, "cells": tower_v_cells},
        ],
        "path": path_cells,
    }


def _straight_candidate(
    h: int,
    w: int,
    div_r: int,
    div_c: int,
) -> dict[str, object] | None:
    orientation = random.choice(("h", "v"))
    row_choices = [r for r in range(1, h - 1) if r != div_r]
    col_choices = [c for c in range(1, w - 1) if c != div_c]
    if not row_choices or not col_choices:
        return None

    midpoint = (int(random.choice(row_choices)), int(random.choice(col_choices)))
    if orientation == "h":
        max_d = min(midpoint[1] - 1, (w - 2) - midpoint[1])
        if max_d < MIN_RAY_LEN:
            return None
        d = random.randint(MIN_RAY_LEN, max_d)
        left_center = (int(midpoint[0]), int(midpoint[1] - d))
        right_center = (int(midpoint[0]), int(midpoint[1] + d))
        left_missing = "right"
        right_missing = "left"
        left_cells = _tower_cells(left_center, left_missing)
        right_cells = _tower_cells(right_center, right_missing)
        if not _valid_tower_cells(left_cells | right_cells, h, w, div_r, div_c):
            return None
        path_cells = set(_ray_cells(left_center, right_center)[:-1])
        if not path_cells:
            return None
        towers = [
            {"center": left_center, "missing": left_missing, "cells": left_cells},
            {"center": right_center, "missing": right_missing, "cells": right_cells},
        ]
    else:
        max_d = min(midpoint[0] - 1, (h - 2) - midpoint[0])
        if max_d < MIN_RAY_LEN:
            return None
        d = random.randint(MIN_RAY_LEN, max_d)
        top_center = (int(midpoint[0] - d), int(midpoint[1]))
        bottom_center = (int(midpoint[0] + d), int(midpoint[1]))
        top_missing = "down"
        bottom_missing = "up"
        top_cells = _tower_cells(top_center, top_missing)
        bottom_cells = _tower_cells(bottom_center, bottom_missing)
        if not _valid_tower_cells(top_cells | bottom_cells, h, w, div_r, div_c):
            return None
        path_cells = set(_ray_cells(top_center, bottom_center)[:-1])
        if not path_cells:
            return None
        towers = [
            {"center": top_center, "missing": top_missing, "cells": top_cells},
            {"center": bottom_center, "missing": bottom_missing, "cells": bottom_cells},
        ]

    return {"kind": "straight", "pivot": midpoint, "towers": towers, "path": path_cells}


def _sample_pair(
    kind: str,
    h: int,
    w: int,
    div_r: int,
    div_c: int,
    forbidden_tower: set[tuple[int, int]],
    occupied_tower: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(PAIR_ATTEMPTS):
        if kind == "L":
            cand = _l_candidate(h, w, div_r, div_c)
        else:
            cand = _straight_candidate(h, w, div_r, div_c)
        if cand is None:
            continue

        tower_cells = set().union(*(tower["cells"] for tower in cand["towers"]))
        path_cells = set(cand["path"])
        if tower_cells & forbidden_tower:
            continue
        if tower_cells & occupied_path:
            continue
        if path_cells & occupied_tower:
            continue
        if path_cells & tower_cells:
            continue
        if path_cells & occupied_path:
            continue
        return cand
    return None


def _gen_attempt() -> np.ndarray | None:
    h = _odd_rand(*GRID_HEIGHT_RANGE)
    w = _odd_rand(*GRID_WIDTH_RANGE)
    div_r = h // 2
    div_c = w // 2

    n_pairs = int(random.choice(PAIR_COUNT_CHOICES))
    pair_kinds = ["L", "straight"]
    if n_pairs == 3:
        pair_kinds.append(random.choice(("L", "straight")))
    random.shuffle(pair_kinds)

    quad_order = tuple(random.sample(BACKGROUND_PALETTE, 4))
    grid = _blank_grid(h, w, quad_order)
    pair_colors = random.sample(TOWER_PALETTE, n_pairs)

    forbidden_tower: set[tuple[int, int]] = set()
    occupied_tower: set[tuple[int, int]] = set()
    occupied_path: set[tuple[int, int]] = set()
    pivot_colors: list[int] = []

    for pair_color, kind in zip(pair_colors, pair_kinds):
        pair = _sample_pair(kind, h, w, div_r, div_c, forbidden_tower, occupied_tower, occupied_path)
        if pair is None:
            return None

        for tower in pair["towers"]:
            for r, c in tower["cells"]:
                grid[int(r), int(c)] = int(pair_color)

        tower_cells = set().union(*(tower["cells"] for tower in pair["towers"]))
        occupied_tower.update(tower_cells)
        forbidden_tower.update(_expand8(tower_cells, h, w))
        occupied_path.update(pair["path"])

        pivot = pair["pivot"]
        pivot_colors.append(int(grid[int(pivot[0]), int(pivot[1])]))

    if len(set(pivot_colors)) < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if inp.ndim != 2 or out.ndim != 2:
                continue
            if max(inp.shape) > 30 or max(out.shape) > 30:
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
