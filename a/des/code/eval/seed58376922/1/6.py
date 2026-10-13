from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PRISM = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

PRISM_SPECS = {
    "UL": {
        "cells": ((0, 0), (0, 1), (1, 0)),
        "corner": (0, 0),
        "contact": {"R": (1, 0), "D": (0, 1)},
        "redirect": {"R": "U", "D": "L"},
    },
    "UR": {
        "cells": ((0, 0), (0, 1), (1, 1)),
        "corner": (0, 1),
        "contact": {"L": (1, 1), "D": (0, 0)},
        "redirect": {"L": "U", "D": "R"},
    },
    "LL": {
        "cells": ((0, 0), (1, 0), (1, 1)),
        "corner": (1, 0),
        "contact": {"R": (0, 0), "U": (1, 1)},
        "redirect": {"R": "D", "U": "L"},
    },
    "LR": {
        "cells": ((0, 1), (1, 0), (1, 1)),
        "corner": (1, 1),
        "contact": {"L": (0, 1), "U": (1, 0)},
        "redirect": {"L": "D", "U": "R"},
    },
}

TURN_TO_ORIENTATION = {
    ("R", "U"): "UL",
    ("R", "D"): "LL",
    ("L", "U"): "UR",
    ("L", "D"): "LR",
    ("U", "L"): "LL",
    ("U", "R"): "LR",
    ("D", "L"): "UL",
    ("D", "R"): "UR",
}

CORNER_FROM_REL = {
    (0, 0): "UL",
    (0, 1): "UR",
    (1, 0): "LL",
    (1, 1): "LR",
}

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# --------------- solver ---------------


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _scan_corner_legend(
    g: np.ndarray, bg: int, top: bool, left: bool
) -> tuple[dict[int, int], set[tuple[int, int]]] | None:
    h, w = g.shape
    if h < 3 or w < 3:
        return None

    r0 = 0 if top else h - 2
    r1 = r0 + 1
    barrier = 2 if top else h - 3

    if left:
        cols: list[int] = []
        c = 0
        while c < w and int(g[r0, c]) != bg and int(g[r1, c]) != bg:
            if int(g[r0, c]) == PRISM or int(g[r1, c]) == PRISM:
                return None
            cols.append(c)
            c += 1
        if not cols:
            return None
        if c < w and (int(g[r0, c]) != bg or int(g[r1, c]) != bg):
            return None
    else:
        cols_rev: list[int] = []
        c = w - 1
        while c >= 0 and int(g[r0, c]) != bg and int(g[r1, c]) != bg:
            if int(g[r0, c]) == PRISM or int(g[r1, c]) == PRISM:
                return None
            cols_rev.append(c)
            c -= 1
        if not cols_rev:
            return None
        if c >= 0 and (int(g[r0, c]) != bg or int(g[r1, c]) != bg):
            return None
        cols = list(reversed(cols_rev))

    if len(cols) < 2:
        return None
    if 0 <= barrier < h and np.any(g[barrier, cols] != bg):
        return None

    in_colors = [int(g[r0, c]) for c in cols]
    out_colors = [int(g[r1, c]) for c in cols]
    if len(set(in_colors)) != len(in_colors) or len(set(out_colors)) != len(out_colors):
        return None
    if any(v in (bg, PRISM) for v in in_colors + out_colors):
        return None

    coords = {(r0, c) for c in cols} | {(r1, c) for c in cols}
    return dict(zip(in_colors, out_colors)), coords


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    candidates = []
    for top, left in ((True, True), (True, False), (False, True), (False, False)):
        candidate = _scan_corner_legend(grid, bg, top, left)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one 2xM corner legend"
    mapping, coords = candidates[0]
    assert set(mapping) == set(mapping.values()), "legend must define a closed color permutation"
    return mapping, coords


def _extract_prisms(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    prisms: list[dict[str, object]] = []
    by_cell: dict[tuple[int, int], int] = {}

    for comp in _components(grid == PRISM):
        assert len(comp) == 3, "each prism must contain exactly three grey cells"
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, c0 = min(rows), min(cols)
        r1, c1 = max(rows), max(cols)
        assert r1 - r0 == 1 and c1 - c0 == 1, "each prism must fit in a 2x2 box"

        comp_set = set(comp)
        corner_rel = None
        for r, c in comp:
            degree = 0
            for dr, dc in NEIGHBORS4:
                if (r + dr, c + dc) in comp_set:
                    degree += 1
            if degree == 2:
                assert corner_rel is None, "prism must have one corner cell"
                corner_rel = (r - r0, c - c0)
        assert corner_rel in CORNER_FROM_REL, "invalid prism orientation"

        orientation = CORNER_FROM_REL[corner_rel]
        prism = {
            "orientation": orientation,
            "bbox": (r0, c0),
            "corner": (r0 + corner_rel[0], c0 + corner_rel[1]),
            "cells": comp_set,
        }
        prism_idx = len(prisms)
        prisms.append(prism)
        for cell in comp:
            by_cell[cell] = prism_idx

    return prisms, by_cell


def _extract_rays(
    grid: np.ndarray,
    bg: int,
    legend_coords: set[tuple[int, int]],
) -> list[dict[str, object]]:
    usable = np.ones_like(grid, dtype=bool)
    for r, c in legend_coords:
        usable[r, c] = False

    rays: list[dict[str, object]] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, PRISM))
    for color in colors:
        for comp in _components((grid == color) & usable):
            rows = sorted(r for r, _c in comp)
            cols = sorted(c for _r, c in comp)
            if len(set(rows)) == 1:
                assert cols == list(range(cols[0], cols[-1] + 1)), "ray must be contiguous"
            else:
                assert len(set(cols)) == 1, "ray must be horizontal or vertical"
                assert rows == list(range(rows[0], rows[-1] + 1)), "ray must be contiguous"

            touched = []
            if rows[0] == 0:
                touched.append("top")
            if rows[-1] == grid.shape[0] - 1:
                touched.append("bottom")
            if cols[0] == 0:
                touched.append("left")
            if cols[-1] == grid.shape[1] - 1:
                touched.append("right")
            assert len(touched) == 1, "ray seed must touch exactly one edge"

            side = touched[0]
            if side == "top":
                direction = "D"
                tip = (rows[-1], cols[0])
            elif side == "bottom":
                direction = "U"
                tip = (rows[0], cols[0])
            elif side == "left":
                direction = "R"
                tip = (rows[0], cols[-1])
            else:
                direction = "L"
                tip = (rows[0], cols[0])

            rays.append({
                "color": color,
                "dir": direction,
                "tip": tip,
                "cells": set(comp),
            })

    assert rays, "expected at least one edge ray"
    rays.sort(key=lambda ray: (ray["tip"], ray["dir"], ray["color"]))
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """Trace edge rays through grey right-triangle prisms using the visible color legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = BACKGROUND
    mapping, legend_coords = _extract_legend(g, bg)
    prisms, prism_by_cell = _extract_prisms(g)
    rays = _extract_rays(g, bg, legend_coords)

    out = g.copy()
    max_steps = g.size * 8

    for ray in rays:
        color = int(ray["color"])
        direction = str(ray["dir"])
        assert color in mapping, "ray color must appear in the legend"
        r, c = ray["tip"]
        seen: set[tuple[int, int, str, int]] = set()

        for _ in range(max_steps):
            state = (r, c, direction, color)
            assert state not in seen, "ray propagation must not loop"
            seen.add(state)

            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            if (nr, nc) in prism_by_cell:
                prism = prisms[prism_by_cell[(nr, nc)]]
                br, bc = prism["bbox"]
                rel = (nr - br, nc - bc)
                spec = PRISM_SPECS[str(prism["orientation"])]
                assert direction in spec["redirect"], "ray hit prism from an unsupported side"
                assert rel == spec["contact"][direction], "ray hit the wrong prism arm"
                color = int(mapping[color])
                r, c = prism["corner"]
                direction = spec["redirect"][direction]
                continue

            if (nr, nc) in legend_coords:
                break
            if int(out[nr, nc]) != bg:
                break

            out[nr, nc] = color
            r, c = nr, nc

    return out


# --------------- generator ---------------

HEIGHT_RANGE = (12, 19)
WIDTH_RANGE = (12, 19)
LEGEND_SIZE_OPTIONS = (3, 4)
RAY_COUNT_OPTIONS = (1, 2, 3)
SEED_LEN_RANGE = (2, 5)
BOUNCE_WEIGHTS = [(0, 2), (1, 3), (2, 3), (3, 1)]
STRAIGHT_LEN_RANGE = (1, 5)
GEN_ATTEMPTS = 500


def _sample_weighted(values_and_weights: list[tuple[int, int]]) -> int:
    values = [v for v, _w in values_and_weights]
    weights = [w for _v, w in values_and_weights]
    return random.choices(values, weights=weights, k=1)[0]


def _derangement(colors: list[int]) -> dict[int, int] | None:
    for _ in range(40):
        shuffled = colors[:]
        random.shuffle(shuffled)
        if all(a != b for a, b in zip(colors, shuffled)):
            return dict(zip(colors, shuffled))
    return None


def _legend_layout(h: int, w: int, size: int, corner: str) -> list[tuple[int, int, int]]:
    if corner == "TL":
        return [(0, c, 0) for c in range(size)] + [(1, c, 1) for c in range(size)]
    if corner == "TR":
        start = w - size
        return [(0, start + c, 0) for c in range(size)] + [(1, start + c, 1) for c in range(size)]
    if corner == "BL":
        return [(h - 2, c, 0) for c in range(size)] + [(h - 1, c, 1) for c in range(size)]
    start = w - size
    return [(h - 2, start + c, 0) for c in range(size)] + [(h - 1, start + c, 1) for c in range(size)]


def _legend_cells(h: int, w: int, size: int, corner: str) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    layout = _legend_layout(h, w, size, corner)
    ordered = [(r, c) for r, c, row_id in layout if row_id == 0] + [(r, c) for r, c, row_id in layout if row_id == 1]
    return ordered, {(r, c) for r, c, _row_id in layout}


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], neighbors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        expanded.add((r, c))
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                expanded.add((nr, nc))
    return expanded


def _walk_cells(start: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    return [(r + dr * i, c + dc * i) for i in range(1, length + 1)]


def _run_to_boundary(start: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _line_seed_cells(side: str, pos: int, length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(r, pos) for r in range(length)]
    if side == "bottom":
        return [(h - 1 - r, pos) for r in range(length)]
    if side == "left":
        return [(pos, c) for c in range(length)]
    return [(pos, w - 1 - c) for c in range(length)]


def _seed_tip(side: str, seed_cells: list[tuple[int, int]]) -> tuple[tuple[int, int], str]:
    if side == "top":
        return seed_cells[-1], "D"
    if side == "bottom":
        return seed_cells[-1], "U"
    if side == "left":
        return seed_cells[-1], "R"
    return seed_cells[-1], "L"


def _prism_from_turn(
    pre_hit: tuple[int, int],
    incoming: str,
    outgoing: str,
) -> tuple[str, tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    orientation = TURN_TO_ORIENTATION[(incoming, outgoing)]
    spec = PRISM_SPECS[orientation]
    dr, dc = DIRS[incoming]
    cr, cc = spec["contact"][incoming]
    br = pre_hit[0] + dr - cr
    bc = pre_hit[1] + dc - cc
    cells = [(br + rr, bc + cc2) for rr, cc2 in spec["cells"]]
    corner = (br + spec["corner"][0], bc + spec["corner"][1])
    return orientation, (br, bc), cells, corner


def _free_run(cells: list[tuple[int, int]], shape: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) and (r, c) not in blocked for r, c in cells)


def _place_ray(
    shape: tuple[int, int],
    mapping: dict[int, int],
    blocked_actual: set[tuple[int, int]],
    blocked_reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    ray_color = random.choice(list(mapping))
    side = random.choice(["top", "bottom", "left", "right"])
    seed_len = random.randint(*SEED_LEN_RANGE)

    if side in ("top", "bottom"):
        if seed_len > h - 4:
            return None
        positions = list(range(1, w - 1))
    else:
        if seed_len > w - 4:
            return None
        positions = list(range(1, h - 1))
    random.shuffle(positions)

    seed_cells = None
    tip = None
    direction = None
    for pos in positions:
        candidate = _line_seed_cells(side, pos, seed_len, shape)
        if any(cell in blocked_reserved for cell in candidate):
            continue
        if any(cell in blocked_actual for cell in candidate):
            continue
        seed_cells = candidate
        tip, direction = _seed_tip(side, candidate)
        break
    if seed_cells is None or tip is None or direction is None:
        return None

    planned_actual = set(seed_cells)
    output_segments: list[tuple[list[tuple[int, int]], int]] = []
    prisms: list[tuple[str, tuple[int, int]]] = []
    current = tip
    current_color = ray_color
    bounce_count = _sample_weighted(BOUNCE_WEIGHTS)

    for _ in range(bounce_count):
        turn_dirs = [d for (src, d), _ori in TURN_TO_ORIENTATION.items() if src == direction]
        random.shuffle(turn_dirs)
        lengths = list(range(STRAIGHT_LEN_RANGE[0], STRAIGHT_LEN_RANGE[1] + 1))
        random.shuffle(lengths)
        placed = False

        for outgoing in turn_dirs:
            for seg_len in lengths:
                path_cells = _walk_cells(current, direction, seg_len)
                if not _free_run(path_cells, shape, blocked_reserved | planned_actual):
                    continue
                pre_hit = path_cells[-1]
                orientation, _bbox, prism_cells, corner = _prism_from_turn(pre_hit, direction, outgoing)
                if not _free_run(prism_cells, shape, blocked_reserved | planned_actual):
                    continue
                next_after_turn = _walk_cells(corner, outgoing, 1)
                if not next_after_turn or not _free_run(next_after_turn, shape, blocked_reserved | planned_actual):
                    continue

                output_segments.append((path_cells, current_color))
                planned_actual.update(path_cells)
                planned_actual.update(prism_cells)
                prisms.append((orientation, _bbox))
                current = corner
                direction = outgoing
                current_color = mapping[current_color]
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    final_cells = _run_to_boundary(current, direction, shape)
    if len(final_cells) < 2:
        return None
    if not _free_run(final_cells, shape, blocked_reserved | planned_actual):
        return None

    output_segments.append((final_cells, current_color))
    planned_actual.update(final_cells)

    return {
        "seed_color": ray_color,
        "seed_cells": seed_cells,
        "segments": output_segments,
        "prisms": prisms,
        "actual_cells": planned_actual,
    }


def _render_grid(
    shape: tuple[int, int],
    legend_map: dict[int, int],
    legend_corner: str,
    rays: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    size = len(legend_map)
    inp = np.full((h, w), BACKGROUND, dtype=int)
    out = np.full((h, w), BACKGROUND, dtype=int)

    in_colors = list(legend_map)
    out_colors = [legend_map[c] for c in in_colors]
    layout = _legend_layout(h, w, size, legend_corner)
    for idx, (r, c, row_id) in enumerate(layout):
        color = in_colors[idx] if row_id == 0 else out_colors[idx - size]
        inp[r, c] = color
        out[r, c] = color

    for ray in rays:
        color = int(ray["seed_color"])
        for r, c in ray["seed_cells"]:
            inp[r, c] = color
            out[r, c] = color
        for orientation, bbox in ray["prisms"]:
            br, bc = bbox
            for rr, cc in PRISM_SPECS[orientation]["cells"]:
                inp[br + rr, bc + cc] = PRISM
                out[br + rr, bc + cc] = PRISM
        for cells, seg_color in ray["segments"]:
            for r, c in cells:
                out[r, c] = seg_color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    legend_size = random.choice(LEGEND_SIZE_OPTIONS)
    if w < legend_size + 4:
        return None

    legend_colors = random.sample(COLOR_POOL, legend_size)
    legend_map = _derangement(legend_colors)
    if legend_map is None:
        return None

    legend_corner = random.choice(["TL", "TR", "BL", "BR"])
    _ordered, legend_set = _legend_cells(h, w, legend_size, legend_corner)
    blocked_actual = set(legend_set)
    blocked_reserved = _expand(legend_set, (h, w), NEIGHBORS8)

    rays: list[dict[str, object]] = []
    ray_count = random.choice(RAY_COUNT_OPTIONS)
    for _ in range(ray_count):
        ray = _place_ray((h, w), legend_map, blocked_actual, blocked_reserved)
        if ray is None:
            return None
        rays.append(ray)
        blocked_actual |= set(ray["actual_cells"])
        blocked_reserved |= _expand(set(ray["actual_cells"]), (h, w), NEIGHBORS8)

    total_prisms = sum(len(ray["prisms"]) for ray in rays)
    if total_prisms == 0:
        return None

    inp, expected = _render_grid((h, w), legend_map, legend_corner, rays)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-prism puzzle")
