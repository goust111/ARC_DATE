from __future__ import annotations

import numpy as np


# Marker colors follow the visible ARC palette semantics from the prompt:
# red -> right, blue -> down, green -> left, yellow -> up.
MARKER_TO_DIR = {
    2: (0, 1),
    1: (1, 0),
    3: (0, -1),
    4: (-1, 0),
}

DIR_NAMES = ("right", "down", "left", "up")
DIR_TO_MARKER = {"right": 2, "down": 1, "left": 3, "up": 4}
DIR_TO_VEC = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator palette and layout bounds.
BG_CHOICES = (0, 5, 6, 7, 8, 9)
SEED_COLOR_POOL = (0, 5, 6, 7, 8, 9)
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
OBJECT_COUNT_RANGE = (2, 6)
SHAPE_H_RANGE = (1, 4)
SHAPE_W_RANGE = (1, 5)
MAX_COPIES = 4
PLACEMENT_TRIES = 80
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    comps = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1 or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            labels[r, c] = next_id
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if labels[nr, nc] != -1 or int(grid[nr, nc]) != color:
                        continue
                    labels[nr, nc] = next_id
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
            next_id += 1
    return comps, labels


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps, labels = _components(g, bg)

    comp_by_id = {comp["id"]: comp for comp in comps}
    marker_to_seed = {}
    seed_to_markers: dict[int, list[int]] = {}

    for comp in comps:
        if int(comp["area"]) != 1 or int(comp["color"]) not in MARKER_TO_DIR:
            continue
        r, c = map(int, comp["coords"][0])
        adjacent = set()
        for nr, nc in _neighbors4(r, c, h, w):
            lab = int(labels[nr, nc])
            if lab >= 0 and lab != int(comp["id"]):
                adjacent.add(lab)
        if len(adjacent) != 1:
            continue
        seed_id = next(iter(adjacent))
        marker_to_seed[int(comp["id"])] = int(seed_id)
        seed_to_markers.setdefault(int(seed_id), []).append(int(comp["id"]))

    out = g.copy()
    for marker_id in marker_to_seed:
        r, c = map(int, comp_by_id[marker_id]["coords"][0])
        out[r, c] = bg

    growth_order = []
    for seed_id, marker_ids in seed_to_markers.items():
        if len(marker_ids) != 1:
            continue
        seed = comp_by_id[int(seed_id)]
        r0, c0, _r1, _c1 = seed["bbox"]
        growth_order.append((int(r0), int(c0), int(seed_id), int(marker_ids[0])))
    growth_order.sort()

    for _r0, _c0, seed_id, marker_id in growth_order:
        seed = comp_by_id[seed_id]
        marker = comp_by_id[marker_id]
        color = int(seed["color"])
        dr, dc = MARKER_TO_DIR[int(marker["color"])]
        r0, c0, r1, c1 = seed["bbox"]
        step = (c1 - c0 + 1) if dc != 0 else (r1 - r0 + 1)
        shift = np.array([dr * step, dc * step], dtype=int)
        translated = seed["coords"] + shift

        while True:
            if not bool(
                np.all(
                    (0 <= translated[:, 0])
                    & (translated[:, 0] < h)
                    & (0 <= translated[:, 1])
                    & (translated[:, 1] < w)
                )
            ):
                break
            if not bool(np.all(out[translated[:, 0], translated[:, 1]] == bg)):
                break
            out[translated[:, 0], translated[:, 1]] = color
            translated = translated + shift

    return out


def _normalize_shape(coords):
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(SHAPE_H_RANGE[0], SHAPE_H_RANGE[1] + 1))
    w = int(rng.integers(SHAPE_W_RANGE[0], SHAPE_W_RANGE[1] + 1))
    total = h * w
    if total == 1:
        return np.array([[0, 0]], dtype=int)
    min_area = 1 if rng.random() < 0.25 else 2
    target_area = int(rng.integers(min_area, total + 1))
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    frontier = set()
    while len(cells) < target_area:
        frontier.clear()
        for r, c in cells:
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            break
        choice = list(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
    shape = _normalize_shape(cells)
    if rng.random() < 0.5:
        shape[:, 1] = int(shape[:, 1].max()) - shape[:, 1]
    if rng.random() < 0.5:
        shape[:, 0] = int(shape[:, 0].max()) - shape[:, 0]
    if rng.random() < 0.3:
        shape = shape[:, ::-1]
    return _normalize_shape({tuple(x) for x in shape.tolist()})


def _marker_candidates(shape: np.ndarray, direction: str):
    shape_set = {tuple(map(int, pt)) for pt in shape.tolist()}
    dr, dc = DIR_TO_VEC[direction]
    candidates = []
    seen = set()
    for r, c in shape_set:
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (rr, cc) in shape_set or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            on_growth_side = (rr - dr, cc - dc) in shape_set
            candidates.append(((rr, cc), on_growth_side))
    return candidates


def _max_copies(r0: int, c0: int, bbox_h: int, bbox_w: int, direction: str, h: int, w: int) -> int:
    if direction == "right":
        return int((w - 1 - (c0 + bbox_w - 1)) // bbox_w)
    if direction == "left":
        return int(c0 // bbox_w)
    if direction == "down":
        return int((h - 1 - (r0 + bbox_h - 1)) // bbox_h)
    return int(r0 // bbox_h)


def _is_input_clear(cells, occ: np.ndarray) -> bool:
    h, w = occ.shape
    cell_set = set(cells)
    for r, c in cell_set:
        if occ[r, c]:
            return False
        for nr, nc in _neighbors4(r, c, h, w):
            if occ[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _output_cells(seed_cells, direction: str, copies: int, bbox_h: int, bbox_w: int):
    dr, dc = DIR_TO_VEC[direction]
    step = bbox_w if dc != 0 else bbox_h
    out = []
    for k in range(copies + 1):
        sr = dr * step * k
        sc = dc * step * k
        for r, c in seed_cells:
            out.append((r + sr, c + sc))
    return out


def _placement_options(
    shape: np.ndarray,
    direction: str,
    input_occ: np.ndarray,
    reserved_output: np.ndarray,
):
    h, w = input_occ.shape
    bbox_h = int(shape[:, 0].max()) + 1
    bbox_w = int(shape[:, 1].max()) + 1
    rel_seed = [tuple(map(int, pt)) for pt in shape.tolist()]
    options = []
    for (mr, mc), on_growth_side in _marker_candidates(shape, direction):
        for r0 in range(h - bbox_h + 1):
            for c0 in range(w - bbox_w + 1):
                marker = (r0 + mr, c0 + mc)
                if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                    continue
                seed_cells = [(r0 + rr, c0 + cc) for rr, cc in rel_seed]
                input_cells = seed_cells + [marker]
                if not _is_input_clear(input_cells, input_occ):
                    continue
                copies = _max_copies(r0, c0, bbox_h, bbox_w, direction, h, w)
                if not (1 <= copies <= MAX_COPIES):
                    continue
                out_cells = _output_cells(seed_cells, direction, copies, bbox_h, bbox_w)
                if any(reserved_output[r, c] for r, c in out_cells):
                    continue
                options.append(
                    {
                        "seed_cells": seed_cells,
                        "marker": marker,
                        "copies": copies,
                        "bbox_h": bbox_h,
                        "bbox_w": bbox_w,
                        "on_growth_side": bool(on_growth_side),
                        "output_cells": out_cells,
                    }
                )
    return options


def _render_expected(grid: np.ndarray, records, bg: int) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        mr, mc = rec["marker"]
        out[mr, mc] = bg
    for rec in records:
        color = int(rec["color"])
        dr, dc = DIR_TO_VEC[rec["direction"]]
        step = int(rec["bbox_w"]) if dc != 0 else int(rec["bbox_h"])
        for k in range(1, int(rec["copies"]) + 1):
            sr = dr * step * k
            sc = dc * step * k
            for r, c in rec["seed_cells"]:
                out[r + sr, c + sc] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    bg = int(BG_CHOICES[int(rng.integers(0, len(BG_CHOICES)))])
    grid = np.full((h, w), bg, dtype=int)

    seed_colors = [c for c in SEED_COLOR_POOL if c != bg]
    rng.shuffle(seed_colors)
    max_objects = min(len(seed_colors), OBJECT_COUNT_RANGE[1] - 1)
    num_objects = int(rng.integers(OBJECT_COUNT_RANGE[0], max_objects + 1))

    direction_pool = list(DIR_NAMES) * ((num_objects + len(DIR_NAMES) - 1) // len(DIR_NAMES))
    rng.shuffle(direction_pool)
    input_occ = np.zeros((h, w), dtype=bool)
    reserved_output = np.zeros((h, w), dtype=bool)
    records = []

    for i in range(num_objects):
        direction = direction_pool[i]
        color = int(seed_colors[i])
        placed = False
        for _ in range(PLACEMENT_TRIES):
            shape = _random_shape(rng)
            options = _placement_options(shape, direction, input_occ, reserved_output)
            if not options:
                continue
            prefer_growth_side = bool(rng.integers(0, 2))
            filtered = [opt for opt in options if opt["on_growth_side"] == prefer_growth_side]
            if not filtered:
                filtered = options
            choice = filtered[int(rng.integers(0, len(filtered)))]

            for r, c in choice["seed_cells"]:
                grid[r, c] = color
                input_occ[r, c] = True
            mr, mc = choice["marker"]
            grid[mr, mc] = DIR_TO_MARKER[direction]
            input_occ[mr, mc] = True
            for r, c in choice["output_cells"]:
                reserved_output[r, c] = True

            records.append(
                {
                    "color": color,
                    "direction": direction,
                    "seed_cells": list(choice["seed_cells"]),
                    "marker": choice["marker"],
                    "copies": int(choice["copies"]),
                    "bbox_h": int(choice["bbox_h"]),
                    "bbox_w": int(choice["bbox_w"]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    if len({rec["direction"] for rec in records}) < min(2, len(records)):
        return None

    expected = _render_expected(grid, records, bg)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
