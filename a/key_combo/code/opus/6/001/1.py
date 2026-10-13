from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES = [("h", 2), ("h", 3), ("h", 4), ("v", 2), ("v", 3), ("v", 4)]
MAX_GENERATE_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHO:
        yield int(r + dr), int(c + dc)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    origin = coords.min(axis=0)
    norm = coords - origin
    height = int(norm[:, 0].max()) + 1
    width = int(norm[:, 1].max()) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return height, width, cells


def _expand4(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _rod_info(g: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    cells = {tuple(map(int, cell)) for cell in coords}
    blue_cells = [p for p in cells if int(g[p]) == BLUE]
    grey_cells = [p for p in cells if int(g[p]) == GREY]
    red_cells = [p for p in cells if int(g[p]) == RED]
    assert len(blue_cells) == 1 and len(grey_cells) == 1
    assert len(red_cells) + 2 == len(cells)

    hinge = blue_cells[0]
    endpoint = grey_cells[0]

    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    if len(rows) == 1:
        orient = "h"
        row = rows[0]
        assert len(cols) == len(cells)
        assert cols == list(range(cols[0], cols[-1] + 1))
        assert hinge[0] == row and endpoint[0] == row
        fixed_sign = int(np.sign(endpoint[1] - hinge[1]))
        assert fixed_sign != 0
        tail_step = (0, -fixed_sign)
        fixed_red = [(row, hinge[1] + fixed_sign * k) for k in range(1, 99) if (row, hinge[1] + fixed_sign * k) in cells]
    else:
        orient = "v"
        col = cols[0]
        assert len(cols) == 1
        assert rows == list(range(rows[0], rows[-1] + 1))
        assert hinge[1] == col and endpoint[1] == col
        fixed_sign = int(np.sign(endpoint[0] - hinge[0]))
        assert fixed_sign != 0
        tail_step = (-fixed_sign, 0)
        fixed_red = [(hinge[0] + fixed_sign * k, col) for k in range(1, 99) if (hinge[0] + fixed_sign * k, col) in cells]

    tail: list[tuple[int, int]] = []
    step = 1
    while True:
        p = (int(hinge[0] + tail_step[0] * step), int(hinge[1] + tail_step[1] * step))
        if p not in cells:
            break
        assert int(g[p]) == RED
        tail.append(p)
        step += 1
    assert 2 <= len(tail) <= 4
    fixed_red = [p for p in fixed_red if p != endpoint and p != hinge]
    for p in fixed_red:
        assert int(g[p]) == RED

    return {
        "coords": coords,
        "orient": orient,
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_red": fixed_red,
        "tail": tail,
        "tail_key": _shape_key(np.array(tail, dtype=int)),
    }


def _parse_scene(g: np.ndarray) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    rods = [_rod_info(g, comp) for comp in _component_coords(np.isin(g, [RED, BLUE, GREY]))]
    templates = _component_coords(g == GREEN)
    assert 2 <= len(rods) <= 3
    assert len(templates) == len(rods)
    return rods, templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked rod's tail toward the side encoded by the matching green template.

    Input structure:
    - The background is black.
    - Each non-green connected object made of red, one blue hinge marker, and one grey endpoint marker is a straight horizontal or vertical rod.
    - Separate green 4-connected template objects have normalized shapes that uniquely match the normalized tail segment of the rods.

    Rule:
    - For each rod, identify the blue hinge and the grey endpoint; the red cells on the opposite side of the hinge form the moving tail.
    - Match that tail's translation-normalized shape against the green templates without rotating or flipping.
    - Use the matched template's centroid relative to the hinge to choose the perpendicular fold direction: horizontal rods fold up or down, vertical rods fold left or right.
    - Erase the original tail and all green templates, then redraw the tail as a straight red perpendicular ray of the same length from the hinge; keep the grey endpoint, blue hinge, and fixed red segment unchanged.

    Key insight:
    - The green template does not mark where to draw; its normalized shape selects which template belongs to a rod, and that matched template's side of the hinge encodes the fold direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG

    rods, templates = _parse_scene(g)
    template_by_key: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[np.ndarray]] = {}
    for comp in templates:
        template_by_key.setdefault(_shape_key(comp), []).append(comp)

    clear: list[tuple[int, int]] = []
    paint: list[tuple[int, int]] = []

    for rod in rods:
        matches = template_by_key.get(rod["tail_key"], [])
        assert len(matches) == 1
        template = matches[0]
        hinge = rod["hinge"]

        if rod["orient"] == "h":
            mean_r = float(template[:, 0].mean())
            assert mean_r != hinge[0]
            fold_step = (-1, 0) if mean_r < hinge[0] else (1, 0)
        else:
            mean_c = float(template[:, 1].mean())
            assert mean_c != hinge[1]
            fold_step = (0, -1) if mean_c < hinge[1] else (0, 1)

        clear.extend(rod["tail"])
        clear.extend([tuple(map(int, cell)) for cell in template])
        for step in range(1, len(rod["tail"]) + 1):
            p = (int(hinge[0] + fold_step[0] * step), int(hinge[1] + fold_step[1] * step))
            assert _in_bounds(g.shape, *p)
            paint.append(p)

    clear_set = set(clear)
    assert len(paint) == len(set(paint))
    for p in paint:
        assert int(g[p]) == bg or p in clear_set

    out = g.copy()
    for p in clear:
        out[p] = bg
    for p in paint:
        out[p] = RED
    return out


def _template_side(orient: str, hinge: tuple[int, int], template: np.ndarray) -> int:
    if orient == "h":
        side = int(np.sign(float(template[:, 0].mean()) - hinge[0]))
    else:
        side = int(np.sign(float(template[:, 1].mean()) - hinge[1]))
    assert side in (-1, 1)
    return side


def _render_expected(grid: np.ndarray, records: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rec in records:
        for p in rec["tail"]:
            out[p] = BG
        for p in rec["template"]:
            out[p] = BG
    for rec in records:
        for p in rec["fold"]:
            out[p] = RED
    return out


def _sample_shapes(count: int) -> list[tuple[str, int]]:
    while True:
        chosen = random.sample(SHAPES, count)
        orients = {orient for orient, _length in chosen}
        if len(orients) == 2:
            return chosen


def _random_template_cells(
    shape: tuple[int, int],
    orient: str,
    hinge: tuple[int, int],
    side: int,
    tail_len: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    if orient == "h":
        max_dist = 5
        dist_choices = [d for d in range(2, max_dist + 1) if 0 <= hinge[0] + side * d < h]
        if not dist_choices:
            return None
        row = hinge[0] + side * random.choice(dist_choices)
        start_choices = [c0 for c0 in range(w - tail_len + 1)]
        if not start_choices:
            return None
        start = random.choice(start_choices)
        return [(int(row), int(start + k)) for k in range(tail_len)]

    max_dist = 5
    dist_choices = [d for d in range(2, max_dist + 1) if 0 <= hinge[1] + side * d < w]
    if not dist_choices:
        return None
    col = hinge[1] + side * random.choice(dist_choices)
    start_choices = [r0 for r0 in range(h - tail_len + 1)]
    if not start_choices:
        return None
    start = random.choice(start_choices)
    return [(int(start + k), int(col)) for k in range(tail_len)]


def _sample_record(
    shape: tuple[int, int],
    tail_shape: tuple[str, int],
    occupied_input_buffer: set[tuple[int, int]],
    occupied_envelope: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    orient, tail_len = tail_shape
    fixed_red_len = random.randint(1, min(4, 7 - tail_len))
    endpoint_sign = random.choice([-1, 1])
    fold_sign = random.choice([-1, 1])

    if orient == "h":
        row = random.randint(0, h - 1)
        min_c = fixed_red_len + 1
        max_c = w - 1 - tail_len
        if min_c > max_c:
            return None
        hinge = (int(row), int(random.randint(min_c, max_c)))
        endpoint = (row, hinge[1] + endpoint_sign * (fixed_red_len + 1))
        fixed_red = [(row, hinge[1] + endpoint_sign * k) for k in range(1, fixed_red_len + 1)]
        tail = [(row, hinge[1] - endpoint_sign * k) for k in range(1, tail_len + 1)]
        fold = [(row + fold_sign * k, hinge[1]) for k in range(1, tail_len + 1)]
    else:
        col = random.randint(0, w - 1)
        min_r = fixed_red_len + 1
        max_r = h - 1 - tail_len
        if min_r > max_r:
            return None
        hinge = (int(random.randint(min_r, max_r)), int(col))
        endpoint = (hinge[0] + endpoint_sign * (fixed_red_len + 1), col)
        fixed_red = [(hinge[0] + endpoint_sign * k, col) for k in range(1, fixed_red_len + 1)]
        tail = [(hinge[0] - endpoint_sign * k, col) for k in range(1, tail_len + 1)]
        fold = [(hinge[0], col + fold_sign * k) for k in range(1, tail_len + 1)]

    rod = [endpoint, hinge, *fixed_red, *tail]
    if len(rod) != len(set(rod)):
        return None
    if any(not _in_bounds(shape, *p) for p in rod + fold):
        return None

    template = _random_template_cells(shape, orient, hinge, fold_sign, tail_len)
    if template is None:
        return None
    if len(template) != len(set(template)):
        return None
    if any(not _in_bounds(shape, *p) for p in template):
        return None

    rod_set = set(rod)
    template_set = set(template)
    if template_set & _expand4(rod_set, shape):
        return None

    input_cells = rod_set | template_set
    envelope = input_cells | set(fold)
    if input_cells & occupied_input_buffer:
        return None
    if envelope & occupied_envelope:
        return None

    return {
        "orient": orient,
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_red": fixed_red,
        "tail": tail,
        "template": template,
        "fold": fold,
        "tail_key": _shape_key(np.array(tail, dtype=int)),
        "input_cells": input_cells,
        "envelope": envelope,
        "fold_sign": fold_sign,
    }


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    count = random.choices([2, 3], weights=[2, 5], k=1)[0]
    tail_shapes = _sample_shapes(count)
    random.shuffle(tail_shapes)

    records: list[dict[str, object]] = []
    occupied_input_buffer: set[tuple[int, int]] = set()
    occupied_envelope: set[tuple[int, int]] = set()

    for tail_shape in sorted(tail_shapes, key=lambda item: item[1], reverse=True):
        placed = None
        for _ in range(200):
            rec = _sample_record((h, w), tail_shape, occupied_input_buffer, occupied_envelope)
            if rec is None:
                continue
            placed = rec
            break
        if placed is None:
            return None
        records.append(placed)
        occupied_input_buffer |= _expand4(placed["input_cells"], (h, w))
        occupied_envelope |= set(placed["envelope"])

    grid = np.zeros((h, w), dtype=int)
    for rec in records:
        for p in rec["tail"] + rec["fixed_red"]:
            grid[p] = RED
        grid[rec["hinge"]] = BLUE
        grid[rec["endpoint"]] = GREY
        for p in rec["template"]:
            grid[p] = GREEN

    expected = _render_expected(grid, records)
    if not _semantic_checks(grid, expected):
        return None
    return grid, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    try:
        rods, templates = _parse_scene(inp)
    except Exception:
        return False

    if len({rod["tail_key"] for rod in rods}) != len(rods):
        return False

    template_items = [{"coords": comp, "key": _shape_key(comp)} for comp in templates]
    matched_centroids: list[tuple[tuple[int, int], tuple[float, float], bool]] = []
    for rod in rods:
        matches = [tpl for tpl in template_items if tpl["key"] == rod["tail_key"]]
        if len(matches) != 1:
            return False
        tpl = matches[0]["coords"]
        side = _template_side(rod["orient"], rod["hinge"], tpl)
        matched_centroids.append(
            (
                rod["hinge"],
                (float(tpl[:, 0].mean()), float(tpl[:, 1].mean())),
                side == 1,
            )
        )

    dual_side_context = False
    nearest_wrong = False
    fold_sides = set()
    for rod in rods:
        hinge = rod["hinge"]
        signed = []
        dists = []
        own_key = rod["tail_key"]
        for tpl in template_items:
            centroid = (float(tpl["coords"][:, 0].mean()), float(tpl["coords"][:, 1].mean()))
            if rod["orient"] == "h":
                side = int(np.sign(centroid[0] - hinge[0]))
            else:
                side = int(np.sign(centroid[1] - hinge[1]))
            if side != 0:
                signed.append(side)
            dist = (centroid[0] - hinge[0]) ** 2 + (centroid[1] - hinge[1]) ** 2
            dists.append((dist, tpl["key"] == own_key))
        if -1 in signed and 1 in signed:
            dual_side_context = True
        dists.sort(key=lambda item: item[0])
        if len(dists) >= 2 and dists[0][0] < dists[1][0] and not dists[0][1]:
            nearest_wrong = True
        own_template = next(tpl["coords"] for tpl in template_items if tpl["key"] == own_key)
        fold_sides.add(_template_side(rod["orient"], hinge, own_template))

    if not dual_side_context:
        return False
    if not nearest_wrong:
        return False
    if len(fold_sides) < 2:
        return False
    if int(np.sum(inp == GREEN)) == 0 or int(np.sum(expected == GREEN)) != 0:
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        scene = _build_scene()
        if scene is not None:
            grid, _expected = scene
            return grid
    raise RuntimeError("generate_input: failed after many attempts")
