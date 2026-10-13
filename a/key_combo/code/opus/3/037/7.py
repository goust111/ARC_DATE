from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


BG = Color.BLACK
GREY = Color.GREY
TRAIL = Color.ORANGE
RECT_COLORS = (Color.BLUE, Color.RED, Color.GREEN)

RECT_DIM_POOL = ((2, 3), (2, 4), (3, 3), (3, 4), (4, 4))
GRID_MIN = 18
GRID_MAX = 24
RECT_GAP = 2
GEN_ATTEMPTS = 400
SPRITE_ATTEMPTS = 200
SHAPE_ATTEMPTS = 300


def _touch_side(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str | None:
    h, w = shape
    r0, c0, r1, c1 = bbox
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return int(comp["area"]) == patch.size and np.all(patch == int(comp["color"]))


def _rect_record(
    bbox: tuple[int, int, int, int],
    color: int,
    side: str,
    shape: tuple[int, int],
) -> dict:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rect_h = bbox_h(bbox)
    rect_w = bbox_w(bbox)
    if side == "top":
        final_bbox = (h - rect_h, c0, h - 1, c1)
        strip_bbox = (0, c0, h - 1, c1)
        axis = "vertical"
    elif side == "bottom":
        final_bbox = (0, c0, rect_h - 1, c1)
        strip_bbox = (0, c0, h - 1, c1)
        axis = "vertical"
    elif side == "left":
        final_bbox = (r0, w - rect_w, r1, w - 1)
        strip_bbox = (r0, 0, r1, w - 1)
        axis = "horizontal"
    else:
        final_bbox = (r0, 0, r1, rect_w - 1)
        strip_bbox = (r0, 0, r1, w - 1)
        axis = "horizontal"
    return {
        "bbox": bbox,
        "color": int(color),
        "side": side,
        "axis": axis,
        "h": rect_h,
        "w": rect_w,
        "canonical": tuple(sorted((rect_h, rect_w))),
        "strip_bbox": strip_bbox,
        "final_bbox": final_bbox,
    }


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rects = []
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    for comp in components:
        color = int(comp["color"])
        if color == GREY:
            continue
        side = _touch_side(comp["bbox"], grid.shape)
        if side is None or not _is_solid_rectangle(grid, comp):
            continue
        rects.append(_rect_record(comp["bbox"], color, side, grid.shape))
    rects.sort(key=lambda rect: rect["bbox"])
    return rects


def _matched_rectangle(sprite_bbox: tuple[int, int, int, int], rects: list[dict]) -> dict | None:
    canon = tuple(sorted((bbox_h(sprite_bbox), bbox_w(sprite_bbox))))
    matches = [rect for rect in rects if rect["canonical"] == canon]
    return matches[0] if len(matches) == 1 else None


def _bbox_in_strip(bbox: tuple[int, int, int, int], rect: dict) -> bool:
    r0, c0, r1, c1 = bbox
    sr0, sc0, sr1, sc1 = rect["strip_bbox"]
    if rect["axis"] == "vertical":
        return sc0 <= c0 and c1 <= sc1
    return sr0 <= r0 and r1 <= sr1


def _classify_sprite_bbox(bbox: tuple[int, int, int, int], rects: list[dict]) -> tuple[str, dict | None]:
    matched = _matched_rectangle(bbox, rects)
    if matched is not None and _bbox_in_strip(bbox, matched):
        return "both", matched
    if matched is not None:
        return "shape_only", matched
    if any(_bbox_in_strip(bbox, rect) for rect in rects):
        return "zone_only", None
    return "neither", None


def _paint_background_only(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch[patch == bg] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each edge rectangle to the opposite side, leave an orange sweep strip, and recolor only the grey sprites whose bbox both matches that rectangle and lies inside its strip.

    Input structure:
    - The background is black.
    - Two or three distinct solid colored rectangles touch exactly one outer edge without occupying a corner.
    - Four to six grey 4-connected free sprites sit away from the edges; some have bbox sizes matching one rectangle up to rotation.

    Rule:
    - Identify every edge-touching solid rectangle and move it straight across the grid to the opposite edge, perpendicular to the edge it touched.
    - The rectangle's full swept strip becomes orange except where the final rectangle lands, and existing sprite pixels are kept on top of that strip.
    - For each grey sprite, find whether its bbox size matches exactly one rectangle size, allowing width and height to swap.
    - Recolor that sprite to the matched rectangle color only when its whole bbox is also contained in that matched rectangle's sweep strip; otherwise leave it grey.

    Key insight:
    - Size matching alone is insufficient: the grey sprite inherits a rectangle's color only if it is both geometrically paired with that rectangle and physically standing inside the path that rectangle sweeps out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    rects = _find_rectangles(g, bg)
    assert 2 <= len(rects) <= 3, "expected two or three edge rectangles"

    sprites = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[GREY])
    assert 4 <= len(sprites) <= 6, "expected four to six grey sprites"

    out = g.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for rect in rects:
        _paint_background_only(out, rect["strip_bbox"], bg, TRAIL)

    for rect in rects:
        r0, c0, r1, c1 = rect["final_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]

    for sprite in sprites:
        category, matched = _classify_sprite_bbox(sprite["bbox"], rects)
        if category != "both":
            continue
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = int(matched["color"])

    return out


def _orientations(canon: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = canon
    return [(a, b)] if a == b else [(a, b), (b, a)]


def _inside_orientations(canon: tuple[int, int], rect: dict, mode: str) -> list[tuple[int, int]]:
    feasible = []
    for dims in _orientations(canon):
        h, w = dims
        if mode == "vertical" and w <= rect["w"]:
            feasible.append(dims)
        if mode == "horizontal" and h <= rect["h"]:
            feasible.append(dims)
    return feasible


def _sample_line_positions(lengths: list[int], total: int) -> list[int] | None:
    order = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)
    for _ in range(SPRITE_ATTEMPTS):
        starts: dict[int, int] = {}
        intervals: list[tuple[int, int]] = []
        ok = True
        for idx in order:
            length = int(lengths[idx])
            candidates = []
            for start in range(1, total - length):
                end = start + length - 1
                if all(end < s - RECT_GAP or start > e + RECT_GAP for s, e in intervals):
                    candidates.append(start)
            if not candidates:
                ok = False
                break
            start = random.choice(candidates)
            starts[idx] = start
            intervals.append((start, start + length - 1))
        if ok:
            return [starts[idx] for idx in range(len(lengths))]
    return None


def _choose_grid_shape(mode: str, dims: list[tuple[int, int]]) -> tuple[int, int] | None:
    heights = [h for h, _w in dims]
    widths = [w for _h, w in dims]
    if mode == "vertical":
        min_w = sum(widths) + RECT_GAP * (len(widths) - 1) + 6
        low_w = max(GRID_MIN, min_w)
        if low_w > GRID_MAX:
            return None
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(low_w, GRID_MAX)
        return h, w
    min_h = sum(heights) + RECT_GAP * (len(heights) - 1) + 6
    low_h = max(GRID_MIN, min_h)
    if low_h > GRID_MAX:
        return None
    h = random.randint(low_h, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    return h, w


def _build_rectangles(
    mode: str,
    shape: tuple[int, int],
    dims: list[tuple[int, int]],
    colors: list[int],
) -> list[dict] | None:
    h, w = shape
    if mode == "vertical":
        starts = _sample_line_positions([rect_w for _rect_h, rect_w in dims], w)
        if starts is None:
            return None
        rects = []
        for dims_i, color, c0 in zip(dims, colors, starts):
            rect_h, rect_w = dims_i
            side = random.choice(("top", "bottom"))
            if side == "top":
                bbox = (0, c0, rect_h - 1, c0 + rect_w - 1)
            else:
                bbox = (h - rect_h, c0, h - 1, c0 + rect_w - 1)
            rects.append(_rect_record(bbox, color, side, shape))
        return rects

    starts = _sample_line_positions([rect_h for rect_h, _rect_w in dims], h)
    if starts is None:
        return None
    rects = []
    for dims_i, color, r0 in zip(dims, colors, starts):
        rect_h, rect_w = dims_i
        side = random.choice(("left", "right"))
        if side == "left":
            bbox = (r0, 0, r0 + rect_h - 1, rect_w - 1)
        else:
            bbox = (r0, w - rect_w, r0 + rect_h - 1, w - 1)
        rects.append(_rect_record(bbox, color, side, shape))
    return rects


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _neighbors_in_bbox(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    if random.random() < 0.5:
        axes = (0, 1)
    else:
        axes = (1, 0)
    for axis in axes:
        if axis == 0:
            step = 1 if rr >= r else -1
            while r != rr:
                r += step
                cells.append((r, c))
        else:
            step = 1 if cc >= c else -1
            while c != cc:
                c += step
                cells.append((r, c))
    return cells


def _random_sprite_shape(h: int, w: int) -> np.ndarray:
    max_cells = min(8, h * w - 1)
    assert max_cells >= 4, "sprite bbox must allow a non-rectangular 4-cell shape"
    for _ in range(SHAPE_ATTEMPTS):
        hub = (random.randrange(h), random.randrange(w))
        anchors = {
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        }
        cells: set[tuple[int, int]] = set()
        for anchor in anchors:
            cells.update(_manhattan_path(anchor, hub))
        if len(cells) > max_cells:
            continue

        target = random.randint(max(4, len(cells)), max_cells)
        frontier = {
            nbr
            for cell in cells
            for nbr in _neighbors_in_bbox(cell[0], cell[1], h, w)
            if nbr not in cells
        }
        while len(cells) < target and frontier:
            nxt = random.choice(tuple(frontier))
            cells.add(nxt)
            frontier = {
                nbr
                for cell in cells
                for nbr in _neighbors_in_bbox(cell[0], cell[1], h, w)
                if nbr not in cells
            }

        if len(cells) != target or len(cells) >= h * w:
            continue
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        return np.array(sorted(cells), dtype=int)
    raise RuntimeError("failed to sample a valid grey sprite shape")


def _bbox_outside_all_strips(
    bbox: tuple[int, int, int, int],
    rects: list[dict],
    mode: str,
) -> bool:
    r0, c0, r1, c1 = bbox
    if mode == "vertical":
        return all(c1 < rect["bbox"][1] or c0 > rect["bbox"][3] for rect in rects)
    return all(r1 < rect["bbox"][0] or r0 > rect["bbox"][2] for rect in rects)


def _candidate_origins(
    blocked: np.ndarray,
    rects: list[dict],
    mode: str,
    bbox_hw: tuple[int, int],
    *,
    inside_rect: dict | None,
) -> list[tuple[int, int]]:
    h, w = bbox_hw
    grid_h, grid_w = blocked.shape
    if inside_rect is not None:
        if mode == "vertical":
            row_range = range(grid_h - h + 1)
            col_range = range(inside_rect["bbox"][1], inside_rect["bbox"][3] - w + 2)
        else:
            row_range = range(inside_rect["bbox"][0], inside_rect["bbox"][2] - h + 2)
            col_range = range(grid_w - w + 1)
    else:
        row_range = range(grid_h - h + 1)
        col_range = range(grid_w - w + 1)

    positions = []
    for r0 in row_range:
        for c0 in col_range:
            if blocked[r0 : r0 + h, c0 : c0 + w].any():
                continue
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if inside_rect is None and not _bbox_outside_all_strips(bbox, rects, mode):
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _zone_host_options(rects: list[dict], remaining_canons: list[tuple[int, int]], mode: str) -> list[tuple[dict, tuple[int, int], list[tuple[int, int]]]]:
    options = []
    for rect in rects:
        for canon in remaining_canons:
            feasible = _inside_orientations(canon, rect, mode)
            if feasible:
                options.append((rect, canon, feasible))
    return options


def _build_sprite_jobs(rects: list[dict], mode: str, total_sprites: int) -> list[dict] | None:
    used_canons = {tuple(rect["canonical"]) for rect in rects}
    remaining_canons = [canon for canon in RECT_DIM_POOL if canon not in used_canons]
    if len(remaining_canons) < 1:
        return None

    zone_hosts = _zone_host_options(rects, remaining_canons, mode)
    if not zone_hosts:
        return None

    jobs = []
    both_rect = random.choice(rects)
    jobs.append(
        {
            "category": "both",
            "inside_rect": both_rect,
            "dims": random.choice(_inside_orientations(both_rect["canonical"], both_rect, mode)),
        }
    )

    shape_rect = random.choice(rects)
    jobs.append(
        {
            "category": "shape_only",
            "inside_rect": None,
            "dims": random.choice(_orientations(shape_rect["canonical"])),
        }
    )

    zone_rect, zone_canon, zone_orients = random.choice(zone_hosts)
    jobs.append(
        {
            "category": "zone_only",
            "inside_rect": zone_rect,
            "dims": random.choice(zone_orients),
        }
    )

    neither_canon = random.choice(remaining_canons)
    jobs.append(
        {
            "category": "neither",
            "inside_rect": None,
            "dims": random.choice(_orientations(neither_canon)),
        }
    )

    extras = total_sprites - len(jobs)
    extra_categories = random.choices(
        ["both", "shape_only", "zone_only", "neither"],
        weights=[3, 2, 2, 1],
        k=extras,
    )
    for category in extra_categories:
        if category == "both":
            rect = random.choice(rects)
            dims = random.choice(_inside_orientations(rect["canonical"], rect, mode))
            jobs.append({"category": category, "inside_rect": rect, "dims": dims})
        elif category == "shape_only":
            rect = random.choice(rects)
            dims = random.choice(_orientations(rect["canonical"]))
            jobs.append({"category": category, "inside_rect": None, "dims": dims})
        elif category == "zone_only":
            rect, canon, feasible = random.choice(zone_hosts)
            dims = random.choice(feasible)
            jobs.append({"category": category, "inside_rect": rect, "dims": dims})
        else:
            canon = random.choice(remaining_canons)
            dims = random.choice(_orientations(canon))
            jobs.append({"category": category, "inside_rect": None, "dims": dims})

    jobs.sort(key=lambda job: (job["inside_rect"] is None, -(job["dims"][0] * job["dims"][1])))
    return jobs


def _place_sprites(grid: np.ndarray, blocked: np.ndarray, rects: list[dict], mode: str, jobs: list[dict]) -> bool:
    for job in jobs:
        bbox_hw = tuple(job["dims"])
        origins = _candidate_origins(blocked, rects, mode, bbox_hw, inside_rect=job["inside_rect"])
        if not origins:
            return False
        shape = _random_sprite_shape(*bbox_hw)
        r0, c0 = origins[0]
        coords = shape + np.array([r0, c0], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = GREY
        _mark_blocked(blocked, (r0, c0, r0 + bbox_hw[0] - 1, c0 + bbox_hw[1] - 1))
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(grid, out):
        return False

    bg = get_bg(grid)
    rects = _find_rectangles(grid, bg)
    if len(rects) < 2:
        return False
    if len({rect["color"] for rect in rects}) != len(rects):
        return False
    if len({rect["canonical"] for rect in rects}) != len(rects):
        return False

    sprites = find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[GREY])
    counts = Counter()
    recolored = 0
    unchanged = 0
    for sprite in sprites:
        category, matched = _classify_sprite_bbox(sprite["bbox"], rects)
        counts[category] += 1
        coords = sprite["coords"]
        output_colors = {int(out[r, c]) for r, c in coords}
        if category == "both":
            if output_colors != {int(matched["color"])}:
                return False
            recolored += 1
        else:
            if output_colors != {GREY}:
                return False
            unchanged += 1

    return (
        counts["both"] >= 1
        and counts["shape_only"] >= 1
        and counts["zone_only"] >= 1
        and counts["neither"] >= 1
        and recolored >= 1
        and unchanged >= 1
    )


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("vertical", "horizontal"))
    rect_count = random.choice((2, 3))
    rect_canons = random.sample(list(RECT_DIM_POOL), rect_count)
    rect_dims = [random.choice(_orientations(canon)) for canon in rect_canons]
    shape = _choose_grid_shape(mode, rect_dims)
    if shape is None:
        return None

    colors = random.sample(list(RECT_COLORS), rect_count)
    rects = _build_rectangles(mode, shape, rect_dims, colors)
    if rects is None:
        return None

    sprite_count = random.randint(4, 6)
    jobs = _build_sprite_jobs(rects, mode, sprite_count)
    if jobs is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]

    blocked = np.zeros(shape, dtype=bool)
    for rect in rects:
        _mark_blocked(blocked, rect["bbox"])
        _mark_blocked(blocked, rect["final_bbox"])

    if not _place_sprites(grid, blocked, rects, mode, jobs):
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-strip recolor input")
