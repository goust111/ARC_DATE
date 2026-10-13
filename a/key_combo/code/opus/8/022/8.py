from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int, connectivity: int = 4) -> list[list[Coord]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[Coord]] = []
    deltas = list(_neighbors4(0, 0)) if connectivity == 4 else list(_neighbors8(0, 0))
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_template(grid: np.ndarray) -> dict[str, object]:
    candidates = []
    for comp in _components_of_color(grid, GREEN, connectivity=4):
        if len(comp) != 5:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        sprite = grid[r0 : r1 + 1, c0 : c1 + 1]
        center = (1, 1)
        plus_mask = np.zeros((3, 3), dtype=bool)
        plus_mask[1, 1] = True
        plus_mask[0, 1] = True
        plus_mask[2, 1] = True
        plus_mask[1, 0] = True
        plus_mask[1, 2] = True
        if int(sprite[center]) != GREEN:
            continue
        if np.all(sprite[plus_mask] == GREEN) and np.all(sprite[~plus_mask] == BLACK):
            candidates.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "horizontal": True,
                    "vertical": True,
                    "coords": comp,
                }
            )
    assert len(candidates) == 1
    return candidates[0]


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for comp in _components_of_color(grid, GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        allowed = (region == GREY) | (region == RED)
        if not np.all(allowed):
            continue
        red_rel = np.argwhere(region == RED)
        if red_rel.size == 0:
            continue
        if np.any(red_rel[:, 0] == 0) or np.any(red_rel[:, 0] == region.shape[0] - 1):
            continue
        if np.any(red_rel[:, 1] == 0) or np.any(red_rel[:, 1] == region.shape[1] - 1):
            continue
        if not np.all(region[0, :] == GREY) or not np.all(region[-1, :] == GREY):
            continue
        if not np.all(region[:, 0] == GREY) or not np.all(region[:, -1] == GREY):
            continue
        if np.count_nonzero(region == GREY) != region.size - len(red_rel):
            continue
        markers = [(r0 + int(rr), c0 + int(cc)) for rr, cc in red_rel]
        rects.append({"bbox": (r0, c0, r1, c1), "markers": markers})
    rects.sort(key=lambda item: item["bbox"])
    assert 1 <= len(rects) <= 2
    return rects


def _find_start(grid: np.ndarray) -> Coord:
    coords = np.argwhere(grid == BLUE)
    assert coords.shape[0] == 1
    return int(coords[0, 0]), int(coords[0, 1])


def _make_stamp_mask(shape: tuple[int, int], rects: list[dict[str, object]], horizontal: bool, vertical: bool) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        for mr, mc in rect["markers"]:
            mask[mr, mc] = True
            if horizontal:
                mask[mr, c0 : c1 + 1] = True
            if vertical:
                mask[r0 : r1 + 1, mc] = True
    return mask


def _simulate_ray(shape: tuple[int, int], start: Coord) -> list[Coord]:
    H, W = shape
    r, c = start
    dr, dc = -1, 1
    path = [(r, c)]
    seen_states = {(r, c, dr, dc)}
    while True:
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= H:
            dr = -dr
            nr = r + dr
        if nc < 0 or nc >= W:
            dc = -dc
            nc = c + dc
        if (nr, nc) == start:
            break
        state = (nr, nc, dr, dc)
        if state in seen_states:
            break
        seen_states.add(state)
        r, c = nr, nc
        path.append((r, c))
    return path


def _ray_background_and_cross_output(grid: np.ndarray, stamp_mask: np.ndarray, ray_path: list[Coord], start: Coord) -> np.ndarray:
    out = grid.copy()
    out[stamp_mask] = GREEN
    for r, c in ray_path:
        if (r, c) == start:
            out[r, c] = BLUE
        elif stamp_mask[r, c]:
            out[r, c] = YELLOW
        elif int(grid[r, c]) == BLACK:
            out[r, c] = BLUE
    return out


def _spec_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Coord], list[dict[str, object]], dict[str, object], Coord]:
    rects = _find_rectangles(grid)
    template = _find_template(grid)
    start = _find_start(grid)
    stamp_mask = _make_stamp_mask(grid.shape, rects, bool(template["horizontal"]), bool(template["vertical"]))
    ray_path = _simulate_ray(grid.shape, start)
    out = _ray_background_and_cross_output(grid, stamp_mask, ray_path, start)
    return out, stamp_mask, ray_path, rects, template, start


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rectangle-wide crosses from the marked centers and color the intersections with a bouncing diagonal ray.

    Input structure:
    - The background is black, with one or two solid grey rectangles placed away from the edge.
    - Each rectangle contains a red marker pixel strictly inside its border.
    - A separate 3x3 green plus sprite sits near one corner, and one isolated blue pixel sits elsewhere on the background.

    Rule:
    - Read the external green plus as a crosshair template with both horizontal and vertical arms enabled.
    - For every red marker, recolor the marker and the full row and column through it inside its own grey rectangle to green.
    - From the blue pixel, trace a diagonal ray moving up-right; reverse horizontal direction at left/right walls and vertical direction at top/bottom walls, and stop when the ray returns to its start.
    - The ray paints only black background cells blue; grey rectangle cells stay grey unless they are on a stamped green cross, in which case cross-ray overlaps become yellow.
    - The external template sprite stays unchanged.

    Key insight:
    - The rectangles and the ray are generated independently, so the yellow output cells can only be found by expanding the full rectangle crosses and the full reflected ray, then intersecting those two geometric sets.
    """
    grid = np.asarray(grid, dtype=int)
    out, _stamp_mask, _ray_path, _rects, _template, _start = _spec_output(grid)
    return out


def _plus_sprite() -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    sprite[1, 1] = GREEN
    sprite[0, 1] = GREEN
    sprite[2, 1] = GREEN
    sprite[1, 0] = GREEN
    sprite[1, 2] = GREEN
    return sprite


def _rect_mask(shape: tuple[int, int], rects: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _expanded_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _place_rectangles(H: int, W: int, sprite_bbox: BBox, count: int) -> list[dict[str, object]] | None:
    occupied = np.zeros((H, W), dtype=bool)
    sr0, sc0, sr1, sc1 = sprite_bbox
    occupied[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    rects: list[dict[str, object]] = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            rh = random.randint(6, 8)
            rw = random.randint(6, 8)
            r0 = random.randint(2, H - rh - 2)
            c0 = random.randint(2, W - rw - 2)
            if not _can_place(occupied, r0, c0, rh, rw, pad=2):
                continue
            marker_r = random.randint(r0 + 1, r0 + rh - 2)
            marker_c = random.randint(c0 + 1, c0 + rw - 2)
            rect = {"bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1), "markers": [(marker_r, marker_c)]}
            rects.append(rect)
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            placed = True
            break
        if not placed:
            return None
    rects.sort(key=lambda item: item["bbox"])
    return rects


def _build_input_grid(H: int, W: int, sprite_bbox: BBox, rects: list[dict[str, object]], start: Coord) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    sr0, sc0, _sr1, _sc1 = sprite_bbox
    grid[sr0 : sr0 + 3, sc0 : sc0 + 3] = _plus_sprite()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        for mr, mc in rect["markers"]:
            grid[mr, mc] = RED
    grid[start] = BLUE
    return grid


def _corner_sprite_bbox(H: int, W: int) -> BBox:
    if random.random() < 0.5:
        r0 = random.randint(0, 2)
        c0 = random.randint(0, 2)
    else:
        r0 = H - 3 - random.randint(0, 2)
        c0 = W - 3 - random.randint(0, 2)
    return r0, c0, r0 + 2, c0 + 2


def _choose_blue_start(H: int, W: int, rects: list[dict[str, object]], sprite_bbox: BBox) -> Coord | None:
    sprite_mask = np.zeros((H, W), dtype=bool)
    sr0, sc0, sr1, sc1 = sprite_bbox
    sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    near_sprite = _expanded_mask(sprite_mask)
    stamp_mask = _make_stamp_mask((H, W), rects, horizontal=True, vertical=True)
    rect_mask = _rect_mask((H, W), rects)
    require_gray_hit = random.random() < 0.7

    candidates: list[tuple[int, Coord]] = []
    for r in range(H):
        for c in range(W):
            if rect_mask[r, c] or near_sprite[r, c] or sprite_mask[r, c]:
                continue
            path = _simulate_ray((H, W), (r, c))
            intersections = sum(1 for pr, pc in path if stamp_mask[pr, pc])
            if intersections == 0:
                continue
            gray_hits = sum(1 for pr, pc in path if rect_mask[pr, pc] and not stamp_mask[pr, pc])
            sprite_hits = sum(1 for pr, pc in path if sprite_mask[pr, pc])
            bg_hits = sum(1 for pr, pc in path if not rect_mask[pr, pc] and not sprite_mask[pr, pc])
            if require_gray_hit and gray_hits == 0:
                continue
            if sprite_hits != 0:
                continue
            if bg_hits < 4:
                continue
            if intersections > 8:
                continue
            score = abs(intersections - 2) + (0 if gray_hits > 0 else 3) + abs(len(path) - (H + W) // 2)
            candidates.append((score, (r, c)))
    if not candidates:
        return None
    best_score = min(score for score, _coord in candidates)
    best = [coord for score, coord in candidates if score <= best_score + 2]
    return random.choice(best)


def _validate_instance(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30
    out, stamp_mask, ray_path, rects, template, start = _spec_output(inp)
    assert inp.shape == out.shape
    assert len(rects) in (1, 2)
    assert bool(template["horizontal"]) and bool(template["vertical"])

    sprite_bbox = template["bbox"]
    r0, c0, r1, c1 = sprite_bbox
    assert np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1])

    rect_mask = _rect_mask(inp.shape, rects)
    intersections = [(r, c) for r, c in ray_path if stamp_mask[r, c]]
    assert intersections

    for rect in rects:
        rr0, cc0, rr1, cc1 = rect["bbox"]
        for mr, mc in rect["markers"]:
            assert rr0 < mr < rr1 and cc0 < mc < cc1
            assert np.all(out[mr, cc0 : cc1 + 1][out[mr, cc0 : cc1 + 1] != YELLOW] == GREEN)
            assert np.all(out[rr0 : rr1 + 1, mc][out[rr0 : rr1 + 1, mc] != YELLOW] == GREEN)

    for r, c in ray_path:
        if (r, c) == start:
            assert int(out[r, c]) == BLUE
        elif stamp_mask[r, c]:
            assert int(out[r, c]) == YELLOW
        elif int(inp[r, c]) == BLACK:
            assert int(out[r, c]) == BLUE
        elif rect_mask[r, c]:
            assert int(out[r, c]) == int(inp[r, c])
        else:
            assert int(out[r, c]) == int(inp[r, c])

    untouched_bg = (inp == BLACK)
    for r, c in ray_path:
        untouched_bg[r, c] = False
    untouched_bg[stamp_mask] = False
    assert np.all(out[untouched_bg] == BLACK)


def generate_input() -> np.ndarray:
    for _ in range(500):
        H = random.randint(15, 22)
        W = random.randint(15, 22)
        sprite_bbox = _corner_sprite_bbox(H, W)
        rect_count = 1 if random.random() < 0.45 else 2
        rects = _place_rectangles(H, W, sprite_bbox, rect_count)
        if rects is None:
            continue
        start = _choose_blue_start(H, W, rects, sprite_bbox)
        if start is None:
            continue
        grid = _build_input_grid(H, W, sprite_bbox, rects, start)
        _validate_instance(grid)
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
