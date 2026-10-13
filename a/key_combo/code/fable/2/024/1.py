from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

BG = 0
OBSTACLE = 5
BODY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS_H = ((0, 1), (0, -1))
DIRS_V = ((1, 0), (-1, 0))

MIN_OBJECTS = 2
MAX_OBJECTS = 4
BODY_H_RANGE = (2, 4)
BODY_W_RANGE = (2, 5)
TAIL_RANGE = (2, 4)
EXTRA_OBSTACLES_RANGE = (1, 3)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield int(r - 1), int(c)
    yield int(r + 1), int(c)
    yield int(r), int(c - 1)
    yield int(r), int(c + 1)


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        pts: list[Coord] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((int(rr), int(cc)))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        out.append(np.asarray(pts, dtype=int))
    return out


def _is_full(mask: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _detect_slider(mask: np.ndarray) -> dict | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_coords(coords)
    candidates: list[dict] = []

    def add_if_slider(body: BBox, direction: Direction, length: int, outside: np.ndarray) -> None:
        if length < 2 or outside.shape[0] != length or not _is_full(mask, body):
            return
        br0, bc0, br1, bc1 = body
        dr, dc = direction
        rows = sorted({int(r) for r in outside[:, 0]})
        cols = sorted({int(c) for c in outside[:, 1]})
        if dr == 0:
            if len(rows) != 1:
                return
            attach = int(rows[0])
            if not (br0 <= attach <= br1):
                return
            expected = list(range(bc1 + 1, c1 + 1)) if dc > 0 else list(range(c0, bc0))
            if cols != expected:
                return
        else:
            if len(cols) != 1:
                return
            attach = int(cols[0])
            if not (bc0 <= attach <= bc1):
                return
            expected = list(range(br1 + 1, r1 + 1)) if dr > 0 else list(range(r0, br0))
            if rows != expected:
                return
        candidates.append(
            {
                "body": body,
                "direction": direction,
                "tail_len": int(length),
                "attach": int(attach),
                "coords": coords,
            }
        )

    for body_c1 in range(c0, c1):
        body = (r0, c0, r1, int(body_c1))
        add_if_slider(body, (0, 1), c1 - body_c1, coords[coords[:, 1] > body_c1])
    for body_c0 in range(c0 + 1, c1 + 1):
        body = (r0, int(body_c0), r1, c1)
        add_if_slider(body, (0, -1), body_c0 - c0, coords[coords[:, 1] < body_c0])
    for body_r1 in range(r0, r1):
        body = (r0, c0, int(body_r1), c1)
        add_if_slider(body, (1, 0), r1 - body_r1, coords[coords[:, 0] > body_r1])
    for body_r0 in range(r0 + 1, r1 + 1):
        body = (int(body_r0), c0, r1, c1)
        add_if_slider(body, (-1, 0), body_r0 - r0, coords[coords[:, 0] < body_r0])

    assert len(candidates) <= 1, "body/tail decomposition must be unique"
    return candidates[0] if candidates else None


def _extract_sliders(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    sliders: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, OBSTACLE}):
        for coords in _components_of_color(grid, color):
            mask = np.zeros_like(grid, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            slider = _detect_slider(mask)
            assert slider is not None, "every non-grey component must be a tailed rectangle"
            slider["color"] = int(color)
            sliders.append(slider)
    sliders.sort(key=lambda s: (s["body"][0], s["body"][1], s["color"]))
    assert sliders, "expected at least one slider"
    return sliders


def _channel_obstacles(grid: np.ndarray, body: BBox, direction: Direction) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = body
    out: list[tuple[int, int, int]] = []
    if direction == (0, 1):
        for rr, cc in np.argwhere(grid[r0 : r1 + 1, c1 + 1 :] == OBSTACLE):
            out.append((int(cc + 1), int(r0 + rr), int(c1 + 1 + cc)))
    elif direction == (0, -1):
        for rr, cc in np.argwhere(grid[r0 : r1 + 1, :c0] == OBSTACLE):
            out.append((int(c0 - cc), int(r0 + rr), int(cc)))
    elif direction == (1, 0):
        for rr, cc in np.argwhere(grid[r1 + 1 :, c0 : c1 + 1] == OBSTACLE):
            out.append((int(rr + 1), int(r1 + 1 + rr), int(c0 + cc)))
    else:
        for rr, cc in np.argwhere(grid[:r0, c0 : c1 + 1] == OBSTACLE):
            out.append((int(r0 - rr), int(rr), int(c0 + cc)))
    out.sort()
    distances = [d for d, _r, _c in out]
    assert len(distances) == len(set(distances)), "obstacle order must be unambiguous"
    return out


def _tail_line_obstacles(
    grid: np.ndarray, body: BBox, direction: Direction, attach: int
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = body
    out: list[tuple[int, int, int]] = []
    if direction == (0, 1):
        for cc in np.where(grid[int(attach), c1 + 1 :] == OBSTACLE)[0]:
            out.append((int(cc + 1), int(attach), int(c1 + 1 + cc)))
    elif direction == (0, -1):
        for cc in np.where(grid[int(attach), :c0] == OBSTACLE)[0]:
            out.append((int(c0 - cc), int(attach), int(cc)))
    elif direction == (1, 0):
        for rr in np.where(grid[r1 + 1 :, int(attach)] == OBSTACLE)[0]:
            out.append((int(rr + 1), int(r1 + 1 + rr), int(attach)))
    else:
        for rr in np.where(grid[:r0, int(attach)] == OBSTACLE)[0]:
            out.append((int(r0 - rr), int(rr), int(attach)))
    out.sort()
    return out


def _apply_true_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    out = g.copy()
    sliders = _extract_sliders(g)
    for slider in sliders:
        coords = slider["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for slider in sliders:
        body = slider["body"]
        direction = slider["direction"]
        tail_len = int(slider["tail_len"])
        obstacles = _channel_obstacles(g, body, direction)
        assert len(obstacles) >= tail_len, "tail length must not exceed obstacle count"
        assert obstacles[0][0] > tail_len, "obstacles must begin beyond the tail"
        stop_distance = int(obstacles[tail_len - 1][0])
        for _dist, r, c in obstacles[: tail_len - 1]:
            out[int(r), int(c)] = bg
        dr, dc = direction
        shift = int(stop_distance - 1)
        r0, c0, r1, c1 = body
        nr0 = int(r0 + dr * shift)
        nr1 = int(r1 + dr * shift)
        nc0 = int(c0 + dc * shift)
        nc1 = int(c1 + dc * shift)
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "moved body must stay in bounds"
        region = out[nr0 : nr1 + 1, nc0 : nc1 + 1]
        assert np.all(region == bg), "slide channels must not interfere"
        region[:, :] = int(slider["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each tail and slide its rectangle until the L-th obstacle in the tail direction blocks it.

    Input structure:
    - The background is black, grey pixels are single-cell obstacles, and every other component is one monochrome solid rectangle with a one-cell-thick straight tail.
    - Each tail points up, down, left, or right and has length at least two.
    - In the full rectangular slide channel ahead of that body, grey obstacle pixels appear at distinct encounter distances.

    Rule:
    - Detect each tailed rectangle, discard the tail, and read the body's slide direction from the tail.
    - Let L be the tail length; within the body's full slide channel, count obstacles by encounter order, not by geometric distance.
    - Erase the first L-1 encountered obstacles, then stop the body immediately before the L-th obstacle; keep the L-th and all farther obstacles unchanged.

    Key insight:
    - The tail length is an ordinal selector for which obstacle becomes the stopper, so the body must count channel obstacles across its whole swept rectangle rather than move by L cells.
    """
    return _apply_true_rule(grid)


def _random_distances(tail_len: int, obstacle_count: int) -> list[int]:
    dist = random.randint(tail_len + 1, tail_len + 3)
    out = [int(dist)]
    for _ in range(obstacle_count - 1):
        dist += random.randint(1, 3)
        out.append(int(dist))
    return out


def _counterexample_lanes(orth: int, attach: int, tail_len: int, obstacle_count: int) -> list[int]:
    other_choices = [idx for idx in range(orth) if idx != attach]
    other = random.choice(other_choices)
    off_idx = random.randrange(tail_len)
    lanes: list[int] = []
    attach_used = 0
    for idx in range(tail_len):
        if idx == off_idx:
            lanes.append(int(other))
        else:
            lanes.append(int(attach))
            attach_used += 1
    while len(lanes) < obstacle_count:
        if attach_used < tail_len:
            lanes.append(int(attach))
            attach_used += 1
        else:
            lane = int(attach) if random.random() < 0.5 else int(random.choice(other_choices))
            lanes.append(lane)
    return lanes


def _regular_lanes(orth: int, attach: int, tail_len: int, obstacle_count: int) -> list[int]:
    other_choices = [idx for idx in range(orth) if idx != attach]
    lanes = [int(attach)] * tail_len
    while len(lanes) < obstacle_count:
        if random.random() < 0.6:
            lanes.append(int(attach))
        else:
            lanes.append(int(random.choice(other_choices)))
    return lanes


def _make_object_spec(family: str, color: int, challenge: bool) -> dict:
    body_h = random.randint(*BODY_H_RANGE)
    body_w = random.randint(*BODY_W_RANGE)
    tail_len = random.randint(*TAIL_RANGE)
    obstacle_count = tail_len + random.randint(*EXTRA_OBSTACLES_RANGE)
    distances = _random_distances(tail_len, obstacle_count)
    direction = random.choice(DIRS_H if family == "horizontal" else DIRS_V)
    orth = body_h if family == "horizontal" else body_w
    attach = random.randrange(orth)
    if challenge:
        lanes = _counterexample_lanes(orth, attach, tail_len, obstacle_count)
    else:
        lanes = _regular_lanes(orth, attach, tail_len, obstacle_count)

    if family == "horizontal":
        zone_h = body_h
        zone_w = body_w + distances[-1]
    else:
        zone_h = body_h + distances[-1]
        zone_w = body_w

    return {
        "color": int(color),
        "body_h": int(body_h),
        "body_w": int(body_w),
        "tail_len": int(tail_len),
        "distances": [int(d) for d in distances],
        "direction": direction,
        "attach": int(attach),
        "lanes": [int(x) for x in lanes],
        "zone_h": int(zone_h),
        "zone_w": int(zone_w),
    }


def _zone_geometry(spec: dict) -> tuple[BBox, list[Coord], list[Coord], BBox]:
    h = int(spec["body_h"])
    w = int(spec["body_w"])
    tail_len = int(spec["tail_len"])
    distances = spec["distances"]
    attach = int(spec["attach"])
    direction = spec["direction"]
    lanes = spec["lanes"]
    shift = int(distances[tail_len - 1] - 1)

    if direction == (0, 1):
        body = (0, 0, h - 1, w - 1)
        tail = [(attach, c) for c in range(w, w + tail_len)]
        obstacles = [(int(lanes[i]), int(w - 1 + distances[i])) for i in range(len(distances))]
        final_body = (0, shift, h - 1, shift + w - 1)
    elif direction == (0, -1):
        origin = int(distances[-1])
        body = (0, origin, h - 1, origin + w - 1)
        tail = [(attach, c) for c in range(origin - tail_len, origin)]
        obstacles = [(int(lanes[i]), int(origin - distances[i])) for i in range(len(distances))]
        final_body = (0, origin - shift, h - 1, origin - shift + w - 1)
    elif direction == (1, 0):
        body = (0, 0, h - 1, w - 1)
        tail = [(r, attach) for r in range(h, h + tail_len)]
        obstacles = [(int(h - 1 + distances[i]), int(lanes[i])) for i in range(len(distances))]
        final_body = (shift, 0, shift + h - 1, w - 1)
    else:
        origin = int(distances[-1])
        body = (origin, 0, origin + h - 1, w - 1)
        tail = [(r, attach) for r in range(origin - tail_len, origin)]
        obstacles = [(int(origin - distances[i]), int(lanes[i])) for i in range(len(distances))]
        final_body = (origin - shift, 0, origin - shift + h - 1, w - 1)
    return body, tail, obstacles, final_body


def _distribute_extra(extra: int, slots: int) -> list[int]:
    out = [0] * slots
    for _ in range(extra):
        out[random.randrange(slots)] += 1
    return out


def _place_horizontal(specs: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(specs)
    base_h = 2 + sum(spec["zone_h"] for spec in specs) + (n - 1)
    if base_h > 30:
        return None
    extra_h = random.randint(0, min(30 - base_h, 6))
    gaps = [1 + add for add in _distribute_extra(extra_h, n + 1)]
    rows: list[int] = []
    r = int(gaps[0])
    for idx, spec in enumerate(specs):
        rows.append(int(r))
        r += int(spec["zone_h"])
        if idx < n - 1:
            r += int(gaps[idx + 1])
    height = int(r + gaps[-1])

    min_w = 2 + max(spec["zone_w"] for spec in specs)
    if min_w > 30:
        return None
    width = random.randint(min_w, min(30, min_w + 8))
    placements = []
    for spec, row in zip(specs, rows):
        left = random.randint(1, width - int(spec["zone_w"]) - 1)
        placements.append((int(row), int(left)))
    return height, width, placements


def _place_vertical(specs: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(specs)
    base_w = 2 + sum(spec["zone_w"] for spec in specs) + (n - 1)
    if base_w > 30:
        return None
    extra_w = random.randint(0, min(30 - base_w, 6))
    gaps = [1 + add for add in _distribute_extra(extra_w, n + 1)]
    cols: list[int] = []
    c = int(gaps[0])
    for idx, spec in enumerate(specs):
        cols.append(int(c))
        c += int(spec["zone_w"])
        if idx < n - 1:
            c += int(gaps[idx + 1])
    width = int(c + gaps[-1])

    min_h = 2 + max(spec["zone_h"] for spec in specs)
    if min_h > 30:
        return None
    height = random.randint(min_h, min(30, min_h + 8))
    placements = []
    for spec, col in zip(specs, cols):
        top = random.randint(1, height - int(spec["zone_h"]) - 1)
        placements.append((int(top), int(col)))
    return height, width, placements


def _paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _build_scene(specs: list[dict], height: int, width: int, placements: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for spec, (top, left) in zip(specs, placements):
        body, tail, obstacles, final_body = _zone_geometry(spec)
        body_abs = (
            int(body[0] + top),
            int(body[1] + left),
            int(body[2] + top),
            int(body[3] + left),
        )
        final_abs = (
            int(final_body[0] + top),
            int(final_body[1] + left),
            int(final_body[2] + top),
            int(final_body[3] + left),
        )
        _paint_rect(inp, body_abs, spec["color"])
        _paint_rect(out, final_abs, spec["color"])
        for r, c in tail:
            inp[int(top + r), int(left + c)] = int(spec["color"])
        for idx, (r, c) in enumerate(obstacles):
            inp[int(top + r), int(left + c)] = OBSTACLE
            if idx >= int(spec["tail_len"] - 1):
                out[int(top + r), int(left + c)] = OBSTACLE
    return inp, out


def _naive_tail_length_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    sliders = _extract_sliders(g)
    for r, c in np.argwhere(g == OBSTACLE):
        out[int(r), int(c)] = OBSTACLE
    for slider in sliders:
        r0, c0, r1, c1 = slider["body"]
        dr, dc = slider["direction"]
        shift = int(slider["tail_len"])
        _paint_rect(
            out,
            (int(r0 + dr * shift), int(c0 + dc * shift), int(r1 + dr * shift), int(c1 + dc * shift)),
            int(slider["color"]),
        )
    return out


def _tail_line_only_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sliders = _extract_sliders(g)
    for slider in sliders:
        coords = slider["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for slider in sliders:
        line_obstacles = _tail_line_obstacles(g, slider["body"], slider["direction"], slider["attach"])
        tail_len = int(slider["tail_len"])
        assert len(line_obstacles) >= tail_len
        for _dist, r, c in line_obstacles[: tail_len - 1]:
            out[int(r), int(c)] = BG
        stop_distance = int(line_obstacles[tail_len - 1][0])
        dr, dc = slider["direction"]
        shift = int(stop_distance - 1)
        r0, c0, r1, c1 = slider["body"]
        _paint_rect(
            out,
            (int(r0 + dr * shift), int(c0 + dc * shift), int(r1 + dr * shift), int(c1 + dc * shift)),
            int(slider["color"]),
        )
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    sliders = _extract_sliders(inp)
    has_tail_line_counterexample = False
    for slider in sliders:
        obstacles = _channel_obstacles(inp, slider["body"], slider["direction"])
        tail_len = int(slider["tail_len"])
        if len(obstacles) < tail_len:
            return False
        if obstacles[0][0] <= tail_len:
            return False
        if obstacles[tail_len - 1][0] - 1 <= tail_len:
            return False
        line_obstacles = _tail_line_obstacles(inp, slider["body"], slider["direction"], slider["attach"])
        if len(line_obstacles) < tail_len:
            return False
        if line_obstacles[tail_len - 1][0] != obstacles[tail_len - 1][0]:
            has_tail_line_counterexample = True
    if not has_tail_line_counterexample:
        return False
    if np.array_equal(_naive_tail_length_output(inp), expected):
        return False
    if np.array_equal(_tail_line_only_output(inp), expected):
        return False
    return True


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    family = random.choice(("horizontal", "vertical"))
    count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    colors = random.sample(BODY_COLORS, count)
    specs = []
    for idx in range(count):
        specs.append(_make_object_spec(family, colors[idx], challenge=(idx == 0)))
    if family == "horizontal":
        placed = _place_horizontal(specs)
    else:
        placed = _place_vertical(specs)
    if placed is None:
        return None
    height, width, placements = placed
    inp, out = _build_scene(specs, height, width, placements)
    if _get_bg(inp) != BG:
        return None
    if not _validate_semantics(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid sliding-obstacle puzzle")
