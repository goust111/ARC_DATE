import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
TRAIL_COLOR = 2
DEVICE_SIZE = 3
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
FUSE_OFFSET = {
    (-1, -1): (0, 0),
    (-1, 1): (0, 2),
    (1, -1): (2, 0),
    (1, 1): (2, 2),
}
CORNER_TO_DIR = {offset: direction for direction, offset in FUSE_OFFSET.items()}
HIT_OFFSET = {
    (-1, -1): (2, 2),
    (-1, 1): (2, 0),
    (1, -1): (0, 2),
    (1, 1): (0, 0),
}

GRID_MIN = 18
GRID_MAX = 30
ACTIVE_LEN_CHOICES = (3, 4)
EXTRA_DEVICE_CHOICES = (1, 2)
GAP_RANGE = (2, 5)
SHADOW_GAP_RANGE = (2, 4)
MARGIN_RANGE = (3, 6)
TERMINAL_MIN_LEN = 2
TERMINAL_MAX_LEN = 10
GEN_ATTEMPTS = 600
EXTRA_PLACE_TRIES = 160
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _non_bg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not visited[nr, nc] and int(grid[nr, nc]) != bg:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _device_cells(tl: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = tl
    return [(r0 + dr, c0 + dc) for dr in range(DEVICE_SIZE) for dc in range(DEVICE_SIZE)]


def _expanded_bbox(tl: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = tl
    return r0 - 1, c0 - 1, r0 + DEVICE_SIZE, c0 + DEVICE_SIZE


def _expanded_overlaps(tl_a: tuple[int, int], tl_b: tuple[int, int]) -> bool:
    ar0, ac0, ar1, ac1 = _expanded_bbox(tl_a)
    br0, bc0, br1, bc1 = _expanded_bbox(tl_b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _ray_start(tl: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = tl
    fr, fc = FUSE_OFFSET[direction]
    dr, dc = direction
    return r0 + fr + dr, c0 + fc + dc


def _step(coord: tuple[int, int], direction: tuple[int, int], times: int = 1) -> tuple[int, int]:
    return coord[0] + direction[0] * times, coord[1] + direction[1] * times


def _place_target_from_start(
    start: tuple[int, int], direction: tuple[int, int], gap: int
) -> tuple[int, int]:
    hit_r, hit_c = _step(start, direction, gap)
    hr, hc = HIT_OFFSET[direction]
    return hit_r - hr, hit_c - hc


def _place_target_from_source(
    source_tl: tuple[int, int], direction: tuple[int, int], gap: int
) -> tuple[int, int]:
    return _place_target_from_start(_ray_start(source_tl, direction), direction, gap)


def _continuation_start_after_block(
    target_tl: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    return _ray_start(target_tl, direction)


def _all_device_cells(devices: list[dict]) -> set[tuple[int, int]]:
    cells = set()
    for device in devices:
        cells.update(_device_cells(device["tl"]))
    return cells


def _parse_devices(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    devices = []
    for coords in _non_bg_components(g, bg):
        assert len(coords) == 9, "every foreground component must be a 3x3 device"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert (r1 - r0, c1 - c0) == (2, 2), "device bbox must be 3x3"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors, counts = np.unique(patch, return_counts=True)
        assert len(colors) == 2, "device must use body color plus one fuse color"
        count_map = {int(color): int(count) for color, count in zip(colors, counts)}
        body = next(color for color, count in count_map.items() if count == 8)
        fuse = next(color for color, count in count_map.items() if count == 1)
        fuse_pos = np.argwhere(patch == fuse)
        assert fuse_pos.shape == (1, 2), "fuse must occupy exactly one cell"
        fr, fc = map(int, fuse_pos[0])
        assert (fr, fc) in CORNER_TO_DIR, "fuse must sit on a corner"
        devices.append(
            {
                "tl": (r0, c0),
                "body": body,
                "fuse": fuse,
                "direction": CORNER_TO_DIR[(fr, fc)],
                "cells": set(coords),
            }
        )

    assert devices, "expected at least one device"
    body_counts = Counter(device["body"] for device in devices)
    ignition_colors = [color for color, count in body_counts.items() if count == 1]
    assert len(ignition_colors) == 1, "expected exactly one ignition body color"
    ignition_color = ignition_colors[0]
    for i, device in enumerate(devices):
        device["ignition"] = device["body"] == ignition_color
        device["index"] = i
    return devices


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    devices = _parse_devices(g)
    cell_to_device = {}
    for device in devices:
        for cell in device["cells"]:
            cell_to_device[cell] = device["index"]

    ignition = next(device["index"] for device in devices if device["ignition"])
    out = g.copy()
    fired = set()
    fire_order = []
    triggered_by = {}
    next_hit = {}
    paths = {}

    queue = deque([ignition])
    while queue:
        idx = queue.popleft()
        if idx in fired:
            continue
        fired.add(idx)
        fire_order.append(idx)

        device = devices[idx]
        direction = device["direction"]
        dr, dc = direction
        r, c = _ray_start(device["tl"], direction)
        path = []
        hit = None
        while _in_bounds(g.shape, r, c):
            other = cell_to_device.get((r, c))
            if other is not None and other != idx:
                hit = other
                break
            path.append((r, c))
            r += dr
            c += dc

        for rr, cc in path:
            out[rr, cc] = TRAIL_COLOR

        paths[idx] = path
        next_hit[idx] = hit
        if hit is not None and hit not in fired and hit not in queue:
            triggered_by.setdefault(hit, idx)
            queue.append(hit)

    return {
        "out": out,
        "devices": devices,
        "fired": fired,
        "fire_order": fire_order,
        "triggered_by": triggered_by,
        "next_hit": next_hit,
        "paths": paths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire the unique ignition device, then draw only the diagonal ray chain triggered by struck corner-coded devices.

    Input structure:
    - The background contains several isolated 3x3 device blocks.
    - Each device has eight body-color cells plus one fuse-color cell on a corner; the fuse corner encodes one of the four outward diagonal directions.
    - Exactly one device uses a unique body color, marking it as the ignition device; all other devices share the other body color.

    Rule:
    - The ignition device fires first, starting from the cell just outside its fuse corner and extending a diagonal ray in the fuse direction.
    - A ray paints the fixed trail color on background cells until it leaves the grid, or until the next ray step would enter any cell of another device, where it stops just outside that device.
    - A struck device fires once using its own fuse-coded direction, regardless of the direction from which it was hit; devices never struck stay unchanged and silent.

    Key insight:
    - Activation is controlled by reachability along first-hit diagonal rays, but each newly activated device emits in its own corner-coded direction rather than continuing the incoming trajectory.
    """
    return _simulate(np.asarray(grid, dtype=int))["out"]


def _direction_choices(exclude: set[tuple[int, int]] = frozenset()) -> list[tuple[int, int]]:
    return [direction for direction in DIRS if direction not in exclude]


def _build_active_layout(active_len: int) -> tuple[list[dict], dict] | None:
    root_dir = random.choice(DIRS)
    first_gap = random.randint(*GAP_RANGE)
    root_tl = (0, 0)
    turn_tl = _place_target_from_source(root_tl, root_dir, first_gap)

    shadow_gap = random.randint(*SHADOW_GAP_RANGE)
    shadow_start = _continuation_start_after_block(turn_tl, root_dir)
    shadow_tl = _place_target_from_start(shadow_start, root_dir, shadow_gap)

    turn_dir = random.choice(_direction_choices({root_dir, (-root_dir[0], -root_dir[1])}))
    second_gap = random.randint(*GAP_RANGE)
    chain_tl = _place_target_from_source(turn_tl, turn_dir, second_gap)

    layout = [
        {"name": "ignition", "tl": root_tl, "direction": root_dir},
        {"name": "turn", "tl": turn_tl, "direction": turn_dir},
        {"name": "shadow", "tl": shadow_tl, "direction": random.choice(DIRS)},
    ]

    terminal_name = "tail"
    if active_len == 4:
        tail_dir = random.choice(_direction_choices({(-turn_dir[0], -turn_dir[1])}))
        third_gap = random.randint(*GAP_RANGE)
        tail_tl = _place_target_from_source(chain_tl, tail_dir, third_gap)
        terminal_dir = random.choice(DIRS)
        layout.append({"name": "chain", "tl": chain_tl, "direction": tail_dir})
        layout.append({"name": "tail", "tl": tail_tl, "direction": terminal_dir})
    else:
        terminal_dir = random.choice(DIRS)
        layout.append({"name": "chain", "tl": chain_tl, "direction": terminal_dir})

    tles = [item["tl"] for item in layout]
    for i in range(len(tles)):
        for j in range(i + 1, len(tles)):
            if _expanded_overlaps(tles[i], tles[j]):
                return None

    return layout, {"active_len": active_len, "terminal_name": terminal_name}


def _layout_bbox(layout: list[dict]) -> tuple[int, int, int, int]:
    rows = []
    cols = []
    for item in layout:
        r0, c0 = item["tl"]
        rows.extend([r0, r0 + 2])
        cols.extend([c0, c0 + 2])
    return min(rows), min(cols), max(rows), max(cols)


def _translate_layout(layout: list[dict], dr: int, dc: int) -> list[dict]:
    translated = []
    for item in layout:
        rr, cc = item["tl"]
        translated.append({**item, "tl": (rr + dr, cc + dc)})
    return translated


def _render_devices(
    shape: tuple[int, int],
    layout: list[dict],
    ignition_body: int,
    common_body: int,
    fuse_color: int,
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for item in layout:
        r0, c0 = item["tl"]
        body = ignition_body if item["name"] == "ignition" else common_body
        grid[r0 : r0 + 3, c0 : c0 + 3] = body
        fr, fc = FUSE_OFFSET[item["direction"]]
        grid[r0 + fr, c0 + fc] = fuse_color
    return grid


def _path_cells(path_map: dict[int, list[tuple[int, int]]]) -> set[tuple[int, int]]:
    cells = set()
    for path in path_map.values():
        cells.update(path)
    return cells


def _free_top_lefts(
    shape: tuple[int, int], existing_tls: list[tuple[int, int]], forbidden_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = shape
    options = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            tl = (r0, c0)
            if any(_expanded_overlaps(tl, other) for other in existing_tls):
                continue
            cells = _device_cells(tl)
            if any(cell in forbidden_cells for cell in cells):
                continue
            options.append(tl)
    return options


def _pick_palette() -> tuple[int, int, int]:
    colors = random.sample(PALETTE, 3)
    return colors[0], colors[1], colors[2]


def _find_device_by_tl(devices: list[dict], tl: tuple[int, int]) -> int:
    matches = [device["index"] for device in devices if device["tl"] == tl]
    assert len(matches) == 1
    return matches[0]


def _semantic_ok(grid: np.ndarray, meta: dict) -> bool:
    try:
        sim = _simulate(grid)
    except AssertionError:
        return False

    devices = sim["devices"]
    fire_order = sim["fire_order"]
    fired = sim["fired"]
    paths = sim["paths"]
    next_hit = sim["next_hit"]
    triggered_by = sim["triggered_by"]

    if len(fired) != meta["active_len"]:
        return False
    if len(devices) < meta["active_len"] + 1:
        return False
    if len(fired) >= len(devices):
        return False

    shadow_idx = _find_device_by_tl(devices, meta["shadow_tl"])
    if shadow_idx in fired:
        return False

    inactive_extras = [
        _find_device_by_tl(devices, tl)
        for tl in meta["extra_tls"]
        if _find_device_by_tl(devices, tl) not in fired
    ]
    if len(inactive_extras) != len(meta["extra_tls"]):
        return False

    if len(fire_order) < 2:
        return False
    turn_idx = _find_device_by_tl(devices, meta["turn_tl"])
    if turn_idx not in fired:
        return False
    predecessor = triggered_by.get(turn_idx)
    if predecessor is None:
        return False
    if devices[predecessor]["direction"] == devices[turn_idx]["direction"]:
        return False

    first_target = next_hit.get(fire_order[0])
    if first_target != turn_idx:
        return False

    last_idx = fire_order[-1]
    if next_hit[last_idx] is not None:
        return False
    last_len = len(paths[last_idx])
    if not (TERMINAL_MIN_LEN <= last_len <= TERMINAL_MAX_LEN):
        return False

    if any(len(paths[idx]) < 2 for idx in fire_order):
        return False

    seen_paths = set()
    for idx in fire_order:
        path = tuple(paths[idx])
        if len(set(path)) != len(path):
            return False
        if seen_paths.intersection(path):
            return False
        seen_paths.update(path)

    shadow_line_idx = _find_device_by_tl(devices, meta["shadow_tl"])
    if shadow_line_idx in fired:
        return False

    root_idx = fire_order[0]
    root_dir = devices[root_idx]["direction"]
    turn_tl = meta["turn_tl"]
    shadow_tl = meta["shadow_tl"]
    exit_start = _continuation_start_after_block(turn_tl, root_dir)
    hit_offset = HIT_OFFSET[root_dir]
    shadow_hit = (shadow_tl[0] + hit_offset[0], shadow_tl[1] + hit_offset[1])
    dr, dc = root_dir
    delta_r = shadow_hit[0] - exit_start[0]
    delta_c = shadow_hit[1] - exit_start[1]
    if dr == 0 or dc == 0:
        return False
    if delta_r * dc != delta_c * dr:
        return False
    if delta_r // dr < 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    active_len = random.choice(ACTIVE_LEN_CHOICES)
    built = _build_active_layout(active_len)
    if built is None:
        return None
    layout, info = built
    bbox_r0, bbox_c0, bbox_r1, bbox_c1 = _layout_bbox(layout)
    core_h = bbox_r1 - bbox_r0 + 1
    core_w = bbox_c1 - bbox_c0 + 1

    for _ in range(80):
        top = random.randint(*MARGIN_RANGE)
        bottom = random.randint(*MARGIN_RANGE)
        left = random.randint(*MARGIN_RANGE)
        right = random.randint(*MARGIN_RANGE)
        h = core_h + top + bottom
        w = core_w + left + right
        if not (GRID_MIN <= h <= GRID_MAX and GRID_MIN <= w <= GRID_MAX):
            continue

        translated = _translate_layout(layout, top - bbox_r0, left - bbox_c0)
        ignition_body, common_body, fuse_color = _pick_palette()
        grid = _render_devices((h, w), translated, ignition_body, common_body, fuse_color)

        pre_meta = {
            "active_len": active_len,
            "turn_tl": next(item["tl"] for item in translated if item["name"] == "turn"),
            "shadow_tl": next(item["tl"] for item in translated if item["name"] == "shadow"),
            "extra_tls": [],
        }

        try:
            pre_sim = _simulate(grid)
        except AssertionError:
            continue

        if len(pre_sim["fired"]) != active_len:
            continue
        if _find_device_by_tl(pre_sim["devices"], pre_meta["shadow_tl"]) in pre_sim["fired"]:
            continue

        reserved = _path_cells(pre_sim["paths"])
        existing_tls = [item["tl"] for item in translated]
        options = _free_top_lefts((h, w), existing_tls, reserved)
        if not options:
            continue

        extra_count = random.choice(EXTRA_DEVICE_CHOICES)
        if len(options) < extra_count:
            continue

        random.shuffle(options)
        extra_tls = []
        for tl in options:
            if any(_expanded_overlaps(tl, other) for other in existing_tls + extra_tls):
                continue
            extra_tls.append(tl)
            if len(extra_tls) == extra_count:
                break
        if len(extra_tls) != extra_count:
            continue

        full_layout = translated[:]
        for tl in extra_tls:
            full_layout.append(
                {
                    "name": f"extra_{len(full_layout)}",
                    "tl": tl,
                    "direction": random.choice(DIRS),
                }
            )

        grid = _render_devices((h, w), full_layout, ignition_body, common_body, fuse_color)
        meta = {**pre_meta, "extra_tls": extra_tls}
        if not _semantic_ok(grid, meta):
            continue

        expected = _simulate(grid)["out"]
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
