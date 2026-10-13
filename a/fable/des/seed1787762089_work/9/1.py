from __future__ import annotations

from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

KEY_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, AZURE)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_RNG = np.random.default_rng()

# Solver/generator shared shape library: distinct up to rotation/reflection.
SHAPE_LIBRARY = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)

MAX_GENERATION_ATTEMPTS = 500
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (26, 34)
BUILD_H_RANGE = (13, 19)
BUILD_W_RANGE = (15, 21)
MIN_ROOM_H = 5
MIN_ROOM_W = 5
PLACEMENT_ATTEMPTS = 240


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _transform_shape(coords, rotation: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rotation % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_shape(pts)


def _all_shape_transforms(coords) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for flip in (False, True):
        for rotation in range(4):
            shape = _transform_shape(coords, rotation, flip)
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
    return out


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    return min(_all_shape_transforms(coords))


def _shape_size(coords) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _color_components(
    grid: np.ndarray,
    color: int,
    allowed_mask: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    if allowed_mask is not None:
        mask &= np.asarray(allowed_mask, dtype=bool)
    return _components_from_mask(mask)


def _room_side_from_lock(lock_coords, wall_mask: np.ndarray) -> str:
    hits = defaultdict(int)
    h, w = wall_mask.shape
    for r, c in lock_coords:
        if r - 1 >= 0 and wall_mask[r - 1, c]:
            hits["top"] += 1
        if r + 1 < h and wall_mask[r + 1, c]:
            hits["bottom"] += 1
        if c - 1 >= 0 and wall_mask[r, c - 1]:
            hits["left"] += 1
        if c + 1 < w and wall_mask[r, c + 1]:
            hits["right"] += 1
    sides = [side for side, count in hits.items() if count > 0]
    assert len(sides) == 1, "each lock must touch exactly one wall"
    return sides[0]


def _fill_room(
    out: np.ndarray,
    room_box: tuple[int, int, int, int],
    side: str,
    depth: int,
    fill_color: int,
    bg: int,
) -> None:
    r0, r1, c0, c1 = room_box
    if side == "top":
        view = out[r0 : min(r1 + 1, r0 + depth), c0 : c1 + 1]
    elif side == "bottom":
        view = out[max(r0, r1 - depth + 1) : r1 + 1, c0 : c1 + 1]
    elif side == "left":
        view = out[r0 : r1 + 1, c0 : min(c1 + 1, c0 + depth)]
    elif side == "right":
        view = out[r0 : r1 + 1, max(c0, c1 - depth + 1) : c1 + 1]
    else:
        raise AssertionError(f"unknown side {side}")
    view[view == int(bg)] = int(fill_color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    wall_components = _color_components(g, GREY)
    assert wall_components, "expected a grey building"
    building = max(wall_components, key=len)
    br0, br1, bc0, bc1 = _bbox(building)

    inner_mask = np.zeros_like(g, dtype=bool)
    inner_mask[br0 + 1 : br1, bc0 + 1 : bc1] = True
    wall_mask = g == GREY

    room_components = _components_from_mask(inner_mask & (~wall_mask))
    assert 3 <= len(room_components) <= 4, "expected 3 or 4 rooms"

    room_id = np.full(g.shape, -1, dtype=int)
    for idx, comp in enumerate(room_components):
        for r, c in comp:
            room_id[r, c] = int(idx)

    lock_components = _color_components(g, MAROON, inner_mask)
    assert len(lock_components) == len(room_components), "one lock per room is required"
    locks_by_room: dict[int, list[tuple[int, int]]] = {}
    for comp in lock_components:
        rid = int(room_id[comp[0]])
        assert rid >= 0, "lock must lie inside a room"
        assert rid not in locks_by_room, "room must contain exactly one lock"
        locks_by_room[rid] = comp

    outside_mask = np.ones_like(g, dtype=bool)
    outside_mask[br0 : br1 + 1, bc0 : bc1 + 1] = False

    key_lookup: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY, MAROON):
            continue
        comps = _color_components(g, color, outside_mask)
        if not comps:
            continue
        sig = _canonical_shape(comps[0])
        assert all(_canonical_shape(comp) == sig for comp in comps), "a keychain must repeat one shape"
        assert sig not in key_lookup, "lock-key matching must be unambiguous"
        key_lookup[sig] = (color, len(comps))

    out = g.copy()
    for rid, room_comp in enumerate(room_components):
        lock_comp = locks_by_room[rid]
        room_box = _bbox(room_comp)
        lock_sig = _canonical_shape(lock_comp)
        assert lock_sig in key_lookup, "every lock must have a matching keychain"
        fill_color, depth = key_lookup[lock_sig]
        side = _room_side_from_lock(lock_comp, wall_mask)
        _fill_room(out, room_box, side, depth, fill_color, bg)

    return out


def _partition_building(
    build_h: int,
    build_w: int,
    room_count: int,
) -> tuple[list[tuple[int, int, int, int]], set[tuple[int, int]]] | None:
    regions = [(1, build_h - 2, 1, build_w - 2)]
    wall_cells: set[tuple[int, int]] = set()

    for _ in range(room_count - 1):
        choices = []
        for idx, (r0, r1, c0, c1) in enumerate(regions):
            if (r1 - r0 + 1) >= 2 * MIN_ROOM_H + 1:
                choices.append((idx, "h"))
            if (c1 - c0 + 1) >= 2 * MIN_ROOM_W + 1:
                choices.append((idx, "v"))
        if not choices:
            return None

        choice_idx = int(_RNG.integers(len(choices)))
        region_idx, orient = choices[choice_idx]
        r0, r1, c0, c1 = regions.pop(region_idx)

        if orient == "h":
            wr = int(_RNG.integers(r0 + MIN_ROOM_H, r1 - MIN_ROOM_H + 1))
            for c in range(c0, c1 + 1):
                wall_cells.add((wr, c))
            regions.extend([(r0, wr - 1, c0, c1), (wr + 1, r1, c0, c1)])
        else:
            wc = int(_RNG.integers(c0 + MIN_ROOM_W, c1 - MIN_ROOM_W + 1))
            for r in range(r0, r1 + 1):
                wall_cells.add((r, wc))
            regions.extend([(r0, r1, c0, wc - 1), (r0, r1, wc + 1, c1)])

    return regions, wall_cells


def _choose_lock_placement(
    room_box: tuple[int, int, int, int],
    base_shape,
) -> tuple[str, tuple[tuple[int, int], ...]] | None:
    r0, r1, c0, c1 = room_box
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    by_side = defaultdict(list)

    for shape in _all_shape_transforms(base_shape):
        shape_h, shape_w = _shape_size(shape)

        if room_h >= shape_h + 1 and room_w >= shape_w + 2:
            lo, hi = c0 + 1, c1 - shape_w
            if lo <= hi:
                by_side["top"].append((shape, r0, -1))
                by_side["bottom"].append((shape, r1 - shape_h + 1, -1))

        if room_w >= shape_w + 1 and room_h >= shape_h + 2:
            lo, hi = r0 + 1, r1 - shape_h
            if lo <= hi:
                by_side["left"].append((shape, -1, c0))
                by_side["right"].append((shape, -1, c1 - shape_w + 1))

    feasible_sides = [side for side, items in by_side.items() if items]
    if not feasible_sides:
        return None

    side = feasible_sides[int(_RNG.integers(len(feasible_sides)))]
    shape, fixed_r, fixed_c = by_side[side][int(_RNG.integers(len(by_side[side])))]
    shape_h, shape_w = _shape_size(shape)

    if side in ("top", "bottom"):
        base_r = fixed_r
        base_c = int(_RNG.integers(c0 + 1, c1 - shape_w + 1))
    else:
        base_r = int(_RNG.integers(r0 + 1, r1 - shape_h + 1))
        base_c = fixed_c

    placed = tuple(sorted((base_r + r, base_c + c) for r, c in shape))
    return side, placed


def _make_keychain_pattern(
    base_shape,
    copies: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    variants = _all_shape_transforms(base_shape)
    shape = variants[int(_RNG.integers(len(variants)))]
    axis = "horizontal" if int(_RNG.integers(2)) == 0 else "vertical"
    shape_h, shape_w = _shape_size(shape)
    pts = []
    for i in range(copies):
        dr = i * (shape_h + 1) if axis == "vertical" else 0
        dc = i * (shape_w + 1) if axis == "horizontal" else 0
        pts.extend((r + dr, c + dc) for r, c in shape)
    return shape, tuple(sorted(pts))


def _mark_buffer(mask: np.ndarray, coords) -> None:
    h, w = mask.shape
    for r, c in coords:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _place_pattern(
    pattern,
    h: int,
    w: int,
    forbidden: np.ndarray,
) -> tuple[tuple[int, int], ...] | None:
    pattern_h, pattern_w = _shape_size(pattern)
    if pattern_h > h or pattern_w > w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = int(_RNG.integers(0, h - pattern_h + 1))
        base_c = int(_RNG.integers(0, w - pattern_w + 1))
        placed = tuple((base_r + r, base_c + c) for r, c in pattern)
        if all(not forbidden[r, c] for r, c in placed):
            return placed
    return None


def _render_expected(
    grid: np.ndarray,
    rooms_meta,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for meta in rooms_meta:
        _fill_room(out, meta["room_box"], meta["side"], meta["depth"], meta["key_color"], BLACK)
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = 3 if int(_RNG.integers(2)) == 0 else 4
    build_h = int(_RNG.integers(BUILD_H_RANGE[0], BUILD_H_RANGE[1] + 1))
    build_w = int(_RNG.integers(BUILD_W_RANGE[0], BUILD_W_RANGE[1] + 1))
    partition = _partition_building(build_h, build_w, room_count)
    if partition is None:
        return None
    room_boxes_local, wall_cells_local = partition

    shape_indices = _RNG.choice(len(SHAPE_LIBRARY), size=room_count, replace=False)
    key_colors = _RNG.choice(np.array(KEY_COLORS, dtype=int), size=room_count, replace=False)

    room_entries = []
    chain_patterns = []
    for idx, room_box in enumerate(room_boxes_local):
        base_shape = SHAPE_LIBRARY[int(shape_indices[idx])]
        placement = _choose_lock_placement(room_box, base_shape)
        if placement is None:
            return None
        side, lock_cells = placement

        r0, r1, c0, c1 = room_box
        depth_limit = (r1 - r0 + 1) if side in ("top", "bottom") else (c1 - c0 + 1)
        depth = int(_RNG.integers(1, min(4, depth_limit) + 1))
        key_color = int(key_colors[idx])
        _, pattern = _make_keychain_pattern(base_shape, depth)

        room_entries.append(
            {
                "room_box_local": room_box,
                "side": side,
                "lock_cells_local": lock_cells,
                "depth": depth,
                "key_color": key_color,
                "base_shape": base_shape,
                "key_pattern": pattern,
            }
        )
        chain_patterns.append(pattern)

    max_chain_h = max(_shape_size(pattern)[0] for pattern in chain_patterns)
    max_chain_w = max(_shape_size(pattern)[1] for pattern in chain_patterns)

    min_grid_h = max(GRID_H_RANGE[0], build_h + 6, max_chain_h + 4)
    min_grid_w = max(GRID_W_RANGE[0], build_w + 6, max_chain_w + 4)
    if min_grid_h > GRID_H_RANGE[1] or min_grid_w > GRID_W_RANGE[1]:
        return None

    grid_h = int(_RNG.integers(min_grid_h, GRID_H_RANGE[1] + 1))
    grid_w = int(_RNG.integers(min_grid_w, GRID_W_RANGE[1] + 1))

    margin_r_lo = 2
    margin_r_hi = grid_h - build_h - 2
    margin_c_lo = 2
    margin_c_hi = grid_w - build_w - 2
    if margin_r_lo > margin_r_hi or margin_c_lo > margin_c_hi:
        return None

    build_r0 = int(_RNG.integers(margin_r_lo, margin_r_hi + 1))
    build_c0 = int(_RNG.integers(margin_c_lo, margin_c_hi + 1))

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    grid[build_r0, build_c0 : build_c0 + build_w] = GREY
    grid[build_r0 + build_h - 1, build_c0 : build_c0 + build_w] = GREY
    grid[build_r0 : build_r0 + build_h, build_c0] = GREY
    grid[build_r0 : build_r0 + build_h, build_c0 + build_w - 1] = GREY
    for r, c in wall_cells_local:
        grid[build_r0 + r, build_c0 + c] = GREY

    rooms_meta = []
    for entry in room_entries:
        lock_cells_global = tuple(
            sorted((build_r0 + r, build_c0 + c) for r, c in entry["lock_cells_local"])
        )
        for r, c in lock_cells_global:
            grid[r, c] = MAROON
        rr0, rr1, cc0, cc1 = entry["room_box_local"]
        rooms_meta.append(
            {
                "room_box": (build_r0 + rr0, build_r0 + rr1, build_c0 + cc0, build_c0 + cc1),
                "side": entry["side"],
                "depth": entry["depth"],
                "key_color": entry["key_color"],
                "lock_cells": lock_cells_global,
            }
        )

    forbidden = np.zeros_like(grid, dtype=bool)
    building_box = []
    for r in range(build_r0, build_r0 + build_h):
        for c in range(build_c0, build_c0 + build_w):
            building_box.append((r, c))
    _mark_buffer(forbidden, building_box)

    for entry in room_entries:
        key_pattern = entry["key_pattern"]
        placed = _place_pattern(key_pattern, grid_h, grid_w, forbidden)
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = entry["key_color"]
        _mark_buffer(forbidden, placed)

    expected = _render_expected(grid, rooms_meta)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
