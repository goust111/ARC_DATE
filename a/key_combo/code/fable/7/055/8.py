from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
PALETTE = [0, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_order(coords: np.ndarray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == -1:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray, *, bg: int, left_color: int, right_color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    inner_mask = g != int(bg)
    inner_mask[:, 0] = False
    inner_mask[:, -1] = False
    comps = _label_components(inner_mask)
    objects: list[dict] = []
    for coords in comps:
        values = {int(g[r, c]) for r, c in coords}
        main_colors = values - {int(BLUE)}
        assert len(main_colors) == 1, "each movable object must have one main color"
        main_color = main_colors.pop()
        assert main_color in {int(left_color), int(right_color)}, "movable object color must match one wall"
        assert values <= {int(main_color), int(BLUE)}, "object may contain only its main color and blue"

        r0, c0, r1, c1 = bbox_coords(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        rel = coords - np.array([r0, c0], dtype=int)
        for (rr, cc), (r, c) in zip(rel, coords):
            patch[int(rr), int(cc)] = int(g[int(r), int(c)])
        objects.append(
            {
                "main_color": int(main_color),
                "blue_count": int(np.sum(patch == int(BLUE))),
                "patch": patch,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
            }
        )
    return objects


def _touch_shift(coords: np.ndarray, settled: np.ndarray, direction: int, width: int) -> int:
    shift = 0
    while True:
        candidate = coords + np.array([0, shift + direction], dtype=int)
        cols = candidate[:, 1]
        if int(cols.min()) < 0 or int(cols.max()) >= int(width):
            break
        if np.any(settled[candidate[:, 0], candidate[:, 1]]):
            break
        shift += direction
    return int(shift)


def _settle_objects(
    objects: list[dict], *, shape: tuple[int, int], wall_col: int, direction: int
) -> list[tuple[dict, np.ndarray]]:
    h, w = shape
    settled = np.zeros((h, w), dtype=bool)
    settled[:, int(wall_col)] = True
    ordered = sorted(objects, key=lambda obj: _component_order(obj["coords"], direction))

    placements: list[tuple[dict, np.ndarray]] = []
    for obj in ordered:
        touch = _touch_shift(obj["coords"], settled, direction, w)
        final_shift = int(touch - direction * obj["blue_count"])
        if direction == -1:
            assert final_shift <= 0, "left-moving object cannot move away from the wall"
        else:
            assert final_shift >= 0, "right-moving object cannot move away from the wall"
        placed = obj["coords"] + np.array([0, final_shift], dtype=int)
        cols = placed[:, 1]
        assert int(cols.min()) >= 0 and int(cols.max()) < w, "settled object must remain in bounds"
        assert not np.any(settled[placed[:, 0], placed[:, 1]]), "settled object may not overlap obstacle"

        if obj["blue_count"] > 0:
            before_touch = placed + np.array([0, direction * obj["blue_count"]], dtype=int)
            assert not np.any(settled[before_touch[:, 0], before_touch[:, 1]])
        bumped = placed + np.array([0, direction * (obj["blue_count"] + 1)], dtype=int)
        if int(bumped[:, 1].min()) < 0 or int(bumped[:, 1].max()) >= w:
            pass
        else:
            assert np.any(settled[bumped[:, 0], bumped[:, 1]])

        settled[placed[:, 0], placed[:, 1]] = True
        placements.append((obj, placed))
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each multicolor object toward its same-colored wall while preserving a blue-count-sized gap.

    Input structure:
    - The dominant background surrounds two full-height wall columns: one color on the left edge and a different color on the right edge.
    - Away from the walls, several 8-connected objects use exactly one wall color as their main color, plus optional blue pixels embedded inside that object.
    - Blue pixels belong to the object that contains them; they are not standalone markers.

    Rule:
    - Keep both wall columns fixed and clear every movable object from its starting position.
    - For each wall color separately, process its objects from nearest to farthest from the matching wall.
    - Rigidly slide each object horizontally toward its own wall until the minimum remaining horizontal gap to the nearest same-color obstacle ahead equals that object's blue-pixel count.
    - After all movement, recolor every blue pixel to its containing object's main color.

    Key insight:
    - Blue pixels do not mark destinations directly; they encode how many empty columns the object must leave before the wall or the already-settled same-color object that blocks it, so order and counting both matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid with interior columns"
    h, w = g.shape
    bg = get_bg(g)
    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])
    assert left_color != right_color, "wall colors must differ"
    assert left_color != int(bg) and right_color != int(bg), "walls must be foreground colors"
    assert left_color != int(BLUE) and right_color != int(BLUE), "wall colors must differ from blue"
    assert np.all(g[:, 0] == left_color), "left wall must fill the full left column"
    assert np.all(g[:, -1] == right_color), "right wall must fill the full right column"
    assert set(map(int, np.unique(g))) <= {int(bg), int(BLUE), left_color, right_color}

    objects = _extract_objects(g, bg=bg, left_color=left_color, right_color=right_color)
    left_objects = [obj for obj in objects if obj["main_color"] == left_color]
    right_objects = [obj for obj in objects if obj["main_color"] == right_color]
    assert left_objects and right_objects, "both wall colors must have movable objects"

    out = np.full_like(g, int(bg))
    out[:, 0] = int(left_color)
    out[:, -1] = int(right_color)
    occupied = out != int(bg)

    placements: list[tuple[int, np.ndarray]] = []
    placements.extend((obj["main_color"], coords) for obj, coords in _settle_objects(left_objects, shape=g.shape, wall_col=0, direction=-1))
    placements.extend((obj["main_color"], coords) for obj, coords in _settle_objects(right_objects, shape=g.shape, wall_col=w - 1, direction=1))

    for color, coords in placements:
        assert not np.any(occupied[coords[:, 0], coords[:, 1]]), "different colors may not overlap after sliding"
        out[coords[:, 0], coords[:, 1]] = int(color)
        occupied[coords[:, 0], coords[:, 1]] = True
    return out


def _random_connected_mask(area: int) -> np.ndarray:
    area = int(area)
    dims = [
        (h, w)
        for h in range(1, 5)
        for w in range(1, 5)
        if max(h, w) <= area <= h * w and area / (h * w) >= 0.45
    ]
    h, w = random.choice(dims)
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if random.choice([False, True]):
        mask = np.flipud(mask)
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    return mask.copy()


def _make_patch(main_color: int, blue_count: int) -> np.ndarray:
    area = random.randint(max(blue_count + 1, 2), max(blue_count + 2, 6))
    mask = _random_connected_mask(area)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(main_color)
    coords = list(map(tuple, np.argwhere(mask)))
    random.shuffle(coords)
    for r, c in coords[:blue_count]:
        patch[int(r), int(c)] = int(BLUE)
    return patch


def _patch_to_coords(patch: np.ndarray, top: int, left: int, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != int(bg))
    return coords + np.array([int(top), int(left)], dtype=int)


def _pair_touches(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    dr = np.abs(coords_a[:, None, 0] - coords_b[None, :, 0])
    dc = np.abs(coords_a[:, None, 1] - coords_b[None, :, 1])
    return bool(np.any((dr <= 1) & (dc <= 1)))


def _side_touch_left(settled: np.ndarray, patch: np.ndarray, top: int, bg: int) -> int | None:
    coords0 = _patch_to_coords(patch, top, 0, bg)
    min_left = 0
    max_left = settled.shape[1] - patch.shape[1]
    if top < 0 or top + patch.shape[0] > settled.shape[0]:
        return None
    for left in range(min_left, max_left + 1):
        coords = coords0 + np.array([0, left], dtype=int)
        if not np.any(settled[coords[:, 0], coords[:, 1]]):
            return int(left)
    return None


def _side_touch_right(settled: np.ndarray, patch: np.ndarray, top: int, bg: int) -> int | None:
    coords0 = _patch_to_coords(patch, top, 0, bg)
    max_left = settled.shape[1] - patch.shape[1]
    if top < 0 or top + patch.shape[0] > settled.shape[0]:
        return None
    for left in range(max_left, -1, -1):
        coords = coords0 + np.array([0, left], dtype=int)
        if not np.any(settled[coords[:, 0], coords[:, 1]]):
            return int(left)
    return None


def _place_output_side(
    h: int,
    w: int,
    color: int,
    count: int,
    *,
    left_side: bool,
    require_chain: bool,
    global_occ: np.ndarray,
    bg: int,
    forced_blue_values: list[int] | None = None,
) -> list[dict] | None:
    wall_col = 0 if left_side else w - 1
    settled = np.zeros((h, w), dtype=bool)
    settled[:, wall_col] = True
    placed: list[dict] = []
    chain_hit = False

    blue_pool = forced_blue_values[:] if forced_blue_values is not None else []
    while len(blue_pool) < count:
        blue_pool.append(random.randint(0, 3))
    random.shuffle(blue_pool)

    for idx in range(count):
        for _ in range(120):
            blue_count = int(blue_pool[idx])
            patch = _make_patch(color, blue_count)
            rows = list(range(0, h - patch.shape[0] + 1))
            if placed and random.random() < 0.7:
                anchor = random.choice(placed)["top"]
                rows.sort(key=lambda r: (abs(r - anchor), random.random()))
            else:
                random.shuffle(rows)

            for top in rows:
                touch_left = (
                    _side_touch_left(settled, patch, top, bg)
                    if left_side
                    else _side_touch_right(settled, patch, top, bg)
                )
                if touch_left is None:
                    continue
                final_left = int(touch_left + blue_count) if left_side else int(touch_left - blue_count)
                if final_left < 0 or final_left + patch.shape[1] > w:
                    continue

                coords = _patch_to_coords(patch, top, final_left, bg)
                if np.any(global_occ[coords[:, 0], coords[:, 1]]):
                    continue

                wall_only = np.zeros((h, w), dtype=bool)
                wall_only[:, wall_col] = True
                wall_touch = (
                    _side_touch_left(wall_only, patch, top, bg)
                    if left_side
                    else _side_touch_right(wall_only, patch, top, bg)
                )
                blocked_by_object = bool(wall_touch is not None and wall_touch != touch_left)

                if require_chain and idx == count - 1 and not chain_hit and not blocked_by_object:
                    continue

                global_occ[coords[:, 0], coords[:, 1]] = True
                settled[coords[:, 0], coords[:, 1]] = True
                placed.append(
                    {
                        "main_color": int(color),
                        "blue_count": blue_count,
                        "patch": patch,
                        "top": int(top),
                        "final_left": int(final_left),
                        "final_coords": coords,
                        "blocked_by_object": blocked_by_object,
                    }
                )
                chain_hit = chain_hit or blocked_by_object
                break
            else:
                continue
            break
        else:
            return None

    if require_chain and not chain_hit:
        return None
    return placed


def _place_input_side(
    inp: np.ndarray,
    side_items: list[dict],
    existing_coords: list[np.ndarray],
    *,
    left_side: bool,
    bg: int,
) -> bool:
    ordered = sorted(side_items, key=lambda item: item["final_left"] if left_side else -(item["final_left"] + item["patch"].shape[1] - 1))
    width = inp.shape[1]
    placed_coords: list[np.ndarray] = []

    def dfs(i: int, prev_metric: int) -> bool:
        if i == len(ordered):
            return True

        item = ordered[i]
        patch = item["patch"]
        patch_w = patch.shape[1]
        if left_side:
            lo = max(1, item["final_left"] + 1, prev_metric + 1)
            hi = width - patch_w - 1
            candidates = list(range(lo, hi + 1))
            random.shuffle(candidates)
            candidates.sort(reverse=True)
        else:
            lo = 1
            hi = item["final_left"] - 1
            candidates = [left for left in range(lo, hi + 1) if left + patch_w - 1 < prev_metric]
            random.shuffle(candidates)
            candidates.sort()

        for left in candidates:
            coords = _patch_to_coords(patch, item["top"], left, bg)
            if np.any(inp[coords[:, 0], coords[:, 1]] != int(bg)):
                continue
            if any(_pair_touches(coords, other) for other in existing_coords + placed_coords):
                continue

            rr = coords[:, 0] - item["top"]
            cc = coords[:, 1] - left
            values = patch[rr, cc]
            inp[coords[:, 0], coords[:, 1]] = values
            placed_coords.append(coords)
            item["input_coords"] = coords
            item["input_left"] = int(left)

            next_metric = int(left) if left_side else int(left + patch_w - 1)
            if dfs(i + 1, next_metric):
                return True

            inp[coords[:, 0], coords[:, 1]] = int(bg)
            placed_coords.pop()

        return False

    start_metric = 0 if left_side else width - 1
    if not dfs(0, start_metric):
        return False
    existing_coords.extend(placed_coords)
    return True


def _build_input_from_layout(
    h: int,
    w: int,
    bg: int,
    left_color: int,
    right_color: int,
    left_items: list[dict],
    right_items: list[dict],
) -> np.ndarray | None:
    inp = np.full((h, w), int(bg), dtype=int)
    inp[:, 0] = int(left_color)
    inp[:, -1] = int(right_color)
    all_coords: list[np.ndarray] = []

    if not _place_input_side(inp, left_items, all_coords, left_side=True, bg=bg):
        return None
    if not _place_input_side(inp, right_items, all_coords, left_side=False, bg=bg):
        return None
    return inp


def _render_expected(
    h: int,
    w: int,
    bg: int,
    left_color: int,
    right_color: int,
    left_items: list[dict],
    right_items: list[dict],
) -> np.ndarray:
    out = np.full((h, w), int(bg), dtype=int)
    out[:, 0] = int(left_color)
    out[:, -1] = int(right_color)
    for item in left_items + right_items:
        out[item["final_coords"][:, 0], item["final_coords"][:, 1]] = int(item["main_color"])
    return out


def _semantic_flags(left_items: list[dict], right_items: list[dict]) -> tuple[bool, bool, bool]:
    items = left_items + right_items
    has_zero = any(item["blue_count"] == 0 for item in items)
    has_nonzero = any(item["blue_count"] > 0 for item in items)
    has_chain = any(item["blocked_by_object"] for item in items)
    return has_zero, has_nonzero, has_chain


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(18, 28)
    bg = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
    colors = [c for c in PALETTE if c not in (bg, BLUE)]
    left_color, right_color = random.sample(colors, 2)

    left_count = random.randint(2, 3)
    right_count = random.randint(2, 3)

    forced = [0, random.randint(1, 2)]
    random.shuffle(forced)
    force_left = random.choice([False, True])

    global_occ = np.zeros((h, w), dtype=bool)
    global_occ[:, 0] = True
    global_occ[:, -1] = True

    left_items = _place_output_side(
        h,
        w,
        left_color,
        left_count,
        left_side=True,
        require_chain=force_left,
        global_occ=global_occ,
        bg=bg,
        forced_blue_values=forced if force_left else None,
    )
    if left_items is None:
        return None
    right_items = _place_output_side(
        h,
        w,
        right_color,
        right_count,
        left_side=False,
        require_chain=not force_left,
        global_occ=global_occ,
        bg=bg,
        forced_blue_values=forced if not force_left else None,
    )
    if right_items is None:
        return None

    has_zero, has_nonzero, has_chain = _semantic_flags(left_items, right_items)
    if not (has_zero and has_nonzero and has_chain):
        return None

    inp = _build_input_from_layout(h, w, bg, left_color, right_color, left_items, right_items)
    if inp is None:
        return None
    expected = _render_expected(h, w, bg, left_color, right_color, left_items, right_items)

    try:
        objects = _extract_objects(inp, bg=bg, left_color=left_color, right_color=right_color)
    except AssertionError:
        return None
    if len(objects) != left_count + right_count:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    if sum(obj["blue_count"] for obj in objects) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
