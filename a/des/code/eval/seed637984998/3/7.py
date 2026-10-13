from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
RED = 2
FULL_SIZES = (5, 6)  # border included; interiors are 3x3 or 4x4
FRAME_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _corner_for_turns(size: int, turns_cw: int) -> tuple[int, int]:
    corners = (
        (0, 0),
        (0, size - 1),
        (size - 1, size - 1),
        (size - 1, 0),
    )
    return corners[int(turns_cw) % 4]


def _has_clear_margin(grid: np.ndarray, top: int, left: int, size: int, bg: int) -> bool:
    h, w = grid.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + size + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + size + 1)
    around = grid[r0:r1, c0:c1]
    outside = np.ones_like(around, dtype=bool)
    outside[top - r0 : top - r0 + size, left - c0 : left - c0 + size] = False
    return not np.any(around[outside] != bg)


def _classify_patch(patch: np.ndarray) -> dict | None:
    size = int(patch.shape[0])
    if patch.shape != (size, size):
        return None

    border = patch[_border_mask(size)]
    interior = patch[1:-1, 1:-1]

    if np.all(interior == BG):
        corners = [_corner_for_turns(size, turns) for turns in range(4)]
        red_turns = [turns for turns, (r, c) in enumerate(corners) if int(patch[r, c]) == RED]
        if len(red_turns) != 1:
            return None
        if int(np.count_nonzero(border == RED)) != 1:
            return None
        frame_vals = [int(v) for v in border.tolist() if int(v) != RED]
        if not frame_vals or len(set(frame_vals)) != 1:
            return None
        frame_color = int(frame_vals[0])
        if frame_color in (BG, RED):
            return None
        return {
            "kind": "canvas",
            "frame_color": frame_color,
            "turns_cw": int(red_turns[0]),
            "inner": interior.copy(),
        }

    border_vals = set(map(int, np.unique(border)))
    if len(border_vals) != 1:
        return None
    frame_color = int(next(iter(border_vals)))
    if frame_color in (BG, RED):
        return None

    inner_colors = set(map(int, np.unique(interior))) - {BG}
    if len(inner_colors) != 1:
        return None
    pattern_color = int(next(iter(inner_colors)))
    if pattern_color == frame_color:
        return None

    return {
        "kind": "stamp",
        "frame_color": frame_color,
        "pattern_color": pattern_color,
        "inner": interior.copy(),
    }


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    objects: list[dict] = []
    for size in FULL_SIZES:
        for top in range(h - size + 1):
            for left in range(w - size + 1):
                patch = grid[top : top + size, left : left + size]
                if np.all(patch == BG):
                    continue
                if not _has_clear_margin(grid, top, left, size, BG):
                    continue
                obj = _classify_patch(patch)
                if obj is None:
                    continue
                obj["top"] = int(top)
                obj["left"] = int(left)
                obj["size"] = int(size)
                objects.append(obj)
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each framed stamp with the canvas that shares its border color, decode
    rotation from the canvas's red corner marker, and stamp the rotated interior
    pattern into the canvas while restoring the border corner color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    objects = _find_objects(g)
    assert objects, "no framed objects found"

    stamps: dict[int, list[dict]] = defaultdict(list)
    canvases: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        if obj["kind"] == "stamp":
            stamps[int(obj["frame_color"])].append(obj)
        else:
            canvases[int(obj["frame_color"])].append(obj)

    colors = sorted(set(stamps) | set(canvases))
    assert colors, "no pairable frame colors found"

    for color in colors:
        assert len(stamps[color]) == 1, f"expected one stamp for frame color {color}"
        assert len(canvases[color]) == 1, f"expected one canvas for frame color {color}"
        stamp = stamps[color][0]
        canvas = canvases[color][0]
        assert int(stamp["size"]) == int(canvas["size"])

        rotated = np.rot90(stamp["inner"], -int(canvas["turns_cw"]))
        top = int(canvas["top"])
        left = int(canvas["left"])
        size = int(canvas["size"])

        patch = out[top : top + size, left : left + size]
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        patch[1:-1, 1:-1] = rotated

    return out


def _neighbors4(r: int, c: int, size: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < size:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < size:
        yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    size = int(mask.shape[0])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, size):
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rotations_are_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, turns)) for turns in (1, 2, 3))


def _make_pattern(inner_size: int, rng: random.Random) -> np.ndarray | None:
    area = inner_size * inner_size
    for _ in range(200):
        target = rng.randint(3, area - 2)
        mask = np.zeros((inner_size, inner_size), dtype=bool)
        start = (rng.randrange(inner_size), rng.randrange(inner_size))
        chosen = {start}
        frontier = set(_neighbors4(start[0], start[1], inner_size))

        while len(chosen) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            for nb in _neighbors4(cell[0], cell[1], inner_size):
                if nb not in chosen:
                    frontier.add(nb)

        if len(chosen) != target:
            continue

        for r, c in chosen:
            mask[r, c] = True

        if not _is_connected(mask):
            continue
        if not _rotations_are_distinct(mask):
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) < 2 or len(cols) < 2:
            continue
        return mask
    return None


def _make_stamp_patch(frame_color: int, pattern_color: int, pattern_mask: np.ndarray) -> np.ndarray:
    inner_size = int(pattern_mask.shape[0])
    size = inner_size + 2
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1:-1, 1:-1][pattern_mask] = pattern_color
    return patch


def _make_canvas_patch(frame_color: int, inner_size: int, turns_cw: int) -> np.ndarray:
    size = inner_size + 2
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    rr, cc = _corner_for_turns(size, turns_cw)
    patch[rr, cc] = RED
    return patch


def _free_positions(occupied: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    for top in range(h - size + 1):
        for left in range(w - size + 1):
            r0 = max(0, top - 1)
            r1 = min(h, top + size + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + size + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            positions.append((top, left))
    return positions


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    size = int(patch.shape[0])
    grid[top : top + size, left : left + size] = patch
    occupied[top : top + size, left : left + size] = True


def _expected_output(grid: np.ndarray, pair_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in pair_specs:
        inner = spec["stamp_patch"][1:-1, 1:-1]
        rotated = np.rot90(inner, -int(spec["turns_cw"]))
        top = int(spec["canvas_top"])
        left = int(spec["canvas_left"])
        size = int(spec["size"])
        frame_color = int(spec["frame_color"])

        patch = out[top : top + size, left : left + size]
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color
        patch[1:-1, 1:-1] = rotated
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 4)
    frame_colors = rng.sample(FRAME_COLORS, pair_count)

    pair_specs = []
    for frame_color in frame_colors:
        inner_size = rng.choice((3, 4))
        pattern_mask = _make_pattern(inner_size, rng)
        if pattern_mask is None:
            return None

        pattern_choices = [c for c in FRAME_COLORS if c != frame_color]
        pattern_color = rng.choice(pattern_choices)
        turns_cw = rng.randrange(4)

        stamp_patch = _make_stamp_patch(frame_color, pattern_color, pattern_mask)
        canvas_patch = _make_canvas_patch(frame_color, inner_size, turns_cw)
        pair_specs.append(
            {
                "frame_color": int(frame_color),
                "turns_cw": int(turns_cw),
                "size": int(inner_size + 2),
                "stamp_patch": stamp_patch,
                "canvas_patch": canvas_patch,
            }
        )

    max_size = max(int(spec["size"]) for spec in pair_specs)
    min_h = max(18, max_size + 4, 10 + pair_count * 2)
    min_w = max(18, max_size + 4, 11 + pair_count * 3)
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    objects = []
    for idx, spec in enumerate(pair_specs):
        objects.append((idx, "stamp", spec["stamp_patch"]))
        objects.append((idx, "canvas", spec["canvas_patch"]))
    rng.shuffle(objects)
    objects.sort(key=lambda item: -int(item[2].shape[0]))

    for pair_idx, role, patch in objects:
        size = int(patch.shape[0])
        positions = _free_positions(occupied, size)
        if not positions:
            return None
        top, left = rng.choice(positions)
        _place_patch(grid, occupied, patch, top, left)
        pair_specs[pair_idx][f"{role}_top"] = int(top)
        pair_specs[pair_idx][f"{role}_left"] = int(left)

    expected = _expected_output(grid, pair_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
