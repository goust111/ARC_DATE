from __future__ import annotations

import random

import numpy as np


_BG = 0
_FRAME = 5
_SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_DIRECTIONS = ("up", "down", "left", "right")
_GEN_COUNTER = 0
_MAX_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _gap_run(line: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~line)
    if missing.size == 0:
        return None
    if missing[0] == 0 or missing[-1] == len(line) - 1:
        return (-1, -1)
    if int(missing[-1] - missing[0] + 1) != int(missing.size):
        return (-1, -1)
    return int(missing[0]), int(missing[-1])


def _frame_info(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros_like(local)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(local & ~border):
        return None

    sides = {
        "up": _gap_run(local[0, :]),
        "down": _gap_run(local[-1, :]),
        "left": _gap_run(local[:, 0]),
        "right": _gap_run(local[:, -1]),
    }
    if any(run == (-1, -1) for run in sides.values()):
        return None

    missing = [(name, run) for name, run in sides.items() if run is not None]
    if len(missing) != 1:
        return None

    direction, run = missing[0]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any(interior != bg):
        return None

    if direction in {"up", "down"}:
        gap = (c0 + run[0], c0 + run[1])
    else:
        gap = (r0 + run[0], r0 + run[1])

    return {
        "direction": direction,
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
    }


def _paste_non_bg(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    h, w = out.shape
    for pr in range(ph):
        rr = top + pr
        if not (0 <= rr < h):
            continue
        for pc in range(pw):
            cc = left + pc
            if not (0 <= cc < w):
                continue
            value = int(patch[pr, pc])
            if value != bg:
                out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each frame's seed pattern out through the frame gap.

    The gap side gives the copy direction. The number of empty rows or columns
    between the seed and the gap gives the copy count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    out = g.copy()
    colors = sorted(set(map(int, g.flatten())) - {bg})

    frames: list[dict] = []
    for color in colors:
        for coords in _components(g == color):
            info = _frame_info(g, coords, bg)
            if info is not None:
                frames.append(info)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        direction = frame["direction"]
        gap0, gap1 = frame["gap"]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_cells = np.argwhere(interior != bg)
        assert seed_cells.size > 0, "each frame interior must contain a seed"

        sr0, sc0 = np.min(seed_cells, axis=0)
        sr1, sc1 = np.max(seed_cells, axis=0)
        sr0 = int(sr0)
        sc0 = int(sc0)
        sr1 = int(sr1)
        sc1 = int(sc1)

        patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        seed_r0 = r0 + 1 + sr0
        seed_c0 = c0 + 1 + sc0
        seed_r1 = r0 + 1 + sr1
        seed_c1 = c0 + 1 + sc1
        patch_h, patch_w = patch.shape

        if direction == "up":
            assert seed_r1 == r1 - 1
            assert (seed_c0, seed_c1) == (gap0, gap1)
            copies = seed_r0 - (r0 + 1)
            for step in range(copies):
                _paste_non_bg(out, patch, r0 - (step + 1) * patch_h, seed_c0, bg)
        elif direction == "down":
            assert seed_r0 == r0 + 1
            assert (seed_c0, seed_c1) == (gap0, gap1)
            copies = (r1 - 1) - seed_r1
            for step in range(copies):
                _paste_non_bg(out, patch, r1 + 1 + step * patch_h, seed_c0, bg)
        elif direction == "left":
            assert seed_c1 == c1 - 1
            assert (seed_r0, seed_r1) == (gap0, gap1)
            copies = seed_c0 - (c0 + 1)
            for step in range(copies):
                _paste_non_bg(out, patch, seed_r0, c0 - (step + 1) * patch_w, bg)
        else:
            assert seed_c0 == c0 + 1
            assert (seed_r0, seed_r1) == (gap0, gap1)
            copies = (c1 - 1) - seed_c1
            for step in range(copies):
                _paste_non_bg(out, patch, seed_r0, c1 + 1 + step * patch_w, bg)

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_components(mask.astype(bool))) == 1


def _random_seed_patch(height: int, width: int) -> np.ndarray:
    for _ in range(200):
        density = random.uniform(0.45, 0.9)
        mask = np.random.random((height, width)) < density
        if not np.any(mask):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        if not _mask_connected(mask):
            continue
        if height * width > 1 and np.all(mask) and random.random() < 0.5:
            continue

        colors = random.sample(_SEED_COLORS, k=random.randint(1, min(3, int(mask.sum()))))
        patch = np.zeros((height, width), dtype=int)
        cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        assignments = [random.choice(colors) for _ in cells]
        for idx, color in enumerate(colors):
            assignments[idx] = color
        random.shuffle(assignments)
        for (r, c), color in zip(cells, assignments, strict=True):
            patch[r, c] = int(color)
        return patch

    patch = np.zeros((height, width), dtype=int)
    patch[:, :] = random.choice(_SEED_COLORS)
    return patch


def _make_spec(direction: str) -> dict:
    seed_h = random.randint(1, 3)
    seed_w = random.randint(1, 3)
    copies = random.randint(1, 3)
    patch = _random_seed_patch(seed_h, seed_w)

    if direction in {"up", "down"}:
        left_pad = random.randint(0, 2)
        right_pad = random.randint(0, 2)
        interior_h = seed_h + copies
        interior_w = left_pad + seed_w + right_pad
        frame_h = interior_h + 2
        frame_w = interior_w + 2
        seed_r = 1 + (copies if direction == "up" else 0)
        seed_c = 1 + left_pad
        reserve = (-copies * seed_h if direction == "up" else 0, 0,
                   frame_h - 1 + (copies * seed_h if direction == "down" else 0), frame_w - 1)
    else:
        top_pad = random.randint(0, 2)
        bottom_pad = random.randint(0, 2)
        interior_h = top_pad + seed_h + bottom_pad
        interior_w = seed_w + copies
        frame_h = interior_h + 2
        frame_w = interior_w + 2
        seed_r = 1 + top_pad
        seed_c = 1 + (copies if direction == "left" else 0)
        reserve = (0, -copies * seed_w if direction == "left" else 0,
                   frame_h - 1, frame_w - 1 + (copies * seed_w if direction == "right" else 0))

    return {
        "direction": direction,
        "patch": patch,
        "seed_h": seed_h,
        "seed_w": seed_w,
        "copies": copies,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "seed_r": seed_r,
        "seed_c": seed_c,
        "reserve": reserve,
    }


def _candidate_positions(spec: dict, used: np.ndarray) -> list[tuple[int, int]]:
    h, w = used.shape
    rr0, cc0, rr1, cc1 = spec["reserve"]
    row_min = 1 - rr0
    row_max = h - 2 - rr1
    col_min = 1 - cc0
    col_max = w - 2 - cc1
    if row_min > row_max or col_min > col_max:
        return []

    positions: list[tuple[int, int]] = []
    for r0 in range(row_min, row_max + 1):
        ur0 = r0 + rr0 - 1
        ur1 = r0 + rr1 + 1
        for c0 in range(col_min, col_max + 1):
            uc0 = c0 + cc0 - 1
            uc1 = c0 + cc1 + 1
            if not np.any(used[ur0 : ur1 + 1, uc0 : uc1 + 1]):
                positions.append((r0, c0))
    return positions


def _draw_frame(grid: np.ndarray, r0: int, c0: int, spec: dict) -> None:
    frame_h = spec["frame_h"]
    frame_w = spec["frame_w"]
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    grid[r0 : r1 + 1, c0] = _FRAME
    grid[r0 : r1 + 1, c1] = _FRAME
    grid[r0, c0 : c1 + 1] = _FRAME
    grid[r1, c0 : c1 + 1] = _FRAME

    seed_r = spec["seed_r"]
    seed_c = spec["seed_c"]
    seed_h = spec["seed_h"]
    seed_w = spec["seed_w"]
    direction = spec["direction"]

    if direction == "up":
        grid[r0, c0 + seed_c : c0 + seed_c + seed_w] = _BG
    elif direction == "down":
        grid[r1, c0 + seed_c : c0 + seed_c + seed_w] = _BG
    elif direction == "left":
        grid[r0 + seed_r : r0 + seed_r + seed_h, c0] = _BG
    else:
        grid[r0 + seed_r : r0 + seed_r + seed_h, c1] = _BG


def _render_pair(shape: tuple[int, int], placements: list[tuple[dict, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, _BG, dtype=int)
    out = np.full(shape, _BG, dtype=int)

    for spec, r0, c0 in placements:
        _draw_frame(inp, r0, c0, spec)
        _draw_frame(out, r0, c0, spec)

        seed_top = r0 + spec["seed_r"]
        seed_left = c0 + spec["seed_c"]
        patch = spec["patch"]
        _paste_non_bg(inp, patch, seed_top, seed_left, _BG)
        _paste_non_bg(out, patch, seed_top, seed_left, _BG)

        for step in range(spec["copies"]):
            if spec["direction"] == "up":
                _paste_non_bg(out, patch, r0 - (step + 1) * spec["seed_h"], seed_left, _BG)
            elif spec["direction"] == "down":
                _paste_non_bg(out, patch, r0 + spec["frame_h"] + step * spec["seed_h"], seed_left, _BG)
            elif spec["direction"] == "left":
                _paste_non_bg(out, patch, seed_top, c0 - (step + 1) * spec["seed_w"], _BG)
            else:
                _paste_non_bg(out, patch, seed_top, c0 + spec["frame_w"] + step * spec["seed_w"], _BG)

    return inp, out


def _sample_directions(count: int) -> list[str]:
    global _GEN_COUNTER
    primary = _DIRECTIONS[_GEN_COUNTER % len(_DIRECTIONS)]
    _GEN_COUNTER += 1

    remaining = [d for d in _DIRECTIONS if d != primary]
    random.shuffle(remaining)
    dirs = [primary] + remaining[: max(0, count - 1)]
    while len(dirs) < count:
        dirs.append(random.choice(_DIRECTIONS))
    random.shuffle(dirs)
    return dirs


def _generate_pair() -> tuple[np.ndarray, np.ndarray]:
    rows = random.randint(18, 30)
    cols = random.randint(18, 30)
    frame_count = random.randint(1, 4)
    specs = [_make_spec(direction) for direction in _sample_directions(frame_count)]
    specs.sort(key=lambda spec: (spec["frame_h"] * spec["frame_w"], spec["copies"]), reverse=True)

    used = np.zeros((rows, cols), dtype=bool)
    placements: list[tuple[dict, int, int]] = []
    for spec in specs:
        positions = _candidate_positions(spec, used)
        if not positions:
            raise ValueError("placement failed")
        r0, c0 = random.choice(positions)
        rr0, cc0, rr1, cc1 = spec["reserve"]
        used[r0 + rr0 - 1 : r0 + rr1 + 2, c0 + cc0 - 1 : c0 + cc1 + 2] = True
        placements.append((spec, r0, c0))

    return _render_pair((rows, cols), placements)


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        try:
            inp, out = _generate_pair()
        except ValueError:
            continue
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid sample")
