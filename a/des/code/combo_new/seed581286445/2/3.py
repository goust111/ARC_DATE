from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
STAMP_RETRIES = 200
GRID_MIN = 12
GRID_MAX = 30
MAX_STAMP_H = 6
MAX_STAMP_W = {0: 6, 1: 6, 2: 6, 3: 5, 4: 5}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size > 0 and cols.size > 0
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components(~mask):
        if any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in comp):
            continue
        holes += 1
    return holes


def _outside_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _grow_outward(mask: np.ndarray, rng: random.Random, max_steps: int) -> np.ndarray:
    canvas = np.zeros((mask.shape[0] + 4, mask.shape[1] + 4), dtype=bool)
    canvas[2 : 2 + mask.shape[0], 2 : 2 + mask.shape[1]] = mask
    for _ in range(rng.randint(0, max_steps)):
        outside = _outside_background(canvas)
        frontier: list[tuple[int, int]] = []
        for r in range(canvas.shape[0]):
            for c in range(canvas.shape[1]):
                if not outside[r, c]:
                    continue
                if any(
                    0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and canvas[nr, nc]
                    for nr, nc in _neighbors4(r, c)
                ):
                    frontier.append((r, c))
        if not frontier:
            break
        rr, cc = rng.choice(frontier)
        canvas[rr, cc] = True
    return _trim(canvas)


def _random_solid_stamp(rng: random.Random, max_w: int) -> np.ndarray:
    for _ in range(STAMP_RETRIES):
        h = rng.randint(1, MAX_STAMP_H)
        w = rng.randint(1, max_w)
        target = rng.randint(1, h * w)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim(mask)
        if rng.random() < 0.7:
            mask = _grow_outward(mask, rng, max_steps=2)
        if len(_components(mask)) != 1:
            continue
        if _count_holes(mask) != 0:
            continue
        if mask.shape[0] > MAX_STAMP_H or mask.shape[1] > max_w:
            continue
        return mask
    raise RuntimeError("failed to build solid stamp")


def _pick_hole_positions(inner_h: int, inner_w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []

    def backtrack(start: int) -> bool:
        if len(chosen) == count:
            return True
        for idx in range(start, len(cells)):
            r, c = cells[idx]
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            if backtrack(idx + 1):
                return True
            chosen.pop()
        return False

    return chosen.copy() if backtrack(0) else None


def _holed_stamp(rng: random.Random, hole_count: int) -> np.ndarray:
    min_side = {1: 3, 2: 4, 3: 5, 4: 5}[hole_count]
    max_w = MAX_STAMP_W[hole_count]
    for _ in range(STAMP_RETRIES):
        h = rng.randint(min_side, MAX_STAMP_H)
        w = rng.randint(min_side, max_w)
        holes = _pick_hole_positions(h - 2, w - 2, hole_count, rng)
        if holes is None:
            continue
        mask = np.ones((h, w), dtype=bool)
        for r, c in holes:
            mask[r + 1, c + 1] = False
        if rng.random() < 0.75:
            mask = _grow_outward(mask, rng, max_steps=2)
        if len(_components(mask)) != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue
        if mask.shape[0] > MAX_STAMP_H or mask.shape[1] > max_w:
            continue
        return mask
    raise RuntimeError(f"failed to build stamp with {hole_count} holes")


def _make_stamp_mask(rng: random.Random, hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return _random_solid_stamp(rng, MAX_STAMP_W[0])
    return _holed_stamp(rng, hole_count)


def _stamp_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _frame_patch(rng: random.Random, color: int, stamp_mask: np.ndarray, hole_count: int) -> tuple[np.ndarray, dict]:
    stamp_h, stamp_w = map(int, stamp_mask.shape)
    if hole_count == 0:
        interior_w = rng.randint(max(2, stamp_w), min(9, max(2, stamp_w + 3)))
    else:
        interior_w = hole_count * stamp_w + (hole_count - 1) + rng.randint(0, 2)
    interior_h = stamp_h
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    patch = np.full((frame_h, frame_w + 1), BG, dtype=int)
    patch[0, 1:] = FRAME
    patch[-1, 1:] = FRAME
    patch[:, 1] = FRAME
    patch[:, -1] = FRAME
    marker_row = rng.randrange(frame_h)
    patch[marker_row, 0] = color
    return patch, {"marker_row": marker_row, "interior_h": interior_h, "interior_w": interior_w}


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if border != {tuple(map(int, pt)) for pt in comp}:
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each colored stamp inside its matching grey frame according to the stamp's hole count.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick hollow rectangles act as blank frames, and each frame has one adjacent colored marker pixel immediately outside its left border.
    - Separate monochrome 4-connected objects of distinct colors act as stamps; a stamp may contain enclosed black holes.

    Rule:
    - For every frame, use its left-side marker color to find the stamp of the same color.
    - Count the number of enclosed black hole components inside that stamp.
    - If the count is N>0, copy the stamp shape, including its holes, into the frame interior N times from left to right, with one black column between neighboring copies and the first copy flush with the frame's interior left edge.
    - If the count is 0, leave that frame interior empty. Keep the original stamps, markers, and grey frames unchanged.

    Key insight:
    - The colored marker does not say where to draw directly; it only pairs a frame with a stamp, and the stamp's topological property, its number of enclosed holes, becomes the repetition count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one frame"

    marker_by_color: dict[int, tuple[int, int]] = {}
    frame_infos: list[dict] = []
    for r0, c0, r1, c1 in frames:
        assert c0 > 0, "frame marker needs one column on the left"
        markers = [(r, c0 - 1) for r in range(r0, r1 + 1) if int(g[r, c0 - 1]) not in (BG, FRAME)]
        assert len(markers) == 1, "each frame must have exactly one left-side marker"
        mr, mc = markers[0]
        color = int(g[mr, mc])
        assert color not in marker_by_color, "marker colors must be unique"
        marker_by_color[color] = (mr, mc)
        frame_infos.append({"bbox": (r0, c0, r1, c1), "color": color})

    stamps: dict[int, dict] = {}
    for color, marker in marker_by_color.items():
        mask = g == color
        mask[marker] = False
        comps = _components(mask)
        assert len(comps) == 1, "each marker color must match exactly one stamp"
        comp = comps[0]
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        stamp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            stamp_mask[int(r) - r0, int(c) - c0] = True
        stamps[color] = {
            "mask": stamp_mask,
            "holes": _count_holes(stamp_mask),
            "h": stamp_mask.shape[0],
            "w": stamp_mask.shape[1],
        }

    for frame in frame_infos:
        color = frame["color"]
        stamp = stamps[color]
        copies = int(stamp["holes"])
        if copies == 0:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        inner_top = r0 + 1
        inner_left = c0 + 1
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        stamp_h = int(stamp["h"])
        stamp_w = int(stamp["w"])
        need_w = copies * stamp_w + (copies - 1)
        assert stamp_h <= inner_h and need_w <= inner_w
        for k in range(copies):
            base_c = inner_left + k * (stamp_w + 1)
            for rr, cc in np.argwhere(stamp["mask"]):
                out[inner_top + int(rr), base_c + int(cc)] = color

    return out


def _pair_hole_counts(rng: random.Random, pair_count: int) -> list[int]:
    if pair_count == 2:
        counts = [0, rng.randint(1, 4)]
    else:
        counts = [0, rng.randint(1, 3), rng.randint(0, 3)]
    rng.shuffle(counts)
    return counts


def _pick_grid_shape(rng: random.Random, items: list[dict]) -> tuple[int, int] | None:
    max_h = max(int(item["patch"].shape[0]) for item in items)
    max_w = max(int(item["patch"].shape[1]) for item in items)
    if max_h > GRID_MAX or max_w > GRID_MAX:
        return None
    required_area = sum((item["patch"].shape[0] + 2) * (item["patch"].shape[1] + 2) for item in items)
    min_h = max(GRID_MIN, max_h + 1)
    min_w = max(GRID_MIN, max_w + 1)
    for _ in range(80):
        h = rng.randint(min_h, GRID_MAX)
        w = rng.randint(min_w, GRID_MAX)
        if h * w >= required_area:
            return h, w
    return None


def _place_items(rng: random.Random, shape: tuple[int, int], items: list[dict]) -> tuple[np.ndarray, dict] | None:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placements: dict[tuple[str, int], tuple[int, int]] = {}

    ordered = sorted(
        items,
        key=lambda item: int(item["patch"].shape[0]) * int(item["patch"].shape[1]),
        reverse=True,
    )
    for item in ordered:
        patch = item["patch"]
        ph, pw = map(int, patch.shape)
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            painted = patch != BG
            if np.any(grid[r0 : r0 + ph, c0 : c0 + pw][painted] != BG):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][painted] = patch[painted]
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            placements[item["key"]] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return grid, placements


def _expected_output(grid: np.ndarray, pairs: list[dict], placements: dict[tuple[str, int], tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for idx, pair in enumerate(pairs):
        copies = int(pair["holes"])
        if copies == 0:
            continue
        frame_r, frame_c = placements[("frame", idx)]
        stamp_mask = pair["stamp_mask"]
        stamp_w = int(stamp_mask.shape[1])
        inner_top = frame_r + 1
        inner_left = frame_c + 2
        for k in range(copies):
            base_c = inner_left + k * (stamp_w + 1)
            for rr, cc in np.argwhere(stamp_mask):
                out[inner_top + int(rr), base_c + int(cc)] = int(pair["color"])
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = rng.choice((2, 3))
    colors = list(STAMP_COLORS)
    rng.shuffle(colors)
    hole_counts = _pair_hole_counts(rng, pair_count)

    pairs: list[dict] = []
    items: list[dict] = []
    for idx in range(pair_count):
        color = colors[idx]
        holes = hole_counts[idx]
        stamp_mask = _make_stamp_mask(rng, holes)
        stamp_patch = _stamp_patch(stamp_mask, color)
        frame_patch, frame_info = _frame_patch(rng, color, stamp_mask, holes)
        pairs.append(
            {
                "color": color,
                "holes": holes,
                "stamp_mask": stamp_mask,
                "frame_info": frame_info,
            }
        )
        items.append({"key": ("stamp", idx), "patch": stamp_patch})
        items.append({"key": ("frame", idx), "patch": frame_patch})

    shape = _pick_grid_shape(rng, items)
    if shape is None:
        return None
    placed = _place_items(rng, shape, items)
    if placed is None:
        return None
    grid, placements = placed
    expected = _expected_output(grid, pairs, placements)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _gen_attempt(rng)
        if built is None:
            continue
        grid, expected = built
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-frame task")
