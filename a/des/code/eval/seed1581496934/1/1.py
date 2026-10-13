from __future__ import annotations

import random

import numpy as np


BG = 0
ROOM = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_room_color(grid: np.ndarray, bg: int) -> int:
    vals = grid[grid != bg]
    assert vals.size > 0, "grid must contain non-background pixels"
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[np.argmax(counts)])


def _best_legend_run(groups: dict[tuple[int, int], list[tuple[int, int]]]) -> list[tuple[int, int]]:
    best_key = (-1, -1)
    best_run: list[tuple[int, int]] = []
    for (c0, c1), entries in groups.items():
        cur: list[tuple[int, int]] = []
        for item in sorted(entries):
            if not cur or item[0] - cur[-1][0] == 2:
                cur.append(item)
            else:
                if (len(cur), c1) > best_key:
                    best_key = (len(cur), c1)
                    best_run = cur[:]
                cur = [item]
        if (len(cur), c1) > best_key:
            best_key = (len(cur), c1)
            best_run = cur[:]
    return best_run


def _find_legend(grid: np.ndarray, bg: int, room_color: int) -> list[int]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    mask = (grid != bg) & (grid != room_color)
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape != (1, 3):
            continue
        vals = np.unique(patch)
        if vals.size != 1:
            continue
        groups.setdefault((c0, c1), []).append((r0, int(vals[0])))
    assert groups, "legend not found"
    run = _best_legend_run(groups)
    assert run, "legend run not found"
    return [color for _, color in sorted(run)]


def _is_room_patch(patch: np.ndarray, room_color: int, bg: int) -> bool:
    h, w = patch.shape
    if h < 4 or w < 4:
        return False
    if not np.all(patch[0, :] == room_color):
        return False
    if not np.all(patch[-1, :] == room_color):
        return False
    if not np.all(patch[:, 0] == room_color):
        return False
    if not np.all(patch[:, -1] == room_color):
        return False
    return set(map(int, np.unique(patch))) <= {bg, room_color}


def _count_windows(patch: np.ndarray, bg: int) -> int:
    count = 0
    for pts in _components(patch[1:-1, 1:-1] == bg):
        r0, c0, r1, c1 = _bbox(pts)
        if len(pts) == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            count += 1
    return count


def _paint_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _find_bg(g)
    room_color = _find_room_color(g, bg)
    legend = _find_legend(g, bg, room_color)

    out = g.copy()
    for pts in _components(g == room_color):
        r0, c0, r1, c1 = _bbox(pts)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not _is_room_patch(patch, room_color, bg):
            continue
        window_count = _count_windows(patch, bg)
        assert 1 <= window_count <= len(legend), "window count exceeds legend"
        _paint_border(out, r0, c0, r1, c1, legend[window_count - 1])
    return out


def _composition(total: int, parts: int, rng: random.Random) -> list[int]:
    vals = [0] * parts
    for _ in range(total):
        vals[rng.randrange(parts)] += 1
    return vals


def _room_slot_choices(window_count: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for slot_rows in range(1, 4):
        for slot_cols in range(1, 4):
            capacity = slot_rows * slot_cols
            if capacity < window_count:
                continue
            if capacity - window_count > 2:
                continue
            choices.append((slot_rows, slot_cols))
    assert choices
    return choices


def _build_room(window_count: int, rng: random.Random) -> np.ndarray:
    slot_rows, slot_cols = rng.choice(_room_slot_choices(window_count))
    extra_h = rng.randint(0, 3)
    extra_w = rng.randint(0, 4)
    row_gaps = [1 + v for v in _composition(extra_h, slot_rows + 1, rng)]
    col_gaps = [1 + v for v in _composition(extra_w, slot_cols + 1, rng)]

    ih = 2 * slot_rows + sum(row_gaps)
    iw = 2 * slot_cols + sum(col_gaps)
    patch = np.full((ih + 2, iw + 2), ROOM, dtype=int)

    slots: list[tuple[int, int]] = []
    rr = row_gaps[0]
    for slot_r in range(slot_rows):
        cc = col_gaps[0]
        for slot_c in range(slot_cols):
            slots.append((rr, cc))
            if slot_c + 1 < slot_cols:
                cc += 2 + col_gaps[slot_c + 1]
        if slot_r + 1 < slot_rows:
            rr += 2 + row_gaps[slot_r + 1]

    for wr, wc in rng.sample(slots, window_count):
        patch[1 + wr : 1 + wr + 2, 1 + wc : 1 + wc + 2] = BG

    assert _count_windows(patch, BG) == window_count
    return patch


def _sample_window_counts(room_count: int, legend_len: int, rng: random.Random) -> list[int]:
    counts = list(range(1, legend_len + 1))
    rng.shuffle(counts)
    counts = counts[: min(room_count, legend_len)]
    while len(counts) < room_count:
        counts.append(rng.randint(1, legend_len))
    rng.shuffle(counts)
    return counts


def _place_rooms(
    room_patches: list[np.ndarray], height: int, width: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    occ = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(room_patches)
    order = sorted(
        range(len(room_patches)),
        key=lambda i: room_patches[i].shape[0] * room_patches[i].shape[1],
        reverse=True,
    )
    for idx in order:
        patch = room_patches[idx]
        ph, pw = patch.shape
        if height - ph - 2 < 1 or width - pw - 5 < 1:
            return None
        for _ in range(300):
            r0 = rng.randint(1, height - ph - 1)
            c0 = rng.randint(1, width - pw - 4)
            halo = occ[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]
            if halo.any():
                continue
            occ[r0 : r0 + ph, c0 : c0 + pw] = True
            positions[idx] = (r0, c0)
            break
        if positions[idx] is None:
            return None
    return [(r, c) for r, c in positions if r is not None and c is not None]


def _expected_output(
    grid: np.ndarray, room_patches: list[np.ndarray], positions: list[tuple[int, int]], counts: list[int], legend: list[int]
) -> np.ndarray:
    out = grid.copy()
    for patch, (r0, c0), count in zip(room_patches, positions, counts):
        ph, pw = patch.shape
        _paint_border(out, r0, c0, r0 + ph - 1, c0 + pw - 1, legend[count - 1])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_len = rng.randint(3, 5)
    legend = rng.sample(list(COLOR_POOL), legend_len)
    room_count = rng.randint(2, min(5, legend_len + 1))
    counts = _sample_window_counts(room_count, legend_len, rng)
    room_patches = [_build_room(count, rng) for count in counts]

    legend_h = 2 * legend_len - 1
    max_room_h = max(p.shape[0] for p in room_patches)
    max_room_w = max(p.shape[1] for p in room_patches)
    total_room_h = sum(p.shape[0] for p in room_patches)
    total_room_w = sum(p.shape[1] for p in room_patches)

    height = max(legend_h + 2, max_room_h + 2, total_room_h // 2 + 3) + rng.randint(2, 8)
    width = max(max_room_w + 6, total_room_w // 2 + 7) + rng.randint(2, 8)
    if height > 30 or width > 36:
        return None

    positions = _place_rooms(room_patches, height, width, rng)
    if positions is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    legend_r0 = rng.randint(1, height - legend_h - 1)
    legend_c0 = width - 3
    for idx, color in enumerate(legend):
        grid[legend_r0 + 2 * idx, legend_c0 : legend_c0 + 3] = color

    for patch, (r0, c0) in zip(room_patches, positions):
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    if int(np.sum(grid == BG)) <= int(np.sum(grid == ROOM)):
        return None
    if _find_bg(grid) != BG or _find_room_color(grid, BG) != ROOM:
        return None
    if _find_legend(grid, BG, ROOM) != legend:
        return None

    expected = _expected_output(grid, room_patches, positions, counts, legend)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
