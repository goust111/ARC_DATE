from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _palette_from_row(grid: np.ndarray, row: int, bg: int) -> list[int] | None:
    h, w = grid.shape
    cols = [c for c in range(w) if int(grid[row, c]) != bg]
    if len(cols) < 2:
        return None

    colors: list[int] = []
    prev = -10
    for c in cols:
        color = int(grid[row, c])
        if color in colors:
            return None
        if c - prev == 1:
            return None
        if c > 0 and int(grid[row, c - 1]) != bg:
            return None
        if c + 1 < w and int(grid[row, c + 1]) != bg:
            return None
        if row > 0 and int(grid[row - 1, c]) != bg:
            return None
        if row + 1 < h and int(grid[row + 1, c]) != bg:
            return None
        colors.append(color)
        prev = c
    return colors


def _find_palette(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    candidates: list[tuple[int, list[int]]] = []
    for row in (0, grid.shape[0] - 1):
        palette = _palette_from_row(grid, row, bg)
        if palette is not None:
            candidates.append((row, palette))
    if not candidates:
        raise ValueError("no palette row found on the top or bottom edge")
    candidates.sort(key=lambda item: (len(item[1]), -item[0]), reverse=True)
    return candidates[0]


def _frame_from_component(
    grid: np.ndarray, points: list[tuple[int, int]], color: int
) -> dict[str, int] | None:
    if len(points) < 8:
        return None
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(points) != 2 * h + 2 * w - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not bool((sub[border] == int(color)).all()):
        return None
    if h > 2 and w > 2 and bool((sub[1:-1, 1:-1] == int(color)).any()):
        return None

    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": int(color)}


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for color in sorted({int(v) for v in grid.flat if int(v) != bg}):
        for comp in _components(grid == int(color)):
            frame = _frame_from_component(grid, comp, int(color))
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow rectangle border by indexing a palette with its interior dot count.

    Input structure:
    - bg is the most common color.
    - The top or bottom edge contains a palette row of isolated 1x1 colored cells separated by bg.
    - Elsewhere there are 1-pixel-thick monochrome rectangular frames, each enclosing several non-bg single-pixel markers.

    Rule:
    - Read the palette colors from left to right.
    - For each rectangular frame, count all interior pixels that are neither bg nor the frame color.
    - Use that count N as a 1-based palette index, then repaint the entire frame border with the Nth palette color.
    - Keep the palette row and all interior marker pixels unchanged.

    Key insight:
    - The palette row is an ordered lookup table, and each frame's interior marker count is the lookup key.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    _palette_row, palette = _find_palette(g, bg)
    frames = _find_frames(g, bg)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        frame_color = int(frame["color"])
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero((inside != bg) & (inside != frame_color)))
        if not (1 <= count <= len(palette)):
            continue
        new_color = int(palette[count - 1])
        out[r0, c0 : c1 + 1] = new_color
        out[r1, c0 : c1 + 1] = new_color
        out[r0 : r1 + 1, c0] = new_color
        out[r0 : r1 + 1, c1] = new_color
    return out


def _separated_slots(h: int, w: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for row_offset in (0, 1):
        for col_offset in (0, 1):
            pts = [
                (r, c)
                for r in range(row_offset, h, 2)
                for c in range(col_offset, w, 2)
            ]
            if pts:
                groups.append(pts)
    return groups


def _sample_palette_positions(rng: random.Random, width: int, length: int) -> list[int]:
    while True:
        gaps = [rng.randint(1, 3) for _ in range(length - 1)]
        span = length + sum(gaps)
        if span <= width:
            break
    start = rng.randint(0, width - span)
    cols = [start]
    cur = start
    for gap in gaps:
        cur += 1 + gap
        cols.append(cur)
    return cols


def _sample_dot_positions(
    rng: random.Random, inner_h: int, inner_w: int, count: int
) -> list[tuple[int, int]]:
    groups = [pts for pts in _separated_slots(inner_h, inner_w) if len(pts) >= count]
    if not groups:
        raise ValueError("not enough separated slots for requested count")
    chosen = list(rng.choice(groups))
    rng.shuffle(chosen)
    return chosen[:count]


def _candidate_inner_sizes(max_count: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for inner_h in range(1, 7):
        for inner_w in range(1, 7):
            if max(len(group) for group in _separated_slots(inner_h, inner_w)) >= max_count:
                sizes.append((inner_h, inner_w))
    return sizes


def _add_halo(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 30)
    w = rng.randint(12, 30)
    bg = rng.choice(ALL_COLORS)

    max_palette_len = min(7, (w + 1) // 2)
    if max_palette_len < 3:
        return None
    palette_len = rng.randint(3, max_palette_len)

    non_bg_colors = [color for color in ALL_COLORS if color != bg]
    rng.shuffle(non_bg_colors)
    palette = non_bg_colors[:palette_len]
    border_color = non_bg_colors[palette_len]
    marker_colors = [color for color in ALL_COLORS if color not in (bg, border_color)]

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    palette_row = 0 if rng.random() < 0.5 else h - 1
    palette_cols = _sample_palette_positions(rng, w, palette_len)
    for c, color in zip(palette_cols, palette):
        grid[palette_row, c] = int(color)
    blocked[palette_row, :] = True
    if palette_row == 0 and h > 1:
        blocked[1, :] = True
    if palette_row == h - 1 and h > 1:
        blocked[h - 2, :] = True

    usable_area = max(1, (h - 4) * (w - 4))
    max_frames = min(5, max(1, usable_area // 60))
    frame_count = rng.randint(1, max_frames)

    counts = [rng.randint(1, palette_len) for _ in range(frame_count)]
    if frame_count > 1 and palette_len > 1 and len(set(counts)) == 1:
        counts[0] = 1 if counts[0] != 1 else palette_len

    placed_bboxes: list[tuple[int, int, int, int]] = []
    size_cache = {count: _candidate_inner_sizes(count) for count in set(counts)}

    for count in counts:
        sizes = list(size_cache[count])
        rng.shuffle(sizes)
        placed = False
        for inner_h, inner_w in sizes:
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            if h - frame_h - 2 < 1 or w - frame_w - 2 < 1:
                continue
            for _ in range(120):
                r0 = rng.randint(1, h - frame_h - 1)
                c0 = rng.randint(1, w - frame_w - 1)
                r1 = r0 + frame_h - 1
                c1 = c0 + frame_w - 1
                halo = blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)]
                if bool(halo.any()):
                    continue

                grid[r0, c0 : c1 + 1] = int(border_color)
                grid[r1, c0 : c1 + 1] = int(border_color)
                grid[r0 : r1 + 1, c0] = int(border_color)
                grid[r0 : r1 + 1, c1] = int(border_color)

                for dr, dc in _sample_dot_positions(rng, inner_h, inner_w, count):
                    grid[r0 + 1 + dr, c0 + 1 + dc] = int(rng.choice(marker_colors))

                _add_halo(blocked, r0, c0, r1, c1)
                placed_bboxes.append((r0, c0, r1, c1))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if _get_bg(grid) != bg:
        return None

    try:
        detected_row, detected_palette = _find_palette(grid, bg)
        frames = _find_frames(grid, bg)
    except Exception:
        return None

    if detected_row != palette_row or detected_palette != palette:
        return None

    detected_bboxes = {(f["r0"], f["c0"], f["r1"], f["c1"]) for f in frames}
    if detected_bboxes != set(placed_bboxes):
        return None

    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero((inside != bg) & (inside != border_color)))
        if not (1 <= count <= palette_len):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
