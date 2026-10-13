from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 3
SEED_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    return pts


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            want = _frame_border_set(r0, c0, r1, c1)
            if set(pts) != want:
                continue
            if (r1 - r0) == (c1 - c0):
                continue
            frames.append((int(r0), int(c0), int(r1), int(c1)))
    frames.sort()
    return frames


def _seed_info(interior: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    coords = np.argwhere(interior != int(bg))
    assert coords.size > 0, "each frame must contain one seed rectangle"
    sr0, sc0 = coords.min(axis=0)
    sr1, sc1 = coords.max(axis=0)
    sr0, sc0, sr1, sc1 = map(int, (sr0, sc0, sr1, sc1))
    seed = interior[sr0 : sr1 + 1, sc0 : sc1 + 1]
    colors = sorted(int(v) for v in np.unique(seed) if int(v) != int(bg))
    assert len(colors) == 1, "seed must be monochrome"
    seed_color = int(colors[0])
    assert bool((seed == seed_color).all()), "seed must be a solid rectangle"
    mask = np.zeros_like(interior, dtype=bool)
    mask[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    assert not bool(((interior != int(bg)) & (~mask)).any()), "frame interior must only contain the seed"
    return sr0, sc0, sr1, sc1, seed_color


def _paint_horizontal(out: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, seed_h: int, seed_w: int, color: int) -> None:
    x = 0
    cur_w = int(seed_w)
    while x < inner_w:
        w = min(int(cur_w), int(inner_w - x))
        out[top : top + seed_h, left + x : left + x + w] = int(color)
        x += w
        cur_w = max(1, int(cur_w) // 2)


def _paint_vertical(out: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, seed_h: int, seed_w: int, color: int) -> None:
    y = 0
    cur_h = int(seed_h)
    while y < inner_h:
        h = min(int(cur_h), int(inner_h - y))
        out[top + y : top + y + h, left : left + seed_w] = int(color)
        y += h
        cur_h = max(1, int(cur_h) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recursively copy the seed rectangle inside each frame while shrinking along the frame's long axis.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping hollow rectangular frames drawn in green.
    - Inside each frame, the top-left inner corner contains a solid monochrome seed rectangle of a non-black, non-green color.
    - Every frame is strictly horizontal or strictly vertical; no frame is a square.

    Rule:
    - Process each frame independently and read the solid seed rectangle from its top-left interior corner.
    - If the frame is wider than it is tall, copy the seed to the right; each new copy keeps the same height but its width is halved with floor rounding and a minimum of 1.
    - If the frame is taller than it is wide, copy the seed downward; each new copy keeps the same width but its height is halved with floor rounding and a minimum of 1.
    - Place each copy flush against the previous one until the copied chain reaches the opposite inner wall; keep the frame itself unchanged and leave everything else black.

    Key insight:
    - The frame's aspect ratio determines whether the recursive shrinking happens horizontally or vertically, and the repeated size update is always "halve only the long-axis extent".
    """
    g = np.asarray(grid, dtype=int)
    bg = int(BG if bool((g == BG).sum()) >= g.size // 2 else np.bincount(g.ravel()).argmax())
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g, bg):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        sr0, sc0, sr1, sc1, seed_color = _seed_info(inner, bg)
        seed_h = int(sr1 - sr0 + 1)
        seed_w = int(sc1 - sc0 + 1)
        inner_h, inner_w = map(int, inner.shape)

        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
        if inner_w > inner_h:
            _paint_horizontal(out, r0 + 1 + sr0, c0 + 1 + sc0, inner_h - sr0, inner_w - sc0, seed_h, seed_w, seed_color)
        else:
            _paint_vertical(out, r0 + 1 + sr0, c0 + 1 + sc0, inner_h - sr0, inner_w - sc0, seed_h, seed_w, seed_color)

    return out


def _halving_sequence(start: int, copies: int) -> list[int]:
    seq: list[int] = []
    cur = int(start)
    for _ in range(copies):
        seq.append(int(cur))
        cur = max(1, int(cur) // 2)
    return seq


def _make_frame_spec(orientation: str, color: int) -> dict[str, int] | None:
    for _ in range(200):
        minor = random.randint(2, 6)
        start = random.randint(2, 7)
        copies = random.randint(3, 8)
        seq = _halving_sequence(start, copies)
        major = sum(seq)
        if orientation == "H":
            inner_h, inner_w = minor, major
            seed_h, seed_w = minor, start
        else:
            inner_h, inner_w = major, minor
            seed_h, seed_w = start, minor
        if inner_w == inner_h:
            continue
        if orientation == "H" and inner_w <= inner_h:
            continue
        if orientation == "V" and inner_h <= inner_w:
            continue
        if inner_h + 2 > 14 or inner_w + 2 > 14:
            continue
        ratio = (seed_h * seed_w) / float(inner_h * inner_w)
        if not 0.18 <= ratio <= 0.5:
            continue
        return {
            "orientation": 0 if orientation == "H" else 1,
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "frame_h": int(inner_h + 2),
            "frame_w": int(inner_w + 2),
            "seed_h": int(seed_h),
            "seed_w": int(seed_w),
            "color": int(color),
        }
    return None


def _row_partition(count: int, rows: int) -> list[int]:
    if rows == 1:
        return [count]
    breaks = sorted(random.sample(range(1, count), rows - 1))
    parts: list[int] = []
    prev = 0
    for b in breaks + [count]:
        parts.append(int(b - prev))
        prev = b
    return parts


def _layout_specs(specs: list[dict[str, int]]) -> tuple[int, int, list[tuple[dict[str, int], int, int]]] | None:
    for _ in range(200):
        ordered = specs[:]
        random.shuffle(ordered)
        row_count = random.randint(1, min(3, len(ordered)))
        parts = _row_partition(len(ordered), row_count)
        rows: list[list[dict[str, int]]] = []
        idx = 0
        for part in parts:
            rows.append(ordered[idx : idx + part])
            idx += part

        placements: list[tuple[dict[str, int], int, int]] = []
        total_w = 0
        y = 0
        for ridx, row in enumerate(rows):
            row_h = max(int(spec["frame_h"]) for spec in row)
            indent = random.randint(0, 3)
            x = indent
            for sidx, spec in enumerate(row):
                dy = random.randint(0, row_h - int(spec["frame_h"]))
                placements.append((spec, int(y + dy), int(x)))
                x += int(spec["frame_w"])
                if sidx != len(row) - 1:
                    x += random.randint(2, 4)
            total_w = max(total_w, x)
            y += row_h
            if ridx != len(rows) - 1:
                y += random.randint(2, 4)

        top = random.randint(1, 4)
        bottom = random.randint(1, 4)
        left = random.randint(1, 4)
        right = random.randint(1, 4)
        height = y + top + bottom
        width = total_w + left + right
        if height > 30 or width > 30:
            continue

        shifted = [(spec, int(r + top), int(c + left)) for spec, r, c in placements]
        return int(height), int(width), shifted
    return None


def _draw_frame(grid: np.ndarray, spec: dict[str, int], top: int, left: int) -> None:
    fh = int(spec["frame_h"])
    fw = int(spec["frame_w"])
    color = int(spec["color"])
    grid[top : top + fh, left : left + fw] = int(BG)
    grid[top, left : left + fw] = int(FRAME)
    grid[top + fh - 1, left : left + fw] = int(FRAME)
    grid[top : top + fh, left] = int(FRAME)
    grid[top : top + fh, left + fw - 1] = int(FRAME)
    grid[top + 1 : top + 1 + int(spec["seed_h"]), left + 1 : left + 1 + int(spec["seed_w"])] = color


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_frames = random.choice([1, 2, 2, 3, 3, 4])
        palette = random.sample(SEED_COLORS, n_frames)
        orientations = [random.choice(["H", "V"]) for _ in range(n_frames)]

        specs: list[dict[str, int]] = []
        ok = True
        for orientation, color in zip(orientations, palette):
            spec = _make_frame_spec(orientation, color)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if not ok:
            continue

        layout = _layout_specs(specs)
        if layout is None:
            continue
        h, w, placements = layout
        grid = np.full((h, w), BG, dtype=int)
        for spec, top, left in placements:
            _draw_frame(grid, spec, top, left)

        frames = _find_frames(grid, BG)
        if len(frames) != n_frames:
            continue
        if max(grid.shape) > 30:
            continue
        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
