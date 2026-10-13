import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _ordered_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        border.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        border.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        border.append((r, c0))
    return border


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _ordered_border(r0, c0, r1, c1)
        if len(comp) != len(border):
            continue
        if set(comp) != set(border):
            continue
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_pts = np.argwhere((inside != BG) & (inside != FRAME))
        if len(inner_pts) != 1:
            continue
        mr = r0 + 1 + int(inner_pts[0, 0])
        mc = c0 + 1 + int(inner_pts[0, 1])
        if (mr, mc) != ((r0 + r1) // 2, (c0 + c1) // 2):
            continue
        frames.append(
            {
                "marker_color": int(grid[mr, mc]),
                "marker_pos": (mr, mc),
                "border": border,
            }
        )
    return frames


def _find_bars(grid: np.ndarray) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:
    bars: dict[int, list[int]] = {}
    bar_cells: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) in (BG, FRAME):
                c += 1
                continue
            c0 = c
            seq: list[int] = []
            while c < w and int(grid[r, c]) not in (BG, FRAME):
                seq.append(int(grid[r, c]))
                c += 1
            if len(seq) <= 1:
                continue
            bars[seq[0]] = seq
            bar_cells.extend((r, cc) for cc in range(c0, c))
    return bars, bar_cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    bars, bar_cells = _find_bars(g)

    for r, c in bar_cells:
        out[r, c] = BG

    for frame in frames:
        marker_color = int(frame["marker_color"])
        marker_r, marker_c = frame["marker_pos"]
        out[marker_r, marker_c] = BG
        seq = bars[marker_color]
        border = frame["border"]
        for idx, (r, c) in enumerate(border):
            out[r, c] = seq[idx % len(seq)]

    return out


def _make_sequence(rng: random.Random, marker_color: int, perimeter: int) -> list[int]:
    repeat_branch = rng.random() < 0.5
    if repeat_branch:
        length = rng.randint(4, min(12, perimeter - 1))
    else:
        length = rng.randint(perimeter, min(perimeter + 8, perimeter + 10))
    cycle_len = 2 if rng.random() < 0.6 else 3
    extras = rng.sample([c for c in PALETTE if c != marker_color], cycle_len - 1)
    cycle = [marker_color] + extras
    return [cycle[i % cycle_len] for i in range(length)]


def _fits(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    er0 = max(0, r0 - 2)
    ec0 = max(0, c0 - 2)
    er1 = min(used.shape[0] - 1, r1 + 2)
    ec1 = min(used.shape[1] - 1, c1 + 2)
    return not used[er0 : er1 + 1, ec0 : ec1 + 1].any()


def _mark_used(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    er0 = max(0, r0 - 2)
    ec0 = max(0, c0 - 2)
    er1 = min(used.shape[0] - 1, r1 + 2)
    ec1 = min(used.shape[1] - 1, c1 + 2)
    used[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.randint(1, 4)
    if num_frames == 1:
        size_choices = (5, 7, 9, 11)
    elif num_frames == 2:
        size_choices = (5, 7, 9)
    else:
        size_choices = (5, 7)

    marker_colors = rng.sample(list(PALETTE), num_frames)
    specs: list[dict[str, object]] = []
    for color in marker_colors:
        h = rng.choice(size_choices)
        w = rng.choice(size_choices)
        perimeter = 2 * (h + w) - 4
        seq = _make_sequence(rng, color, perimeter)
        specs.append({"marker_color": color, "h": h, "w": w, "seq": seq})

    max_bar_len = max(len(spec["seq"]) for spec in specs)
    max_frame_h = max(int(spec["h"]) for spec in specs)
    max_frame_w = max(int(spec["w"]) for spec in specs)

    width_lo = max(22 + 3 * max(0, num_frames - 2), max_bar_len + 2, max_frame_w + 6)
    height_lo = max(20 + 2 * max(0, num_frames - 2), max_frame_h + 8)
    width_hi = min(40, width_lo + 6)
    height_hi = min(32, height_lo + 6)
    if width_lo > width_hi or height_lo > height_hi:
        return None

    h = rng.randint(height_lo, height_hi)
    w = rng.randint(width_lo, width_hi)
    grid = np.zeros((h, w), dtype=int)

    bar_rows = [0, 2, h - 3, h - 1]
    rng.shuffle(bar_rows)
    for spec, row in zip(specs, bar_rows):
        seq = spec["seq"]
        start = rng.randint(0, w - len(seq))
        grid[row, start : start + len(seq)] = seq

    used = np.zeros((h, w), dtype=bool)
    used[bar_rows[:num_frames], :] = True

    for spec in sorted(specs, key=lambda item: int(item["h"]) * int(item["w"]), reverse=True):
        fh = int(spec["h"])
        fw = int(spec["w"])
        placed = False
        for _ in range(200):
            r0 = rng.randint(3, h - fh - 4)
            c0 = rng.randint(1, w - fw - 2)
            if not _fits(used, r0, c0, fh, fw):
                continue
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            grid[r0, c0 : c1 + 1] = FRAME
            grid[r1, c0 : c1 + 1] = FRAME
            grid[r0 : r1 + 1, c0] = FRAME
            grid[r0 : r1 + 1, c1] = FRAME
            grid[(r0 + r1) // 2, (c0 + c1) // 2] = int(spec["marker_color"])
            _mark_used(used, r0, c0, fh, fw)
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if Counter(grid.reshape(-1))[BG] <= Counter(grid.reshape(-1))[FRAME]:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
