from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
MAROON = 9

LEGEND_COLORS = [YELLOW, GREEN, BLUE, RED, MAGENTA]
LEGEND_SET = set(LEGEND_COLORS)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_palette(grid: np.ndarray) -> list[int]:
    mask = np.isin(grid, list(LEGEND_SET))
    assert mask.any(), "expected a legend strip using the five palette colors"
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    r0, c0, r1, c1 = _bbox(pts)

    colors: list[int] = []
    last = None
    for c in range(c0, c1 + 1):
        col_vals = grid[r0 : r1 + 1, c]
        vals = [int(v) for v in col_vals if int(v) in LEGEND_SET]
        if not vals:
            continue
        unique, counts = np.unique(np.array(vals, dtype=int), return_counts=True)
        color = int(unique[int(np.argmax(counts))])
        if color != last:
            colors.append(color)
            last = color

    assert len(colors) == 5, f"legend should compress to 5 runs, got {colors}"
    return colors


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    maroon_mask = grid == MAROON
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(maroon_mask):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have a non-empty interior"
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def _room_count(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0, "frame interior cannot be empty"
    passable = interior != GREY
    comps = _components(passable)
    assert 1 <= len(comps) <= 5, f"expected 1-5 subregions, got {len(comps)}"
    return len(comps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partitioned frame with the legend color indexed by its room count.

    Input structure:
    - The background is black.
    - A horizontal legend strip near the top contains five contiguous color blocks:
      yellow, green, blue, red, and magenta from left to right.
    - Below it are several one-pixel-thick maroon rectangular frames whose interiors
      may be split by grey horizontal or vertical divider lines.

    Rule:
    - Read the legend strip from left to right to obtain the 1st through 5th fill colors.
    - For each maroon frame, count how many 4-connected non-grey interior regions its
      divider pattern creates.
    - Fill every black interior cell of that frame with the legend color at that index,
      while keeping the maroon borders, grey dividers, and legend strip unchanged.

    Key insight:
    - The relevant number is the count of interior rooms created by the grey partitions;
      that room count directly indexes the visible top legend.
    """
    g = np.asarray(grid, dtype=int)
    palette = _legend_palette(g)
    out = g.copy()

    for box in _frame_bboxes(g):
        rooms = _room_count(g, box)
        fill = int(palette[rooms - 1])
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        target = interior == BG
        out[r0 + 1 : r1, c0 + 1 : c1][target] = fill

    return out


def _count_rooms_from_interior(interior: np.ndarray) -> int:
    return len(_components(interior != GREY))


def _build_partition(target_rooms: int, rng: random.Random) -> np.ndarray:
    for _ in range(300):
        height = rng.randint(max(3, target_rooms + 1), min(8, target_rooms + 5))
        width = rng.randint(4, 10)
        if height * width < 3 * target_rooms:
            continue

        interior = np.full((height, width), BG, dtype=int)
        leaves = [(0, 0, height, width)]
        ok = True

        for _split_index in range(target_rooms - 1):
            candidates: list[tuple[int, str, int]] = []
            for idx, (r0, c0, r1, c1) in enumerate(leaves):
                rh = r1 - r0
                rw = c1 - c0
                area = rh * rw
                if rh >= 3:
                    candidates.append((idx, "h", area))
                if rw >= 3:
                    candidates.append((idx, "v", area))
            if not candidates:
                ok = False
                break

            weights = [area for _, _, area in candidates]
            choice_idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
            leaf_idx, axis, _area = candidates[choice_idx]
            r0, c0, r1, c1 = leaves.pop(leaf_idx)
            rh = r1 - r0
            rw = c1 - c0

            if axis == "h":
                min_span = 2 if rh >= 5 else 1
                rows = list(range(r0 + min_span, r1 - min_span))
                if not rows:
                    rows = list(range(r0 + 1, r1 - 1))
                if not rows:
                    ok = False
                    break
                cut = rng.choice(rows)
                interior[cut, c0:c1] = GREY
                leaves.extend([(r0, c0, cut, c1), (cut + 1, c0, r1, c1)])
            else:
                min_span = 2 if rw >= 5 else 1
                cols = list(range(c0 + min_span, c1 - min_span))
                if not cols:
                    cols = list(range(c0 + 1, c1 - 1))
                if not cols:
                    ok = False
                    break
                cut = rng.choice(cols)
                interior[r0:r1, cut] = GREY
                leaves.extend([(r0, c0, r1, cut), (r0, cut + 1, r1, c1)])

        if not ok:
            continue

        room_sizes = [int(np.sum(interior[tuple(np.array(comp).T)] == BG)) for comp in _components(interior != GREY)]
        if _count_rooms_from_interior(interior) != target_rooms:
            continue
        if min(room_sizes) < 2:
            continue
        return interior

    raise RuntimeError(f"failed to build a {target_rooms}-room partition")


def _build_frame(target_rooms: int, rng: random.Random) -> np.ndarray:
    interior = _build_partition(target_rooms, rng)
    h, w = map(int, interior.shape)
    frame = np.full((h + 2, w + 2), MAROON, dtype=int)
    frame[1:-1, 1:-1] = interior
    return frame


def _assign_rows(frames: list[np.ndarray], rng: random.Random) -> list[list[np.ndarray]]:
    order = frames[:]
    rng.shuffle(order)
    n_rows = rng.randint(1, min(3, len(order)))
    rows: list[list[np.ndarray]] = [[] for _ in range(n_rows)]
    widths = [0] * n_rows

    for frame in order:
        gap = rng.randint(1, 3)
        best_width = min(widths)
        choices = [i for i, width in enumerate(widths) if width == best_width]
        idx = rng.choice(choices)
        rows[idx].append(frame)
        widths[idx] += int(frame.shape[1]) + (gap if rows[idx][:-1] else 0)

    return [row for row in rows if row]


def _layout(frames: list[np.ndarray], block_widths: list[int], legend_h: int, rng: random.Random):
    rows = _assign_rows(frames, rng)

    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    top_margin = rng.randint(0, 1)
    legend_gap = rng.randint(1, 3)
    row_gap = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    legend_w = sum(block_widths)

    row_specs = []
    max_row_w = 0
    total_rows_h = 0
    for row in rows:
        gaps = [rng.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        row_h = max(int(frame.shape[0]) for frame in row)
        row_w = sum(int(frame.shape[1]) for frame in row) + sum(gaps)
        row_specs.append((row, gaps, row_h, row_w))
        max_row_w = max(max_row_w, row_w)
        total_rows_h += row_h

    width = left_margin + max(legend_w, max_row_w) + right_margin
    height = top_margin + legend_h + legend_gap + total_rows_h + row_gap * (len(row_specs) - 1) + bottom_margin
    if width > 30 or height > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    legend_start = left_margin + rng.randint(0, width - left_margin - right_margin - legend_w)
    c = legend_start
    for color, block_w in zip(LEGEND_COLORS, block_widths):
        grid[top_margin : top_margin + legend_h, c : c + block_w] = color
        expected[top_margin : top_margin + legend_h, c : c + block_w] = color
        c += block_w

    placements: list[tuple[np.ndarray, int, int]] = []
    y = top_margin + legend_h + legend_gap
    for row, gaps, row_h, row_w in row_specs:
        x_slack = width - left_margin - right_margin - row_w
        x = left_margin + rng.randint(0, x_slack)
        for i, frame in enumerate(row):
            fh, fw = map(int, frame.shape)
            y_offset = rng.randint(0, row_h - fh)
            grid[y + y_offset : y + y_offset + fh, x : x + fw] = frame
            placements.append((frame, y + y_offset, x))
            x += fw
            if i < len(gaps):
                x += gaps[i]
        y += row_h + row_gap

    for frame, r0, c0 in placements:
        rooms = _count_rooms_from_interior(frame[1:-1, 1:-1])
        fill = LEGEND_COLORS[rooms - 1]
        patch = expected[r0 : r0 + frame.shape[0], c0 : c0 + frame.shape[1]]
        patch[:, :] = frame
        patch[frame == BG] = fill

    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 5)
    target_rooms = [rng.randint(1, 5) for _ in range(n_frames)]
    if n_frames > 1 and len(set(target_rooms)) == 1:
        idx = rng.randrange(n_frames)
        choices = [v for v in range(1, 6) if v != target_rooms[idx]]
        target_rooms[idx] = rng.choice(choices)

    frames = [_build_frame(rooms, rng) for rooms in target_rooms]
    block_widths = [rng.randint(1, 3) for _ in LEGEND_COLORS]
    legend_h = rng.randint(1, 2)

    laid_out = _layout(frames, block_widths, legend_h, rng)
    if laid_out is None:
        return None
    grid, expected = laid_out

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
