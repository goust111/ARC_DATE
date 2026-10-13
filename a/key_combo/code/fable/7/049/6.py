from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, int, int]:
    best = None
    for color in np.unique(grid):
        coords = np.argwhere(grid == int(color))
        if coords.shape[0] < 8:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(patch[0, :] == color)
            and np.all(patch[-1, :] == color)
            and np.all(patch[:, 0] == color)
            and np.all(patch[:, -1] == color)
        ):
            continue

        perim = 2 * (patch.shape[0] + patch.shape[1]) - 4
        if int(np.count_nonzero(patch == color)) != perim:
            continue

        inner = patch[1:-1, 1:-1]
        if np.any(inner == color):
            continue
        counts = Counter(map(int, inner.ravel()))
        ranked = counts.most_common()
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        fill = int(ranked[0][0])

        score = patch.shape[0] * patch.shape[1]
        candidate = (int(r0), int(c0), int(r1), int(c1), int(color), fill)
        if best is None or score > best[0]:
            best = (score, candidate)

    if best is None:
        raise ValueError("frame not found")
    return best[1]


def _project_contact(
    touch: tuple[int, int], frame: tuple[int, int, int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1, _border, _fill = frame
    r, c = touch
    out_h = r1 - r0 - 1
    out_w = c1 - c0 - 1
    if c == c0:
        return r - r0 - 1, 0
    if c == c1:
        return r - r0 - 1, out_w - 1
    if r == r0:
        return 0, c - c0 - 1
    if r == r1:
        return out_h - 1, c - c0 - 1
    raise ValueError("touch point is not on the frame")


def _wall_map(
    grid: np.ndarray, frame: tuple[int, int, int, int, int, int], bg: int
) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1, border, fill = frame
    touches: dict[int, set[tuple[int, int]]] = {}

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (bg, border, fill):
                continue
            if r0 < r < r1 and c0 < c < c1:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                    continue
                if int(grid[nr, nc]) == border:
                    touches.setdefault(color, set()).add((nr, nc))

    out_h = r1 - r0 - 1
    out_w = c1 - c0 - 1
    walls: dict[int, list[tuple[int, int]]] = {}
    for color, pts in touches.items():
        if len(pts) != 2:
            continue
        a, b = sorted(pts)
        pa = _project_contact(a, frame)
        pb = _project_contact(b, frame)
        if pa[0] == pb[0]:
            rr = pa[0]
            walls[color] = [(rr, cc) for cc in range(out_w)]
        elif pa[1] == pb[1]:
            cc = pa[1]
            walls[color] = [(rr, cc) for rr in range(out_h)]
        else:
            raise ValueError("projected touch points must align")
    return walls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project external touch traces into straight interior walls, then fill the resulting regions by marker-majority color.

    Input structure:
    - A grid contains a one-pixel rectangular frame of one color, with a uniform interior fill color.
    - Outside the frame, several colored traces touch the frame at exactly two border points per color.
    - Inside the frame, isolated one-pixel marker cells use colors distinct from the frame, fill, and outside traces.

    Rule:
    - Crop the output to the frame interior.
    - For each outside trace color, project its two touch points one cell inward and connect those projections with a full horizontal or vertical wall of that same color.
    - Treat the new walls as barriers: every wall-bounded region that contains markers is repainted with the unique majority marker color in that region, while marker-free regions stay the interior fill color.

    Key insight:
    - The answer is determined by regions that do not explicitly exist in the input until the external touch traces are projected inward as partitioning walls.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _get_bg(g)
    frame = _find_frame(g)
    r0, c0, r1, c1, _border, fill = frame
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    out_h, out_w = interior.shape

    walls = _wall_map(g, frame, bg)
    wall_mask = np.zeros((out_h, out_w), dtype=bool)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    for color, cells in walls.items():
        for r, c in cells:
            if wall_mask[r, c]:
                raise ValueError("crossing walls are not supported")
            wall_mask[r, c] = True
            out[r, c] = int(color)

    visited = np.zeros((out_h, out_w), dtype=bool)
    for sr in range(out_h):
        for sc in range(out_w):
            if wall_mask[sr, sc] or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            region = []
            marker_colors = []
            while q:
                r, c = q.popleft()
                region.append((r, c))
                val = int(interior[r, c])
                if val != int(fill):
                    marker_colors.append(val)
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < out_h and 0 <= nc < out_w and not wall_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            region_color = int(fill)
            if marker_colors:
                counts = Counter(marker_colors).most_common()
                if len(counts) > 1 and counts[0][1] == counts[1][1]:
                    raise ValueError("marker majority must be unique")
                region_color = int(counts[0][0])
            for r, c in region:
                out[r, c] = region_color

    return out


def _choose_distinct_colors(count: int, avoid: set[int] | None = None) -> list[int]:
    banned = set() if avoid is None else set(avoid)
    pool = [c for c in PALETTE if c not in banned]
    if len(pool) < count:
        raise ValueError("not enough colors available")
    return random.sample(pool, count)


def _sample_region_layout(orientation: str) -> tuple[int, int, list[tuple[int, int]], list[int]]:
    region_count = random.randint(2, 4)
    if orientation == "horizontal":
        heights = [random.randint(2, 4) for _ in range(region_count)]
        width = random.randint(8, 18)
        total_h = sum(heights) + region_count - 1
        spans = []
        cursor = 0
        wall_rows = []
        for idx, h in enumerate(heights):
            spans.append((cursor, cursor + h - 1))
            cursor += h
            if idx + 1 < region_count:
                wall_rows.append(cursor)
                cursor += 1
        return total_h, width, spans, wall_rows

    widths = [random.randint(2, 4) for _ in range(region_count)]
    height = random.randint(8, 18)
    total_w = sum(widths) + region_count - 1
    spans = []
    cursor = 0
    wall_cols = []
    for idx, w in enumerate(widths):
        spans.append((cursor, cursor + w - 1))
        cursor += w
        if idx + 1 < region_count:
            wall_cols.append(cursor)
            cursor += 1
    return height, total_w, spans, wall_cols


def _region_cells(
    orientation: str,
    region_index: int,
    spans: list[tuple[int, int]],
    out_h: int,
    out_w: int,
) -> list[tuple[int, int]]:
    start, end = spans[region_index]
    if orientation == "horizontal":
        return [(r, c) for r in range(start, end + 1) for c in range(out_w)]
    return [(r, c) for r in range(out_h) for c in range(start, end + 1)]


def _place_markers(
    inp: np.ndarray,
    expected: np.ndarray,
    orientation: str,
    spans: list[tuple[int, int]],
    reserved_colors: set[int],
) -> None:
    out_h, out_w = inp.shape
    region_count = len(spans)
    marked = [random.random() < 0.75 for _ in range(region_count)]
    if not any(marked):
        marked[random.randrange(region_count)] = True
    if region_count > 1 and all(marked):
        marked[random.randrange(region_count)] = False

    marker_palette = _choose_distinct_colors(3, reserved_colors)
    for idx in range(region_count):
        cells = _region_cells(orientation, idx, spans, out_h, out_w)
        if not marked[idx]:
            continue

        total = random.randint(1, min(5, len(cells)))
        max_color_count = min(3, total, (total + 1) // 2)
        if total == 1:
            color_count = 1
        else:
            color_count = random.randint(1, max_color_count)
            if color_count == total:
                color_count -= 1

        chosen_colors = random.sample(marker_palette, color_count)
        majority_color = random.choice(chosen_colors)
        if color_count == 1:
            counts = {majority_color: total}
        else:
            majority = random.randint(total // 2 + 1, total - (color_count - 1))
            counts = {majority_color: majority}
            remaining = total - majority
            others = [c for c in chosen_colors if c != majority_color]
            for j, color in enumerate(others):
                need = len(others) - j - 1
                amount = 1 if j == len(others) - 1 else random.randint(1, remaining - need)
                counts[color] = amount
                remaining -= amount

        positions = random.sample(cells, total)
        colors = []
        for color, count in counts.items():
            colors.extend([int(color)] * int(count))
        random.shuffle(colors)
        for (r, c), color in zip(positions, colors):
            inp[r, c] = int(color)

        expected_color = max(counts.items(), key=lambda kv: kv[1])[0]
        rr, cc = zip(*cells)
        expected[np.array(rr), np.array(cc)] = int(expected_color)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    orientation = random.choice(["horizontal", "vertical"])
    out_h, out_w, spans, wall_axis = _sample_region_layout(orientation)
    wall_count = len(wall_axis)

    bg, border, fill = _choose_distinct_colors(3)
    wall_colors = _choose_distinct_colors(wall_count, {bg, border, fill})

    interior = np.full((out_h, out_w), int(fill), dtype=int)
    expected = np.full((out_h, out_w), int(fill), dtype=int)

    if orientation == "horizontal":
        for row, color in zip(wall_axis, wall_colors):
            expected[row, :] = int(color)
    else:
        for col, color in zip(wall_axis, wall_colors):
            expected[:, col] = int(color)

    _place_markers(interior, expected, orientation, spans, {bg, border, fill, *wall_colors})

    frame_h = out_h + 2
    frame_w = out_w + 2
    if orientation == "horizontal":
        left_lens = [random.randint(1, 4) for _ in range(wall_count)]
        right_lens = [random.randint(1, 4) for _ in range(wall_count)]
        top_margin = random.randint(1, 4)
        bottom_margin = random.randint(1, 4)
        left_margin = max(left_lens, default=1) + random.randint(1, 3)
        right_margin = max(right_lens, default=1) + random.randint(1, 3)
    else:
        top_lens = [random.randint(1, 4) for _ in range(wall_count)]
        bottom_lens = [random.randint(1, 4) for _ in range(wall_count)]
        left_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        top_margin = max(top_lens, default=1) + random.randint(1, 3)
        bottom_margin = max(bottom_lens, default=1) + random.randint(1, 3)

    grid_h = top_margin + frame_h + bottom_margin
    grid_w = left_margin + frame_w + right_margin
    if grid_h > 30 or grid_w > 30:
        raise ValueError("grid exceeds ARC limits")

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0 : r1 + 1, c0 : c1 + 1] = int(border)
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior

    if orientation == "horizontal":
        for row, color, left_len, right_len in zip(wall_axis, wall_colors, left_lens, right_lens):
            frame_row = r0 + 1 + row
            grid[frame_row, c0 - left_len : c0] = int(color)
            grid[frame_row, c1 + 1 : c1 + 1 + right_len] = int(color)
    else:
        for col, color, top_len, bottom_len in zip(wall_axis, wall_colors, top_lens, bottom_lens):
            frame_col = c0 + 1 + col
            grid[r0 - top_len : r0, frame_col] = int(color)
            grid[r1 + 1 : r1 + 1 + bottom_len, frame_col] = int(color)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _build_case()
        except ValueError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
