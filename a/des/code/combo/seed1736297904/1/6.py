from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_from(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    blocked: np.ndarray,
    seen: np.ndarray,
) -> np.ndarray:
    h, w = grid.shape
    q = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                if blocked[nr, nc]:
                    continue
                if int(grid[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
    return np.array(coords, dtype=int)


def _outside_reachable(open_mask: np.ndarray) -> np.ndarray:
    h, w = open_mask.shape
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not reach[r, c]:
                reach[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not reach[r, c]:
                reach[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _interior_from_boundary_coords(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 2), dtype=int)

    min_r = int(coords[:, 0].min())
    max_r = int(coords[:, 0].max())
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())
    local = np.zeros((max_r - min_r + 3, max_c - min_c + 3), dtype=bool)
    for r, c in coords:
        local[int(r) - min_r + 1, int(c) - min_c + 1] = True

    outside = _outside_reachable(~local)
    inside = (~local) & (~outside)
    if not inside.any():
        return np.zeros((0, 2), dtype=int)

    pts = np.argwhere(inside)
    pts[:, 0] += min_r - 1
    pts[:, 1] += min_c - 1
    return pts.astype(int)


def _legend_row_signature(row: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    w = len(row)
    prefix = 0
    while prefix < w and int(row[prefix]) != bg:
        prefix += 1
    if prefix < 2:
        return None
    marker = int(row[0])
    if marker == bg:
        return None
    if any(int(v) != marker for v in row[: prefix - 1]):
        return None
    mapped = int(row[prefix - 1])
    if mapped in (bg, marker):
        return None
    return marker, prefix - 1, mapped


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    by_marker: dict[int, dict[int, tuple[int, int]]] = {}
    for r in range(h):
        sig = _legend_row_signature(grid[r], bg)
        if sig is None:
            continue
        marker, count, mapped = sig
        by_marker.setdefault(marker, {})[r] = (count, mapped)

    best: tuple[int, int, dict[int, int]] | None = None
    for marker, rows in by_marker.items():
        for start_row, (count, _mapped) in rows.items():
            if count != 1:
                continue
            cur_map: dict[int, int] = {}
            expected = 1
            r = start_row
            while r in rows and rows[r][0] == expected:
                cur_map[expected] = int(rows[r][1])
                expected += 1
                r += 1
            if not cur_map:
                continue
            if best is None or len(cur_map) > best[1]:
                best = (marker, len(cur_map), cur_map)

    assert best is not None, "failed to detect left-edge legend"
    _marker, length, mapping = best

    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        sig = _legend_row_signature(grid[r], bg)
        if sig is None:
            continue
        marker, count, mapped = sig
        if marker != best[0]:
            continue
        if 1 <= count <= length and mapping[count] == mapped:
            mask[r, : count + 1] = True
    assert mapping, "legend mapping must be non-empty"
    return mapping, mask


def _find_loops(grid: np.ndarray, legend_mask: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    loops = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or legend_mask[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            coords = _component_from(grid, (r, c), color, legend_mask, seen)
            inside = _interior_from_boundary_coords(coords)
            if inside.size == 0:
                continue
            loops.append({"color": color, "coords": coords, "inside": inside})
    assert loops, "expected at least one closed loop"
    return loops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed loop according to the number of distinct dot colors trapped
    inside it, then erase all dots while preserving the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    mapping, legend_mask = _detect_legend(g, bg)
    loops = _find_loops(g, legend_mask, bg)

    out = np.full_like(g, bg)
    out[legend_mask] = g[legend_mask]

    for loop in loops:
        loop_color = int(loop["color"])
        inside_colors = {
            int(g[int(r), int(c)])
            for r, c in loop["inside"]
            if int(g[int(r), int(c)]) not in (bg, loop_color)
        }
        count = len(inside_colors)
        assert count in mapping, "missing legend row for trapped-color count"
        coords = loop["coords"]
        out[coords[:, 0], coords[:, 1]] = int(mapping[count])

    return out


def _interior_mask_from_boundary(boundary: np.ndarray) -> np.ndarray:
    outside = _outside_reachable(~boundary)
    return (~boundary) & (~outside)


def _draw_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    for t in range(steps + 1):
        mask[a[0] + dr * t, a[1] + dc * t] = True


def _rectangle_loop(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = rng.randint(5, 9)
    w = rng.randint(5, 11)
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return boundary, _interior_mask_from_boundary(boundary)


def _diamond_loop(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    radius = rng.randint(3, 5)
    size = 2 * radius + 1
    boundary = np.zeros((size, size), dtype=bool)
    center = radius
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) == radius:
                boundary[r, c] = True
    return boundary, _interior_mask_from_boundary(boundary)


def _octagon_loop(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = rng.randint(7, 11)
    w = rng.randint(7, 12)
    chamfer = rng.randint(1, max(1, min(h, w) // 3))
    boundary = np.zeros((h, w), dtype=bool)
    pts = [
        (0, chamfer),
        (chamfer, 0),
        (h - 1 - chamfer, 0),
        (h - 1, chamfer),
        (h - 1, w - 1 - chamfer),
        (h - 1 - chamfer, w - 1),
        (chamfer, w - 1),
        (0, w - 1 - chamfer),
    ]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        _draw_segment(boundary, a, b)
    return boundary, _interior_mask_from_boundary(boundary)


def _random_loop_shape(
    rng: random.Random, min_inside: int
) -> tuple[np.ndarray, np.ndarray] | None:
    factories = [_rectangle_loop, _diamond_loop, _octagon_loop]
    for _ in range(40):
        boundary, inside = rng.choice(factories)(rng)
        if int(inside.sum()) >= min_inside:
            if rng.random() < 0.35:
                boundary = boundary.T.copy()
                inside = inside.T.copy()
            return boundary, inside
    return None


def _mark_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_spaced_cells(
    cells: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    if len(cells) < count:
        return None
    pool = list(cells)
    rng.shuffle(pool)
    blocked: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if cell in blocked:
            continue
        chosen.append(cell)
        r, c = cell
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((r + dr, c + dc))
        if len(chosen) == count:
            return chosen
    return rng.sample(cells, count)


def _place_boundary(
    grid: np.ndarray,
    blocked: np.ndarray,
    boundary: np.ndarray,
    color: int,
    min_col: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid.shape
    bh, bw = boundary.shape
    rows = list(range(1, h - bh))
    cols = list(range(min_col, w - bw - 1))
    if not rows or not cols:
        return None
    for _ in range(120):
        r0 = rng.choice(rows)
        c0 = rng.choice(cols)
        if blocked[r0 - 1 : r0 + bh + 1, c0 - 1 : c0 + bw + 1].any():
            continue
        coords = np.argwhere(boundary)
        grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color
        _mark_box(blocked, r0, c0, r0 + bh - 1, c0 + bw - 1, pad=1)
        return r0, c0
    return None


GEN_ATTEMPTS = 400


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    max_count = rng.randint(3, 5)
    legend_width = max_count + 1
    n_loops = rng.randint(2, 4)

    height = rng.randint(max(16, max_count + 8), 28)
    width = rng.randint(legend_width + 16, 38)

    all_colors = list(range(10))
    bg = rng.choice(all_colors)
    remaining = [c for c in all_colors if c != bg]
    rng.shuffle(remaining)
    loop_color = remaining.pop()
    marker_color = remaining.pop()

    content_palette = remaining[: max(4, max_count)]
    if len(content_palette) < max(4, max_count):
        return None
    legend_colors = rng.sample(content_palette, max_count)
    legend_map = {k + 1: legend_colors[k] for k in range(max_count)}

    grid = np.full((height, width), bg, dtype=int)
    legend_mask = np.zeros((height, width), dtype=bool)
    legend_top = rng.randint(0, height - max_count)
    for count in range(1, max_count + 1):
        r = legend_top + count - 1
        grid[r, :count] = marker_color
        grid[r, count] = legend_map[count]
        legend_mask[r, : count + 1] = True

    blocked = np.zeros((height, width), dtype=bool)
    _mark_box(blocked, legend_top, 0, legend_top + max_count - 1, legend_width - 1, pad=1)
    play_min_col = legend_width + rng.randint(2, 4)

    target_counts = [rng.randint(1, max_count) for _ in range(n_loops)]
    if max_count > 1 and len(set(target_counts)) == 1:
        idx = rng.randrange(n_loops)
        choices = [v for v in range(1, max_count + 1) if v != target_counts[idx]]
        target_counts[idx] = rng.choice(choices)

    loop_specs = []
    for count in target_counts:
        min_inside = count + 2
        shape = _random_loop_shape(rng, min_inside)
        if shape is None:
            return None
        boundary, inside = shape
        loop_specs.append(
            {
                "count": count,
                "boundary": boundary,
                "inside": inside,
                "area": int(boundary.sum() + inside.sum()),
            }
        )
    loop_specs.sort(key=lambda spec: spec["area"], reverse=True)

    placed = []
    for spec in loop_specs:
        placed_at = _place_boundary(
            grid, blocked, spec["boundary"], loop_color, play_min_col, rng
        )
        if placed_at is None:
            return None
        r0, c0 = placed_at
        interior_cells = [
            (r0 + int(r), c0 + int(c))
            for r, c in np.argwhere(spec["inside"])
        ]
        chosen_colors = rng.sample(content_palette, spec["count"])
        max_extra = min(3, len(interior_cells) - spec["count"])
        extra = 0
        if max_extra > 0 and rng.random() < 0.75:
            extra = rng.randint(1, max_extra)
        dot_colors = chosen_colors + [rng.choice(chosen_colors) for _ in range(extra)]
        cells = _sample_spaced_cells(interior_cells, len(dot_colors), rng)
        if cells is None:
            return None
        rng.shuffle(dot_colors)
        for (r, c), color in zip(cells, dot_colors):
            grid[r, c] = color
        loop_coords = np.argwhere(spec["boundary"])
        loop_coords[:, 0] += r0
        loop_coords[:, 1] += c0
        placed.append(
            {
                "coords": loop_coords.astype(int),
                "count": spec["count"],
            }
        )

    external_candidates = [
        (r, c)
        for r in range(height)
        for c in range(play_min_col, width)
        if int(grid[r, c]) == bg and not blocked[r, c]
    ]
    if not external_candidates:
        return None
    ext_count = min(len(external_candidates), rng.randint(n_loops + 2, n_loops + 7))
    ext_cells = _sample_spaced_cells(external_candidates, ext_count, rng)
    if ext_cells is None:
        return None
    for r, c in ext_cells:
        grid[r, c] = rng.choice(content_palette)

    expected = np.full_like(grid, bg)
    expected[legend_mask] = grid[legend_mask]
    for item in placed:
        coords = item["coords"]
        expected[coords[:, 0], coords[:, 1]] = legend_map[item["count"]]

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
