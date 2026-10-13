from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EDGE_NAMES = ("top", "bottom", "left", "right")


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_info(row: np.ndarray) -> tuple[list[int], np.ndarray]:
    cols = np.flatnonzero(row != BLACK)
    assert cols.size > 0, "top row must contain a non-black legend strip"
    if cols.size > 1:
        assert bool(np.all(np.diff(cols) == 1)), "legend strip must be contiguous"
    colors = [int(row[int(c)]) for c in cols]
    return colors, cols.astype(int, copy=False)


def _nearest_edge_name(r: int, c: int, h: int, w: int, *, require_unique: bool = True) -> str | None:
    dists = {
        "top": int(r),
        "bottom": int(h - 1 - r),
        "left": int(c),
        "right": int(w - 1 - c),
    }
    best = min(dists.values())
    winners = [name for name, dist in dists.items() if int(dist) == int(best)]
    if len(winners) != 1:
        if require_unique:
            raise AssertionError("ring center must have a unique nearest edge")
        return None
    return winners[0]


def _draw_ray(out: np.ndarray, center: tuple[int, int], edge_name: str, color: int) -> None:
    r, c = center
    if edge_name == "top":
        for rr in range(r, -1, -1):
            out[int(rr), int(c)] = int(color)
        return
    if edge_name == "bottom":
        for rr in range(r, out.shape[0]):
            out[int(rr), int(c)] = int(color)
        return
    if edge_name == "left":
        for cc in range(c, -1, -1):
            out[int(r), int(cc)] = int(color)
        return
    for cc in range(c, out.shape[1]):
        out[int(r), int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map each grey ring's enclosed black-cell count through the top legend, then replace the ring
    by a ray from its center toward the uniquely nearest grid edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_colors, legend_cols = _legend_info(g[0])

    ring_mask = g == GREY
    ring_mask[0, :] = False
    rings = _components4(ring_mask)
    assert rings, "expected at least one grey ring"

    specs: list[tuple[tuple[int, int], str, int]] = []
    for comp in rings:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "ring bbox must have a non-empty interior"

        for cc in range(c0, c1 + 1):
            assert int(g[r0, cc]) == GREY and int(g[r1, cc]) == GREY
        for rr in range(r0, r1 + 1):
            assert int(g[rr, c0]) == GREY and int(g[rr, c1]) == GREY

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0, "ring interior must be non-empty"
        assert bool(np.all(inner == BLACK)), "ring interior must contain only black cells"

        hole_pixels = int(inner.size)
        assert 1 <= hole_pixels <= len(legend_colors), "hole count must index the legend"
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        edge_name = _nearest_edge_name(center[0], center[1], h, w)
        color = int(legend_colors[hole_pixels - 1])
        specs.append((center, edge_name, color))

    out = np.zeros_like(g)
    for center, edge_name, color in specs:
        _draw_ray(out, center, edge_name, color)

    out[0, legend_cols] = g[0, legend_cols]
    return out


def _odd_factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for d in range(1, n + 1, 2):
        if n % d == 0:
            q = n // d
            if q % 2 == 1:
                pairs.append((d, q))
    return pairs


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _bbox_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _ray_cells(center: tuple[int, int], edge_name: str, h: int, w: int) -> set[tuple[int, int]]:
    r, c = center
    if edge_name == "top":
        return {(rr, c) for rr in range(r, -1, -1)}
    if edge_name == "bottom":
        return {(rr, c) for rr in range(r, h)}
    if edge_name == "left":
        return {(r, cc) for cc in range(c, -1, -1)}
    return {(r, cc) for cc in range(c, w)}


def _draw_ring(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.zeros_like(grid)
    for spec in specs:
        _draw_ray(out, spec["center"], spec["edge_name"], spec["color"])
    legend_mask = grid[0] != BLACK
    out[0, legend_mask] = grid[0, legend_mask]
    return out


def _choose_shape(n: int) -> tuple[int, int]:
    return random.choice(_odd_factor_pairs(n))


def _candidate_center(
    edge_name: str,
    half_h: int,
    half_w: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    min_r = half_h + 1
    max_r = h - 1 - half_h
    min_c = half_w
    max_c = w - 1 - half_w
    assert min_r <= max_r and min_c <= max_c

    if edge_name == "top":
        top_hi = max(min_r, min(max_r, h // 3 + 1))
        return random.randint(min_r, top_hi), random.randint(min_c, max_c)
    if edge_name == "bottom":
        bot_lo = min(max_r, max(min_r, h - 1 - (h // 3 + 1)))
        return random.randint(bot_lo, max_r), random.randint(min_c, max_c)
    if edge_name == "left":
        left_hi = max(min_c, min(max_c, w // 3))
        return random.randint(min_r, max_r), random.randint(min_c, left_hi)
    right_lo = min(max_c, max(min_c, w - 1 - (w // 3)))
    return random.randint(min_r, max_r), random.randint(right_lo, max_c)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    if w < 8:
        return None

    legend_len = random.randint(4, min(8, w - 3))
    legend_start = random.randint(0, w - legend_len)
    legend_end = legend_start + legend_len - 1
    legend_colors = random.sample(list(LEGEND_COLORS), legend_len)

    usable_counts = [n for n in range(1, legend_len + 1) if n % 2 == 1]
    max_objects = min(4, len(usable_counts))
    if max_objects < 2:
        return None
    num_objects = random.randint(2, max_objects)
    hole_counts = random.sample(usable_counts, num_objects)
    directions = random.sample(list(EDGE_NAMES), num_objects)
    random.shuffle(hole_counts)

    items = []
    for hole_count, edge_name in zip(hole_counts, directions, strict=False):
        inner_h, inner_w = _choose_shape(hole_count)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        items.append(
            {
                "hole_count": int(hole_count),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "edge_name": edge_name,
                "color": int(legend_colors[hole_count - 1]),
            }
        )

    items.sort(key=lambda spec: max(spec["outer_h"], spec["outer_w"]), reverse=True)

    placed: list[dict] = []
    occupied_boxes: list[tuple[int, int, int, int]] = []
    occupied_box_cells: set[tuple[int, int]] = set()
    occupied_ray_cells: set[tuple[int, int]] = set()

    for item in items:
        half_h = item["outer_h"] // 2
        half_w = item["outer_w"] // 2
        placed_ok = False
        for _ in range(300):
            center = _candidate_center(item["edge_name"], half_h, half_w, h, w)
            if item["edge_name"] == "top" and legend_start <= center[1] <= legend_end:
                continue
            if _nearest_edge_name(center[0], center[1], h, w, require_unique=False) != item["edge_name"]:
                continue

            box = (
                center[0] - half_h,
                center[1] - half_w,
                center[0] + half_h,
                center[1] + half_w,
            )
            if box[0] < 1 or box[2] >= h or box[1] < 0 or box[3] >= w:
                continue
            if any(_boxes_overlap(box, other, pad=1) for other in occupied_boxes):
                continue

            box_cells = _bbox_cells(box)
            ray_cells = _ray_cells(center, item["edge_name"], h, w)
            if bool((ray_cells - box_cells) & occupied_box_cells):
                continue
            if bool(box_cells & occupied_ray_cells):
                continue
            if bool(ray_cells & occupied_ray_cells):
                continue

            spec = dict(item)
            spec["center"] = center
            spec["box"] = box
            spec["box_cells"] = box_cells
            spec["ray_cells"] = ray_cells
            placed.append(spec)
            occupied_boxes.append(box)
            occupied_box_cells |= box_cells
            occupied_ray_cells |= ray_cells
            placed_ok = True
            break
        if not placed_ok:
            return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, legend_start : legend_start + legend_len] = np.array(legend_colors, dtype=int)
    for spec in placed:
        _draw_ring(grid, spec["box"])

    expected = _render_expected(grid, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")
