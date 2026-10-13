import numpy as np
from collections import defaultdict


BG = 0
GROUP_GAP = 2
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

OUTPUT_MAX = 30
GRID_MAX = 30
GROUP_COUNT_RANGE = (1, 3)
GEN_RETRIES = 300


def _components_of_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            area = len(coords)

            assert area == rect_h * rect_w, "every component must be a solid rectangle"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "h": rect_h,
                    "w": rect_w,
                    "area": area,
                }
            )

    return components


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _axis_gap(a: dict, b: dict) -> list[int]:
    (ar0, ac0, ar1, ac1) = a["bbox"]
    (br0, bc0, br1, bc1) = b["bbox"]
    gaps: list[int] = []

    if _overlap_1d(ar0, ar1, br0, br1):
        if ac1 < bc0:
            gaps.append(bc0 - ac1 - 1)
        elif bc1 < ac0:
            gaps.append(ac0 - bc1 - 1)

    if _overlap_1d(ac0, ac1, bc0, bc1):
        if ar1 < br0:
            gaps.append(br0 - ar1 - 1)
        elif br1 < ar0:
            gaps.append(ar0 - br1 - 1)

    return gaps


def _min_group_gap(rects: list[dict]) -> int:
    if len(rects) <= 1:
        return 0

    best = None
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            for gap in _axis_gap(rects[i], rects[j]):
                if best is None or gap < best:
                    best = gap
    assert best is not None and best >= 1, "same-color rectangles must expose a positive axis gap"
    return int(best)


def _paint_frame(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top, left : left + width] = color
    canvas[top + height - 1, left : left + width] = color
    canvas[top : top + height, left] = color
    canvas[top : top + height, left + width - 1] = color


def _group_order_key(rects: list[dict]) -> tuple[int, int]:
    r0 = min(rect["bbox"][0] for rect in rects)
    c0 = min(rect["bbox"][1] for rect in rects)
    return (c0, r0)


def _build_group_canvas(rects: list[dict], gap: int) -> np.ndarray:
    color = int(rects[0]["color"])
    outer_h = int(rects[0]["h"])
    outer_w = int(rects[0]["w"])
    canvas = np.full((outer_h, outer_w), BG, dtype=int)

    prev_top = None
    prev_left = None
    for rect in rects:
        height = int(rect["h"])
        width = int(rect["w"])
        assert (outer_h - height) % 2 == 0
        assert (outer_w - width) % 2 == 0
        top = (outer_h - height) // 2
        left = (outer_w - width) // 2

        if prev_top is not None:
            assert top - prev_top - 1 == gap
            assert left - prev_left - 1 == gap

        _paint_frame(canvas, top, left, height, width, color)
        prev_top = top
        prev_left = left

    return canvas


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _components_of_rectangles(g)
    assert components, "grid must contain at least one rectangle"

    groups_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        groups_by_color[int(comp["color"])].append(comp)

    ordered_groups: list[np.ndarray] = []
    group_keys: list[tuple[int, int]] = []
    for color, rects in groups_by_color.items():
        rects = sorted(rects, key=lambda rect: (-int(rect["area"]), -int(rect["h"]), -int(rect["w"])))
        gap = _min_group_gap(rects)
        ordered_groups.append(_build_group_canvas(rects, gap))
        group_keys.append(_group_order_key(rects))

    packed = sorted(zip(group_keys, ordered_groups), key=lambda item: item[0])
    canvases = [canvas for _, canvas in packed]

    out_h = max(canvas.shape[0] for canvas in canvases)
    out_w = sum(canvas.shape[1] for canvas in canvases) + GROUP_GAP * (len(canvases) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    cur_c = 0
    for canvas in canvases:
        hh, ww = canvas.shape
        mask = canvas != BG
        out[:hh, cur_c : cur_c + ww][mask] = canvas[mask]
        cur_c += ww + GROUP_GAP

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored solid rectangles, infer the smallest axis-aligned gap per color,
    and redraw each color group as concentric rectangular frames packed left-to-right.
    """
    return _solve(grid)


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _choose_group_shape(rng: np.random.Generator, outer_limit: int) -> tuple[int, int, list[tuple[int, int]]]:
    configs = []
    for count in range(2, 5):
        for gap in range(1, 4):
            step = 2 * (gap + 1)
            if 1 + step * (count - 1) <= outer_limit:
                configs.append((count, gap))

    assert configs
    count, gap = configs[_randint(rng, 0, len(configs) - 1)]
    step = 2 * (gap + 1)
    inner_max = outer_limit - step * (count - 1)
    inner_h = _randint(rng, 1, inner_max)
    inner_w = _randint(rng, 1, inner_max)

    dims = []
    for level in range(count - 1, -1, -1):
        dims.append((inner_h + step * level, inner_w + step * level))
    return gap, count, dims


def _align_offset(total: int, size: int, mode: int) -> int:
    if mode == 0:
        return 0
    if mode == 1:
        return (total - size) // 2
    return total - size


def _cluster_layout(
    rng: np.random.Generator,
    dims: list[tuple[int, int]],
    group_gap: int,
) -> tuple[str, int, int, list[tuple[int, int, int, int]]]:
    order = list(map(int, rng.permutation(len(dims))))
    ordered_dims = [dims[i] for i in order]
    orientation = "horizontal" if _randint(rng, 0, 1) == 0 else "vertical"
    mode = _randint(rng, 0, 2)
    gaps = [_randint(rng, 0, 3) for _ in range(max(0, len(dims) - 1))]
    if gaps:
        gaps[_randint(rng, 0, len(gaps) - 1)] = 0
    gaps = [group_gap + extra for extra in gaps]

    rects: list[tuple[int, int, int, int]] = []
    if orientation == "horizontal":
        cluster_h = max(h for h, _ in ordered_dims)
        cur_c = 0
        for idx, (height, width) in enumerate(ordered_dims):
            top = _align_offset(cluster_h, height, mode)
            rects.append((top, cur_c, height, width))
            cur_c += width
            if idx < len(gaps):
                cur_c += gaps[idx]
        cluster_w = cur_c
    else:
        cluster_w = max(w for _, w in ordered_dims)
        cur_r = 0
        for idx, (height, width) in enumerate(ordered_dims):
            left = _align_offset(cluster_w, width, mode)
            rects.append((cur_r, left, height, width))
            cur_r += height
            if idx < len(gaps):
                cur_r += gaps[idx]
        cluster_h = cur_r

    return orientation, cluster_h, cluster_w, rects


def _make_group_spec(rng: np.random.Generator, color: int, outer_limit: int) -> dict:
    gap, count, dims = _choose_group_shape(rng, outer_limit)
    orientation, cluster_h, cluster_w, rects = _cluster_layout(rng, dims, gap)
    return {
        "color": int(color),
        "gap": int(gap),
        "count": int(count),
        "dims": dims,
        "orientation": orientation,
        "cluster_h": int(cluster_h),
        "cluster_w": int(cluster_w),
        "cluster_rects": rects,
    }


def _paint_input(specs: list[dict], rng: np.random.Generator) -> np.ndarray | None:
    left_margin = _randint(rng, 1, 4)
    right_margin = _randint(rng, 1, 4)
    inner_gaps = [_randint(rng, 2, 6) for _ in range(max(0, len(specs) - 1))]

    min_w = left_margin + right_margin + sum(spec["cluster_w"] for spec in specs) + sum(inner_gaps)
    if min_w > GRID_MAX:
        return None

    top_margin = _randint(rng, 1, 4)
    bottom_margin = _randint(rng, 1, 4)
    min_h = top_margin + bottom_margin + max(spec["cluster_h"] for spec in specs)
    if min_h > GRID_MAX:
        return None

    grid_w = _randint(rng, min_w, min(GRID_MAX, min_w + 6))
    grid_h = _randint(rng, min_h, min(GRID_MAX, min_h + 6))

    extra_slots = np.zeros(len(specs) + 1, dtype=int)
    slack = grid_w - min_w
    if slack > 0:
        extra_slots = rng.multinomial(slack, np.full(len(specs) + 1, 1.0 / (len(specs) + 1)))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    cur_c = left_margin + int(extra_slots[0])

    for idx, spec in enumerate(specs):
        cluster_h = int(spec["cluster_h"])
        cluster_w = int(spec["cluster_w"])
        top = _randint(rng, 0, grid_h - cluster_h)

        for rel_r, rel_c, height, width in spec["cluster_rects"]:
            r0 = top + rel_r
            c0 = cur_c + rel_c
            if np.any(grid[r0 : r0 + height, c0 : c0 + width] != BG):
                return None
            grid[r0 : r0 + height, c0 : c0 + width] = int(spec["color"])

        cur_c += cluster_w
        if idx < len(inner_gaps):
            cur_c += inner_gaps[idx] + int(extra_slots[idx + 1])

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    group_count = _randint(rng, GROUP_COUNT_RANGE[0], GROUP_COUNT_RANGE[1])
    colors = list(map(int, rng.choice(PALETTE, size=group_count, replace=False)))
    outer_limit = 14 - 2 * (group_count - 1)

    specs = [_make_group_spec(rng, color, outer_limit) for color in colors]

    out_w = sum(spec["dims"][0][1] for spec in specs) + GROUP_GAP * (group_count - 1)
    out_h = max(spec["dims"][0][0] for spec in specs)
    if out_w > OUTPUT_MAX or out_h > OUTPUT_MAX:
        return None

    specs = [specs[i] for i in map(int, rng.permutation(len(specs)))]
    grid = _paint_input(specs, rng)
    if grid is None:
        return None

    solved = _solve(grid)
    if solved.shape[0] > OUTPUT_MAX or solved.shape[1] > OUTPUT_MAX:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
