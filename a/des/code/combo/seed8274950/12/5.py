from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()


GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
OBJECT_COUNT_CHOICES = np.array([1, 2, 3, 4], dtype=int)
OBJECT_COUNT_WEIGHTS = np.array([1.0, 3.0, 3.0, 1.0], dtype=float)
CANVAS_MARGIN = 1
PLACEMENT_MARGIN = 1
PLACEMENT_TRIES = 120
GENERATION_ATTEMPTS = 400


def _randint(low: int, high: int) -> int:
    return int(RNG.integers(int(low), int(high) + 1))


def get_bg(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    parts = [g[0, :], g[-1, :]]
    if g.shape[0] > 2:
        parts.append(g[1:-1, 0])
        parts.append(g[1:-1, -1])
    border = np.concatenate(parts)
    values, counts = np.unique(border, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_monochrome_components(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comp_ids = -np.ones((h, w), dtype=int)
    components: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if int(g[r0, c0]) == int(bg) or int(comp_ids[r0, c0]) != -1:
                continue

            color = int(g[r0, c0])
            cid = len(components)
            queue = deque([(r0, c0)])
            comp_ids[r0, c0] = cid
            coords: list[tuple[int, int]] = []
            r_min = r_max = int(r0)
            c_min = c_max = int(c0)

            while queue:
                r, c = queue.popleft()
                coords.append((int(r), int(c)))
                r_min = min(r_min, int(r))
                r_max = max(r_max, int(r))
                c_min = min(c_min, int(c))
                c_max = max(c_max, int(c))
                for dr, dc in DIRS4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and int(comp_ids[rr, cc]) == -1 and int(g[rr, cc]) == int(color):
                        comp_ids[rr, cc] = cid
                        queue.append((rr, cc))

            components.append(
                {
                    "color": int(color),
                    "coords": tuple(coords),
                    "bbox": (int(r_min), int(c_min), int(r_max), int(c_max)),
                }
            )

    return comp_ids, components


def _is_thin_cycle(component: dict[str, object], grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    color = int(component["color"])
    for r, c in component["coords"]:
        degree = 0
        for dr, dc in DIRS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(color):
                degree += 1
        if degree != 2:
            return False
    return True


def _hole_areas_by_component(grid: np.ndarray, bg: int, comp_ids: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    areas: dict[int, int] = defaultdict(int)

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) != int(bg):
                continue

            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            touches_border = False
            area = 0
            adjacent_components: set[int] = set()

            while queue:
                r, c = queue.popleft()
                area += 1
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True

                for dr, dc in DIRS4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if int(g[rr, cc]) == int(bg):
                        if not bool(seen[rr, cc]):
                            seen[rr, cc] = True
                            queue.append((rr, cc))
                    else:
                        cid = int(comp_ids[rr, cc])
                        if cid >= 0:
                            adjacent_components.add(cid)

            if not touches_border and len(adjacent_components) == 1:
                cid = next(iter(adjacent_components))
                areas[int(cid)] += int(area)

    return dict(areas)


def _paint_materialized_bar(
    canvas: np.ndarray,
    *,
    top: int,
    left: int,
    width: int,
    area: int,
    color: int,
) -> None:
    if area <= 0 or width <= 0:
        return

    height = int((int(area) + int(width) - 1) // int(width))
    remaining = int(area)
    for offset in range(height):
        row = int(top + height - 1 - offset)
        take = min(int(width), int(remaining))
        canvas[row, left : left + take] = int(color)
        remaining -= take
        if remaining == 0:
            break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find one-pixel monochrome loops, measure their enclosed background area,
    then materialize that area as a width-bounded bar below each loop.

    The bar starts one row below the loop bbox, uses the loop bbox width, and
    packs exactly A pixels bottom-aligned (full rows first, then a partial top
    row when A is not divisible by the width).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    comp_ids, components = _find_monochrome_components(g, bg)
    hole_areas = _hole_areas_by_component(g, bg, comp_ids)

    out = g.copy()
    ordered = sorted(
        (
            cid
            for cid, area in hole_areas.items()
            if area > 0 and _is_thin_cycle(components[int(cid)], g)
        ),
        key=lambda cid: tuple(int(x) for x in components[int(cid)]["bbox"]),
    )

    for cid in ordered:
        comp = components[int(cid)]
        area = int(hole_areas[int(cid)])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        width = int(c1 - c0 + 1)
        height = int((area + width - 1) // width)
        top = int(r1 + 2)
        if top < 0 or c0 < 0 or top + height > out.shape[0] or c1 >= out.shape[1]:
            continue
        _paint_materialized_bar(out, top=top, left=c0, width=width, area=area, color=int(comp["color"]))

    return out


def _draw_rect_loop(canvas: np.ndarray, *, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top, left : left + width] = int(color)
    canvas[top + height - 1, left : left + width] = int(color)
    canvas[top : top + height, left] = int(color)
    canvas[top : top + height, left + width - 1] = int(color)


def _sample_loop_template(grid_h: int, grid_w: int, object_count: int) -> dict[str, int] | None:
    ring_h_max = 10 if object_count <= 2 else 8
    ring_w_max = 10 if object_count <= 2 else 8
    max_h = min(int(ring_h_max), int(grid_h - 2 * CANVAS_MARGIN - 2))
    max_w = min(int(ring_w_max), int(grid_w - 2 * CANVAS_MARGIN))
    if max_h < 4 or max_w < 4:
        return None

    candidates: list[dict[str, int]] = []
    for ring_h in range(4, max_h + 1):
        for ring_w in range(4, max_w + 1):
            hole_area = int((ring_h - 2) * (ring_w - 2))
            bar_h = int((hole_area + ring_w - 1) // ring_w)
            patch_h = int(ring_h + 1 + bar_h)
            if patch_h <= int(grid_h - 2 * CANVAS_MARGIN):
                candidates.append(
                    {
                        "ring_h": int(ring_h),
                        "ring_w": int(ring_w),
                        "hole_area": int(hole_area),
                        "bar_h": int(bar_h),
                        "patch_h": int(patch_h),
                    }
                )

    if not candidates:
        return None

    want_partial = bool(int(RNG.integers(0, 2)))
    filtered = [cand for cand in candidates if (int(cand["hole_area"]) % int(cand["ring_w"]) != 0) == want_partial]
    pool = filtered if filtered else candidates
    return dict(pool[_randint(0, len(pool) - 1)])


def _place_specs(grid_h: int, grid_w: int, specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placed = [dict(spec) for spec in specs]
    order = sorted(range(len(placed)), key=lambda i: (-int(placed[i]["patch_h"]) * int(placed[i]["ring_w"]), i))

    for idx in order:
        spec = placed[idx]
        patch_h = int(spec["patch_h"])
        patch_w = int(spec["ring_w"])
        top_low = int(CANVAS_MARGIN)
        top_high = int(grid_h - patch_h - CANVAS_MARGIN)
        left_low = int(CANVAS_MARGIN)
        left_high = int(grid_w - patch_w - CANVAS_MARGIN)
        if top_high < top_low or left_high < left_low:
            return None

        success = False
        for _ in range(PLACEMENT_TRIES):
            top = _randint(top_low, top_high)
            left = _randint(left_low, left_high)
            r0 = max(0, int(top - PLACEMENT_MARGIN))
            r1 = min(grid_h, int(top + patch_h + PLACEMENT_MARGIN))
            c0 = max(0, int(left - PLACEMENT_MARGIN))
            c1 = min(grid_w, int(left + patch_w + PLACEMENT_MARGIN))
            if bool(blocked[r0:r1, c0:c1].any()):
                continue

            blocked[r0:r1, c0:c1] = True
            spec["top"] = int(top)
            spec["left"] = int(left)
            success = True
            break

        if not success:
            return None

    return placed


def _render_input(grid_h: int, grid_w: int, bg: int, specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    for spec in specs:
        _draw_rect_loop(
            grid,
            top=int(spec["top"]),
            left=int(spec["left"]),
            height=int(spec["ring_h"]),
            width=int(spec["ring_w"]),
            color=int(spec["color"]),
        )
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict[str, int]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _paint_materialized_bar(
            out,
            top=int(spec["top"]) + int(spec["ring_h"]) + 1,
            left=int(spec["left"]),
            width=int(spec["ring_w"]),
            area=int(spec["hole_area"]),
            color=int(spec["color"]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = _randint(*GRID_H_RANGE)
    grid_w = _randint(*GRID_W_RANGE)
    object_count = int(RNG.choice(OBJECT_COUNT_CHOICES, p=OBJECT_COUNT_WEIGHTS / OBJECT_COUNT_WEIGHTS.sum()))

    colors = list(range(10))
    bg = int(RNG.choice(np.array(colors, dtype=int)))
    fg_pool = [color for color in colors if int(color) != int(bg)]
    if len(fg_pool) < object_count:
        return None
    fg_colors = [int(color) for color in RNG.choice(np.array(fg_pool, dtype=int), size=object_count, replace=False)]

    specs: list[dict[str, int]] = []
    for color in fg_colors:
        template = _sample_loop_template(grid_h, grid_w, object_count)
        if template is None:
            return None
        template["color"] = int(color)
        specs.append(template)

    placed = _place_specs(grid_h, grid_w, specs)
    if placed is None:
        return None

    inp = _render_input(grid_h, grid_w, bg, placed)
    expected = _render_expected(inp, placed)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
