from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS = (-1, 1)
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 800


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_monochrome(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _components_monochrome(g, bg)
    assert comps, "expected at least one foreground rectangle"
    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors)), "foreground rectangles must use distinct colors"

    rects: list[dict[str, int]] = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        area = int(comp["area"])
        assert area == hh * ww, "every foreground object must be a solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be uniformly filled"
        rects.append(
            {
                "id": idx,
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "h": hh,
                "w": ww,
                "area": area,
            }
        )

    areas = [rect["area"] for rect in rects]
    assert len(areas) == len(set(areas)), "rectangle areas must be pairwise distinct"
    return bg, rects


def _clip_box(shape: tuple[int, int], r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int] | None:
    hh, ww = shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(hh, int(r0 + h))
    cc1 = min(ww, int(c0 + w))
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    return rr0, cc0, rr1, cc1


def _order_rects(rects: list[dict[str, int]], order: str) -> list[dict[str, int]]:
    if order == "area_desc":
        key = lambda rect: (-rect["area"], rect["r0"], rect["c0"], rect["color"])
    elif order == "area_asc":
        key = lambda rect: (rect["area"], rect["r0"], rect["c0"], rect["color"])
    elif order == "top_left":
        key = lambda rect: (rect["r0"], rect["c0"], -rect["area"], rect["color"])
    else:
        raise ValueError(f"unsupported order: {order}")
    return sorted(rects, key=key)


def _simulate(
    grid: np.ndarray,
    *,
    order: str = "area_desc",
    stop_on_overlap: bool = True,
    return_events: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, int]]] | np.ndarray:
    bg, rects = _extract_rectangles(grid)
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    occupied = g != bg
    occupied_kind = np.zeros((h, w), dtype=np.int8)
    occupied_kind[occupied] = 1  # 1 = original, 2 = stamp
    events: list[dict[str, object]] = []

    for rect in _order_rects(rects, order):
        for direction in DIRS:
            step = 1
            while True:
                rr = rect["r0"] + step * rect["h"]
                cc = rect["c0"] + direction * step * rect["w"]
                clip = _clip_box((h, w), rr, cc, rect["h"], rect["w"])
                if clip is None:
                    events.append(
                        {
                            "rect_id": rect["id"],
                            "rect_color": rect["color"],
                            "area": rect["area"],
                            "dir": direction,
                            "step": step,
                            "status": "out",
                        }
                    )
                    break

                rr0, cc0, rr1, cc1 = clip
                partial = rr0 != rr or cc0 != cc or rr1 != rr + rect["h"] or cc1 != cc + rect["w"]
                overlap_mask = occupied[rr0:rr1, cc0:cc1]
                if stop_on_overlap and bool(overlap_mask.any()):
                    overlap_kind = occupied_kind[rr0:rr1, cc0:cc1][overlap_mask]
                    cause = "original" if int(np.any(overlap_kind == 1)) else "stamp"
                    events.append(
                        {
                            "rect_id": rect["id"],
                            "rect_color": rect["color"],
                            "area": rect["area"],
                            "dir": direction,
                            "step": step,
                            "status": "blocked",
                            "cause": cause,
                            "partial": partial,
                        }
                    )
                    break

                out[rr0:rr1, cc0:cc1] = rect["color"]
                occupied[rr0:rr1, cc0:cc1] = True
                occupied_kind[rr0:rr1, cc0:cc1] = 2
                events.append(
                    {
                        "rect_id": rect["id"],
                        "rect_color": rect["color"],
                        "area": rect["area"],
                        "dir": direction,
                        "step": step,
                        "status": "success",
                        "partial": partial,
                    }
                )
                step += 1

    if return_events:
        return out, events, rects
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Process the solid rectangles by descending area and let each one stamp itself down both diagonals until overlap blocks that side.

    Input structure:
    - The grid has a single uniform background color.
    - Every foreground object is a color-distinct solid axis-aligned rectangle.
    - The rectangles have pairwise different areas.

    Rule:
    - Keep all original rectangles.
    - Process rectangles from largest area to smallest area.
    - For a rectangle of height h and width w, stamp the same rectangle at offsets
      (h, -w), (2h, -2w), ... and (h, +w), (2h, +2w), ... .
    - If a candidate stamp overlaps any already colored cell, that one trajectory
      stops immediately.
    - If a candidate stamp is partly outside the grid, clip the out-of-bounds part
      and keep the visible part; if it is completely outside, that trajectory stops.
    - The two diagonal trajectories of the same rectangle are independent.

    Key insight:
    - The rectangle's own size defines both the stamp shape and the diagonal stride,
      while the correct output also depends on global processing order because
      earlier large rectangles can occlude later smaller trajectories.
    """
    return _simulate(grid, order="area_desc", stop_on_overlap=True)


def _make_grid(shape: tuple[int, int], rect_specs: list[tuple[int, int, int, int, int]], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, r0, c0, h, w in rect_specs:
        grid[r0 : r0 + h, c0 : c0 + w] = color
    return grid


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    return not (ar0 + ah <= br0 or br0 + bh <= ar0 or ac0 + aw <= bc0 or bc0 + bw <= ac0)


def _count_side_successes(events: list[dict[str, object]], rect_id: int, direction: int) -> int:
    return sum(1 for event in events if event["rect_id"] == rect_id and event["dir"] == direction and event["status"] == "success")


def _first_side_block(events: list[dict[str, object]], rect_id: int, direction: int) -> dict[str, object] | None:
    for event in events:
        if event["rect_id"] == rect_id and event["dir"] == direction and event["status"] == "blocked":
            return event
    return None


def _candidate_specs() -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(2, 6):
        for w in range(2, 6):
            dims.append((h, w))
    return dims


DIM_CHOICES = _candidate_specs()


def _choose_dims_for_roles() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    for _ in range(200):
        a_h, a_w = random.choice([dim for dim in DIM_CHOICES if dim[0] * dim[1] >= 9])
        a_area = a_h * a_w
        b_options = [dim for dim in DIM_CHOICES if 5 <= dim[0] * dim[1] < a_area]
        if not b_options:
            continue
        b_h, b_w = random.choice(b_options)
        b_area = b_h * b_w
        c_options = [
            dim
            for dim in DIM_CHOICES
            if dim[0] * dim[1] < b_area and dim[0] * dim[1] not in {a_area, b_area}
        ]
        if not c_options:
            continue
        c_h, c_w = random.choice(c_options)
        if len({a_area, b_area, c_h * c_w}) == 3:
            return (a_h, a_w), (b_h, b_w), (c_h, c_w)
    return None


def _build_attempt() -> np.ndarray | None:
    dims = _choose_dims_for_roles()
    if dims is None:
        return None
    (a_h, a_w), (b_h, b_w), (c_h, c_w) = dims
    h = random.randint(21, 29)
    w = random.randint(21, 29)
    bg = 0
    free_dir = random.choice(DIRS)
    block_dir = -free_dir
    k_hit = random.choice((2, 3))
    k_block = random.choice((1, 2))
    b_step = random.choice((1, 2))
    colors = random.sample(COLOR_POOL, 3)
    color_a, color_b, color_c = colors

    min_c = max(k_block * a_w, 1)
    max_c = min(w - (k_hit + 1) * a_w, w - a_w - 1)
    if free_dir < 0:
        min_c = max(k_hit * a_w, min_c)
        max_c = min(max_c, w - (k_block + 1) * a_w)
    else:
        min_c = max(min_c, k_block * a_w)
        max_c = min(max_c, w - (k_hit + 1) * a_w)
    if min_c > max_c:
        return None

    min_r = 2
    max_r = h - max(k_hit * a_h + 1, k_block * a_h + c_h + 1, 4 * a_h)
    if max_r < min_r:
        return None

    a_r0 = random.randint(min_r, max_r)
    a_c0 = random.randint(min_c, max_c)

    hit_r = a_r0 + k_hit * a_h
    hit_c = a_c0 + free_dir * k_hit * a_w
    if not (0 <= hit_c and hit_c + a_w <= w):
        return None

    block_r = a_r0 + k_block * a_h
    block_c = a_c0 + block_dir * k_block * a_w
    if not (0 <= block_c and block_c + a_w <= w):
        return None

    b_r0 = hit_r - b_step * b_h
    b_c0 = hit_c - free_dir * b_step * b_w
    c_r0 = block_r
    c_c0 = block_c

    rect_a = (a_r0, a_c0, a_h, a_w)
    rect_b = (b_r0, b_c0, b_h, b_w)
    rect_c = (c_r0, c_c0, c_h, c_w)

    for r0, c0, hh, ww in (rect_a, rect_b, rect_c):
        if r0 < 0 or c0 < 0 or r0 + hh > h or c0 + ww > w:
            return None

    if _rectangles_overlap(rect_a, rect_b) or _rectangles_overlap(rect_a, rect_c) or _rectangles_overlap(rect_b, rect_c):
        return None

    if not (b_r0 < a_r0):
        return None

    rect_specs = [
        (color_a, a_r0, a_c0, a_h, a_w),
        (color_b, b_r0, b_c0, b_h, b_w),
        (color_c, c_r0, c_c0, c_h, c_w),
    ]
    grid = _make_grid((h, w), rect_specs, bg)
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out, events, rects = _simulate(grid, return_events=True)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    success_events = [event for event in events if event["status"] == "success"]
    if len(success_events) < 4:
        return False

    if not any(bool(event.get("partial")) for event in success_events):
        return False

    blocked_original = [event for event in events if event["status"] == "blocked" and event.get("cause") == "original"]
    blocked_stamp = [event for event in events if event["status"] == "blocked" and event.get("cause") == "stamp"]
    if not blocked_original or not blocked_stamp:
        return False

    has_independent_sides = False
    for rect in rects:
        left_block = _first_side_block(events, rect["id"], -1)
        right_block = _first_side_block(events, rect["id"], 1)
        left_successes = _count_side_successes(events, rect["id"], -1)
        right_successes = _count_side_successes(events, rect["id"], 1)
        if (left_block is not None and right_successes > 0) or (right_block is not None and left_successes > 0):
            has_independent_sides = True
            break
    if not has_independent_sides:
        return False

    asc = _simulate(grid, order="area_asc", stop_on_overlap=True)
    if np.array_equal(asc, out):
        return False

    top_left = _simulate(grid, order="top_left", stop_on_overlap=True)
    if np.array_equal(top_left, out):
        return False

    no_stop = _simulate(grid, order="area_desc", stop_on_overlap=False)
    if np.array_equal(no_stop, out):
        return False

    _, rects_again = _extract_rectangles(grid)
    area_rank = [rect["id"] for rect in sorted(rects_again, key=lambda rect: -rect["area"])]
    row_rank = [rect["id"] for rect in sorted(rects_again, key=lambda rect: (rect["r0"], rect["c0"]))]
    if area_rank == row_rank:
        return False

    new_pixels = int(np.count_nonzero((out != _get_bg(grid)) & (grid == _get_bg(grid))))
    if new_pixels < 12:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is None:
            continue
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for the diagonal train puzzle")
