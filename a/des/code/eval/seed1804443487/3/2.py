from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _legend_is_isolated(grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = grid.shape
    if r0 > 0 and np.any(grid[r0 - 1, c0 : c1 + 1] != bg):
        return False
    if r1 + 1 < h and np.any(grid[r1 + 1, c0 : c1 + 1] != bg):
        return False
    if c0 > 0 and np.any(grid[r0 : r1 + 1, c0 - 1] != bg):
        return False
    if c1 + 1 < w and np.any(grid[r0 : r1 + 1, c1 + 1] != bg):
        return False
    return True


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    mask = grid != bg
    candidates: list[tuple[tuple[int, int], tuple[int, int, int, int], list[int]]] = []

    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh not in (3, 4) or ww != 2:
            continue
        if len(comp) != hh * ww:
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner:
            continue
        if not _legend_is_isolated(grid, bg, r0, c0, r1, c1):
            continue

        row_colors: list[int] = []
        ok = True
        for r in range(r0, r1 + 1):
            vals = {int(v) for v in grid[r, c0 : c1 + 1]}
            if bg in vals or len(vals) != 1:
                ok = False
                break
            row_colors.append(next(iter(vals)))
        if not ok or len(set(row_colors)) != hh:
            continue

        # Prefer the true corner legend over accidental tiny edge structures.
        score = (hh, len(set(row_colors)))
        candidates.append((score, (r0, c0, r1, c1), row_colors))

    assert len(candidates) == 1, "legend must be uniquely identifiable"
    _, (r0, c0, r1, c1), row_colors = candidates[0]
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return row_colors, legend_mask


def _reconstruct_loops(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    for color in colors:
        pts_arr = np.argwhere((grid == color) & (~legend_mask))
        if pts_arr.size == 0:
            continue
        pts = [(int(r), int(c)) for r, c in pts_arr]
        r0, c0, r1, c1 = _bbox(pts)
        for r, c in pts:
            assert r in (r0, r1) or c in (c0, c1), "segment pixels must lie on rectangle edges"
        touches = {
            "top": any(r == r0 for r, _ in pts),
            "bottom": any(r == r1 for r, _ in pts),
            "left": any(c == c0 for _, c in pts),
            "right": any(c == c1 for _, c in pts),
        }
        assert all(touches.values()), "segments must determine all four rectangle sides"
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
    return out


def _find_enclosed_regions(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    free = (grid == bg) & (~legend_mask)
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(free):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adjacent_colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not legend_mask[nr, nc]:
                    val = int(grid[nr, nc])
                    if val != bg:
                        adjacent_colors.add(val)
        assert len(adjacent_colors) == 1, "each enclosed region must belong to one loop"
        regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored path fragments into rectangular loops, then fill each loop
    according to the corner legend after sorting loop interiors by area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_colors, legend_mask = _find_legend(g, bg)

    out = _reconstruct_loops(g, bg, legend_mask)
    regions = _find_enclosed_regions(out, bg, legend_mask)
    assert len(regions) == len(legend_colors), "legend size must match the number of loops"

    regions.sort(key=lambda comp: len(comp))
    for fill_color, comp in zip(legend_colors, regions, strict=True):
        for r, c in comp:
            out[r, c] = fill_color
    return out


GEN_RETRIES = 400


def _sample_colors(count: int) -> list[int]:
    return random.sample(list(range(10)), count)


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    if np.any(mask[rr0:rr1, cc0:cc1]):
        return False
    mask[rr0:rr1, cc0:cc1] = True
    return True


def _inner_interval(length: int) -> tuple[int, int]:
    seg_len = random.randint(2, min(5, length - 2))
    start = random.randint(1, length - seg_len - 1)
    return start, start + seg_len - 1


def _sample_rect_spec(mode: int, used_areas: set[int]) -> dict[str, int] | None:
    for _ in range(200):
        if mode == 2:
            family = random.choice(["vertical_pair", "horizontal_pair"])
            if family == "vertical_pair":
                h = random.randint(3, 5)
                w = random.randint(4, 10)
            else:
                h = random.randint(4, 10)
                w = random.randint(3, 5)
        elif mode == 3:
            family = random.choice(["vertical_pair_extra", "horizontal_pair_extra"])
            if family == "vertical_pair_extra":
                h = random.randint(3, 5)
                w = random.randint(6, 10)
            else:
                h = random.randint(6, 10)
                w = random.randint(3, 5)
        else:
            family = "four_sides"
            h = random.randint(6, 10)
            w = random.randint(6, 10)

        area = (h - 2) * (w - 2)
        if area <= 0 or area in used_areas:
            continue
        return {"mode": mode, "family": family, "h": h, "w": w, "area": area}
    return None


def _segment_cells(spec: dict[str, int]) -> list[tuple[int, int]]:
    h = int(spec["h"])
    w = int(spec["w"])
    family = spec["family"]
    cells: set[tuple[int, int]] = set()

    if family == "vertical_pair":
        for r in range(h):
            cells.add((r, 0))
            cells.add((r, w - 1))
    elif family == "horizontal_pair":
        for c in range(w):
            cells.add((0, c))
            cells.add((h - 1, c))
    elif family == "vertical_pair_extra":
        for r in range(h):
            cells.add((r, 0))
            cells.add((r, w - 1))
        start, end = _inner_interval(w)
        side_r = random.choice([0, h - 1])
        for c in range(start, end + 1):
            cells.add((side_r, c))
    elif family == "horizontal_pair_extra":
        for c in range(w):
            cells.add((0, c))
            cells.add((h - 1, c))
        start, end = _inner_interval(h)
        side_c = random.choice([0, w - 1])
        for r in range(start, end + 1):
            cells.add((r, side_c))
    else:
        top_s, top_e = _inner_interval(w)
        bot_s, bot_e = _inner_interval(w)
        left_s, left_e = _inner_interval(h)
        right_s, right_e = _inner_interval(h)
        for c in range(top_s, top_e + 1):
            cells.add((0, c))
        for c in range(bot_s, bot_e + 1):
            cells.add((h - 1, c))
        for r in range(left_s, left_e + 1):
            cells.add((r, 0))
        for r in range(right_s, right_e + 1):
            cells.add((r, w - 1))

    return sorted(cells)


def _legend_bbox(corner: str, h: int, w: int, count: int) -> tuple[int, int]:
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, w - 2
    if corner == "bottom_left":
        return h - count, 0
    return h - count, w - 2


def _draw_legend(grid: np.ndarray, corner: str, colors: list[int]) -> np.ndarray:
    h, w = grid.shape
    legend_mask = np.zeros_like(grid, dtype=bool)
    r0, c0 = _legend_bbox(corner, h, w, len(colors))
    for i, color in enumerate(colors):
        rr = r0 + i
        grid[rr, c0 : c0 + 2] = color
        legend_mask[rr, c0 : c0 + 2] = True
    return legend_mask


def _place_rectangles(h: int, w: int, specs: list[dict[str, int]], legend_mask: np.ndarray) -> list[dict[str, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    legend_pts = np.argwhere(legend_mask)
    lr0 = int(legend_pts[:, 0].min())
    lc0 = int(legend_pts[:, 1].min())
    lr1 = int(legend_pts[:, 0].max())
    lc1 = int(legend_pts[:, 1].max())
    if not _reserve(occupied, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=2):
        return None

    placed: list[dict[str, int]] = []
    for spec in sorted(specs, key=lambda item: item["h"] * item["w"], reverse=True):
        rh = int(spec["h"])
        rw = int(spec["w"])
        if h - rh - 1 < 1 or w - rw - 1 < 1:
            return None
        success = False
        for _ in range(300):
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            if _reserve(occupied, r0, c0, rh, rw, pad=1):
                placed.append({**spec, "r0": r0, "c0": c0})
                success = True
                break
        if not success:
            return None
    return placed


def _render_input(h: int, w: int, bg: int, corner: str, legend_colors: list[int], rects: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    _draw_legend(grid, corner, legend_colors)
    for rect in rects:
        for dr, dc in _segment_cells(rect):
            grid[int(rect["r0"]) + dr, int(rect["c0"]) + dc] = int(rect["color"])
    return grid


def _render_expected(h: int, w: int, bg: int, corner: str, legend_colors: list[int], rects: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    _draw_legend(grid, corner, legend_colors)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        r1 = r0 + int(rect["h"]) - 1
        c1 = c0 + int(rect["w"]) - 1
        color = int(rect["color"])
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color

    ranked = sorted(rects, key=lambda rect: int(rect["area"]))
    for fill_color, rect in zip(legend_colors, ranked, strict=True):
        r0 = int(rect["r0"]) + 1
        c0 = int(rect["c0"]) + 1
        r1 = int(rect["r0"]) + int(rect["h"]) - 1
        c1 = int(rect["c0"]) + int(rect["w"]) - 1
        grid[r0:r1, c0:c1] = fill_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    loop_count = random.choice([3, 4])
    if loop_count == 3:
        h = random.randint(18, 24)
        w = random.randint(18, 24)
    else:
        h = random.randint(22, 28)
        w = random.randint(22, 28)

    colors = _sample_colors(1 + 2 * loop_count)
    bg = colors[0]
    legend_colors = colors[1 : 1 + loop_count]
    loop_colors = colors[1 + loop_count :]
    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    blank = np.full((h, w), bg, dtype=int)
    legend_mask = _draw_legend(blank, corner, legend_colors)

    modes = [2, 3, 4]
    if loop_count == 4:
        modes.append(random.choice([2, 3, 4]))
    random.shuffle(modes)

    specs: list[dict[str, int]] = []
    used_areas: set[int] = set()
    for mode, color in zip(modes, loop_colors, strict=True):
        spec = _sample_rect_spec(mode, used_areas)
        if spec is None:
            return None
        used_areas.add(int(spec["area"]))
        specs.append({**spec, "color": color})

    rects = _place_rectangles(h, w, specs, legend_mask)
    if rects is None:
        return None

    inp = _render_input(h, w, bg, corner, legend_colors, rects)
    expected = _render_expected(h, w, bg, corner, legend_colors, rects)

    if len(np.unique([int(rect["area"]) for rect in rects])) != loop_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
