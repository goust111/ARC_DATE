from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 5
BLACK = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 600


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _direction_from_centroid(
    indicator: tuple[int, int],
    centroid: tuple[float, float],
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    ir, ic = indicator
    dr = ir - centroid[0]
    dc = ic - centroid[1]
    if abs(dr) > abs(dc):
        return _sign(dr), 0
    if abs(dc) > abs(dr):
        return 0, _sign(dc)
    r0, c0, r1, c1 = bbox
    if ir < r0:
        return -1, 0
    if ir > r1:
        return 1, 0
    if ic < c0:
        return 0, -1
    if ic > c1:
        return 0, 1
    raise ValueError("indicator direction is ambiguous")


def _single_color_component_size(grid: np.ndarray, start: tuple[int, int]) -> int:
    sr, sc = start
    target = int(grid[sr, sc])
    h, w = grid.shape
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == target:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen)


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    black_components = _extract_components(g == BLACK)
    if not black_components:
        raise ValueError("expected a black template")
    template = max(black_components, key=lambda comp: int(comp.shape[0]))
    bbox = _bbox(template)

    indicator_set = set()
    template_cells = {tuple(map(int, cell)) for cell in template}
    for r, c in template_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                value = int(g[nr, nc])
                if value not in (bg, BLACK) and (nr, nc) not in template_cells:
                    if _single_color_component_size(g, (nr, nc)) == 1:
                        indicator_set.add((nr, nc))
    indicators = sorted(indicator_set)
    if not indicators:
        raise ValueError("expected at least one indicator pixel")
    return template, bbox, indicators, bg


def _paint_tiles(
    grid: np.ndarray,
    template: np.ndarray,
    bbox: tuple[int, int, int, int],
    indicators: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    centroid = (float(template[:, 0].mean()), float(template[:, 1].mean()))
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    uniform_rows = {r for r in range(g.shape[0]) if np.all(g[r] == g[r, 0])}
    indicator_set = set(indicators)
    protected = {
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) not in (bg, BLACK) and (r, c) not in indicator_set
    }

    template_list = [tuple(map(int, cell)) for cell in template]
    for ir, ic in indicators:
        color = int(g[ir, ic])
        step_r, step_c = _direction_from_centroid((ir, ic), centroid, bbox)
        stride = height if step_r else width
        k = 1
        while True:
            shifted = [
                (r + step_r * stride * k, c + step_c * stride * k)
                for r, c in template_list
            ]
            if any(not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]) for rr, cc in shifted):
                break
            if any(rr in uniform_rows for rr, _cc in shifted):
                break
            for rr, cc in shifted:
                if (rr, cc) not in protected:
                    out[rr, cc] = color
            k += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the black template in each indicator's color and direction until a uniform row blocks the next copy.

    Input structure:
    - A mostly gray background contains one irregular black 4-connected template.
    - One or two single-pixel colored indicators touch the template orthogonally.
    - Two or three full rows are uniform single-color rows, and other rows may contain scattered colored pixels.

    Rule:
    - Find the black template and its touching one-pixel indicators.
    - For each indicator, infer the cardinal direction from the template centroid toward the indicator.
    - Reuse the template shape in the indicator color, translating by one template bbox span each step along that direction.
    - Do not draw a candidate copy if any of its pixels would fall on a row whose entire input row is one color; stop farther copies in that direction as well.
    - Keep the original template, uniform rows, and scattered non-indicator colored pixels unchanged.

    Key insight:
    - The blocking condition is structural row uniformity, not a specific barrier color, so colored but non-uniform rows do not stop the propagation.
    """
    template, bbox, indicators, bg = _parse_scene(np.asarray(grid, dtype=int))
    return _paint_tiles(np.asarray(grid, dtype=int), template, bbox, indicators, bg)


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    arr = np.array([(r - r0, c - c0) for r, c in cells], dtype=int)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _indicator_candidates(local_template: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    centroid = (float(local_template[:, 0].mean()), float(local_template[:, 1].mean()))
    r0, c0, r1, c1 = _bbox(local_template)
    cells = {tuple(map(int, cell)) for cell in local_template}
    candidates = {"up": [], "down": []}
    for r, c in cells:
        if (r - 1, c) not in cells:
            pos = (r - 1, c)
            if pos[0] >= r0:
                pos = None
            if pos is not None:
                try:
                    if _direction_from_centroid(pos, centroid, (r0, c0, r1, c1)) == (-1, 0):
                        candidates["up"].append(pos)
                except ValueError:
                    pass
        if (r + 1, c) not in cells:
            pos = (r + 1, c)
            if pos[0] <= r1:
                pos = None
            if pos is not None:
                try:
                    if _direction_from_centroid(pos, centroid, (r0, c0, r1, c1)) == (1, 0):
                        candidates["down"].append(pos)
                except ValueError:
                    pass
    for key in candidates:
        uniq = sorted(set(candidates[key]), key=lambda p: (abs(p[1] - centroid[1]), p[0], p[1]))
        candidates[key] = uniq
    return candidates


def _generate_template_shape() -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    for _ in range(200):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        if box_h * box_w <= 4:
            continue
        area = random.randint(4, min(7, box_h * box_w - 1))
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        template = _normalize_cells(cells)
        h = int(template[:, 0].max()) + 1
        w = int(template[:, 1].max()) + 1
        if h < 2 or w < 2 or len(cells) == h * w:
            continue
        candidates = _indicator_candidates(template)
        if candidates["up"] and candidates["down"]:
            return template, candidates
    raise RuntimeError("failed to generate a valid template shape")


def _choose_indicator_layout(
    template_h: int,
    grid_h: int,
    want_up: bool,
    want_down: bool,
) -> tuple[int, int, int, int]:
    up_choices = [0] if not want_up else [1, 2, 3]
    down_choices = [0] if not want_down else [1, 2, 3]
    feasible = []
    for n_up in up_choices:
        for n_down in down_choices:
            lower = n_up * 0 + (n_up + 1) * template_h
            upper = grid_h - 1 - (n_down + 1) * template_h
            if lower <= upper:
                feasible.append((n_up, n_down, lower, upper))
    if not feasible:
        raise ValueError("no feasible layout")
    n_up, n_down, lower, upper = random.choice(feasible)
    top = random.randint(lower, upper)
    return top, n_up, n_down, top + template_h - 1


def _choose_extra_barriers(
    total_rows: int,
    template_top: int,
    template_bottom: int,
    base_barriers: set[int],
    want_up: bool,
    want_down: bool,
) -> list[int]:
    target_total = random.choice([2, 3])
    need = max(0, target_total - len(base_barriers))
    if need == 0:
        return []

    pool = []
    for r in range(total_rows):
        if r in base_barriers or template_top - 1 <= r <= template_bottom + 1:
            continue
        if want_up and r < template_top:
            continue
        if want_down and r > template_bottom:
            continue
        pool.append(r)
    if len(pool) < need:
        return []
    return random.sample(pool, need)


def _scatter_columns(width: int, bbox_left: int, bbox_right: int) -> list[int]:
    cols = [c for c in range(width) if c < bbox_left - 1 or c > bbox_right + 1]
    return cols if cols else list(range(width))


def _fill_non_barrier_rows(
    grid: np.ndarray,
    barrier_rows: set[int],
    safe_cols: list[int],
) -> None:
    for r in range(grid.shape[0]):
        if r in barrier_rows or not np.all(grid[r] == BG):
            continue
        if random.random() < 0.35 and len(safe_cols) >= 3:
            cols = random.sample(safe_cols, random.randint(2, min(3, len(safe_cols))))
            main_color = random.choice(PALETTE)
            for c in cols[:-1]:
                grid[r, c] = main_color
            grid[r, cols[-1]] = random.choice([c for c in PALETTE if c != main_color])
        else:
            grid[r, random.choice(safe_cols)] = random.choice(PALETTE)


def _expected_copy_counts(
    template: np.ndarray,
    bbox: tuple[int, int, int, int],
    indicators: list[tuple[int, int]],
    uniform_rows: set[int],
    shape: tuple[int, int],
) -> dict[tuple[int, int], int]:
    centroid = (float(template[:, 0].mean()), float(template[:, 1].mean()))
    r0, c0, r1, c1 = bbox
    stride_v = r1 - r0 + 1
    stride_h = c1 - c0 + 1
    template_list = [tuple(map(int, cell)) for cell in template]
    out: dict[tuple[int, int], int] = {}
    for indicator in indicators:
        step_r, step_c = _direction_from_centroid(indicator, centroid, bbox)
        stride = stride_v if step_r else stride_h
        count = 0
        while True:
            k = count + 1
            shifted = [
                (r + step_r * stride * k, c + step_c * stride * k)
                for r, c in template_list
            ]
            if any(not (0 <= rr < shape[0] and 0 <= cc < shape[1]) for rr, cc in shifted):
                break
            if any(rr in uniform_rows for rr, _cc in shifted):
                break
            count += 1
        out[indicator] = count
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        template_local, candidates = _generate_template_shape()
        t_h = int(template_local[:, 0].max()) + 1
        t_w = int(template_local[:, 1].max()) + 1

        branch = random.random()
        if branch < 0.35:
            chosen_dirs = ["up"]
        elif branch < 0.7:
            chosen_dirs = ["down"]
        else:
            chosen_dirs = ["up", "down"]

        grid_h = random.randint(max(12, 4 * t_h + 2), 20)
        grid_w = random.randint(max(14, t_w + 8), 22)

        try:
            top, n_up, n_down, bottom = _choose_indicator_layout(
                t_h, grid_h, "up" in chosen_dirs, "down" in chosen_dirs
            )
        except ValueError:
            continue

        min_left = 2
        max_left = grid_w - t_w - 3
        if max_left < min_left:
            continue
        left = random.randint(min_left, max_left)

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        template_abs = template_local + np.array([top, left], dtype=int)
        for r, c in template_abs:
            grid[int(r), int(c)] = BLACK

        barrier_rows: set[int] = set()
        if "up" in chosen_dirs:
            barrier_rows.add(top - (n_up + 1) * t_h)
        if "down" in chosen_dirs:
            barrier_rows.add(top + (n_down + 1) * t_h)

        extra_barriers = _choose_extra_barriers(
            grid_h,
            top,
            bottom,
            barrier_rows,
            "up" in chosen_dirs,
            "down" in chosen_dirs,
        )
        if len(barrier_rows) + len(extra_barriers) < 2:
            continue
        barrier_rows.update(extra_barriers)

        barrier_colors = [random.choice(PALETTE) for _ in range(len(barrier_rows))]
        for row, color in zip(sorted(barrier_rows), barrier_colors):
            grid[row, :] = color

        indicator_cells = []
        indicator_colors = random.sample(PALETTE, len(chosen_dirs))
        for color, direction in zip(indicator_colors, chosen_dirs):
            local_pos = random.choice(candidates[direction][: min(2, len(candidates[direction]))])
            abs_pos = (top + local_pos[0], left + local_pos[1])
            if not (0 <= abs_pos[0] < grid_h and 0 <= abs_pos[1] < grid_w):
                break
            grid[abs_pos] = color
            indicator_cells.append(abs_pos)
        if len(indicator_cells) != len(chosen_dirs):
            continue

        safe_cols = _scatter_columns(grid_w, left, left + t_w - 1)
        _fill_non_barrier_rows(grid, barrier_rows, safe_cols)

        uniform_rows = {r for r in range(grid_h) if np.all(grid[r] == grid[r, 0])}
        if uniform_rows != barrier_rows:
            continue

        try:
            parsed_template, parsed_bbox, parsed_indicators, bg = _parse_scene(grid)
        except Exception:
            continue
        if bg != BG:
            continue
        if set(map(tuple, parsed_template.tolist())) != set(map(tuple, template_abs.tolist())):
            continue
        if sorted(parsed_indicators) != sorted(indicator_cells):
            continue

        parsed_counts = _expected_copy_counts(parsed_template, parsed_bbox, parsed_indicators, uniform_rows, grid.shape)
        if any(count == 0 for count in parsed_counts.values()):
            continue
        if "up" in chosen_dirs:
            up_indicator = min((p for p in parsed_indicators if p[0] < top), default=None)
            if up_indicator is None or parsed_counts[up_indicator] != n_up:
                continue
        if "down" in chosen_dirs:
            down_indicator = min((p for p in parsed_indicators if p[0] > bottom), default=None)
            if down_indicator is None or parsed_counts[down_indicator] != n_down:
                continue

        expected = _paint_tiles(grid, parsed_template, parsed_bbox, parsed_indicators, bg)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(actual, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
