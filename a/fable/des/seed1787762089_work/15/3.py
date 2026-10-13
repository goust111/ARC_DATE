import random
from collections import Counter

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (1, 2, 3, 4, 6, 7, 8)
GRID_MIN = 12
GRID_MAX = 18
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
OBJECT_SIZE_MIN = 3
OBJECT_SIZE_MAX = 8
DOT_COUNT_MIN = 1
DOT_COUNT_MAX = 5
SHAPE_MAX_H = 5
SHAPE_MAX_W = 4
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D grid")
    if g.size == 0:
        return g.copy()

    bg = _background_color(g)
    out = np.full(g.shape, bg, dtype=int)
    top_counts = Counter(int(color) for color in g[0] if int(color) != bg)

    for color in np.unique(g[1:]):
        color = int(color)
        if color == bg:
            continue
        shift = int(top_counts.get(color, 0))
        coords = np.argwhere(g[1:] == color)
        for rr, cc in coords:
            nr = int(rr) + 1 + shift
            if nr < g.shape[0]:
                out[nr, int(cc)] = color

    return out


def _normalize(cells) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in pts))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = list(cells)
    variants: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                transformed = []
                for r, c in pts:
                    rr, cc = (c, r) if swap else (r, c)
                    transformed.append((sr * rr, sc * cc))
                norm = _normalize(transformed)
                if norm not in seen:
                    seen.add(norm)
                    variants.append(norm)
    return variants


def _random_shape(size: int) -> tuple[tuple[int, int], ...]:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    cand = (r + dr, c + dc)
                    if cand in cells:
                        continue
                    trial = _normalize(cells | {cand})
                    h, w = _shape_dims(trial)
                    if h <= SHAPE_MAX_H and w <= SHAPE_MAX_W:
                        frontier.add(cand)
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != size:
            continue
        shape = random.choice(_shape_variants(_normalize(cells)))
        h, w = _shape_dims(shape)
        if h <= SHAPE_MAX_H and w <= SHAPE_MAX_W:
            return shape
    raise RuntimeError("failed to sample a connected shape")


def _bounded_composition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    values: list[int] = []
    remaining = int(total)
    for i in range(parts):
        slots = parts - i - 1
        low = max(lo, remaining - hi * slots)
        high = min(hi, remaining - lo * slots)
        if low > high:
            return None
        values.append(random.randint(low, high))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _nonnegative_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(int(cut - prev - 1))
        prev = cut
    return out


def _isolated_positions(width: int, count: int) -> list[int]:
    bases = sorted(random.sample(range(width - count + 1), count))
    return [int(base + i) for i, base in enumerate(bases)]


def _assign_dot_colors(positions: list[int], specs: list[dict]) -> list[int] | None:
    dots: list[int] = []
    for spec in specs:
        dots.extend([int(spec["color"])] * int(spec["dot_count"]))

    for _ in range(200):
        random.shuffle(dots)
        ok = True
        for spec in specs:
            left = int(spec["left"])
            right = left + int(spec["width"]) - 1
            cols = [positions[i] for i, color in enumerate(dots) if int(color) == int(spec["color"])]
            if cols and all(left <= col <= right for col in cols):
                ok = False
                break
        if ok:
            return list(dots)

    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    colors = random.sample(PALETTE, n_objects)

    max_top_dots = (w + 1) // 2
    counts = _bounded_composition(
        total=random.randint(n_objects, min(max_top_dots, n_objects * DOT_COUNT_MAX)),
        parts=n_objects,
        lo=DOT_COUNT_MIN,
        hi=DOT_COUNT_MAX,
    )
    if counts is None:
        return None

    specs: list[dict] = []
    widths: list[int] = []
    total_pixels = 0
    for color, dot_count in zip(colors, counts):
        size = random.randint(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX)
        shape = _random_shape(size)
        shape_h, shape_w = _shape_dims(shape)
        specs.append(
            {
                "color": int(color),
                "dot_count": int(dot_count),
                "shape": shape,
                "height": int(shape_h),
                "width": int(shape_w),
            }
        )
        widths.append(int(shape_w))
        total_pixels += size

    required_width = sum(widths) + (n_objects - 1)
    if required_width > w:
        return None

    gap_slack = _nonnegative_composition(w - required_width, n_objects + 1)
    cursor = gap_slack[0]
    for i, spec in enumerate(specs):
        spec["left"] = int(cursor)
        cursor += int(spec["width"]) + 1 + gap_slack[i + 1]

    want_clipping = random.random() < 0.5
    clipped_specs = set()
    if want_clipping:
        clipped_specs = set(random.sample(range(n_objects), random.randint(1, n_objects)))

    clipped_pixels = 0
    for idx, spec in enumerate(specs):
        shape_h = int(spec["height"])
        shift = int(spec["dot_count"])
        if idx in clipped_specs:
            row_low = max(2, h - shape_h - shift + 1)
            row_high = h - shape_h
        else:
            row_low = 2
            row_high = h - shape_h - shift
        if row_low > row_high:
            return None
        top = random.randint(row_low, row_high)
        spec["top"] = int(top)
        for dr, _ in spec["shape"]:
            if top + int(dr) + shift >= h:
                clipped_pixels += 1

    total_dots = sum(counts)
    positions = _isolated_positions(w, total_dots)
    dot_colors = _assign_dot_colors(positions, specs)
    if dot_colors is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        color = int(spec["color"])
        for dr, dc in spec["shape"]:
            grid[top + int(dr), left + int(dc)] = color
    for col, color in zip(positions, dot_colors):
        grid[0, int(col)] = int(color)

    if np.any(grid[1] != 0):
        return None

    row0_counts = Counter(int(color) for color in grid[0] if int(color) != 0)
    if {int(spec["color"]) for spec in specs} != set(row0_counts):
        return None

    for spec in specs:
        color = int(spec["color"])
        if row0_counts[color] != int(spec["dot_count"]):
            return None

    for col in positions:
        if col > 0 and grid[0, col - 1] != 0:
            return None
        if col + 1 < w and grid[0, col + 1] != 0:
            return None

    spans = []
    for spec in specs:
        color = int(spec["color"])
        body_mask = grid[1:] == color
        comps = _components(body_mask)
        if len(comps) != 1:
            return None
        area = int(comps[0].shape[0])
        if not (OBJECT_SIZE_MIN <= area <= OBJECT_SIZE_MAX):
            return None
        cols = comps[0][:, 1]
        spans.append((int(cols.min()), int(cols.max())))

    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1 :]:
            if not (a1 < b0 or b1 < a0):
                return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out[0] != 0):
        return None
    if int(np.count_nonzero(out)) != total_pixels - clipped_pixels:
        return None
    if want_clipping != (clipped_pixels > 0):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
