from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIVIDER_COLOR = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, DIVIDER_COLOR))
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and np.all(grid[r] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected exactly one non-black full-width divider row"
    return int(rows[0])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(coords))
    comps.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))
    return comps


def _coords_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _tail_is_clean(
    coord_set: set[tuple[int, int]],
    base: tuple[int, int],
    tail: list[tuple[int, int]],
) -> bool:
    for i, cell in enumerate(tail):
        allowed = set()
        if i == 0:
            allowed.add(base)
        if i > 0:
            allowed.add(tail[i - 1])
        if i + 1 < len(tail):
            allowed.add(tail[i + 1])
        for nbr in _neighbors4(*cell):
            if nbr in coord_set and nbr not in allowed:
                return False
    return True


def _split_body_and_tail(
    grid: np.ndarray, component: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    coord_set = set(component)
    colors = {coord: int(grid[coord]) for coord in component}
    candidates = []

    for base in component:
        base_color = colors[base]
        for dr, dc in DIRS:
            first = (base[0] + dr, base[1] + dc)
            if first not in coord_set or colors[first] != base_color:
                continue
            tail: list[tuple[int, int]] = []
            cur = first
            while cur in coord_set and colors[cur] == base_color:
                tail.append(cur)
                cur = (cur[0] + dr, cur[1] + dc)
            tail_set = set(tail)
            body = coord_set - tail_set
            if not tail:
                continue
            if not (2 <= len(body) <= 6):
                continue
            if not _coords_connected(body):
                continue
            if not _tail_is_clean(coord_set, base, tail):
                continue
            candidates.append((sorted(body), (dr, dc), len(tail), tuple(sorted(tail))))

    assert candidates, "component must contain a clean straight tail"
    best_length = max(length for _, _, length, _ in candidates)
    best = [
        (body, direction, length)
        for body, direction, length, _ in candidates
        if length == best_length
    ]
    unique = {(tuple(body), direction, length): (body, direction, length) for body, direction, length in best}
    assert len(unique) == 1, "longest valid tail decomposition must be unique"
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each upper-half object body across the lower canvas using its attached tail as a direction-and-spacing guide.

    Input structure:
    - Background is black.
    - A single full-width grey row splits the grid into an upper object area and an equally tall lower blank canvas.
    - The upper half contains several 4-connected multicolor objects, each with a same-color one-cell-wide straight tail attached to exactly one body pixel.
    - Tails are horizontal or vertical, stay inside the upper half, and do not touch other objects.

    Rule:
    - Keep the upper half and the grey divider unchanged.
    - For each upper-half component, separate the straight tail from the object body and measure the tail length N.
    - Copy only the body to the corresponding location in the lower half.
    - Stamp additional body copies on the lower half by translating the whole body by multiples of N along the tail direction, stopping before a copy would leave the lower canvas.

    Key insight:
    The tail encodes two independent parameters at once: its axis gives the repetition direction, and its length gives the translation step used to place repeated body stamps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    split = _find_divider_row(g)
    upper = g[:split, :]
    lower = g[split + 1 :, :]
    assert upper.shape == lower.shape

    mask = upper != BG
    components = _find_components(mask)
    out = g.copy()
    out[split + 1 :, :] = BG

    lower_h, lower_w = lower.shape
    for component in components:
        body, (dr, dc), step = _split_body_and_tail(g, component)
        body_colors = {coord: int(g[coord]) for coord in body}
        k = 0
        while True:
            shifted = [
                (r + k * step * dr, c + k * step * dc)
                for r, c in body
            ]
            if not all(0 <= r < lower_h and 0 <= c < lower_w for r, c in shifted):
                break
            for (src_r, src_c), (dst_r, dst_c) in zip(body, shifted):
                out[split + 1 + dst_r, dst_c] = body_colors[(src_r, src_c)]
            k += 1

    return out


BODY_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)

RNG = np.random.default_rng()
MAX_RETRIES = 500
OBJECT_TRIES = 250
UPPER_H_RANGE = (8, 15)
WIDTH_MAX_EXCLUSIVE = 29
OBJECT_COUNT_RANGE = (2, 5)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    coords = list(shape)
    for _ in range(int(RNG.integers(0, 4))):
        coords = [(-c, r) for r, c in coords]
    if bool(RNG.integers(0, 2)):
        coords = [(r, -c) for r, c in coords]
    return _normalize(coords)


def _eligible_bases(
    shape: tuple[tuple[int, int], ...], dr: int, dc: int
) -> list[tuple[int, int]]:
    shape_set = set(shape)
    bh = 1 + max(r for r, _ in shape)
    bw = 1 + max(c for _, c in shape)
    limit = max(bh, bw) + 2
    bases = []
    for br, bc in shape:
        if (br + dr, bc + dc) in shape_set:
            continue
        blocked = False
        for rr, cc in shape:
            if dr != 0 and cc == bc and (rr - br) * dr > 0:
                blocked = True
                break
            if dc != 0 and rr == br and (cc - bc) * dc > 0:
                blocked = True
                break
        if not blocked:
            clean = True
            for i in range(1, limit + 1):
                cell = (br + i * dr, bc + i * dc)
                prev = (br + (i - 1) * dr, bc + (i - 1) * dc)
                for nbr in _neighbors4(*cell):
                    if nbr in shape_set and nbr != prev:
                        clean = False
                        break
                if not clean:
                    break
            if not clean:
                continue
            bases.append((br, bc))
    return bases


def _sample_body_colors(
    shape: tuple[tuple[int, int], ...], base: tuple[int, int]
) -> dict[tuple[int, int], int]:
    palette = list(RNG.permutation(PALETTE))
    base_color = int(palette[0])
    other_palette = [int(v) for v in palette[1:]]
    color_count = int(RNG.integers(2, min(5, len(shape)) + 1))
    non_base_choices = other_palette[: color_count - 1]
    colors = {base: base_color}
    for cell in shape:
        if cell == base:
            continue
        colors[cell] = int(non_base_choices[int(RNG.integers(0, len(non_base_choices)))])
    return colors


def _coords_clear(mask: np.ndarray, coords: list[tuple[int, int]], gap: int) -> bool:
    h, w = mask.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - gap)
        r1 = min(h, r + gap + 1)
        c0 = max(0, c - gap)
        c1 = min(w, c + gap + 1)
        if bool(mask[r0:r1, c0:c1].any()):
            return False
    return True


def _compute_copies(
    body_abs: list[tuple[int, int]], direction: tuple[int, int], step: int, upper_h: int, width: int
) -> list[list[tuple[int, int]]]:
    dr, dc = direction
    copies: list[list[tuple[int, int]]] = []
    k = 0
    while True:
        shifted = [(r + k * step * dr, c + k * step * dc) for r, c in body_abs]
        if not all(0 <= r < upper_h and 0 <= c < width for r, c in shifted):
            break
        copies.append(shifted)
        k += 1
    return copies


def _paint_expected(
    grid: np.ndarray,
    objects: list[dict[str, object]],
    split: int,
    upper_h: int,
    width: int,
) -> np.ndarray:
    out = grid.copy()
    out[split + 1 :, :] = BG
    for spec in objects:
        body_abs = spec["body_abs"]
        body_colors = spec["body_colors"]
        direction = spec["direction"]
        step = spec["step"]
        for copy_coords in _compute_copies(body_abs, direction, step, upper_h, width):
            for (src_r, src_c), (dst_r, dst_c) in zip(body_abs, copy_coords):
                out[split + 1 + dst_r, dst_c] = body_colors[(src_r, src_c)]
    return out


def _gen_attempt() -> np.ndarray | None:
    upper_h = int(RNG.integers(*UPPER_H_RANGE))
    object_count = int(RNG.integers(*OBJECT_COUNT_RANGE))
    width_min = max(14, object_count * 5 + 2)
    if width_min >= WIDTH_MAX_EXCLUSIVE:
        return None
    width = int(RNG.integers(width_min, WIDTH_MAX_EXCLUSIVE))
    height = 2 * upper_h + 1
    split = upper_h

    grid = np.full((height, width), BG, dtype=int)
    grid[split, :] = DIVIDER_COLOR

    upper_occ = np.zeros((upper_h, width), dtype=bool)
    lower_occ = np.zeros((upper_h, width), dtype=bool)
    objects: list[dict[str, object]] = []
    directions = [DIRS[int(i)] for i in RNG.permutation(len(DIRS))[:object_count]]

    for direction in directions:
        placed = False
        dr, dc = direction
        for _ in range(OBJECT_TRIES):
            shape = _transform_shape(BODY_TEMPLATES[int(RNG.integers(0, len(BODY_TEMPLATES)))])
            bases = _eligible_bases(shape, dr, dc)
            if not bases:
                continue
            base = bases[int(RNG.integers(0, len(bases)))]
            bh = 1 + max(r for r, _ in shape)
            bw = 1 + max(c for _, c in shape)
            min_step = (bw + 1) if dc != 0 else (bh + 1)
            max_step = (width - bw) if dc != 0 else (upper_h - bh)
            if max_step < min_step:
                continue
            step = int(RNG.integers(min_step, max_step + 1))

            if dr == -1:
                top_lo, top_hi = step, upper_h - bh
                left_lo, left_hi = 0, width - bw
            elif dr == 1:
                top_lo, top_hi = 0, upper_h - bh - step
                left_lo, left_hi = 0, width - bw
            elif dc == -1:
                top_lo, top_hi = 0, upper_h - bh
                left_lo, left_hi = step, width - bw
            else:
                top_lo, top_hi = 0, upper_h - bh
                left_lo, left_hi = 0, width - bw - step
            if top_lo > top_hi or left_lo > left_hi:
                continue

            top = int(RNG.integers(top_lo, top_hi + 1))
            left = int(RNG.integers(left_lo, left_hi + 1))
            body_abs = [(top + r, left + c) for r, c in shape]
            tail_abs = [(top + base[0] + i * dr, left + base[1] + i * dc) for i in range(1, step + 1)]
            comp_abs = body_abs + tail_abs
            if len(set(comp_abs)) != len(comp_abs):
                continue
            if not _coords_clear(upper_occ, comp_abs, gap=1):
                continue

            copies = _compute_copies(body_abs, direction, step, upper_h, width)
            flat_copies = [coord for copy in copies for coord in copy]
            if len(copies) < 2:
                continue
            if len(set(flat_copies)) != len(flat_copies):
                continue
            if not _coords_clear(lower_occ, flat_copies, gap=0):
                continue

            body_colors_local = _sample_body_colors(shape, base)
            body_colors_abs = {
                (top + r, left + c): int(color) for (r, c), color in body_colors_local.items()
            }

            comp_grid = np.zeros((upper_h, width), dtype=int)
            for r, c in body_abs:
                comp_grid[r, c] = body_colors_abs[(r, c)]
            base_color = body_colors_local[base]
            for r, c in tail_abs:
                comp_grid[r, c] = base_color
            comp_only = _find_components(comp_grid != BG)
            if len(comp_only) != 1:
                continue
            try:
                parsed_body, parsed_dir, parsed_step = _split_body_and_tail(comp_grid, comp_only[0])
            except AssertionError:
                continue
            if set(parsed_body) != set(body_abs) or parsed_dir != direction or parsed_step != step:
                continue

            for r, c in body_abs:
                grid[r, c] = body_colors_abs[(r, c)]
                upper_occ[r, c] = True
            for r, c in tail_abs:
                grid[r, c] = base_color
                upper_occ[r, c] = True
            for r, c in flat_copies:
                lower_occ[r, c] = True

            objects.append(
                {
                    "body_abs": sorted(body_abs),
                    "body_colors": body_colors_abs,
                    "direction": direction,
                    "step": step,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _paint_expected(grid, objects, split, upper_h, width)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
