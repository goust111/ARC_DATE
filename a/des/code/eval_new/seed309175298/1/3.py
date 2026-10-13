import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIVIDER_COLOR = 5
OBSTACLE_COLOR = 9
PALETTE = tuple(range(10))
FREE_COLORS = tuple(color for color in PALETTE if color not in (DIVIDER_COLOR, OBSTACLE_COLOR))
MAX_GENERATION_ATTEMPTS = 400


def _coords_from_rows(rows: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    coords = np.argwhere(mask)
    center = np.array(mask.shape) // 2
    return tuple(sorted((int(r - center[0]), int(c - center[1])) for r, c in coords))


PATTERN_LIBRARY = (
    _coords_from_rows(("010", "111", "010")),
    _coords_from_rows(("111", "010", "010")),
    _coords_from_rows(("010", "010", "111")),
    _coords_from_rows(("011", "010", "110")),
    _coords_from_rows(("110", "010", "011")),
    _coords_from_rows(("111", "111", "010")),
    _coords_from_rows(("010", "111", "111")),
    _coords_from_rows(("00100", "01110", "11111", "01110", "00100")),
    _coords_from_rows(("11111", "00100", "00100", "00100", "00100")),
    _coords_from_rows(("00100", "00100", "00100", "00100", "11111")),
    _coords_from_rows(("11111", "01110", "00100", "01110", "11111")),
    _coords_from_rows(("00100", "01100", "11111", "00110", "00100")),
    _coords_from_rows(("10001", "10001", "11111", "10001", "10001")),
    _coords_from_rows(("10001", "10001", "10001", "01110", "00100")),
)


def _dominant_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_divider(grid: np.ndarray) -> tuple[int, int, int]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        if rows.size == 1 and cols.size == 1:
            candidates.append((color, int(rows[0]), int(cols[0])))
    assert len(candidates) == 1, "expected one full-row/full-column divider color"
    return candidates[0]


def _extract_templates(lookup: np.ndarray, bg: int) -> dict[int, tuple[tuple[int, int], ...]]:
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in np.unique(lookup):
        color = int(color)
        if color == bg:
            continue
        comps = _components(lookup == color)
        assert len(comps) == 1, "each lookup color must form one component"
        coords = comps[0]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert (int(r1) - int(r0)) % 2 == 0 and (int(c1) - int(c0)) % 2 == 0, "template bbox must have a center"
        center = ((int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2)
        assert int(lookup[center]) == color, "template center pixel must be present"
        rel = tuple(sorted((int(r - center[0]), int(c - center[1])) for r, c in coords))
        templates[color] = rel
    assert len(templates) >= 2, "lookup quadrant must contain multiple templates"
    return templates


def _apply_template(
    out: np.ndarray,
    source: np.ndarray,
    center: tuple[int, int],
    rel_coords: tuple[tuple[int, int], ...],
    color: int,
    bounds: tuple[int, int, int, int],
    obstacle_color: int,
) -> None:
    r0, r1, c0, c1 = bounds
    sr, sc = center
    for dr, dc in rel_coords:
        rr = sr + dr
        cc = sc + dc
        if r0 <= rr < r1 and c0 <= cc < c1 and int(source[rr, cc]) != obstacle_color:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed color to look up a template in the top-left legend, then grow that template inside each other quadrant while clipping it against maroon obstacles and the grey cross.

    Input structure:
    - A background-colored grid is split into four quadrants by one full grey row and one full grey column.
    - The top-left quadrant is a lookup table containing several separated monochrome template sprites, one per color, each with an occupied center pixel.
    - Each of the other three quadrants contains exactly one seed pixel whose color matches one lookup template, plus one or more maroon rectangular obstacles.

    Rule:
    - Keep the divider cross, lookup table, background, and maroon obstacles unchanged.
    - For each non-lookup quadrant, find its single seed pixel, use the seed color to select the same-colored lookup template, and align the template's center onto the seed.
    - Paint only the aligned template cells that stay inside that quadrant and do not land on maroon obstacle cells; clipped cells are omitted.

    Key insight:
    - The seed color is a legend key: the real task is color-based template retrieval followed by local stamping with obstacle and boundary clipping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider_color, split_r, split_c = _find_divider(g)
    bg = _dominant_color(g[g != divider_color])

    lookup = g[:split_r, :split_c]
    assert lookup.size > 0, "top-left lookup quadrant must be non-empty"
    templates = _extract_templates(lookup, bg)
    template_colors = set(templates)

    other_colors = set(map(int, np.unique(g))) - {bg, divider_color} - template_colors
    assert len(other_colors) == 1, "expected one obstacle color outside the lookup palette"
    obstacle_color = int(next(iter(other_colors)))

    quadrants = (
        (0, split_r, split_c + 1, g.shape[1]),
        (split_r + 1, g.shape[0], 0, split_c),
        (split_r + 1, g.shape[0], split_c + 1, g.shape[1]),
    )

    out = g.copy()
    for r0, r1, c0, c1 in quadrants:
        sub = g[r0:r1, c0:c1]
        assert sub.size > 0, "each target quadrant must be non-empty"
        seed_mask = np.isin(sub, list(template_colors))
        seed_coords = np.argwhere(seed_mask)
        assert seed_coords.shape[0] == 1, "each target quadrant must contain exactly one seed pixel"
        sr, sc = map(int, seed_coords[0])
        seed_color = int(sub[sr, sc])
        bounds = (r0, r1, c0, c1)
        _apply_template(
            out,
            g,
            (r0 + sr, c0 + sc),
            templates[seed_color],
            seed_color,
            bounds,
            obstacle_color,
        )
    return out


def _pattern_cells(
    rel_coords: tuple[tuple[int, int], ...],
    center: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    sr, sc = center
    cells = set()
    for dr, dc in rel_coords:
        rr = sr + dr
        cc = sc + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def _candidate_centers(
    rel_coords: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    require_full_fit: bool,
) -> list[tuple[int, int]]:
    total = len(rel_coords)
    centers = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            visible = _pattern_cells(rel_coords, (r, c), shape)
            if require_full_fit and len(visible) == total:
                centers.append((r, c))
            if not require_full_fit and len(visible) < total:
                centers.append((r, c))
    return centers


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, rh, rw = rect
    return {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}


def _buffer_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_rect(
    shape: tuple[int, int],
    blocked_zone: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    require_overlap: set[tuple[int, int]] | None,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    for _ in range(250):
        rh = random.randint(1, min(4, h))
        rw = random.randint(1, min(4, w))
        if rh * rw == 1:
            continue
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        rect = (r0, c0, rh, rw)
        cells = _rect_cells(rect)
        if cells & blocked_zone:
            continue
        if cells & forbidden:
            continue
        if require_overlap is not None and not (cells & require_overlap):
            continue
        return rect
    return None


def _build_target_quadrant(
    shape: tuple[int, int],
    color: int,
    rel_coords: tuple[tuple[int, int], ...],
    bg: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    full_fit_centers = _candidate_centers(rel_coords, shape, require_full_fit=True)
    edge_centers = _candidate_centers(rel_coords, shape, require_full_fit=False)
    if mode in {"free", "blocked"}:
        if not full_fit_centers:
            return None
        center = random.choice(full_fit_centers)
    else:
        if not edge_centers:
            return None
        center = random.choice(edge_centers)

    visible = _pattern_cells(rel_coords, center, shape)
    quad_in = np.full(shape, bg, dtype=int)
    quad_out = quad_in.copy()

    obstacle_count = random.randint(1, 3)
    blocked_zone: set[tuple[int, int]] = set()
    obstacles: list[tuple[int, int, int, int]] = []

    if mode == "blocked":
        overlap_target = visible - {center}
        if not overlap_target:
            return None
        rect = _place_rect(shape, blocked_zone, {center}, overlap_target)
        if rect is None:
            return None
        cells = _rect_cells(rect)
        blocked_zone |= _buffer_cells(cells, shape)
        obstacles.append(rect)

    safe_forbidden = set(visible)
    while len(obstacles) < obstacle_count:
        rect = _place_rect(shape, blocked_zone, safe_forbidden, None)
        if rect is None:
            return None
        cells = _rect_cells(rect)
        blocked_zone |= _buffer_cells(cells, shape)
        obstacles.append(rect)

    for rect in obstacles:
        r0, c0, rh, rw = rect
        quad_in[r0 : r0 + rh, c0 : c0 + rw] = OBSTACLE_COLOR
        quad_out[r0 : r0 + rh, c0 : c0 + rw] = OBSTACLE_COLOR

    quad_in[center] = color
    quad_out[:] = quad_in
    for rr, cc in visible:
        if int(quad_out[rr, cc]) != OBSTACLE_COLOR:
            quad_out[rr, cc] = color

    if mode == "free":
        assert len(visible) == len(rel_coords)
    elif mode == "edge":
        assert len(visible) < len(rel_coords)
    else:
        assert len(visible) == len(rel_coords)
        assert any((rr, cc) != center and int(quad_in[rr, cc]) == OBSTACLE_COLOR for rr, cc in visible)
    return quad_in, quad_out


def _choose_layout() -> tuple[int, int, int, int]:
    top_h = random.randint(13, 15)
    left_w = random.randint(13, 15)
    bottom_h = random.randint(8, min(14, 29 - top_h))
    right_w = random.randint(8, min(14, 29 - left_w))
    return top_h, left_w, bottom_h, right_w


def _render_lookup(
    grid: np.ndarray,
    expected: np.ndarray,
    top_h: int,
    left_w: int,
    palette: dict[int, tuple[tuple[int, int], ...]],
) -> None:
    slots = [(3, 3), (3, left_w - 4), (top_h - 4, 3), (top_h - 4, left_w - 4)]
    chosen_slots = random.sample(slots, len(palette))
    items = list(palette.items())
    random.shuffle(items)
    for (color, rel_coords), (cr, cc) in zip(items, chosen_slots, strict=False):
        for dr, dc in rel_coords:
            rr = cr + dr
            cc2 = cc + dc
            grid[rr, cc2] = color
            expected[rr, cc2] = color


def _paste_block(dst: np.ndarray, origin: tuple[int, int], block: np.ndarray) -> None:
    r0, c0 = origin
    h, w = block.shape
    dst[r0 : r0 + h, c0 : c0 + w] = block


def _gen_attempt() -> np.ndarray | None:
    top_h, left_w, bottom_h, right_w = _choose_layout()
    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w

    bg = random.choice(FREE_COLORS)
    n_patterns = random.randint(2, 4)
    colors = random.sample([c for c in FREE_COLORS if c != bg], n_patterns)
    patterns = random.sample(PATTERN_LIBRARY, n_patterns)
    lookup_palette = {color: pattern for color, pattern in zip(colors, patterns, strict=False)}

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()
    grid[top_h, :] = DIVIDER_COLOR
    grid[:, left_w] = DIVIDER_COLOR
    expected[top_h, :] = DIVIDER_COLOR
    expected[:, left_w] = DIVIDER_COLOR

    _render_lookup(grid, expected, top_h, left_w, lookup_palette)

    quadrants = (
        ((0, left_w + 1), (top_h, right_w)),
        ((top_h + 1, 0), (bottom_h, left_w)),
        ((top_h + 1, left_w + 1), (bottom_h, right_w)),
    )
    modes = ["free", "edge", "blocked"]
    random.shuffle(modes)

    for ((r0, c0), shape), mode in zip(quadrants, modes, strict=False):
        seed_color = random.choice(colors)
        block_pair = _build_target_quadrant(shape, seed_color, lookup_palette[seed_color], bg, mode)
        if block_pair is None:
            return None
        quad_in, quad_out = block_pair
        _paste_block(grid, (r0, c0), quad_in)
        _paste_block(expected, (r0, c0), quad_out)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lookup-and-clipping ARC input")
