from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEP = 5
PATTERN_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _enclosed_non_wall(wall: np.ndarray) -> np.ndarray:
    h, w = wall.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not bool(wall[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(wall[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not bool(wall[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    return (~seen) & (~wall)


def _pattern_rows(right: np.ndarray, bg: int, sep_color: int) -> list[np.ndarray]:
    sep_rows = [r for r in range(right.shape[0]) if bool(np.all(right[r, :] == int(sep_color)))]
    patterns: list[np.ndarray] = []
    start = 0
    for end in sep_rows + [right.shape[0]]:
        band = right[start:end, :]
        content = (band != int(bg)) & (band != int(sep_color))
        if bool(content.any()):
            coords = np.argwhere(content)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            patch = band[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
            assert patch.size > 0
            patterns.append(patch)
        start = end + 1
    assert patterns, "expected at least one library pattern"
    return patterns


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != int(bg):
            candidates.append((c, int(vals[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return candidates[0]


def _tile_fill(out: np.ndarray, fill_mask: np.ndarray, pattern: np.ndarray) -> None:
    coords = np.argwhere(fill_mask)
    assert coords.size > 0
    anchor_r = int(coords[:, 0].min())
    anchor_c = int(coords[:, 1].min())
    ph, pw = pattern.shape
    for r, c in coords:
        rr = (int(r) - anchor_r) % int(ph)
        cc = (int(c) - anchor_c) % int(pw)
        out[int(r), int(c)] = int(pattern[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow object's interior with the library texture indexed by its hole count.

    Input structure:
    - The background is black, and one full-height grey separator column splits the grid into a left object half and a right pattern-library half.
    - The left half contains several closed single-color outer borders; some objects also contain one or more smaller same-color inner rings that enclose black hole regions.
    - The right half contains rectangular texture samples stacked vertically, with grey horizontal separator rows between samples.

    Rule:
    - For each left-half object, count how many hole regions are enclosed by its inner rings.
    - Select the texture sample at that top-to-bottom index in the right library, using 0-based indexing.
    - Tile that sample over the object's fillable interior, aligned to the interior region's top-left cell, while keeping the outer border, inner-ring borders, hole interiors, the grey separator column, and the library unchanged.

    Key insight:
    - The border color is irrelevant; the correct fill comes from a topological property of the object, namely how many enclosed holes it contains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(BG if np.any(g == BG) else np.bincount(g.ravel()).argmax())
    sep_col, sep_color = _find_separator(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    patterns = _pattern_rows(right, bg, sep_color)

    out = g.copy()
    object_colors = [int(c) for c in np.unique(left) if int(c) not in {bg, sep_color}]

    for color in object_colors:
        comps = _components(left == int(color))
        assert comps, "object color must have at least one component"
        ranked = sorted(
            comps,
            key=lambda comp: (
                (_bbox(comp)[2] - _bbox(comp)[0] + 1) * (_bbox(comp)[3] - _bbox(comp)[1] + 1),
                comp.shape[0],
            ),
            reverse=True,
        )
        outer = ranked[0]
        r0, c0, r1, c1 = _bbox(outer)
        sub = left[r0 : r1 + 1, c0 : c1 + 1]

        outer_wall = np.zeros_like(sub, dtype=bool)
        outer_wall[outer[:, 0] - r0, outer[:, 1] - c0] = True
        inside_outer = _enclosed_non_wall(outer_wall)
        fill_mask = inside_outer & (sub == int(bg))

        hole_count = 0
        for inner in ranked[1:]:
            ir0, ic0, ir1, ic1 = _bbox(inner)
            ring = left[ir0 : ir1 + 1, ic0 : ic1 + 1]
            inner_wall = np.zeros_like(ring, dtype=bool)
            inner_wall[inner[:, 0] - ir0, inner[:, 1] - ic0] = True
            hole_bg = _enclosed_non_wall(inner_wall) & (ring == int(bg))
            if not bool(hole_bg.any()):
                continue
            hole_count += 1
            fill_mask[ir0 - r0 : ir1 - r0 + 1, ic0 - c0 : ic1 - c0 + 1][hole_bg] = False

        assert hole_count < len(patterns), "hole count must map into the pattern library"
        if bool(fill_mask.any()):
            pattern = patterns[hole_count]
            fill_global = np.zeros_like(out, dtype=bool)
            fill_global[r0 : r1 + 1, c0 : c1 + 1] = fill_mask
            _tile_fill(out, fill_global, pattern)

    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def _rect_ring(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    inside = np.zeros((h, w), dtype=bool)
    inside[1:-1, 1:-1] = True
    return border, inside


def _diamond_ring(radius: int) -> tuple[np.ndarray, np.ndarray]:
    size = 2 * int(radius) + 1
    rr, cc = np.indices((size, size))
    dist = np.abs(rr - int(radius)) + np.abs(cc - int(radius))
    border = dist == int(radius)
    inside = dist < int(radius)
    return border, inside


def _overlap_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _make_object_patch(
    rng: np.random.Generator,
    color: int,
    hole_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    hole_shapes: list[tuple[np.ndarray, np.ndarray]] = [_rect_ring(3, 3), _rect_ring(3, 5), _rect_ring(5, 3), _diamond_ring(1)]
    for _ in range(120):
        h = int(rng.integers(7 + hole_count, 12 + hole_count))
        w = int(rng.integers(7 + hole_count, 12 + hole_count))
        h = min(h, 13)
        w = min(w, 13)
        outer_border, outer_inside = _rect_ring(h, w)

        ring_mask = outer_border.copy()
        hole_bg = np.zeros((h, w), dtype=bool)
        padded_holes: list[tuple[int, int, int, int]] = []

        ok = True
        for _hole in range(hole_count):
            placed = False
            local_shapes = hole_shapes[:]
            rng.shuffle(local_shapes)
            if min(h, w) >= 10 and hole_count <= 1:
                local_shapes.append(_rect_ring(5, 5))
            for _ in range(80):
                border, inside = local_shapes[int(rng.integers(0, len(local_shapes)))]
                ph, pw = border.shape
                if h - ph - 4 < 0 or w - pw - 4 < 0:
                    continue
                r0 = int(rng.integers(2, h - ph - 1))
                c0 = int(rng.integers(2, w - pw - 1))
                padded = (r0 - 1, c0 - 1, r0 + ph, c0 + pw)
                if any(_overlap_bbox(padded, prev) for prev in padded_holes):
                    continue
                ring_mask[r0 : r0 + ph, c0 : c0 + pw] |= border
                hole_bg[r0 : r0 + ph, c0 : c0 + pw] |= inside
                padded_holes.append(padded)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        fill_mask = outer_inside & (~ring_mask) & (~hole_bg)
        if not bool(fill_mask.any()):
            continue
        if len(_components(fill_mask)) != 1:
            continue

        patch = np.full((h, w), BG, dtype=int)
        patch[ring_mask] = int(color)
        return patch, fill_mask
    return None


def _pattern_from_family(
    rng: np.random.Generator,
    family: str,
    colors: list[int],
    h: int,
    w: int,
) -> np.ndarray:
    colors = list(colors)
    arr = np.zeros((h, w), dtype=int)
    if family == "checker":
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r + c) % 2]
    elif family == "hstripe":
        for r in range(h):
            arr[r, :] = colors[r % len(colors)]
    elif family == "vstripe":
        for c in range(w):
            arr[:, c] = colors[c % len(colors)]
    elif family == "diag":
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r - c) % len(colors)]
    elif family == "cross":
        base, alt1, alt2 = colors[0], colors[1], colors[2]
        arr[:, :] = base
        for r in range(h):
            for c in range(w):
                if (r % 2 == 0) and (c % 2 == 0):
                    arr[r, c] = alt1
                elif (r + c) % 2 == 1:
                    arr[r, c] = alt2
    else:
        raise AssertionError("unknown pattern family")
    return arr


def _make_patterns(
    rng: np.random.Generator,
    right_w: int,
    color_pool: list[int],
) -> list[np.ndarray] | None:
    families = ["checker", "hstripe", "vstripe", "diag", "cross"]
    rng.shuffle(families)
    chosen = families[:PATTERN_COUNT]
    patterns: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for family in chosen:
        pw = int(rng.integers(3, right_w + 1))
        ph = int(rng.integers(2, 5))
        if family == "cross":
            cols = list(rng.choice(color_pool, size=3, replace=False))
        elif rng.random() < 0.55 and len(color_pool) >= 3:
            cols = list(rng.choice(color_pool, size=3, replace=False))
        else:
            cols = list(rng.choice(color_pool, size=2, replace=False))
        pattern = _pattern_from_family(rng, family, cols, ph, pw)
        key = (pattern.shape, pattern.tobytes())
        if key in seen:
            return None
        seen.add(key)
        patterns.append(pattern)
    return patterns


def _tile_mask_relative(fill_mask: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    out = np.zeros(fill_mask.shape, dtype=int)
    coords = np.argwhere(fill_mask)
    anchor_r = int(coords[:, 0].min())
    anchor_c = int(coords[:, 1].min())
    ph, pw = pattern.shape
    for r, c in coords:
        out[int(r), int(c)] = int(pattern[(int(r) - anchor_r) % ph, (int(c) - anchor_c) % pw])
    return out


def _paste_pattern_library(
    grid: np.ndarray,
    sep_col: int,
    patterns: list[np.ndarray],
    rng: np.random.Generator,
) -> None:
    right = grid[:, sep_col + 1 :]
    right[:, :] = BG
    row = int(rng.integers(0, 3))
    for idx, pattern in enumerate(patterns):
        ph, pw = pattern.shape
        col = int(rng.integers(0, right.shape[1] - pw + 1))
        right[row : row + ph, col : col + pw] = pattern
        row += ph
        if idx + 1 < len(patterns):
            right[row, :] = SEP
            row += 1


def _place_objects(
    rng: np.random.Generator,
    left_h: int,
    left_w: int,
    objects: list[dict],
) -> list[tuple[int, int, dict]] | None:
    occupied = np.zeros((left_h, left_w), dtype=bool)
    placed: list[tuple[int, int, dict]] = []

    for obj in sorted(objects, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        patch = obj["patch"]
        ph, pw = patch.shape
        if ph + 2 > left_h or pw + 2 > left_w:
            return None
        positions = []
        for r0 in range(1, left_h - ph):
            for c0 in range(1, left_w - pw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(left_h, r0 + ph + 1)
                cc1 = min(left_w, c0 + pw + 1)
                if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(0, len(positions)))]
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(left_h, r0 + ph + 1)
        cc1 = min(left_w, c0 + pw + 1)
        occupied[rr0:rr1, cc0:cc1] = True
        placed.append((r0, c0, obj))
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = int(rng.integers(2, 5))
    hole_counts = list(rng.choice(np.arange(PATTERN_COUNT), size=n_objects, replace=False))
    rng.shuffle(hole_counts)

    usable_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    border_colors = list(rng.choice(usable_colors, size=n_objects, replace=False))
    remaining = [c for c in usable_colors if c not in border_colors]
    pattern_palette_size = int(rng.integers(3, min(5, len(remaining)) + 1))
    pattern_palette = list(rng.choice(remaining, size=pattern_palette_size, replace=False))

    objects: list[dict] = []
    for color, holes in zip(border_colors, hole_counts, strict=False):
        built = _make_object_patch(rng, int(color), int(holes))
        if built is None:
            return None
        patch, fill_mask = built
        objects.append({"patch": patch, "fill_mask": fill_mask, "holes": int(holes), "color": int(color)})

    max_obj_h = max(obj["patch"].shape[0] for obj in objects)
    max_obj_w = max(obj["patch"].shape[1] for obj in objects)

    right_w = int(rng.integers(5, 10))
    patterns = _make_patterns(rng, right_w, pattern_palette)
    if patterns is None:
        return None

    lib_h = int(rng.integers(0, 3)) + sum(p.shape[0] for p in patterns) + (PATTERN_COUNT - 1)
    min_h = max(18, lib_h + 1, max_obj_h + 2)
    if min_h > 30:
        return None
    grid_h = int(rng.integers(min_h, 31))

    min_left_w = max(12, max_obj_w + 2)
    max_left_w = 30 - 1 - right_w
    if min_left_w > max_left_w:
        return None
    left_w = int(rng.integers(min_left_w, max_left_w + 1))
    grid_w = left_w + 1 + right_w

    placements = _place_objects(rng, grid_h, left_w, objects)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, left_w] = SEP
    expected = grid.copy()

    _paste_pattern_library(grid, left_w, patterns, rng)
    expected[:, :] = grid

    for r0, c0, obj in placements:
        patch = obj["patch"]
        ph, pw = patch.shape
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        expected[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]

        filled = _tile_mask_relative(obj["fill_mask"], patterns[obj["holes"]])
        fill_mask = obj["fill_mask"]
        expected[r0 : r0 + ph, c0 : c0 + pw][fill_mask] = filled[fill_mask]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, expected):
        return grid
    return None


def _fast_draw_ring(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = int(color)
    grid[r0 + h - 1, c0 : c0 + w] = int(color)
    grid[r0 : r0 + h, c0] = int(color)
    grid[r0 : r0 + h, c0 + w - 1] = int(color)


def _fast_generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid_h = int(rng.integers(20, 31))
    left_w = int(rng.integers(17, 21))
    right_w = int(rng.integers(5, 9))
    grid_w = left_w + 1 + right_w
    if grid_w > 30:
        left_w = 30 - 1 - right_w
        grid_w = left_w + 1 + right_w

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, left_w] = SEP

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(palette)
    pattern_colors = palette[:4]
    row = 1
    for idx, color in enumerate(pattern_colors):
        ph = int(rng.integers(2, 4))
        pw = int(rng.integers(3, right_w + 1))
        patch = np.full((ph, pw), int(color), dtype=int)
        if idx % 2 == 0 and pw > 3:
            patch[:, 1::2] = int(palette[(idx + 4) % len(palette)])
        else:
            patch[1::2, :] = int(palette[(idx + 4) % len(palette)])
        c0 = int(rng.integers(0, right_w - pw + 1))
        grid[row : row + ph, left_w + 1 + c0 : left_w + 1 + c0 + pw] = patch
        row += ph
        if idx + 1 < 4:
            grid[row, left_w + 1 :] = SEP
            row += 1

    object_colors = [c for c in palette if c not in pattern_colors][:3]
    hole_options = [0, 1, 2]
    rng.shuffle(hole_options)
    slots = [(1, 1), (1, 9), (11, 3)]
    for (r0, c0), color, holes in zip(slots, object_colors, hole_options, strict=False):
        h = int(rng.integers(7, 10))
        w = int(rng.integers(7, 10))
        if r0 + h >= grid_h or c0 + w >= left_w:
            continue
        _fast_draw_ring(grid, r0, c0, h, w, int(color))
        if holes >= 1 and h >= 7 and w >= 7:
            _fast_draw_ring(grid, r0 + 2, c0 + 2, 3, 3, int(color))
        if holes >= 2 and h >= 8 and w >= 9:
            _fast_draw_ring(grid, r0 + 2, c0 + w - 5, 3, 3, int(color))

    return grid


def generate_input() -> np.ndarray:
    for _ in range(20):
        grid = _fast_generate_input()
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    for _ in range(40):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
