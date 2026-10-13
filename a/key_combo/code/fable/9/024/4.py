from __future__ import annotations

import numpy as np


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors(shape: tuple[int, int], r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    height, width = shape
    out = []
    for dr, dc in deltas:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray, connectivity: int = 8) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        stack = [(rr, cc)]
        seen[rr, cc] = True
        while stack:
            cr, cc2 = stack.pop()
            for nr, nc in _neighbors(mask.shape, cr, cc2, connectivity):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _separator_and_background(grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    candidates: list[tuple[int, int]] = []
    for row in range(1, height - 1):
        if not np.all(grid[row, :] == grid[row, 0]):
            continue
        sep_color = int(grid[row, 0])
        lower = grid[row + 1 :, :]
        bg = int(lower[0, 0])
        if sep_color == bg:
            continue
        if not np.all(lower == bg):
            continue
        upper = grid[:row, :]
        if np.any(upper == sep_color):
            continue
        upper_colors = _colors_present(upper, exclude={bg})
        if len(upper_colors) < 2:
            continue
        candidates.append((row, bg))
    assert len(candidates) == 1, "expected one full-width separator above a uniform lower region"
    return candidates[0]


def _color_is_untouching(region: np.ndarray, color: int, bg: int, connectivity: int = 4) -> bool:
    coords = np.argwhere(region == int(color))
    assert coords.size > 0, "candidate color must be present"
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in _neighbors(region.shape, r, c, connectivity):
            neighbor = int(region[nr, nc])
            if neighbor not in {bg, int(color)}:
                return False
    return True


def _unique_untouching_color(region: np.ndarray, bg: int) -> int:
    colors = _colors_present(region, exclude={bg})
    winners = [color for color in colors if _color_is_untouching(region, color, bg, connectivity=4)]
    assert len(winners) == 1, "expected exactly one color that never 4-touches any other color"
    return int(winners[0])


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_row, bg = _separator_and_background(g)
    upper = g[:sep_row, :]
    lower = g[sep_row + 1 :, :]
    isolated = _unique_untouching_color(upper, bg)

    coords = np.argwhere(upper == isolated)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    box_h = int(r1 - r0 + 1)
    box_w = int(c1 - c0 + 1)
    assert box_h <= lower.shape[0] and box_w <= lower.shape[1], "isolated color bbox must fit in lower region"
    assert (lower.shape[0] - box_h) % 2 == 0, "bbox vertical center must align to lower-region center"
    assert (lower.shape[1] - box_w) % 2 == 0, "bbox horizontal center must align to lower-region center"

    target_r0 = sep_row + 1 + (lower.shape[0] - box_h) // 2
    target_c0 = (lower.shape[1] - box_w) // 2

    out = g.copy()
    out[sep_row + 1 :, :] = bg
    out[:sep_row, :][upper == isolated] = bg
    for r_raw, c_raw in coords:
        rr = int(target_r0 + int(r_raw) - r0)
        cc = int(target_c0 + int(c_raw) - c0)
        out[rr, cc] = isolated
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Exile the unique upper color that never 4-touches any other color by moving its whole pixel pattern to the centered lower panel.

    Input structure
    - One full-width separator row splits the grid into an upper pattern region and a lower result region.
    - The lower region is entirely background, while the upper region contains background plus several foreground colors.
    - Among the upper foreground colors, exactly one color has all of its pixels 4-neighbor-adjacent only to background or to the same color.

    Rule
    - Find that unique untouching color using only 4-neighbor contact within the upper region.
    - Remove all pixels of that color from the upper region.
    - Translate the full set of removed pixels, preserving relative offsets, into the lower region so the color's bounding-box center aligns with the lower region's center.

    Key insight
    - The target color is identified by adjacency topology rather than by size, count, or shape: it is the only upper color that never directly 4-touches a different foreground color.
    """
    return _expected_output(np.asarray(grid, dtype=int))


_PALETTE = tuple(range(10))
_GRID_RETRIES = 500
_PATCH_RETRIES = 400
_PLACEMENT_RETRIES = 150
_MIN_UPPER = 7
_MAX_UPPER = 13
_MIN_LOWER = 7
_MAX_LOWER = 13
_MIN_WIDTH = 12
_MAX_WIDTH = 24


def _random_patch(height: int, width: int, rng: np.random.Generator, multi_component: bool) -> np.ndarray | None:
    max_cells = min(height * width - 1, 12)
    min_cells = max(4, min(max_cells, (height + width) // 2))
    if min_cells > max_cells:
        return None

    desired_components = {1} if not multi_component else {2, 3}
    for _ in range(_PATCH_RETRIES):
        cell_count = int(rng.integers(min_cells, max_cells + 1))
        coords = {
            (0, int(rng.integers(width))),
            (height - 1, int(rng.integers(width))),
            (int(rng.integers(height)), 0),
            (int(rng.integers(height)), width - 1),
        }
        while len(coords) < cell_count:
            coords.add((int(rng.integers(height)), int(rng.integers(width))))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in coords:
            mask[r, c] = True

        bbox = _bbox_from_coords(np.argwhere(mask))
        if bbox != (0, 0, height - 1, width - 1):
            continue
        if _component_count(mask, connectivity=8) not in desired_components:
            continue
        return mask
    return None


def _guard_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    guard = np.zeros(shape, dtype=bool)
    for r, c in coords:
        guard[r, c] = True
        for nr, nc in _neighbors(shape, r, c, 4):
            guard[nr, nc] = True
    return guard


def _find_chain_candidates(free: np.ndarray, length: int) -> list[list[tuple[int, int]]]:
    height, width = free.shape
    candidates: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width - length + 1):
            if np.all(free[r, c : c + length]):
                candidates.append([(r, c + dc) for dc in range(length)])
    for c in range(width):
        for r in range(height - length + 1):
            if np.all(free[r : r + length, c]):
                candidates.append([(r + dr, c) for dr in range(length)])
    return candidates


def _diagonal_candidates(iso_mask: np.ndarray, guard: np.ndarray) -> list[tuple[int, int]]:
    candidates = set()
    for r_raw, c_raw in np.argwhere(iso_mask):
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < iso_mask.shape[0] and 0 <= cc < iso_mask.shape[1]:
                if not iso_mask[rr, cc] and not guard[rr, cc]:
                    candidates.add((rr, cc))
    return sorted(candidates)


def _choose_targets(
    iso_count: int,
    n_other: int,
    available: int,
    rng: np.random.Generator,
) -> list[int] | None:
    if available < n_other or iso_count <= 1:
        return None

    for _ in range(80):
        targets = [1] * n_other
        large_idx, small_idx = rng.choice(n_other, size=2, replace=False)
        large_max = min(iso_count + 6, available - (n_other - 1))
        if large_max <= iso_count:
            continue
        targets[large_idx] = int(rng.integers(iso_count + 1, large_max + 1))
        targets[small_idx] = int(rng.integers(1, iso_count))

        assigned = targets[large_idx] + targets[small_idx]
        remaining_slots = available - assigned
        remaining_colors = [idx for idx in range(n_other) if idx not in {large_idx, small_idx}]
        ok = True
        for pos, idx in enumerate(remaining_colors):
            min_left = len(remaining_colors) - pos - 1
            hi = min(iso_count + 3, remaining_slots - min_left)
            if hi < 1:
                ok = False
                break
            targets[idx] = int(rng.integers(1, hi + 1))
            remaining_slots -= targets[idx]
        if ok and sum(targets) <= available:
            if max(targets) > iso_count and min(targets) < iso_count:
                return targets
    return None


def _expand_color_cells(
    seeds: set[tuple[int, int]],
    target: int,
    free_cells: set[tuple[int, int]],
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    while len(seeds) < target:
        neighbor_options = set()
        for r, c in seeds:
            for nr, nc in _neighbors(shape, r, c, 8):
                if (nr, nc) in free_cells:
                    neighbor_options.add((nr, nc))
        use_neighbor = bool(neighbor_options) and (rng.random() < 0.75 or len(free_cells) == len(neighbor_options))
        if use_neighbor:
            chosen = list(neighbor_options)[int(rng.integers(len(neighbor_options)))]
        elif free_cells:
            chosen = list(free_cells)[int(rng.integers(len(free_cells)))]
        else:
            return False
        free_cells.remove(chosen)
        seeds.add(chosen)
    return True


def _build_clutter(
    upper: np.ndarray,
    bg: int,
    other_colors: list[int],
    iso_mask: np.ndarray,
    require_diagonal: bool,
    rng: np.random.Generator,
) -> bool:
    guard = _guard_mask(upper.shape, [tuple(map(int, rc)) for rc in np.argwhere(iso_mask)])
    free = (upper == bg) & ~guard
    chain_candidates = _find_chain_candidates(free, len(other_colors))
    if not chain_candidates:
        return False

    iso_count = int(iso_mask.sum())
    diagonal_cell: tuple[int, int] | None = None
    if require_diagonal:
        diagonal_options = _diagonal_candidates(iso_mask, guard)
        if not diagonal_options:
            return False
        diagonal_cell = diagonal_options[int(rng.integers(len(diagonal_options)))]

    chain = chain_candidates[int(rng.integers(len(chain_candidates)))]
    seeds = {color: {chain[idx]} for idx, color in enumerate(other_colors)}
    reserved = set(chain)
    large_color = other_colors[0]
    if diagonal_cell is not None:
        if diagonal_cell in reserved:
            return False
        seeds[large_color].add(diagonal_cell)
        reserved.add(diagonal_cell)

    available_count = int(np.count_nonzero(free))
    targets = _choose_targets(iso_count, len(other_colors), available_count, rng)
    if targets is None:
        return False
    target_by_color = {color: targets[idx] for idx, color in enumerate(other_colors)}
    for color in other_colors:
        target_by_color[color] = max(target_by_color[color], len(seeds[color]))
    if sum(target_by_color.values()) > available_count:
        return False

    free_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(free)
        if (int(r), int(c)) not in reserved
    }
    for color in sorted(other_colors, key=lambda c: target_by_color[c], reverse=True):
        if not _expand_color_cells(seeds[color], target_by_color[color], free_cells, upper.shape, rng):
            return False

    for color, coords in seeds.items():
        for r, c in coords:
            upper[r, c] = int(color)
    return True


def _explicit_validate(grid: np.ndarray, expect_diagonal: bool) -> bool:
    try:
        sep_row, bg = _separator_and_background(grid)
        upper = grid[:sep_row, :]
        lower = grid[sep_row + 1 :, :]
        if not np.all(lower == bg):
            return False

        fg_colors = _colors_present(upper, exclude={bg})
        if len(fg_colors) < 3:
            return False

        isolated = [c for c in fg_colors if _color_is_untouching(upper, c, bg, connectivity=4)]
        if len(isolated) != 1:
            return False
        iso_color = isolated[0]

        for color in fg_colors:
            if color == iso_color:
                continue
            if _color_is_untouching(upper, color, bg, connectivity=4):
                return False

        counts = {color: int(np.sum(upper == color)) for color in fg_colors}
        if not any(counts[color] < counts[iso_color] for color in fg_colors if color != iso_color):
            return False
        if not any(counts[color] > counts[iso_color] for color in fg_colors if color != iso_color):
            return False

        if expect_diagonal and _color_is_untouching(upper, iso_color, bg, connectivity=8):
            return False

        expected = _expected_output(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        moved_mask = actual[sep_row + 1 :, :] == iso_color
        moved_coords = np.argwhere(moved_mask)
        if moved_coords.size == 0:
            return False
        mr0, mc0, mr1, mc1 = _bbox_from_coords(moved_coords)
        if (mr1 - mr0 + 1) % 2 != lower.shape[0] % 2:
            return False
        if (mc1 - mc0 + 1) % 2 != lower.shape[1] % 2:
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    upper_h = int(rng.integers(_MIN_UPPER, _MAX_UPPER + 1))
    lower_h = int(rng.integers(_MIN_LOWER, _MAX_LOWER + 1))
    width = int(rng.integers(_MIN_WIDTH, _MAX_WIDTH + 1))
    if upper_h + 1 + lower_h > 30:
        return None

    bg, sep, isolated = rng.choice(_PALETTE, size=3, replace=False)
    palette = [color for color in _PALETTE if color not in {int(bg), int(sep), int(isolated)}]
    other_count = int(rng.integers(2, 5))
    other_colors = list(map(int, rng.choice(palette, size=other_count, replace=False)))

    height_choices = [
        h
        for h in range(2, min(upper_h, lower_h, 8) + 1)
        if (lower_h - h) % 2 == 0
    ]
    width_choices = [
        w
        for w in range(3, min(width - 3, 10) + 1)
        if (width - w) % 2 == 0
    ]
    if not height_choices or not width_choices:
        return None

    multi_component = bool(rng.integers(0, 2))
    require_diagonal = bool(rng.integers(0, 2))
    patch_h = int(height_choices[int(rng.integers(len(height_choices)))])
    patch_w = int(width_choices[int(rng.integers(len(width_choices)))])
    patch = _random_patch(patch_h, patch_w, rng, multi_component=multi_component)
    if patch is None:
        return None

    full_h = upper_h + 1 + lower_h
    grid = np.full((full_h, width), int(bg), dtype=int)
    grid[upper_h, :] = int(sep)
    upper = grid[:upper_h, :]

    iso_coords_local = np.argwhere(patch)
    for _ in range(_PLACEMENT_RETRIES):
        top = int(rng.integers(0, upper_h - patch_h + 1))
        left = int(rng.integers(0, width - patch_w + 1))
        iso_coords = [(top + int(r), left + int(c)) for r, c in iso_coords_local]
        guard = _guard_mask(upper.shape, iso_coords)
        if np.count_nonzero(~guard) < 12:
            continue

        upper.fill(int(bg))
        for r, c in iso_coords:
            upper[r, c] = int(isolated)
        iso_mask = upper == int(isolated)

        shuffled_other = list(other_colors)
        rng.shuffle(shuffled_other)
        if not _build_clutter(upper, int(bg), shuffled_other, iso_mask, require_diagonal, rng):
            continue

        if _explicit_validate(grid, expect_diagonal=require_diagonal):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GRID_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid isolated-color-exile input")
