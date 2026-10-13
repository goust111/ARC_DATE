from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 9

PANEL_H_RANGE = (8, 12)
PANEL_W_RANGE = (8, 12)
PALETTE_LENGTHS = (2, 3, 4)
PALETTE_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SMALL_SHARED_PATTERNS = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _neighbors4(r: int, c: int):
    for dr, dc in _N4:
        yield r + dr, c + dc


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _region_center(coords: np.ndarray) -> tuple[int, int]:
    return _round_half_up(coords[:, 0].mean()), _round_half_up(coords[:, 1].mean())


def _region_depths(coords: np.ndarray, center: tuple[int, int]) -> dict[tuple[int, int], int]:
    cr, cc = center
    return {
        (int(r), int(c)): max(abs(int(r) - cr), abs(int(c) - cc))
        for r, c in coords
    }


def _paint_component(out: np.ndarray, coords: np.ndarray, palette: list[int], offset: int) -> None:
    depths = _region_depths(coords, _region_center(coords))
    plen = len(palette)
    for (r, c), d in depths.items():
        out[r, c] = palette[(d + offset) % plen]


def _solve_from_panels(left: np.ndarray, right: np.ndarray, palette: list[int]) -> np.ndarray:
    assert left.shape == right.shape
    assert len(palette) >= 2

    out = np.zeros_like(left)
    same_fg = (left != BG) & (right == left)
    left_only = (left != BG) & (right == BG)
    right_only = (left == BG) & (right != BG)
    invalid = (left != BG) & (right != BG) & (left != right)
    assert not np.any(invalid), "panels may overlap only with identical colors"

    out[same_fg] = left[same_fg]
    for coords in _components(left_only):
        _paint_component(out, coords, palette, offset=0)
    for coords in _components(right_only):
        _paint_component(out, coords, palette, offset=1)
    return out


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 7, "grid too small"

    palette_row = g[-1]
    palette = [int(v) for v in palette_row if int(v) != BG]
    assert 2 <= len(palette) <= 4, "palette row must contain 2-4 colors"
    assert len(set(palette)) == len(palette), "palette colors must be unique"
    assert SEP not in palette, "palette must not include the separator color"

    top = g[:-1]
    sep_candidates = []
    for col in range(w):
        if np.all(top[:, col] == SEP) and g[-1, col] == BG:
            if col > 0 and col < w - 1 and col == w - col - 1:
                sep_candidates.append(col)
    assert len(sep_candidates) == 1, "expected one centered separator column above the palette row"
    sep = sep_candidates[0]

    left = top[:, :sep]
    right = top[:, sep + 1 :]
    assert left.shape == right.shape, "separator must split equal panels"
    return left, right, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two aligned panels, keep shared pixels, and recolor left-only/right-only difference regions by cyclic centroid-depth layers.

    Input structure:
    - The grid contains two equal-size panels separated by a vertical maroon divider, with a palette row on the bottom.
    - Above the palette row, both panels use black background plus non-black foreground pixels; overlapping foreground positions always match in color.
    - The bottom row contains 2-4 distinct palette colors in left-to-right order, with all remaining cells black.

    Rule:
    - Remove the bottom palette row and the divider, then compare the left and right panels cell by cell.
    - Shared foreground cells keep their original color; black/black cells stay black.
    - For each 4-connected left-only region, compute the rounded centroid of its coordinates, set each cell's layer to the Chebyshev distance from that centroid, and recolor by palette[layer mod palette_length].
    - For each 4-connected right-only region, use the same layer rule but shift the palette by one step: palette[(layer + 1) mod palette_length].

    Key insight:
    - Pixel comparison only creates the masks; the real transformation is the cyclic layer fill driven by each difference region's own centroid, with left-only and right-only regions distinguished solely by a one-step palette offset.
    """
    left, right, palette = _parse_grid(np.asarray(grid, dtype=int))
    return _solve_from_panels(left, right, palette)


def _grow_connected_mask(
    shape: tuple[int, int],
    blocked: np.ndarray,
    area_range: tuple[int, int],
    palette_len: int,
    require_cycle: bool,
) -> np.ndarray | None:
    h, w = shape
    min_area, max_area = area_range
    free_cells = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    if len(free_cells) < min_area:
        return None

    for _ in range(180):
        area = random.randint(min_area, min(max_area, len(free_cells)))
        seed_choices = [(r, c) for r, c in free_cells if 0 < r < h - 1 and 0 < c < w - 1]
        seed = random.choice(seed_choices or free_cells)
        cells = {seed}
        frontier = set()
        for nr, nc in _neighbors4(*seed):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]:
                frontier.add((nr, nc))

        while frontier and len(cells) < area:
            options = list(frontier)
            random.shuffle(options)
            pick = None
            for cand in options:
                rr, cc = cand
                degree = sum((nr, nc) in cells for nr, nc in _neighbors4(rr, cc))
                if degree == 1 and random.random() < 0.7:
                    pick = cand
                    break
            if pick is None:
                pick = random.choice(options)
            frontier.remove(pick)
            cells.add(pick)
            for nr, nc in _neighbors4(*pick):
                if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) < min_area:
            continue

        mask = np.zeros(shape, dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 3 or bw < 3:
            continue
        if int(mask.sum()) == bh * bw:
            continue

        center = _region_center(coords)
        depths = _region_depths(coords, center)
        depth_values = set(depths.values())
        if len(depth_values) < 2:
            continue
        if require_cycle and max(depth_values) < palette_len:
            continue

        bbox_center = (_round_half_up((r0 + r1) / 2), _round_half_up((c0 + c1) / 2))
        bbox_depths = _region_depths(coords, bbox_center)
        if all(depths[pos] == bbox_depths[pos] for pos in depths):
            continue

        if not any(int(r) != center[0] and int(c) != center[1] for r, c in coords):
            continue
        return mask
    return None


def _place_shared_marks(
    shape: tuple[int, int],
    blocked: np.ndarray,
    shared_colors: list[int],
) -> np.ndarray | None:
    h, w = shape
    out = np.zeros(shape, dtype=int)
    occupied = blocked.copy()
    motif_count = random.randint(2, 4)
    used_colors: list[int] = []

    for i in range(motif_count):
        pattern = random.choice(_SMALL_SHARED_PATTERNS)
        ph = max(r for r, _ in pattern) + 1
        pw = max(c for _, c in pattern) + 1
        anchors = [
            (r0, c0)
            for r0 in range(h - ph + 1)
            for c0 in range(w - pw + 1)
        ]
        random.shuffle(anchors)
        placed = False
        for r0, c0 in anchors:
            cells = [(r0 + dr, c0 + dc) for dr, dc in pattern]
            if any(occupied[r, c] for r, c in cells):
                continue
            color = shared_colors[i % len(shared_colors)]
            for r, c in cells:
                out[r, c] = color
            pad = np.zeros(shape, dtype=bool)
            for r, c in cells:
                pad[r, c] = True
            occupied |= _dilate4(pad)
            used_colors.append(color)
            placed = True
            break
        if not placed:
            return None

    if np.count_nonzero(out) < 3:
        return None
    if len(set(int(v) for v in out[out != 0])) < 2:
        return None
    return out


def _fill_mask_with_colors(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    coords = np.argwhere(mask)
    for r, c in coords:
        choice = colors[(int(r) + 2 * int(c) + random.randrange(len(colors))) % len(colors)]
        out[r, c] = choice
    if len(colors) > 1 and len(coords) >= 2:
        present = set(int(v) for v in out[mask])
        if len(present) == 1:
            rr, cc = map(int, coords[-1])
            alternatives = [c for c in colors if c != out[rr, cc]]
            out[rr, cc] = random.choice(alternatives)
    return out


def _touches(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.any(_dilate4(mask_a) & mask_b) or np.any(_dilate4(mask_b) & mask_a))


def _semantic_checks(
    left: np.ndarray,
    right: np.ndarray,
    palette: list[int],
    require_cycle: bool,
) -> np.ndarray | None:
    shared = (left != BG) & (right == left)
    left_only = (left != BG) & (right == BG)
    right_only = (left == BG) & (right != BG)
    if not np.any(shared) or not np.any(left_only) or not np.any(right_only):
        return None
    if len(_components(left_only)) != 1 or len(_components(right_only)) != 1:
        return None
    if _touches(left_only, right_only):
        return None
    shared_colors = set(int(v) for v in left[shared])
    if len(shared_colors) < 2:
        return None

    panel_colors = set(int(v) for v in np.concatenate([left.flatten(), right.flatten()]) if int(v) != BG)
    if panel_colors & set(palette):
        return None
    if SEP in panel_colors or SEP in palette:
        return None

    l_coords = _components(left_only)[0]
    r_coords = _components(right_only)[0]
    l_depths = set(_region_depths(l_coords, _region_center(l_coords)).values())
    r_depths = set(_region_depths(r_coords, _region_center(r_coords)).values())
    if len(l_depths) < 2 or len(r_depths) < 2:
        return None
    if not (l_depths & r_depths):
        return None
    if require_cycle and max(max(l_depths), max(r_depths)) < len(palette):
        return None

    out = _solve_from_panels(left, right, palette)
    if np.array_equal(out, left) or np.array_equal(out, right):
        return None
    if len(set(int(v) for v in out[left_only])) < min(2, len(palette)):
        return None
    if len(set(int(v) for v in out[right_only])) < min(2, len(palette)):
        return None
    return out


def _assemble_input(left: np.ndarray, right: np.ndarray, palette: list[int]) -> np.ndarray:
    h, w = left.shape
    grid = np.zeros((h + 1, 2 * w + 1), dtype=int)
    grid[:h, :w] = left
    grid[:h, w] = SEP
    grid[:h, w + 1 :] = right

    side = random.choice(["left", "right"])
    if side == "left":
        start = random.randint(0, w - len(palette))
    else:
        start = random.randint(w + 1, 2 * w - len(palette))
    for i, color in enumerate(palette):
        grid[h, start + i] = color
    return grid


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*PANEL_H_RANGE)
    w = random.randint(*PANEL_W_RANGE)
    palette = random.sample(list(PALETTE_POOL), random.choice(PALETTE_LENGTHS))

    panel_color_pool = [c for c in range(1, 10) if c not in palette and c != SEP]
    if len(panel_color_pool) < 4:
        return None
    shared_colors = random.sample(panel_color_pool, random.choice([2, 3]))
    left_colors = random.sample(panel_color_pool, random.choice([1, 2]))
    right_colors = random.sample(panel_color_pool, random.choice([1, 2]))

    require_cycle = random.random() < 0.6
    panel_shape = (h, w)
    empty = np.zeros(panel_shape, dtype=bool)

    min_area = 6
    max_area = min(16, h * w // 3)
    left_mask = _grow_connected_mask(panel_shape, empty, (min_area, max_area), len(palette), require_cycle and random.random() < 0.7)
    if left_mask is None:
        return None

    right_blocked = _dilate4(left_mask)
    right_mask = _grow_connected_mask(panel_shape, right_blocked, (min_area, max_area), len(palette), require_cycle and random.random() < 0.7)
    if right_mask is None:
        return None

    shared_blocked = _dilate4(left_mask | right_mask)
    shared_grid = _place_shared_marks(panel_shape, shared_blocked, shared_colors)
    if shared_grid is None:
        return None

    left = shared_grid.copy()
    right = shared_grid.copy()
    left[left_mask] = _fill_mask_with_colors(left_mask, left_colors)[left_mask]
    right[right_mask] = _fill_mask_with_colors(right_mask, right_colors)[right_mask]

    expected = _semantic_checks(left, right, palette, require_cycle)
    if expected is None:
        return None

    grid = _assemble_input(left, right, palette)
    parsed_left, parsed_right, parsed_palette = _parse_grid(grid)
    if not np.array_equal(parsed_left, left):
        return None
    if not np.array_equal(parsed_right, right):
        return None
    if parsed_palette != palette:
        return None
    if not np.array_equal(_solve_from_panels(left, right, palette), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
