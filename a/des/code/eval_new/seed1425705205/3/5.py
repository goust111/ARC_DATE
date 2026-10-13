from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = np.concatenate((grid[0], grid[h - 1], grid[1 : h - 1, 0], grid[1 : h - 1, w - 1]))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _axis_period(arr: np.ndarray, axis: int) -> int:
    n = arr.shape[axis]
    for period in range(1, n + 1):
        lhs = [slice(None)] * arr.ndim
        rhs = [slice(None)] * arr.ndim
        lhs[axis] = slice(period, None)
        rhs[axis] = slice(None, n - period)
        if np.array_equal(arr[tuple(lhs)], arr[tuple(rhs)]):
            return period
    return n


def _scan_colors(patches: list[np.ndarray]) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for patch in patches:
        for value in patch.ravel():
            color = int(value)
            if color not in seen:
                seen.add(color)
                order.append(color)
    return order


def _extract_patches(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    pieces: list[tuple[tuple[int, int], np.ndarray]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        assert patch.size == len(coords), "each foreground object must be a solid rectangle"
        assert np.all(patch != bg), "rectangle interiors must not contain background"
        pieces.append(((r0, c0), patch.copy()))

    pieces.sort(key=lambda item: item[0])
    assert 2 <= len(pieces) <= 4, "expected 2-4 damaged pattern rectangles"
    shapes = {patch.shape for _pos, patch in pieces}
    assert len(shapes) == 1, "all rectangles must have the same size"
    return [patch for _pos, patch in pieces]


def _reconstruct_candidate(
    patches: list[np.ndarray], candidate: int
) -> tuple[np.ndarray, int, int, int] | None:
    n = len(patches)
    h, w = patches[0].shape
    out = np.empty((h, w), dtype=int)
    mixed_positions = 0
    partners: set[int] = set()

    for r in range(h):
        for c in range(w):
            colors_here = [int(patch[r, c]) for patch in patches]
            visible = [color for color in colors_here if color != candidate]
            if not visible:
                return None
            base = visible[0]
            if any(color != base for color in visible[1:]):
                return None
            out[r, c] = base
            distinct = set(colors_here)
            assert 1 <= len(distinct) <= 2, "each aligned position should show one true color plus optional damage"
            if candidate in distinct and len(distinct) == 2:
                mixed_positions += 1
                partners.add(base)

    if mixed_positions == 0:
        return None

    for patch in patches:
        mismatch = patch != out
        if not mismatch.any():
            continue
        mismatch_colors = {int(value) for value in patch[mismatch]}
        if mismatch_colors != {candidate}:
            return None

    period_r = _axis_period(out, axis=0)
    period_c = _axis_period(out, axis=1)
    return out, mixed_positions, len(partners), period_r * period_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge several damaged copies of the same periodic pattern into one repaired rectangle.

    Input structure:
    - The border/background is one color, with 2-4 non-overlapping foreground rectangles separated by background gaps.
    - Every foreground rectangle has the same size and contains the same repeated 2-3-color pattern.
    - Some cells inside each rectangle are replaced by a shared damage color, while other copies keep the original color at that same relative position.

    Rule:
    - Detect the equal-sized rectangles and align them by local coordinates inside their bounding boxes.
    - Identify the shared damage color as the candidate whose removal makes all copies agree cell-by-cell and whose repaired overlay yields a small repeating tile.
    - For each relative position, copy the non-damage color seen in any intact rectangle.
    - Output a single repaired rectangle of the common size.

    Key insight:
    - The damaged blocks are complementary views of one template, so the missing cells are recovered by cross-referencing aligned positions across copies, with periodicity used only to recognize which color is the damage marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    patches = _extract_patches(g, bg)
    colors = _scan_colors(patches)

    best_patch: np.ndarray | None = None
    best_key: tuple[int, int, int, int] | None = None

    for candidate in colors:
        rebuilt = _reconstruct_candidate(patches, candidate)
        if rebuilt is None:
            continue
        out, mixed_positions, partner_count, tile_area = rebuilt
        nontrivial_period = int(tile_area < out.size)
        key = (nontrivial_period, -tile_area, partner_count, mixed_positions)
        if best_key is None or key > best_key:
            best_key = key
            best_patch = out

    assert best_patch is not None, "failed to identify the damage color and reconstruct the pattern"
    out_colors = {int(value) for value in best_patch.ravel()}
    assert 2 <= len(out_colors) <= 3, "repaired pattern should use 2-3 colors"
    return best_patch


# --------------- generator ---------------

BACKGROUND = 0
DAMAGE = 5
PATTERN_COLOR_COUNTS = (2, 3)
RECT_COUNTS = (2, 3, 4)
RECT_HEIGHTS = tuple(range(5, 10))
RECT_WIDTHS = tuple(range(5, 10))
TILE_SHAPES = (
    (1, 2),
    (1, 3),
    (2, 1),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
)
LAYOUTS = {
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    ],
}
MAX_GENERATION_ATTEMPTS = 300


def _make_tile(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    feasible = [shape for shape in TILE_SHAPES if shape[0] <= height and shape[1] <= width]
    random.shuffle(feasible)
    for tile_h, tile_w in feasible:
        if tile_h * tile_w < len(colors):
            continue
        for _ in range(60):
            values = colors.copy()
            values.extend(random.choices(colors, k=tile_h * tile_w - len(colors)))
            random.shuffle(values)
            tile = np.asarray(values, dtype=int).reshape(tile_h, tile_w)
            if len({int(value) for value in tile.ravel()}) != len(colors):
                continue
            if _axis_period(tile, 0) != tile_h or _axis_period(tile, 1) != tile_w:
                continue
            return tile
    return None


def _tile_pattern(height: int, width: int, tile: np.ndarray) -> np.ndarray:
    tile_h, tile_w = tile.shape
    pattern = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            pattern[r, c] = int(tile[r % tile_h, c % tile_w])
    return pattern


def _sample_masks(pattern: np.ndarray, rect_count: int) -> list[set[tuple[int, int]]] | None:
    height, width = pattern.shape
    area = height * width
    masks = [set() for _ in range(rect_count)]

    cells_by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            cells_by_color.setdefault(int(pattern[r, c]), []).append((r, c))

    target = random.randint(max(len(cells_by_color), area // 4), min(area - 1, (3 * area) // 5))
    selected: set[tuple[int, int]] = set()
    for positions in cells_by_color.values():
        selected.add(random.choice(positions))
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    while len(selected) < target:
        selected.add(random.choice(all_cells))

    cover_counts: Counter[tuple[int, int]] = Counter()
    for pos in selected:
        damaged_in = random.randint(1, rect_count - 1)
        for idx in random.sample(range(rect_count), damaged_in):
            masks[idx].add(pos)
            cover_counts[pos] += 1

    min_per_patch = max(2, area // 8)
    max_per_patch = (2 * area) // 3
    for idx in range(rect_count):
        refillable = [pos for pos in selected if pos not in masks[idx] and cover_counts[pos] < rect_count - 1]
        while len(masks[idx]) < min_per_patch and refillable:
            pos = random.choice(refillable)
            masks[idx].add(pos)
            cover_counts[pos] += 1
            refillable = [cell for cell in refillable if cell != pos and cover_counts[cell] < rect_count - 1]
        if len(masks[idx]) < min_per_patch or len(masks[idx]) > max_per_patch:
            return None

    if any(cover_counts[pos] == rect_count for pos in selected):
        return None
    if any(len(mask) == 0 for mask in masks):
        return None
    return masks


def _place_rectangles(rect_count: int, rect_h: int, rect_w: int) -> tuple[list[tuple[int, int]], int, int] | None:
    layouts = list(LAYOUTS[rect_count])
    random.shuffle(layouts)
    for layout in layouts:
        max_lr = max(r for r, _ in layout)
        max_lc = max(c for _, c in layout)
        for _ in range(40):
            gap_r = random.randint(1, 4)
            gap_c = random.randint(1, 4)
            top = random.randint(1, 3)
            left = random.randint(1, 3)
            bottom = random.randint(1, 3)
            right = random.randint(1, 3)
            grid_h = top + (max_lr + 1) * rect_h + max_lr * gap_r + bottom
            grid_w = left + (max_lc + 1) * rect_w + max_lc * gap_c + right
            if grid_h > 30 or grid_w > 30:
                continue
            positions = [(top + lr * (rect_h + gap_r), left + lc * (rect_w + gap_c)) for lr, lc in layout]
            return positions, grid_h, grid_w
    return None


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNTS)
    rect_h = random.choice(RECT_HEIGHTS)
    rect_w = random.choice(RECT_WIDTHS)
    palette = [color for color in range(10) if color not in {BACKGROUND, DAMAGE}]
    pattern_colors = random.sample(palette, k=random.choice(PATTERN_COLOR_COUNTS))

    tile = _make_tile(rect_h, rect_w, pattern_colors)
    if tile is None:
        return None
    expected = _tile_pattern(rect_h, rect_w, tile)

    masks = _sample_masks(expected, rect_count)
    if masks is None:
        return None

    placed = _place_rectangles(rect_count, rect_h, rect_w)
    if placed is None:
        return None
    positions, grid_h, grid_w = placed

    patches: list[np.ndarray] = []
    for mask in masks:
        patch = expected.copy()
        for r, c in mask:
            patch[r, c] = DAMAGE
        patches.append(patch)
    random.shuffle(patches)

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = patch

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid damaged-pattern puzzle")
