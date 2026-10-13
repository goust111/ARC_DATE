from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
GREY = 5
SPRITE_COLORS = (1, 2, 3, 4)
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))  # TL, TR, BL, BR
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _make_sprite(color: int, missing: tuple[int, int]) -> np.ndarray:
    sprite = np.full((2, 2), int(color), dtype=int)
    sprite[missing] = BLACK
    return sprite


def _slot_capacity(band_height: int, width: int) -> int:
    return (band_height // 2) * (width // 2)


def _collect_sprites(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    sprites: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, GREY}):
        for coords in _components_of_color(grid, color):
            assert coords.shape[0] == 3, "each colored object must contain exactly three pixels"
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each sprite must occupy a 2x2 bbox"
            sprite = _crop(grid, box)
            miss = np.argwhere(sprite == bg)
            assert miss.shape == (1, 2), "each L sprite must have exactly one missing corner"
            missing = tuple(map(int, miss[0]))
            sprites.append(
                {
                    "color": int(color),
                    "bbox": box,
                    "sprite": sprite,
                    "missing": missing,
                    "group": MISSING_CORNERS.index(missing),
                }
            )
    assert 8 <= len(sprites) <= 12, "expected 8-12 L sprites"
    groups_present = {sprite["group"] for sprite in sprites}
    assert groups_present == {0, 1, 2, 3}, "all four missing-corner orientations must appear"
    return sprites


def _collect_bars(grid: np.ndarray) -> list[dict]:
    bars = []
    for coords in _components_of_color(grid, GREY):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert c0 == c1, "each grey component must be a vertical bar"
        assert coords.shape[0] == r1 - r0 + 1, "grey bars must be solid 1-pixel-wide segments"
        bars.append({"bbox": box, "length": int(coords.shape[0])})
    assert len(bars) == 4, "expected exactly four grey bars"
    lengths = [bar["length"] for bar in bars]
    assert len(set(lengths)) == 4, "grey bar lengths must be distinct"
    bars.sort(key=lambda bar: (bar["bbox"][0], bar["bbox"][1]))
    return bars


def _render_output(width: int, heights: list[int], grouped_sprites: list[list[dict]], bg: int = BLACK) -> np.ndarray:
    out = np.full((sum(heights), width), int(bg), dtype=int)
    band_top = 0
    for group_index, band_height in enumerate(heights):
        sprites = sorted(grouped_sprites[group_index], key=lambda item: (item["bbox"][0], item["bbox"][1]))
        placed = 0
        for local_r in range(0, band_height - 1, 2):
            for local_c in range(0, width - 1, 2):
                if placed >= len(sprites):
                    break
                _paste(out, sprites[placed]["sprite"], band_top + local_r, local_c, transparent=bg)
                placed += 1
            if placed >= len(sprites):
                break
        assert placed == len(sprites), "band height must be sufficient for its sprite group"
        band_top += band_height
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort L sprites into horizontal strips by missing-corner orientation, with strip heights encoded by grey bar lengths.

    Input structure:
    - A black-background grid contains 8-12 separated monochrome L sprites, each occupying a 2x2 box with one missing corner.
    - Sprite colors may vary independently of orientation; the four missing-corner orientations all appear at least once.
    - The grid also contains exactly four grey vertical 1-pixel-wide bars whose lengths are distinct.

    Rule:
    - Group sprites by their missing corner: top-left, top-right, bottom-left, bottom-right.
    - Sort the four grey bars from top to bottom; their lengths give the heights of the four output strips for those four groups in that same order.
    - Build an output whose width matches the input and whose height is the sum of the four bar lengths, then tile each group's cropped 2x2 sprites into its strip in source reading order (top-to-bottom, then left-to-right), using row-major 2x2 slots and leaving all unused cells black.

    Key insight:
    - Orientation decides which strip a sprite belongs to, while the independent grey bars do not mark positions directly: they encode the strip heights that control how the packed sprites occupy space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert int(bg) == BLACK, "background must be black"

    sprites = _collect_sprites(g)
    bars = _collect_bars(g)
    grouped: list[list[dict]] = [[], [], [], []]
    for sprite in sprites:
        grouped[int(sprite["group"])].append(sprite)

    heights = [int(bar["length"]) for bar in bars]
    for idx, band_height in enumerate(heights):
        assert _slot_capacity(band_height, g.shape[1]) >= len(grouped[idx]), "strip cannot hold its sprite group"
    return _render_output(g.shape[1], heights, grouped, bg=bg)


def _choose_dimensions(force_wrap: bool) -> tuple[int, int]:
    if force_wrap:
        width = random.randint(12, 14)
    else:
        width = random.randint(14, 18)
    left_zone_width = width // 2
    return width, left_zone_width


def _choose_band_heights() -> list[int]:
    heights = random.sample([2, 3, 4, 5, 6], 4)
    random.shuffle(heights)
    return heights


def _distribute_counts(total: int, capacities: list[int], slots_per_row: int, force_wrap: bool) -> list[int] | None:
    counts = [1, 1, 1, 1]
    remaining = total - 4
    per_group_limit = capacities[:]

    heavy_group = None
    if force_wrap:
        candidates = [idx for idx, cap in enumerate(capacities) if cap >= slots_per_row + 1]
        if not candidates:
            return None
        heavy_group = random.choice(candidates)
        heavy_max = min(capacities[heavy_group], total - 3)
        heavy_min = slots_per_row + 1
        if heavy_max < heavy_min:
            return None
        heavy_count = random.randint(heavy_min, heavy_max)
        counts[heavy_group] = heavy_count
        remaining -= heavy_count - 1

    while remaining > 0:
        candidates = []
        for idx in range(4):
            limit = per_group_limit[idx]
            if not force_wrap:
                limit = min(limit, slots_per_row)
            if counts[idx] < limit:
                candidates.append(idx)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        counts[chosen] += 1
        remaining -= 1

    if force_wrap and heavy_group is not None and counts[heavy_group] <= slots_per_row:
        return None
    if not force_wrap and any(count > slots_per_row for count in counts):
        return None
    return counts


def _assign_colors(counts: list[int]) -> list[list[int]]:
    group_colors: list[list[int]] = []
    bridge_color = random.choice(SPRITE_COLORS)
    bridge_groups = random.sample(range(4), 2)
    mixed_candidates = [idx for idx, count in enumerate(counts) if count >= 2]
    mixed_group = random.choice(mixed_candidates)
    mixed_pair = random.sample(SPRITE_COLORS, 2)

    for group_idx, count in enumerate(counts):
        colors = [random.choice(SPRITE_COLORS) for _ in range(count)]
        if group_idx in bridge_groups:
            colors[0] = bridge_color
        if group_idx == mixed_group:
            colors[0], colors[1] = mixed_pair
        group_colors.append(colors)
    return group_colors


def _sprite_candidates(grid_h: int, left_zone_width: int, occupied: np.ndarray) -> list[tuple[int, int]]:
    candidates = []
    for r in range(grid_h - 1):
        for c in range(left_zone_width - 1):
            rr0 = max(0, r - 1)
            rr1 = min(grid_h, r + 3)
            cc0 = max(0, c - 1)
            cc1 = min(left_zone_width, c + 3)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_sprites(
    grid: np.ndarray,
    counts: list[int],
    group_colors: list[list[int]],
    left_zone_width: int,
) -> list[dict] | None:
    occupied = np.zeros((grid.shape[0], left_zone_width), dtype=bool)
    to_place = []
    for group_idx, count in enumerate(counts):
        for color in group_colors[group_idx]:
            to_place.append({"group": group_idx, "color": int(color), "missing": MISSING_CORNERS[group_idx]})
    random.shuffle(to_place)

    placed = []
    for spec in to_place:
        candidates = _sprite_candidates(grid.shape[0], left_zone_width, occupied)
        if not candidates:
            return None
        r, c = candidates[0]
        sprite = _make_sprite(spec["color"], spec["missing"])
        _paste(grid, sprite, r, c, transparent=BLACK)
        occupied[r : r + 2, c : c + 2] = True
        placed.append(
            {
                "group": spec["group"],
                "color": spec["color"],
                "missing": spec["missing"],
                "bbox": (r, c, r + 1, c + 1),
                "sprite": sprite,
            }
        )
    return placed


def _place_bars(grid: np.ndarray, heights: list[int], right_zone_start: int) -> None:
    top_margin = random.randint(0, 1)
    gaps = [random.randint(2, 3) for _ in range(3)]
    current_row = top_margin
    for idx, length in enumerate(heights):
        col = random.randint(right_zone_start, grid.shape[1] - 1)
        grid[current_row : current_row + length, col] = GREY
        current_row += length
        if idx < 3:
            current_row += gaps[idx]


def _build_expected(width: int, heights: list[int], placed_sprites: list[dict]) -> np.ndarray:
    grouped: list[list[dict]] = [[], [], [], []]
    for sprite in placed_sprites:
        grouped[int(sprite["group"])].append(sprite)
    return _render_output(width, heights, grouped, bg=BLACK)


def _validate_generated_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    counts: list[int],
    heights: list[int],
    force_wrap: bool,
) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    sprites = _collect_sprites(grid)
    grouped = defaultdict(list)
    for sprite in sprites:
        grouped[int(sprite["group"])].append(sprite)
    if [len(grouped[idx]) for idx in range(4)] != counts:
        return False

    color_sets = [set(int(sprite["color"]) for sprite in grouped[idx]) for idx in range(4)]
    if not any(len(colors) >= 2 for colors in color_sets):
        return False
    color_to_groups = defaultdict(set)
    for idx, sprites_in_group in grouped.items():
        for sprite in sprites_in_group:
            color_to_groups[int(sprite["color"])].add(int(idx))
    if max(len(groups) for groups in color_to_groups.values()) < 2:
        return False

    bars = _collect_bars(grid)
    if [bar["length"] for bar in bars] != heights:
        return False

    slots_per_row = grid.shape[1] // 2
    wrap_happened = any(count > slots_per_row for count in counts)
    if force_wrap and not wrap_happened:
        return False
    if not force_wrap and wrap_happened:
        return False

    if len(set(tuple(row.tolist()) for row in expected)) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    force_wrap = random.random() < 0.65
    width, left_zone_width = _choose_dimensions(force_wrap)
    right_zone_start = width // 2 + 1
    if right_zone_start >= width:
        return None

    heights = _choose_band_heights()
    capacities = [_slot_capacity(height, width) for height in heights]
    slots_per_row = width // 2
    total = random.randint(10, 12) if force_wrap else random.randint(8, 11)
    counts = _distribute_counts(total, capacities, slots_per_row, force_wrap)
    if counts is None:
        return None

    top_margin = 1
    bottom_margin = 1
    bar_gaps = [random.randint(2, 3) for _ in range(3)]
    min_grid_h = top_margin + sum(heights) + sum(bar_gaps) + bottom_margin
    grid_h = max(min_grid_h, random.randint(18, 24))
    if grid_h > 30:
        return None

    grid = np.full((grid_h, width), BLACK, dtype=int)
    group_colors = _assign_colors(counts)
    placed_sprites = _place_sprites(grid, counts, group_colors, left_zone_width)
    if placed_sprites is None:
        return None

    # Reserve the divider column as black to keep the two input object families visually separated.
    divider_col = width // 2
    grid[:, divider_col] = BLACK
    _place_bars(grid, heights, right_zone_start)

    expected = _build_expected(width, heights, placed_sprites)
    if expected.shape[0] > 30:
        return None
    if not _validate_generated_instance(grid, expected, counts, heights, force_wrap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
