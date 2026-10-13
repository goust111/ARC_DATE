from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GREY = 5
NON_BLACK_NON_GREY = [1, 2, 3, 4, 6, 7, 8, 9]
ZONE_WIDTH = 6
ZONE_GAP = 2
RING_HEIGHT = 4
RING_STEP = 5
GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_components(mask: np.ndarray, *, conn8: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_id = 0
    neigh_fn = _neighbors8 if conn8 else _neighbors4

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neigh_fn(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return labels, components


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _component_key(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].min()), int(coords[:, 1].min())


def _read_palette(grid: np.ndarray) -> list[int]:
    palette = []
    for c in range(grid.shape[1]):
        value = int(grid[0, c])
        if value == BLACK:
            break
        palette.append(value)
    assert palette, "expected a non-empty header palette"
    return palette


def _find_seed_pixels(grid: np.ndarray, palette_len: int) -> list[tuple[int, int]]:
    counts: dict[int, int] = defaultdict(int)
    for value in grid.flat:
        counts[int(value)] += 1

    h, w = grid.shape
    seeds = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if r == 0 and c < palette_len:
                continue
            value = int(grid[r, c])
            if value in (BLACK, GREY):
                continue
            if counts[value] != 1:
                continue
            seeds.append((r, c))
    seeds.sort()
    return seeds


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    palette = _read_palette(g)
    seeds = _find_seed_pixels(g, len(palette))
    assert 2 <= len(palette) <= 3
    assert 2 <= len(seeds) <= 3

    grey_labels, _ = _label_components(g == GREY, conn8=True)
    seed_to_component: dict[tuple[int, int], int] = {}
    for seed in seeds:
        r, c = seed
        comp_ids = {
            int(grey_labels[nr, nc])
            for nr, nc in _neighbors8(r, c)
            if _in_bounds(g.shape, nr, nc) and grey_labels[nr, nc] > 0
        }
        assert len(comp_ids) == 1, "each seed must connect to exactly one grey component"
        seed_to_component[seed] = next(iter(comp_ids))

    black_labels, black_components = _label_components(g == BLACK, conn8=False)
    cavities_by_component: dict[int, list[np.ndarray]] = defaultdict(list)
    for coords in black_components:
        if _touches_border(coords, g.shape):
            continue
        touched = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] == GREY:
                    touched.add(int(grey_labels[nr, nc]))
        assert len(touched) == 1, "each cavity must touch exactly one grey component"
        comp_id = next(iter(touched))
        cavities_by_component[comp_id].append(coords)

    for seed in seeds:
        assert seed_to_component[seed] in cavities_by_component, "each seed group must own at least one cavity"

    return {
        "palette": palette,
        "seeds": seeds,
        "seed_to_component": seed_to_component,
        "cavities_by_component": cavities_by_component,
        "black_labels": black_labels,
    }


def _fill_groupwise(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    palette = info["palette"]
    for seed in info["seeds"]:
        component_id = info["seed_to_component"][seed]
        cavities = list(info["cavities_by_component"].get(component_id, []))
        cavities.sort(key=_component_key)
        for index, coords in enumerate(cavities):
            color = int(palette[index % len(palette)])
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _fill_with_seed_colors(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for seed in info["seeds"]:
        component_id = info["seed_to_component"][seed]
        color = int(grid[seed])
        for coords in info["cavities_by_component"].get(component_id, []):
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _fill_global_palette(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    palette = info["palette"]
    all_cavities = []
    for seed in info["seeds"]:
        component_id = info["seed_to_component"][seed]
        for coords in info["cavities_by_component"].get(component_id, []):
            all_cavities.append(coords)
    all_cavities.sort(key=_component_key)
    for index, coords in enumerate(all_cavities):
        out[coords[:, 0], coords[:, 1]] = int(palette[index % len(palette)])
    return out


def _fill_reverse_sorted(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    palette = info["palette"]
    for seed in info["seeds"]:
        component_id = info["seed_to_component"][seed]
        cavities = list(info["cavities_by_component"].get(component_id, []))
        cavities.sort(key=_component_key, reverse=True)
        for index, coords in enumerate(cavities):
            out[coords[:, 0], coords[:, 1]] = int(palette[index % len(palette)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seed-grouped cavities with a cyclic palette read from the top-row header.

    Input structure:
    - A black background contains grey line components that include closed ring boundaries around black cavities.
    - Two or three unique single-pixel seeds lie on the grid border outside the top-row header; each seed is 8-connected to exactly one grey component.
    - The top row starts with a contiguous non-black header sequence, followed by black cells.

    Rule:
    - Read the header palette as the consecutive non-black cells from row 0 starting at column 0.
    - For each border seed, take the 8-connected grey component touching that seed and collect every 4-connected black cavity whose boundary is 4-adjacent to that grey component.
    - Within each seed group, sort cavities by the smallest row in the cavity, then by the smallest column, and fill the i-th cavity with palette[i mod len(palette)].
    - Keep the seeds, grey lines, and header unchanged.

    Key insight:
    - Seed pixels determine which cavities belong together, but the fill colors come only from the shared top-row palette, which restarts independently inside each seed group.
    """
    info = _analyze(grid)
    return _fill_groupwise(np.asarray(grid, dtype=int), info)


def _draw_ring(grid: np.ndarray, top: int, left: int, width: int) -> np.ndarray:
    grid[top, left : left + width] = GREY
    grid[top + RING_HEIGHT - 1, left : left + width] = GREY
    grid[top : top + RING_HEIGHT, left] = GREY
    grid[top : top + RING_HEIGHT, left + width - 1] = GREY
    rows = np.arange(top + 1, top + RING_HEIGHT - 1)
    cols = np.arange(left + 1, left + width - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.column_stack([rr.ravel(), cc.ravel()])


def _valid_group_counts(seed_count: int, palette_len: int) -> list[tuple[int, ...]]:
    choices = []

    def rec(prefix: list[int], remaining: int):
        if len(prefix) == seed_count:
            if remaining == 0:
                counts = tuple(prefix)
                if max(counts) <= 4 and max(counts) > palette_len and len(set(counts)) > 1:
                    choices.append(counts)
            return
        slots_left = seed_count - len(prefix)
        for value in range(1, 5):
            min_rest = slots_left - 1
            max_rest = 4 * (slots_left - 1)
            if remaining - value < min_rest or remaining - value > max_rest:
                continue
            prefix.append(value)
            rec(prefix, remaining - value)
            prefix.pop()

    for total in range(4, 9):
        rec([], total)
    return choices


def _build_case(rng: random.Random) -> np.ndarray | None:
    seed_count = rng.choice([2, 2, 3])
    palette_len = rng.choice([2, 3])
    count_options = _valid_group_counts(seed_count, palette_len)
    if not count_options:
        return None
    group_counts = list(rng.choice(count_options))

    palette = rng.sample(NON_BLACK_NON_GREY, palette_len)
    seed_pool = [color for color in NON_BLACK_NON_GREY if color not in palette]
    seed_colors = rng.sample(seed_pool, seed_count)
    sides = [rng.choice(["left", "right"]) for _ in range(seed_count)]
    group_shifts = [rng.randint(0, 1) for _ in range(seed_count)]

    base_top = 2
    last_ring_bottom = max(base_top + shift + (count - 1) * RING_STEP + RING_HEIGHT - 1 for shift, count in zip(group_shifts, group_counts))
    seed_row = last_ring_bottom + 2 + rng.randint(0, 1)
    h = seed_row + 1
    w = (seed_count - 1) * (ZONE_WIDTH + ZONE_GAP) + ZONE_WIDTH + rng.randint(0, 2)
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :palette_len] = np.array(palette, dtype=int)

    all_cavity_coords: list[np.ndarray] = []
    for group_idx in range(seed_count):
        zone_left = group_idx * (ZONE_WIDTH + ZONE_GAP)
        side = sides[group_idx]
        trunk_col = zone_left if side == "left" else zone_left + ZONE_WIDTH - 1
        grid[1:seed_row, trunk_col] = GREY
        grid[seed_row, trunk_col] = seed_colors[group_idx]

        for cavity_idx in range(group_counts[group_idx]):
            top = base_top + group_shifts[group_idx] + cavity_idx * RING_STEP
            cavity_w = rng.choice([2, 3])
            ring_w = cavity_w + 2
            left = zone_left + 1 if side == "left" else zone_left + ZONE_WIDTH - 1 - ring_w
            if left < 0 or left + ring_w > w:
                return None
            cavity = _draw_ring(grid, top, left, ring_w)
            all_cavity_coords.append(cavity)

    info = _analyze(grid)
    group_sizes = []
    for seed in info["seeds"]:
        component_id = info["seed_to_component"][seed]
        group_sizes.append(len(info["cavities_by_component"][component_id]))

    if sorted(group_sizes) != sorted(group_counts):
        return None
    if len(all_cavity_coords) != sum(group_counts):
        return None
    if not (4 <= len(all_cavity_coords) <= 8):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, _fill_with_seed_colors(grid, info)):
        return None
    if np.array_equal(out, _fill_global_palette(grid, info)):
        return None
    if np.array_equal(out, _fill_reverse_sorted(grid, info)):
        return None

    palette_set = set(info["palette"])
    seed_color_set = {int(grid[seed]) for seed in info["seeds"]}
    if palette_set & seed_color_set:
        return None
    if max(group_sizes) <= len(info["palette"]):
        return None
    if max(group_sizes) <= 1 or len(set(group_sizes)) == 1:
        return None

    filled_colors = set(int(v) for v in np.unique(out)) - {BLACK, GREY}
    if len(filled_colors) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _build_case(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
