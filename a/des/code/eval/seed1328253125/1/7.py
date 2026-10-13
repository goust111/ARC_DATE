from __future__ import annotations

import math
import random

import numpy as np


BG_COLOR = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            tight = _trim_mask(variant)
            key = _mask_key(tight)
            if key not in seen:
                seen.add(key)
                out.append(tight)
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "mask": _mask_from_coords(coords),
                }
            )

    return components


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask.astype(bool), factor, axis=0), factor, axis=1)


def _scaled_match(small_mask: np.ndarray, large_mask: np.ndarray) -> int | None:
    for factor in (2, 3):
        if large_mask.shape != (small_mask.shape[0] * factor, small_mask.shape[1] * factor):
            continue
        if np.array_equal(_scale_mask(small_mask, factor), large_mask):
            return factor
    return None


def _legend_row_index(grid: np.ndarray, bg: int) -> int:
    for r in range(grid.shape[0] - 1, -1, -1):
        if np.any(grid[r] != bg):
            assert np.all(grid[r + 1 :] == bg)
            return r
    raise AssertionError("expected a legend row")


def _parse_mix_legend(grid: np.ndarray, bg: int) -> tuple[int, dict[frozenset[int], int]]:
    legend_row = _legend_row_index(grid, bg)
    row = grid[legend_row]
    mapping: dict[frozenset[int], int] = {}
    col = 0

    while col < row.shape[0]:
        if int(row[col]) == bg:
            col += 1
            continue
        start = col
        while col < row.shape[0] and int(row[col]) != bg:
            col += 1
        run = row[start:col]
        assert run.shape[0] == 3, "legend strips must be 1x3"
        left, mid, right = map(int, run)
        key = frozenset((left, right))
        assert key not in mapping or mapping[key] == mid
        mapping[key] = mid

    assert mapping, "legend must contain at least one mixing strip"
    return legend_row, mapping


def _pair_components(components: list[dict]) -> list[tuple[int, int, int]]:
    n = len(components)
    assert n % 2 == 0 and n > 0

    edges: list[tuple[int, int, int]] = []
    adjacency: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            mask_i = components[i]["mask"]
            mask_j = components[j]["mask"]
            factor = _scaled_match(mask_i, mask_j)
            if factor is not None:
                edge = (i, j, factor)
            else:
                factor = _scaled_match(mask_j, mask_i)
                if factor is None:
                    continue
                edge = (j, i, factor)
            edges.append(edge)
            adjacency[edge[0]].append(edge)
            adjacency[edge[1]].append(edge)

    solutions: list[list[tuple[int, int, int]]] = []

    def search(used: set[int], chosen: list[tuple[int, int, int]]) -> None:
        if len(solutions) > 1:
            return
        if len(used) == n:
            solutions.append(chosen.copy())
            return

        first = next(idx for idx in range(n) if idx not in used)
        candidates = []
        for edge in adjacency[first]:
            small_idx, large_idx, _factor = edge
            other = large_idx if small_idx == first else small_idx
            if other in used:
                continue
            candidates.append(edge)

        for edge in candidates:
            small_idx, large_idx, _factor = edge
            if small_idx in used or large_idx in used:
                continue
            used.add(small_idx)
            used.add(large_idx)
            chosen.append(edge)
            search(used, chosen)
            chosen.pop()
            used.remove(large_idx)
            used.remove(small_idx)

    search(set(), [])
    assert len(solutions) == 1, "shape pairing must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome shapes by exact 2x/3x integer scaling, erase the smaller one,
    and recolor the larger one using the bottom-row mixing legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    legend_row, mix_map = _parse_mix_legend(g, bg)
    components = _find_components(g[:legend_row, :], bg)
    pairs = _pair_components(components)

    out = g.copy()
    for small_idx, large_idx, _factor in pairs:
        small = components[small_idx]
        large = components[large_idx]
        key = frozenset((int(small["color"]), int(large["color"])))
        assert key in mix_map, "missing legend entry for a shape pair"
        mix_color = int(mix_map[key])
        out[small["coords"][:, 0], small["coords"][:, 1]] = bg
        out[large["coords"][:, 0], large["coords"][:, 1]] = mix_color

    return out


GEN_RETRIES = 400
MASK_RETRIES = 100
PLACEMENT_PADDING = 2
GRID_EXTRA_ROWS = (4, 8)
GRID_EXTRA_COLS = (5, 10)


def _random_connected_mask() -> np.ndarray:
    for _ in range(MASK_RETRIES):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w)
        area = random.randint(3, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(
                        0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                        for dr, dc in NEIGHBORS4
                    ):
                        frontier.append((r, c))
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True

        tight = _trim_mask(mask)
        if tight.shape == (1, 1):
            continue
        return tight
    raise RuntimeError("failed to sample a connected mask")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _sample_factors(pair_count: int) -> list[int]:
    if pair_count == 2:
        factors = [2, 3]
    else:
        factors = [2, 2, 3] if random.random() < 0.5 else [2, 3, 3]
    random.shuffle(factors)
    return factors


def _make_pair_specs(pair_count: int) -> list[dict]:
    colors = random.sample(PALETTE, 3 * pair_count)
    object_colors = colors[: 2 * pair_count]
    mix_colors = colors[2 * pair_count :]
    seen_keys = set()
    specs = []

    for pair_idx, factor in enumerate(_sample_factors(pair_count)):
        oriented_mask = None
        for _ in range(MASK_RETRIES):
            base = _random_connected_mask()
            candidate = random.choice(_all_orientations(base))
            key = _mask_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            oriented_mask = candidate
            break
        assert oriented_mask is not None

        specs.append(
            {
                "small_mask": oriented_mask,
                "large_mask": _scale_mask(oriented_mask, factor),
                "small_color": object_colors[2 * pair_idx],
                "large_color": object_colors[2 * pair_idx + 1],
                "mix_color": mix_colors[pair_idx],
                "factor": factor,
            }
        )

    return specs


def _grid_size_for(specs: list[dict]) -> tuple[int, int]:
    patches = [spec["small_mask"] for spec in specs] + [spec["large_mask"] for spec in specs]
    max_h = max(mask.shape[0] for mask in patches)
    max_w = max(mask.shape[1] for mask in patches)
    footprint = sum((mask.shape[0] + PLACEMENT_PADDING) * (mask.shape[1] + PLACEMENT_PADDING) for mask in patches)
    side_hint = int(math.ceil(math.sqrt(footprint))) + 2
    legend_min_cols = 4 * len(specs) - 1

    body_rows = max(side_hint, max_h + 2) + random.randint(*GRID_EXTRA_ROWS)
    cols = max(side_hint, max_w + 2, legend_min_cols) + random.randint(*GRID_EXTRA_COLS)
    rows = body_rows + 2
    return rows, cols


def _place_specs(rows: int, cols: int, specs: list[dict]) -> dict[tuple[int, str], tuple[int, int]] | None:
    body_rows = rows - 2
    occupied = np.zeros((body_rows, cols), dtype=bool)
    patches = []
    for pair_idx, spec in enumerate(specs):
        patches.append((pair_idx, "large", spec["large_mask"]))
        patches.append((pair_idx, "small", spec["small_mask"]))
    patches.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)

    placements: dict[tuple[int, str], tuple[int, int]] = {}
    for pair_idx, size_tag, mask in patches:
        h, w = mask.shape
        candidates = [(r, c) for r in range(body_rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(candidates)

        placed = False
        for top, left in candidates:
            r0 = max(0, top - PLACEMENT_PADDING)
            c0 = max(0, left - PLACEMENT_PADDING)
            r1 = min(body_rows, top + h + PLACEMENT_PADDING)
            c1 = min(cols, left + w + PLACEMENT_PADDING)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            placements[(pair_idx, size_tag)] = (top, left)
            placed = True
            break

        if not placed:
            return None

    return placements


def _legend_entries(specs: list[dict]) -> list[tuple[int, int, int]]:
    entries = []
    for spec in specs:
        a = int(spec["small_color"])
        b = int(spec["large_color"])
        c = int(spec["mix_color"])
        if random.random() < 0.5:
            entries.append((a, c, b))
        else:
            entries.append((b, c, a))
    random.shuffle(entries)
    return entries


def _place_legend(grid: np.ndarray, entries: list[tuple[int, int, int]]) -> bool:
    width = grid.shape[1]
    needed = 3 * len(entries) + max(0, len(entries) - 1)
    extra = width - needed
    if extra < 0:
        return False

    slack = [0] * (len(entries) + 1)
    for _ in range(extra):
        slack[random.randrange(len(slack))] += 1

    row = grid.shape[0] - 1
    col = slack[0]
    for idx, (left_color, mid_color, right_color) in enumerate(entries):
        grid[row, col : col + 3] = [left_color, mid_color, right_color]
        col += 3
        if idx < len(entries) - 1:
            col += 1 + slack[idx + 1]
    return True


def _expected_output(grid: np.ndarray, specs: list[dict], placements: dict[tuple[int, str], tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for pair_idx, spec in enumerate(specs):
        small_top, small_left = placements[(pair_idx, "small")]
        large_top, large_left = placements[(pair_idx, "large")]
        _paint_mask(out, spec["small_mask"], small_top, small_left, BG_COLOR)
        _paint_mask(out, spec["large_mask"], large_top, large_left, int(spec["mix_color"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([2, 3])
    specs = _make_pair_specs(pair_count)
    rows, cols = _grid_size_for(specs)
    placements = _place_specs(rows, cols, specs)
    if placements is None:
        return None

    grid = np.full((rows, cols), BG_COLOR, dtype=int)
    for pair_idx, spec in enumerate(specs):
        small_top, small_left = placements[(pair_idx, "small")]
        large_top, large_left = placements[(pair_idx, "large")]
        _paint_mask(grid, spec["small_mask"], small_top, small_left, int(spec["small_color"]))
        _paint_mask(grid, spec["large_mask"], large_top, large_left, int(spec["large_color"]))

    if not _place_legend(grid, _legend_entries(specs)):
        return None

    expected = _expected_output(grid, specs, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid instance for the scaled-shape mixing task."""
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
