from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
DIVIDER_BAND = 3
CORE_SIZE = 5
GEN_ATTEMPTS = 400
QUADRANT_ORDER = ("tl", "tr", "br", "bl")
ANCHOR_BY_QUADRANT = {
    0: (1, 1),
    1: (1, 3),
    2: (3, 3),
    3: (3, 1),
}
TEMPLATES = (
    np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
)
LAYOUTS = tuple(
    (size, n_rows, n_cols)
    for size in (5, 7)
    for n_rows in (2, 3)
    for n_cols in (2, 3)
    if max(
        n_rows * size + (n_rows + 1) * DIVIDER_BAND,
        n_cols * size + (n_cols + 1) * DIVIDER_BAND,
    )
    <= 30
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _quadrant_of(coord: tuple[int, int], size: int) -> int:
    r, c = map(int, coord)
    mid = size // 2
    if r < mid and c < mid:
        return 0
    if r < mid and c > mid:
        return 1
    if r > mid and c > mid:
        return 2
    if r > mid and c < mid:
        return 3
    raise ValueError("anchor lies on a central axis")


def _full_line_indices(grid: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(np.all(grid == color, axis=1))
    cols = np.flatnonzero(np.all(grid == color, axis=0))
    return rows.astype(int), cols.astype(int)


def _find_divider_color(grid: np.ndarray) -> int:
    best_color = None
    best_score = -1
    for color in np.unique(grid):
        rows, cols = _full_line_indices(grid, int(color))
        if len(rows) < 3 or len(cols) < 3:
            continue
        score = int(len(rows) + len(cols))
        if score > best_score:
            best_score = score
            best_color = int(color)
    if best_color is None:
        raise ValueError("failed to identify divider color")
    return best_color


def _progression_candidates(indices: np.ndarray) -> list[tuple[int, ...]]:
    values = sorted(set(map(int, indices)))
    value_set = set(values)
    candidates: set[tuple[int, ...]] = set()
    for i, start in enumerate(values):
        for nxt in values[i + 1 :]:
            diff = int(nxt - start)
            if diff <= DIVIDER_BAND:
                continue
            seq = [start]
            cur = start
            while cur in value_set:
                if cur != start:
                    seq.append(cur)
                cur += diff
            if len(seq) >= 3:
                candidates.add(tuple(seq))
    return sorted(candidates, key=lambda seq: (-len(seq), seq))


def _try_regions(
    grid: np.ndarray, divider: int, row_seq: tuple[int, ...], col_seq: tuple[int, ...]
) -> tuple[int, list[tuple[int, int, int, int]]] | None:
    if len(row_seq) < 3 or len(col_seq) < 3:
        return None
    height, width = grid.shape
    if any(r <= 0 or r >= height - 1 for r in row_seq):
        return None
    if any(c <= 0 or c >= width - 1 for c in col_seq):
        return None
    if any(np.count_nonzero(grid[r - 1] == divider) <= width // 2 or np.count_nonzero(grid[r + 1] == divider) <= width // 2 for r in row_seq):
        return None
    if any(np.count_nonzero(grid[:, c - 1] == divider) <= height // 2 or np.count_nonzero(grid[:, c + 1] == divider) <= height // 2 for c in col_seq):
        return None

    row_steps = np.diff(np.array(row_seq, dtype=int))
    col_steps = np.diff(np.array(col_seq, dtype=int))
    if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
        return None

    size_r = int(row_steps[0] - DIVIDER_BAND)
    size_c = int(col_steps[0] - DIVIDER_BAND)
    if size_r != size_c or size_r <= 0 or size_r % 2 == 0:
        return None

    regions = []
    for r_idx in range(len(row_seq) - 1):
        rs = int(row_seq[r_idx] + 2)
        re = int(row_seq[r_idx + 1] - 2)
        if rs < 0 or re >= grid.shape[0] or re - rs + 1 != size_r:
            return None
        for c_idx in range(len(col_seq) - 1):
            cs = int(col_seq[c_idx] + 2)
            ce = int(col_seq[c_idx + 1] - 2)
            if cs < 0 or ce >= grid.shape[1] or ce - cs + 1 != size_r:
                return None
            corners = _corner_coords((rs, re, cs, ce))
            marked = sum(int(grid[r, c]) != divider for r, c in corners)
            if marked != 1:
                return None
            regions.append((rs, re, cs, ce))
    return size_r, regions


def _parse_regions(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    divider = _find_divider_color(grid)
    full_rows, full_cols = _full_line_indices(grid, divider)
    row_candidates = _progression_candidates(full_rows)
    col_candidates = _progression_candidates(full_cols)
    if not row_candidates or not col_candidates:
        raise ValueError("divider grid is incomplete")

    matches = []
    for row_seq in row_candidates:
        for col_seq in col_candidates:
            tried = _try_regions(grid, divider, row_seq, col_seq)
            if tried is not None:
                matches.append(tried)

    if not matches:
        raise ValueError("failed to parse any divider-grid candidate")
    max_regions = max(len(regions) for _size, regions in matches)
    unique_matches = {
        (size, tuple(regions))
        for size, regions in matches
        if len(regions) == max_regions
    }
    if len(unique_matches) != 1:
        raise ValueError("failed to uniquely parse the divider grid")
    size, regions = next(iter(unique_matches))
    return divider, int(size), list(regions)


def _corner_coords(bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    rs, re, cs, ce = bounds
    return [
        (rs - 1, cs - 1),
        (rs - 1, ce + 1),
        (re + 1, ce + 1),
        (re + 1, cs - 1),
    ]


def _target_quadrant(grid: np.ndarray, bounds: tuple[int, int, int, int], divider: int) -> int:
    marked = [idx for idx, (r, c) in enumerate(_corner_coords(bounds)) if int(grid[r, c]) != divider]
    if len(marked) != 1:
        raise ValueError("each region must have exactly one marked outer corner")
    return int(marked[0])


def _find_anchor(patch: np.ndarray) -> tuple[int, int]:
    bg = _most_common_color(patch)
    values, counts = np.unique(patch, return_counts=True)
    singleton_colors = [int(v) for v, c in zip(values, counts) if int(v) != bg and int(c) == 1]
    if len(singleton_colors) != 1:
        raise ValueError("region must contain exactly one singleton anchor color")
    coords = np.argwhere(patch == singleton_colors[0])
    if len(coords) != 1:
        raise ValueError("anchor pixel must be unique")
    return int(coords[0, 0]), int(coords[0, 1])


def _anchor_quadrant(patch: np.ndarray) -> int:
    return _quadrant_of(_find_anchor(patch), patch.shape[0])


def _component_count(mask: np.ndarray) -> int:
    seen: set[tuple[int, int]] = set()
    count = 0
    for r, c in np.argwhere(mask):
        node = (int(r), int(c))
        if node in seen:
            continue
        count += 1
        queue = deque([node])
        seen.add(node)
        while queue:
            rr, cc = queue.popleft()
            for dr, dc in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                nr, nc = rr + dr, cc + dc
                nxt = (nr, nc)
                if (
                    0 <= nr < mask.shape[0]
                    and 0 <= nc < mask.shape[1]
                    and mask[nr, nc]
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)
    return count


def _render_region(
    size: int,
    template_idx: int,
    visible_anchor_quadrant: int,
    body_rotation: int,
    body_color: int,
    anchor_color: int,
) -> np.ndarray:
    core = np.full((CORE_SIZE, CORE_SIZE), BACKGROUND, dtype=int)
    mask = TEMPLATES[int(template_idx)] > 0
    core[mask] = int(body_color)

    canonical_quadrant = (int(visible_anchor_quadrant) + int(body_rotation)) % 4
    ar, ac = ANCHOR_BY_QUADRANT[canonical_quadrant]
    core[ar, ac] = int(anchor_color)
    core = np.rot90(core, int(body_rotation))

    if size == CORE_SIZE:
        return core

    patch = np.full((size, size), BACKGROUND, dtype=int)
    offset = (size - CORE_SIZE) // 2
    patch[offset : offset + CORE_SIZE, offset : offset + CORE_SIZE] = core
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each partition so its anchor pixel ends up in the quadrant indicated by that partition's marked outer corner.

    Input structure:
    - A divider-color grid splits the canvas into odd-sized square partitions.
    - Each partition contains one small sprite on a background, and exactly one sprite pixel is a singleton-color anchor.
    - Around each partition, exactly one of the four outer-corner marker cells differs from the divider color.

    Rule:
    - Parse every square partition independently from the divider grid.
    - For each partition, consider the four quarter-turns of that partition's interior only and pick the unique rotation that moves the anchor into the quadrant adjacent to the marked outer corner.
    - Keep the divider grid and all corner markers unchanged, and replace each partition interior by its chosen rotation.

    Key insight:
    - The corner marker does not label the region; it specifies a target quadrant, and the anchor's current quadrant determines the unique local rotation needed to reach it.
    """
    g = np.asarray(grid, dtype=int)
    divider, _size, regions = _parse_regions(g)
    out = g.copy()

    for bounds in regions:
        rs, re, cs, ce = bounds
        patch = g[rs : re + 1, cs : ce + 1]
        target = _target_quadrant(g, bounds, divider)

        chosen = None
        for k in range(4):
            rotated = np.rot90(patch, k)
            if _anchor_quadrant(rotated) == target:
                if chosen is not None:
                    raise ValueError("rotation is not uniquely determined")
                chosen = rotated

        if chosen is None:
            raise ValueError("no valid rotation found")
        out[rs : re + 1, cs : ce + 1] = chosen

    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray]:
    size, n_rows, n_cols = random.choice(LAYOUTS)
    height = n_rows * size + (n_rows + 1) * DIVIDER_BAND
    width = n_cols * size + (n_cols + 1) * DIVIDER_BAND

    palette = list(range(1, 10))
    divider_color, marker_color = random.sample(palette, 2)
    sprite_palette = [color for color in palette if color not in {divider_color, marker_color}]

    grid = np.full((height, width), divider_color, dtype=int)
    expected = np.full((height, width), divider_color, dtype=int)

    full_rows = [1 + idx * (size + DIVIDER_BAND) for idx in range(n_rows + 1)]
    full_cols = [1 + idx * (size + DIVIDER_BAND) for idx in range(n_cols + 1)]
    region_bounds = [
        (full_rows[r] + 2, full_rows[r + 1] - 2, full_cols[c] + 2, full_cols[c + 1] - 2)
        for r in range(n_rows)
        for c in range(n_cols)
    ]
    n_regions = len(region_bounds)

    current_quadrants = list(range(4)) + [random.randrange(4) for _ in range(n_regions - 4)]
    deltas = list(range(4)) + [random.randrange(4) for _ in range(n_regions - 4)]
    random.shuffle(current_quadrants)
    random.shuffle(deltas)

    duplicate_template = random.randrange(len(TEMPLATES))
    duplicate_rotation = random.randrange(4)
    duplicate_body = random.choice(sprite_palette)
    duplicate_anchor = random.choice([c for c in sprite_palette if c != duplicate_body])

    params = []
    for idx in range(n_regions):
        body_color = random.choice(sprite_palette)
        anchor_color = random.choice([c for c in sprite_palette if c != body_color])
        params.append(
            {
                "current": int(current_quadrants[idx]),
                "delta": int(deltas[idx]),
                "template": random.randrange(len(TEMPLATES)),
                "body_rotation": random.randrange(4),
                "body_color": int(body_color),
                "anchor_color": int(anchor_color),
            }
        )

    params[0]["template"] = duplicate_template
    params[1]["template"] = duplicate_template
    params[0]["body_rotation"] = duplicate_rotation
    params[1]["body_rotation"] = duplicate_rotation
    params[0]["body_color"] = duplicate_body
    params[1]["body_color"] = duplicate_body
    params[0]["anchor_color"] = duplicate_anchor
    params[1]["anchor_color"] = duplicate_anchor
    if params[0]["current"] == params[1]["current"]:
        params[1]["current"] = (params[0]["current"] + 1) % 4

    order = list(range(n_regions))
    random.shuffle(order)

    for slot_idx, region_idx in enumerate(order):
        bounds = region_bounds[slot_idx]
        rs, re, cs, ce = bounds
        spec = params[region_idx]
        target = (spec["current"] - spec["delta"]) % 4

        patch = _render_region(
            size=size,
            template_idx=spec["template"],
            visible_anchor_quadrant=spec["current"],
            body_rotation=spec["body_rotation"],
            body_color=spec["body_color"],
            anchor_color=spec["anchor_color"],
        )
        rotated = np.rot90(patch, spec["delta"])

        grid[rs : re + 1, cs : ce + 1] = patch
        expected[rs : re + 1, cs : ce + 1] = rotated

        mark_r, mark_c = _corner_coords(bounds)[target]
        grid[mark_r, mark_c] = marker_color
        expected[mark_r, mark_c] = marker_color

    return grid, expected


def _validate_attempt(grid: np.ndarray, expected: np.ndarray) -> bool:
    divider, size, regions = _parse_regions(grid)
    if size not in (5, 7) or len(regions) < 4:
        return False

    seen_rotations = set()
    seen_currents = set()
    mask_counter: Counter[tuple[tuple[int, ...], ...]] = Counter()

    for bounds in regions:
        rs, re, cs, ce = bounds
        patch = grid[rs : re + 1, cs : ce + 1]
        target = _target_quadrant(grid, bounds, divider)
        current = _anchor_quadrant(patch)
        seen_currents.add(current)

        values, counts = np.unique(patch, return_counts=True)
        bg = _most_common_color(patch)
        non_bg = {int(v): int(c) for v, c in zip(values, counts) if int(v) != bg}
        if len(non_bg) != 2 or sorted(non_bg.values())[:1] != [1]:
            return False
        if _component_count(patch != bg) != 1:
            return False

        valid_rotations = [k for k in range(4) if _anchor_quadrant(np.rot90(patch, k)) == target]
        if len(valid_rotations) != 1:
            return False
        seen_rotations.add(valid_rotations[0])
        mask_counter[tuple(tuple(int(x) for x in row) for row in (patch != bg).astype(int))] += 1

    if seen_rotations != {0, 1, 2, 3}:
        return False
    if seen_currents != {0, 1, 2, 3}:
        return False
    if max(mask_counter.values()) < 2:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected = _build_attempt()
        if _validate_attempt(grid, expected):
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle")
