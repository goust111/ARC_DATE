from __future__ import annotations

import numpy as np


BLOCK_COLORS = (1, 2, 3, 4, 6)
COUNT_PROFILES = (
    (4, 3, 2),
    (4, 3, 1),
    (4, 2, 1),
    (3, 2, 1),
)
GRID_MIN = 18
GRID_MAX = 24
GEN_ATTEMPTS = 400
INNER_ATTEMPTS = 120


def _count_solid_2x2_blocks(grid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color == 0:
                continue
            if (
                int(grid[r + 1, c]) == color
                and int(grid[r, c + 1]) == color
                and int(grid[r + 1, c + 1]) == color
            ):
                counts[color] = counts.get(color, 0) + 1
    return counts


def _build_output_from_counts(counts: dict[int, int]) -> np.ndarray:
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    widths = [count for _color, count in ordered]
    assert widths and len(widths) == len(set(widths))

    total_width = sum(widths)
    side = 2 * total_width + 1
    out = np.zeros((side, side), dtype=int)
    center = total_width
    radius = total_width

    for color, width in ordered:
        for layer in range(radius, radius - width, -1):
            r0 = center - layer
            r1 = center + layer
            c0 = center - layer
            c1 = center + layer
            out[r0 : r1 + 1, c0] = int(color)
            out[r0 : r1 + 1, c1] = int(color)
            out[r0, c0 : c1 + 1] = int(color)
            out[r1, c0 : c1 + 1] = int(color)
        radius -= width

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count solid 2x2 color blocks, then turn those counts into outside-in concentric square rings whose widths equal the counts.

    Input structure:
    - The grid has a black background with 3 or more non-black colors.
    - Each active color may appear in isolated solid 2x2 squares plus same-color single-pixel distractors.
    - Only fully monochrome non-black 2x2 windows are meaningful; singleton noise is irrelevant.

    Rule:
    - Scan every 2x2 window and count, for each color, how many windows are filled entirely by that same non-black color.
    - Ignore all other pixels and windows.
    - Sort colors by these block counts from largest to smallest; ties are not intended.
    - Let S be the sum of all counts. Create a black (2S+1)x(2S+1) output grid.
    - Fill concentric square rings from the outside inward, giving each color a ring band whose thickness equals its own count.

    Key insight:
    - The same per-color 2x2-block count controls both the rank order of colors and the geometric width of each ring, so noise removal is necessary before either property can be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    counts = _count_solid_2x2_blocks(g)
    assert counts, "expected at least one solid 2x2 block"
    values = list(counts.values())
    assert len(values) == len(set(values)), "block counts must be distinct"
    return _build_output_from_counts(counts)


def _sorted_colors_by_value(values: dict[int, int]) -> tuple[int, ...]:
    return tuple(color for color, _value in sorted(values.items(), key=lambda item: item[1], reverse=True))


def _sample_noise_counts(block_counts: dict[int, int]) -> dict[int, int] | None:
    colors = list(block_counts)
    block_order = _sorted_colors_by_value(block_counts)

    for _ in range(INNER_ATTEMPTS):
        noise_counts = {color: int(np.random.randint(1, 7)) for color in colors}
        pixel_totals = {color: 4 * block_counts[color] + noise_counts[color] for color in colors}
        component_totals = {color: block_counts[color] + noise_counts[color] for color in colors}

        pixel_values = list(pixel_totals.values())
        component_values = list(component_totals.values())
        if len(pixel_values) != len(set(pixel_values)):
            continue
        if len(component_values) != len(set(component_values)):
            continue
        if _sorted_colors_by_value(pixel_totals) == block_order:
            continue
        if _sorted_colors_by_value(component_totals) == block_order:
            continue
        return noise_counts

    return None


def _can_place(grid: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rows, cols = grid.shape
    if r < 1 or c < 1 or r + h > rows - 1 or c + w > cols - 1:
        return False
    r0 = max(0, r - 1)
    r1 = min(rows, r + h + 1)
    c0 = max(0, c - 1)
    c1 = min(cols, c + w + 1)
    return not np.any(grid[r0:r1, c0:c1] != 0)


def _place_patch(grid: np.ndarray, color: int, h: int, w: int) -> bool:
    rows, cols = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, rows - h):
        for c in range(1, cols - w):
            if _can_place(grid, r, c, h, w):
                candidates.append((r, c))
    if not candidates:
        return False
    r, c = candidates[int(np.random.randint(len(candidates)))]
    grid[r : r + h, c : c + w] = int(color)
    return True


def _render_input(block_counts: dict[int, int], noise_counts: dict[int, int]) -> np.ndarray | None:
    total_objects = sum(block_counts.values()) + sum(noise_counts.values())
    base = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    extra = 2 if total_objects >= 16 else 0
    rows = min(30, base + extra + int(np.random.randint(0, 2)))
    cols = min(30, base + extra + int(np.random.randint(0, 2)))
    grid = np.zeros((rows, cols), dtype=int)

    items: list[tuple[int, int, int]] = []
    for color, count in block_counts.items():
        items.extend((color, 2, 2) for _ in range(count))
    for color, count in noise_counts.items():
        items.extend((color, 1, 1) for _ in range(count))

    areas = np.array([h * w for _color, h, w in items], dtype=int)
    order = np.argsort(-areas)
    for idx in order:
        color, h, w = items[int(idx)]
        if not _place_patch(grid, color, h, w):
            return None
    return grid


def _component_shapes_are_clean(grid: np.ndarray) -> bool:
    rows, cols = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            if len(coords) == 1:
                continue
            if len(coords) != 4:
                return False
            rs = sorted({rr for rr, _cc in coords})
            cs = sorted({cc for _rr, cc in coords})
            if rs != [rs[0], rs[0] + 1] or cs != [cs[0], cs[0] + 1]:
                return False
    return True


def _validate_generated_grid(grid: np.ndarray, block_counts: dict[int, int], noise_counts: dict[int, int]) -> bool:
    if not _component_shapes_are_clean(grid):
        return False

    counted = _count_solid_2x2_blocks(grid)
    if counted != block_counts:
        return False

    for color, noise_count in noise_counts.items():
        if int(np.count_nonzero(grid == color)) != 4 * block_counts[color] + noise_count:
            return False

    out = main(grid)
    expected = _build_output_from_counts(block_counts)
    if not np.array_equal(out, expected):
        return False

    unique_colors = {int(v) for v in grid.ravel() if int(v) != 0}
    if unique_colors != set(block_counts):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        counts_desc = COUNT_PROFILES[int(np.random.randint(len(COUNT_PROFILES)))]
        colors = np.random.choice(np.array(BLOCK_COLORS, dtype=int), size=len(counts_desc), replace=False)
        permuted_counts = np.random.permutation(np.array(counts_desc, dtype=int))
        block_counts = {int(color): int(count) for color, count in zip(colors.tolist(), permuted_counts.tolist())}

        noise_counts = _sample_noise_counts(block_counts)
        if noise_counts is None:
            continue

        grid = _render_input(block_counts, noise_counts)
        if grid is None:
            continue

        if _validate_generated_grid(grid, block_counts, noise_counts):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
