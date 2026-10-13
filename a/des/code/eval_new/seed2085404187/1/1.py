from __future__ import annotations

import random
from typing import Optional

import numpy as np


GEN_RETRIES = 300
PLACEMENT_RETRIES = 200
COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _tile_to_shape(block: np.ndarray, height: int, width: int) -> np.ndarray:
    bh, bw = block.shape
    reps = ((height + bh - 1) // bh, (width + bw - 1) // bw)
    return np.tile(block, reps)[:height, :width]


def _infer_repeat_unit(sample: np.ndarray, bg: int, seed: int) -> Optional[np.ndarray]:
    matches: list[np.ndarray] = []
    for unit in (2, 3):
        h, w = sample.shape
        if h % unit != 0 or w % unit != 0:
            continue
        block = sample[:unit, :unit]
        colors = {int(v) for v in np.unique(block)}
        if colors != {bg, seed}:
            continue
        tiled = np.tile(block, (h // unit, w // unit))
        if np.array_equal(sample, tiled):
            matches.append(block.copy())
    if len(matches) == 1:
        return matches[0]
    return None


def _same_color_components(grid: np.ndarray, start_row: int, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(start_row, h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if nr < start_row or nr >= h or nc < 0 or nc >= w:
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append((color, coords))
    return components


def _rectangular_fences(
    grid: np.ndarray, start_row: int, bg: int
) -> list[dict[str, object]]:
    fences: list[dict[str, object]] = []
    for color, coords in _same_color_components(grid, start_row, bg):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        expected_area = 2 * (height + width) - 4
        if len(coords) != expected_area:
            continue
        border = set(coords)
        ok = True
        for r, c in coords:
            if r not in (r0, r1) and c not in (c0, c1):
                ok = False
                break
        if not ok:
            continue
        for c in range(c0, c1 + 1):
            if (r0, c) not in border or (r1, c) not in border:
                ok = False
                break
        if not ok:
            continue
        for r in range(r0, r1 + 1):
            if (r, c0) not in border or (r, c1) not in border:
                ok = False
                break
        if not ok:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_positions = np.argwhere(inner != bg)
        if seed_positions.shape[0] != 1:
            continue
        sr, sc = map(int, seed_positions[0])
        sr += r0 + 1
        sc += c0 + 1
        seed_color = int(grid[sr, sc])
        fences.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "seed": (sr, sc),
                "seed_color": seed_color,
            }
        )
    return fences


def _parse_top_legend(
    grid: np.ndarray, bg: int
) -> Optional[tuple[int, dict[int, np.ndarray], list[dict[str, object]]]]:
    h, w = grid.shape
    best: Optional[tuple[int, dict[int, np.ndarray], list[dict[str, object]]]] = None

    for cell_size in range(min(7, h - 1), 2, -1):
        if (w + 1) % (cell_size + 1) != 0:
            continue
        legend_count = (w + 1) // (cell_size + 1)
        if legend_count < 2:
            continue

        sep_cols = [cell_size + i * (cell_size + 1) for i in range(legend_count - 1)]
        sep_color = int(grid[0, sep_cols[0]])
        if sep_color == bg:
            continue
        if not all(np.all(grid[:cell_size, col] == sep_color) for col in sep_cols):
            continue

        cell_mask = np.ones((cell_size, w), dtype=bool)
        cell_mask[:, sep_cols] = False
        if np.any(grid[:cell_size, :][cell_mask] == sep_color):
            continue

        mapping: dict[int, np.ndarray] = {}
        ok = True
        for idx in range(legend_count):
            c0 = idx * (cell_size + 1)
            panel = grid[:cell_size, c0 : c0 + cell_size]
            seed = int(panel[0, 0])
            if seed in (bg, sep_color) or seed in mapping:
                ok = False
                break
            if not np.all(panel[0, 1:] == bg):
                ok = False
                break
            if not np.all(panel[1:, 0] == bg):
                ok = False
                break
            unit = _infer_repeat_unit(panel[1:, 1:], bg, seed)
            if unit is None:
                ok = False
                break
            mapping[seed] = unit
        if not ok:
            continue

        fences = _rectangular_fences(grid, cell_size, bg)
        if not fences:
            continue
        if any(int(fence["seed_color"]) not in mapping for fence in fences):
            continue

        allowed: set[tuple[int, int]] = set()
        for fence in fences:
            allowed.update(fence["border"])  # type: ignore[arg-type]
            allowed.add(fence["seed"])  # type: ignore[arg-type]

        body_ok = True
        for rr, cc in np.argwhere(grid[cell_size:, :] != bg):
            coord = (int(rr) + cell_size, int(cc))
            if coord not in allowed:
                body_ok = False
                break
        if not body_ok:
            continue

        if best is None or len(fences) > len(best[2]) or len(mapping) > len(best[1]):
            best = (cell_size, mapping, fences)

    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced region by tiling the legend pattern indexed by its interior seed color.

    Input structure:
    - The background color is the most common value in the grid.
    - Several single-pixel-thick rectangular fences sit away from the legend edge; each fence encloses one non-background seed pixel inside an otherwise background interior.
    - One grid edge contains a legend strip split into equal cells by separator lines; each cell has a seed-colored pixel in its top-left corner and a repeated 2x2 or 3x3 two-color pattern in the remaining area.

    Rule:
    - Find the legend cell whose top-left color matches a fence's seed color.
    - Extract that cell's repeated unit pattern.
    - Tile the pattern from the fence interior's top-left corner to cover the whole interior, replacing the seed pixel as well.
    - Leave the fences, legend strip, and all background outside the fences unchanged.

    Key insight:
    - The seed pixel is only an index into the edge legend; the real work is decoding the legend's repeated unit and reusing it as a tiled fill inside each enclosed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    for rotation in range(4):
        rotated = np.rot90(g, rotation)
        parsed = _parse_top_legend(rotated, bg)
        if parsed is None:
            continue
        cell_size, mapping, fences = parsed
        out = rotated.copy()
        for fence in fences:
            r0, c0, r1, c1 = fence["bbox"]  # type: ignore[misc]
            seed_color = int(fence["seed_color"])
            pattern = mapping[seed_color]
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            out[r0 + 1 : r1, c0 + 1 : c1] = _tile_to_shape(pattern, inner_h, inner_w)
        return np.rot90(out, -rotation)

    raise AssertionError("Could not locate a valid legend strip and fenced regions")


def _pattern_signature(block: np.ndarray, bg: int) -> tuple[int, ...]:
    return tuple(int(v != bg) for v in block.ravel())


def _random_pattern(
    rng: random.Random, unit: int, repeat: int, bg: int, color: int, used: set[tuple[int, ...]]
) -> np.ndarray:
    for _ in range(200):
        bits = np.array([rng.randrange(2) for _ in range(unit * unit)], dtype=int).reshape(unit, unit)
        on_count = int(bits.sum())
        if on_count < 2 or on_count > unit * unit - 2:
            continue
        block = np.where(bits == 1, color, bg)
        sig = _pattern_signature(block, bg)
        if sig in used:
            continue
        sample = np.tile(block, (repeat, repeat))
        inferred = _infer_repeat_unit(sample, bg, color)
        if inferred is None or inferred.shape != block.shape or not np.array_equal(inferred, block):
            continue
        used.add(sig)
        return block
    raise RuntimeError("Failed to construct an unambiguous legend pattern")


def _build_canonical_grid(rng: random.Random) -> np.ndarray:
    unit = rng.choice((2, 3))
    repeat = rng.choice((1, 2))
    cell_size = 1 + unit * repeat
    max_legend = min(4, (31 // (cell_size + 1)))
    legend_count = rng.randint(2, max_legend)
    width = legend_count * cell_size + (legend_count - 1)

    palette = list(COLORS)
    bg = rng.choice(palette)
    palette.remove(bg)
    sep_color = rng.choice(palette)
    palette.remove(sep_color)
    fence_color = rng.choice(palette)
    palette.remove(fence_color)
    legend_colors = rng.sample(palette, legend_count)

    used_patterns: set[tuple[int, ...]] = set()
    legend_patterns = {
        color: _random_pattern(rng, unit, repeat, bg, color, used_patterns) for color in legend_colors
    }

    max_fences = min(4, legend_count + 1)
    fence_count = rng.randint(legend_count, max_fences)
    gap = rng.randint(1, 2)
    max_body_h = 30 - cell_size - gap
    min_body_h = max(9, 5 * fence_count - 1)
    if min_body_h > max_body_h:
        raise RuntimeError("No room for the fenced regions")
    body_h = rng.randint(min_body_h, max_body_h)
    height = cell_size + gap + body_h

    grid = np.full((height, width), bg, dtype=int)

    for idx, color in enumerate(legend_colors):
        c0 = idx * (cell_size + 1)
        grid[0, c0] = color
        grid[1:cell_size, c0 + 1 : c0 + cell_size] = np.tile(legend_patterns[color], (repeat, repeat))
    for idx in range(legend_count - 1):
        sep_col = cell_size + idx * (cell_size + 1)
        grid[:cell_size, sep_col] = sep_color

    fence_seeds = legend_colors.copy()
    while len(fence_seeds) < fence_count:
        fence_seeds.append(rng.choice(legend_colors))
    rng.shuffle(fence_seeds)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[: cell_size + gap, :] = True
    max_box_w = min(8, width - 2)
    if max_box_w < 4:
        raise RuntimeError("Grid too narrow for fences")

    for seed_color in fence_seeds:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            box_h = rng.randint(4, min(8, body_h))
            box_w = rng.randint(4, max_box_w)
            r0 = rng.randint(cell_size + gap, height - box_h)
            c0 = rng.randint(1, width - box_w - 1)
            window = occupied[max(0, r0 - 1) : min(height, r0 + box_h + 1), max(0, c0 - 1) : min(width, c0 + box_w + 1)]
            if np.any(window):
                continue

            grid[r0, c0 : c0 + box_w] = fence_color
            grid[r0 + box_h - 1, c0 : c0 + box_w] = fence_color
            grid[r0 : r0 + box_h, c0] = fence_color
            grid[r0 : r0 + box_h, c0 + box_w - 1] = fence_color

            sr = rng.randint(r0 + 1, r0 + box_h - 2)
            sc = rng.randint(c0 + 1, c0 + box_w - 2)
            grid[sr, sc] = seed_color

            occupied[max(0, r0 - 1) : min(height, r0 + box_h + 1), max(0, c0 - 1) : min(width, c0 + box_w + 1)] = True
            placed = True
            break
        if not placed:
            raise RuntimeError("Failed to place all fenced rectangles")

    return grid


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    try:
        grid = _build_canonical_grid(rng)
    except RuntimeError:
        return None

    rotation = rng.randrange(4)
    grid = np.rot90(grid, rotation)

    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
