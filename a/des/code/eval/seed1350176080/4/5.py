from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

INSTRUCTION_TO_TRANSFORM = {
    RED: "hflip",
    BLUE: "vflip",
    GREEN: "cw",
    YELLOW: "ccw",
}

LAYOUTS_BY_COUNT = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2), (1, 4), (4, 1)],
}
CELL_SIZE_RANGE = (7, 10)
POINT_COUNT_RANGE = (3, 8)
SEARCH_ATTEMPTS = 1200
GENERATION_ATTEMPTS = 400
CORNERS = ("tl", "tr", "bl", "br")


def _separator_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), int(length)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = int(left) + 1
        end = int(right)
        assert start < end, "separator lines must leave non-empty regions"
        intervals.append((start, end))
    return intervals


def _corner_slices(size: int, corner: str) -> tuple[slice, slice]:
    assert size >= 2
    if corner == "tl":
        return slice(0, 2), slice(0, 2)
    if corner == "tr":
        return slice(0, 2), slice(size - 2, size)
    if corner == "bl":
        return slice(size - 2, size), slice(0, 2)
    if corner == "br":
        return slice(size - 2, size), slice(size - 2, size)
    raise AssertionError("unknown corner")


def _instruction_block_mask(height: int, width: int, corner: str) -> np.ndarray:
    assert height >= 2 and width >= 2
    mask = np.zeros((height, width), dtype=bool)
    rs, cs = _corner_slices(height, corner)
    mask[rs, cs] = True
    return mask


def _detect_instruction(patch: np.ndarray) -> tuple[int, np.ndarray]:
    height, width = patch.shape
    assert min(height, width) >= 2
    found: list[tuple[int, np.ndarray]] = []
    for corner in CORNERS:
        rs, cs = _corner_slices(height, corner)
        block = patch[rs, cs]
        if block.shape != (2, 2):
            continue
        color = int(block[0, 0])
        if color not in INSTRUCTION_TO_TRANSFORM:
            continue
        if np.all(block == color):
            found.append((color, _instruction_block_mask(height, width, corner)))
    assert len(found) == 1, "each region must contain exactly one 2x2 instruction block in a corner"
    return found[0]


def _apply_transform(
    coords: set[tuple[int, int]],
    height: int,
    width: int,
    transform: str,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    if transform == "hflip":
        for r, c in coords:
            out.add((height - 1 - int(r), int(c)))
        return out
    if transform == "vflip":
        for r, c in coords:
            out.add((int(r), width - 1 - int(c)))
        return out

    assert height == width, "90-degree rotation requires a square region"
    size = int(height)
    if transform == "cw":
        for r, c in coords:
            out.add((int(c), size - 1 - int(r)))
        return out
    if transform == "ccw":
        for r, c in coords:
            out.add((size - 1 - int(c), int(r)))
        return out
    raise AssertionError("unknown transform")


def _mask_from_coords(height: int, width: int, coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        assert 0 <= int(r) < height and 0 <= int(c) < width
        mask[int(r), int(c)] = True
    return mask


def _classify_symmetry(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal and vertical:
        return "hv"
    if horizontal:
        return "h"
    if vertical:
        return "v"
    return "none"


def _recolor_for_symmetry(kind: str) -> int:
    if kind == "hv":
        return MAGENTA
    if kind == "h":
        return RED
    if kind == "v":
        return BLUE
    return ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each grey-separated region contains a corner control block and scattered orange pixels.
    The control color selects a region-centered spatial transform for those orange pixels;
    the original and transformed pixels are merged, then recolored by whether the merged
    region pattern is horizontally symmetric, vertically symmetric, or both.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    separator_rows = [r for r in range(height) if bool(np.all(g[r, :] == GREY))]
    separator_cols = [c for c in range(width) if bool(np.all(g[:, c] == GREY))]
    row_intervals = _separator_intervals(separator_rows, height)
    col_intervals = _separator_intervals(separator_cols, width)
    assert 2 <= len(row_intervals) * len(col_intervals) <= 4

    out = np.full_like(g, BLACK)
    for r in separator_rows:
        out[int(r), :] = GREY
    for c in separator_cols:
        out[:, int(c)] = GREY

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            patch = g[r0:r1, c0:c1]
            assert not bool(np.any(patch == GREY)), "grey may only appear in separator lines"

            instruction_color, block_mask = _detect_instruction(patch)
            transform = INSTRUCTION_TO_TRANSFORM[instruction_color]

            orange_coords = {
                (int(r), int(c))
                for r, c in np.argwhere(patch == ORANGE)
            }
            assert orange_coords, "each region must contain at least one orange pixel"
            assert not any(block_mask[r, c] for r, c in orange_coords), "orange pixels may not overlap the instruction block"

            transformed = _apply_transform(orange_coords, patch.shape[0], patch.shape[1], transform)
            merged = orange_coords | transformed
            merged_mask = _mask_from_coords(patch.shape[0], patch.shape[1], merged)
            symmetry_kind = _classify_symmetry(merged_mask)
            recolor = _recolor_for_symmetry(symmetry_kind)

            visible_mask = merged_mask & (~block_mask)
            out[r0:r1, c0:c1][visible_mask] = recolor

    return out


def _region_origins(rows: int, cols: int, cell_size: int) -> list[tuple[int, int]]:
    return [
        (rr * (cell_size + 1), cc * (cell_size + 1))
        for rr in range(rows)
        for cc in range(cols)
    ]


def _empty_partition(rows: int, cols: int, cell_size: int) -> np.ndarray:
    height = rows * cell_size + (rows - 1)
    width = cols * cell_size + (cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)
    for rr in range(1, rows):
        grid[rr * cell_size + (rr - 1), :] = GREY
    for cc in range(1, cols):
        grid[:, cc * cell_size + (cc - 1)] = GREY
    return grid


def _sparse_sample(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for r, c in shuffled:
        if (r, c) in blocked:
            continue
        chosen.append((int(r), int(c)))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((int(r) + dr, int(c) + dc))
        if len(chosen) == count:
            return chosen
    return None


def _feasible_targets(instruction_color: int) -> tuple[str, ...]:
    transform = INSTRUCTION_TO_TRANSFORM[instruction_color]
    if transform == "hflip":
        return ("h", "hv")
    if transform == "vflip":
        return ("v", "hv")
    return ("h", "v", "hv")


def _find_points_for_region(
    cell_size: int,
    instruction_color: int,
    corner: str,
    target_kind: str,
) -> list[tuple[int, int]] | None:
    block_mask = _instruction_block_mask(cell_size, cell_size, corner)
    transform = INSTRUCTION_TO_TRANSFORM[instruction_color]

    candidates: list[tuple[int, int]] = []
    for r in range(cell_size):
        for c in range(cell_size):
            if block_mask[r, c]:
                continue
            mapped = _apply_transform({(r, c)}, cell_size, cell_size, transform)
            ((tr, tc),) = tuple(mapped)
            if block_mask[tr, tc]:
                continue
            candidates.append((r, c))

    if len(candidates) < POINT_COUNT_RANGE[0]:
        return None

    for _ in range(SEARCH_ATTEMPTS):
        count = random.randint(POINT_COUNT_RANGE[0], POINT_COUNT_RANGE[1])
        points = _sparse_sample(candidates, count)
        if points is None:
            continue
        original = {(int(r), int(c)) for r, c in points}
        merged = original | _apply_transform(original, cell_size, cell_size, transform)
        mask = _mask_from_coords(cell_size, cell_size, merged)
        if _classify_symmetry(mask) != target_kind:
            continue
        if len(merged) < max(4, count):
            continue
        if mask.sum() > (cell_size * cell_size) // 2:
            continue
        return points
    return None


def _choose_instruction_colors(region_count: int) -> list[int]:
    flips = [RED, BLUE]
    rotations = [GREEN, YELLOW]
    if region_count == 2:
        colors = [random.choice(flips), random.choice(rotations)]
    elif region_count == 3:
        picked_flip = random.choice(flips)
        picked_rotation = random.choice(rotations)
        remaining = [color for color in (RED, BLUE, GREEN, YELLOW) if color not in {picked_flip, picked_rotation}]
        colors = [picked_flip, picked_rotation, random.choice(remaining)]
    else:
        colors = [RED, BLUE, GREEN, YELLOW]
    random.shuffle(colors)
    return colors


def _choose_layout() -> tuple[int, int]:
    region_count = random.randint(2, 4)
    return random.choice(LAYOUTS_BY_COUNT[region_count])


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rows, cols = _choose_layout()
        cell_size = random.randint(CELL_SIZE_RANGE[0], CELL_SIZE_RANGE[1])
        region_count = rows * cols
        grid = _empty_partition(rows, cols, cell_size)
        origins = _region_origins(rows, cols, cell_size)
        instruction_colors = _choose_instruction_colors(region_count)

        region_specs: list[dict] = []
        target_kinds: list[str] = []
        failed = False
        for idx in range(region_count):
            instruction_color = instruction_colors[idx]
            corner = random.choice(CORNERS)
            target_kind = random.choice(_feasible_targets(instruction_color))
            points = _find_points_for_region(cell_size, instruction_color, corner, target_kind)
            if points is None:
                failed = True
                break
            region_specs.append(
                {
                    "origin": origins[idx],
                    "instruction_color": instruction_color,
                    "corner": corner,
                    "points": points,
                }
            )
            target_kinds.append(target_kind)
        if failed:
            continue

        if region_count > 1 and len(set(target_kinds)) == 1:
            continue

        for spec in region_specs:
            r0, c0 = spec["origin"]
            patch = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
            rs, cs = _corner_slices(cell_size, spec["corner"])
            patch[rs, cs] = int(spec["instruction_color"])
            for r, c in spec["points"]:
                patch[int(r), int(c)] = ORANGE

        try:
            out = main(grid)
        except Exception:
            continue

        if not isinstance(out, np.ndarray):
            continue
        if not bool(np.any(out == MAGENTA) or np.any(out == RED) or np.any(out == BLUE)):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
