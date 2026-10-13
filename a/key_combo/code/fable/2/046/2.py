from __future__ import annotations

import hashlib
import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, get_bg, paste


BG = 0
GEN_RETRIES = 400
COLOR_POOL = tuple(range(1, 10))

_MASK_CACHE: dict[int, list[np.ndarray]] = {}
_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()
_GEN_COUNTER = 0


def _grid_hash(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode(), usedforsecurity=False).hexdigest()


def _separator_candidates(grid: np.ndarray) -> list[tuple[str, int, int]]:
    bg = int(get_bg(grid))
    row_hits = [
        ("row", int(r), int(grid[r, 0]))
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_hits = [
        ("col", int(c), int(grid[0, c]))
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    return row_hits + col_hits


def _cell_bounds(cell_size: int, cell_r: int, cell_c: int) -> tuple[int, int, int, int]:
    r0 = 1 + int(cell_r) * (int(cell_size) + 1)
    c0 = 1 + int(cell_c) * (int(cell_size) + 1)
    return r0, c0, r0 + int(cell_size), c0 + int(cell_size)


def _flip_to_target(mask: np.ndarray, src_r: int, src_c: int, dst_r: int, dst_c: int) -> np.ndarray:
    out = mask
    if int(src_r) != int(dst_r):
        out = np.flipud(out)
    if int(src_c) != int(dst_c):
        out = np.fliplr(out)
    return out


def _try_parse_lattice(panel: np.ndarray, divider: int) -> dict | None:
    if panel.ndim != 2 or panel.shape[0] != panel.shape[1]:
        return None
    size = int(panel.shape[0])
    if size < 7 or (size - 3) % 2 != 0:
        return None

    cell_size = (size - 3) // 2
    cross = cell_size + 1
    lines = (0, cross, size - 1)
    if any(not np.all(panel[r, :] == int(divider)) for r in lines):
        return None
    if any(not np.all(panel[:, c] == int(divider)) for c in lines):
        return None

    line_mask = np.zeros_like(panel, dtype=bool)
    line_mask[list(lines), :] = True
    line_mask[:, list(lines)] = True
    if np.any((panel == int(divider)) & ~line_mask):
        return None

    cells = []
    sprite_colors: set[int] = set()
    missing_count = 0
    for cell_r in range(2):
        for cell_c in range(2):
            r0, c0, r1, c1 = _cell_bounds(cell_size, cell_r, cell_c)
            cell = panel[r0:r1, c0:c1]
            vals = set(map(int, np.unique(cell))) - {BG, int(divider)}
            if len(vals) > 1:
                return None
            if vals:
                color = vals.pop()
                sprite_colors.add(int(color))
                if np.any((cell != BG) & (cell != int(color))):
                    return None
                mask = cell == int(color)
            else:
                color = None
                mask = np.zeros((cell_size, cell_size), dtype=bool)
                missing_count += 1
            cells.append(
                {
                    "row": int(cell_r),
                    "col": int(cell_c),
                    "color": color,
                    "mask": mask.copy(),
                    "cell": cell.copy(),
                }
            )

    if len(sprite_colors) != 1 or missing_count != 1:
        return None
    sprite_color = next(iter(sprite_colors))
    missing = next(cell for cell in cells if not cell["mask"].any())

    predictions = []
    present_masks = []
    for cell in cells:
        if cell is missing:
            continue
        present_masks.append(cell["mask"].copy())
        predictions.append(_flip_to_target(cell["mask"], cell["row"], cell["col"], missing["row"], missing["col"]))
    if not predictions:
        return None
    if not all(np.array_equal(predictions[0], pred) for pred in predictions[1:]):
        return None

    return {
        "cell_size": int(cell_size),
        "sprite_color": int(sprite_color),
        "missing_row": int(missing["row"]),
        "missing_col": int(missing["col"]),
        "missing_mask": predictions[0].copy(),
        "present_masks": present_masks,
        "cells": cells,
        "panel": panel.copy(),
    }


def _extract_problem(grid: np.ndarray) -> tuple[dict, np.ndarray]:
    matches: list[tuple[dict, np.ndarray]] = []
    for mode, idx, divider in _separator_candidates(grid):
        if mode == "row":
            parts = [grid[:idx, :], grid[idx + 1 :, :]]
        else:
            parts = [grid[:, :idx], grid[:, idx + 1 :]]
        if any(part.size == 0 for part in parts):
            continue

        parsed = [_try_parse_lattice(part, divider) for part in parts]
        good = [i for i, item in enumerate(parsed) if item is not None]
        if len(good) != 1:
            continue

        lattice = parsed[good[0]]
        assert lattice is not None
        source = np.asarray(parts[1 - good[0]], dtype=int)
        if source.ndim != 2 or source.shape[0] != source.shape[1] or source.shape[0] == 0:
            continue
        if np.any(source == int(divider)):
            continue
        matches.append((lattice, source))

    assert len(matches) == 1, "expected a unique separator line that isolates one lattice side and one source panel"
    return matches[0]


def _stamp_from_source(mask: np.ndarray, source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=int)
    block = int(mask.shape[0])
    side = int(source.shape[0])
    out = np.zeros((side * block, side * block), dtype=int)
    for r, c in np.argwhere(source != BG):
        color = int(source[r, c])
        patch = np.where(mask, color, BG)
        paste(out, patch, int(r) * block, int(c) * block, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing mirrored sprite in a 2x2 lattice and use it as a recolored block stamp.

    Input structure:
    - A single full-length non-black separator line splits the input into two sides.
    - One side is a square 2x2 lattice panel whose outer border and inner cross use the separator color.
    - Three lattice cells contain one monochrome sprite's mirror variants on black; the fourth cell is empty.
    - The other side is a square black source panel containing scattered non-black pixels.

    Rule:
    - Identify the lattice side and the source-panel side.
    - Reflect each visible cell-local sprite vertically and/or horizontally according to its position relative to the empty cell; the three reflected predictions must agree, giving the missing sprite mask.
    - Build a black output whose blocks have the lattice cell size.
    - For every non-black source pixel at (r, c), stamp the inferred missing mask recolored to that pixel's color into output block (r, c); black source cells leave black blocks.

    Key insight:
    - The template to stamp never appears directly: it is the absent member of a four-way mirror quartet and must be inferred before the source-panel recoloring and tiling step can happen.
    """
    lattice, source = _extract_problem(np.asarray(grid, dtype=int))
    return _stamp_from_source(lattice["missing_mask"], source)


def _mask_library(cell_size: int) -> list[np.ndarray]:
    cell_size = int(cell_size)
    if cell_size in _MASK_CACHE:
        return [mask.copy() for mask in _MASK_CACHE[cell_size]]

    masks: list[np.ndarray] = []
    area = cell_size * cell_size
    for bits in range(1, 1 << area):
        if bits == (1 << area) - 1:
            continue
        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for idx in range(area):
            if bits & (1 << idx):
                r, c = divmod(idx, cell_size)
                mask[r, c] = True
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        variants = [
            mask,
            np.flipud(mask),
            np.fliplr(mask),
            np.flipud(np.fliplr(mask)),
        ]
        if len({variant.tobytes() for variant in variants}) != 4:
            continue
        masks.append(mask.copy())

    assert masks, f"no valid asymmetric sprite masks for cell size {cell_size}"
    _MASK_CACHE[cell_size] = [mask.copy() for mask in masks]
    return [mask.copy() for mask in masks]


def _variant_for_cell(base_mask: np.ndarray, cell_r: int, cell_c: int) -> np.ndarray:
    return _flip_to_target(base_mask, 0, 0, cell_r, cell_c)


def _render_lattice(cell_size: int, divider: int, sprite_color: int, base_mask: np.ndarray, missing_idx: int) -> np.ndarray:
    panel_size = 2 * int(cell_size) + 3
    panel = np.zeros((panel_size, panel_size), dtype=int)
    cross = int(cell_size) + 1
    for line in (0, cross, panel_size - 1):
        panel[line, :] = int(divider)
        panel[:, line] = int(divider)

    for idx, (cell_r, cell_c) in enumerate(itertools.product(range(2), range(2))):
        if idx == int(missing_idx):
            continue
        r0, c0, r1, c1 = _cell_bounds(cell_size, cell_r, cell_c)
        mask = _variant_for_cell(base_mask, cell_r, cell_c)
        panel[r0:r1, c0:c1] = np.where(mask, int(sprite_color), BG)
    return panel


def _forced_coords(side: int, token: int) -> list[tuple[int, int]]:
    coords = []
    seeds = ((1, 2), (3, 1), (2, 4), (4, 3))
    for mul, add in seeds:
        coords.append(((token * mul + add) % side, (token * (mul + 1) + add + 1) % side))

    chosen: list[tuple[int, int]] = []
    for cand in coords:
        if cand not in chosen:
            chosen.append(cand)
        if len(chosen) == 2:
            break

    if len(chosen) < 2:
        for cand in itertools.product(range(side), repeat=2):
            if cand not in chosen:
                chosen.append(cand)
            if len(chosen) == 2:
                break

    if chosen[0][0] == chosen[1][0]:
        chosen[1] = ((chosen[1][0] + 1) % side, chosen[1][1])
    if chosen[0][1] == chosen[1][1]:
        chosen[1] = (chosen[1][0], (chosen[1][1] + 1) % side)
    if chosen[0] == chosen[1]:
        chosen[1] = ((chosen[1][0] + 1) % side, (chosen[1][1] + 2) % side)
    return chosen


def _render_source(side: int, palette: list[int], token: int) -> np.ndarray:
    source = np.zeros((side, side), dtype=int)
    forced = _forced_coords(side, token)
    source[forced[0]] = int(palette[0])
    source[forced[1]] = int(palette[1])

    extra_count = random.randint(1, min(4, side))
    remaining = [(r, c) for r, c in itertools.product(range(side), repeat=2) if source[r, c] == BG]
    random.shuffle(remaining)
    for r, c in remaining[:extra_count]:
        source[r, c] = int(random.choice(palette))

    coords = np.argwhere(source != BG)
    colors = {int(source[r, c]) for r, c in coords}
    if len(coords) < 3 or len(colors) < 2:
        raise AssertionError("source panel must exercise both recolor and block-placement steps")
    if len({int(r) for r, _ in coords}) < 2 or len({int(c) for _, c in coords}) < 2:
        raise AssertionError("source pixels must occupy multiple rows and columns")
    return source


def _compose_input(lattice: np.ndarray, source: np.ndarray, divider: int, orientation: str, lattice_first: bool) -> np.ndarray:
    if orientation == "vertical":
        sep = np.full((lattice.shape[0], 1), int(divider), dtype=int)
        left, right = (lattice, source) if lattice_first else (source, lattice)
        return np.concatenate([left, sep, right], axis=1)

    assert orientation == "horizontal"
    sep = np.full((1, lattice.shape[1]), int(divider), dtype=int)
    top, bottom = (lattice, source) if lattice_first else (source, lattice)
    return np.concatenate([top, sep, bottom], axis=0)


def _wrongly_permuted_source(source: np.ndarray) -> np.ndarray:
    coords = np.argwhere(source != BG)
    values = [int(source[r, c]) for r, c in coords]
    permuted = np.zeros_like(source)
    shifted_coords = np.roll(coords, shift=1, axis=0)
    for (r, c), value in zip(shifted_coords, values):
        permuted[int(r), int(c)] = int(value)
    return permuted


def _validate_semantics(grid: np.ndarray) -> None:
    lattice, source = _extract_problem(grid)
    missing_mask = lattice["missing_mask"]
    present_masks = lattice["present_masks"]
    correct = _stamp_from_source(missing_mask, source)

    assert len({missing_mask.tobytes(), *(mask.tobytes() for mask in present_masks)}) == 4, (
        "the inferred template must differ from every visible mirror variant"
    )
    assert int(correct.shape[0]) <= 30 and int(correct.shape[1]) <= 30

    square_wrong = _stamp_from_source(np.ones_like(missing_mask, dtype=bool), source)
    assert not np.array_equal(square_wrong, correct), "sprite geometry must matter"

    for wrong_mask in present_masks:
        wrong = _stamp_from_source(wrong_mask, source)
        assert not np.array_equal(wrong, correct), "using any visible variant directly must fail"

    flat_source = np.zeros_like(source)
    first_color = int(source[source != BG][0])
    flat_source[source != BG] = first_color
    assert not np.array_equal(_stamp_from_source(missing_mask, flat_source), correct), "recoloring must matter"

    permuted_source = _wrongly_permuted_source(source)
    assert not np.array_equal(_stamp_from_source(missing_mask, permuted_source), correct), (
        "source-panel block coordinates must matter"
    )


def _generate_candidate(branch: int) -> tuple[np.ndarray, np.ndarray]:
    cell_size = 2 + (int(branch // 16) % 2)
    orientation = "vertical" if int(branch) % 2 == 0 else "horizontal"
    lattice_first = ((int(branch) // 2) % 2) == 0
    missing_idx = (int(branch) // 4) % 4

    panel_side = 2 * cell_size + 3
    divider, sprite_color, *palette = random.sample(COLOR_POOL, k=5)
    base_mask = random.choice(_mask_library(cell_size))
    lattice = _render_lattice(cell_size, divider, sprite_color, base_mask, missing_idx)
    source = _render_source(panel_side, palette[: random.randint(2, 3)], branch + missing_idx * 11)
    grid = _compose_input(lattice, source, divider, orientation, lattice_first)
    return grid, _stamp_from_source(_variant_for_cell(base_mask, missing_idx // 2, missing_idx % 2), source)


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(GEN_RETRIES):
        branch = _GEN_COUNTER
        _GEN_COUNTER += 1
        try:
            grid, expected = _generate_candidate(branch)
            solved = main(grid)
            if not np.array_equal(solved, expected):
                continue
            _validate_semantics(grid)
        except AssertionError:
            continue

        input_hash = _grid_hash(grid)
        output_hash = _grid_hash(expected)
        if input_hash in _SEEN_INPUTS or output_hash in _SEEN_OUTPUTS:
            continue
        _SEEN_INPUTS.add(input_hash)
        _SEEN_OUTPUTS.add(output_hash)
        return grid

    raise RuntimeError("failed to generate a valid input for the absent-fairy stamp puzzle")
