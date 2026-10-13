from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox, colors_present, crop_bbox
except Exception:
    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.unique(grid)))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


MACRO_N = 4
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _most_common(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _mul_alpha(x: int) -> int:
    table = (0, 2, 3, 1)
    return table[int(x)]


def _base_dual_latin_squares() -> tuple[np.ndarray, np.ndarray]:
    rot = np.zeros((MACRO_N, MACRO_N), dtype=int)
    color = np.zeros((MACRO_N, MACRO_N), dtype=int)
    for r in range(MACRO_N):
        for c in range(MACRO_N):
            rot[r, c] = r ^ c
            color[r, c] = r ^ _mul_alpha(c)
    return rot, color


def _permute_square(square: np.ndarray, row_perm: list[int], col_perm: list[int], sym_perm: list[int]) -> np.ndarray:
    permuted = square[np.ix_(row_perm, col_perm)]
    mapping = np.asarray(sym_perm, dtype=int)
    return mapping[permuted]


def _visible_coords(blank_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(MACRO_N) for c in range(MACRO_N) if (r, c) not in blank_cells]


def _has_functional_mapping(rot_square: np.ndarray, color_square: np.ndarray, coords: list[tuple[int, int]], *, source: str) -> bool:
    mapping: dict[int, int] = {}
    for r, c in coords:
        a = int(rot_square[r, c]) if source == "rotation" else int(color_square[r, c])
        b = int(color_square[r, c]) if source == "rotation" else int(rot_square[r, c])
        old = mapping.get(a)
        if old is not None and old != b:
            return False
        mapping[a] = b
    return len(mapping) == MACRO_N


def _choose_blank_cells(rng: random.Random, rot_square: np.ndarray, color_square: np.ndarray) -> list[tuple[int, int]] | None:
    perms = list(itertools.permutations(range(MACRO_N)))
    rng.shuffle(perms)
    for perm in perms:
        blanks = [(r, perm[r]) for r in range(MACRO_N)]
        blank_set = set(blanks)
        visible = _visible_coords(blank_set)
        rot_counts = np.bincount([int(rot_square[r, c]) for r, c in visible], minlength=MACRO_N)
        color_counts = np.bincount([int(color_square[r, c]) for r, c in visible], minlength=MACRO_N)
        if np.any(rot_counts == 0) or np.any(color_counts == 0):
            continue
        if _has_functional_mapping(rot_square, color_square, visible, source="rotation"):
            continue
        if _has_functional_mapping(rot_square, color_square, visible, source="color"):
            continue
        return blanks
    return None


def _neighbors4(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _random_sprite_mask(cell_size: int, rng: random.Random) -> np.ndarray | None:
    inner = cell_size - 2
    min_pixels = max(5, inner + 1)
    max_pixels = min(inner * inner - 1, inner * inner // 2 + 3)
    if min_pixels > max_pixels:
        return None

    start = (rng.randrange(inner), rng.randrange(inner))
    cells = {start}
    target = rng.randint(min_pixels, max_pixels)
    while len(cells) < target:
        frontier = sorted({nb for cell in cells for nb in _neighbors4(cell, inner) if nb not in cells})
        if not frontier:
            return None
        cells.add(rng.choice(frontier))

    inner_mask = np.zeros((inner, inner), dtype=bool)
    for r, c in cells:
        inner_mask[r, c] = True

    mask = np.zeros((cell_size, cell_size), dtype=bool)
    mask[1:-1, 1:-1] = inner_mask

    box = bbox(mask)
    if box is None:
        return None
    tight = crop_bbox(mask.astype(int), box)
    if tight.shape[0] < 2 or tight.shape[1] < 2:
        return None

    rotations = [np.rot90(mask, k) for k in range(MACRO_N)]
    for i in range(MACRO_N):
        for j in range(i + 1, MACRO_N):
            if np.array_equal(rotations[i], rotations[j]):
                return None
    return mask


def _paint_block(mask: np.ndarray, rotation: int, color: int, bg: int) -> np.ndarray:
    block = np.full(mask.shape, bg, dtype=int)
    block[np.rot90(mask, int(rotation))] = int(color)
    return block


def _build_case(
    mask: np.ndarray,
    rot_square: np.ndarray,
    color_square: np.ndarray,
    blank_cells: list[tuple[int, int]],
    bg: int,
    line_color: int,
    sprite_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cell_size = int(mask.shape[0])
    side = MACRO_N * cell_size + MACRO_N + 1
    inp = np.full((side, side), line_color, dtype=int)
    out = np.full((side, side), line_color, dtype=int)
    blank_set = set(blank_cells)

    for br in range(MACRO_N):
        for bc in range(MACRO_N):
            r0 = 1 + br * (cell_size + 1)
            c0 = 1 + bc * (cell_size + 1)
            inp[r0 : r0 + cell_size, c0 : c0 + cell_size] = bg
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = bg

            rotation = int(rot_square[br, bc])
            color = int(sprite_colors[int(color_square[br, bc])])
            block = _paint_block(mask, rotation, color, bg)
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = block
            if (br, bc) not in blank_set:
                inp[r0 : r0 + cell_size, c0 : c0 + cell_size] = block

    return inp, out


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = rng or random.Random()
    rot_base, color_base = _base_dual_latin_squares()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        cell_size = rng.choice((5, 6))
        mask = _random_sprite_mask(cell_size, rng)
        if mask is None:
            continue

        bg, line_color = rng.sample(PALETTE, 2)
        sprite_colors = rng.sample([c for c in PALETTE if c not in {bg, line_color}], MACRO_N)

        row_perm = rng.sample(range(MACRO_N), MACRO_N)
        col_perm = rng.sample(range(MACRO_N), MACRO_N)
        rot_labels = rng.sample(range(MACRO_N), MACRO_N)
        color_labels = rng.sample(range(MACRO_N), MACRO_N)

        rot_square = _permute_square(rot_base, row_perm, col_perm, rot_labels)
        color_square = _permute_square(color_base, row_perm, col_perm, color_labels)
        if rng.random() < 0.5:
            rot_square = rot_square.T.copy()
            color_square = color_square.T.copy()

        if len({(int(rot_square[r, c]), int(color_square[r, c])) for r in range(MACRO_N) for c in range(MACRO_N)}) != MACRO_N * MACRO_N:
            continue

        blank_cells = _choose_blank_cells(rng, rot_square, color_square)
        if blank_cells is None:
            continue

        inp, out = _build_case(mask, rot_square, color_square, blank_cells, bg, line_color, sprite_colors)
        meta = {
            "mask": mask,
            "rot_square": rot_square,
            "color_square": color_square,
            "blank_cells": blank_cells,
            "bg": int(bg),
            "line_color": int(line_color),
            "sprite_colors": list(map(int, sprite_colors)),
            "cell_size": int(cell_size),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid dual-latin macro-grid case")


def _find_macro_structure(grid: np.ndarray) -> tuple[int, list[int], list[int], int, int]:
    h, w = grid.shape
    for color in colors_present(grid):
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(row_lines) != MACRO_N + 1 or len(col_lines) != MACRO_N + 1:
            continue
        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) != int(col_steps[0]) or int(row_steps[0]) < 2:
            continue
        cell_size = int(row_steps[0] - 1)
        if row_lines[0] != 0 or col_lines[0] != 0 or row_lines[-1] != h - 1 or col_lines[-1] != w - 1:
            continue
        line_mask = np.zeros_like(grid, dtype=bool)
        line_mask[row_lines, :] = True
        line_mask[:, col_lines] = True
        bg = _most_common(grid[~line_mask])
        return int(color), row_lines, col_lines, cell_size, int(bg)
    raise AssertionError("failed to infer the 4x4 macro-grid separators")


def _block_box(row_lines: list[int], col_lines: list[int], br: int, bc: int) -> tuple[int, int, int, int]:
    return row_lines[br] + 1, col_lines[bc] + 1, row_lines[br + 1] - 1, col_lines[bc + 1] - 1


def _extract_blocks(grid: np.ndarray) -> tuple[dict[tuple[int, int], np.ndarray | None], dict[tuple[int, int], int], dict]:
    line_color, row_lines, col_lines, cell_size, bg = _find_macro_structure(grid)
    masks: dict[tuple[int, int], np.ndarray | None] = {}
    colors: dict[tuple[int, int], int] = {}

    for br in range(MACRO_N):
        for bc in range(MACRO_N):
            block = crop_bbox(grid, _block_box(row_lines, col_lines, br, bc))
            assert block.shape == (cell_size, cell_size)
            if np.all(block == bg):
                masks[(br, bc)] = None
                continue
            block_colors = colors_present(block, exclude={bg})
            assert len(block_colors) == 1, "each sprite block must contain exactly one foreground color"
            color = int(block_colors[0])
            mask = (block == color)
            assert bbox(mask) is not None
            masks[(br, bc)] = mask
            colors[(br, bc)] = color

    structure = {
        "line_color": line_color,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "cell_size": cell_size,
        "bg": bg,
    }
    return masks, colors, structure


def _classify_rotations(masks: dict[tuple[int, int], np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    filled = [(pos, mask) for pos, mask in masks.items() if mask is not None]
    assert filled, "expected at least one visible sprite block"
    reference = filled[0][1]
    assert reference is not None

    rotation_grid = np.full((MACRO_N, MACRO_N), -1, dtype=int)
    empty_cells: list[tuple[int, int]] = []
    for pos, mask in masks.items():
        if mask is None:
            empty_cells.append(pos)
            continue
        matches = [k for k in range(MACRO_N) if np.array_equal(mask, np.rot90(reference, k))]
        assert len(matches) == 1, "all sprites must be rotations of one shared base mask"
        rotation_grid[pos] = int(matches[0])
    return rotation_grid, reference, empty_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the blank macro-blocks so both sprite orientation and color each appear exactly once per row and column.

    Input structure:
    - The grid is partitioned by separator lines into a 4x4 array of equal square macro-blocks.
    - Every non-empty macro-block contains the same non-rotationally-symmetric sprite in one of four quarter-turn rotations and in one of four colors.
    - Some macro-blocks are pure background blanks; among the given blocks, each row and column already has no repeated rotation and no repeated color.

    Rule:
    - Treat the four rotations as one Latin-square symbol set and the four colors as a second, independent symbol set.
    - For every blank macro-block, find the unique rotation missing from both its row and its column, and separately the unique color missing from both its row and its column.
    - Draw that rotated sprite in that color, leaving all existing blocks and separator lines unchanged.

    Key insight:
    - The picture overlays two independent row-column completion problems on the same cells: the shape solves orientation, and the pigment solves color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    masks, block_colors, structure = _extract_blocks(g)
    rotation_grid, reference_mask, empty_cells = _classify_rotations(masks)
    bg = int(structure["bg"])
    row_lines = structure["row_lines"]
    col_lines = structure["col_lines"]

    color_grid = np.full((MACRO_N, MACRO_N), -1, dtype=int)
    for (br, bc), color in block_colors.items():
        color_grid[br, bc] = int(color)

    out = g.copy()
    all_rotations = set(range(MACRO_N))
    all_colors = set(map(int, np.unique([color for color in block_colors.values()])))
    assert len(all_colors) == MACRO_N, "expected four visible color classes across the macro-grid"

    for br, bc in empty_cells:
        row_rot_missing = all_rotations - set(map(int, rotation_grid[br, :][rotation_grid[br, :] >= 0]))
        col_rot_missing = all_rotations - set(map(int, rotation_grid[:, bc][rotation_grid[:, bc] >= 0]))
        row_color_missing = all_colors - set(map(int, color_grid[br, :][color_grid[br, :] >= 0]))
        col_color_missing = all_colors - set(map(int, color_grid[:, bc][color_grid[:, bc] >= 0]))

        rot_choice = row_rot_missing & col_rot_missing
        color_choice = row_color_missing & col_color_missing
        assert len(rot_choice) == 1 and len(color_choice) == 1, "each blank must have a unique rotation and color"

        rotation = rot_choice.pop()
        color = color_choice.pop()
        block = _paint_block(reference_mask, rotation, color, bg)
        r0, c0, r1, c1 = _block_box(row_lines, col_lines, br, bc)
        out[r0 : r1 + 1, c0 : c1 + 1] = block

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, out, meta = _generate_case()
        if not np.array_equal(main(inp), out):
            continue
        try:
            _validate_case(inp, out, meta)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input for the dual-latin macro-grid puzzle")


def _validate_case(inp: np.ndarray, out: np.ndarray, meta: dict) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.array_equal(main(inp), out)

    masks, colors, structure = _extract_blocks(inp)
    rotation_grid, ref_mask, empty_cells = _classify_rotations(masks)
    assert len(empty_cells) == MACRO_N
    assert sorted(r for r, _ in empty_cells) == list(range(MACRO_N))
    assert sorted(c for _, c in empty_cells) == list(range(MACRO_N))

    expected_rot = np.asarray(meta["rot_square"], dtype=int)
    expected_color_labels = np.asarray(meta["color_square"], dtype=int)
    sprite_colors = list(map(int, meta["sprite_colors"]))
    expected_color = np.vectorize(lambda idx: sprite_colors[int(idx)])(expected_color_labels)

    visible = _visible_coords(set(empty_cells))
    assert len({(int(expected_rot[r, c]), int(expected_color[r, c])) for r, c in visible}) > MACRO_N
    assert not _has_functional_mapping(expected_rot, expected_color_labels, visible, source="rotation")
    assert not _has_functional_mapping(expected_rot, expected_color_labels, visible, source="color")

    color_grid = np.full((MACRO_N, MACRO_N), -1, dtype=int)
    for (br, bc), color in colors.items():
        color_grid[br, bc] = int(color)

    out_masks, out_colors, _ = _extract_blocks(out)
    for br, bc in empty_cells:
        row_rot_missing = set(range(MACRO_N)) - set(map(int, rotation_grid[br, :][rotation_grid[br, :] >= 0]))
        col_rot_missing = set(range(MACRO_N)) - set(map(int, rotation_grid[:, bc][rotation_grid[:, bc] >= 0]))
        row_color_missing = set(sprite_colors) - set(map(int, color_grid[br, :][color_grid[br, :] >= 0]))
        col_color_missing = set(sprite_colors) - set(map(int, color_grid[:, bc][color_grid[:, bc] >= 0]))
        assert len(row_rot_missing) == len(col_rot_missing) == 1
        assert len(row_color_missing) == len(col_color_missing) == 1
        rotation = next(iter(row_rot_missing & col_rot_missing))
        color = next(iter(row_color_missing & col_color_missing))
        assert np.array_equal(out_masks[(br, bc)], np.rot90(ref_mask, rotation))
        assert out_colors[(br, bc)] == color

    expected_out = out
    actual_out = main(inp)
    assert np.array_equal(actual_out, expected_out)
    assert structure["bg"] == int(meta["bg"])
    assert structure["line_color"] == int(meta["line_color"])
