from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SEPARATOR = 3
FRAGMENT_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _separator_rows(grid: np.ndarray) -> list[int]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR)]
    assert rows, "expected separator rows"
    assert len(rows) == len(set(rows)), "separator rows must be unique"
    return rows


def _split_groups(grid: np.ndarray) -> list[np.ndarray]:
    sep_rows = _separator_rows(grid)
    starts = [0] + [r + 1 for r in sep_rows]
    stops = sep_rows + [int(grid.shape[0])]
    groups = []
    for start, stop in zip(starts, stops):
        assert start < stop, "group must have positive height"
        groups.append(grid[start:stop].copy())
    assert 3 <= len(groups) <= 4, "expected 3 or 4 fragment groups"
    return groups


def _extract_fragment(group: np.ndarray) -> np.ndarray:
    mask = (group != BACKGROUND) & (group != SEPARATOR)
    comps = _components(mask)
    assert len(comps) == 1, "each group must contain one fragment"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = group[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert patch.size > 0
    assert np.all(patch != BACKGROUND), "fragment bbox must be fully rectangular"
    assert np.all(patch != SEPARATOR), "fragment cannot contain separator color"
    return patch


def _rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(grid, -int(quarter_turns) % 4).copy()


def _diag_layout(blocks: list[np.ndarray]) -> np.ndarray:
    row_heights = [int(block.shape[0]) for block in blocks]
    col_widths = [int(block.shape[1]) for block in blocks]
    out = np.zeros((sum(row_heights), sum(col_widths)), dtype=int)

    row_offset = 0
    col_offset = 0
    for block, h, w in zip(blocks, row_heights, col_widths):
        out[row_offset : row_offset + h, col_offset : col_offset + w] = block
        row_offset += h
        col_offset += w
    return out


def _canonical_rotation_key(patch: np.ndarray) -> tuple:
    variants = []
    for k in range(4):
        rot = _rotate_cw(patch, k)
        variants.append((rot.shape, tuple(int(x) for x in rot.ravel())))
    return min(variants)


def _all_rotations_distinct(patch: np.ndarray) -> bool:
    keys = {
        (rot.shape, tuple(int(x) for x in rot.ravel()))
        for rot in (_rotate_cw(patch, k) for k in range(4))
    }
    return len(keys) == 4


def _wrong_layouts(patches: list[np.ndarray]) -> list[np.ndarray]:
    wrong = []
    wrong.append(_diag_layout(patches))
    wrong.append(_diag_layout([_rotate_cw(p, k + 1) for k, p in enumerate(patches)]))
    wrong.append(_diag_layout([_rotate_cw(p, -k) for k, p in enumerate(patches)]))
    wrong.append(_diag_layout([_rotate_cw(patches[::-1][k], k) for k in range(len(patches))]))
    return wrong


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the input by green separator rows, rotate each fragment by its group index, and place the rotated fragments on a diagonal block layout.

    Input structure:
    - The background is black.
    - One or more full green rows divide the tall grid into 3 or 4 horizontal groups.
    - Each group contains exactly one rectangular non-black non-green fragment on the background.

    Rule:
    - Use the full green rows as group boundaries and number groups from top to bottom starting at 0.
    - Extract the unique fragment from each group by its tight bounding box.
    - Rotate the fragment in group k clockwise by k*90 degrees.
    - Build an output grid whose diagonal blocks have the rotated fragments' sizes and place fragment k in diagonal block (k, k).
    - Fill every non-diagonal block with black and omit the green separator rows.

    Key insight:
    - The separator rows do two jobs at once: they split the input into fragment groups and their top-to-bottom order implicitly encodes the rotation amount for each fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    groups = _split_groups(g)
    patches = [_extract_fragment(group) for group in groups]
    rotated = [_rotate_cw(patch, k) for k, patch in enumerate(patches)]
    out = _diag_layout(rotated)
    assert max(out.shape) <= 30, "output exceeds ARC size limit"
    return out


def _make_structured_patch(height: int, width: int) -> np.ndarray | None:
    palette_size = random.randint(2, 4)
    colors = random.sample(FRAGMENT_COLORS, palette_size)
    patch = np.full((height, width), colors[0], dtype=int)

    mode = random.choice(("row_cell", "col_cell", "row_col", "double_cell"))
    row = random.randrange(height)
    col = random.randrange(width)

    if mode == "row_cell":
        patch[row, :] = colors[1]
        cell_choices = [(r, c) for r in range(height) for c in range(width) if r != row]
        if not cell_choices:
            return None
        r2, c2 = random.choice(cell_choices)
        patch[r2, c2] = colors[0 if palette_size == 2 else 2]
        if palette_size == 4:
            cell_choices = [(r, c) for r in range(height) for c in range(width) if (r, c) not in {(r2, c2)}]
            r3, c3 = random.choice(cell_choices)
            patch[r3, c3] = colors[3]
    elif mode == "col_cell":
        patch[:, col] = colors[1]
        cell_choices = [(r, c) for r in range(height) for c in range(width) if c != col]
        if not cell_choices:
            return None
        r2, c2 = random.choice(cell_choices)
        patch[r2, c2] = colors[0 if palette_size == 2 else 2]
        if palette_size == 4:
            cell_choices = [(r, c) for r in range(height) for c in range(width) if (r, c) not in {(r2, c2)}]
            r3, c3 = random.choice(cell_choices)
            patch[r3, c3] = colors[3]
    elif mode == "row_col":
        patch[row, :] = colors[1]
        patch[:, col] = colors[0 if palette_size == 2 else 2]
        if palette_size >= 3:
            patch[row, col] = colors[1]
        if palette_size == 4:
            cell_choices = [(r, c) for r in range(height) for c in range(width) if (r, c) != (row, col)]
            r3, c3 = random.choice(cell_choices)
            patch[r3, c3] = colors[3]
    else:
        cell_choices = [(r, c) for r in range(height) for c in range(width)]
        a = random.choice(cell_choices)
        b_choices = [p for p in cell_choices if p != a]
        b = random.choice(b_choices)
        patch[a] = colors[1]
        patch[b] = colors[0 if palette_size == 2 else 2]
        if palette_size == 4:
            c_choices = [p for p in cell_choices if p not in {a, b}]
            patch[random.choice(c_choices)] = colors[3]

    if len(set(int(x) for x in patch.ravel())) < 2:
        return None
    if not _all_rotations_distinct(patch):
        return None
    return patch


def _generate_patches(count: int, height: int, width: int) -> list[np.ndarray] | None:
    patches: list[np.ndarray] = []
    seen_keys: set[tuple] = set()
    for _ in range(count):
        for _attempt in range(80):
            patch = _make_structured_patch(height, width)
            if patch is None:
                continue
            key = _canonical_rotation_key(patch)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            patches.append(patch)
            break
        else:
            return None
    return patches


def _render_input(patches: list[np.ndarray]) -> np.ndarray:
    height, width = patches[0].shape
    count = len(patches)
    side_margin = random.randint(1, 3)
    extra_width = random.randint(1, 3)
    canvas_width = width + 2 * side_margin + extra_width
    top_pads = [random.randint(0, 1) for _ in range(count)]
    bottom_pads = [random.randint(0, 1) for _ in range(count)]
    group_heights = [top_pads[i] + height + bottom_pads[i] for i in range(count)]
    total_height = sum(group_heights) + (count - 1)
    assert total_height <= 30
    canvas = np.zeros((total_height, canvas_width), dtype=int)

    row = 0
    for i, patch in enumerate(patches):
        c0 = random.randint(side_margin, canvas_width - width - side_margin)
        r0 = row + top_pads[i]
        canvas[r0 : r0 + height, c0 : c0 + width] = patch
        row += group_heights[i]
        if i < count - 1:
            canvas[row, :] = SEPARATOR
            row += 1
    return canvas


def _semantic_checks(inp: np.ndarray, patches: list[np.ndarray]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    try:
        groups = _split_groups(inp)
        extracted = [_extract_fragment(group) for group in groups]
        out = main(inp)
    except Exception:
        return False

    if len(extracted) != len(patches):
        return False
    for actual, expected in zip(extracted, patches):
        if not np.array_equal(actual, expected):
            return False
        if not _all_rotations_distinct(actual):
            return False

    expected_out = _diag_layout([_rotate_cw(patch, k) for k, patch in enumerate(patches)])
    if not np.array_equal(out, expected_out):
        return False
    if np.any(out == SEPARATOR):
        return False
    if max(out.shape) > 30:
        return False

    for wrong in _wrong_layouts(patches):
        if np.array_equal(wrong, expected_out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choices([3, 4], weights=[2, 3], k=1)[0]
    dim_choices = [(h, w) for h in range(2, 5) for w in range(3, 6) if h != w and h * w >= 8]
    height, width = random.choice(dim_choices)

    patches = _generate_patches(count, height, width)
    if patches is None:
        return None

    inp = _render_input(patches)
    if not _semantic_checks(inp, patches):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
