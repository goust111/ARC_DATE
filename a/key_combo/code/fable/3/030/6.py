from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _mirror(cell: Coord, size: int) -> Coord:
    r, c = cell
    return int(size - 1 - r), int(size - 1 - c)


def _restore_block(crop: np.ndarray, bg: int, fg: int, occ: int) -> np.ndarray:
    n, m = crop.shape
    assert n == m and n >= 3, "expected square block crops"

    restored = crop.copy()
    for rr, cc in np.argwhere(restored == int(occ)):
        r, c = int(rr), int(cc)
        mr, mc = _mirror((r, c), n)
        assert (mr, mc) != (r, c), "a center occlusion has no visible symmetric mate"
        mate = int(restored[mr, mc])
        assert mate != int(occ), "symmetric occlusion pairs are not allowed"
        assert mate in {int(bg), int(fg)}, "occlusion mate must be foreground or background"
        restored[r, c] = mate

    assert not np.any(restored == int(occ)), "restoration must remove all occluder cells"
    assert np.all((restored == int(bg)) | (restored == int(fg))), "restored block has invalid colors"
    assert np.array_equal(restored, restored[::-1, ::-1]), "restored block must be point symmetric"
    assert np.all(restored[0, :] == int(fg)), "top border must be solid foreground"
    assert np.all(restored[-1, :] == int(fg)), "bottom border must be solid foreground"
    assert np.all(restored[:, 0] == int(fg)), "left border must be solid foreground"
    assert np.all(restored[:, -1] == int(fg)), "right border must be solid foreground"
    return restored


def _solve_with_roles(grid: np.ndarray, bg: int, fg: int, occ: int) -> np.ndarray:
    comps = _components4(grid != int(bg))
    assert len(comps) >= 2, "expected several disconnected blocks"

    restored_blocks: list[tuple[int, np.ndarray]] = []
    block_shape: tuple[int, int] | None = None
    for comp in comps:
        crop = _crop_bbox(grid, _bbox_coords(comp))
        assert np.all((crop == int(bg)) | (crop == int(fg)) | (crop == int(occ)))
        if block_shape is None:
            block_shape = crop.shape
        else:
            assert crop.shape == block_shape, "all blocks must be the same size"

        restored = _restore_block(crop, bg, fg, occ)
        holes = int(np.count_nonzero(restored == int(bg)))
        assert holes > 0, "each block must contain background holes"
        restored_blocks.append((holes, restored))

    counts = Counter(holes for holes, _ in restored_blocks)
    unique_counts = [holes for holes, count in counts.items() if count == 1]
    assert len(unique_counts) == 1, "expected exactly one unique restored hole count"

    matches = [block for holes, block in restored_blocks if holes == unique_counts[0]]
    assert len(matches) == 1, "unique hole count must select exactly one block"
    return matches[0].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete occluded point-symmetric blocks, then crop the restored block with the unique hole count.

    Input structure:
    - The dominant color is the background.
    - One foreground color forms several disconnected equal-size square blocks with solid borders.
    - The block interiors contain background holes arranged with point symmetry about each block center.
    - A second foreground color occludes some cells inside the blocks, never hiding both cells of a symmetric pair.

    Rule:
    - For each block, replace every occluder cell with the color of its point-symmetric mate inside that same block.
    - Count the background cells in each fully restored block.
    - Return the restored square block whose hole count occurs exactly once, cropped to its bounding box.

    Key insight:
    - The outlier is defined by the hidden hole count, so the occluders must first be decoded using each block's own point symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected one block color and one occluder color"

    valid: list[np.ndarray] = []
    for fg in colors:
        occ = colors[1] if fg == colors[0] else colors[0]
        try:
            valid.append(_solve_with_roles(g, bg, int(fg), int(occ)))
        except AssertionError:
            continue

    assert len(valid) == 1, "foreground and occluder roles must be structurally unique"
    return valid[0]


# --------------- generator ---------------

MAX_ATTEMPTS = 400
BLOCK_SIZES = (5, 6, 7, 8)
BLOCK_COUNT_RANGE = (4, 6)
MIN_BLOCK_GAP = 2
ARC_COLORS = tuple(range(10))


def _interior_pairs(size: int) -> list[tuple[Coord, Coord]]:
    pairs: list[tuple[Coord, Coord]] = []
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            mate = _mirror((r, c), size)
            if (r, c) == mate:
                continue
            if (r, c) < mate:
                pairs.append(((r, c), mate))
    return pairs


def _holes_are_scattered(holes: set[Coord]) -> bool:
    for r, c in holes:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in holes:
                return False
    return True


def _sample_holes(size: int, count: int) -> set[Coord] | None:
    if count % 2 != 0:
        return None
    pairs = _interior_pairs(size)
    if count // 2 > len(pairs):
        return None
    for _ in range(300):
        chosen = random.sample(pairs, count // 2)
        holes = {cell for pair in chosen for cell in pair}
        if _holes_are_scattered(holes):
            return holes
    return None


def _valid_hole_counts(size: int) -> list[int]:
    max_count = 2 * len(_interior_pairs(size))
    counts = []
    for count in range(2, max_count + 1, 2):
        if _sample_holes(size, count) is not None:
            counts.append(count)
    return counts


def _make_block(size: int, bg: int, fg: int, holes: int) -> np.ndarray | None:
    hole_cells = _sample_holes(size, holes)
    if hole_cells is None:
        return None
    block = np.full((size, size), int(fg), dtype=int)
    for r, c in hole_cells:
        block[r, c] = int(bg)
    return block


def _sample_no_mirror(
    candidates: list[Coord],
    size: int,
    count: int,
    chosen: set[Coord],
) -> list[Coord] | None:
    options = candidates[:]
    random.shuffle(options)
    out: list[Coord] = []
    blocked = set(chosen)
    for cell in options:
        if len(out) == count:
            break
        mate = _mirror(cell, size)
        if cell == mate or cell in blocked or mate in blocked:
            continue
        out.append(cell)
        blocked.add(cell)
    if len(out) != count:
        return None
    chosen.update(out)
    return out


def _choose_occlusions(
    block: np.ndarray,
    bg: int,
    fg: int,
    hidden_holes: int,
    force_foreground: bool,
) -> tuple[set[Coord], int, int] | None:
    size = int(block.shape[0])
    chosen: set[Coord] = set()

    hole_candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(block == int(bg))
        if _mirror((int(r), int(c)), size) != (int(r), int(c))
    ]
    hidden_hole_cells = _sample_no_mirror(hole_candidates, size, hidden_holes, chosen)
    if hidden_hole_cells is None:
        return None

    border_fg = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r in {0, size - 1} or c in {0, size - 1}) and int(block[r, c]) == int(fg)
    ]
    fg_count = random.choice((0, 0, 1, 1, 2))
    if force_foreground:
        fg_count = max(1, fg_count)
    hidden_fg_cells = _sample_no_mirror(border_fg, size, fg_count, chosen)
    if hidden_fg_cells is None:
        return None

    return set(hidden_hole_cells) | set(hidden_fg_cells), len(hidden_hole_cells), len(hidden_fg_cells)


def _layout_options(block_count: int, block_size: int) -> list[tuple[int, int, int, int]]:
    options = []
    for rows in range(2, block_count + 1):
        cols = int(np.ceil(block_count / rows))
        if cols < 2:
            continue
        base_h = rows * block_size + (rows - 1) * MIN_BLOCK_GAP
        base_w = cols * block_size + (cols - 1) * MIN_BLOCK_GAP
        if base_h <= 30 and base_w <= 30:
            options.append((rows, cols, base_h, base_w))
    return options


def _axis_positions(total: int, count: int, size: int) -> list[int] | None:
    base = count * size + (count - 1) * MIN_BLOCK_GAP
    if base > total:
        return None
    extras = [0] * (count + 1)
    for _ in range(total - base):
        extras[random.randrange(count + 1)] += 1

    positions = []
    cursor = extras[0]
    for i in range(count):
        positions.append(cursor)
        if i + 1 < count:
            cursor += size + MIN_BLOCK_GAP + extras[i + 1]
    return positions


def _sample_locations(block_count: int, block_size: int) -> tuple[int, int, list[Coord]] | None:
    options = _layout_options(block_count, block_size)
    if not options:
        return None
    rows, cols, base_h, base_w = random.choice(options)
    height = random.randint(base_h, min(30, base_h + 6))
    width = random.randint(base_w, min(30, base_w + 6))
    row_positions = _axis_positions(height, rows, block_size)
    col_positions = _axis_positions(width, cols, block_size)
    if row_positions is None or col_positions is None:
        return None
    slots = [(r, c) for r in row_positions for c in col_positions]
    random.shuffle(slots)
    return height, width, slots[:block_count]


def _paint_block(grid: np.ndarray, block: np.ndarray, r0: int, c0: int) -> None:
    h, w = block.shape
    grid[r0 : r0 + h, c0 : c0 + w] = block


def _gen_attempt() -> np.ndarray | None:
    bg, fg, occ = random.sample(ARC_COLORS, 3)
    block_size = random.choice(BLOCK_SIZES)
    valid_counts = _valid_hole_counts(block_size)
    if len(valid_counts) < 2:
        return None

    adjacent_pairs = [
        (valid_counts[i], valid_counts[i + 1])
        for i in range(len(valid_counts) - 1)
        if valid_counts[i + 1] - valid_counts[i] <= valid_counts[i + 1] // 2
    ]
    if not adjacent_pairs:
        return None
    lower_holes, higher_holes = random.choice(adjacent_pairs)
    if random.choice((True, False)):
        common_holes, unique_holes = lower_holes, higher_holes
    else:
        common_holes, unique_holes = higher_holes, lower_holes
    visible_holes = min(common_holes, unique_holes)

    block_count = random.randint(*BLOCK_COUNT_RANGE)
    locations = _sample_locations(block_count, block_size)
    if locations is None:
        return None
    height, width, anchors = locations

    unique_index = random.randrange(block_count)
    force_fg_index = random.randrange(block_count)
    grid = np.full((height, width), int(bg), dtype=int)
    expected: np.ndarray | None = None
    hidden_original_values: list[int] = []
    raw_visible_counts: list[int] = []

    for i, (r0, c0) in enumerate(anchors):
        true_holes = unique_holes if i == unique_index else common_holes
        block = _make_block(block_size, bg, fg, true_holes)
        if block is None:
            return None

        hidden_holes = true_holes - visible_holes
        occlusion = _choose_occlusions(
            block,
            bg,
            fg,
            hidden_holes,
            force_foreground=(i == force_fg_index),
        )
        if occlusion is None:
            return None
        occ_cells, _hidden_hole_count, _hidden_fg_count = occlusion

        inp_block = block.copy()
        for r, c in occ_cells:
            hidden_original_values.append(int(block[r, c]))
            inp_block[r, c] = int(occ)

        raw_visible_counts.append(int(np.count_nonzero(inp_block == int(bg))))
        _paint_block(grid, inp_block, r0, c0)
        if i == unique_index:
            expected = block

    if expected is None:
        return None
    true_counts = [unique_holes if i == unique_index else common_holes for i in range(block_count)]
    if Counter(true_counts).most_common() and sum(1 for c in Counter(true_counts).values() if c == 1) != 1:
        return None
    if len(set(raw_visible_counts)) != 1:
        return None
    if int(bg) not in hidden_original_values or int(fg) not in hidden_original_values:
        return None

    try:
        result = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fable/3/030/6")
