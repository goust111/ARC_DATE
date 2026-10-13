from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_intersects(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    min_gap: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + min_gap < br0
        or br1 + min_gap < ar0
        or ac1 + min_gap < bc0
        or bc1 + min_gap < ac0
    )


def _paint_block(grid: np.ndarray, top: tuple[int, int], size: int, color: int) -> None:
    r, c = top
    grid[r : r + size, c : c + size] = int(color)


def _find_thumbnail(
    grid: np.ndarray, bg: int
) -> tuple[np.ndarray, tuple[int, int, int, int], set[tuple[int, int]]]:
    candidates: list[tuple[int, int, int, tuple[int, int, int, int], np.ndarray, set[tuple[int, int]]]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox_coords(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        cells = {tuple(map(int, rc)) for rc in coords}
        if len(cells) != sub.size:
            continue
        if np.any(sub == bg):
            continue
        flat = [int(v) for v in sub.flat]
        if len(set(flat)) != len(flat):
            continue
        if sub.shape[0] < 2 or sub.shape[1] < 2 or sub.size < 4:
            continue
        candidates.append((r0 + c0, r0, c0, (r0, c0, r1, c1), sub.copy(), cells))

    assert candidates, "expected a filled multicolor thumbnail in the top-left"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[4].size))
    best = candidates[0]
    assert len(candidates) == 1 or (candidates[1][0], candidates[1][1], candidates[1][2]) != (
        best[0],
        best[1],
        best[2],
    ), "thumbnail location must be unique"
    return best[4], best[3], best[5]


def _find_blocks(
    grid: np.ndarray, bg: int, thumb_cells: set[tuple[int, int]]
) -> tuple[list[tuple[int, int, int]], int]:
    mask = grid != bg
    for r, c in thumb_cells:
        mask[r, c] = False

    blocks: list[tuple[int, int, int, int]] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 2 or len(coords) != h * w:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(sub[0, 0])
        if color == bg or not np.all(sub == color):
            continue
        blocks.append((r0, c0, h, color))

    assert blocks, "expected square monochrome blocks in the main area"
    sizes = {size for _, _, size, _ in blocks}
    assert len(sizes) == 1, "all blocks must share one block size"
    block_size = sizes.pop()
    colors = [color for *_rest, color in blocks]
    assert len(colors) == len(set(colors)), "thumbnail colors should map to unique block colors"
    return [(r0, c0, color) for r0, c0, _size, color in blocks], block_size


def _layout_fits(
    dr: int,
    dc: int,
    step: int,
    thumb_shape: tuple[int, int],
    block_size: int,
    grid_shape: tuple[int, int],
    thumb_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid_shape
    th, tw = thumb_shape
    r0 = dr
    c0 = dc
    r1 = dr + (th - 1) * step + block_size - 1
    c1 = dc + (tw - 1) * step + block_size - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    for i in range(th):
        for j in range(tw):
            rect = (
                dr + i * step,
                dc + j * step,
                dr + i * step + block_size - 1,
                dc + j * step + block_size - 1,
            )
            if _rect_intersects(rect, thumb_bbox, min_gap=0):
                return False
    return True


def _candidate_scores(
    thumb: np.ndarray,
    thumb_bbox: tuple[int, int, int, int],
    blocks: list[tuple[int, int, int]],
    block_size: int,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    color_to_pos = {int(thumb[i, j]): (i, j) for i in range(thumb.shape[0]) for j in range(thumb.shape[1])}
    block_by_color = {color: (r, c) for r, c, color in blocks}
    assert set(block_by_color).issubset(set(color_to_pos)), "every block color must appear in the thumbnail"

    h, w = grid_shape
    th, tw = thumb.shape
    max_step_r = h - block_size if th == 1 else (h - block_size) // (th - 1)
    max_step_c = w - block_size if tw == 1 else (w - block_size) // (tw - 1)
    max_step = min(max_step_r, max_step_c)

    scored: dict[tuple[int, int, int], tuple[int, int]] = {}
    for step in range(block_size + 1, max_step + 1):
        for color, (br, bc) in block_by_color.items():
            i, j = color_to_pos[color]
            dr = br - i * step
            dc = bc - j * step
            if not _layout_fits(dr, dc, step, thumb.shape, block_size, grid_shape, thumb_bbox):
                continue

            exact = 0
            total_dist = 0
            ok = True
            for other_color, (rr, cc) in block_by_color.items():
                oi, oj = color_to_pos[other_color]
                er = dr + oi * step
                ec = dc + oj * step
                dist = abs(rr - er) + abs(cc - ec)
                if dist == 0:
                    exact += 1
                elif dist == 1:
                    total_dist += 1
                else:
                    ok = False
                    break
            if ok and exact >= 2:
                scored[(dr, dc, step)] = (total_dist, -exact)

    ranked = [(score[0], score[1], dr, dc, step) for (dr, dc, step), score in scored.items()]
    ranked.sort()
    return ranked


def _infer_transform(
    thumb: np.ndarray,
    thumb_bbox: tuple[int, int, int, int],
    blocks: list[tuple[int, int, int]],
    block_size: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int, int]:
    ranked = _candidate_scores(thumb, thumb_bbox, blocks, block_size, grid_shape)
    assert ranked, "could not infer a valid scale and offset"
    best = ranked[0]
    tied = [cand for cand in ranked if cand[:2] == best[:2]]
    assert len(tied) == 1, "scale and offset must be uniquely determined"
    _score_dist, _score_exact, dr, dc, step = best
    return dr, dc, step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide offset blocks back to the thumbnail-indicated lattice and complete the missing ones.

    Input structure:
    - The background is black.
    - A small filled multicolor thumbnail sits in the top-left as a compact rectangle of unique colors.
    - Elsewhere, the same colors appear as separated monochrome square blocks of one shared size.
    - Some blocks already occupy the thumbnail-mapped lattice points, some are shifted by exactly one cell in one cardinal direction, and some are absent.

    Rule:
    - Identify the thumbnail rectangle and the monochrome square blocks.
    - Infer the block lattice step and top-left offset from the correctly placed blocks so that thumbnail cell `(i, j)` maps to block top-left `(dr + i*S, dc + j*S)`.
    - For each thumbnail color in reading order, keep a correctly placed block, slide a same-colored block from one cardinally adjacent top-left cell back onto the expected site, or draw a new block if none exists.
    - Preserve the thumbnail itself.

    Key insight:
    - The thumbnail is both the completion template and the alignment reference, so the solver must distinguish exact placements from near-miss placements before filling true gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = 0

    thumb, thumb_bbox, thumb_cells = _find_thumbnail(g, bg)
    blocks, block_size = _find_blocks(g, bg, thumb_cells)
    dr, dc, step = _infer_transform(thumb, thumb_bbox, blocks, block_size, g.shape)

    color_to_block = {color: (r, c) for r, c, color in blocks}
    out = g.copy()

    for i in range(thumb.shape[0]):
        for j in range(thumb.shape[1]):
            color = int(thumb[i, j])
            expected = (dr + i * step, dc + j * step)
            current = color_to_block.get(color)
            if current == expected:
                continue
            if current is not None and abs(current[0] - expected[0]) + abs(current[1] - expected[1]) == 1:
                out[current[0] : current[0] + block_size, current[1] : current[1] + block_size] = bg
                _paint_block(out, expected, block_size, color)
                color_to_block[color] = expected

    for i in range(thumb.shape[0]):
        for j in range(thumb.shape[1]):
            color = int(thumb[i, j])
            expected = (dr + i * step, dc + j * step)
            if color_to_block.get(color) != expected:
                _paint_block(out, expected, block_size, color)
                color_to_block[color] = expected

    return out


_THUMB_SHAPES = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
_BLOCK_SIZES = (2, 3, 4)
_GAPS = (2, 3, 4)
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MAX_ATTEMPTS = 400


def _expected_positions(
    base_r: int,
    base_c: int,
    step: int,
    thumb_shape: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, int]]:
    return {
        (i, j): (base_r + i * step, base_c + j * step)
        for i in range(thumb_shape[0])
        for j in range(thumb_shape[1])
    }


def _render_case(
    shape: tuple[int, int],
    thumb_origin: tuple[int, int],
    thumb: np.ndarray,
    block_size: int,
    placements: dict[tuple[int, int], tuple[int, int] | None],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    tr, tc = thumb_origin
    grid[tr : tr + thumb.shape[0], tc : tc + thumb.shape[1]] = thumb
    for (i, j), top in placements.items():
        if top is None:
            continue
        _paint_block(grid, top, block_size, int(thumb[i, j]))
    return grid


def _build_output(
    shape: tuple[int, int],
    thumb_origin: tuple[int, int],
    thumb: np.ndarray,
    block_size: int,
    expected: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    tr, tc = thumb_origin
    grid[tr : tr + thumb.shape[0], tc : tc + thumb.shape[1]] = thumb
    for i in range(thumb.shape[0]):
        for j in range(thumb.shape[1]):
            _paint_block(grid, expected[(i, j)], block_size, int(thumb[i, j]))
    return grid


def _choose_state_counts(area: int) -> tuple[int, int, int]:
    misaligned = random.randint(1, max(1, area - 3))
    correct = random.randint(2, area - misaligned - 1)
    missing = area - correct - misaligned
    assert correct >= 2 and misaligned >= 1 and missing >= 1
    return correct, misaligned, missing


def _candidate_shift_rect(
    expected_top: tuple[int, int],
    direction: tuple[int, int],
    block_size: int,
) -> tuple[int, int, int, int]:
    er, ec = expected_top
    dr, dc = direction
    rr = er + dr
    cc = ec + dc
    return rr, cc, rr + block_size - 1, cc + block_size - 1


def _in_bounds_rect(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h, w = shape
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 29)
    w = random.randint(18, 29)
    thumb_h, thumb_w = random.choice(_THUMB_SHAPES)
    block_size = random.choice(_BLOCK_SIZES)
    step = block_size + random.choice(_GAPS)

    max_base_r = h - 1 - block_size - (thumb_h - 1) * step
    max_base_c = w - 1 - block_size - (thumb_w - 1) * step
    thumb_r = random.randint(1, 2)
    thumb_c = random.randint(1, 2)
    min_base_r = thumb_r + thumb_h + 2
    min_base_c = thumb_c + thumb_w + 4
    if min_base_r > max_base_r or min_base_c > max_base_c:
        return None

    base_r = random.randint(min_base_r, max_base_r)
    base_c = random.randint(min_base_c, max_base_c)
    colors = random.sample(list(range(1, 10)), thumb_h * thumb_w)
    thumb = np.array(colors, dtype=int).reshape(thumb_h, thumb_w)
    if random.random() < 0.5:
        thumb = thumb[:, ::-1]
    if random.random() < 0.5:
        thumb = thumb[::-1, :]

    expected = _expected_positions(base_r, base_c, step, thumb.shape)
    correct_n, misaligned_n, _missing_n = _choose_state_counts(thumb.size)
    cells = [(i, j) for i in range(thumb.shape[0]) for j in range(thumb.shape[1])]
    random.shuffle(cells)
    correct_cells = set(cells[:correct_n])
    misaligned_cells = cells[correct_n : correct_n + misaligned_n]

    placements: dict[tuple[int, int], tuple[int, int] | None] = {cell: None for cell in cells}
    occupied = [
        (top[0], top[1], top[0] + block_size - 1, top[1] + block_size - 1)
        for cell, top in expected.items()
        if cell in correct_cells
    ]
    thumb_rect = (thumb_r, thumb_c, thumb_r + thumb.shape[0] - 1, thumb_c + thumb.shape[1] - 1)
    for cell in correct_cells:
        placements[cell] = expected[cell]

    chosen_dirs: list[tuple[int, int]] = []
    for cell in misaligned_cells:
        exp_top = expected[cell]
        directions = _DIRS4[:]
        random.shuffle(directions)
        placed = False
        for direction in directions:
            rect = _candidate_shift_rect(exp_top, direction, block_size)
            if not _in_bounds_rect(rect, (h, w)):
                continue
            if _rect_intersects(rect, thumb_rect, min_gap=1):
                continue
            if any(_rect_intersects(rect, other, min_gap=1) for other in occupied):
                continue
            placements[cell] = (rect[0], rect[1])
            occupied.append(rect)
            chosen_dirs.append(direction)
            placed = True
            break
        if not placed:
            return None

    if misaligned_n >= 2 and len(set(chosen_dirs)) < 2:
        return None

    inp = _render_case((h, w), (thumb_r, thumb_c), thumb, block_size, placements)
    out = _build_output((h, w), (thumb_r, thumb_c), thumb, block_size, expected)

    thumb2, thumb_bbox2, thumb_cells2 = _find_thumbnail(inp, 0)
    blocks2, block_size2 = _find_blocks(inp, 0, thumb_cells2)
    ranked = _candidate_scores(thumb2, thumb_bbox2, blocks2, block_size2, inp.shape)
    if not ranked:
        return None
    if len([cand for cand in ranked if cand[:2] == ranked[0][:2]]) != 1:
        return None
    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    moved = 0
    missing = 0
    color_to_block = {color: (r, c) for r, c, color in blocks2}
    for i in range(thumb.shape[0]):
        for j in range(thumb.shape[1]):
            color = int(thumb[i, j])
            current = color_to_block.get(color)
            target = expected[(i, j)]
            if current is None:
                missing += 1
            elif current != target:
                moved += 1
    if moved < 1 or missing < 1:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        case = _attempt_generate()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid thumbnail-slide puzzle")
