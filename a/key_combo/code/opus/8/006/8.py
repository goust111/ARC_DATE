from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 0
RED = 2
CYAN = 8
GROUP_COLORS = (1, 3, 4, 5, 6, 7, 9)
SOURCE_PATTERNS = (
    ((0, 1), (0, 2), (1, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((1, 0), (2, 0), (2, 1)),
    ((1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 2), (2, 2)),
)
SCALE_SETS_2 = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3))
SCALE_SETS_3 = ((1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3))
RETRY_LIMIT = 400


def _is_square_component(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int] | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    size_r = int(r1 - r0 + 1)
    size_c = int(c1 - c0 + 1)
    if size_r != size_c:
        return None
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
        return None
    return int(r0), int(c0), int(size_r)


def _source_patch_and_anchor(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    mixed = components_mask((grid == RED) | (grid == CYAN), connectivity=4)
    candidates: list[tuple[int, int, int, int]] = []
    for comp in mixed:
        colors = {int(grid[r, c]) for r, c in comp}
        if colors != {RED, CYAN}:
            continue
        red_coords = np.array([(int(r), int(c)) for r, c in comp if int(grid[r, c]) == RED], dtype=int)
        if red_coords.size == 0:
            continue
        square = _is_square_component(grid, red_coords, RED)
        if square is None:
            continue
        r0, c0, size = square
        score = min(r0, c0, grid.shape[0] - (r0 + size), grid.shape[1] - (c0 + size))
        candidates.append((score, r0, c0, size))
    assert candidates, "expected one mixed red-cyan source object"
    _, r0, c0, size = min(candidates)
    patch_r0 = r0 - size
    patch_c0 = c0 - size
    patch = grid[patch_r0 : patch_r0 + 3 * size, patch_c0 : patch_c0 + 3 * size].copy()
    return patch, (int(r0), int(c0), int(size))


def _local_red_anchors(grid: np.ndarray, source_anchor: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    sr0, sc0, ssize = source_anchor
    for comp in components_mask(grid == RED, connectivity=4):
        square = _is_square_component(grid, comp, RED)
        if square is None:
            continue
        if square == (sr0, sc0, ssize):
            continue
        out.append(square)
    return out


def _diagonal_groups(grid: np.ndarray, bg: int) -> list[dict]:
    comps_by_color: dict[int, list[tuple[int, int, int]]] = {}
    for color in sorted(set(map(int, grid.ravel())) - {bg, RED, CYAN}):
        for comp in components_mask(grid == color, connectivity=4):
            square = _is_square_component(grid, comp, color)
            if square is not None:
                comps_by_color.setdefault(int(color), []).append(square)

    groups: list[dict] = []
    for color, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        (r0a, c0a, size_a), (r0b, c0b, size_b) = sorted(comps)
        if size_a != size_b:
            continue
        size = int(size_a)
        ur0 = min(r0a, r0b)
        uc0 = min(c0a, c0b)
        pos = {
            ((r0a - ur0) // size, (c0a - uc0) // size),
            ((r0b - ur0) // size, (c0b - uc0) // size),
        }
        if pos == {(0, 0), (1, 1)}:
            vacancies = [(0, 1), (1, 0)]
        elif pos == {(0, 1), (1, 0)}:
            vacancies = [(0, 0), (1, 1)]
        else:
            continue
        groups.append(
            {
                "color": int(color),
                "size": size,
                "bbox": (ur0, uc0, ur0 + 2 * size - 1, uc0 + 2 * size - 1),
                "vacancies": [
                    (ur0 + br * size, uc0 + bc * size, br, bc) for br, bc in vacancies
                ],
            }
        )
    return groups


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(dr + dc)


def _assign_anchors(
    groups: list[dict],
    anchors: list[tuple[int, int, int]],
    source_patch: np.ndarray,
    source_size: int,
) -> None:
    used = [False] * len(anchors)
    patch_h = int(source_patch.shape[0])
    for group in groups:
        k = int(group["size"])
        candidates: list[tuple[int, int]] = []
        for idx, (r0, c0, size) in enumerate(anchors):
            if used[idx]:
                continue
            if patch_h * size != k * source_size:
                continue
            dist = _bbox_gap(group["bbox"], (r0, c0, r0 + size - 1, c0 + size - 1))
            candidates.append((dist, idx))
        assert candidates, "each diagonal group needs one matching red anchor"
        candidates.sort()
        _, best_idx = candidates[0]
        used[best_idx] = True
        group["anchor"] = anchors[best_idx]


def _scaled_patch(source_patch: np.ndarray, factor: int, flip_mode: str) -> np.ndarray:
    patch = np.kron(source_patch, np.ones((factor, factor), dtype=int))
    if flip_mode == "h":
        patch = np.fliplr(patch)
    elif flip_mode == "v":
        patch = np.flipud(patch)
    return patch


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Stamp a scaled cyan motif into the diagonal-complement vacancies of each square-block pair.

    Input structure:
    - The grid has a black background, one corner source object made from a red square anchor and an attached cyan motif,
      plus 2 or 3 diagonal groups of same-color solid squares.
    - Each diagonal group consists of two equal square blocks occupying one diagonal of a 2x2 block footprint.
    - A separate red square anchor sits near each diagonal group, and its side length is an integer multiple of the source anchor size.

    Rule:
    - Extract the source motif from the corner object and treat the source red square as its anchor.
    - For each diagonal group, use the ratio between the nearby red anchor size and the source anchor size as the integer scale factor.
    - Find the two complementary empty blocks of that 2x2 footprint, scale the source motif by the factor, and stamp it into each vacancy.
    - The vacancy in the top row receives a horizontal mirror of the scaled motif, while the vacancy in the bottom row receives a vertical mirror.
    - Keep the original source object, square blocks, and local anchors unchanged.

    Key insight:
    - The empty diagonal positions tell where to stamp, and the nearby red anchor size tells how large the copied cyan motif must be.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = get_bg(grid)
    out = grid.copy()

    source_patch, source_anchor = _source_patch_and_anchor(grid)
    source_size = int(source_anchor[2])
    groups = _diagonal_groups(grid, bg)
    assert groups, "expected diagonal square-block groups"
    anchors = _local_red_anchors(grid, source_anchor)
    _assign_anchors(groups, anchors, source_patch, source_size)

    for group in groups:
        _ar0, _ac0, target_size = group["anchor"]
        factor = target_size // source_size
        assert factor >= 1 and factor * source_size == target_size
        for vr, vc, block_r, _block_c in group["vacancies"]:
            flip_mode = "h" if block_r == 0 else "v"
            patch = _scaled_patch(source_patch, factor, flip_mode)
            red_coords = np.argwhere(patch == RED)
            assert red_coords.size > 0
            rr0, cc0, rr1, cc1 = bbox_coords(red_coords)
            red_h = rr1 - rr0 + 1
            red_w = cc1 - cc0 + 1
            assert red_h == target_size and red_w == target_size
            target_r0 = vr + (group["size"] - target_size) // 2
            target_c0 = vc + (group["size"] - target_size) // 2
            place_r0 = target_r0 - rr0
            place_c0 = target_c0 - cc0
            stamp = np.where(patch == CYAN, CYAN, bg)
            paste(out, stamp, int(place_r0), int(place_c0), transparent=bg, clip=True)

    return out


def _source_object_patch() -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[1, 1] = RED
    for r, c in random.choice(SOURCE_PATTERNS):
        patch[r, c] = CYAN
    return patch


def _dilate_once(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _paint_rect(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    grid[r0 : r0 + size, c0 : c0 + size] = int(color)


def _group_blocks(
    r0: int, c0: int, size: int, diag_kind: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    if diag_kind == 0:
        filled = [(r0, c0), (r0 + size, c0 + size)]
        vacancies = [(r0, c0 + size, 0, 1), (r0 + size, c0, 1, 0)]
    else:
        filled = [(r0, c0 + size), (r0 + size, c0)]
        vacancies = [(r0, c0, 0, 0), (r0 + size, c0 + size, 1, 1)]
    return filled, vacancies


def _anchor_candidates(
    vr: int,
    vc: int,
    block_r: int,
    block_c: int,
    block_size: int,
    anchor_size: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    outward_sides = []
    if block_r == 0:
        outward_sides.append("top")
    else:
        outward_sides.append("bottom")
    if block_c == 0:
        outward_sides.append("left")
    else:
        outward_sides.append("right")
    for gap in (0, 1):
        for side in outward_sides:
            if side == "top":
                row = vr - gap - anchor_size
                if row >= 0:
                    for col in range(vc, vc + block_size - anchor_size + 1):
                        out.append((row, col))
            elif side == "bottom":
                row = vr + block_size + gap
                if row + anchor_size <= h:
                    for col in range(vc, vc + block_size - anchor_size + 1):
                        out.append((row, col))
            elif side == "left":
                col = vc - gap - anchor_size
                if col >= 0:
                    for row in range(vr, vr + block_size - anchor_size + 1):
                        out.append((row, col))
            else:
                col = vc + block_size + gap
                if col + anchor_size <= w:
                    for row in range(vr, vr + block_size - anchor_size + 1):
                        out.append((row, col))
    random.shuffle(out)
    return out


def _composite_cells(
    r0: int,
    c0: int,
    block_size: int,
    diag_kind: int,
    anchor_pos: tuple[int, int],
    anchor_size: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for br0, bc0 in _group_blocks(r0, c0, block_size, diag_kind)[0]:
        for rr in range(br0, br0 + block_size):
            for cc in range(bc0, bc0 + block_size):
                cells.add((rr, cc))
    ar0, ac0 = anchor_pos
    for rr in range(ar0, ar0 + anchor_size):
        for cc in range(ac0, ac0 + anchor_size):
            cells.add((rr, cc))
    return cells


def _try_place_group(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    color: int,
    scale: int,
    diag_kind: int,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    anchor_size = scale
    block_size = 3 * scale
    footprint = 2 * block_size
    row_options = list(range(0, h - footprint + 1))
    col_options = list(range(0, w - footprint + 1))
    random.shuffle(row_options)
    random.shuffle(col_options)
    for r0 in row_options:
        for c0 in col_options:
            _filled, vacancies = _group_blocks(r0, c0, block_size, diag_kind)
            vacancy_choices = vacancies[:]
            random.shuffle(vacancy_choices)
            for vr, vc, block_r, block_c in vacancy_choices:
                for ar0, ac0 in _anchor_candidates(
                    vr, vc, block_r, block_c, block_size, anchor_size, grid.shape
                ):
                    cells = _composite_cells(r0, c0, block_size, diag_kind, (ar0, ac0), anchor_size)
                    if cells & forbidden:
                        continue
                    for br0, bc0 in _filled:
                        _paint_rect(grid, br0, bc0, block_size, color)
                    _paint_rect(grid, ar0, ac0, anchor_size, RED)
                    return _dilate_once(cells, grid.shape, radius=1)
    return None


def _place_source_object(grid: np.ndarray, patch: np.ndarray) -> set[tuple[int, int]]:
    h, w = grid.shape
    options = (
        (0, 0),
        (0, w - 3),
        (h - 3, 0),
        (h - 3, w - 3),
    )
    r0, c0 = random.choice(options)
    paste(grid, patch, r0, c0)
    cells = {(r0 + r, c0 + c) for r in range(3) for c in range(3) if int(patch[r, c]) != BG}
    return _dilate_once(cells, grid.shape, radius=1)


def _generate_once() -> np.ndarray | None:
    n_groups = random.choice((2, 3))
    scales = list(random.choice(SCALE_SETS_2 if n_groups == 2 else SCALE_SETS_3))
    random.shuffle(scales)
    colors = random.sample(GROUP_COLORS, n_groups)
    diags = [random.randint(0, 1) for _ in range(n_groups)]

    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)
    forbidden = _place_source_object(grid, _source_object_patch())

    group_specs = sorted(
        zip(scales, colors, diags, strict=False),
        key=lambda item: item[0],
        reverse=True,
    )
    for scale, color, diag_kind in group_specs:
        added = _try_place_group(grid, forbidden, color, scale, diag_kind)
        if added is None:
            return None
        forbidden |= added
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_once()
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("generate_input failed")
