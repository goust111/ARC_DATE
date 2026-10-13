from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6

ROTATION_BY_BORDER = {
    RED: 0,
    GREEN: 1,
    YELLOW: 2,
    MAGENTA: 3,
}
BORDER_BY_ROTATION = {turns: color for color, turns in ROTATION_BY_BORDER.items()}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size, "expected a non-empty mask"
    r0 = int(rows[0])
    r1 = int(rows[-1])
    c0 = int(cols[0])
    c1 = int(cols[-1])
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_rotated_pieces(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BLACK)
    assert comps, "expected at least one fragment"
    pieces: list[np.ndarray] = []

    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert BLUE in colors and len(colors) == 2, "each fragment must contain blue plus one border color"
        border_colors = [color for color in colors if color != BLUE]
        assert len(border_colors) == 1 and border_colors[0] in ROTATION_BY_BORDER, "invalid border color"
        border_color = border_colors[0]

        blue_cells = [(r, c) for r, c in comp if int(g[r, c]) == BLUE]
        assert blue_cells, "fragment interior must contain blue cells"
        rs = [r for r, _ in blue_cells]
        cs = [c for _, c in blue_cells]
        r0 = min(rs)
        r1 = max(rs)
        c0 = min(cs)
        c1 = max(cs)

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in blue_cells:
            mask[r - r0, c - c0] = True

        turns = ROTATION_BY_BORDER[border_color]
        rotated = np.rot90(mask, -turns)
        pieces.append(_crop_mask(rotated))

    return pieces


def _candidate_rectangles(area: int) -> list[tuple[int, int]]:
    rects = [(h, area // h) for h in range(1, area + 1) if area % h == 0]
    rects.sort(key=lambda shape: (shape[0] + shape[1], shape[0]))
    return rects


def _coverings_for_piece(piece: np.ndarray, target_h: int, target_w: int) -> dict[int, list[int]]:
    piece = np.asarray(piece, dtype=bool)
    ph, pw = piece.shape
    assert ph <= target_h and pw <= target_w

    coords = [(int(r), int(c)) for r, c in np.argwhere(piece)]
    coverings: dict[int, list[int]] = {}
    for top in range(target_h - ph + 1):
        for left in range(target_w - pw + 1):
            placement = 0
            covered: list[int] = []
            for dr, dc in coords:
                idx = (top + dr) * target_w + (left + dc)
                placement |= 1 << idx
                covered.append(idx)
            for idx in covered:
                coverings.setdefault(idx, []).append(placement)
    return coverings


def _tiles_rectangle(pieces: list[np.ndarray], target_h: int, target_w: int) -> bool:
    total_area = sum(int(piece.sum()) for piece in pieces)
    if total_area != target_h * target_w:
        return False

    order = sorted(
        range(len(pieces)),
        key=lambda idx: (-int(pieces[idx].sum()), -(pieces[idx].shape[0] * pieces[idx].shape[1])),
    )
    ordered = [np.asarray(pieces[idx], dtype=bool) for idx in order]
    areas = [int(piece.sum()) for piece in ordered]
    coverings = [_coverings_for_piece(piece, target_h, target_w) for piece in ordered]
    full_mask = (1 << (target_h * target_w)) - 1
    all_used = (1 << len(ordered)) - 1

    @lru_cache(maxsize=None)
    def dfs(filled_mask: int, used_mask: int) -> bool:
        if used_mask == all_used:
            return filled_mask == full_mask

        empty_bits = full_mask & ~filled_mask
        lowest_empty = empty_bits & -empty_bits
        anchor = lowest_empty.bit_length() - 1

        choices = []
        for idx, area in enumerate(areas):
            if (used_mask >> idx) & 1:
                continue
            valid = [placement for placement in coverings[idx].get(anchor, ()) if (placement & filled_mask) == 0]
            if valid:
                choices.append((len(valid), -area, idx, valid))
        if not choices:
            return False

        choices.sort()
        for _, _, idx, valid in choices:
            next_used = used_mask | (1 << idx)
            for placement in valid:
                if dfs(filled_mask | placement, next_used):
                    return True
        return False

    return dfs(0, 0)


def _find_rectangle_shape(pieces: list[np.ndarray]) -> tuple[int, int]:
    total_area = sum(int(piece.sum()) for piece in pieces)
    for target_h, target_w in _candidate_rectangles(total_area):
        if _tiles_rectangle(pieces, target_h, target_w):
            return int(target_h), int(target_w)
    raise AssertionError("rotated pieces do not tile any solid rectangle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate border-coded blue fragments, then assemble them into a solid rectangle.

    Red/green/yellow/magenta borders encode 0/90/180/270 clockwise turns.
    After rotating each fragment's blue interior accordingly, the pieces tile a
    single filled rectangle. The output keeps only that blue rectangle.
    """
    pieces = _extract_rotated_pieces(grid)
    target_h, target_w = _find_rectangle_shape(pieces)
    return np.full((target_h, target_w), BLUE, dtype=int)


def _sample_partition_sizes(total: int, count: int, rng: random.Random) -> list[int]:
    min_piece = 2 if total >= 2 * count else 1
    sizes = [min_piece] * count
    remaining = total - min_piece * count
    for _ in range(remaining):
        sizes[rng.randrange(count)] += 1
    rng.shuffle(sizes)
    return sizes


def _same_region_neighbors(owner: np.ndarray, label: int, cell: tuple[int, int]) -> int:
    h, w = owner.shape
    r, c = cell
    score = 0
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and int(owner[nr, nc]) == int(label):
            score += 1
    return score


def _grow_partition(height: int, width: int, count: int, rng: random.Random) -> list[np.ndarray] | None:
    total = height * width
    cells = [(r, c) for r in range(height) for c in range(width)]

    for _ in range(80):
        sizes = _sample_partition_sizes(total, count, rng)
        owner = np.full((height, width), -1, dtype=int)
        seeds = rng.sample(cells, count)
        frontiers = [set() for _ in range(count)]
        areas = [1] * count

        for idx, (r, c) in enumerate(seeds):
            owner[r, c] = idx

        for idx, (r, c) in enumerate(seeds):
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and int(owner[nr, nc]) < 0:
                    frontiers[idx].add((int(nr), int(nc)))

        assigned = count
        failed = False
        while assigned < total:
            candidates = [idx for idx in range(count) if areas[idx] < sizes[idx] and frontiers[idx]]
            if not candidates:
                failed = True
                break

            weights = [sizes[idx] - areas[idx] for idx in candidates]
            idx = rng.choices(candidates, weights=weights, k=1)[0]

            frontier = list(frontiers[idx])
            rng.shuffle(frontier)
            frontier.sort(key=lambda cell: _same_region_neighbors(owner, idx, cell), reverse=True)
            r, c = frontier[0]

            owner[r, c] = idx
            areas[idx] += 1
            assigned += 1

            for other in frontiers:
                other.discard((r, c))
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and int(owner[nr, nc]) < 0:
                    frontiers[idx].add((int(nr), int(nc)))

        if failed or np.any(owner < 0):
            continue

        pieces = [_crop_mask(owner == idx) for idx in range(count)]
        if all(piece.any() for piece in pieces):
            return pieces

    return None


def _is_non_rectangular(piece: np.ndarray) -> bool:
    return int(piece.sum()) < piece.shape[0] * piece.shape[1]


def _make_fragment_patch(piece: np.ndarray, border_color: int) -> np.ndarray:
    piece = np.asarray(piece, dtype=bool)
    h, w = piece.shape
    patch = np.full((h + 2, w + 2), BLACK, dtype=int)
    blue = np.zeros((h + 2, w + 2), dtype=bool)
    border = np.zeros((h + 2, w + 2), dtype=bool)

    for r, c in np.argwhere(piece):
        rr = int(r) + 1
        cc = int(c) + 1
        blue[rr, cc] = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                border[rr + dr, cc + dc] = True

    border &= ~blue
    patch[border] = border_color
    patch[blue] = BLUE
    return patch


def _place_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    total_bbox_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    base_span = int(np.ceil(np.sqrt(total_bbox_area))) + len(patches) * 2

    for _ in range(80):
        canvas_h = rng.randint(max(max_h + 4, 10), max(max_h + 7, base_span + 4))
        canvas_w = rng.randint(max(max_w + 4, 10), max(max_w + 9, base_span + 7))
        grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
        occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

        order = list(range(len(patches)))
        rng.shuffle(order)
        order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for top in range(canvas_h - ph + 1):
                for left in range(canvas_w - pw + 1):
                    r0 = max(0, top - 1)
                    r1 = min(canvas_h, top + ph + 1)
                    c0 = max(0, left - 1)
                    c1 = min(canvas_w, left + pw + 1)
                    if occupied[r0:r1, c0:c1].any():
                        continue
                    candidates.append((int(top), int(left)))

            if not candidates:
                ok = False
                break

            top, left = rng.choice(candidates)
            sub = grid[top : top + ph, left : left + pw]
            mask = patch != BLACK
            sub[mask] = patch[mask]
            occupied[top : top + ph, left : left + pw] = True

        if ok:
            return grid

    return None


def _sample_target_shape(rng: random.Random) -> tuple[int, int]:
    while True:
        height = rng.randint(3, 6)
        width = rng.randint(4, 8)
        if 12 <= height * width <= 28:
            return int(height), int(width)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    target_h, target_w = _sample_target_shape(rng)
    total_area = target_h * target_w
    max_pieces = min(5, max(2, total_area // 4 + 1))
    piece_count = rng.randint(2, max_pieces)

    target_pieces = _grow_partition(target_h, target_w, piece_count, rng)
    if target_pieces is None:
        return None

    if not any(_is_non_rectangular(piece) for piece in target_pieces) and rng.random() < 0.6:
        return None

    turns = [rng.randrange(4) for _ in target_pieces]
    if all(turn == 0 for turn in turns) and len(turns) > 1 and rng.random() < 0.8:
        turns[rng.randrange(len(turns))] = rng.choice([1, 2, 3])

    patches = []
    for piece, turn in zip(target_pieces, turns, strict=False):
        input_piece = _crop_mask(np.rot90(piece, turn))
        patches.append(_make_fragment_patch(input_piece, BORDER_BY_ROTATION[turn]))

    inp = _place_patches(patches, rng)
    if inp is None:
        return None

    expected = np.full((target_h, target_w), BLUE, dtype=int)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        case = _generate_case(rng)
        if case is not None:
            inp, _ = case
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 2")
