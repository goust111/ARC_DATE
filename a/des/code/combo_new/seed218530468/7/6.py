from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


@dataclass(frozen=True)
class Piece:
    color: int
    mask: np.ndarray
    area: int


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches(a: np.ndarray, b: np.ndarray) -> bool:
    h, w = a.shape
    for r, c in np.argwhere(a):
        rr, cc = int(r), int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and b[nr, nc]:
                return True
    return False


def _count_contacts(piece_mask: np.ndarray, occupied: np.ndarray, top: int, left: int) -> int:
    h, w = piece_mask.shape
    contacts = 0
    for r, c in np.argwhere(piece_mask):
        rr, cc = int(top + r), int(left + c)
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                contacts += 1
    return contacts


def _paste(arr: np.ndarray, piece_mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = piece_mask.shape
    view = arr[top : top + h, left : left + w]
    view[piece_mask] = int(color)


def _extract_frame(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey target frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have an interior"
    return int(r1 - r0 - 1), int(c1 - c0 - 1)


def _extract_pieces(grid: np.ndarray) -> list[Piece]:
    h, w = grid.shape
    fg = (grid != BLACK) & (grid != GREY)
    seen = np.zeros_like(fg, dtype=bool)
    pieces: list[Piece] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            r0, c0 = min(rs), min(cs)
            r1, c1 = max(rs), max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True
            pieces.append(Piece(color=color, mask=mask, area=int(mask.sum())))
    assert pieces, "expected at least one piece"
    pieces.sort(key=lambda piece: (int(piece.area), int(piece.mask.shape[0] * piece.mask.shape[1]), int(piece.color)))
    return pieces


def _candidate_positions(piece: Piece, occupied: np.ndarray) -> list[tuple[int, int]]:
    out_h, out_w = occupied.shape
    h, w = piece.mask.shape
    positions = []
    for top in range(out_h - h + 1):
        for left in range(out_w - w + 1):
            view = occupied[top : top + h, left : left + w]
            if bool((view & piece.mask).any()):
                continue
            contacts = _count_contacts(piece.mask, occupied, top, left)
            if contacts == 0:
                continue
            positions.append((-contacts, top + left, top, left))
    positions.sort()
    return [(top, left) for _neg, _diag, top, left in positions]


def _solve_layout(
    pieces: list[Piece],
    out_shape: tuple[int, int],
    max_solutions: int = 1,
) -> tuple[int, list[tuple[int, int]] | None]:
    out_h, out_w = map(int, out_shape)
    total_area = sum(int(piece.area) for piece in pieces)
    if total_area != out_h * out_w:
        return 0, None

    first = pieces[0]
    fh, fw = first.mask.shape
    if fh > out_h or fw > out_w:
        return 0, None
    if not first.mask[0, 0]:
        return 0, None

    occupied = np.zeros((out_h, out_w), dtype=bool)
    first_view = occupied[:fh, :fw]
    first_view[first.mask] = True
    placements: list[tuple[int, int]] = [(0, 0)] + [(-1, -1)] * (len(pieces) - 1)
    first_solution: list[tuple[int, int]] | None = None
    solutions = 0

    def dfs(i: int) -> None:
        nonlocal solutions, first_solution
        if solutions >= max_solutions:
            return
        if i == len(pieces):
            if bool(occupied.all()):
                solutions += 1
                if first_solution is None:
                    first_solution = placements.copy()
            return

        piece = pieces[i]
        for top, left in _candidate_positions(piece, occupied):
            h, w = piece.mask.shape
            view = occupied[top : top + h, left : left + w]
            view[piece.mask] = True
            placements[i] = (int(top), int(left))
            dfs(i + 1)
            view[piece.mask] = False
            placements[i] = (-1, -1)
            if solutions >= max_solutions:
                return

    dfs(1)
    return solutions, first_solution


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered toothed jigsaw pieces into the grey target frame by area order.

    Input structure:
    - Black background with several differently colored 4-connected pieces scattered apart.
    - Each piece is a translation-only puzzle fragment whose boundary may include 1-pixel tabs and matching 1-pixel notches.
    - A single grey rectangular border marks the target frame near the bottom of the grid.

    Rule:
    - Ignore the pieces' current positions, crop each piece to its own bounding box, and sort the pieces by pixel area from small to large.
    - Place the smallest piece first with its bounding-box corner aligned to the target interior's top-left corner.
    - Place every later piece by translation only so that the pieces interlock inside the target frame and together exactly fill the frame interior; when several cells would coincide, later pieces would be painted last.
    - Return only the frame interior, without the grey border itself.

    Key insight:
    - The hidden order is area, not input location, and the exact translation of each later piece is determined by complementary boundary geometry inside the target rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out_shape = _extract_frame(g)
    pieces = _extract_pieces(g)
    count, placements = _solve_layout(pieces, out_shape, max_solutions=1)
    assert count == 1 and placements is not None, "expected a unique assembly"

    out = np.zeros(out_shape, dtype=int)
    for piece, (top, left) in zip(pieces, placements):
        _paste(out, piece.mask, piece.color, int(top), int(left))
    return out


def _boundary_cells(src: np.ndarray, dst: np.ndarray) -> list[tuple[int, int]]:
    cells = []
    h, w = src.shape
    for r, c in np.argwhere(src):
        rr, cc = int(r), int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and dst[nr, nc]:
                cells.append((rr, cc))
                break
    return cells


def _piece_is_clear(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(mask.sum())
    return area >= 5 and h >= 2 and w >= 2 and area * 10 >= 4 * h * w


def _split_region(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray] | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    options: list[tuple[np.ndarray, np.ndarray]] = []
    for orient in random.sample(["v", "h"], 2):
        if orient == "v":
            cuts = list(range(c0 + 1, c1 + 1))
            random.shuffle(cuts)
            for cut in cuts:
                left = mask.copy()
                left[:, cut:] = False
                right = mask & (~left)
                if int(left.sum()) < min_area or int(right.sum()) < min_area:
                    continue
                if not _is_connected(left) or not _is_connected(right):
                    continue
                cells = _boundary_cells(right, left) + _boundary_cells(left, right)
                random.shuffle(cells)
                for rr, cc in cells:
                    if right[rr, cc]:
                        a, b = left.copy(), right.copy()
                        b[rr, cc] = False
                        a[rr, cc] = True
                    else:
                        a, b = left.copy(), right.copy()
                        a[rr, cc] = False
                        b[rr, cc] = True
                    if int(a.sum()) < min_area or int(b.sum()) < min_area:
                        continue
                    if _is_connected(a) and _is_connected(b) and _piece_is_clear(a) and _piece_is_clear(b):
                        options.append((a, b))
        else:
            cuts = list(range(r0 + 1, r1 + 1))
            random.shuffle(cuts)
            for cut in cuts:
                top = mask.copy()
                top[cut:, :] = False
                bottom = mask & (~top)
                if int(top.sum()) < min_area or int(bottom.sum()) < min_area:
                    continue
                if not _is_connected(top) or not _is_connected(bottom):
                    continue
                cells = _boundary_cells(bottom, top) + _boundary_cells(top, bottom)
                random.shuffle(cells)
                for rr, cc in cells:
                    if bottom[rr, cc]:
                        a, b = top.copy(), bottom.copy()
                        b[rr, cc] = False
                        a[rr, cc] = True
                    else:
                        a, b = top.copy(), bottom.copy()
                        a[rr, cc] = False
                        b[rr, cc] = True
                    if int(a.sum()) < min_area or int(b.sum()) < min_area:
                        continue
                    if _is_connected(a) and _is_connected(b) and _piece_is_clear(a) and _piece_is_clear(b):
                        options.append((a, b))
    if not options:
        return None
    return random.choice(options)


def _top_left_piece(out_h: int, out_w: int) -> np.ndarray | None:
    for _ in range(100):
        h0 = random.randint(2, min(4, out_h - 2))
        w0 = random.randint(2, min(4, out_w - 2))
        mask = np.zeros((out_h, out_w), dtype=bool)
        mask[:h0, :w0] = True
        tab_count = random.randint(1, 2)
        sides = []
        if w0 < out_w and h0 >= 3:
            sides.append("right")
        if h0 < out_h and w0 >= 3:
            sides.append("bottom")
        random.shuffle(sides)
        for side in sides[:tab_count]:
            if side == "right":
                rr = random.randint(1, h0 - 2)
                mask[rr, w0] = True
            else:
                cc = random.randint(1, w0 - 2)
                mask[h0, cc] = True
        remainder = np.ones((out_h, out_w), dtype=bool) & (~mask)
        if _piece_is_clear(mask) and _is_connected(remainder):
            return mask
    return None


def _assemble_partition() -> tuple[list[np.ndarray], np.ndarray] | None:
    out_h = random.randint(7, 11)
    out_w = random.randint(8, 13)
    piece_count = random.randint(3, 5)

    first = _top_left_piece(out_h, out_w)
    if first is None:
        return None

    regions = [np.ones((out_h, out_w), dtype=bool) & (~first)]
    while len(regions) + 1 < piece_count:
        idx = max(range(len(regions)), key=lambda i: int(regions[i].sum()))
        region = regions.pop(idx)
        split = _split_region(region, min_area=5)
        if split is None:
            return None
        regions.extend(split)

    pieces = [first] + regions
    areas = [int(piece.sum()) for piece in pieces]
    if len(set(areas)) != len(areas):
        return None
    smallest_idx = min(range(len(pieces)), key=lambda i: areas[i])
    if smallest_idx != 0:
        return None
    if any(area <= areas[0] for area in areas[1:]):
        return None

    order = sorted(range(len(pieces)), key=lambda i: areas[i])
    union = pieces[order[0]].copy()
    if not union[0, 0]:
        return None
    for idx in order[1:]:
        if not _touches(pieces[idx], union):
            return None
        union |= pieces[idx]
    if not bool(union.all()):
        return None

    sorted_pieces = [pieces[i] for i in order]
    return sorted_pieces, union


def _piece_patch(mask: np.ndarray, color: int) -> np.ndarray:
    crop = _crop(mask)
    out = np.zeros_like(crop, dtype=int)
    out[crop] = int(color)
    return out


def _place_patch_random(grid: np.ndarray, patch: np.ndarray, occupied: np.ndarray, forbidden: np.ndarray) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    spots = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(spots)
    mask = patch != 0
    for top, left in spots:
        if bool((forbidden[top : top + h, left : left + w] & mask).any()):
            continue
        if bool((occupied[top : top + h, left : left + w] & mask).any()):
            continue
        view = grid[top : top + h, left : left + w]
        view[mask] = patch[mask]
        occupied[top : top + h, left : left + w][mask] = True
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(H, top + h + 1)
        c1 = min(W, left + w + 1)
        forbidden[r0:r1, c0:c1] |= True
        return True
    return False


def _build_input_from_partition(piece_masks: list[np.ndarray], target: np.ndarray) -> np.ndarray | None:
    out_h, out_w = target.shape
    n = len(piece_masks)
    colors = random.sample(PIECE_COLORS, n)
    output = np.zeros((out_h, out_w), dtype=int)
    patches = []
    for mask, color in zip(piece_masks, colors):
        output[mask] = int(color)
        patches.append(_piece_patch(mask, int(color)))

    max_piece_h = max(p.shape[0] for p in patches)
    max_piece_w = max(p.shape[1] for p in patches)
    min_grid_h = max(out_h + max_piece_h + 6, 18)
    min_grid_w = max(out_w + max_piece_w + 6, 18)
    if min_grid_h > 30 or min_grid_w > 30:
        return None
    grid_h = random.randint(min_grid_h, 30)
    grid_w = random.randint(min_grid_w, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    frame_h = out_h + 2
    frame_w = out_w + 2
    frame_top = random.randint(max(0, grid_h - frame_h - 2), grid_h - frame_h)
    frame_left = random.randint(0, grid_w - frame_w)
    if frame_top < grid_h // 2:
        frame_top = grid_h - frame_h

    grid[frame_top, frame_left : frame_left + frame_w] = GREY
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = GREY
    grid[frame_top : frame_top + frame_h, frame_left] = GREY
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = GREY

    occupied = np.zeros_like(grid, dtype=bool)
    forbidden = np.zeros_like(grid, dtype=bool)
    r0 = max(0, frame_top - 1)
    c0 = max(0, frame_left - 1)
    r1 = min(grid_h, frame_top + frame_h + 1)
    c1 = min(grid_w, frame_left + frame_w + 1)
    forbidden[r0:r1, c0:c1] = True

    random.shuffle(patches)
    for patch in patches:
        if not _place_patch_random(grid, patch, occupied, forbidden):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        assembled = _assemble_partition()
        if assembled is None:
            continue
        piece_masks, target_mask = assembled
        grid = _build_input_from_partition(piece_masks, target_mask)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        expected = np.zeros(target_mask.shape, dtype=int)
        pieces = _extract_pieces(grid)
        count, placements = _solve_layout(pieces, target_mask.shape, max_solutions=2)
        if count != 1 or placements is None:
            continue
        for piece, (top, left) in zip(pieces, placements):
            _paste(expected, piece.mask, piece.color, int(top), int(left))
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
