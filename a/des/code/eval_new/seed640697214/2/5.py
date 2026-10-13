from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
LAYOUT_RETRIES = 250
PLACEMENT_RETRIES = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    cur = mask.astype(bool)
    for _ in range(4):
        key = _mask_key(cur)
        if key not in seen:
            seen.add(key)
            variants.append(cur.copy())
        cur = np.rot90(cur)
    return variants


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(rot) for rot in _rotation_variants(mask))


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _board_from_grey(grid: np.ndarray) -> dict[str, object]:
    grey_pts = np.argwhere(grid == GREY)
    assert grey_pts.size > 0, "expected a grey puzzle board"
    ir0, ic0 = grey_pts.min(axis=0)
    ir1, ic1 = grey_pts.max(axis=0)
    ir0 = int(ir0)
    ic0 = int(ic0)
    ir1 = int(ir1)
    ic1 = int(ic1)
    r0, c0, r1, c1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    assert r0 >= 0 and c0 >= 0 and r1 < grid.shape[0] and c1 < grid.shape[1]

    top = _mode(grid[r0, c0 + 1 : c1])
    right = _mode(grid[r0 + 1 : r1, c1])
    bottom = _mode(grid[r1, c0 + 1 : c1])
    left = _mode(grid[r0 + 1 : r1, c0])
    side_colors = [int(top), int(right), int(bottom), int(left)]
    assert len(set(side_colors)) == 4, "side colors must be distinct"

    interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    assert np.all((interior == GREY) | (interior == BG)), "board interior should be grey with black gaps"
    hole_union = interior == BG
    assert bool(hole_union.any()), "board must contain at least one gap"

    return {
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (ir0, ic0, ir1, ic1),
        "side_colors": side_colors,
        "priority": {color: rank for rank, color in enumerate(side_colors)},
        "hole_union": hole_union,
    }


def _pieces_outside_board(grid: np.ndarray, board_bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    h, w = grid.shape
    r0, c0, r1, c1 = board_bbox
    outside = np.ones((h, w), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components((grid != BG) & outside)
    pieces: list[dict[str, object]] = []
    for pts in comps:
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1, "each loose fragment must be monochrome"
        color = colors.pop()
        pr0, pc0, pr1, pc1 = _bbox(pts)
        patch = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
        for r, c in pts:
            patch[r - pr0, c - pc0] = True
        pieces.append({"color": color, "mask": _trim_mask(patch)})
    pieces.sort(key=lambda item: (int(item["color"]), _mask_key(np.asarray(item["mask"], dtype=bool))))
    assert pieces, "expected at least one loose fragment"
    return pieces


def _candidate_placements(hole_union: np.ndarray, piece_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = hole_union.shape
    candidates: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for rot in _rotation_variants(piece_mask):
        ph, pw = rot.shape
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                window = hole_union[r0 : r0 + ph, c0 : c0 + pw]
                if not np.all(window[rot]):
                    continue
                full = np.zeros_like(hole_union, dtype=bool)
                full[r0 : r0 + ph, c0 : c0 + pw] = rot
                key = full.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"r0": r0, "c0": c0, "mask": full})
    candidates.sort(key=lambda item: (int(item["r0"]), int(item["c0"]), item["mask"].sum()))
    return candidates


def _find_layout_solutions(
    hole_union: np.ndarray, piece_masks: list[np.ndarray], limit: int = 2
) -> list[list[dict[str, object]]]:
    n = len(piece_masks)
    if n == 0:
        return []

    candidates = [_candidate_placements(hole_union, mask) for mask in piece_masks]
    if any(len(options) == 0 for options in candidates):
        return []

    order = sorted(range(n), key=lambda idx: (len(candidates[idx]), -int(piece_masks[idx].sum())))
    remaining_cover = [np.zeros_like(hole_union, dtype=bool) for _ in range(n + 1)]
    for pos in range(n - 1, -1, -1):
        cur = remaining_cover[pos + 1].copy()
        idx = order[pos]
        for cand in candidates[idx]:
            cur |= np.asarray(cand["mask"], dtype=bool)
        remaining_cover[pos] = cur

    solutions: list[list[dict[str, object]]] = []
    chosen: list[dict[str, object] | None] = [None] * n

    def backtrack(pos: int, covered: np.ndarray) -> None:
        if len(solutions) >= limit:
            return
        if np.any(hole_union & ~(covered | remaining_cover[pos])):
            return
        if pos == n:
            if np.array_equal(covered, hole_union):
                solution = [cand for cand in chosen if cand is not None]
                assert len(solution) == n
                solutions.append(solution)
            return

        idx = order[pos]
        for cand in candidates[idx]:
            new_covered = covered | np.asarray(cand["mask"], dtype=bool)
            if np.any(hole_union & ~(new_covered | remaining_cover[pos + 1])):
                continue
            chosen[idx] = cand
            backtrack(pos + 1, new_covered)
            chosen[idx] = None

    backtrack(0, np.zeros_like(hole_union, dtype=bool))
    return solutions


def _render_completed_board(
    grid: np.ndarray,
    board: dict[str, object],
    pieces: list[dict[str, object]],
    layout: list[dict[str, object]],
) -> np.ndarray:
    out = np.full_like(grid, BG)
    r0, c0, r1, c1 = board["bbox"]  # type: ignore[index]
    ir0, ic0, ir1, ic1 = board["inner_bbox"]  # type: ignore[index]
    out[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = GREY

    priority = dict(board["priority"])  # type: ignore[arg-type]
    inner = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
    order = sorted(range(len(pieces)), key=lambda idx: int(priority[int(pieces[idx]["color"])]))
    for idx in order:
        inner[np.asarray(layout[idx]["mask"], dtype=bool)] = int(pieces[idx]["color"])
    return out


def _solve_grid(grid: np.ndarray, solution_limit: int = 1) -> tuple[dict[str, object], list[dict[str, object]], list[list[dict[str, object]]]]:
    board = _board_from_grey(grid)
    pieces = _pieces_outside_board(grid, board["bbox"])  # type: ignore[arg-type]
    priority = dict(board["priority"])  # type: ignore[arg-type]
    for piece in pieces:
        assert int(piece["color"]) in priority, "fragment color must come from a board side"
    layouts = _find_layout_solutions(
        np.asarray(board["hole_union"], dtype=bool),
        [np.asarray(piece["mask"], dtype=bool) for piece in pieces],
        limit=solution_limit,
    )
    return board, pieces, layouts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit the loose colored fragments back into the grey puzzle board and resolve overlaps by border priority.

    Input structure:
    - The background is black.
    - A large central grey rectangle forms the puzzle board; inside it, black cells mark the missing area to be filled.
    - The board border is split into four differently colored sides read clockwise as top, right, bottom, and left.
    - Outside the board sit several monochrome loose fragments whose colors reuse board-side colors.

    Rule:
    - Detect the board, read the four side colors in clockwise order, and treat that order as the layering priority.
    - For each loose fragment, try all four rotations and place it onto the board's black missing area so the fragment lies entirely inside that missing region.
    - Choose one placement per fragment whose union exactly reconstructs the board's black area; then paint those placed fragments back onto the board.
    - When multiple placed fragments cover the same cell, the fragment whose color comes later in the side order top -> right -> bottom -> left stays visible; loose fragments disappear from outside the board.

    Key insight:
    - The black region on the board is only the union of the hidden holes, so you must recover the latent placement of each rotated fragment first and only then apply the border-derived stacking order.
    """
    g = np.asarray(grid, dtype=int)
    board, pieces, layouts = _solve_grid(g, solution_limit=1)
    assert layouts, "no valid fragment placement found"
    return _render_completed_board(g, board, pieces, layouts[0])


def _random_piece_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_area = min(8, h * w - 1)
        min_area = max(3, (h + w) // 2)
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = [start]
        while len(cells) < target:
            base_r, base_c = rng.choice(frontier)
            nbrs = []
            for nr, nc in ((base_r - 1, base_c), (base_r + 1, base_c), (base_r, base_c - 1), (base_r, base_c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((base_r, base_c))
                if not frontier:
                    frontier = list(cells)
                continue
            pick = rng.choice(nbrs)
            cells.add(pick)
            frontier.append(pick)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        rows = np.any(mask, axis=1).sum()
        cols = np.any(mask, axis=0).sum()
        if rows < 2 or cols < 2:
            continue
        if mask.all():
            continue
        if len(_rotation_variants(mask)) == 1:
            continue
        return mask
    raise RuntimeError("failed to sample a fragment shape")


def _sample_piece_set(rng: random.Random, count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    while len(shapes) < count:
        mask = _random_piece_mask(rng)
        key = _canonical_rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return shapes


def _full_mask(shape: tuple[int, int], piece_mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    h, w = piece_mask.shape
    full[r0 : r0 + h, c0 : c0 + w] = piece_mask
    return full


def _sample_hole_layout(
    inner_shape: tuple[int, int],
    oriented_masks: list[np.ndarray],
    want_overlap: bool,
    rng: random.Random,
) -> list[np.ndarray] | None:
    h, w = inner_shape
    n = len(oriented_masks)
    overlap_pair = tuple(rng.sample(range(n), 2)) if want_overlap else None

    for _ in range(LAYOUT_RETRIES):
        placements: list[np.ndarray | None] = [None] * n
        union = np.zeros(inner_shape, dtype=bool)

        if overlap_pair is not None:
            first, second = overlap_pair
            first_mask = oriented_masks[first]
            fh, fw = first_mask.shape
            if h - fh - 1 < 1 or w - fw - 1 < 1:
                continue
            fr0 = rng.randint(1, h - fh - 1)
            fc0 = rng.randint(1, w - fw - 1)
            first_full = _full_mask(inner_shape, first_mask, fr0, fc0)
            placements[first] = first_full
            union |= first_full

            second_mask = oriented_masks[second]
            sh, sw = second_mask.shape
            placed = False
            for _attempt in range(PLACEMENT_RETRIES):
                if h - sh - 1 < 1 or w - sw - 1 < 1:
                    break
                sr0 = rng.randint(1, h - sh - 1)
                sc0 = rng.randint(1, w - sw - 1)
                second_full = _full_mask(inner_shape, second_mask, sr0, sc0)
                overlap = first_full & second_full
                overlap_count = int(overlap.sum())
                if overlap_count == 0:
                    continue
                if overlap_count >= min(int(first_mask.sum()), int(second_mask.sum())):
                    continue
                placements[second] = second_full
                union |= second_full
                placed = True
                break
            if not placed:
                continue

        for idx, piece_mask in enumerate(oriented_masks):
            if placements[idx] is not None:
                continue
            ph, pw = piece_mask.shape
            if h - ph - 1 < 1 or w - pw - 1 < 1:
                break
            placed = False
            for _attempt in range(PLACEMENT_RETRIES):
                r0 = rng.randint(1, h - ph - 1)
                c0 = rng.randint(1, w - pw - 1)
                full = _full_mask(inner_shape, piece_mask, r0, c0)
                if bool((_dilate8(union) & full).any()):
                    continue
                placements[idx] = full
                union |= full
                placed = True
                break
            if not placed:
                break
        else:
            final = [mask for mask in placements if mask is not None]
            if len(final) == n:
                return final

    return None


def _place_loose_fragments(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    pieces: list[dict[str, object]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = board_bbox
    blocked = np.zeros((h, w), dtype=bool)
    br0 = max(0, r0 - 1)
    bc0 = max(0, c0 - 1)
    br1 = min(h - 1, r1 + 1)
    bc1 = min(w - 1, c1 + 1)
    blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True

    order = list(range(len(pieces)))
    order.sort(key=lambda idx: -int(np.asarray(pieces[idx]["loose_mask"], dtype=bool).sum()))
    for idx in order:
        mask = np.asarray(pieces[idx]["loose_mask"], dtype=bool)
        ph, pw = mask.shape
        rows = range(1, h - ph) if h - ph > 1 else range(0, h - ph + 1)
        cols = range(1, w - pw) if w - pw > 1 else range(0, w - pw + 1)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r_start = rng.choice(list(rows))
            c_start = rng.choice(list(cols))
            full = _full_mask((h, w), mask, r_start, c_start)
            halo = _dilate8(full)
            if bool((blocked & halo).any()):
                continue
            grid[r_start : r_start + ph, c_start : c_start + pw][mask] = int(pieces[idx]["color"])
            blocked |= halo
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(
    grid_shape: tuple[int, int],
    board_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    side_colors: list[int],
    holes: list[dict[str, object]],
) -> np.ndarray:
    h, w = grid_shape
    out = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = board_bbox
    ir0, ic0, ir1, ic1 = inner_bbox

    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = GREY
    out[r0, c0 : c1 + 1] = side_colors[0]
    out[r0 : r1 + 1, c1] = side_colors[1]
    out[r1, c0 : c1 + 1] = side_colors[2]
    out[r0 : r1 + 1, c0] = side_colors[3]

    rank = {color: idx for idx, color in enumerate(side_colors)}
    inner = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
    for hole in sorted(holes, key=lambda item: rank[int(item["color"])]):
        inner[np.asarray(hole["placed_mask"], dtype=bool)] = int(hole["color"])
    return out


def _make_sample(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = rng.choice((2, 3, 4))
    want_overlap = bool(rng.randrange(2))
    base_masks = _sample_piece_set(rng, piece_count)

    max_h = max(mask.shape[0] for mask in base_masks)
    max_w = max(mask.shape[1] for mask in base_masks)
    inner_h = rng.randint(max(9, max_h + 4), 15)
    inner_w = rng.randint(max(9, max_w + 4), 15)

    oriented_holes = [rng.choice(_rotation_variants(mask)) for mask in base_masks]
    placed_holes = _sample_hole_layout((inner_h, inner_w), oriented_holes, want_overlap, rng)
    if placed_holes is None:
        return None

    side_colors = rng.sample(list(PALETTE), 4)
    side_indices = rng.sample(range(4), piece_count)

    holes: list[dict[str, object]] = []
    for idx, base_mask in enumerate(base_masks):
        color = side_colors[side_indices[idx]]
        holes.append(
            {
                "color": color,
                "placed_mask": placed_holes[idx],
                "loose_mask": rng.choice(_rotation_variants(base_mask)),
            }
        )

    board_h = inner_h + 2
    board_w = inner_w + 2
    min_h = max(board_h + 8, 20)
    min_w = max(board_w + 8, 20)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    top_center = (h - board_h) // 2
    left_center = (w - board_w) // 2
    board_r0 = min(max(3, top_center + rng.randint(-2, 2)), h - board_h - 3)
    board_c0 = min(max(3, left_center + rng.randint(-2, 2)), w - board_w - 3)
    board_r1 = board_r0 + board_h - 1
    board_c1 = board_c0 + board_w - 1
    inner_bbox = (board_r0 + 1, board_c0 + 1, board_r1 - 1, board_c1 - 1)

    grid = np.full((h, w), BG, dtype=int)
    grid[inner_bbox[0] : inner_bbox[2] + 1, inner_bbox[1] : inner_bbox[3] + 1] = GREY
    grid[board_r0, board_c0 : board_c1 + 1] = side_colors[0]
    grid[board_r0 : board_r1 + 1, board_c1] = side_colors[1]
    grid[board_r1, board_c0 : board_c1 + 1] = side_colors[2]
    grid[board_r0 : board_r1 + 1, board_c0] = side_colors[3]

    inner = grid[inner_bbox[0] : inner_bbox[2] + 1, inner_bbox[1] : inner_bbox[3] + 1]
    for hole in holes:
        inner[np.asarray(hole["placed_mask"], dtype=bool)] = BG

    if not _place_loose_fragments(grid, (board_r0, board_c0, board_r1, board_c1), holes, rng):
        return None

    expected = _expected_output(
        (h, w),
        (board_r0, board_c0, board_r1, board_c1),
        inner_bbox,
        side_colors,
        holes,
    )

    try:
        board, pieces, layouts = _solve_grid(grid, solution_limit=2)
    except Exception:
        return None
    if len(layouts) != 1:
        return None

    solved = _render_completed_board(grid, board, pieces, layouts[0])
    if not np.array_equal(solved, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        sample = _make_sample(rng)
        if sample is None:
            continue
        inp, _expected = sample
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
