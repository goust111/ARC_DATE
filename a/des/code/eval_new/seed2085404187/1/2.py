from __future__ import annotations

import random

import numpy as np


BOARD_COLOR = 9
GREY_COLOR = 5
ORANGE_COLOR = 7
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL8_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_BG_POOL = [0, 1, 2, 3, 4, 6, 8]
GEN_GRID_RANGE = (20, 28)
GEN_BOARD_SIDE_RANGE = (10, 15)
GEN_SLOT_COUNT_RANGE = (2, 4)
GEN_SHAPE_AREA_RANGE = (4, 9)
GEN_MAX_SHAPE_SIDE = 5
GEN_MAX_GREY = 3
GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 120
GEN_SLOT_ATTEMPTS = 120
GEN_PIECE_ATTEMPTS = 160


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in ORTHO_DIRS:
        nr, nc = int(r) + dr, int(c) + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield nr, nc


def _neighbors8(shape: tuple[int, int], r: int, c: int):
    for dr, dc in ALL8_DIRS:
        nr, nc = int(r) + dr, int(c) + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield nr, nc


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((H, W), rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _largest_monochrome_component(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    best_color = -1
    best_coords: np.ndarray | None = None
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for comp in _mask_components(g == int(color)):
            if best_coords is None or len(comp) > len(best_coords):
                best_color = int(color)
                best_coords = comp
    assert best_coords is not None, "expected a non-background component"
    return best_color, best_coords


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(np.asarray(mask, dtype=bool))]


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), int(k))


def _find_rotation(piece_mask: np.ndarray, slot_mask: np.ndarray) -> int | None:
    for k in range(4):
        if np.array_equal(_rotate_mask(piece_mask, k), slot_mask):
            return int(k)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored piece into its labeled board slot, then recolor filled slots by slot depth parity.

    Input structure:
    - bg is the most common color. A large central monochrome board (maroon in the task description)
      is the largest non-background component.
    - Inside the board bbox are several 4-connected slot regions made of bg cells and optional grey
      counting pixels; each slot has exactly one adjacent single-pixel label of a distinct color.
    - Outside the board are monochrome connected pieces whose colors match the slot-label colors.

    Rule:
    - For each slot, collect its bg cells as the placement mask and count its grey cells as depth D.
    - Find the outside piece with the same color as the slot label; up to a 0/90/180/270 rotation,
      its tight mask must match the slot's bg-only mask.
    - Erase every outside piece from the background and fill the entire slot region on the board.
    - If D is odd, paint the slot orange; if D is even (including 0), keep the piece's original color.

    Key insight:
    - Color labels identify which piece belongs to which slot, while the grey-dot count controls a
      second-stage recoloring that is independent of the geometric match.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    board_color, board_coords = _largest_monochrome_component(g, bg)
    r0, c0, r1, c1 = _bbox_from_coords(board_coords)
    board_patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    open_mask = (board_patch == int(bg)) | (board_patch == int(GREY_COLOR))
    slot_comps = _mask_components(open_mask)
    assert slot_comps, "expected at least one slot inside the board"

    slot_id = np.full(board_patch.shape, -1, dtype=int)
    for idx, comp in enumerate(slot_comps):
        for rr, cc in comp:
            slot_id[int(rr), int(cc)] = int(idx)

    label_map: dict[int, int] = {}
    for rr in range(board_patch.shape[0]):
        for cc in range(board_patch.shape[1]):
            color = int(board_patch[rr, cc])
            if color in {int(bg), int(board_color), int(GREY_COLOR)}:
                continue
            touched = {
                int(slot_id[nr, nc])
                for nr, nc in _neighbors4(board_patch.shape, rr, cc)
                if slot_id[nr, nc] >= 0
            }
            assert len(touched) == 1, "each label must touch exactly one slot"
            slot_idx = next(iter(touched))
            assert slot_idx not in label_map, "each slot should have a unique label"
            label_map[slot_idx] = int(color)

    assert len(label_map) == len(slot_comps), "every slot should have one label"

    outside_piece_components: dict[int, np.ndarray] = {}
    for label_color in label_map.values():
        color_comps = _mask_components(g == int(label_color))
        outside = []
        for comp in color_comps:
            inside_board = np.all(
                (comp[:, 0] >= int(r0))
                & (comp[:, 0] <= int(r1))
                & (comp[:, 1] >= int(c0))
                & (comp[:, 1] <= int(c1))
            )
            if not inside_board:
                outside.append(comp)
        assert len(outside) == 1, "each label color must have exactly one outside piece"
        outside_piece_components[int(label_color)] = outside[0]

    out = g.copy()
    for comp in outside_piece_components.values():
        out[comp[:, 0], comp[:, 1]] = int(bg)

    for idx, slot_comp in enumerate(slot_comps):
        label_color = int(label_map[idx])
        piece_mask = _tight_mask_from_coords(outside_piece_components[label_color])

        rel_fill = np.asarray(
            [
                (int(rr), int(cc))
                for rr, cc in slot_comp
                if int(board_patch[int(rr), int(cc)]) == int(bg)
            ],
            dtype=int,
        )
        assert rel_fill.size > 0, "slot fill area should be nonempty"
        slot_fill_mask = _tight_mask_from_coords(rel_fill)
        assert _find_rotation(piece_mask, slot_fill_mask) is not None, "piece must match slot up to rotation"

        grey_count = int(sum(int(board_patch[int(rr), int(cc)]) == int(GREY_COLOR) for rr, cc in slot_comp))
        paint = int(ORANGE_COLOR if grey_count % 2 == 1 else label_color)
        for rr, cc in slot_comp:
            out[int(r0) + int(rr), int(c0) + int(cc)] = paint

    return out


def _is_connected_coords(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    seen = set()
    stack = [next(iter(coords))]
    seen.add(stack[0])
    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHO_DIRS:
            nxt = (int(r) + dr, int(c) + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        h = random.randint(2, GEN_MAX_SHAPE_SIDE)
        w = random.randint(2, GEN_MAX_SHAPE_SIDE)
        if int(area) > h * w:
            continue
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < int(area):
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4((h, w), r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) != int(area):
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue
        norm = {(r - min(rows), c - min(cols)) for r, c in cells}
        nh = max(r for r, _ in norm) + 1
        nw = max(c for _, c in norm) + 1
        mask = np.zeros((nh, nw), dtype=bool)
        for r, c in norm:
            mask[int(r), int(c)] = True
        return mask
    return None


def _remove_grey_cells(support_mask: np.ndarray, grey_count: int) -> np.ndarray | None:
    coords = set(_coords_from_mask(support_mask))
    active = set(coords)
    for _ in range(int(grey_count)):
        candidates = []
        for cell in sorted(active):
            trial = active - {cell}
            if trial and _is_connected_coords(trial):
                candidates.append(cell)
        if not candidates:
            return None
        active.remove(random.choice(candidates))
    if not active:
        return None
    fill = np.zeros_like(support_mask, dtype=bool)
    for r, c in active:
        fill[int(r), int(c)] = True
    return fill


def _dilate4(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _can_place_support(
    support_occ: np.ndarray,
    label_occ: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> bool:
    bh, bw = support_occ.shape
    h, w = mask.shape
    if top < 2 or left < 2 or top + h > bh - 2 or left + w > bw - 2:
        return False
    blocked = _dilate4(support_occ | label_occ)
    for dr, dc in np.argwhere(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        if blocked[rr, cc]:
            return False
    return True


def _choose_label_cell(
    support_occ: np.ndarray,
    label_occ: np.ndarray,
    placed_mask: np.ndarray,
    top: int,
    left: int,
) -> tuple[int, int] | None:
    bh, bw = support_occ.shape
    abs_cells = {(int(top + dr), int(left + dc)) for dr, dc in np.argwhere(placed_mask)}
    candidates = set()
    for r, c in abs_cells:
        for nr, nc in _neighbors4((bh, bw), r, c):
            if (nr, nc) in abs_cells or support_occ[nr, nc] or label_occ[nr, nc]:
                continue
            if nr in {0, bh - 1} or nc in {0, bw - 1}:
                continue
            if any(support_occ[ar, ac] for ar, ac in _neighbors4((bh, bw), nr, nc)):
                continue
            nearby_labels = False
            for ar, ac in _neighbors8((bh, bw), nr, nc):
                if label_occ[ar, ac]:
                    nearby_labels = True
                    break
            if not nearby_labels:
                candidates.add((int(nr), int(nc)))
    if not candidates:
        return None
    return random.choice(sorted(candidates))


def _pick_board_dims(H: int, W: int) -> tuple[int, int]:
    limit = int(0.48 * H * W)
    choices = []
    for bh in range(GEN_BOARD_SIDE_RANGE[0], min(GEN_BOARD_SIDE_RANGE[1], H - 5) + 1):
        for bw in range(GEN_BOARD_SIDE_RANGE[0], min(GEN_BOARD_SIDE_RANGE[1], W - 5) + 1):
            if bh * bw <= limit:
                choices.append((bh, bw))
    assert choices, "expected feasible board sizes"
    return random.choice(choices)


def _centered_board_origin(H: int, W: int, bh: int, bw: int) -> tuple[int, int]:
    base_r = (H - bh) // 2
    base_c = (W - bw) // 2
    r_candidates = [r for r in range(max(2, base_r - 1), min(H - bh - 2, base_r + 1) + 1)]
    c_candidates = [c for c in range(max(2, base_c - 1), min(W - bw - 2, base_c + 1) + 1)]
    return random.choice(r_candidates), random.choice(c_candidates)


def _make_slot_shape(parity: int) -> tuple[np.ndarray, np.ndarray, int] | None:
    grey_choices = [d for d in range(GEN_MAX_GREY + 1) if d % 2 == parity]
    random.shuffle(grey_choices)
    for grey_count in grey_choices:
        min_area = max(GEN_SHAPE_AREA_RANGE[0], grey_count + 3)
        max_area = GEN_SHAPE_AREA_RANGE[1]
        if min_area > max_area:
            continue
        total_area = random.randint(min_area, max_area)
        support = _random_polyomino(total_area)
        if support is None:
            continue
        fill = _remove_grey_cells(support, grey_count)
        if fill is None:
            continue
        if int(fill.sum()) < 3:
            continue
        if _find_rotation(_tight_mask_from_coords(np.argwhere(fill)), _tight_mask_from_coords(np.argwhere(fill))) is None:
            continue
        rot = random.randrange(4)
        support = _rotate_mask(support, rot)
        fill = _rotate_mask(fill, rot)
        return support, fill, int(grey_count)
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[int(top + dr), int(left + dc)] = int(color)


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _can_place_piece(
    grid: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    bg: int,
    board_rect: tuple[int, int, int, int],
) -> bool:
    H, W = grid.shape
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    if _rectangles_intersect((top, left, top + h - 1, left + w - 1), board_rect):
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        if int(grid[rr, cc]) != int(bg):
            return False
        r0 = max(0, rr - 1)
        c0 = max(0, cc - 1)
        r1 = min(H, rr + 2)
        c1 = min(W, cc + 2)
        if np.any(grid[r0:r1, c0:c1] != int(bg)):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_GRID_RANGE)
    W = random.randint(*GEN_GRID_RANGE)
    bg = random.choice(GEN_BG_POOL)
    bh, bw = _pick_board_dims(H, W)
    board_top, board_left = _centered_board_origin(H, W, bh, bw)
    board_rect = (board_top, board_left, board_top + bh - 1, board_left + bw - 1)

    n_slots = random.randint(*GEN_SLOT_COUNT_RANGE)
    label_pool = [c for c in range(10) if c not in {bg, BOARD_COLOR, GREY_COLOR, ORANGE_COLOR}]
    if len(label_pool) < n_slots:
        return None
    label_colors = random.sample(label_pool, n_slots)

    parities = [i % 2 for i in range(n_slots)]
    random.shuffle(parities)

    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    inp[board_top : board_top + bh, board_left : board_left + bw] = int(BOARD_COLOR)
    out[board_top : board_top + bh, board_left : board_left + bw] = int(BOARD_COLOR)

    support_occ = np.zeros((bh, bw), dtype=bool)
    label_occ = np.zeros((bh, bw), dtype=bool)
    slot_specs = []

    for slot_idx in range(n_slots):
        placed = False
        for _ in range(GEN_SLOT_ATTEMPTS):
            shape = _make_slot_shape(parities[slot_idx])
            if shape is None:
                continue
            support_mask, fill_mask, grey_count = shape
            h, w = support_mask.shape
            if h > bh - 4 or w > bw - 4:
                continue
            top = random.randint(2, bh - h - 2)
            left = random.randint(2, bw - w - 2)
            if not _can_place_support(support_occ, label_occ, support_mask, top, left):
                continue
            label_cell = _choose_label_cell(support_occ, label_occ, support_mask, top, left)
            if label_cell is None:
                continue

            support_occ[top : top + h, left : left + w] |= support_mask
            lr, lc = label_cell
            label_occ[lr, lc] = True

            abs_top = board_top + top
            abs_left = board_left + left
            grey_mask = support_mask & ~fill_mask

            _paint_mask(inp, fill_mask, abs_top, abs_left, bg)
            _paint_mask(inp, grey_mask, abs_top, abs_left, GREY_COLOR)

            final_color = ORANGE_COLOR if grey_count % 2 == 1 else label_colors[slot_idx]
            _paint_mask(out, support_mask, abs_top, abs_left, final_color)

            inp[board_top + lr, board_left + lc] = int(label_colors[slot_idx])
            out[board_top + lr, board_left + lc] = int(label_colors[slot_idx])

            slot_specs.append(
                {
                    "label_color": int(label_colors[slot_idx]),
                    "fill_mask": np.asarray(fill_mask, dtype=bool),
                    "grey_count": int(grey_count),
                }
            )
            placed = True
            break
        if not placed:
            return None

    for spec in slot_specs:
        piece_mask = _tight_mask_from_coords(np.argwhere(spec["fill_mask"]))
        piece_mask = _rotate_mask(piece_mask, random.randrange(4))
        placed = False
        for _ in range(GEN_PIECE_ATTEMPTS):
            h, w = piece_mask.shape
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            if not _can_place_piece(inp, piece_mask, top, left, bg, board_rect):
                continue
            _paint_mask(inp, piece_mask, top, left, spec["label_color"])
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
