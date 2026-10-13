from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BG = 0
WILD = 1
PALETTE = (2, 3, 4, 7)
BLOCK = 5
MARK = 6

GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _find_multicolor_components(grid: np.ndarray, allowed: set[int]) -> list[dict]:
    mask = np.isin(grid, list(allowed))
    seen = np.zeros_like(mask, dtype=bool)
    pieces = []
    h, w = grid.shape
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
            arr = np.array(coords, dtype=int)
            bbox = _bbox_coords(arr)
            patch = _crop_bbox(grid, bbox)
            area = int(arr.shape[0])
            bh = bbox[2] - bbox[0] + 1
            bw = bbox[3] - bbox[1] + 1
            assert area == bh * bw, "expected solid rectangular pieces"
            pieces.append(
                {
                    "coords": arr,
                    "bbox": bbox,
                    "array": patch,
                    "area": area,
                }
            )
    pieces.sort(key=lambda item: item["bbox"])
    for idx, piece in enumerate(pieces):
        piece["index"] = idx
    return pieces


def _find_marker(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    coords = np.argwhere(grid == MARK)
    assert coords.shape[0] == 3, "expected one 3-cell marker"
    r0, c0, r1, c1 = _bbox_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker must occupy a 2x2 box"
    cells = {tuple(map(int, rc)) for rc in coords}
    all_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(all_cells - cells)
    assert len(missing) == 1
    miss = missing[0]
    if miss == (r0, c0):
        return "BR", miss
    if miss == (r0, c1):
        return "BL", miss
    if miss == (r1, c0):
        return "TR", miss
    return "TL", miss


def _rect_top_left(anchor: tuple[int, int], corner: str, rh: int, rw: int) -> tuple[int, int]:
    ar, ac = anchor
    if corner == "TL":
        return ar, ac
    if corner == "TR":
        return ar, ac - rw + 1
    if corner == "BL":
        return ar - rh + 1, ac
    return ar - rh + 1, ac - rw + 1


def _piece_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _row_first_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    cells = []
    step_r = 0 if er == sr else (1 if er > sr else -1)
    cur_r = sr
    while cur_r != er:
        cur_r += step_r
        cells.append((cur_r, sc))
    step_c = 0 if ec == sc else (1 if ec > sc else -1)
    cur_c = sc
    while cur_c != ec:
        cur_c += step_c
        cells.append((er, cur_c))
    return cells


def _obstacle_score(
    piece_bbox: tuple[int, int, int, int],
    rect_bbox: tuple[int, int, int, int],
    obstacles: set[tuple[int, int]],
) -> tuple[int, int]:
    sr, sc = _piece_center(piece_bbox)
    r0, c0, r1, c1 = rect_bbox
    tr = (r0 + r1) // 2
    tc = (c0 + c1) // 2
    path = _row_first_path((sr, sc), (tr, tc))
    blocked = sum((cell in obstacles) for cell in path)
    dist = abs(sr - tr) + abs(sc - tc)
    return int(blocked), int(dist)


def _edge_compatible(a: np.ndarray, b: np.ndarray) -> bool:
    mask = (a != WILD) & (b != WILD)
    return bool(np.all(a[mask] == b[mask]))


def _candidate_positions(
    piece: dict,
    step: int,
    rect_shape: tuple[int, int],
    corner: str,
    occ_mask: np.ndarray,
    occ_color: np.ndarray,
) -> list[tuple[int, int]]:
    ph, pw = piece["array"].shape
    rh, rw = rect_shape
    out = []
    if ph > rh or pw > rw:
        return out
    if step == 0:
        if corner == "TL":
            return [(0, 0)]
        if corner == "TR":
            return [(0, rw - pw)]
        if corner == "BL":
            return [(rh - ph, 0)]
        return [(rh - ph, rw - pw)]

    for top in range(rh - ph + 1):
        for left in range(rw - pw + 1):
            if occ_mask[top : top + ph, left : left + pw].any():
                continue
            patch = piece["array"]
            touch = False

            if left > 0:
                neigh = occ_mask[top : top + ph, left - 1]
                if neigh.any():
                    if not _edge_compatible(patch[:, 0][neigh], occ_color[top : top + ph, left - 1][neigh]):
                        continue
                    touch = True
            if left + pw < rw:
                neigh = occ_mask[top : top + ph, left + pw]
                if neigh.any():
                    if not _edge_compatible(patch[:, -1][neigh], occ_color[top : top + ph, left + pw][neigh]):
                        continue
                    touch = True
            if top > 0:
                neigh = occ_mask[top - 1, left : left + pw]
                if neigh.any():
                    if not _edge_compatible(patch[0, :][neigh], occ_color[top - 1, left : left + pw][neigh]):
                        continue
                    touch = True
            if top + ph < rh:
                neigh = occ_mask[top + ph, left : left + pw]
                if neigh.any():
                    if not _edge_compatible(patch[-1, :][neigh], occ_color[top + ph, left : left + pw][neigh]):
                        continue
                    touch = True

            if touch:
                out.append((top, left))
    return out


def _render_rect(rect_shape: tuple[int, int], placements: list[tuple[int, int]], pieces: list[dict], order: list[int]) -> np.ndarray:
    rh, rw = rect_shape
    out = np.full((rh, rw), BG, dtype=int)
    for pos, idx in zip(placements, order):
        top, left = pos
        patch = pieces[idx]["array"]
        _paste(out, patch, top, left)
    return out


def _solve_for_order(
    pieces: list[dict],
    rect_bbox: tuple[int, int, int, int],
    corner: str,
    order: list[int],
    max_solutions: int = 2,
) -> list[np.ndarray]:
    r0, c0, r1, c1 = rect_bbox
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1
    rect_shape = (rh, rw)
    total_area = sum(piece["area"] for piece in pieces)
    if total_area != rh * rw:
        return []

    occ_mask = np.zeros(rect_shape, dtype=bool)
    occ_color = np.full(rect_shape, BG, dtype=int)
    placements: list[tuple[int, int]] = []
    solutions: list[np.ndarray] = []

    def dfs(step: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if step == len(order):
            if occ_mask.all():
                solutions.append(_render_rect(rect_shape, placements, pieces, order))
            return

        idx = order[step]
        piece = pieces[idx]
        for top, left in _candidate_positions(piece, step, rect_shape, corner, occ_mask, occ_color):
            patch = piece["array"]
            ph, pw = patch.shape
            occ_mask[top : top + ph, left : left + pw] = True
            occ_color[top : top + ph, left : left + pw] = patch
            placements.append((top, left))
            dfs(step + 1)
            placements.pop()
            occ_mask[top : top + ph, left : left + pw] = False
            occ_color[top : top + ph, left : left + pw] = BG

    dfs(0)
    return solutions


def _all_factor_pairs(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return sorted(out)


def _solve_unique(grid: np.ndarray) -> np.ndarray:
    corner, anchor = _find_marker(grid)
    pieces = _find_multicolor_components(grid, allowed={WILD, *PALETTE})
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLOCK)}
    area = sum(piece["area"] for piece in pieces)
    gh, gw = grid.shape

    outputs = []
    for rh, rw in _all_factor_pairs(area):
        tr, tc = _rect_top_left(anchor, corner, rh, rw)
        if tr < 0 or tc < 0 or tr + rh > gh or tc + rw > gw:
            continue
        rect_bbox = (tr, tc, tr + rh - 1, tc + rw - 1)
        order = sorted(
            range(len(pieces)),
            key=lambda idx: (_obstacle_score(pieces[idx]["bbox"], rect_bbox, obstacles), pieces[idx]["bbox"]),
        )
        sols = _solve_for_order(pieces, rect_bbox, corner, order, max_solutions=2)
        if len(sols) == 1:
            out = np.full_like(grid, BG)
            out[grid == MARK] = MARK
            out[tr : tr + rh, tc : tc + rw] = sols[0]
            outputs.append(out)

    unique = []
    for cand in outputs:
        if not any(np.array_equal(cand, prev) for prev in unique):
            unique.append(cand)
    assert len(unique) == 1, "expected a unique output"
    return unique[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered blue-heavy rectangles into the marked rectangle, with obstacle counts deciding the placement order.

    Input structure:
    - Black is the background; one magenta 3-cell L-marker marks a target rectangle corner by its missing cell.
    - Three or four scattered solid rectangular pieces are multicolor 4-connected components made of blue wildcard pixels
      plus a few pixels from {red, green, yellow, orange}.
    - One to three gray blocker pixels lie on the row-first Manhattan paths from piece centers toward the target center.

    Rule:
    - Use the marker gap as the target rectangle corner, and use the total area of all rectangular pieces to determine the
      target rectangle dimensions.
    - Order the pieces by the number of gray blockers on the row-first Manhattan path from each piece-bbox center to the
      target-bbox center; break ties by smaller Manhattan distance.
    - Starting from the marked corner, translate the pieces without rotation into the unique rectangle-growing placement
      order where touching edges are compatible and blue acts as a wildcard.
    - Paint only the assembled rectangle and keep the marker; erase the source pieces and all blockers to black.

    Key insight:
    The obstacle pixels do not move anything themselves; they only choose which otherwise swappable piece reaches the
    growing assembly earlier, and that order resolves the final rectangle arrangement.
    """
    g = np.asarray(grid, dtype=int)
    return _solve_unique(g)


def _pattern_column(h: int, row: int, color: int) -> np.ndarray:
    col = np.full(h, WILD, dtype=int)
    col[row] = color
    return col


def _make_piece(
    h: int,
    w: int,
    *,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    interior: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    piece = np.full((h, w), WILD, dtype=int)
    if left is not None:
        piece[:, 0] = left
    if right is not None:
        piece[:, -1] = right
    for r, c, color in interior or []:
        piece[r, c] = color
    nonblue = int(np.count_nonzero(piece != WILD))
    assert 2 <= nonblue <= 4
    return piece


def _interior_spots(h: int, w: int) -> list[tuple[int, int]]:
    spots = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            spots.append((r, c))
    if not spots:
        spots.append((h // 2, max(1, w // 2)))
    return spots


def _build_row_template() -> dict:
    four_pieces = random.random() < 0.5
    anchor_side = random.choice(["left", "right"])
    h = random.randint(3, 5)
    w_mid = random.randint(3, 5)
    w_a_choices = [w for w in (3, 4, 5) if w != w_mid]
    w_a = random.choice(w_a_choices)
    if four_pieces:
        w_d_choices = [w for w in (3, 4, 5) if w != w_mid]
        w_d = random.choice(w_d_choices)
    else:
        w_d = 0

    rows = list(range(h))
    random.shuffle(rows)
    r_l = rows[0]
    r_r = rows[1] if h >= 2 else rows[0]
    color_l, color_r, color_a, color_d = random.sample(PALETTE, 4)

    pat_l = _pattern_column(h, r_l, color_l)
    pat_r = _pattern_column(h, r_r, color_r)
    pat_a = _pattern_column(h, r_r, color_a)
    pat_d = _pattern_column(h, r_l, color_d)

    spots_a = _interior_spots(h, w_a)
    spots_mid = _interior_spots(h, w_mid)
    piece_a = _make_piece(
        h,
        w_a,
        left=pat_a,
        right=pat_l,
        interior=[(*random.choice(spots_a), random.choice([c for c in PALETTE if c not in {color_l, color_a}]))],
    )
    piece_x = _make_piece(
        h,
        w_mid,
        left=pat_l,
        right=pat_r,
        interior=[(*random.choice(spots_mid), color_a)],
    )
    piece_y = _make_piece(
        h,
        w_mid,
        left=pat_l,
        right=pat_r,
        interior=[(*random.choice(spots_mid), color_d)],
    )

    pieces = [piece_a, piece_x, piece_y]
    names = ["A", "X", "Y"]
    swap_pair = (1, 2)

    if four_pieces:
        spots_d = _interior_spots(h, w_d)
        piece_d = _make_piece(
            h,
            w_d,
            left=pat_r,
            right=pat_d,
            interior=[(*random.choice(spots_d), color_l)],
        )
        pieces.append(piece_d)
        names.append("D")

    if anchor_side == "right":
        pieces = [np.fliplr(piece) for piece in pieces]

    if not four_pieces:
        if anchor_side == "left":
            layout_order = [0, 1, 2] if random.random() < 0.5 else [0, 2, 1]
            placement_order = list(layout_order)
        else:
            layout_order = [1, 2, 0] if random.random() < 0.5 else [2, 1, 0]
            placement_order = list(reversed(layout_order))
    else:
        if anchor_side == "left":
            layout_order = [0, 1, 2, 3] if random.random() < 0.5 else [0, 2, 1, 3]
            placement_order = list(layout_order)
        else:
            layout_order = [3, 1, 2, 0] if random.random() < 0.5 else [3, 2, 1, 0]
            placement_order = list(reversed(layout_order))

    assembled = np.concatenate([pieces[idx] for idx in layout_order], axis=1)
    return {
        "pieces": pieces,
        "names": names,
        "layout_order": layout_order,
        "placement_order": placement_order,
        "swap_pair": swap_pair,
        "assembled": assembled,
        "anchor_side": anchor_side,
    }


def _marker_cells_for_bbox(rect_bbox: tuple[int, int, int, int], corner: str) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect_bbox
    if corner == "TL":
        anchor = (r0, c0)
        cells = [(r0 - 1, c0 - 1), (r0 - 1, c0), (r0, c0 - 1)]
    elif corner == "TR":
        anchor = (r0, c1)
        cells = [(r0 - 1, c1), (r0 - 1, c1 + 1), (r0, c1 + 1)]
    elif corner == "BL":
        anchor = (r1, c0)
        cells = [(r1, c0 - 1), (r1 + 1, c0 - 1), (r1 + 1, c0)]
    else:
        anchor = (r1, c1)
        cells = [(r1, c1 + 1), (r1 + 1, c1), (r1 + 1, c1 + 1)]
    return anchor, cells


def _place_source_pieces(
    shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    marker_cells: list[tuple[int, int]],
    pieces: list[np.ndarray],
) -> list[tuple[int, int]]:
    h, w = shape
    occ = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect_bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    for mr, mc in marker_cells:
        occ[max(0, mr - 1) : min(h, mr + 2), max(0, mc - 1) : min(w, mc + 2)] = True

    placements = []
    for patch in pieces:
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            rs = slice(max(0, top - 1), min(h, top + ph + 1))
            cs = slice(max(0, left - 1), min(w, left + pw + 1))
            if occ[rs, cs].any():
                continue
            occ[top : top + ph, left : left + pw] = True
            placements.append((top, left))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place a source piece")
    return placements


def _row_first_path_cells_for_bbox(
    piece_bbox: tuple[int, int, int, int],
    rect_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect_bbox
    sr, sc = _piece_center(piece_bbox)
    tr = (r0 + r1) // 2
    tc = (c0 + c1) // 2
    path = []
    for rr, cc in _row_first_path((sr, sc), (tr, tc)):
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            path.append((rr, cc))
    return path


def _find_obstacles_for_order(
    piece_bboxes: list[tuple[int, int, int, int]],
    rect_bbox: tuple[int, int, int, int],
    desired_order: list[int],
    blocked_budget: int,
    forbidden: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    paths = [
        [cell for cell in _row_first_path_cells_for_bbox(bbox, rect_bbox, shape) if cell not in forbidden]
        for bbox in piece_bboxes
    ]
    candidates = sorted(set(itertools.chain.from_iterable(paths)))
    if len(candidates) < blocked_budget:
        return None

    order_rank = {idx: pos for pos, idx in enumerate(desired_order)}
    dists = []
    for bbox in piece_bboxes:
        sr, sc = _piece_center(bbox)
        r0, c0, r1, c1 = rect_bbox
        tr = (r0 + r1) // 2
        tc = (c0 + c1) // 2
        dists.append(abs(sr - tr) + abs(sc - tc))

    for k in range(1, min(3, blocked_budget) + 1):
        combos = list(itertools.combinations(candidates, k))
        for combo in random.sample(combos, min(200, len(combos))):
            blocked = set(combo)
            scores = []
            for idx, path in enumerate(paths):
                scores.append((sum(cell in blocked for cell in path), dists[idx], idx))
            if [idx for _, _, idx in sorted(scores)] == desired_order and max(score[0] for score in scores) > 0:
                return list(combo)
    return None


def _build_expected_output(
    shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    assembled: np.ndarray,
    marker_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in marker_cells:
        out[r, c] = MARK
    r0, c0, r1, c1 = rect_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = assembled
    return out


def _piece_bboxes_from_placements(placements: list[tuple[int, int]], pieces: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    out = []
    for (top, left), patch in zip(placements, pieces):
        ph, pw = patch.shape
        out.append((top, left, top + ph - 1, left + pw - 1))
    return out


def _match_piece_indices(parsed: list[dict], original_pieces: list[np.ndarray]) -> list[int]:
    remaining = list(range(len(parsed)))
    matched = []
    for patch in original_pieces:
        found = None
        for idx in remaining:
            if np.array_equal(parsed[idx]["array"], patch):
                found = idx
                break
        assert found is not None
        remaining.remove(found)
        matched.append(found)
    return matched


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    parsed = _find_multicolor_components(inp, allowed={WILD, *PALETTE})
    mapped = _match_piece_indices(parsed, meta["pieces"])
    target_order = [mapped[idx] for idx in meta["placement_order"]]
    alt_order = list(target_order)
    s0 = meta["placement_order"].index(meta["swap_pair"][0])
    s1 = meta["placement_order"].index(meta["swap_pair"][1])
    alt_order[s0], alt_order[s1] = alt_order[s1], alt_order[s0]

    sols = _solve_for_order(parsed, meta["rect_bbox"], meta["corner"], target_order, max_solutions=2)
    if len(sols) != 1 or not np.array_equal(sols[0], meta["assembled"]):
        return False
    alt_sols = _solve_for_order(parsed, meta["rect_bbox"], meta["corner"], alt_order, max_solutions=2)
    if len(alt_sols) != 1:
        return False
    if np.array_equal(alt_sols[0], meta["assembled"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template = _build_row_template()
    assembled = template["assembled"]
    rh, rw = assembled.shape
    corner = random.choice(["TL", "BL"] if template["anchor_side"] == "left" else ["TR", "BR"])

    gh = random.randint(max(16, rh + 6), min(28, rh + 16))
    gw = random.randint(max(16, rw + 8), min(28, rw + 16))

    if corner == "TL":
        rect_bbox = (1, 1, rh, rw)
    elif corner == "TR":
        rect_bbox = (1, gw - rw - 1, rh, gw - 2)
    elif corner == "BL":
        rect_bbox = (gh - rh - 1, 1, gh - 2, rw)
    else:
        rect_bbox = (gh - rh - 1, gw - rw - 1, gh - 2, gw - 2)

    _, marker_cells = _marker_cells_for_bbox(rect_bbox, corner)
    if any(not (0 <= r < gh and 0 <= c < gw) for r, c in marker_cells):
        return None

    try:
        placements = _place_source_pieces((gh, gw), rect_bbox, marker_cells, template["pieces"])
    except RuntimeError:
        return None

    piece_bboxes = _piece_bboxes_from_placements(placements, template["pieces"])
    forbidden = set(marker_cells)
    r0, c0, r1, c1 = rect_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            forbidden.add((r, c))
    for bbox in piece_bboxes:
        br0, bc0, br1, bc1 = bbox
        for r in range(br0, br1 + 1):
            for c in range(bc0, bc1 + 1):
                forbidden.add((r, c))

    desired_order = template["placement_order"]
    blocked_budget = 3 if len(template["pieces"]) == 4 else 2
    obstacles = _find_obstacles_for_order(piece_bboxes, rect_bbox, desired_order, blocked_budget, forbidden, (gh, gw))
    if not obstacles:
        return None

    inp = np.full((gh, gw), BG, dtype=int)
    for r, c in marker_cells:
        inp[r, c] = MARK
    for (top, left), patch in zip(placements, template["pieces"]):
        _paste(inp, patch, top, left)
    for r, c in obstacles:
        inp[r, c] = BLOCK

    expected = _build_expected_output((gh, gw), rect_bbox, assembled, marker_cells)
    meta = {
        "pieces": template["pieces"],
        "placement_order": template["placement_order"],
        "swap_pair": template["swap_pair"],
        "assembled": assembled,
        "rect_bbox": rect_bbox,
        "corner": corner,
    }
    if not _validate_semantics(inp, expected, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
