import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
BODY_COLORS = (BLUE, GREEN, ORANGE)

GEN_RETRIES = 400
ASSEMBLY_SOLUTION_CAP = 2


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
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
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords):
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_to_coords(grid, coords):
    r0, c0, r1, c1 = _bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _extract_pieces(grid):
    pieces = []
    for coords in _components(grid != BG):
        arr, bbox = _crop_to_coords(grid, coords)
        nz = np.argwhere(arr != BG)
        reds = [tuple(map(int, x)) for x in np.argwhere(arr == RED)]
        pieces.append(
            {
                "arr": arr,
                "bbox": bbox,
                "shape": arr.shape,
                "nz": nz,
                "reds": reds,
                "area": int(len(nz)),
            }
        )
    return pieces


def _has_anchor_block(arr):
    h, w = arr.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(arr[r : r + 3, c : c + 3] == GREY):
                return True
    return False


def _try_place(canvas, arr, top, left):
    h, w = arr.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return None
    out = canvas.copy()
    for r, c in np.argwhere(arr != BG):
        rr = top + int(r)
        cc = left + int(c)
        src = int(arr[r, c])
        dst = int(out[rr, cc])
        if dst == BG:
            out[rr, cc] = src
        elif dst == RED and src == RED:
            continue
        else:
            return None
    return out


def _assemble_unique(grid, limit=ASSEMBLY_SOLUTION_CAP):
    pieces = _extract_pieces(grid)
    assert pieces, "expected at least one piece"

    anchors = [p for p in pieces if _has_anchor_block(p["arr"])]
    assert len(anchors) == 1, "expected exactly one anchor piece"
    anchor = anchors[0]
    remaining = [p for p in pieces if p is not anchor]

    h, w = grid.shape
    ar0, ac0, _ar1, _ac1 = anchor["bbox"]
    canvas = np.full((h, w), BG, dtype=int)
    placed = _try_place(canvas, anchor["arr"], ar0, ac0)
    assert placed is not None, "anchor placement failed"

    total_reds = sum(len(p["reds"]) for p in pieces)
    assert total_reds % 2 == 0 and total_reds > 0
    target_unique_red = total_reds // 2

    red_counts = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions = {}

    def dfs(cur, rem, rc):
        if len(solutions) >= limit:
            return
        if not rem:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions[cur.tobytes()] = cur.copy()
            return
        if len(rc) > target_unique_red:
            return

        unpaired = [coord for coord, cnt in rc.items() if cnt == 1]
        if not unpaired:
            return
        unpaired_set = set(unpaired)

        best_moves = None
        for tr, tc in unpaired:
            moves = []
            for idx, piece in enumerate(rem):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    overlap = 0
                    bad = False
                    for rr, cc in piece["reds"]:
                        key = (top + rr, left + cc)
                        prev = rc.get(key, 0)
                        if prev >= 2:
                            bad = True
                            break
                        if key in unpaired_set:
                            overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        dedup = {}
        assert best_moves is not None
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves.sort(key=lambda x: (-x[3], -rem[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _overlap in moves:
            piece = rem[idx]
            nxt = _try_place(cur, piece["arr"], top, left)
            if nxt is None:
                continue
            next_rc = dict(rc)
            for rr, cc in piece["reds"]:
                key = (top + rr, left + cc)
                next_rc[key] = next_rc.get(key, 0) + 1
            dfs(nxt, rem[:idx] + rem[idx + 1 :], next_rc)
            if len(solutions) >= limit:
                return

    dfs(placed, remaining, red_counts)
    return list(solutions.values())


def _split_and_shift(assembled):
    yellow = np.argwhere(assembled == YELLOW)
    assert len(yellow) > 0, "expected a crack path"
    r0, c0 = yellow.min(axis=0)
    r1, c1 = yellow.max(axis=0)
    horizontal = (c1 - c0) >= (r1 - r0)

    residual = assembled.copy()
    residual[(residual == YELLOW) | (residual == RED)] = BG
    comps = _components(residual != BG)
    assert len(comps) == 2, "erasing the crack must produce exactly two halves"

    out = np.full_like(assembled, BG)
    if horizontal:
        comps.sort(key=lambda coords: float(coords[:, 0].mean()))
        shifts = [(-1, 0), (1, 0)]
    else:
        comps.sort(key=lambda coords: float(coords[:, 1].mean()))
        shifts = [(0, -1), (0, 1)]

    for coords, (dr, dc) in zip(comps, shifts):
        for r, c in coords:
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            color = int(residual[r, c])
            assert out[rr, cc] in (BG, color)
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered connector-marked sprites, erase the emergent crack path, and pull the two sides one cell apart.

    Input structure:
    - A black background contains 4-6 scattered 4-connected sprites made from blue/green/orange body pixels plus red connector pixels and yellow internal crack markers.
    - Exactly one sprite also contains a solid 3x3 grey anchor block; that anchor sprite stays fixed while the others are translated.
    - Each red connector pixel belongs to exactly one matching pair, and after correct assembly all yellow pixels form one continuous crack path through the whole object.

    Rule:
    - Keep the grey-anchor sprite in place and assemble every other sprite by translating it so one or more of its red pixels overlap currently unpaired red pixels, with no non-red overlap.
    - After the full assembly is reconstructed, erase every yellow crack pixel and every red connector pixel.
    - The erased yellow path splits the remaining object into exactly two halves; if the crack is mainly horizontal move the upper half up and the lower half down by 1, otherwise move the left half left and the right half right by 1.

    Key insight:
    - The yellow pixels are meaningless while the sprites are scattered; only after red-guided assembly do they emerge as one complete crack line that determines where the final split happens.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    solutions = _assemble_unique(g, limit=ASSEMBLY_SOLUTION_CAP)
    assert len(solutions) == 1, "expected a unique assembly"
    return _split_and_shift(solutions[0])


def _is_connected_cells(cells):
    if not cells:
        return False
    coords = np.array(list(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return len(_components(mask)) == 1


def _cells_to_array(cells):
    coords = np.array(list(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in cells.items():
        arr[r - r0, c - c0] = color
    return arr, (r0, c0, r1, c1)


def _transpose_cells(cells):
    return {(c, r): color for (r, c), color in cells.items()}


def _paint_cells(shape, piece_cells):
    grid = np.zeros(shape, dtype=int)
    for cells in piece_cells:
        for (r, c), color in cells.items():
            prev = int(grid[r, c])
            if prev == BG:
                grid[r, c] = color
            else:
                assert prev == RED and color == RED
    return grid


def _bbox_sep(a0, a1, b0, b1):
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return -1


def _bboxes_far_enough(b1, b2, gap):
    rsep = _bbox_sep(b1[0], b1[2], b2[0], b2[2])
    csep = _bbox_sep(b1[1], b1[3], b2[1], b2[3])
    return max(rsep, csep) >= gap


def _shift_bbox(bbox, top, left):
    r0, c0, r1, c1 = bbox
    return top + r0, left + c0, top + r1, left + c1


def _paste_no_overlap(grid, arr, top, left):
    h, w = arr.shape
    sub = grid[top : top + h, left : left + w]
    mask = arr != BG
    if np.any(sub[mask] != BG):
        return False
    sub[mask] = arr[mask]
    return True


def _random_lengths(piece_count):
    for _ in range(100):
        lengths = [random.choice((1, 2)) for _ in range(piece_count)]
        if lengths.count(2) >= 2 and 6 <= sum(lengths) <= 9:
            return lengths
    return None


def _make_local_piece_cells():
    piece_count = random.randint(4, 6)
    lengths = _random_lengths(piece_count)
    if lengths is None:
        return None

    top_h, bot_h = random.choice(((3, 2), (2, 3)))
    seam_r = top_h
    total_w = sum(lengths)
    starts = []
    x = 0
    for ln in lengths:
        starts.append(x)
        x += ln

    anchor_options = []
    for idx, ln in enumerate(lengths):
        if ln != 2:
            continue
        if idx > 0 and lengths[idx - 1] == 2:
            anchor_options.append((idx, -1))
        if idx + 1 < piece_count and lengths[idx + 1] == 2:
            anchor_options.append((idx, 1))
    if not anchor_options:
        return None
    anchor_idx, extend_dir = random.choice(anchor_options)
    neighbor_idx = anchor_idx + extend_dir

    piece_cells = []
    colors = [random.choice(BODY_COLORS) for _ in range(piece_count)]
    for idx, ln in enumerate(lengths):
        a = starts[idx]
        cells = {}
        for c in range(a, a + ln):
            for r in range(top_h + 1 + bot_h):
                if r == seam_r:
                    cells[(r, c)] = YELLOW
                else:
                    cells[(r, c)] = colors[idx]
        piece_cells.append(cells)

    if top_h == 3:
        grey_rows = range(0, 3)
    else:
        grey_rows = range(top_h + 1, top_h + 1 + 3)

    a = starts[anchor_idx]
    extra_col = a - 1 if extend_dir < 0 else a + lengths[anchor_idx]
    grey_cols = sorted([a, a + 1, extra_col])
    for r in grey_rows:
        for c in grey_cols:
            piece_cells[anchor_idx][(r, c)] = GREY
    for r in grey_rows:
        piece_cells[neighbor_idx].pop((r, extra_col), None)

    for idx in range(piece_count - 1):
        left_end = starts[idx] + lengths[idx] - 1
        right_col = left_end + 1
        candidates = []
        for r in range(top_h + 1 + bot_h):
            if r == seam_r:
                continue
            left_ok = piece_cells[idx].get((r, left_end), BG) in BODY_COLORS
            right_ok = piece_cells[idx + 1].get((r, right_col), BG) in BODY_COLORS
            if left_ok and right_ok:
                candidates.append(r)
        if not candidates:
            return None
        rr = random.choice(candidates)
        piece_cells[idx][(rr, right_col)] = RED
        piece_cells[idx + 1][(rr, right_col)] = RED

    for idx, cells in enumerate(piece_cells):
        colors_present = list(cells.values())
        red_count = colors_present.count(RED)
        yellow_count = colors_present.count(YELLOW)
        assert red_count in (1, 2)
        assert yellow_count in (1, 2)
        if not _is_connected_cells(set(cells)):
            return None
        body_count = sum(color in BODY_COLORS or color == GREY for color in colors_present)
        if idx == anchor_idx:
            if body_count < 9:
                return None
        elif not (5 <= body_count <= 10):
            return None

    assembled = _paint_cells((top_h + 1 + bot_h, total_w), piece_cells)
    if len(_components(assembled == YELLOW)) != 1:
        return None
    residual = assembled.copy()
    residual[(residual == RED) | (residual == YELLOW)] = BG
    if len(_components(residual != BG)) != 2:
        return None

    horizontal = random.choice((True, False))
    if not horizontal:
        piece_cells = [_transpose_cells(cells) for cells in piece_cells]
        assembled = assembled.T.copy()
    return piece_cells, assembled, anchor_idx


def _scatter_input(piece_cells, assembled, anchor_idx):
    local_arrays = []
    local_bboxes = []
    for cells in piece_cells:
        arr, bbox = _cells_to_array(cells)
        local_arrays.append(arr)
        local_bboxes.append(bbox)

    ah, aw = assembled.shape
    grid_h = random.randint(max(18, ah + 6), 24)
    grid_w = random.randint(max(18, aw + 6), 24)

    assembly_top = random.randint(2, grid_h - ah - 2)
    assembly_left = random.randint(2, grid_w - aw - 2)

    placed = {}
    anchor_bbox = _shift_bbox(local_bboxes[anchor_idx], assembly_top, assembly_left)
    placed[anchor_idx] = (
        assembly_top + local_bboxes[anchor_idx][0],
        assembly_left + local_bboxes[anchor_idx][1],
    )
    placed_bboxes = [anchor_bbox]

    for idx, arr in enumerate(local_arrays):
        if idx == anchor_idx:
            continue
        h, w = arr.shape
        assembled_piece_bbox = _shift_bbox(local_bboxes[idx], assembly_top, assembly_left)
        ok = False
        for _ in range(200):
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            bbox = (top, left, top + h - 1, left + w - 1)
            if not all(_bboxes_far_enough(bbox, other, 2) for other in placed_bboxes):
                continue
            if not _bboxes_far_enough(bbox, assembled_piece_bbox, 2):
                continue
            placed[idx] = (top, left)
            placed_bboxes.append(bbox)
            ok = True
            break
        if not ok:
            return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, arr in enumerate(local_arrays):
        top, left = placed[idx]
        if not _paste_no_overlap(grid, arr, top, left):
            return None

    expected = np.zeros_like(grid)
    if not _paste_no_overlap(expected, assembled, assembly_top, assembly_left):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        made = _make_local_piece_cells()
        if made is None:
            continue
        piece_cells, assembled, anchor_idx = made
        scattered = _scatter_input(piece_cells, assembled, anchor_idx)
        if scattered is None:
            continue
        grid, expected_assembled = scattered
        try:
            solutions = _assemble_unique(grid, limit=ASSEMBLY_SOLUTION_CAP)
            if len(solutions) != 1:
                continue
            if not np.array_equal(solutions[0], expected_assembled):
                continue
            out = _split_and_shift(solutions[0])
        except AssertionError:
            continue
        if out.shape != grid.shape or max(out.shape) > 30 or max(grid.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
