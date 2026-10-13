from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
LAYOUTS = [(2, 2), (2, 3), (3, 2)]
DIM_CHOICES = [3, 4, 5]
CANVAS_H_RANGE = (20, 30)
CANVAS_W_RANGE = (20, 30)
OUTER_MARGIN = 1
PLACE_TRIES = 300
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _rotations(mask: np.ndarray) -> list[dict]:
    rots: list[dict] = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        rot = _crop_mask(rot)
        key = rot.tobytes()
        if key in seen:
            continue
        seen.add(key)
        coords = [(int(r), int(c)) for r, c in np.argwhere(rot)]
        rots.append(
            {
                "mask": rot,
                "coords": coords,
                "shape": rot.shape,
                "area": len(coords),
            }
        )
    return rots


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    best: tuple[int, float, int, tuple[int, int, int, int]] | None = None
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components(grid == color):
            if len(comp) < 8:
                continue
            r0, c0, r1, c1 = _bbox_from_coords(comp)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if bh < 3 or bw < 3:
                continue
            if len(comp) != 2 * (bh + bw) - 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((bh, bw), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if np.any(sub[1:-1, 1:-1] == color):
                continue
            area = (bh - 2) * (bw - 2)
            center_penalty = abs((r0 + r1) / 2 - (h - 1) / 2) + abs((c0 + c1) / 2 - (w - 1) / 2)
            cand = (area, -center_penalty, color, (r0, c0, r1, c1))
            if best is None or cand > best:
                best = cand
    assert best is not None, "expected a unique rectangular frame"
    return int(best[2]), best[3]


def _extract_pieces(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    mask = (grid != bg) & (grid != frame_color)
    pieces: list[dict] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "every piece must be monochrome"
        pr0, pc0, pr1, pc1 = _bbox_from_coords(comp)
        assert pr1 < r0 or pr0 > r1 or pc1 < c0 or pc0 > c1, "pieces must lie outside the frame"
        local = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
        for rr, cc in comp:
            local[rr - pr0, cc - pc0] = True
        pieces.append(
            {
                "color": next(iter(colors)),
                "mask": local,
                "area": len(comp),
                "rots": _rotations(local),
            }
        )
    pieces.sort(key=lambda piece: (-piece["area"], piece["color"]))
    return pieces


def _solve_exact_cover(
    pieces: list[dict], shape: tuple[int, int], max_solutions: int = 2
) -> list[list[tuple[int, int, int]]]:
    ih, iw = shape
    total_area = sum(int(piece["area"]) for piece in pieces)
    if total_area != ih * iw:
        return []

    order = list(range(len(pieces)))
    order.sort(key=lambda i: (-pieces[i]["area"], len(pieces[i]["rots"])))
    ordered = [pieces[i] for i in order]

    occ = np.zeros((ih, iw), dtype=bool)
    used = [False] * len(ordered)
    placements: list[tuple[int, int, int] | None] = [None] * len(ordered)
    solutions: list[list[tuple[int, int, int]]] = []

    def dfs(depth: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if depth == len(ordered):
            if np.all(occ):
                sol = [(-1, -1, -1)] * len(pieces)
                for idx, place in enumerate(placements):
                    assert place is not None
                    sol[order[idx]] = place
                solutions.append(sol)
            return

        flat = np.flatnonzero(~occ.ravel())
        if flat.size == 0:
            return
        anchor = int(flat[0])
        ar, ac = divmod(anchor, iw)

        for piece_idx, piece in enumerate(ordered):
            if used[piece_idx]:
                continue
            used[piece_idx] = True
            for rot_idx, rot in enumerate(piece["rots"]):
                tried_positions: set[tuple[int, int]] = set()
                rh, rw = rot["shape"]
                for rr, cc in rot["coords"]:
                    top = ar - rr
                    left = ac - cc
                    if (top, left) in tried_positions:
                        continue
                    tried_positions.add((top, left))
                    if top < 0 or left < 0 or top + rh > ih or left + rw > iw:
                        continue
                    ok = True
                    for pr, pc in rot["coords"]:
                        if occ[top + pr, left + pc]:
                            ok = False
                            break
                    if not ok:
                        continue
                    for pr, pc in rot["coords"]:
                        occ[top + pr, left + pc] = True
                    placements[piece_idx] = (rot_idx, top, left)
                    dfs(depth + 1)
                    placements[piece_idx] = None
                    for pr, pc in rot["coords"]:
                        occ[top + pr, left + pc] = False
                    if len(solutions) >= max_solutions:
                        used[piece_idx] = False
                        return
            used[piece_idx] = False

    dfs(0)
    return solutions


def _render_solution(
    grid_shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    pieces: list[dict],
    solution: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color

    ir0, ic0 = r0 + 1, c0 + 1
    for piece, (rot_idx, top, left) in zip(pieces, solution):
        rot = piece["rots"][rot_idx]
        for rr, cc in rot["coords"]:
            out[ir0 + top + rr, ic0 + left + cc] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw-like pieces into the empty slots of a central frame.

    Input structure:
    - bg = background color. A large one-pixel-thick rectangular frame sits near the center.
    - The frame interior is an empty rectangle whose hidden partition consists of interlocking slot shapes.
    - Outside the frame are several monochrome connected pieces with tooth-like tabs and notches.

    Rule:
    - Detect the unique rectangular frame and treat its interior as the target packing region.
    - Extract every outside piece, allowing only 0°/90°/180°/270° rotations.
    - Place the rotated pieces so they exactly tile the frame interior with no overlap or gaps, then erase the original scattered copies.

    Key insight: The tabs and notches act as a shape key, so only one rotation and placement arrangement exactly covers the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    pieces = _extract_pieces(g, bg, frame_color, frame_bbox)
    assert pieces, "expected at least one outside piece"

    r0, c0, r1, c1 = frame_bbox
    interior_shape = (r1 - r0 - 1, c1 - c0 - 1)
    assert interior_shape[0] > 0 and interior_shape[1] > 0, "frame must have non-empty interior"

    solutions = _solve_exact_cover(pieces, interior_shape, max_solutions=2)
    assert len(solutions) == 1, "expected a unique tiling of the frame interior"
    return _render_solution(g.shape, bg, frame_color, frame_bbox, pieces, solutions[0])


def _sample_dims(count: int, rng: random.Random) -> list[int]:
    if count == 2:
        dims = rng.sample(DIM_CHOICES, 2)
        rng.shuffle(dims)
        return dims
    dims = DIM_CHOICES[:]
    rng.shuffle(dims)
    return dims


def _segment_start(span: int, seg_len: int, idx: int, total: int) -> int:
    low = 1
    high = span - seg_len - 1
    assert low <= high
    if total <= 1 or low == high:
        return low
    if total == 2:
        return low if idx == 0 else high
    middle = max(low, min(high, (span - seg_len) // 2))
    anchors = [low, middle, high]
    return anchors[idx]


def _segment_len(span: int, idx: int) -> int:
    if span <= 3:
        return 1
    if span == 4:
        return 1 + int(idx % 2 == 1)
    return 1 + int(idx % 2 == 0)


def _owner_partition(rows: int, cols: int, rng: random.Random) -> np.ndarray | None:
    heights = _sample_dims(rows, rng)
    widths = _sample_dims(cols, rng)
    h = sum(heights)
    w = sum(widths)

    owner = np.zeros((h, w), dtype=int)
    cell_ids = np.arange(rows * cols, dtype=int).reshape(rows, cols)
    row_starts = [0]
    col_starts = [0]
    for value in heights[:-1]:
        row_starts.append(row_starts[-1] + value)
    for value in widths[:-1]:
        col_starts.append(col_starts[-1] + value)

    for r in range(rows):
        for c in range(cols):
            rr = row_starts[r]
            cc = col_starts[c]
            owner[rr : rr + heights[r], cc : cc + widths[c]] = int(cell_ids[r, c])

    for r in range(rows):
        for c in range(cols - 1):
            rr = row_starts[r]
            cc = col_starts[c] + widths[c] - 1
            seg_len = _segment_len(heights[r], c)
            seg_top = rr + _segment_start(heights[r], seg_len, r, rows)
            left_id = int(cell_ids[r, c])
            right_id = int(cell_ids[r, c + 1])
            if (r + c) % 2 == 0:
                owner[seg_top : seg_top + seg_len, cc + 1] = left_id
            else:
                owner[seg_top : seg_top + seg_len, cc] = right_id

    for r in range(rows - 1):
        for c in range(cols):
            rr = row_starts[r] + heights[r] - 1
            cc = col_starts[c]
            seg_len = _segment_len(widths[c], r)
            seg_left = cc + _segment_start(widths[c], seg_len, c, cols)
            top_id = int(cell_ids[r, c])
            bot_id = int(cell_ids[r + 1, c])
            if (r + c) % 2 == 1:
                owner[rr + 1, seg_left : seg_left + seg_len] = top_id
            else:
                owner[rr, seg_left : seg_left + seg_len] = bot_id

    irregular = 0
    for piece_id in range(rows * cols):
        mask = owner == piece_id
        if not _is_connected(mask):
            return None
        cropped = _crop_mask(mask)
        if int(cropped.sum()) != cropped.shape[0] * cropped.shape[1]:
            irregular += 1

    if irregular != rows * cols:
        return None
    return owner


def _make_piece_specs(owner: np.ndarray) -> list[dict]:
    piece_count = int(owner.max()) + 1
    pieces: list[dict] = []
    for piece_id in range(piece_count):
        mask = owner == piece_id
        cropped = _crop_mask(mask)
        pieces.append(
            {
                "id": piece_id,
                "mask": cropped,
                "area": int(cropped.sum()),
                "rots": _rotations(cropped),
            }
        )
    pieces.sort(key=lambda piece: (-piece["area"], piece["id"]))
    return pieces


def _frame_positions(canvas: tuple[int, int], frame_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = canvas
    fh, fw = frame_shape
    center_r = (h - fh) // 2
    center_c = (w - fw) // 2
    positions: list[tuple[int, int]] = []
    for dr in [-2, -1, 0, 1, 2]:
        for dc in [-2, -1, 0, 1, 2]:
            r0 = center_r + dr
            c0 = center_c + dc
            if OUTER_MARGIN <= r0 and OUTER_MARGIN <= c0 and r0 + fh <= h - OUTER_MARGIN and c0 + fw <= w - OUTER_MARGIN:
                positions.append((r0, c0))
    return positions


def _can_place(coords: list[tuple[int, int]], occupied: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    for rr, cc in coords:
        r = top + rr
        c = left + cc
        if not (OUTER_MARGIN <= r < h - OUTER_MARGIN and OUTER_MARGIN <= c < w - OUTER_MARGIN):
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _scatter_pieces(
    canvas_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    piece_payloads: list[tuple[np.ndarray, int]],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = canvas_shape
    grid = np.full((h, w), -1, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    fr0, fc0, fr1, fc1 = frame_bbox
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    order = list(range(len(piece_payloads)))
    order.sort(key=lambda i: -int(piece_payloads[i][0].sum()))
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        mask, _color = piece_payloads[idx]
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        ph, pw = mask.shape
        placed = False
        for _ in range(PLACE_TRIES):
            top = rng.randint(OUTER_MARGIN, h - ph - OUTER_MARGIN)
            left = rng.randint(OUTER_MARGIN, w - pw - OUTER_MARGIN)
            if not _can_place(coords, occupied, top, left):
                continue
            placements[idx] = (top, left)
            for rr, cc in coords:
                occupied[top + rr, left + cc] = True
            placed = True
            break
        if not placed:
            return None

    for idx, (mask, color) in enumerate(piece_payloads):
        top, left = placements[idx]
        for rr, cc in np.argwhere(mask):
            grid[top + int(rr), left + int(cc)] = color
    return grid


def _build_task(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = rng.choice(LAYOUTS)
    owner = _owner_partition(rows, cols, rng)
    if owner is None:
        return None

    pieces = _make_piece_specs(owner)
    solutions = _solve_exact_cover(pieces, owner.shape, max_solutions=2)
    if len(solutions) != 1:
        return None

    canvas_h = rng.randint(*CANVAS_H_RANGE)
    canvas_w = rng.randint(*CANVAS_W_RANGE)
    inner_h, inner_w = owner.shape
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    frame_pos = _frame_positions((canvas_h, canvas_w), (frame_h, frame_w))
    if not frame_pos:
        return None
    fr0, fc0 = rng.choice(frame_pos)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    color_pool = list(range(10))
    bg = rng.choice(color_pool)
    color_pool.remove(bg)
    frame_color = rng.choice(color_pool)
    color_pool.remove(frame_color)
    if len(color_pool) < len(pieces):
        return None
    piece_colors = rng.sample(color_pool, len(pieces))

    piece_payloads: list[tuple[np.ndarray, int]] = []
    for piece, color in zip(pieces, piece_colors):
        rot_idx = rng.randrange(4)
        rot = np.rot90(piece["mask"], rot_idx)
        rot = _crop_mask(rot)
        piece_payloads.append((rot, color))

    scattered = _scatter_pieces((canvas_h, canvas_w), frame_bbox, piece_payloads, rng)
    if scattered is None:
        return None

    inp = np.full((canvas_h, canvas_w), bg, dtype=int)
    inp[scattered >= 0] = scattered[scattered >= 0]
    fr0, fc0, fr1, fc1 = frame_bbox
    inp[fr0 : fr1 + 1, fc0] = frame_color
    inp[fr0 : fr1 + 1, fc1] = frame_color
    inp[fr0, fc0 : fc1 + 1] = frame_color
    inp[fr1, fc0 : fc1 + 1] = frame_color

    extracted = _extract_pieces(inp, bg, frame_color, frame_bbox)
    extracted_solutions = _solve_exact_cover(extracted, owner.shape, max_solutions=2)
    if len(extracted_solutions) != 1:
        return None
    expected = _render_solution(inp.shape, bg, frame_color, frame_bbox, extracted, extracted_solutions[0])
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _build_task(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid jigsaw-frame input")
