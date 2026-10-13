from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 8
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _majority(values: np.ndarray) -> int:
    flat = [int(v) for v in values.ravel()]
    assert flat
    return Counter(flat).most_common(1)[0][0]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    best_key = None
    for pts in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(pts)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        if len(pts) != 2 * (hh + ww) - 4:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                border = r in (r0, r1) or c in (c0, c1)
                if border != bool(grid[r, c] == FRAME):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        key = (r1 + c1, hh * ww)
        if best is None or key > best_key:
            best = (r0, c0, r1, c1)
            best_key = key
    assert best is not None, "could not find grey frame"
    return best


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces: list[dict] = []
    for pts in _components((grid != BG) & (grid != FRAME)):
        r0, c0, r1, c1 = _bbox(pts)
        patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        hh, ww = patch.shape
        assert hh >= 3 and ww >= 3
        assert not np.any(patch == BG)
        interior = _majority(patch[1:-1, 1:-1])
        top = _majority(patch[0, 1:-1])
        bottom = _majority(patch[-1, 1:-1])
        left = _majority(patch[1:-1, 0])
        right = _majority(patch[1:-1, -1])
        row_sig = (hh, None if top == interior else top, None if bottom == interior else bottom)
        col_sig = (ww, None if left == interior else left, None if right == interior else right)
        pieces.append(
            {
                "patch": patch,
                "row_sig": row_sig,
                "col_sig": col_sig,
            }
        )
    assert pieces, "could not find any puzzle pieces"
    return pieces


def _order_tracks(signatures: list[tuple[int, int | None, int | None]]) -> list[tuple[int, int | None, int | None]]:
    if len(signatures) == 1:
        only = signatures[0]
        assert only[1] is None and only[2] is None
        return signatures

    starts = [i for i, sig in enumerate(signatures) if sig[1] is None]
    assert len(starts) == 1
    order = [starts[0]]
    used = {starts[0]}

    while len(order) < len(signatures):
        want = signatures[order[-1]][2]
        assert want is not None
        nxt = [i for i, sig in enumerate(signatures) if sig[1] == want and i not in used]
        assert len(nxt) == 1
        order.append(nxt[0])
        used.add(nxt[0])

    assert signatures[order[-1]][2] is None
    return [signatures[i] for i in order]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fr0, fc0, fr1, fc1 = _find_frame_bbox(g)
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1

    pieces = _extract_pieces(g)
    row_sigs = list(dict.fromkeys(piece["row_sig"] for piece in pieces))
    col_sigs = list(dict.fromkeys(piece["col_sig"] for piece in pieces))
    row_order = _order_tracks(row_sigs)
    col_order = _order_tracks(col_sigs)

    row_total = sum(sig[0] for sig in row_order)
    col_total = sum(sig[0] for sig in col_order)
    assert row_total == inner_h
    assert col_total == inner_w
    assert len(pieces) == len(row_order) * len(col_order)

    row_offsets: dict[tuple[int, int | None, int | None], int] = {}
    acc = 0
    for sig in row_order:
        row_offsets[sig] = acc
        acc += sig[0]

    col_offsets: dict[tuple[int, int | None, int | None], int] = {}
    acc = 0
    for sig in col_order:
        col_offsets[sig] = acc
        acc += sig[0]

    piece_map: dict[tuple[tuple[int, int | None, int | None], tuple[int, int | None, int | None]], np.ndarray] = {}
    for piece in pieces:
        key = (piece["row_sig"], piece["col_sig"])
        assert key not in piece_map
        piece_map[key] = piece["patch"]

    out = np.full((h, w), BG, dtype=int)
    _draw_frame(out, fr0, fc0, inner_h, inner_w)
    for row_sig in row_order:
        for col_sig in col_order:
            patch = piece_map[(row_sig, col_sig)]
            pr, pc = patch.shape
            rr = fr0 + 1 + row_offsets[row_sig]
            cc = fc0 + 1 + col_offsets[col_sig]
            out[rr : rr + pr, cc : cc + pc] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular fragments into the grey frame.

    Each fragment is a filled rectangle with optional colored side borders.
    Matching border colors indicate adjacency; sides whose border color equals
    the fragment fill color belong on the outer boundary next to the frame.
    """
    return _solve(grid)


def _paint_piece(hh: int, ww: int, interior: int, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    patch = np.full((hh, ww), interior, dtype=int)
    patch[0, :] = top
    patch[-1, :] = bottom
    patch[:, 0] = left
    patch[:, -1] = right
    return patch


def _can_place(occ: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r0 + hh > h or c0 + ww > w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + hh + 1)
    cc1 = min(w, c0 + ww + 1)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_pieces(
    rng: random.Random,
    h: int,
    w: int,
    frame_box: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_box
    occ[max(0, fr0 - 1) : min(h, fr1 + 2), max(0, fc0 - 1) : min(w, fc1 + 2)] = True

    order = list(range(len(patches)))
    rng.shuffle(order)
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

    placements: list[tuple[int, int] | None] = [None] * len(patches)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                if _can_place(occ, r0, c0, ph, pw):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements[idx] = (r0, c0)
        occ[r0 : r0 + ph, c0 : c0 + pw] = True

    return [(r, c) for r, c in placements if r is not None]


def _choose_layout(rng: random.Random) -> tuple[int, int]:
    kind = rng.choice(["row", "col", "block"])
    if kind == "row":
        return 1, rng.randint(2, 4)
    if kind == "col":
        return rng.randint(2, 4), 1
    return rng.randint(2, 3), rng.randint(2, 3)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = _choose_layout(rng)
    if rows * cols < 2:
        return None

    row_heights = [rng.randint(3, 6) for _ in range(rows)]
    col_widths = [rng.randint(3, 6) for _ in range(cols)]
    total_h = sum(row_heights)
    total_w = sum(col_widths)

    seam_count = (rows - 1) + (cols - 1)
    seam_colors = rng.sample(PALETTE, seam_count)
    h_seams = seam_colors[: rows - 1]
    v_seams = seam_colors[rows - 1 :]
    interior_palette = [color for color in PALETTE if color not in seam_colors]
    if not interior_palette:
        return None

    interiors: list[list[int]] = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            blocked = set()
            if r > 0:
                blocked.add(h_seams[r - 1])
            if r < rows - 1:
                blocked.add(h_seams[r])
            if c > 0:
                blocked.add(v_seams[c - 1])
            if c < cols - 1:
                blocked.add(v_seams[c])
            if r > 0:
                blocked.add(interiors[r - 1][c])
            if c > 0:
                blocked.add(interiors[r][c - 1])
            choices = [color for color in interior_palette if color not in blocked]
            if not choices:
                choices = [color for color in interior_palette if color not in blocked - set(interior_palette)]
            interiors[r][c] = rng.choice(choices)

    if rows * cols >= 3 and len({color for row in interiors for color in row}) < 2:
        return None

    patches: list[np.ndarray] = []
    expected_inner = np.full((total_h, total_w), BG, dtype=int)
    row_sig_list: list[tuple[int, int | None, int | None]] = []
    col_sig_list: list[tuple[int, int | None, int | None]] = []

    row_offset = 0
    for r in range(rows):
        col_offset = 0
        for c in range(cols):
            interior = interiors[r][c]
            top = interior if r == 0 else h_seams[r - 1]
            bottom = interior if r == rows - 1 else h_seams[r]
            left = interior if c == 0 else v_seams[c - 1]
            right = interior if c == cols - 1 else v_seams[c]
            patch = _paint_piece(row_heights[r], col_widths[c], interior, top, bottom, left, right)
            patches.append(patch)
            expected_inner[
                row_offset : row_offset + row_heights[r],
                col_offset : col_offset + col_widths[c],
            ] = patch
            row_sig_list.append((row_heights[r], None if top == interior else top, None if bottom == interior else bottom))
            col_sig_list.append((col_widths[c], None if left == interior else left, None if right == interior else right))
            col_offset += col_widths[c]
        row_offset += row_heights[r]

    frame_outer_h = total_h + 2
    frame_outer_w = total_w + 2
    extra_h = max(10, rows * 5 + 6)
    extra_w = max(10, cols * 5 + 6)
    grid_h = frame_outer_h + extra_h + rng.randint(0, 5)
    grid_w = frame_outer_w + extra_w + rng.randint(0, 5)

    bottom_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    fr0 = grid_h - frame_outer_h - bottom_margin
    fc0 = grid_w - frame_outer_w - right_margin
    frame_box = (fr0, fc0, fr0 + frame_outer_h - 1, fc0 + frame_outer_w - 1)

    placements = _place_pieces(rng, grid_h, grid_w, frame_box, patches)
    if placements is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(inp, fr0, fc0, total_h, total_w)
    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = patch.shape
        inp[r0 : r0 + ph, c0 : c0 + pw] = patch

    expected = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(expected, fr0, fc0, total_h, total_w)
    expected[fr0 + 1 : fr0 + 1 + total_h, fc0 + 1 : fc0 + 1 + total_w] = expected_inner

    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("generate_input: exhausted retries")
