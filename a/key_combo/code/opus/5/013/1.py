from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
RAY = 5
CROSS = 7
PIECE_COLORS = [1, 3, 4, 6]
EXTRA_FAMILY3_COLOR = 8


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    comps = _components_mask(grid != BG)
    pieces: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == RED)
        piece = {
            "bbox": (r0, c0, r1, c1),
            "arr": arr,
            "shape": arr.shape,
            "reds": [(int(r), int(c)) for r, c in reds],
            "nonred_count": int(np.count_nonzero((arr != BG) & (arr != RED))),
            "size": int(np.count_nonzero(arr != BG)),
        }
        pieces.append(piece)
    return pieces


def _try_place(canvas: np.ndarray, arr: np.ndarray, top: int, left: int) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None
    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != BG
    conflict = mask & (sub != BG) & ~((arr == RED) & (sub == RED))
    if np.any(conflict):
        return None
    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _assemble_unique(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pieces = _extract_pieces(g)
    assert 3 <= len(pieces) <= 5

    counts = [p["nonred_count"] for p in pieces]
    best = max(counts)
    assert counts.count(best) == 1
    anchor_idx = counts.index(best)
    anchor = pieces[anchor_idx]
    rest = [p for i, p in enumerate(pieces) if i != anchor_idx]

    canvas = np.zeros((h, w), dtype=int)
    ar0, ac0, _ar1, _ac1 = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0)
    assert placed is not None

    total_ports = sum(len(p["reds"]) for p in pieces)
    assert total_ports % 2 == 0
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return
        if len(rc) > target_unique_red:
            return

        unpaired = [coord for coord, n in rc.items() if n == 1]
        if not unpaired:
            return

        best_moves = None
        for tr, tc in unpaired:
            moves = []
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc in piece["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if prev == 1:
                            overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves

        assert best_moves is not None
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [(idx, top, left, overlap) for (idx, top, left), overlap in dedup.items()]
        moves2.sort(key=lambda item: (-item[3], -remaining[item[0]]["nonred_count"], item[1], item[2]))

        for idx, top, left, _overlap in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left)
            if nxt is None:
                continue
            next_rc = dict(rc)
            for rr, cc in piece["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1
    return solutions[0]


def _find_holes(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    holes: list[np.ndarray] = []
    for comp in _components_mask(~mask):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            continue
        holes.append(comp)
    return holes


def _hole_direction(hole: np.ndarray, mask: np.ndarray) -> str:
    all_fg = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(all_fg)
    hr0, hc0, hr1, hc1 = _bbox(hole)
    dists = {
        "up": hr0 - r0,
        "down": r1 - hr1,
        "left": hc0 - c0,
        "right": c1 - hc1,
    }
    best = min(dists.values())
    winners = [name for name, dist in dists.items() if dist == best]
    assert len(winners) == 1
    return winners[0]


def _paint_ray_pixel(out: np.ndarray, mask: np.ndarray, r: int, c: int) -> None:
    if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
        return
    if mask[r, c]:
        return
    if out[r, c] == BG:
        out[r, c] = RAY
    elif out[r, c] == RAY:
        out[r, c] = CROSS


def _emit_rays(assembled: np.ndarray) -> np.ndarray:
    out = assembled.copy()
    mask = assembled != BG
    holes = _find_holes(mask)
    for hole in holes:
        direction = _hole_direction(hole, mask)
        if direction in ("left", "right"):
            rows = sorted({int(r) for r, _c in hole})
            for r in rows:
                for c in range(out.shape[1]):
                    _paint_ray_pixel(out, mask, r, c)
        else:
            cols = sorted({int(c) for _r, c in hole})
            for c in cols:
                for r in range(out.shape[0]):
                    _paint_ray_pixel(out, mask, r, c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered connector-marked sprites, detect emergent enclosed holes, and cast rays from their rows or columns.

    Input structure:
    - The grid uses black (0) as background and contains 3 to 5 scattered 4-connected sprites.
    - Each sprite has one body color, plus 1 or 2 red (2) connector pixels on its edge.
    - The sprite with the most non-red body pixels is the anchor; the other sprites are detached fragments with no holes on their own.

    Rule:
    - Keep the anchor fixed and translate the remaining sprites until every red connector overlaps exactly one red connector from another sprite, yielding one unique assembled composite.
    - In that composite, find every enclosed 4-connected black hole.
    - For each hole, use the nearest outer side of the composite bbox to decide whether it emits horizontal rays (left/right case, one per hole row) or vertical rays (up/down case, one per hole column).
    - Paint the ray lines grey (5) on background cells only, leaving the assembled non-background pixels unchanged; if two rays hit the same background pixel, recolor it orange (7).

    Key insight:
    The holes do not exist in any single input sprite; only the red-connector assembly creates the enclosed regions that then determine which ray rows or columns appear.
    """
    assembled = _assemble_unique(np.asarray(grid, dtype=int))
    return _emit_rays(assembled)


BASE_FAMILIES = [
    {
        "holes": 1,
        "pieces": [
            {
                "reds": {(0, 4), (2, 3)},
                "cells": {(0, 2), (0, 3), (0, 4), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)},
            },
            {
                "reds": {(0, 4), (2, 4)},
                "cells": {(0, 4), (0, 5), (1, 5), (2, 4), (2, 5), (2, 6), (2, 7)},
            },
            {
                "reds": {(2, 3), (2, 4)},
                "cells": {(2, 3), (2, 4), (3, 3), (4, 3)},
            },
        ],
    },
    {
        "holes": 2,
        "pieces": [
            {
                "reds": {(2, 5), (3, 3)},
                "cells": {(0, 4), (0, 5), (1, 4), (2, 4), (2, 5), (3, 3), (3, 4)},
            },
            {
                "reds": {(2, 5)},
                "cells": {(0, 6), (0, 7), (1, 7), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)},
            },
            {
                "reds": {(3, 2)},
                "cells": {(3, 0), (3, 1), (3, 2), (4, 0), (5, 0), (6, 0)},
            },
            {
                "reds": {(3, 2), (3, 3)},
                "cells": {(3, 2), (3, 3), (4, 2), (5, 2), (6, 1), (6, 2), (7, 1)},
            },
        ],
    },
    {
        "holes": 3,
        "pieces": [
            {
                "reds": {(2, 5), (3, 3)},
                "cells": {(0, 4), (0, 5), (0, 6), (1, 4), (2, 4), (2, 5), (3, 3), (3, 4)},
            },
            {
                "reds": {(2, 5), (3, 8)},
                "cells": {(0, 7), (1, 7), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 8)},
            },
            {
                "reds": {(3, 2)},
                "cells": {(3, 0), (3, 1), (3, 2), (4, 0), (5, 0), (5, 1), (6, 1)},
            },
            {
                "reds": {(3, 2), (3, 3)},
                "cells": {(3, 2), (3, 3), (4, 2), (5, 2)},
            },
            {
                "reds": {(3, 8)},
                "cells": {(3, 7), (3, 8), (3, 9), (4, 7), (4, 9), (5, 7), (5, 8), (5, 9)},
            },
        ],
    },
]


def _transform_point(r: int, c: int, rot: int, flip: bool) -> tuple[int, int]:
    rr, cc = int(r), int(c)
    if flip:
        cc = -cc
    for _ in range(rot % 4):
        rr, cc = cc, -rr
    return rr, cc


def _transform_family(family: dict, rot: int, flip: bool) -> dict:
    transformed = []
    all_cells: list[tuple[int, int]] = []
    for piece in family["pieces"]:
        cells = {_transform_point(r, c, rot, flip) for r, c in piece["cells"]}
        reds = {_transform_point(r, c, rot, flip) for r, c in piece["reds"]}
        transformed.append({"cells": cells, "reds": reds})
        all_cells.extend(cells)
    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    for piece in transformed:
        piece["cells"] = {(r - min_r, c - min_c) for r, c in piece["cells"]}
        piece["reds"] = {(r - min_r, c - min_c) for r, c in piece["reds"]}
    return {"holes": family["holes"], "pieces": transformed}


def _piece_array(piece: dict, color: int) -> dict:
    coords = np.array(sorted(piece["cells"]), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in piece["cells"]:
        arr[r - r0, c - c0] = color
    for r, c in piece["reds"]:
        arr[r - r0, c - c0] = RED
    return {
        "arr": arr,
        "top": int(r0),
        "left": int(c0),
        "nonred_count": int(len(piece["cells"] - piece["reds"])),
        "size": int(len(piece["cells"])),
        "red_count": int(len(piece["reds"])),
    }


def _render_assembly(placed_pieces: list[dict], top: int, left: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for piece in placed_pieces:
        arr = piece["arr"]
        ph, pw = arr.shape
        rr = top + piece["top"]
        cc = left + piece["left"]
        sub = out[rr : rr + ph, cc : cc + pw]
        mask = arr != BG
        conflict = mask & (sub != BG) & ~((arr == RED) & (sub == RED))
        assert not np.any(conflict)
        sub[mask] = arr[mask]
        out[rr : rr + ph, cc : cc + pw] = sub
    return out


def _can_scatter(board: np.ndarray, arr: np.ndarray, top: int, left: int, gap: int = 1) -> bool:
    ph, pw = arr.shape
    h, w = board.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    rr0 = max(0, top - gap)
    cc0 = max(0, left - gap)
    rr1 = min(h, top + ph + gap)
    cc1 = min(w, left + pw + gap)
    if np.any(board[rr0:rr1, cc0:cc1] != BG):
        return False
    return True


def _place_opaque(board: np.ndarray, arr: np.ndarray, top: int, left: int) -> None:
    ph, pw = arr.shape
    mask = arr != BG
    sub = board[top : top + ph, left : left + pw]
    assert not np.any(mask & (sub != BG))
    sub[mask] = arr[mask]
    board[top : top + ph, left : left + pw] = sub


def _independent_hole_dirs(grid: np.ndarray) -> list[str]:
    mask = grid != BG
    dirs = []
    for hole in _find_holes(mask):
        dirs.append(_hole_direction(hole, mask))
    return sorted(dirs)


def _validate_generated(inp: np.ndarray, expected_holes: int, expect_cross: bool) -> None:
    pieces = _extract_pieces(inp)
    assert 3 <= len(pieces) <= 5
    for piece in pieces:
        assert 4 <= piece["size"] <= 8
        assert 1 <= len(piece["reds"]) <= 2
        colors = {int(v) for v in piece["arr"].flatten() if v not in (BG, RED)}
        assert len(colors) == 1
        crop_mask = piece["arr"] != BG
        assert len(_find_holes(crop_mask)) == 0

    assembled = _assemble_unique(inp)
    holes = _find_holes(assembled != BG)
    assert len(holes) == expected_holes
    dirs = _independent_hole_dirs(assembled)
    assert len(dirs) == expected_holes

    out = main(inp)
    if expect_cross:
        assert np.any(out == CROSS)
    assert np.any(out == RAY) or np.any(out == CROSS)


def _gen_attempt() -> np.ndarray | None:
    family = random.choices(BASE_FAMILIES, weights=[3, 4, 4], k=1)[0]
    transformed = _transform_family(family, rot=random.randrange(4), flip=bool(random.randrange(2)))

    n_pieces = len(transformed["pieces"])
    palette = PIECE_COLORS.copy()
    if n_pieces == 5:
        palette.append(EXTRA_FAMILY3_COLOR)
    colors = random.sample(palette, n_pieces)
    random.shuffle(colors)

    placed_pieces = [_piece_array(piece, color) for piece, color in zip(transformed["pieces"], colors)]
    assembly_h = max(piece["top"] + piece["arr"].shape[0] for piece in placed_pieces)
    assembly_w = max(piece["left"] + piece["arr"].shape[1] for piece in placed_pieces)

    min_h = max(18, assembly_h + 6)
    min_w = max(18, assembly_w + 6)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, min(26, 30))
    grid_w = random.randint(min_w, min(26, 30))

    top = random.randint(2, grid_h - assembly_h - 2)
    left = random.randint(2, grid_w - assembly_w - 2)
    assembled = _render_assembly(placed_pieces, top, left, (grid_h, grid_w))
    expected = _emit_rays(assembled)

    counts = [piece["nonred_count"] for piece in placed_pieces]
    best = max(counts)
    if counts.count(best) != 1:
        return None
    anchor_idx = counts.index(best)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    anchor = placed_pieces[anchor_idx]
    _place_opaque(inp, anchor["arr"], top + anchor["top"], left + anchor["left"])

    order = [i for i in range(n_pieces) if i != anchor_idx]
    random.shuffle(order)
    for idx in order:
        piece = placed_pieces[idx]
        arr = piece["arr"]
        placed = False
        for _ in range(300):
            rr = random.randint(0, grid_h - arr.shape[0])
            cc = random.randint(0, grid_w - arr.shape[1])
            if abs(rr - (top + piece["top"])) <= 1 and abs(cc - (left + piece["left"])) <= 1:
                continue
            if not _can_scatter(inp, arr, rr, cc, gap=1):
                continue
            _place_opaque(inp, arr, rr, cc)
            placed = True
            break
        if not placed:
            return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None

    expect_cross = family["holes"] >= 2
    try:
        _validate_generated(inp, family["holes"], expect_cross=expect_cross)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid emergent-hole assembly input")
