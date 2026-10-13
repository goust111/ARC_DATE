from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
CARDINALS = ("top", "right", "bottom", "left")
SECTORS = ("up", "down", "left", "right")
OPPOSITE = {"up": "bottom", "down": "top", "left": "right", "right": "left"}
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_cells(
    cells: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...]
) -> tuple[tuple[tuple[int, int, int], ...], int, int]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    r0, c0 = min(rs), min(cs)
    norm = tuple(sorted((r - r0, c - c0, int(col)) for r, c, col in cells))
    h = max(r for r, _, _ in norm) + 1
    w = max(c for _, c, _ in norm) + 1
    return norm, int(h), int(w)


def _rot90(cells: tuple[tuple[int, int, int], ...]) -> tuple[tuple[tuple[int, int, int], ...], int, int]:
    norm, h, _w = _normalize_cells(cells)
    rot = [(c, h - 1 - r, col) for r, c, col in norm]
    return _normalize_cells(rot)


def _unique_rotations(cells: tuple[tuple[int, int, int], ...]) -> list[tuple[tuple[int, int, int], ...]]:
    out: list[tuple[tuple[int, int, int], ...]] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    cur = cells
    for _ in range(4):
        norm, _, _ = _normalize_cells(cur)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
        cur = _rot90(cur)[0]
    return out


def _anchor_info(
    cells: tuple[tuple[int, int, int], ...], anchor_color: int
) -> tuple[tuple[int, int], tuple[str, ...], int, int]:
    norm, h, w = _normalize_cells(cells)
    pts = [(r, c) for r, c, col in norm if int(col) == int(anchor_color)]
    assert len(pts) == 1, "piece must contain one anchor pixel"
    r, c = pts[0]
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    return (int(r), int(c)), tuple(sides), int(h), int(w)


def _relative_to_anchor(
    cells: tuple[tuple[int, int, int], ...], anchor_color: int
) -> tuple[tuple[int, int, int], ...]:
    (ar, ac), _sides, _h, _w = _anchor_info(cells, anchor_color)
    return tuple(sorted((r - ar, c - ac, int(col)) for r, c, col in cells))


def _sector_for_point(r: float, c: float, center_r: float, center_c: float) -> str:
    dr = float(r) - float(center_r)
    dc = float(c) - float(center_c)
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        sub = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            sub[r - r0, c - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub, border):
            candidates.append((len(comp), r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one grey rectangular frame"
    _, r0, c0, r1, c1 = candidates[0]
    return int(r0), int(c0), int(r1), int(c1)


def _extract_pieces(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[int, dict]:
    r0, c0, r1, c1 = frame_bbox
    outside = (grid != BG).copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    pieces: dict[int, dict] = {}
    for comp in _components(outside):
        colors = [int(grid[r, c]) for r, c in comp]
        counts = Counter(colors)
        if len(counts) != 2:
            continue
        anchor_candidates = [col for col, cnt in counts.items() if int(cnt) == 1]
        body_candidates = [col for col, cnt in counts.items() if int(cnt) > 1]
        if len(anchor_candidates) != 1 or len(body_candidates) != 1:
            continue
        anchor_color = int(anchor_candidates[0])
        body_color = int(body_candidates[0])
        local = [(r, c, int(grid[r, c])) for r, c in comp]
        cells, _, _ = _normalize_cells(local)
        pieces[body_color] = {"body_color": body_color, "anchor_color": anchor_color, "cells": cells}
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the grey frame and fill it by matching each interior color to the outside piece
    with the same body color, rotating that piece so its unique anchor pixel faces the frame center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _find_frame(g)
    r0, c0, r1, c1 = frame_bbox
    fh = int(r1 - r0 + 1)
    fw = int(c1 - c0 + 1)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    markers: dict[int, tuple[int, int]] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            col = int(g[r, c])
            if col in (BG, FRAME):
                continue
            assert col not in markers, "interior marker colors must be unique"
            markers[col] = (int(r), int(c))
    assert markers, "frame must contain at least one interior marker"

    pieces = _extract_pieces(g, frame_bbox)
    assert set(markers) <= set(pieces), "every interior color must have a matching outside piece"

    out = np.full((fh, fw), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME

    for color, (mr, mc) in markers.items():
        piece = pieces[int(color)]
        sector = _sector_for_point(float(mr), float(mc), center_r, center_c)
        want_side = OPPOSITE[sector]

        matches = []
        for rot in _unique_rotations(piece["cells"]):
            _anchor, sides, _h, _w = _anchor_info(rot, piece["anchor_color"])
            if want_side in sides:
                matches.append(rot)
        assert matches, "piece must be rotatable to face the frame center"

        chosen = None
        for rot in matches:
            _anchor, sides, _h, _w = _anchor_info(rot, piece["anchor_color"])
            if sides == (want_side,):
                chosen = rot
                break
        if chosen is None:
            chosen = matches[0]

        for dr, dc, col in _relative_to_anchor(chosen, piece["anchor_color"]):
            rr = int(mr - r0 + dr)
            cc = int(mc - c0 + dc)
            assert 0 <= rr < fh and 0 <= cc < fw, "placed piece must stay inside the cropped frame"
            assert out[rr, cc] == BG, "pieces may only cover background cells inside the frame"
            out[rr, cc] = int(col)

    return out


def _make_piece(body_color: int, anchor_color: int) -> dict | None:
    for _ in range(400):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        if w < 3:
            continue
        anchor_c = random.randint(1, w - 2)
        body_target = random.randint(3, min(7, (h - 1) * w))
        body = {(1, anchor_c)}
        frontier = {(1, anchor_c - 1), (1, anchor_c + 1), (2, anchor_c)}
        while len(body) < body_target and frontier:
            choices = []
            for r, c in frontier:
                if 1 <= r < h and 0 <= c < w and (r, c) not in body:
                    choices.append((r, c))
            if not choices:
                break
            pick = random.choice(choices)
            body.add(pick)
            frontier.discard(pick)
            r, c = pick
            frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
        if len(body) != body_target:
            continue

        raw = [(0, anchor_c, anchor_color)] + [(r, c, body_color) for r, c in sorted(body)]
        cells, _hh, _ww = _normalize_cells(raw)
        _anchor, sides, ph, pw = _anchor_info(cells, anchor_color)
        if sides != ("top",):
            continue
        if ph < 3 or pw < 2:
            continue

        rotations = _unique_rotations(cells)
        if len(rotations) != 4:
            continue

        by_side: dict[str, tuple[tuple[int, int, int], ...]] = {}
        ok = True
        for rot in rotations:
            _anchor, rot_sides, _rh, _rw = _anchor_info(rot, anchor_color)
            if len(rot_sides) != 1 or rot_sides[0] in by_side:
                ok = False
                break
            by_side[rot_sides[0]] = rot
        if not ok or set(by_side) != set(CARDINALS):
            continue

        return {
            "body_color": int(body_color),
            "anchor_color": int(anchor_color),
            "cells": cells,
            "rotations": rotations,
            "by_side": by_side,
            "area": int(len(cells)),
        }
    return None


def _buffer_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 1 <= rr < h - 1 and 1 <= cc < w - 1:
                    out.add((rr, cc))
    return out


def _inside_candidates(piece: dict, sector: str, frame_h: int, frame_w: int) -> list[dict]:
    center_r = (frame_h - 1) / 2.0
    center_c = (frame_w - 1) / 2.0
    rot = piece["by_side"][OPPOSITE[sector]]
    rel = _relative_to_anchor(rot, piece["anchor_color"])
    candidates = []
    for ar in range(1, frame_h - 1):
        for ac in range(1, frame_w - 1):
            if _sector_for_point(float(ar), float(ac), center_r, center_c) != sector:
                continue
            occ: set[tuple[int, int]] = set()
            placed: list[tuple[int, int, int]] = []
            ok = True
            for dr, dc, col in rel:
                rr = int(ar + dr)
                cc = int(ac + dc)
                if not (1 <= rr < frame_h - 1 and 1 <= cc < frame_w - 1):
                    ok = False
                    break
                occ.add((rr, cc))
                placed.append((rr, cc, int(col)))
            if not ok:
                continue
            candidates.append(
                {
                    "anchor": (int(ar), int(ac)),
                    "cells": tuple(placed),
                    "occ": occ,
                    "buffer": _buffer_cells(occ, frame_h, frame_w),
                }
            )
    random.shuffle(candidates)
    return candidates


def _place_inside_frame(pieces: list[dict], sectors: list[str], frame_h: int, frame_w: int) -> list[dict] | None:
    cand_lists = [_inside_candidates(piece, sector, frame_h, frame_w) for piece, sector in zip(pieces, sectors)]
    if any(not cands for cands in cand_lists):
        return None

    chosen: list[dict | None] = [None] * len(pieces)

    def search(used_forbidden: set[tuple[int, int]], remaining: set[int]) -> bool:
        if not remaining:
            return True
        best_i = None
        best_options = None
        for idx in sorted(remaining):
            options = [cand for cand in cand_lists[idx] if cand["occ"].isdisjoint(used_forbidden)]
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_i = idx
                best_options = options
        assert best_i is not None and best_options is not None
        next_remaining = set(remaining)
        next_remaining.remove(best_i)
        for cand in best_options:
            chosen[best_i] = cand
            if search(used_forbidden | cand["buffer"], next_remaining):
                return True
            chosen[best_i] = None
        return False

    if not search(set(), set(range(len(pieces)))):
        return None
    return [cand for cand in chosen if cand is not None]


def _draw_frame_with_markers(
    frame_h: int, frame_w: int, pieces: list[dict], placements: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    inp_frame = np.full((frame_h, frame_w), BG, dtype=int)
    out = np.full((frame_h, frame_w), BG, dtype=int)

    for arr in (inp_frame, out):
        arr[0, :] = FRAME
        arr[-1, :] = FRAME
        arr[:, 0] = FRAME
        arr[:, -1] = FRAME

    for piece, placement in zip(pieces, placements):
        ar, ac = placement["anchor"]
        inp_frame[int(ar), int(ac)] = int(piece["body_color"])
        for r, c, col in placement["cells"]:
            out[int(r), int(c)] = int(col)

    return inp_frame, out


def _mark_bbox(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _scatter_pieces(
    canvas: np.ndarray, pieces: list[dict], frame_bbox: tuple[int, int, int, int]
) -> bool:
    h, w = canvas.shape
    blocked = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    _mark_bbox(blocked, fr0 - 1, fc0 - 1, fr1 + 1, fc1 + 1)

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        variants = list(piece["rotations"])
        random.shuffle(variants)
        placed = False
        for rot in variants:
            _norm, ph, pw = _normalize_cells(rot)
            positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            random.shuffle(positions)
            for top, left in positions:
                cells = [(top + r, left + c, int(col)) for r, c, col in rot]
                if any(blocked[r, c] for r, c, _ in cells):
                    continue
                for r, c, col in cells:
                    canvas[r, c] = int(col)
                occ = {(r, c) for r, c, _ in cells}
                for r, c in occ:
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            if 0 <= rr < h and 0 <= cc < w:
                                blocked[rr, cc] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_pieces = random.randint(2, 4)
    color_pool = random.sample(PIECE_COLORS, 2 * n_pieces)
    body_colors = color_pool[:n_pieces]
    anchor_colors = color_pool[n_pieces:]

    pieces = []
    for body_color, anchor_color in zip(body_colors, anchor_colors):
        piece = _make_piece(body_color, anchor_color)
        if piece is None:
            return None
        pieces.append(piece)
    pieces.sort(key=lambda piece: int(piece["area"]), reverse=True)

    sectors = random.sample(list(SECTORS), n_pieces)

    placements = None
    for _ in range(80):
        frame_h = random.randint(9, 16)
        frame_w = random.randint(9, 16)
        placements = _place_inside_frame(pieces, sectors, frame_h, frame_w)
        if placements is not None:
            break
    if placements is None:
        return None

    frame_inp, out = _draw_frame_with_markers(frame_h, frame_w, pieces, placements)

    grid_h = random.randint(max(18, frame_h + 8), max(22, frame_h + 14))
    grid_w = random.randint(max(18, frame_w + 8), max(22, frame_w + 14))
    canvas = np.full((grid_h, grid_w), BG, dtype=int)
    fr0 = (grid_h - frame_h) // 2
    fc0 = (grid_w - frame_w) // 2
    canvas[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = frame_inp
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    if not _scatter_pieces(canvas, pieces, frame_bbox):
        return None

    solved = main(canvas)
    if not np.array_equal(solved, out):
        return None
    return canvas, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
