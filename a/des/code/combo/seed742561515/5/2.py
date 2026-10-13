from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
FRAME = 5
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_MOVE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
MOVE_TO_SIDE = {v: k for k, v in SIDE_TO_MOVE.items()}
ARROW_SHAPES = {
    (-1, 0): {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    (1, 0): {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    (0, -1): {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    (0, 1): {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(pts, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return r0, c0, mask


def _match_arrow(mask: np.ndarray) -> tuple[int, int] | None:
    if mask.shape != (3, 3):
        return None
    pts = {tuple(map(int, p)) for p in np.argwhere(mask)}
    for move, shape in ARROW_SHAPES.items():
        if pts == shape:
            return move
    return None


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == FRAME)
    assert len(pts) > 0
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return r0, c0, r1, c1


def _extract_pegs(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict[str, int | str]]:
    r0, c0, r1, c1 = frame_bbox
    pegs: list[dict[str, int | str]] = []
    for c in range(c0 + 1, c1):
        depth = 0
        while r0 + 1 + depth < r1 and grid[r0 + 1 + depth, c] == FRAME:
            depth += 1
        if depth:
            pegs.append({"side": "top", "anchor": c, "depth": depth})
    for c in range(c0 + 1, c1):
        depth = 0
        while r1 - 1 - depth > r0 and grid[r1 - 1 - depth, c] == FRAME:
            depth += 1
        if depth:
            pegs.append({"side": "bottom", "anchor": c, "depth": depth})
    for r in range(r0 + 1, r1):
        depth = 0
        while c0 + 1 + depth < c1 and grid[r, c0 + 1 + depth] == FRAME:
            depth += 1
        if depth:
            pegs.append({"side": "left", "anchor": r, "depth": depth})
    for r in range(r0 + 1, r1):
        depth = 0
        while c1 - 1 - depth > c0 and grid[r, c1 - 1 - depth] == FRAME:
            depth += 1
        if depth:
            pegs.append({"side": "right", "anchor": r, "depth": depth})
    return pegs


def _extract_notch(piece_mask: np.ndarray, move: tuple[int, int]) -> tuple[int, int]:
    missing = np.argwhere(~piece_mask)
    assert len(missing) > 0
    if move == (1, 0):
        cols = {int(c) for _, c in missing}
        rows = sorted(int(r) for r, _ in missing)
        depth = len(rows)
        assert len(cols) == 1 and rows == list(range(1, depth + 1))
        return int(next(iter(cols))), depth
    if move == (-1, 0):
        h = piece_mask.shape[0]
        cols = {int(c) for _, c in missing}
        rows = sorted(int(r) for r, _ in missing)
        depth = len(rows)
        assert len(cols) == 1 and rows == list(range(h - depth - 1, h - 1))
        return int(next(iter(cols))), depth
    if move == (0, 1):
        rows = {int(r) for r, _ in missing}
        cols = sorted(int(c) for _, c in missing)
        depth = len(cols)
        assert len(rows) == 1 and cols == list(range(1, depth + 1))
        return int(next(iter(rows))), depth
    if move == (0, -1):
        w = piece_mask.shape[1]
        rows = {int(r) for r, _ in missing}
        cols = sorted(int(c) for _, c in missing)
        depth = len(cols)
        assert len(rows) == 1 and cols == list(range(w - depth - 1, w - 1))
        return int(next(iter(rows))), depth
    raise AssertionError("invalid move")


def _place_mask(out: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    view = out[top : top + h, left : left + w]
    view[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_bbox = _frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    pegs = _extract_pegs(g, frame_bbox)
    peg_lookup: dict[tuple[str, int, int], dict[str, int | str]] = {}
    for peg in pegs:
        key = (str(peg["side"]), int(peg["anchor"]), int(peg["depth"]))
        peg_lookup[key] = peg

    out = np.full_like(g, BG)
    out[g == FRAME] = FRAME

    inner_peg_mask = np.zeros_like(g, dtype=bool)
    inner_peg_mask[r0 + 1 : r1, c0 + 1 : c1] = g[r0 + 1 : r1, c0 + 1 : c1] == FRAME

    colors = [c for c in sorted(set(map(int, g.ravel()))) if c not in (BG, FRAME)]
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2
        arrow_move: tuple[int, int] | None = None
        piece_coords: np.ndarray | None = None
        piece_mask: np.ndarray | None = None
        piece_top = 0
        piece_left = 0
        for comp in comps:
            cr0, cc0, cmask = _coords_to_mask(comp)
            move = _match_arrow(cmask)
            if move is not None:
                assert arrow_move is None
                arrow_move = move
            else:
                assert piece_coords is None
                piece_coords = comp
                piece_mask = cmask
                piece_top = cr0
                piece_left = cc0
        assert arrow_move is not None and piece_coords is not None and piece_mask is not None

        side = MOVE_TO_SIDE[arrow_move]
        notch_offset, notch_depth = _extract_notch(piece_mask, arrow_move)
        if side in ("top", "bottom"):
            anchor = piece_left + notch_offset
        else:
            anchor = piece_top + notch_offset
        assert (side, anchor, notch_depth) in peg_lookup

        ph, pw = piece_mask.shape
        if side == "top":
            final_top, final_left = r0, piece_left
            assert final_top > piece_top
        elif side == "bottom":
            final_top, final_left = r1 - ph + 1, piece_left
            assert final_top < piece_top
        elif side == "left":
            final_top, final_left = piece_top, c0
            assert final_left > piece_left
        else:
            final_top, final_left = piece_top, c1 - pw + 1
            assert final_left < piece_left

        translated = np.zeros_like(g, dtype=bool)
        translated[final_top : final_top + ph, final_left : final_left + pw] = piece_mask
        assert final_top >= r0 and final_left >= c0
        assert final_top + ph - 1 <= r1 and final_left + pw - 1 <= c1
        assert not np.any(translated & inner_peg_mask)
        _place_mask(out, final_top, final_left, piece_mask, color)

    return out


def _piece_mask(side: str, height: int, width: int, offset: int, depth: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if side == "top":
        mask[1 : depth + 1, offset] = False
    elif side == "bottom":
        mask[height - depth - 1 : height - 1, offset] = False
    elif side == "left":
        mask[offset, 1 : depth + 1] = False
    elif side == "right":
        mask[offset, width - depth - 1 : width - 1] = False
    else:
        raise AssertionError("invalid side")
    return mask


def _peg_cells(side: str, anchor: int, depth: int, frame_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return {(r0 + d, anchor) for d in range(1, depth + 1)}
    if side == "bottom":
        return {(r1 - d, anchor) for d in range(1, depth + 1)}
    if side == "left":
        return {(anchor, c0 + d) for d in range(1, depth + 1)}
    if side == "right":
        return {(anchor, c1 - d) for d in range(1, depth + 1)}
    raise AssertionError("invalid side")


def _piece_cells(top: int, left: int, mask: np.ndarray) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in np.argwhere(mask):
        cells.add((top + int(rr), left + int(cc)))
    return cells


def _touches(existing: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in existing:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in existing:
                return True
    return False


def _arrow_mask(move: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    for r, c in ARROW_SHAPES[move]:
        mask[r, c] = True
    return mask


def _sample_piece_spec(
    rng: random.Random,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> dict[str, int | str | np.ndarray] | None:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    if side in ("top", "bottom"):
        height = rng.randint(3, 4)
        width = rng.randint(3, min(5, w - 2))
        offset = rng.randint(1, width - 2)
        depth = rng.randint(1, min(2, height - 1))
        lo = c0 + offset
        hi = c1 - (width - offset - 1)
        if lo > hi:
            return None
        anchor = rng.randint(lo, hi)
        final_top = r0 if side == "top" else r1 - height + 1
        final_left = anchor - offset
        if side == "top":
            max_gap = final_top - height
            if max_gap < 2:
                return None
            gap = rng.randint(2, min(4, max_gap))
            init_top = final_top - gap - height
        else:
            max_gap = h - (r1 + height + 1)
            if max_gap < 2:
                return None
            gap = rng.randint(2, min(4, max_gap))
            init_top = r1 + gap + 1
        init_left = final_left
    else:
        height = rng.randint(3, min(5, h - 2))
        width = rng.randint(3, 4)
        offset = rng.randint(1, height - 2)
        depth = rng.randint(1, min(2, width - 1))
        lo = r0 + offset
        hi = r1 - (height - offset - 1)
        if lo > hi:
            return None
        anchor = rng.randint(lo, hi)
        final_top = anchor - offset
        final_left = c0 if side == "left" else c1 - width + 1
        if side == "left":
            max_gap = final_left - width
            if max_gap < 2:
                return None
            gap = rng.randint(2, min(4, max_gap))
            init_left = final_left - gap - width
        else:
            max_gap = w - (c1 + width + 1)
            if max_gap < 2:
                return None
            gap = rng.randint(2, min(4, max_gap))
            init_left = c1 + gap + 1
        init_top = final_top

    mask = _piece_mask(side, height, width, offset, depth)
    return {
        "side": side,
        "height": height,
        "width": width,
        "offset": offset,
        "depth": depth,
        "anchor": anchor,
        "init_top": init_top,
        "init_left": init_left,
        "final_top": final_top,
        "final_left": final_left,
        "mask": mask,
    }


def _side_positions(side: str, frame_bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame_bbox
    if side in ("top", "bottom"):
        return list(range(c0 + 1, c1))
    return list(range(r0 + 1, r1))


def _pick_pegs(
    rng: random.Random,
    frame_bbox: tuple[int, int, int, int],
    selected_specs: dict[str, dict[str, int | str | np.ndarray]],
    final_piece_cells: set[tuple[int, int]],
) -> list[dict[str, int | str]]:
    pegs: list[dict[str, int | str]] = []
    used: dict[str, set[int]] = defaultdict(set)
    used_cells: set[tuple[int, int]] = set()
    for side in SIDES:
        if side in selected_specs:
            spec = selected_specs[side]
            anchor = int(spec["anchor"])
            depth = int(spec["depth"])
            pegs.append({"side": side, "anchor": anchor, "depth": depth})
            used[side].add(anchor)
            used_cells |= _peg_cells(side, anchor, depth, frame_bbox)

    for side in SIDES:
        count = rng.randint(1, 2) if side not in selected_specs else rng.randint(0, 1)
        positions = _side_positions(side, frame_bbox)
        rng.shuffle(positions)
        span_block: set[int] = set()
        matched_depth = None
        if side in selected_specs:
            spec = selected_specs[side]
            matched_depth = int(spec["depth"])
            if side in ("top", "bottom"):
                start = int(spec["final_left"])
                stop = start + int(spec["width"]) - 1
            else:
                start = int(spec["final_top"])
                stop = start + int(spec["height"]) - 1
            span_block = set(range(start, stop + 1))

        added = 0
        for pos in positions:
            if pos in used[side]:
                continue
            if pos in span_block:
                continue
            if any(abs(pos - other) < 2 for other in used[side]):
                continue
            depth = rng.randint(1, 2)
            if matched_depth is not None and depth == matched_depth:
                depth = 3 - matched_depth
            cells = _peg_cells(side, pos, depth, frame_bbox)
            if cells & final_piece_cells:
                continue
            if _touches(used_cells, cells):
                continue
            pegs.append({"side": side, "anchor": pos, "depth": depth})
            used[side].add(pos)
            used_cells |= cells
            added += 1
            if added == count:
                break
        if side not in selected_specs and not any(str(p["side"]) == side for p in pegs):
            return []
    return pegs


def _draw_frame(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], pegs: list[dict[str, int | str]]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    grid = np.full((h, w), BG, dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    for peg in pegs:
        for rr, cc in _peg_cells(str(peg["side"]), int(peg["anchor"]), int(peg["depth"]), frame_bbox):
            grid[rr, cc] = FRAME
    return grid


def _arrow_positions(
    spec: dict[str, int | str | np.ndarray],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    side = str(spec["side"])
    top = int(spec["init_top"])
    left = int(spec["init_left"])
    ph = int(spec["height"])
    pw = int(spec["width"])
    positions: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        offsets = list(range(ph - 2))
        for off in offsets:
            positions.append((top + off, left - 4))
            positions.append((top + off, left + pw + 1))
    else:
        offsets = list(range(pw - 2))
        for off in offsets:
            positions.append((top - 4, left + off))
            positions.append((top + ph + 1, left + off))
    return [(r, c) for r, c in positions if 0 <= r and 0 <= c and r + 2 < h and c + 2 < w]


def _oracle_output(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    pegs: list[dict[str, int | str]],
    specs: list[dict[str, int | str | np.ndarray]],
) -> np.ndarray:
    out = _draw_frame(shape, frame_bbox, pegs)
    for spec in specs:
        _place_mask(
            out,
            int(spec["final_top"]),
            int(spec["final_left"]),
            np.asarray(spec["mask"], dtype=bool),
            int(spec["color"]),
        )
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(25, 30)
    w = rng.randint(25, 30)
    frame_h = rng.randint(8, 10)
    frame_w = rng.randint(8, 10)
    r0 = rng.randint(7, h - frame_h - 7)
    c0 = rng.randint(7, w - frame_w - 7)
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    piece_count = rng.randint(2, 4)
    chosen_sides = rng.sample(list(SIDES), piece_count)
    colors = rng.sample(COLOR_POOL, piece_count)

    specs: list[dict[str, int | str | np.ndarray]] = []
    selected_specs: dict[str, dict[str, int | str | np.ndarray]] = {}
    final_piece_cells: set[tuple[int, int]] = set()
    initial_occ: set[tuple[int, int]] = set()
    matched_peg_cells: set[tuple[int, int]] = set()

    for side, color in zip(chosen_sides, colors):
        placed = False
        for _ in range(80):
            spec = _sample_piece_spec(rng, side, frame_bbox, (h, w))
            if spec is None:
                continue
            mask = np.asarray(spec["mask"], dtype=bool)
            final_cells = _piece_cells(int(spec["final_top"]), int(spec["final_left"]), mask)
            init_cells = _piece_cells(int(spec["init_top"]), int(spec["init_left"]), mask)
            peg_cells = _peg_cells(side, int(spec["anchor"]), int(spec["depth"]), frame_bbox)
            if final_cells & final_piece_cells:
                continue
            if init_cells & initial_occ:
                continue
            if _touches(matched_peg_cells, peg_cells):
                continue
            spec["color"] = color
            specs.append(spec)
            selected_specs[side] = spec
            final_piece_cells |= final_cells
            initial_occ |= init_cells
            matched_peg_cells |= peg_cells
            placed = True
            break
        if not placed:
            return None

    pegs = _pick_pegs(rng, frame_bbox, selected_specs, final_piece_cells)
    if not pegs:
        return None

    frame_grid = _draw_frame((h, w), frame_bbox, pegs)
    all_peg_cells: set[tuple[int, int]] = set()
    for peg in pegs:
        all_peg_cells |= _peg_cells(str(peg["side"]), int(peg["anchor"]), int(peg["depth"]), frame_bbox)
    if all_peg_cells & final_piece_cells:
        return None

    grid = frame_grid.copy()
    occupied = {(int(r), int(c)) for r, c in np.argwhere(frame_grid != BG)}
    arrow_mask_cache: dict[tuple[int, int], np.ndarray] = {}

    for spec in specs:
        color = int(spec["color"])
        move = SIDE_TO_MOVE[str(spec["side"])]
        piece_mask = np.asarray(spec["mask"], dtype=bool)
        init_top = int(spec["init_top"])
        init_left = int(spec["init_left"])
        piece_cells = _piece_cells(init_top, init_left, piece_mask)
        if piece_cells & occupied:
            return None
        for r, c in piece_cells:
            grid[r, c] = color
        occupied |= piece_cells

        arrow_mask = arrow_mask_cache.setdefault(move, _arrow_mask(move))
        candidates = _arrow_positions(spec, (h, w))
        rng.shuffle(candidates)
        arrow_done = False
        for ar, ac in candidates:
            arrow_cells = _piece_cells(ar, ac, arrow_mask)
            if arrow_cells & occupied:
                continue
            for r, c in arrow_cells:
                grid[r, c] = color
            occupied |= arrow_cells
            arrow_done = True
            break
        if not arrow_done:
            return None

    expected = _oracle_output((h, w), frame_bbox, pegs, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
