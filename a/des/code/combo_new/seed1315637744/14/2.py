from __future__ import annotations

from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ("top", "bottom", "left", "right")
BG = 0
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


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
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _side_cells(side: str, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _gap_cells_on_side(side_cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(side_cells) < 4:
        return None
    internal = side_cells[1:-1]
    missing = [i for i, cell in enumerate(internal) if cell not in occupied]
    if not missing:
        return []
    if len(missing) < 2:
        return None
    start = missing[0]
    if missing != list(range(start, start + len(missing))):
        return None
    return [internal[i] for i in missing]


def _analyze_frame(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    occ = {(int(r), int(c)) for r, c in coords}
    for r, c in occ:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners <= occ:
        return None

    gap_side = None
    gap_cells: list[tuple[int, int]] = []
    for side in DIRECTIONS:
        cells = _side_cells(side, r0, c0, r1, c1)
        missing = _gap_cells_on_side(cells, occ)
        if missing is None:
            return None
        if missing:
            if gap_side is not None:
                return None
            gap_side = side
            gap_cells = missing

    if gap_side is None:
        return None

    expected = 2 * h + 2 * w - 4 - len(gap_cells)
    if len(occ) != expected:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": gap_side,
        "gap_cells": gap_cells,
        "coords": coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color fragment through its matching frame opening until it reaches the opposite inner wall.

    Input structure:
    - The background is black.
    - Several hollow rectangular frames appear in different colors; each frame is missing one contiguous segment on exactly one side.
    - A single solid fragment of each frame color sits outside the frame on the side of that opening.
    - Each fragment is 4-connected and narrow enough along the opening axis to pass through the gap.

    Rule:
    - Match every fragment to the frame of the same color.
    - Detect which side of the frame contains the gap: top, bottom, left, or right.
    - Translate the fragment straight through that opening, perpendicular to the gapped side, until the fragment is as far as possible inside the frame and touches the opposite inner wall.
    - Fill the missing border pixels with the frame color, and erase the fragment from its original exterior location.

    Key insight:
    The opening determines only the direction of motion; the fragment does not stop in the gap, but keeps sliding until constrained by the far wall inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    moves: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        if not comps:
            continue

        frame = None
        frame_idx = None
        for i, comp in enumerate(comps):
            info = _analyze_frame(comp)
            if info is None:
                continue
            if frame is not None:
                if len(comp) > len(frame["coords"]):
                    frame = info
                    frame_idx = i
            else:
                frame = info
                frame_idx = i

        if frame is None or frame_idx is None:
            continue

        piece_candidates = [comp for i, comp in enumerate(comps) if i != frame_idx]
        if len(piece_candidates) != 1:
            if not piece_candidates:
                continue
            piece = max(piece_candidates, key=len)
        else:
            piece = piece_candidates[0]

        r0, c0, r1, c1 = frame["bbox"]
        side = str(frame["side"])
        rows = piece[:, 0]
        cols = piece[:, 1]
        dr = 0
        dc = 0
        if side == "top":
            dr = (r1 - 1) - int(rows.max())
        elif side == "bottom":
            dr = (r0 + 1) - int(rows.min())
        elif side == "left":
            dc = (c1 - 1) - int(cols.max())
        else:
            dc = (c0 + 1) - int(cols.min())

        final_coords = piece + np.array([dr, dc], dtype=int)
        valid = np.all(final_coords[:, 0] >= 0) and np.all(final_coords[:, 0] < h)
        valid = valid and np.all(final_coords[:, 1] >= 0) and np.all(final_coords[:, 1] < w)
        if not valid:
            continue

        moves.append(
            {
                "color": color,
                "piece": piece,
                "gap_cells": frame["gap_cells"],
                "final": final_coords,
            }
        )

    out = g.copy()
    for move in moves:
        piece = move["piece"]
        out[piece[:, 0], piece[:, 1]] = bg
    for move in moves:
        for r, c in move["gap_cells"]:
            out[int(r), int(c)] = int(move["color"])
    for move in moves:
        final = move["final"]
        out[final[:, 0], final[:, 1]] = int(move["color"])
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _grow_piece(mask: np.ndarray, target_area: int, rng: np.random.Generator) -> np.ndarray:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    h, w = mask.shape
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(0, len(frontier)))]
        cells.add((nr, nc))
    out = np.zeros_like(mask, dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return _crop_mask(out)


def _horizontal_piece(gap_w: int, max_h: int, rng: np.random.Generator) -> np.ndarray:
    raw_h = int(rng.integers(1, max_h + 1))
    row = int(rng.integers(0, raw_h))
    seed = np.zeros((raw_h, gap_w), dtype=bool)
    seed[row, :] = True
    max_area = raw_h * gap_w
    base = int(seed.sum())
    target = int(rng.integers(base, max_area + 1))
    return _grow_piece(seed, target, rng)


def _vertical_piece(gap_h: int, max_w: int, rng: np.random.Generator) -> np.ndarray:
    raw_w = int(rng.integers(1, max_w + 1))
    col = int(rng.integers(0, raw_w))
    seed = np.zeros((gap_h, raw_w), dtype=bool)
    seed[:, col] = True
    max_area = gap_h * raw_w
    base = int(seed.sum())
    target = int(rng.integers(base, max_area + 1))
    return _grow_piece(seed, target, rng)


def _frame_mask(height: int, width: int, side: str | None = None, gap_start: int = 0, gap_len: int = 0) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if side is not None and gap_len > 0:
        if side == "top":
            mask[0, gap_start : gap_start + gap_len] = False
        elif side == "bottom":
            mask[-1, gap_start : gap_start + gap_len] = False
        elif side == "left":
            mask[gap_start : gap_start + gap_len, 0] = False
        else:
            mask[gap_start : gap_start + gap_len, -1] = False
    return mask


def _paint(base: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    h, w = patch.shape
    sub = base[top : top + h, left : left + w]
    sub[mask] = patch[mask]


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _make_cluster(direction: str, color: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    frame_h = int(rng.integers(6, 11))
    frame_w = int(rng.integers(6, 11))
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    gap_sep = int(rng.integers(1, 4))

    if direction in ("top", "bottom"):
        max_gap = inner_w if inner_w <= 2 else inner_w - 1
        gap_len = int(rng.integers(2, max_gap + 1))
        gap_start = int(rng.integers(1, frame_w - gap_len))
        piece_mask = _horizontal_piece(gap_len, inner_h - 1, rng)
        piece_h, piece_w = piece_mask.shape
        if direction == "top":
            frame_r, frame_c = piece_h + gap_sep, 0
            piece_r, piece_c = 0, gap_start
            patch_h = piece_h + gap_sep + frame_h
            patch_w = frame_w
            final_r = frame_r + frame_h - 1 - piece_h
            final_c = frame_c + gap_start
        else:
            frame_r, frame_c = 0, 0
            piece_r, piece_c = frame_h + gap_sep, gap_start
            patch_h = frame_h + gap_sep + piece_h
            patch_w = frame_w
            final_r = frame_r + 1
            final_c = frame_c + gap_start
    else:
        max_gap = inner_h if inner_h <= 2 else inner_h - 1
        gap_len = int(rng.integers(2, max_gap + 1))
        gap_start = int(rng.integers(1, frame_h - gap_len))
        piece_mask = _vertical_piece(gap_len, inner_w - 1, rng)
        piece_h, piece_w = piece_mask.shape
        if direction == "left":
            frame_r, frame_c = 0, piece_w + gap_sep
            piece_r, piece_c = gap_start, 0
            patch_h = frame_h
            patch_w = piece_w + gap_sep + frame_w
            final_r = frame_r + gap_start
            final_c = frame_c + frame_w - 1 - piece_w
        else:
            frame_r, frame_c = 0, 0
            piece_r, piece_c = gap_start, frame_w + gap_sep
            patch_h = frame_h
            patch_w = frame_w + gap_sep + piece_w
            final_r = frame_r + gap_start
            final_c = frame_c + 1

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = np.full((patch_h, patch_w), BG, dtype=int)

    open_frame = _mask_to_patch(_frame_mask(frame_h, frame_w, direction, gap_start, gap_len), color, BG)
    closed_frame = _mask_to_patch(_frame_mask(frame_h, frame_w), color, BG)
    piece_patch = _mask_to_patch(piece_mask, color, BG)

    _paint(inp, open_frame, frame_r, frame_c, BG)
    _paint(inp, piece_patch, piece_r, piece_c, BG)
    _paint(out, closed_frame, frame_r, frame_c, BG)
    _paint(out, piece_patch, final_r, final_c, BG)
    return inp, out


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = occupied.shape
    pts = np.argwhere(patch != bg)
    for r, c in pts:
        rr = top + int(r)
        cc = left + int(c)
        if occupied[rr, cc]:
            return False
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _place_case(patches_in: list[np.ndarray], patches_out: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    nonbg = [int(np.count_nonzero(p != BG)) for p in patches_in]
    max_h = max(int(p.shape[0]) for p in patches_in)
    max_w = max(int(p.shape[1]) for p in patches_in)
    base = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(sum(nonbg) * 4))) + len(patches_in))
    low = min(base, 30)
    if low > 30:
        return None

    order = sorted(range(len(patches_in)), key=lambda i: (-nonbg[i], -patches_in[i].shape[0], -patches_in[i].shape[1]))
    for _ in range(60):
        h_hi = min(30, max(low, low + 5))
        w_hi = min(30, max(low, low + 5))
        h = int(rng.integers(low, h_hi + 1))
        w = int(rng.integers(low, w_hi + 1))
        canvas_in = np.full((h, w), BG, dtype=int)
        canvas_out = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True
        for idx in order:
            patch_in = patches_in[idx]
            patch_out = patches_out[idx]
            ph, pw = patch_in.shape
            if ph > h or pw > w:
                ok = False
                break
            placed = False
            for _ in range(200):
                top = int(rng.integers(0, h - ph + 1))
                left = int(rng.integers(0, w - pw + 1))
                if not _can_place(occupied, patch_in, top, left, BG):
                    continue
                _paint(canvas_in, patch_in, top, left, BG)
                _paint(canvas_out, patch_out, top, left, BG)
                occupied[top : top + ph, left : left + pw] |= patch_in != BG
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return canvas_in, canvas_out
    return None


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = int(rng.integers(2, 5))
    dirs = [DIRECTIONS[i] for i in rng.permutation(len(DIRECTIONS))[:n_objects]]
    colors = [int(c) for c in rng.choice(np.arange(1, 10), size=n_objects, replace=False)]

    patches_in: list[np.ndarray] = []
    patches_out: list[np.ndarray] = []
    for direction, color in zip(dirs, colors, strict=False):
        patch_in, patch_out = _make_cluster(direction, color, rng)
        patches_in.append(patch_in)
        patches_out.append(patch_out)

    placed = _place_case(patches_in, patches_out, rng)
    if placed is None:
        return None
    return placed


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is None:
            continue
        inp, expected = case
        if inp.ndim != 2 or max(inp.shape) > 30:
            continue
        if expected.ndim != 2 or max(expected.shape) > 30:
            continue
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
