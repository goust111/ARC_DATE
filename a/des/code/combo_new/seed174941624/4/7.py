from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER_COLOR = 9
PIECE_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
MARKER_BASE = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ],
    dtype=int,
)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _is_elbow3(mask: np.ndarray) -> bool:
    arr = np.asarray(mask, dtype=bool)
    if int(arr.sum()) != 3:
        return False
    rows = np.where(arr.sum(axis=1) > 0)[0]
    cols = np.where(arr.sum(axis=0) > 0)[0]
    if len(rows) != 2 or len(cols) != 2:
        return False
    sub = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return sub.shape == (2, 2) and int(sub.sum()) == 3


def _extract_piece_masks(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = []
    for color in sorted({int(v) for v in np.unique(g) if int(v) != int(bg)}):
        for coords in _components(g == int(color)):
            comps.append({"color": int(color), "coords": coords, "mask": _crop_component(coords)})

    marker_ids = [i for i, comp in enumerate(comps) if _is_elbow3(comp["mask"])]
    assert len(marker_ids) == 1, "expected exactly one 3-pixel elbow marker"
    marker_id = marker_ids[0]

    pieces = [np.asarray(comp["mask"], dtype=bool) for i, comp in enumerate(comps) if i != marker_id]
    assert 2 <= len(pieces) <= 4, "expected 2 to 4 puzzle pieces"
    piece_colors = {comp["color"] for i, comp in enumerate(comps) if i != marker_id}
    assert len(piece_colors) == 1, "all pieces should share one color"
    piece_color = next(iter(piece_colors))
    return pieces, int(piece_color)


def _piece_data(mask: np.ndarray) -> dict:
    coords = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    return {
        "mask": np.asarray(mask, dtype=bool),
        "coords": coords,
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "area": int(mask.sum()),
    }


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        if h <= 30 and w <= 30:
            out.append((int(h), int(w)))
            if h != w:
                out.append((int(w), int(h)))
    return sorted(out)


def _count_tilings(pieces: list[dict], rect_h: int, rect_w: int, limit: int = 2) -> int:
    if any(p["h"] > rect_h or p["w"] > rect_w for p in pieces):
        return 0

    cell_count = rect_h * rect_w
    uncovered_all = (1 << cell_count) - 1
    piece_opts: list[dict[int, list[int]]] = []

    for piece in pieces:
        by_cell: dict[int, list[int]] = {}
        rel = piece["coords"]
        for r0 in range(rect_h - piece["h"] + 1):
            for c0 in range(rect_w - piece["w"] + 1):
                cover = 0
                for dr, dc in rel:
                    cover |= 1 << ((r0 + dr) * rect_w + (c0 + dc))
                for dr, dc in rel:
                    idx = (r0 + dr) * rect_w + (c0 + dc)
                    by_cell.setdefault(int(idx), []).append(int(cover))
        piece_opts.append(by_cell)

    count = 0
    n = len(pieces)

    def dfs(uncovered: int, unused_mask: int) -> None:
        nonlocal count
        if count >= limit:
            return
        if uncovered == 0:
            count += 1
            return

        first_cell = (uncovered & -uncovered).bit_length() - 1
        moves: list[tuple[int, int]] = []
        for i in range(n):
            if not ((unused_mask >> i) & 1):
                continue
            for cover in piece_opts[i].get(first_cell, ()):
                if cover & ~uncovered:
                    continue
                moves.append((i, int(cover)))
        if not moves:
            return

        moves.sort(key=lambda item: -pieces[item[0]]["area"])
        for i, cover in moves:
            dfs(uncovered ^ cover, unused_mask & ~(1 << i))
            if count >= limit:
                return

    dfs(uncovered_all, (1 << n) - 1)
    return int(count)


def _solve_rectangle_dims(piece_masks: list[np.ndarray]) -> tuple[int, int]:
    pieces = sorted((_piece_data(mask) for mask in piece_masks), key=lambda d: (-d["area"], -d["h"], -d["w"]))
    total_area = int(sum(p["area"] for p in pieces))
    dims = []
    for rect_h, rect_w in _factor_pairs(total_area):
        if _count_tilings(pieces, int(rect_h), int(rect_w), limit=1) > 0:
            dims.append((int(rect_h), int(rect_w)))
    assert len(dims) == 1, f"expected one rectangle size, got {dims}"
    return dims[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered toothed fragments into one rectangle, then rotate it by the fragment count.

    Input structure:
    - The background is black.
    - There are 2 to 4 non-overlapping monochrome fragments of the same color, each 4-connected and separated by background.
    - A separate 3-pixel maroon elbow marker sits in one corner area and is not part of the puzzle pieces.

    Rule:
    - Ignore the marker and extract the monochrome fragments as separate pieces.
    - Translate the pieces without rotating them until they tile one complete solid rectangle.
    - Let N be the number of pieces; rotate that rectangle clockwise by N × 90 degrees.
    - Return only the rotated rectangle on its own minimal canvas, with no marker.

    Key insight:
    - The jagged edges are complementary translation-only jigsaw locks; once the rectangle is reconstructed, the hidden second rule is that the piece count itself selects the final rotation.
    """
    piece_masks, piece_color = _extract_piece_masks(grid)
    rect_h, rect_w = _solve_rectangle_dims(piece_masks)
    n_pieces = len(piece_masks)

    if n_pieces == 3:
        out_h, out_w = rect_w, rect_h
    else:
        out_h, out_w = rect_h, rect_w
    return np.full((out_h, out_w), int(piece_color), dtype=int)


def _connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _partition_rectangles(rect_h: int, rect_w: int, n_pieces: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, int(rect_h), int(rect_w))]
    while len(rects) < n_pieces:
        candidates = []
        for idx, (r0, c0, h, w) in enumerate(rects):
            if w >= 6 and h >= 3:
                for cut in range(3, w - 2):
                    left = h * cut
                    right = h * (w - cut)
                    if left >= 6 and right >= 6:
                        candidates.append((idx, "v", int(cut)))
            if h >= 6 and w >= 3:
                for cut in range(3, h - 2):
                    top = w * cut
                    bottom = w * (h - cut)
                    if top >= 6 and bottom >= 6:
                        candidates.append((idx, "h", int(cut)))
        if not candidates:
            return None

        idx, axis, cut = random.choice(candidates)
        r0, c0, h, w = rects.pop(idx)
        if axis == "v":
            rects.append((r0, c0, h, cut))
            rects.append((r0, c0 + cut, h, w - cut))
        else:
            rects.append((r0, c0, cut, w))
            rects.append((r0 + cut, c0, h - cut, w))
    return rects


def _owner_from_rectangles(rect_h: int, rect_w: int, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    owner = np.full((rect_h, rect_w), -1, dtype=int)
    for idx, (r0, c0, h, w) in enumerate(rects):
        owner[r0 : r0 + h, c0 : c0 + w] = int(idx)
    assert bool(np.all(owner >= 0))
    return owner


def _boundary_segments(owner: np.ndarray) -> list[dict]:
    h, w = owner.shape
    segments: list[dict] = []

    for c in range(w - 1):
        r = 0
        while r < h:
            left = int(owner[r, c])
            right = int(owner[r, c + 1])
            if left == right:
                r += 1
                continue
            r0 = r
            while r < h and int(owner[r, c]) == left and int(owner[r, c + 1]) == right:
                r += 1
            segments.append({"axis": "v", "line": int(c), "start": int(r0), "stop": int(r), "a": left, "b": right})

    for r in range(h - 1):
        c = 0
        while c < w:
            top = int(owner[r, c])
            bottom = int(owner[r + 1, c])
            if top == bottom:
                c += 1
                continue
            c0 = c
            while c < w and int(owner[r, c]) == top and int(owner[r + 1, c]) == bottom:
                c += 1
            segments.append({"axis": "h", "line": int(r), "start": int(c0), "stop": int(c), "a": top, "b": bottom})
    return segments


def _sample_non_adjacent(values: list[int], count: int) -> list[int]:
    chosen: list[int] = []
    pool = values[:]
    random.shuffle(pool)
    for v in pool:
        if all(abs(v - u) >= 2 for u in chosen):
            chosen.append(int(v))
            if len(chosen) == count:
                break
    return sorted(chosen)


def _add_teeth(owner: np.ndarray) -> np.ndarray | None:
    out = owner.copy()
    for seg in _boundary_segments(owner):
        interior = list(range(seg["start"] + 1, seg["stop"] - 1))
        if not interior:
            return None
        tooth_count = 1 if len(interior) < 4 else random.choice([1, 2])
        picks = _sample_non_adjacent(interior, tooth_count)
        if len(picks) != tooth_count:
            return None
        push_forward = bool(random.getrandbits(1))
        if seg["axis"] == "v":
            c = int(seg["line"])
            for r in picks:
                if push_forward:
                    out[r, c + 1] = int(seg["a"])
                else:
                    out[r, c] = int(seg["b"])
        else:
            r = int(seg["line"])
            for c in picks:
                if push_forward:
                    out[r + 1, c] = int(seg["a"])
                else:
                    out[r, c] = int(seg["b"])
    return out


def _piece_masks_from_owner(owner: np.ndarray) -> list[np.ndarray]:
    pieces = []
    for idx in range(int(owner.max()) + 1):
        mask = owner == int(idx)
        if not _connected(mask):
            return []
        pieces.append(_crop_component([tuple(map(int, pt)) for pt in np.argwhere(mask)]))
    return pieces


def _generate_piece_masks(n_pieces: int) -> tuple[list[np.ndarray], int, int] | None:
    for _ in range(200):
        rect_h = random.randint(6, 11)
        rect_w = random.randint(6, 11)
        rects = _partition_rectangles(rect_h, rect_w, n_pieces)
        if rects is None:
            continue
        owner = _owner_from_rectangles(rect_h, rect_w, rects)
        owner = _add_teeth(owner)
        if owner is None:
            continue
        pieces = _piece_masks_from_owner(owner)
        if len(pieces) != n_pieces:
            continue
        if any(int(mask.sum()) < 6 for mask in pieces):
            continue
        if len({_shape_key(mask) for mask in pieces}) != len(pieces):
            continue
        dims = []
        data = sorted((_piece_data(mask) for mask in pieces), key=lambda d: (-d["area"], -d["h"], -d["w"]))
        total_area = int(sum(p["area"] for p in data))
        for h, w in _factor_pairs(total_area):
            cnt = _count_tilings(data, int(h), int(w), limit=2)
            if cnt > 0:
                dims.append((int(h), int(w), int(cnt)))
        if len(dims) != 1 or dims[0][2] != 1:
            continue
        return pieces, int(rect_h), int(rect_w)
    return None


def _marker_patch(rot_k: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[np.rot90(MARKER_BASE, -int(rot_k)).astype(bool)] = MARKER_COLOR
    return patch


def _patch_fits(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - pad)
        r2 = min(H, rr + pad + 1)
        c1 = max(0, cc - pad)
        c2 = min(W, cc + pad + 1)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _random_positions(limit: int) -> list[int]:
    vals = list(range(limit))
    random.shuffle(vals)
    return vals


def _scatter_input(piece_masks: list[np.ndarray], piece_color: int) -> np.ndarray | None:
    max_h = max(int(mask.shape[0]) for mask in piece_masks)
    max_w = max(int(mask.shape[1]) for mask in piece_masks)
    canvas_h = random.randint(max(12, max_h + 6), 24)
    canvas_w = random.randint(max(12, max_w + 6), 24)
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)

    marker = _marker_patch(random.randint(0, 3))
    corners = [
        (0, 0),
        (0, canvas_w - 3),
        (canvas_h - 3, 0),
        (canvas_h - 3, canvas_w - 3),
    ]
    mr, mc = random.choice(corners)
    _paste_patch(canvas, marker, mr, mc)

    order = piece_masks[:]
    random.shuffle(order)
    for mask in sorted(order, key=lambda arr: -(int(arr.shape[0]) * int(arr.shape[1]))):
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = int(piece_color)
        placed = False
        for r0 in _random_positions(canvas_h - patch.shape[0] + 1):
            for c0 in _random_positions(canvas_w - patch.shape[1] + 1):
                if _patch_fits(canvas, patch, int(r0), int(c0), pad=1):
                    _paste_patch(canvas, patch, int(r0), int(c0))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_pieces = random.choice([2, 3, 4])
        piece_color = random.choice(PIECE_COLORS)
        built = _generate_piece_masks(n_pieces)
        if built is None:
            continue
        piece_masks, rect_h, rect_w = built
        inp = _scatter_input(piece_masks, piece_color)
        if inp is None:
            continue

        exp_h, exp_w = (rect_w, rect_h) if n_pieces == 3 else (rect_h, rect_w)
        expected = np.full((exp_h, exp_w), int(piece_color), dtype=int)
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
