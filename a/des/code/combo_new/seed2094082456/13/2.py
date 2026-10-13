import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in pts))


def _shape_rel_coords(shape: tuple[tuple[int, int], ...], anchor_idx: int) -> tuple[tuple[int, int], ...]:
    ar, ac = shape[anchor_idx]
    return tuple((int(r - ar), int(c - ac)) for r, c in shape)


def _fits_anchor(rel_coords: tuple[tuple[int, int], ...], anchor: tuple[int, int], h: int, w: int) -> bool:
    ar, ac = anchor
    return all(0 <= ar + dr < h and 0 <= ac + dc < w for dr, dc in rel_coords)


def _render_shape(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components_mask(grid == color):
            if comp.shape[0] < 12:
                continue
            r0, c0, r1, c1 = _bbox_coords(comp)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 4 or w < 4:
                continue
            if int(comp.shape[0]) != 2 * (h + w) - 4:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(mask, border):
                continue
            if np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] != BG) == 0:
                continue
            candidates.append((h * w, color, (r0, c0, r1, c1)))
    assert candidates, "must contain a framed rectangle"
    candidates.sort(reverse=True)
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "frame must be uniquely largest"
    _, frame_color, bbox = best[0]
    return int(frame_color), bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert each scattered monochrome fragment into the framed template by aligning it to its same-colored anchor pixel.

    Input structure:
    - The background is black.
    - One large single-color rectangular border encloses a template whose interior is mostly black.
    - Inside that rectangle, each fragment color appears exactly once as a singleton anchor pixel.
    - Outside the rectangle, there is one 4-connected monochrome fragment for each anchor color.

    Rule:
    - Find the unique large rectangular border and crop to it for the output canvas.
    - For each outside fragment, match it to the unique interior anchor of the same color.
    - Translate the fragment without rotating or flipping so that exactly one of its pixels lands on the anchor and every other translated pixel lands on black template interior cells.
    - Paint all matched fragments into the frame and return only the cropped framed rectangle.

    Key insight:
    Color identifies which anchor belongs to which fragment, but shape is still needed to determine which fragment pixel must coincide with the anchor.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1) = _find_frame(g)

    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    interior = g[r0 + 1 : r1, c0 + 1 : c1]

    anchor_map: dict[int, tuple[int, int]] = {}
    for rr, cc in np.argwhere((interior != BG) & (interior != frame_color)):
        color = int(interior[rr, cc])
        assert color not in anchor_map, "anchor colors must be unique inside the frame"
        anchor_map[color] = (int(r0 + 1 + rr), int(c0 + 1 + cc))
    assert anchor_map, "frame must contain anchor pixels"

    for color, anchor in anchor_map.items():
        piece_comps: list[np.ndarray] = []
        for comp in _components_mask(g == color):
            inside_frame = [(r0 <= int(r) <= r1) and (c0 <= int(c) <= c1) for r, c in comp]
            if any(inside_frame):
                assert int(comp.shape[0]) == 1 and all(inside_frame), "interior color occurrence must be a singleton anchor"
                continue
            if int(comp.shape[0]) >= 2:
                piece_comps.append(comp)
        assert len(piece_comps) == 1, "each anchor color must have exactly one outside fragment"
        piece = piece_comps[0]
        pr0, pc0, _, _ = _bbox_coords(piece)
        shape = tuple(sorted((int(r - pr0), int(c - pc0)) for r, c in piece))

        candidates: list[list[tuple[int, int]]] = []
        for anchor_idx, (sr, sc) in enumerate(shape):
            translated = [(int(anchor[0] + r - sr), int(anchor[1] + c - sc)) for r, c in shape]
            if not all(r0 < tr < r1 and c0 < tc < c1 for tr, tc in translated):
                continue
            ok = True
            overlap = 0
            for tr, tc in translated:
                val = int(g[tr, tc])
                if (tr, tc) == anchor:
                    if val != color:
                        ok = False
                        break
                    overlap += 1
                elif val != BG:
                    ok = False
                    break
            if ok and overlap == 1:
                candidates.append(translated)
        assert len(candidates) == 1, "fragment-anchor alignment must be unique"

        for tr, tc in candidates[0]:
            lr = int(tr - r0)
            lc = int(tc - c0)
            if (tr, tc) != anchor:
                assert out[lr, lc] == BG, "placed fragments must not overlap"
            out[lr, lc] = int(color)

    return out


def _random_polyomino(size: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))

    pts = list(cells)
    mode = rng.randrange(8)
    transformed = []
    for r, c in pts:
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = r, -c
        elif mode == 2:
            nr, nc = -r, c
        elif mode == 3:
            nr, nc = -r, -c
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, -r
        elif mode == 6:
            nr, nc = -c, r
        else:
            nr, nc = -c, -r
        transformed.append((nr, nc))
    return _normalize_coords(transformed)


def _unique_anchor_positions(
    shape: tuple[tuple[int, int], ...],
    anchor_idx: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    rels = [_shape_rel_coords(shape, i) for i in range(len(shape))]
    good: list[tuple[int, int]] = []
    for ar in range(h):
        for ac in range(w):
            anchor = (ar, ac)
            if not _fits_anchor(rels[anchor_idx], anchor, h, w):
                continue
            if any(_fits_anchor(rels[i], anchor, h, w) for i in range(len(shape)) if i != anchor_idx):
                continue
            good.append(anchor)
    return good


def _cells_from_anchor(
    shape: tuple[tuple[int, int], ...],
    anchor_idx: int,
    anchor: tuple[int, int],
) -> list[tuple[int, int]]:
    rel = _shape_rel_coords(shape, anchor_idx)
    return [(int(anchor[0] + dr), int(anchor[1] + dc)) for dr, dc in rel]


def _reserve_piece(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        reserved[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                reserved[nr, nc] = True


def _build_frame_patch(
    ih: int,
    iw: int,
    border_color: int,
    anchors: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((ih + 2, iw + 2), dtype=int)
    out = np.zeros((ih + 2, iw + 2), dtype=int)
    inp[0, :] = border_color
    inp[-1, :] = border_color
    inp[:, 0] = border_color
    inp[:, -1] = border_color
    out[:, :] = inp
    for piece in anchors:
        color = int(piece["color"])
        ar, ac = piece["anchor"]
        inp[1 + ar, 1 + ac] = color
        for r, c in piece["cells"]:
            out[1 + r, 1 + c] = color
    return inp, out


def _can_place_bbox(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, top - pad)
    rr1 = min(blocked.shape[0], top + h + pad)
    cc0 = max(0, left - pad)
    cc1 = min(blocked.shape[1], left + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, top - pad)
    rr1 = min(blocked.shape[0], top + h + pad)
    cc0 = max(0, left - pad)
    cc1 = min(blocked.shape[1], left + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _paste_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _scatter_input(
    frame_patch: np.ndarray,
    pieces: list[dict],
    rng: random.Random,
) -> np.ndarray | None:
    frame_h, frame_w = frame_patch.shape
    piece_dims = [(_render_shape(piece["shape"], int(piece["color"])).shape) for piece in pieces]
    area_budget = frame_h * frame_w + sum((h + 2) * (w + 2) for h, w in piece_dims) + 25
    min_side = max(frame_h + 4, frame_w + 4, int(np.ceil(area_budget ** 0.5)))
    low = max(16, min_side)
    if low > 30:
        return None
    grid_h = rng.randint(low, 30)
    grid_w = rng.randint(low, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    placements = [(frame_patch, frame_h, frame_w, "frame", None)]
    for piece in pieces:
        patch = _render_shape(piece["shape"], int(piece["color"]))
        ph, pw = patch.shape
        placements.append((patch, ph, pw, "piece", piece))

    frame_patch_, _, _, _, _ = placements[0]
    frame_tries = 120
    frame_top = frame_left = None
    for _ in range(frame_tries):
        top = rng.randint(1, grid_h - frame_h - 1)
        left = rng.randint(1, grid_w - frame_w - 1)
        if _can_place_bbox(blocked, top, left, frame_h, frame_w, pad=1):
            frame_top, frame_left = top, left
            _paste_patch(grid, frame_patch_, top, left)
            _mark_bbox(blocked, top, left, frame_h, frame_w, pad=1)
            break
    if frame_top is None or frame_left is None:
        return None

    for patch, ph, pw, kind, _piece in placements[1:]:
        placed = False
        for _ in range(200):
            top = rng.randint(1, grid_h - ph - 1)
            left = rng.randint(1, grid_w - pw - 1)
            if frame_top <= top + ph - 1 and top <= frame_top + frame_h - 1 and frame_left <= left + pw - 1 and left <= frame_left + frame_w - 1:
                continue
            if not _can_place_bbox(blocked, top, left, ph, pw, pad=1):
                continue
            _paste_patch(grid, patch, top, left)
            _mark_bbox(blocked, top, left, ph, pw, pad=1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(2, 5)
    colors = rng.sample(PALETTE, piece_count + 1)
    border_color = int(colors[0])
    piece_colors = [int(c) for c in colors[1:]]

    ih = rng.randint(6, 12)
    iw = rng.randint(6, 12)
    reserved = np.zeros((ih, iw), dtype=bool)
    pieces: list[dict] = []

    for color in piece_colors:
        placed = False
        for _ in range(120):
            size = rng.randint(2, 5)
            shape = _random_polyomino(size, rng)
            anchor_order = list(range(len(shape)))
            rng.shuffle(anchor_order)
            for anchor_idx in anchor_order:
                anchors = _unique_anchor_positions(shape, anchor_idx, ih, iw)
                rng.shuffle(anchors)
                for anchor in anchors:
                    cells = _cells_from_anchor(shape, anchor_idx, anchor)
                    if any(reserved[r, c] for r, c in cells):
                        continue
                    pieces.append(
                        {
                            "color": int(color),
                            "shape": shape,
                            "anchor_idx": int(anchor_idx),
                            "anchor": (int(anchor[0]), int(anchor[1])),
                            "cells": [(int(r), int(c)) for r, c in cells],
                        }
                    )
                    _reserve_piece(reserved, cells)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    frame_in, expected = _build_frame_patch(ih, iw, border_color, pieces)
    grid = _scatter_input(frame_in, pieces, rng)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
