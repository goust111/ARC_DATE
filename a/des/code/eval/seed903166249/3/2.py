from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_coords_from_mask(np.rot90(mask, k)) for k in range(4))


def _neighbor_colors(sub: np.ndarray, hole: list[tuple[int, int]]) -> set[int]:
    h, w = sub.shape
    colors: set[int] = set()
    for r, c in hole:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                val = int(sub[nr, nc])
                if val not in (BLACK, GREY):
                    colors.add(val)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each border notch of the grey template with the uniquely matching rotated fragment,
    recoloring the inserted shape to the notch's marker color.
    """
    g = np.asarray(grid, dtype=int)

    foreground = _components(g != BLACK)
    assert foreground, "expected a template component"
    template = max(foreground, key=len)
    template_set = set(template)
    r0, c0, r1, c1 = _bbox(template)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    sh, sw = sub.shape

    notches = []
    for comp in _components(sub == BLACK):
        if not any(rr in (0, sh - 1) or cc in (0, sw - 1) for rr, cc in comp):
            continue
        colors = _neighbor_colors(sub, comp)
        assert len(colors) == 1, "each notch must expose exactly one marker color"
        notch_mask = _mask_from_coords(comp)
        notches.append(
            {
                "coords": [(r0 + rr, c0 + cc) for rr, cc in comp],
                "mask": notch_mask,
                "color": next(iter(colors)),
            }
        )

    fragments = []
    for comp in foreground:
        if set(comp) == template_set:
            continue
        frag_mask = _mask_from_coords(comp)
        fragments.append({"coords": comp, "mask": frag_mask})

    assert notches, "expected at least one template notch"
    assert len(fragments) == len(notches), "fragment count must equal notch count"

    matches: list[tuple[int, int, int]] = []
    used_pieces: set[int] = set()
    for hole_index, hole in enumerate(notches):
        candidates = []
        for piece_index, piece in enumerate(fragments):
            for rot in range(4):
                if np.array_equal(np.rot90(piece["mask"], rot), hole["mask"]):
                    candidates.append((piece_index, rot))
        assert len(candidates) == 1, "each notch must have a unique fragment/rotation match"
        piece_index, rot = candidates[0]
        assert piece_index not in used_pieces, "fragment matches must be one-to-one"
        used_pieces.add(piece_index)
        matches.append((hole_index, piece_index, rot))

    out = g.copy()
    for piece in fragments:
        for r, c in piece["coords"]:
            out[r, c] = BLACK
    for hole_index, _piece_index, _rot in matches:
        hole = notches[hole_index]
        for r, c in hole["coords"]:
            out[r, c] = int(hole["color"])
    return out


RAW_SHAPES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (1, 0), (1, 2), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)],
    [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
]


def _build_shape_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for raw in RAW_SHAPES:
        mask = _mask_from_coords(raw)
        if len(_components(mask)) != 1:
            continue
        rotations = {_coords_from_mask(np.rot90(mask, k)) for k in range(4)}
        if len(rotations) != 4:
            continue
        key = _rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        library.append(mask)
    assert len(library) >= 8, "shape library should offer several non-symmetric classes"
    return library


SHAPE_LIBRARY = _build_shape_library()


def _padded_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _place_hole(mask: np.ndarray, side: str, top: int, left: int, th: int, tw: int) -> set[tuple[int, int]]:
    h, w = mask.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    cells: set[tuple[int, int]] = set()
    if side == "left":
        for r, c in coords:
            cells.add((top + r, c))
    elif side == "right":
        for r, c in coords:
            cells.add((top + r, tw - w + c))
    elif side == "top":
        for r, c in coords:
            cells.add((r, left + c))
    elif side == "bottom":
        for r, c in coords:
            cells.add((th - h + r, left + c))
    else:
        raise ValueError(side)
    return cells


def _hole_boundary(hole: set[tuple[int, int]], blocked: set[tuple[int, int]], th: int, tw: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for r, c in hole:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < th and 0 <= nc < tw and (nr, nc) not in blocked:
                boundary.add((nr, nc))
    return boundary


def _valid_hole(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], th: int, tw: int) -> bool:
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= r < th and 0 <= c < tw):
            return False
        if (r, c) in occupied:
            return False
    return True


def _place_piece(mask: np.ndarray, blocked: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    gh, gw = blocked.shape
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    candidates: list[tuple[int, int]] = []
    for top in range(gh - h + 1):
        for left in range(gw - w + 1):
            ok = True
            for r, c in cells:
                if blocked[top + r, left + c]:
                    ok = False
                    break
            if ok:
                candidates.append((top, left))
    if not candidates:
        return None
    return candidates[int(RNG.integers(len(candidates)))]


def _add_piece_to_grid(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
) -> None:
    gh, gw = grid.shape
    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = color
    coords = {(int(top + r), int(left + c)) for r, c in np.argwhere(mask)}
    for rr, cc in _padded_cells(coords, gh, gw):
        blocked[rr, cc] = True


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(20, 31))
    w = int(RNG.integers(20, 31))
    hole_count = int(RNG.choice([2, 3, 4]))
    th = int(RNG.integers(10, min(17, h - 4) + 1))
    tw = int(RNG.integers(10, min(17, w - 4) + 1))
    if h - th < 4 or w - tw < 4:
        return None

    top0 = int(RNG.integers(2, h - th - 1))
    left0 = int(RNG.integers(2, w - tw - 1))
    bottom0 = top0 + th - 1
    right0 = left0 + tw - 1

    shape_indices = RNG.choice(len(SHAPE_LIBRARY), size=hole_count, replace=False)
    marker_colors = list(RNG.choice(PALETTE, size=hole_count, replace=False))
    piece_pool = [c for c in PALETTE if c not in marker_colors]
    piece_colors = list(RNG.choice(piece_pool, size=hole_count, replace=False))
    required_rots = list(RNG.choice(4, size=hole_count, replace=False))
    RNG.shuffle(marker_colors)
    RNG.shuffle(piece_colors)
    RNG.shuffle(required_rots)

    hole_records = []
    occupied: set[tuple[int, int]] = set()
    side_choices = ["left", "right", "top", "bottom"]

    for i, shape_idx in enumerate(shape_indices):
        placed = None
        for _ in range(80):
            side = str(RNG.choice(side_choices))
            oriented = np.rot90(SHAPE_LIBRARY[int(shape_idx)], int(RNG.integers(4)))
            hh, ww = oriented.shape
            if side in ("left", "right"):
                if th - hh <= 2:
                    continue
                anchor = int(RNG.integers(1, th - hh))
                cells = _place_hole(oriented, side, anchor, 0, th, tw)
            else:
                if tw - ww <= 2:
                    continue
                anchor = int(RNG.integers(1, tw - ww))
                cells = _place_hole(oriented, side, 0, anchor, th, tw)
            if not _valid_hole(cells, occupied, th, tw):
                continue
            boundary = _hole_boundary(cells, cells | occupied, th, tw)
            if not boundary:
                continue
            padded = _padded_cells(cells, th, tw)
            if any(cell in occupied for cell in padded):
                continue
            placed = {
                "cells": cells,
                "mask": oriented,
                "marker": int(marker_colors[i]),
                "piece_color": int(piece_colors[i]),
                "required_rot": int(required_rots[i]),
                "boundary": boundary,
            }
            break
        if placed is None:
            return None
        hole_records.append(placed)
        occupied |= _padded_cells(placed["cells"], th, tw)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[top0 : bottom0 + 1, left0 : right0 + 1] = GREY

    expected = grid.copy()
    for record in hole_records:
        for r, c in record["cells"]:
            grid[top0 + r, left0 + c] = BLACK
            expected[top0 + r, left0 + c] = record["marker"]
        for r, c in record["boundary"]:
            grid[top0 + r, left0 + c] = record["marker"]
            expected[top0 + r, left0 + c] = record["marker"]

    blocked = np.zeros((h, w), dtype=bool)
    rr0 = max(0, top0 - 1)
    cc0 = max(0, left0 - 1)
    rr1 = min(h, bottom0 + 2)
    cc1 = min(w, right0 + 2)
    blocked[rr0:rr1, cc0:cc1] = True

    for record in sorted(hole_records, key=lambda x: int(x["mask"].sum()), reverse=True):
        piece_mask = np.rot90(record["mask"], (4 - record["required_rot"]) % 4)
        pos = _place_piece(piece_mask, blocked)
        if pos is None:
            return None
        _add_piece_to_grid(grid, blocked, piece_mask, record["piece_color"], pos[0], pos[1])

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
