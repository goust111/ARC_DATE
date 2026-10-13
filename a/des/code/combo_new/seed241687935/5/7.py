from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
TEXTURE_POOL = (1, 4, 5, 6, 7, 8, 9)
SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    arr = np.asarray(cells, dtype=float)
    mean = arr.mean(axis=0)
    return float(mean[0]), float(mean[1])


def _texture_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    texture_mask = (grid != BG) & (grid != GREEN)
    coords = np.argwhere(texture_mask)
    assert coords.size, "expected a textured rectangle and red markers"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    sub = texture_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub), "texture area must be one solid rectangle when markers are included"
    return int(r0), int(c0), int(r1), int(c1)


def _marker_side(marker: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = box
    if c == c0 and r0 <= r <= r1:
        return "left"
    if c == c1 and r0 <= r <= r1:
        return "right"
    if r == r0 and c0 <= c <= c1:
        return "top"
    if r == r1 and c0 <= c <= c1:
        return "bottom"
    raise AssertionError("marker must sit on the texture rectangle border")


def _fragment_side(cells: list[tuple[int, int]], box: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = _bbox(cells)
    r0, c0, r1, c1 = box
    if fc1 < c0:
        return "left"
    if fc0 > c1:
        return "right"
    if fr1 < r0:
        return "top"
    if fr0 > r1:
        return "bottom"
    raise AssertionError("green fragment must lie outside the textured rectangle")


def _pair_fragments(
    fragments: list[list[tuple[int, int]]], markers: list[tuple[int, int]], box: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    markers_by_side = {side: [] for side in SIDES}
    for marker in markers:
        markers_by_side[_marker_side(marker, box)].append(marker)

    pairs: list[tuple[int, int] | None] = [None] * len(fragments)
    for side in SIDES:
        frag_entries = [(idx, cells) for idx, cells in enumerate(fragments) if _fragment_side(cells, box) == side]
        target_side = OPPOSITE[side]
        cand = markers_by_side[target_side]
        assert len(frag_entries) == len(cand), "each fragment side needs the same number of opposite-edge markers"
        if not frag_entries:
            continue
        if side in ("left", "right"):
            frag_entries.sort(key=lambda item: _centroid(item[1])[0])
            cand.sort(key=lambda pos: pos[0])
        else:
            frag_entries.sort(key=lambda item: _centroid(item[1])[1])
            cand.sort(key=lambda pos: pos[1])
        for (idx, _cells), marker in zip(frag_entries, cand):
            pairs[idx] = marker

    assert all(pair is not None for pair in pairs), "every fragment must receive a marker"
    return [pair for pair in pairs if pair is not None]


def _anchor_cell(cells: list[tuple[int, int]], marker: tuple[int, int]) -> tuple[int, int]:
    mr, mc = marker
    return min(
        cells,
        key=lambda rc: (
            abs(rc[0] - mr) + abs(rc[1] - mc),
            (rc[0] - mr) ** 2 + (rc[1] - mc) ** 2,
            rc[0],
            rc[1],
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each green fragment as a cutting template to move a textured patch out of the large rectangle.

    Input structure:
    - The background is black.
    - One large axis-aligned rectangle is filled with a multicolor texture, and one-pixel red markers sit on its border.
    - Several 4-connected green fragments lie outside the rectangle; each fragment is an irregular template shape.

    Rule:
    - Pair fragments with the red markers on the opposite rectangle edge, using row alignment for left/right fragments and column alignment for top/bottom fragments.
    - For each pair, align the fragment pixel nearest to the marker with that marker position and translate the whole fragment shape onto the texture rectangle.
    - Copy the visible colors under that translated shape back onto the fragment's original cells, turn the cut-out texture cells into background, and remove the red markers.

    Key insight:
    - The green objects are only shape stencils; the actual payload comes from cutting an equally shaped patch out of the textured rectangle via the red anchor pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    box = _texture_bbox(g)
    r0, c0, r1, c1 = box
    markers = [tuple(map(int, pos)) for pos in np.argwhere(g == RED)]
    assert markers, "expected at least one red marker"
    assert all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in markers), "markers must be on the texture rectangle"

    fragments = _components(g == GREEN)
    assert fragments, "expected at least one green fragment"
    pairs = _pair_fragments(fragments, markers, box)

    out = g.copy()
    out[g == RED] = BG
    used_cut: set[tuple[int, int]] = set()

    for cells, marker in zip(fragments, pairs):
        anchor = _anchor_cell(cells, marker)
        dr = marker[0] - anchor[0]
        dc = marker[1] - anchor[1]
        cut = [(r + dr, c + dc) for r, c in cells]
        assert all(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in cut), "cut patch must stay inside the texture rectangle"
        assert not any(pos in used_cut for pos in cut), "cut patches must not overlap"
        used_cut.update(cut)

        colors = [int(g[rr, cc]) for rr, cc in cut]
        for rr, cc in cut:
            out[rr, cc] = BG
        for (fr, fc), color in zip(cells, colors):
            out[fr, fc] = color

    return out


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _grow_shape(side: str, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        if side == "left":
            ar, ac = rng.randrange(h), w - 1
            seed = (ar, w - 2)
            allowed = lambda r, c: (r, c) == (ar, ac) or c < w - 1
        elif side == "right":
            ar, ac = rng.randrange(h), 0
            seed = (ar, 1)
            allowed = lambda r, c: (r, c) == (ar, ac) or c > 0
        elif side == "top":
            ar, ac = h - 1, rng.randrange(w)
            seed = (h - 2, ac)
            allowed = lambda r, c: (r, c) == (ar, ac) or r < h - 1
        elif side == "bottom":
            ar, ac = 0, rng.randrange(w)
            seed = (1, ac)
            allowed = lambda r, c: (r, c) == (ar, ac) or r > 0
        else:
            raise ValueError(side)

        cells = {(ar, ac), seed}
        capacity = sum(1 for r in range(h) for c in range(w) if allowed(r, c))
        target = rng.randint(min(3, capacity), min(capacity, 8))

        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if _inside(h, w, nr, nc) and allowed(nr, nc) and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target:
            continue

        local = sorted((r - ar, c - ac) for r, c in cells)
        if side == "left" and sum(dc == 0 for _dr, dc in local) != 1:
            continue
        if side == "right" and sum(dc == 0 for _dr, dc in local) != 1:
            continue
        if side == "top" and sum(dr == 0 for dr, _dc in local) != 1:
            continue
        if side == "bottom" and sum(dr == 0 for dr, _dc in local) != 1:
            continue
        return local
    return None


def _extents(local: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in local]
    cols = [c for _, c in local]
    return min(rows), min(cols), max(rows), max(cols)


def _render_texture(h: int, w: int, rng: random.Random) -> np.ndarray:
    palette = rng.sample(TEXTURE_POOL, rng.randint(3, min(5, len(TEXTURE_POOL))))
    tex = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            tex[r, c] = rng.choice(palette)
    for color in palette:
        rr = rng.randrange(h)
        cc = rng.randrange(w)
        tex[rr, cc] = color
    return tex


def _cells_from_local(anchor: tuple[int, int], local: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in local]


def _place_fragment(
    side: str,
    local: list[tuple[int, int]],
    box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = box
    gh, gw = grid_shape
    rmin, cmin, rmax, cmax = _extents(local)

    if side == "left":
        row_lo = max(-rmin, r0 - rmin, r0 + 1)
        row_hi = min(gh - 1 - rmax, r1 - rmax, r1 - 1)
        col_lo = max(1 - cmin, 1 - cmin)
        col_hi = c0 - 2
        if row_lo > row_hi or col_lo > col_hi:
            return None
        anchor = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
        marker = (anchor[0], c1)
    elif side == "right":
        row_lo = max(-rmin, r0 - rmin, r0 + 1)
        row_hi = min(gh - 1 - rmax, r1 - rmax, r1 - 1)
        col_lo = c1 + 2
        col_hi = min(gw - 2 - cmax, gw - 1 - cmax)
        if row_lo > row_hi or col_lo > col_hi:
            return None
        anchor = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
        marker = (anchor[0], c0)
    elif side == "top":
        row_lo = max(1 - rmin, 1 - rmin)
        row_hi = r0 - 2
        col_lo = max(-cmin, c0 - cmin, c0 + 1)
        col_hi = min(gw - 1 - cmax, c1 - cmax, c1 - 1)
        if row_lo > row_hi or col_lo > col_hi:
            return None
        anchor = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
        marker = (r1, anchor[1])
    elif side == "bottom":
        row_lo = r1 + 2
        row_hi = min(gh - 2 - rmax, gh - 1 - rmax)
        col_lo = max(-cmin, c0 - cmin, c0 + 1)
        col_hi = min(gw - 1 - cmax, c1 - cmax, c1 - 1)
        if row_lo > row_hi or col_lo > col_hi:
            return None
        anchor = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
        marker = (r0, anchor[1])
    else:
        raise ValueError(side)

    fragment_cells = _cells_from_local(anchor, local)
    cut_cells = _cells_from_local(marker, local)
    return anchor, marker, fragment_cells, cut_cells


def _oracle_output(
    inp: np.ndarray,
    fragments: list[list[tuple[int, int]]],
    cut_cells: list[list[tuple[int, int]]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in markers:
        out[r, c] = BG
    for frag_cells, cut in zip(fragments, cut_cells):
        colors = [int(inp[r, c]) for r, c in cut]
        for r, c in cut:
            out[r, c] = BG
        for (r, c), color in zip(frag_cells, colors):
            out[r, c] = color
    return out


def _attempt(rng: random.Random) -> np.ndarray | None:
    top_margin = rng.randint(4, 7)
    bottom_margin = rng.randint(4, 7)
    left_margin = rng.randint(4, 7)
    right_margin = rng.randint(4, 7)

    rect_h_max = min(14, 30 - top_margin - bottom_margin)
    rect_w_max = min(14, 30 - left_margin - right_margin)
    if rect_h_max < 8 or rect_w_max < 8:
        return None

    rect_h = rng.randint(8, rect_h_max)
    rect_w = rng.randint(8, rect_w_max)
    grid_h = top_margin + rect_h + bottom_margin
    grid_w = left_margin + rect_w + right_margin
    box = (top_margin, left_margin, top_margin + rect_h - 1, left_margin + rect_w - 1)

    count = rng.choice((2, 2, 3, 3, 4))
    sides = rng.sample(SIDES, count)
    locals_by_side: dict[str, list[tuple[int, int]]] = {}
    for side in sides:
        local = _grow_shape(side, rng)
        if local is None:
            return None
        locals_by_side[side] = local

    texture = _render_texture(rect_h, rect_w, rng)
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[box[0] : box[2] + 1, box[1] : box[3] + 1] = texture

    outside_used: set[tuple[int, int]] = set()
    cut_used: set[tuple[int, int]] = set()
    markers: list[tuple[int, int]] = []
    fragments: list[list[tuple[int, int]]] = []
    cuts: list[list[tuple[int, int]]] = []

    order = sides[:]
    rng.shuffle(order)
    for side in order:
        placed = _place_fragment(side, locals_by_side[side], box, (grid_h, grid_w), rng)
        if placed is None:
            return None
        _anchor, marker, frag_cells, cut_cells = placed

        if any(cell in outside_used for cell in frag_cells):
            return None
        if any(cell in cut_used for cell in cut_cells):
            return None
        if marker in markers:
            return None

        outside_used.update(frag_cells)
        cut_used.update(cut_cells)
        markers.append(marker)
        fragments.append(frag_cells)
        cuts.append(cut_cells)

    for r, c in markers:
        inp[r, c] = RED
    for frag_cells in fragments:
        for r, c in frag_cells:
            inp[r, c] = GREEN

    expected = _oracle_output(inp, fragments, cuts, markers)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
