from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
PATCH_ATTEMPTS = 120
PACK_ATTEMPTS = 160
PLACE_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 if np.any(mask) else False


def _distance_to_cells(point: tuple[int, int], cells: list[tuple[int, int]]) -> int:
    pr, pc = point
    return min(abs(pr - r) + abs(pc - c) for r, c in cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each grey shape along the row or column named by its paired boundary markers, keeping the side nearest the guide pixel.

    Input structure:
    - The grid has a dominant black background plus one or more 4-connected grey shapes.
    - Each grey shape has exactly two same-colored non-grey boundary markers that share one row or one column.
    - Each shape also has one extra single-pixel guide color placed inside the shape as a hole or just outside it.

    Rule:
    - For every grey component, find the two same-colored adjacent markers and treat the row or column through them as the cut line.
    - Split the component into the two sides on either side of that line, using the guide pixel's Manhattan distance to decide which side is nearer.
    - Keep the nearer half together with the cut line itself, and erase the farther half to background.
    - Remove all marker pixels and all guide pixels.

    Key insight:
    - The paired markers define a slicing line, not a movement direction, and the third color only selects which side of that slice survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert colors, "expected at least one foreground color"

    color_counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    grey_count = max(color_counts.values())
    grey_candidates = [color for color, count in color_counts.items() if count == grey_count]
    assert len(grey_candidates) == 1, "expected one dominant grey color"
    grey = grey_candidates[0]

    out = np.full_like(g, int(bg))
    for comp in _components(g == grey):
        special: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    value = int(g[nr, nc])
                    if value != bg and value != grey:
                        special.add((int(nr), int(nc)))
        assert len(special) == 3, "each grey shape must touch exactly two markers and one guide"

        by_color: dict[int, list[tuple[int, int]]] = {}
        for cell in sorted(special):
            by_color.setdefault(int(g[cell]), []).append(cell)
        sizes = sorted(len(cells) for cells in by_color.values())
        assert sizes == [1, 2], "expected one singleton guide color and one paired marker color"

        marker_cells = next(cells for cells in by_color.values() if len(cells) == 2)
        guide = next(cells[0] for cells in by_color.values() if len(cells) == 1)
        (r0, c0), (r1, c1) = marker_cells

        if r0 == r1:
            cut_row = int(r0)
            upper = [(r, c) for r, c in comp if r < cut_row]
            lower = [(r, c) for r, c in comp if r > cut_row]
            assert upper and lower, "horizontal cut must leave cells on both sides"
            du = _distance_to_cells(guide, upper)
            dl = _distance_to_cells(guide, lower)
            assert du != dl, "guide must choose one side unambiguously"
            keep = [(r, c) for r, c in comp if r <= cut_row] if du < dl else [(r, c) for r, c in comp if r >= cut_row]
        else:
            assert c0 == c1, "marker pair must share a row or a column"
            cut_col = int(c0)
            left = [(r, c) for r, c in comp if c < cut_col]
            right = [(r, c) for r, c in comp if c > cut_col]
            assert left and right, "vertical cut must leave cells on both sides"
            dl = _distance_to_cells(guide, left)
            dr = _distance_to_cells(guide, right)
            assert dl != dr, "guide must choose one side unambiguously"
            keep = [(r, c) for r, c in comp if c <= cut_col] if dl < dr else [(r, c) for r, c in comp if c >= cut_col]

        for r, c in keep:
            out[r, c] = int(grey)
    return out


def _next_interval(prev_left: int, prev_right: int, width: int) -> tuple[int, int]:
    for _ in range(60):
        left = prev_left + random.choice((-1, 0, 0, 1))
        right = prev_right + random.choice((-1, 0, 0, 1))
        left = max(0, min(left, width - 3))
        right = max(left + 2, min(right, width - 1))
        if left <= prev_right and right >= prev_left:
            return left, right
    return prev_left, prev_right


def _horizontal_body(height: int, width: int, cut_row: int) -> np.ndarray:
    rows: list[tuple[int, int] | None] = [None] * height
    rows[cut_row] = (0, width - 1)

    left, right = rows[cut_row]
    for r in range(cut_row - 1, -1, -1):
        left, right = _next_interval(int(left), int(right), width)
        rows[r] = (left, right)

    left, right = rows[cut_row]
    for r in range(cut_row + 1, height):
        left, right = _next_interval(int(left), int(right), width)
        rows[r] = (left, right)

    body = np.zeros((height, width), dtype=bool)
    for r, interval in enumerate(rows):
        assert interval is not None
        left, right = interval
        body[r, int(left) : int(right) + 1] = True
    return body


def _generate_body(orientation: str) -> tuple[np.ndarray, int]:
    if orientation == "H":
        height = random.randint(6, 10)
        width = random.randint(6, 11)
        cut_row = random.randint(2, height - 3)
        return _horizontal_body(height, width, cut_row), cut_row

    base_height = random.randint(6, 11)
    base_width = random.randint(6, 10)
    base_cut = random.randint(2, base_height - 3)
    return _horizontal_body(base_height, base_width, base_cut).T, base_cut


def _marker_positions(body: np.ndarray, orientation: str, cut_index: int) -> list[tuple[int, int]]:
    h, w = map(int, body.shape)
    if orientation == "H":
        return [(cut_index, 0), (cut_index, w - 1)]
    return [(0, cut_index), (h - 1, cut_index)]


def _inside_candidates(
    body: np.ndarray,
    orientation: str,
    cut_index: int,
    keep_side: str,
    markers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    marker_set = set(markers)
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(body):
        rr, cc = int(r), int(c)
        if (rr, cc) in marker_set:
            continue
        if orientation == "H":
            if keep_side == "upper" and rr >= cut_index:
                continue
            if keep_side == "lower" and rr <= cut_index:
                continue
        else:
            if keep_side == "left" and cc >= cut_index:
                continue
            if keep_side == "right" and cc <= cut_index:
                continue
        temp = body.copy()
        for mr, mc in markers:
            temp[mr, mc] = False
        temp[rr, cc] = False
        if _is_connected(temp):
            candidates.append((rr, cc))
    random.shuffle(candidates)
    return candidates


def _outside_candidates(
    body: np.ndarray,
    orientation: str,
    cut_index: int,
    keep_side: str,
) -> list[tuple[int, int]]:
    h, w = map(int, body.shape)
    if orientation == "H":
        dr, dc = (-1, 0) if keep_side == "upper" else (1, 0)
    else:
        dr, dc = (0, -1) if keep_side == "left" else (0, 1)

    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(body):
        rr, cc = int(r), int(c)
        if orientation == "H":
            if keep_side == "upper" and rr >= cut_index:
                continue
            if keep_side == "lower" and rr <= cut_index:
                continue
        else:
            if keep_side == "left" and cc >= cut_index:
                continue
            if keep_side == "right" and cc <= cut_index:
                continue
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < h and 0 <= nc < w and body[nr, nc]:
            continue
        candidates.append((rr + 1 + dr, cc + 1 + dc))
    random.shuffle(candidates)
    return candidates


def _make_patch(marker_color: int, guide_color: int) -> np.ndarray:
    for _ in range(PATCH_ATTEMPTS):
        orientation = random.choice(("H", "V"))
        body, cut_index = _generate_body(orientation)
        h, w = map(int, body.shape)
        markers = _marker_positions(body, orientation, cut_index)
        keep_side = random.choice(("upper", "lower")) if orientation == "H" else random.choice(("left", "right"))

        patch = np.full((h + 2, w + 2), BG, dtype=int)
        patch[1 : h + 1, 1 : w + 1][body] = GREY
        for mr, mc in markers:
            patch[1 + mr, 1 + mc] = marker_color

        use_inside = random.choice((True, False))
        inside = _inside_candidates(body, orientation, cut_index, keep_side, markers)
        outside = _outside_candidates(body, orientation, cut_index, keep_side)

        if use_inside and inside:
            ir, ic = inside[0]
            patch[1 + ir, 1 + ic] = guide_color
        elif outside:
            pr, pc = outside[0]
            patch[pr, pc] = guide_color
        elif inside:
            ir, ic = inside[0]
            patch[1 + ir, 1 + ic] = guide_color
        else:
            continue

        grey_mask = patch == GREY
        if not _is_connected(grey_mask):
            continue
        special = set()
        for r, c in np.argwhere(grey_mask):
            rr, cc = int(r), int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1]:
                    value = int(patch[nr, nc])
                    if value not in (BG, GREY):
                        special.add((nr, nc))
        if len(special) != 3:
            continue
        return patch
    raise RuntimeError("failed to generate a valid local patch")


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    occupied = {(int(r), int(c)) for r, c in np.argwhere(patch != BG)}
    ph, pw = map(int, patch.shape)
    h, w = map(int, grid.shape)
    for r, c in occupied:
        gr, gc = top + r, left + c
        if not (0 <= gr < h and 0 <= gc < w):
            return False
        if int(grid[gr, gc]) != BG:
            return False
        for nr in range(gr - 1, gr + 2):
            for nc in range(gc - 1, gc + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                local = (nr - top, nc - left)
                if 0 <= local[0] < ph and 0 <= local[1] < pw and local in occupied:
                    continue
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    sum_h = sum(int(p.shape[0]) for p in patches)
    sum_w = sum(int(p.shape[1]) for p in patches)

    min_h = max(12, max_h + 1)
    min_w = max(12, max_w + 1)
    max_h_lim = min(30, max(min_h, sum_h + 4))
    max_w_lim = min(30, max(min_w, sum_w + 4))

    for _ in range(PACK_ATTEMPTS):
        h = random.randint(min_h, max_h_lim)
        w = random.randint(min_w, max_w_lim)
        grid = np.full((h, w), BG, dtype=int)
        ok = True
        for patch in random.sample(patches, len(patches)):
            ph, pw = map(int, patch.shape)
            placed = False
            for _ in range(PLACE_ATTEMPTS):
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
                if _can_place(grid, patch, top, left):
                    sub = grid[top : top + ph, left : left + pw]
                    mask = patch != BG
                    sub[mask] = patch[mask]
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and _most_common_color(grid) == BG:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    shape_count = random.randint(1, 3)
    colors = random.sample(PALETTE, 2 * shape_count)
    patches = [
        _make_patch(colors[2 * i], colors[2 * i + 1])
        for i in range(shape_count)
    ]
    grid = _pack_patches(patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not np.all((out == BG) | (out == GREY)):
        return None
    if np.count_nonzero(grid == GREY) <= np.count_nonzero(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the cut-and-keep puzzle")
