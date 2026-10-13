from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 300
PLACE_RETRIES = 200
MIN_SIZE = 20
MAX_SIZE = 30
EDGE_MARGIN = 3
RESERVE_MARGIN = 3


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _flood_outside(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    outside = np.zeros_like(blocked, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _component_context(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(coords)
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)

    boundary = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
    boundary[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
    outside = _flood_outside(boundary)
    interior = (~boundary) & (~outside)
    return pr0, pc0, boundary, outside, interior


def _outer_red_halo(filled: np.ndarray, outside: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    halo = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not outside[r, c]:
                continue
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and filled[nr, nc]:
                    halo[r, c] = True
                    break
    return halo


def _blue_targets(boundary: np.ndarray, outside: np.ndarray, filled: np.ndarray) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    center = np.mean(np.argwhere(filled), axis=0)
    h, w = boundary.shape

    for r, c in np.argwhere(boundary):
        r = int(r)
        c = int(c)
        neigh_dirs = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and boundary[nr, nc]:
                neigh_dirs.append((dr, dc))
        if len(neigh_dirs) != 2:
            continue
        if neigh_dirs[0][0] == neigh_dirs[1][0] or neigh_dirs[0][1] == neigh_dirs[1][1]:
            continue

        sdr = neigh_dirs[0][0] + neigh_dirs[1][0]
        sdc = neigh_dirs[0][1] + neigh_dirs[1][1]
        candidates = []
        for dr, dc in ((sdr, sdc), (-sdr, -sdc)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and outside[nr, nc]:
                score = (nr - center[0]) ** 2 + (nc - center[1]) ** 2
                candidates.append((score, (nr, nc)))
        if candidates:
            candidates.sort()
            targets.add(candidates[-1][1])
    return targets


def _extreme_specs(coords: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    top_r = int(coords[:, 0].min())
    bottom_r = int(coords[:, 0].max())
    left_c = int(coords[:, 1].min())
    right_c = int(coords[:, 1].max())

    top = coords[coords[:, 0] == top_r]
    bottom = coords[coords[:, 0] == bottom_r]
    left = coords[coords[:, 1] == left_c]
    right = coords[coords[:, 1] == right_c]

    top_pt = (top_r, int(top[:, 1].min()))
    bottom_pt = (bottom_r, int(bottom[:, 1].min()))
    left_pt = (int(left[:, 0].min()), left_c)
    right_pt = (int(right[:, 0].min()), right_c)

    return [
        (top_pt, (-1, 0)),
        (bottom_pt, (1, 0)),
        (left_pt, (0, -1)),
        (right_pt, (0, 1)),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-coded pattern outward from each closed grey outline while erasing the seed itself.

    Input structure:
    - The background is black.
    - The grid contains one or more disjoint grey one-pixel-thick closed outlines.
    - The strict interior of each outline contains exactly one non-grey seed pixel,
      whose color is red, blue, or green; all remaining interior cells are background.

    Rule:
    - Keep every grey outline unchanged and erase each interior seed back to background.
    - Red seed: color the one-cell outside halo of that shape red.
    - Blue seed: for each boundary corner pixel whose two grey 4-neighbors form a right angle,
      color the outward diagonal cell blue.
    - Green seed: from the topmost-leftmost, bottommost-leftmost, leftmost-topmost,
      and rightmost-topmost boundary pixels, draw a two-cell outward line in the
      corresponding cardinal direction using green.

    Key insight:
    - The seed color is a branch selector: the same kind of closed outline can trigger
      either a full outer halo, only corner diagonals, or only four extremal rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _components(g == GREY):
        pr0, pc0, boundary, outside, interior = _component_context(g, coords)
        patch = g[pr0 : pr0 + boundary.shape[0], pc0 : pc0 + boundary.shape[1]]

        seed_mask = interior & (patch != BG)
        seed_coords = np.argwhere(seed_mask)
        assert seed_coords.shape == (1, 2), "each outline interior must contain exactly one seed"
        sr, sc = map(int, seed_coords[0])
        seed_color = int(patch[sr, sc])
        assert seed_color in {RED, BLUE, GREEN}
        assert np.all(patch[interior & ~seed_mask] == BG), "interior must be background except for the seed"

        out[pr0 + sr, pc0 + sc] = BG
        filled = boundary | interior

        if seed_color == RED:
            halo = _outer_red_halo(filled, outside)
            out_patch = out[pr0 : pr0 + boundary.shape[0], pc0 : pc0 + boundary.shape[1]]
            out_patch[halo] = RED
        elif seed_color == BLUE:
            out_patch = out[pr0 : pr0 + boundary.shape[0], pc0 : pc0 + boundary.shape[1]]
            for tr, tc in _blue_targets(boundary, outside, filled):
                out_patch[tr, tc] = BLUE
        else:
            for (r, c), (dr, dc) in _extreme_specs(coords):
                for step in (1, 2):
                    nr = r + dr * step
                    nc = c + dc * step
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == BG:
                        out[nr, nc] = GREEN

    return out


def _filled_to_boundary_and_interior(filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = filled.shape
    boundary = np.zeros_like(filled, dtype=bool)
    for r, c in np.argwhere(filled):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not filled[nr, nc]:
                boundary[r, c] = True
                break
    interior = filled & ~boundary
    return boundary, interior


def _make_rectangle() -> np.ndarray:
    h = random.randint(5, 9)
    w = random.randint(5, 9)
    return np.ones((h, w), dtype=bool)


def _make_plus() -> np.ndarray:
    h = random.choice([7, 9, 11])
    w = random.choice([7, 9, 11])
    vert_w = random.choice([3, 5])
    horiz_h = random.choice([3, 5])
    if vert_w >= w:
        vert_w = 3
    if horiz_h >= h:
        horiz_h = 3

    mask = np.zeros((h, w), dtype=bool)
    vc0 = w // 2 - vert_w // 2
    hr0 = h // 2 - horiz_h // 2
    mask[:, vc0 : vc0 + vert_w] = True
    mask[hr0 : hr0 + horiz_h, :] = True
    return mask


def _make_l_shape() -> np.ndarray:
    h = random.randint(7, 10)
    w = random.randint(7, 10)
    thickness = random.randint(3, min(4, h - 2, w - 2))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[h - thickness :, :] = True

    orientation = random.choice([0, 1, 2, 3])
    if orientation == 1:
        mask = np.fliplr(mask)
    elif orientation == 2:
        mask = np.flipud(mask)
    elif orientation == 3:
        mask = np.flipud(np.fliplr(mask))
    return mask


def _make_t_shape() -> np.ndarray:
    h = random.randint(7, 10)
    w = random.randint(7, 10)
    top_h = random.randint(3, min(4, h - 2))
    stem_w = random.randint(3, min(4, w - 2))

    mask = np.zeros((h, w), dtype=bool)
    mask[:top_h, :] = True
    c0 = w // 2 - stem_w // 2
    mask[:, c0 : c0 + stem_w] = True

    orientation = random.choice([0, 1, 2, 3])
    if orientation == 1:
        mask = np.rot90(mask, 1)
    elif orientation == 2:
        mask = np.rot90(mask, 2)
    elif orientation == 3:
        mask = np.rot90(mask, 3)
    return mask


def _make_step_shape() -> np.ndarray:
    top_h = random.randint(3, 4)
    bottom_h = random.randint(3, 4)
    left_w = random.randint(4, 6)
    right_w = random.randint(4, 6)
    overlap_h = random.randint(2, min(top_h, bottom_h))
    offset = random.randint(2, 4)

    h = top_h + bottom_h - overlap_h
    w = max(left_w, offset + right_w)
    mask = np.zeros((h, w), dtype=bool)
    mask[:top_h, :left_w] = True
    mask[top_h - overlap_h :, offset : offset + right_w] = True

    orientation = random.choice([0, 1, 2, 3])
    if orientation:
        mask = np.rot90(mask, orientation)
    return mask


def _sample_shape() -> np.ndarray:
    builders = [_make_rectangle, _make_plus, _make_l_shape, _make_t_shape, _make_step_shape]
    for _ in range(50):
        filled = random.choice(builders)()
        boundary, interior = _filled_to_boundary_and_interior(filled)
        if np.any(interior) and len(_components(boundary)) == 1 and len(_components(interior)) == 1:
            return filled
    raise RuntimeError("failed to sample a valid filled shape")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _place_shapes(grid_shape: tuple[int, int], shapes: list[np.ndarray]) -> list[tuple[np.ndarray, int, int]] | None:
    h, w = grid_shape
    placed: list[tuple[np.ndarray, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    for filled in shapes:
        ph, pw = filled.shape
        if ph + 2 * EDGE_MARGIN > h or pw + 2 * EDGE_MARGIN > w:
            return None

        ok = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(EDGE_MARGIN, h - ph - EDGE_MARGIN)
            c0 = random.randint(EDGE_MARGIN, w - pw - EDGE_MARGIN)
            box = (
                r0 - RESERVE_MARGIN,
                c0 - RESERVE_MARGIN,
                r0 + ph - 1 + RESERVE_MARGIN,
                c0 + pw - 1 + RESERVE_MARGIN,
            )
            if any(_boxes_overlap(box, other) for other in reserved):
                continue
            placed.append((filled, r0, c0))
            reserved.append(box)
            ok = True
            break
        if not ok:
            return None

    return placed


def _render_case(grid_shape: tuple[int, int], placed: list[tuple[np.ndarray, int, int]], seed_colors: list[int]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)

    for (filled, r0, c0), seed_color in zip(placed, seed_colors):
        boundary, interior = _filled_to_boundary_and_interior(filled)
        interior_coords = np.argwhere(interior)
        assert interior_coords.size > 0
        sr, sc = map(int, interior_coords[random.randrange(len(interior_coords))])

        grid[r0 : r0 + filled.shape[0], c0 : c0 + filled.shape[1]][boundary] = GREY
        grid[r0 + sr, c0 + sc] = seed_color

    return grid


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice([3, 4])
    grid_shape = (
        random.randint(MIN_SIZE, MAX_SIZE),
        random.randint(MIN_SIZE, MAX_SIZE),
    )

    seed_colors = [RED, BLUE, GREEN]
    if object_count == 4:
        seed_colors.append(random.choice([RED, BLUE, GREEN]))
    random.shuffle(seed_colors)

    shapes = [_sample_shape() for _ in range(object_count)]
    placed = _place_shapes(grid_shape, shapes)
    if placed is None:
        return None

    inp = _render_case(grid_shape, placed, seed_colors)

    try:
        out = main(inp)
    except Exception:
        return None

    if out.shape != inp.shape or np.array_equal(out, inp):
        return None
    if not np.all(np.isin(inp, [BG, BLUE, RED, GREEN, GREY])):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
