from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PINK = 6

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 13
GRID_MAX = 19
MARKER_SIZES = (3, 4)
BRANCHES = ("quadrant", "cross_h", "cross_v")
BRANCH_WEIGHTS = np.array([0.34, 0.33, 0.33], dtype=float)
GEN_ATTEMPTS = 300

FRAGMENT_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)

RNG = np.random.default_rng()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _flood_passable(grid: np.ndarray, start: tuple[int, int], passable: set[int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    if int(grid[sr, sc]) not in passable:
        return set()
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in seen or not _in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) not in passable:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _mirror_h(r: int, axis: int) -> int:
    return 2 * axis - r


def _mirror_v(c: int, axis: int) -> int:
    return 2 * axis - c


def _blue_symmetric_visible(grid: np.ndarray, axis: int, horizontal: bool) -> bool:
    h, w = grid.shape
    for r, c in np.argwhere(grid == BLUE):
        rr, cc = (int(_mirror_h(int(r), axis)), int(c)) if horizontal else (int(r), int(_mirror_v(int(c), axis)))
        if not _in_bounds((h, w), rr, cc):
            return False
        if int(grid[rr, cc]) not in (BLUE, PINK):
            return False
    return True


def _find_unique_axis(grid: np.ndarray, horizontal: bool) -> int:
    limit = grid.shape[0] if horizontal else grid.shape[1]
    axes = [axis for axis in range(limit) if _blue_symmetric_visible(grid, axis, horizontal)]
    assert len(axes) == 1, f"expected unique {'horizontal' if horizontal else 'vertical'} axis, got {axes}"
    return int(axes[0])


def _marker_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == PINK)
    assert pts.size > 0, "missing marker rectangle"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == PINK), "marker must be a solid rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase non-boundary blue disguise fragments, then restore the magenta-occluded patch from the cleaned figure's double reflection symmetry.

    Input structure:
    - The grid uses black background, one blue foreground figure, one solid magenta rectangle, and one red single-pixel corner marker.
    - The blue figure is the true boundary-touching structure, while extra blue 2-4 pixel fragments may sit strictly inside the grid and remain disconnected from the boundary.
    - If those fragments are erased and the magenta rectangle is removed, the remaining blue/black pattern is symmetric about one horizontal axis and one vertical axis.

    Rule:
    - Start from the red corner pixel and flood-fill 4-connected black background cells; blue cells block the flood.
    - Remove every blue connected component that does not touch the outer grid boundary, because those components are decorative fragments rather than part of the true structure.
    - Remove the red corner marker to black.
    - Determine the unique horizontal and vertical reflection axes from the cleaned visible blue pixels, treating magenta cells as unknown hidden positions.
    - For each magenta cell, copy the color from a non-hidden horizontal reflection mate if available, otherwise from a non-hidden vertical reflection mate; if both direct mates are hidden, use the 180-degree rotational mate.

    Key insight:
    - The visible blue figure is only symmetric after the internal blue disguises are erased, so symmetry-based completion of the hidden rectangle is impossible until reachability-style fragment cleanup happens first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    red_corners = [(r, c) for r, c in corners if int(g[r, c]) == RED]
    assert len(red_corners) == 1, "expected exactly one red corner marker"
    start = red_corners[0]

    _ = _flood_passable(g, start, {BLACK, RED})

    work = g.copy()
    for comp in _components(work == BLUE):
        if not _touches_boundary(comp, work.shape):
            work[comp[:, 0], comp[:, 1]] = BLACK

    work[start] = BLACK

    r_axis = _find_unique_axis(work, horizontal=True)
    c_axis = _find_unique_axis(work, horizontal=False)

    reference = work.copy()
    out = work.copy()
    for r, c in np.argwhere(reference == PINK):
        r = int(r)
        c = int(c)
        hr, hc = _mirror_h(r, r_axis), c
        vr, vc = r, _mirror_v(c, c_axis)
        rr, rc = _mirror_h(r, r_axis), _mirror_v(c, c_axis)

        vals: list[int] = []
        if _in_bounds(reference.shape, hr, hc) and int(reference[hr, hc]) != PINK:
            vals.append(int(reference[hr, hc]))
        if _in_bounds(reference.shape, vr, vc) and int(reference[vr, vc]) != PINK:
            vals.append(int(reference[vr, vc]))
        if not vals:
            assert _in_bounds(reference.shape, rr, rc) and int(reference[rr, rc]) != PINK
            vals.append(int(reference[rr, rc]))
        assert len(set(vals)) == 1, "hidden cell must have a unique reflected value"
        out[r, c] = vals[0]

    return out


def _paint_orbit_rect(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = mask.shape
    cr = h // 2
    cc = w // 2
    r0, c0, r1, c1 = bbox
    rects = {
        (r0, c0, r1, c1),
        (_mirror_h(r1, cr), c0, _mirror_h(r0, cr), c1),
        (r0, _mirror_v(c1, cc), r1, _mirror_v(c0, cc)),
        (_mirror_h(r1, cr), _mirror_v(c1, cc), _mirror_h(r0, cr), _mirror_v(c0, cc)),
    }
    for rr0, cc0, rr1, cc1 in rects:
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _build_clean_pattern(h: int, w: int) -> np.ndarray | None:
    cr = h // 2
    cc = w // 2
    for _ in range(40):
        mask = np.zeros((h, w), dtype=bool)

        v_thick = int(RNG.choice([1, 3]))
        h_thick = int(RNG.choice([1, 3]))
        v_half = v_thick // 2
        h_half = h_thick // 2

        mask[:, cc - v_half : cc + v_half + 1] = True
        mask[cr - h_half : cr + h_half + 1, :] = True

        shelf_count = int(RNG.integers(3, 6))
        for _ in range(shelf_count):
            fw = int(RNG.integers(2, min(6, cc) + 1))
            fh = int(RNG.integers(1, 4))
            max_r0 = max(2, cr - h_half - fh + 1)
            r0 = int(RNG.integers(1, max_r0))
            r1 = r0 + fh - 1
            c1 = cc - v_half
            c0 = max(1, c1 - fw + 1)
            _paint_orbit_rect(mask, (r0, c0, r1, c1))

        pillar_count = int(RNG.integers(3, 6))
        for _ in range(pillar_count):
            fh = int(RNG.integers(2, min(6, cr) + 1))
            fw = int(RNG.integers(1, 4))
            max_c0 = max(2, cc - v_half - fw + 1)
            c0 = int(RNG.integers(1, max_c0))
            c1 = c0 + fw - 1
            r1 = cr - h_half
            r0 = max(1, r1 - fh + 1)
            _paint_orbit_rect(mask, (r0, c0, r1, c1))

        grid = np.zeros((h, w), dtype=int)
        grid[mask] = BLUE

        blue_ratio = float(mask.mean())
        if not (0.18 <= blue_ratio <= 0.70):
            continue
        if not all(int(grid[r, c]) == BLACK for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
            continue
        comps = _components(grid == BLUE)
        if len(comps) == 1 and _touches_boundary(comps[0], grid.shape):
            return grid
    return None


def _branch_ok(
    bbox: tuple[int, int, int, int],
    branch: str,
    cr: int,
    cc: int,
) -> bool:
    r0, c0, r1, c1 = bbox
    cross_h = r0 <= cr <= r1
    cross_v = c0 <= cc <= c1
    if r0 <= cr <= r1 and c0 <= cc <= c1:
        return False
    if cross_h and cross_v:
        return False
    if branch == "quadrant":
        return not cross_h and not cross_v
    if branch == "cross_h":
        return cross_h and not cross_v
    return cross_v and not cross_h


def _marker_candidate_ok(clean: np.ndarray, bbox: tuple[int, int, int, int], branch: str) -> bool:
    h, w = clean.shape
    cr = h // 2
    cc = w // 2
    r0, c0, r1, c1 = bbox
    if not _branch_ok(bbox, branch, cr, cc):
        return False

    patch = clean[r0 : r1 + 1, c0 : c1 + 1]
    if len(np.unique(patch)) < 2:
        return False

    masked = clean.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = PINK

    if not _blue_symmetric_visible(masked, cr, True):
        return False
    if not _blue_symmetric_visible(masked, cc, False):
        return False

    visible = clean.copy()
    visible[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for comp in _components(visible == BLUE):
        if not _touches_boundary(comp, visible.shape):
            return False

    needs_hidden_h = False
    needs_hidden_v = False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            hr = _mirror_h(r, cr)
            vc = _mirror_v(c, cc)
            h_hidden = r0 <= hr <= r1
            v_hidden = c0 <= vc <= c1
            if h_hidden and v_hidden:
                return False
            needs_hidden_h |= h_hidden
            needs_hidden_v |= v_hidden

    if branch == "cross_h" and not needs_hidden_h:
        return False
    if branch == "cross_v" and not needs_hidden_v:
        return False
    return True


def _marker_candidates(clean: np.ndarray, branch: str) -> list[tuple[int, int, int, int]]:
    h, w = clean.shape
    candidates: list[tuple[int, int, int, int]] = []
    for mh in MARKER_SIZES:
        for mw in MARKER_SIZES:
            for r0 in range(0, h - mh + 1):
                for c0 in range(0, w - mw + 1):
                    bbox = (r0, c0, r0 + mh - 1, c0 + mw - 1)
                    if _marker_candidate_ok(clean, bbox, branch):
                        candidates.append(bbox)
    return candidates


def _cells_from_shape(shape: tuple[tuple[int, int], ...], top: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top
    return [(r0 + dr, c0 + dc) for dr, dc in shape]


def _fragment_candidates(
    clean: np.ndarray,
    marker_bbox: tuple[int, int, int, int],
    existing: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    r0m, c0m, r1m, c1m = marker_bbox
    candidates: list[list[tuple[int, int]]] = []
    for shape in FRAGMENT_SHAPES:
        shape_cells = np.array(shape, dtype=int)
        sh = int(shape_cells[:, 0].max()) + 1
        sw = int(shape_cells[:, 1].max()) + 1
        for r0 in range(1, h - sh):
            for c0 in range(1, w - sw):
                cells = _cells_from_shape(shape, (r0, c0))
                cell_set = set(cells)
                if any(r0m <= r <= r1m and c0m <= c <= c1m for r, c in cells):
                    continue
                if any(int(clean[r, c]) != BLACK for r, c in cells):
                    continue
                if cell_set & existing:
                    continue
                ok = True
                for r, c in cells:
                    if r in (0, h - 1) or c in (0, w - 1):
                        ok = False
                        break
                    for dr, dc in NEIGHBORS4:
                        nr = r + dr
                        nc = c + dc
                        if not _in_bounds((h, w), nr, nc):
                            ok = False
                            break
                        if (nr, nc) in cell_set:
                            continue
                        if int(clean[nr, nc]) == BLUE or (nr, nc) in existing:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append(cells)
    return candidates


def _visible_blue_breaks_symmetry(grid: np.ndarray, cr: int, cc: int) -> bool:
    return not _blue_symmetric_visible(grid, cr, True) and not _blue_symmetric_visible(grid, cc, False)


def _place_fragments(clean: np.ndarray, marker_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, list[list[tuple[int, int]]]] | None:
    h, w = clean.shape
    cr = h // 2
    cc = w // 2
    target_count = int(RNG.integers(1, 4))
    for _ in range(80):
        fragments: list[list[tuple[int, int]]] = []
        used: set[tuple[int, int]] = set()

        candidates = _fragment_candidates(clean, marker_bbox, used)
        if not candidates:
            return None
        first = candidates[int(RNG.integers(0, len(candidates)))]
        fragments.append(first)
        used |= set(first)

        while len(fragments) < target_count:
            candidates = _fragment_candidates(clean, marker_bbox, used)
            if not candidates:
                break
            pick = candidates[int(RNG.integers(0, len(candidates)))]
            fragments.append(pick)
            used |= set(pick)

        grid = clean.copy()
        for frag in fragments:
            for r, c in frag:
                grid[r, c] = BLUE
        grid[marker_bbox[0] : marker_bbox[2] + 1, marker_bbox[1] : marker_bbox[3] + 1] = PINK
        if _visible_blue_breaks_symmetry(grid, cr, cc):
            return grid, fragments
    return None


def _validate_semantics(
    inp: np.ndarray,
    clean: np.ndarray,
    marker_bbox: tuple[int, int, int, int],
    fragment_count: int,
) -> bool:
    h, w = inp.shape
    cr = h // 2
    cc = w // 2

    if h % 2 == 0 or w % 2 == 0:
        return False

    red_corners = [(r, c) for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)) if int(inp[r, c]) == RED]
    if len(red_corners) != 1:
        return False

    if np.any(inp == RED) and np.count_nonzero(inp == RED) != 1:
        return False

    reach = _flood_passable(inp, red_corners[0], {BLACK, RED})
    if len(reach) < 6:
        return False

    blue_comps = _components(inp == BLUE)
    interior = [comp for comp in blue_comps if not _touches_boundary(comp, inp.shape)]
    if len(interior) != fragment_count:
        return False
    if any(comp.shape[0] < 2 or comp.shape[0] > 4 for comp in interior):
        return False

    masked = inp.copy()
    masked[red_corners[0]] = BLACK
    for comp in interior:
        masked[comp[:, 0], comp[:, 1]] = BLACK
    if not _blue_symmetric_visible(masked, cr, True):
        return False
    if not _blue_symmetric_visible(masked, cc, False):
        return False

    if not _visible_blue_breaks_symmetry(inp, cr, cc):
        return False

    r0, c0, r1, c1 = marker_bbox
    if _marker_bbox(inp) != marker_bbox:
        return False
    if r0 <= cr <= r1 and c0 <= cc <= c1:
        return False
    if len(np.unique(clean[r0 : r1 + 1, c0 : c1 + 1])) < 2:
        return False

    solved = main(inp)
    if not np.array_equal(solved, clean):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    h = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
    w = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
    if h % 2 == 0:
        h += 1
    if w % 2 == 0:
        w += 1
    if h > GRID_MAX or w > GRID_MAX:
        return None

    clean = _build_clean_pattern(h, w)
    if clean is None:
        return None

    branch = str(RNG.choice(BRANCHES, p=BRANCH_WEIGHTS))
    marker_options = _marker_candidates(clean, branch)
    if not marker_options:
        return None
    marker_bbox = marker_options[int(RNG.integers(0, len(marker_options)))]

    placed = _place_fragments(clean, marker_bbox)
    if placed is None:
        return None
    inp, fragments = placed

    corner = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))[int(RNG.integers(0, 4))]
    if int(inp[corner]) != BLACK:
        return None
    inp = inp.copy()
    inp[corner] = RED

    if not _validate_semantics(inp, clean, marker_bbox, len(fragments)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
