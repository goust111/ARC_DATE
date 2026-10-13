from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TEMPLATE = 1
OVERLAP = 2
WALL = 9

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (22, 28)
TEMPLATE_BBOX_CHOICES = [(3, 3), (3, 4), (4, 3)]
TEMPLATE_AREA_RANGE = (5, 8)
COPY_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    out = sorted((r - min_r, c - min_c) for r, c in cells)
    return out


def _right_frontier(rel_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cellset = set(rel_cells)
    return sorted((r, c) for r, c in rel_cells if (r, c + 1) not in cellset)


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blue_comps = _components(g == TEMPLATE)
    template_comps = [comp for comp in blue_comps if len(comp) > 1]
    assert len(template_comps) == 1, "expected exactly one blue template component"
    template_cells = sorted(template_comps[0])
    template_bbox = _bbox(template_cells)
    r0, c0, r1, c1 = template_bbox
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    rel_cells = sorted((r - r0, c - c0) for r, c in template_cells)
    frontier = _right_frontier(rel_cells)

    blue_singletons = [comp[0] for comp in blue_comps if len(comp) == 1]
    red_cells = sorted(map(tuple, np.argwhere(g == OVERLAP)))
    assert len(blue_singletons) == 1, "expected exactly one blue layout origin"
    assert 3 <= len(red_cells) <= 5, "expected 3-5 red layout markers"

    markers = [(blue_singletons[0][0], blue_singletons[0][1], True)]
    markers.extend((r, c, False) for r, c in red_cells)
    return {
        "grid": g,
        "shape": (h, w),
        "template_bbox": template_bbox,
        "template_rel": rel_cells,
        "frontier": frontier,
        "box_h": box_h,
        "box_w": box_w,
        "markers": markers,
    }


def _stamp_copies(parsed: dict) -> list[dict]:
    r0, c0, _r1, _c1 = parsed["template_bbox"]
    box_h = parsed["box_h"]
    box_w = parsed["box_w"]
    origin_r, origin_c, _ = parsed["markers"][0]
    copies: list[dict] = []
    for mr, mc, is_origin in parsed["markers"]:
        dr = mr - origin_r
        dc = mc - origin_c
        assert dr % box_h == 0 and dc % box_w == 0, "marker offsets must be whole template units"
        unit_r = dr // box_h
        unit_c = dc // box_w
        copies.append(
            {
                "top": (r0 + unit_r * box_h, c0 + unit_c * box_w),
                "is_origin": bool(is_origin),
            }
        )
    return copies


def _copy_cells(rel_cells: list[tuple[int, int]], top: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = top
    return [(tr + r, tc + c) for r, c in rel_cells]


def _simulate_slide(parsed: dict, copies: list[dict]) -> list[dict]:
    rel_cells = parsed["template_rel"]
    frontier = parsed["frontier"]
    g = parsed["grid"]
    h, w = g.shape
    wall_pixels = set(map(tuple, np.argwhere(g == WALL)))
    active = list(range(len(copies)))
    stopped_pixels: set[tuple[int, int]] = set()

    for _ in range(w + 2):
        if not active:
            break
        blocked_now: list[int] = []
        for idx in active:
            tr, tc = copies[idx]["top"]
            blocked = False
            for rr, cc in frontier:
                nr = tr + rr
                nc = tc + cc + 1
                if nc >= w or (nr, nc) in wall_pixels or (nr, nc) in stopped_pixels:
                    blocked = True
                    break
            if blocked:
                blocked_now.append(idx)

        movable = [idx for idx in active if idx not in set(blocked_now)]
        for idx in blocked_now:
            stopped_pixels.update(_copy_cells(rel_cells, copies[idx]["top"]))
        if not movable:
            active = []
            break
        for idx in movable:
            tr, tc = copies[idx]["top"]
            copies[idx]["top"] = (tr, tc + 1)
        active = movable
    else:
        raise AssertionError("slide did not settle within the grid width")

    assert not active
    return copies


def _render_output(parsed: dict, copies: list[dict]) -> tuple[np.ndarray, dict]:
    h, w = parsed["shape"]
    rel_cells = parsed["template_rel"]
    out = np.zeros((h, w), dtype=int)
    wall_mask = parsed["grid"] == WALL

    coverage = np.zeros((h, w), dtype=int)
    origin_mask = np.zeros((h, w), dtype=bool)
    cell_sets: list[set[tuple[int, int]]] = []
    for copy in copies:
        pts = set(_copy_cells(rel_cells, copy["top"]))
        cell_sets.append(pts)
        for r, c in pts:
            coverage[r, c] += 1
            if copy["is_origin"]:
                origin_mask[r, c] = True

    out[coverage > 0] = TEMPLATE
    out[origin_mask] = OVERLAP
    out[coverage >= 2] = OVERLAP
    out[wall_mask] = WALL

    pair_overlap_count = 0
    pair_overlap_pixels: set[tuple[int, int]] = set()
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            inter = cell_sets[i] & cell_sets[j]
            if inter:
                pair_overlap_count += 1
                pair_overlap_pixels |= inter

    meta = {
        "coverage": coverage,
        "origin_mask": origin_mask,
        "pair_overlap_count": pair_overlap_count,
        "pair_overlap_pixels": pair_overlap_pixels,
        "final_cell_sets": cell_sets,
    }
    return out, meta


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    parsed = _parse_input(grid)
    copies = _stamp_copies(parsed)
    final_copies = _simulate_slide(parsed, [dict(copy) for copy in copies])
    out, meta = _render_output(parsed, final_copies)
    meta["parsed"] = parsed
    meta["initial_copies"] = copies
    meta["final_copies"] = final_copies
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp template copies from a singleton layout, slide them rightward, and recolor the emergent overlap cells.

    Input structure:
    - Black background with a full maroon wall on the right edge.
    - A blue 4-connected template shape sits near the upper-left and a separate singleton layout contains exactly one blue origin pixel plus several red offset pixels.
    - Singleton offsets are integer multiples of the template bbox height and width.

    Rule:
    - Use the blue singleton as the layout origin, convert every singleton offset into template-grid units, and stamp one template copy per layout cell relative to the original template position.
    - Recolor the origin copy red, erase the source template and all singleton markers, then repeatedly slide every not-yet-stopped copy one column right whenever none of its own right-frontier pixels is immediately blocked by the wall or a previously stopped copy.
    - After all copies settle, color every covered cell blue except the origin copy and every cell covered by two or more copies, which are red; keep the maroon wall and black background.

    Key insight:
    - The singleton layout only determines where template copies start; the final red overlap pattern can only be found by simulating the rightward settling and then counting coverage on the settled sparse copies.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _random_template(rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    box_h, box_w = rng.choice(TEMPLATE_BBOX_CHOICES)
    area = rng.randint(TEMPLATE_AREA_RANGE[0], min(TEMPLATE_AREA_RANGE[1], box_h * box_w - 1))
    for _ in range(200):
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < area:
            r, c = rng.choice(tuple(cells))
            nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            nbrs = [(nr, nc) for nr, nc in nbrs if 0 <= nr < box_h and 0 <= nc < box_w]
            candidates = [pt for pt in nbrs if pt not in cells]
            if not candidates:
                break
            cells.add(rng.choice(candidates))
        if len(cells) != area:
            continue
        rel = _normalize(cells)
        if len(_components(_shape_mask(rel, box_h, box_w))) != 1:
            continue
        rows = {r for r, _ in rel}
        cols = {c for _, c in rel}
        if rows != set(range(box_h)) or cols != set(range(box_w)):
            continue
        if len(rel) == box_h * box_w:
            continue
        if all(sum(1 for rr, _ in rel if rr == r) == box_w for r in range(box_h)):
            continue
        contact = _same_row_contact_distance(rel, box_w)
        if contact is None:
            continue
        overlap = _same_row_overlap_count(rel, contact)
        if overlap <= 0 or overlap > max(4, area - 1):
            continue
        return rel, (box_h, box_w)
    return None


def _shape_mask(rel_cells: list[tuple[int, int]], box_h: int, box_w: int) -> np.ndarray:
    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in rel_cells:
        mask[r, c] = True
    return mask


def _same_row_contact_distance(rel_cells: list[tuple[int, int]], box_w: int) -> int | None:
    frontier = _right_frontier(rel_cells)
    cellset = set(rel_cells)
    for d in range(1, box_w):
        for r, c in frontier:
            if (r, c + 1 - d) in cellset:
                return d
    return None


def _same_row_overlap_count(rel_cells: list[tuple[int, int]], offset: int) -> int:
    cellset = set(rel_cells)
    count = 0
    for r, c in rel_cells:
        if (r, c - offset) in cellset:
            count += 1
    return count


def _random_polyomino(rng: random.Random, n: int) -> list[tuple[int, int]] | None:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < n:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        rel = _normalize(cells)
        r0, c0, r1, c1 = _bbox(rel)
        if r1 + 1 > 3 or c1 + 1 > 4:
            continue
        row_counts = Counter(r for r, _ in rel)
        col_counts = Counter(c for _, c in rel)
        if len(row_counts) < 2 or len(col_counts) < 2:
            continue
        if len(set(col_counts.values())) == 1:
            continue
        pair_capacity = sum(max(0, k - 1) for k in row_counts.values())
        if pair_capacity < 2:
            continue
        min_r = min(r for r, _ in rel)
        max_r = max(r for r, _ in rel)
        min_c = min(c for _, c in rel)
        max_c = max(c for _, c in rel)
        interiors = [(r, c) for r, c in rel if min_r < r < max_r and min_c < c < max_c]
        if not interiors:
            continue
        return rel
    return None


def _choose_origin(rng: random.Random, lattice: list[tuple[int, int]]) -> tuple[int, int]:
    min_r = min(r for r, _ in lattice)
    max_r = max(r for r, _ in lattice)
    min_c = min(c for _, c in lattice)
    max_c = max(c for _, c in lattice)
    interiors = [(r, c) for r, c in lattice if min_r < r < max_r and min_c < c < max_c]
    return rng.choice(interiors)


def _marker_positions(
    lattice: list[tuple[int, int]],
    origin: tuple[int, int],
    box_h: int,
    box_w: int,
    marker_origin: tuple[int, int],
) -> list[tuple[int, int, bool]]:
    or_r, or_c = origin
    mr0, mc0 = marker_origin
    out = []
    for lr, lc in lattice:
        rr = mr0 + (lr - or_r) * box_h
        cc = mc0 + (lc - or_c) * box_w
        out.append((rr, cc, (lr, lc) == origin))
    return sorted(out)


def _copy_positions_from_lattice(
    lattice: list[tuple[int, int]],
    origin: tuple[int, int],
    box_h: int,
    box_w: int,
    template_top: tuple[int, int],
) -> list[dict]:
    or_r, or_c = origin
    tr0, tc0 = template_top
    out = []
    for lr, lc in lattice:
        rr = tr0 + (lr - or_r) * box_h
        cc = tc0 + (lc - or_c) * box_w
        out.append({"top": (rr, cc), "is_origin": (lr, lc) == origin})
    return sorted(out, key=lambda d: (d["top"][0], d["top"][1], not d["is_origin"]))


def _all_fit(
    rel_cells: list[tuple[int, int]],
    copies: list[dict],
    shape: tuple[int, int],
    *,
    max_col: int | None = None,
) -> bool:
    h, w = shape
    for copy in copies:
        for r, c in _copy_cells(rel_cells, copy["top"]):
            if not (0 <= r < h and 0 <= c < w):
                return False
            if max_col is not None and c > max_col:
                return False
    return True


def _copies_disjoint(rel_cells: list[tuple[int, int]], copies: list[dict]) -> bool:
    used: set[tuple[int, int]] = set()
    for copy in copies:
        pts = set(_copy_cells(rel_cells, copy["top"]))
        if used & pts:
            return False
        used |= pts
    return True


def _place_marker_cluster(
    rng: random.Random,
    lattice: list[tuple[int, int]],
    origin: tuple[int, int],
    box_h: int,
    box_w: int,
    grid_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, bool]] | None:
    h, w = grid_shape
    or_r, or_c = origin
    min_dr = min(lr - or_r for lr, _ in lattice)
    max_dr = max(lr - or_r for lr, _ in lattice)
    min_dc = min(lc - or_c for _, lc in lattice)
    max_dc = max(lc - or_c for _, lc in lattice)
    row_lo = max(0, -min_dr * box_h)
    row_hi = min(h - 1 - max_dr * box_h, h - 1)
    col_lo = max(0, -min_dc * box_w)
    col_hi = min(w - 2 - max_dc * box_w, w - 2)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    tr0, tc0, tr1, tc1 = template_bbox
    for _ in range(200):
        mr0 = rng.randint(max(row_lo, tr1 + 3), row_hi) if tr1 + 3 <= row_hi else rng.randint(row_lo, row_hi)
        mc0 = rng.randint(col_lo, col_hi)
        markers = _marker_positions(lattice, origin, box_h, box_w, (mr0, mc0))
        coords = [(r, c) for r, c, _ in markers]
        if len(coords) != len(set(coords)):
            continue
        if any(c == w - 1 for _, c in coords):
            continue
        if any(tr0 - 1 <= r <= tr1 + 1 and tc0 - 1 <= c <= tc1 + 1 for r, c in coords):
            continue
        return markers
    return None


def _build_input(
    rel_cells: list[tuple[int, int]],
    box_hw: tuple[int, int],
    lattice: list[tuple[int, int]],
    origin: tuple[int, int],
    grid_shape: tuple[int, int],
    template_top: tuple[int, int],
    markers: list[tuple[int, int, bool]],
) -> np.ndarray:
    h, w = grid_shape
    box_h, box_w = box_hw
    g = np.zeros((h, w), dtype=int)
    g[:, w - 1] = WALL
    tr0, tc0 = template_top
    for rr, cc in _copy_cells(rel_cells, (tr0, tc0)):
        g[rr, cc] = TEMPLATE
    for r, c, is_origin in markers:
        g[r, c] = TEMPLATE if is_origin else OVERLAP
    return g


def _validate_generated(
    grid: np.ndarray,
    rel_cells: list[tuple[int, int]],
    lattice: list[tuple[int, int]],
    origin: tuple[int, int],
    template_top: tuple[int, int],
    box_hw: tuple[int, int],
) -> bool:
    box_h, box_w = box_hw
    h, w = grid.shape
    marker_positions = sorted(map(tuple, np.argwhere((grid == TEMPLATE) | (grid == OVERLAP))))
    if len(marker_positions) != 1 + (len(lattice) - 1) + len(rel_cells):
        return False
    if not np.all(grid[:, w - 1] == WALL):
        return False

    initial_copies = _copy_positions_from_lattice(lattice, origin, box_h, box_w, template_top)
    if not _all_fit(rel_cells, initial_copies, (h, w), max_col=w - 4):
        return False
    if not _copies_disjoint(rel_cells, initial_copies):
        return False

    out, meta = _solve(grid)
    pair_overlap_count = meta["pair_overlap_count"]
    overlap_pixels = meta["pair_overlap_pixels"]
    coverage = meta["coverage"]
    origin_mask = meta["origin_mask"]

    if pair_overlap_count < 2:
        return False
    if not overlap_pixels:
        return False
    if np.any((grid == WALL) & (coverage > 0)):
        return False
    if np.any(coverage >= 3):
        return False
    if len(overlap_pixels) > max(6, len(rel_cells)):
        return False
    if np.count_nonzero(origin_mask & (coverage == 1)) == 0:
        return False
    if np.count_nonzero((coverage == 1) & ~origin_mask) == 0:
        return False
    if np.count_nonzero(out == TEMPLATE) == 0:
        return False
    if np.count_nonzero(out == OVERLAP) <= len(rel_cells):
        return False

    row_counts = Counter(r for r, _ in lattice)
    if sum(max(0, k - 1) for k in row_counts.values()) < 2:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template = _random_template(rng)
    if template is None:
        return None
    rel_cells, (box_h, box_w) = template

    copy_count = rng.randint(*COPY_COUNT_RANGE)
    lattice = _random_polyomino(rng, copy_count)
    if lattice is None:
        return None
    origin = _choose_origin(rng, lattice)

    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    min_dr = min(lr - origin[0] for lr, _ in lattice)
    max_dr = max(lr - origin[0] for lr, _ in lattice)
    min_dc = min(lc - origin[1] for _, lc in lattice)
    max_dc = max(lc - origin[1] for _, lc in lattice)

    row_lo = max(1, -min_dr * box_h)
    row_hi = min(4, grid_h - box_h - max_dr * box_h)
    col_lo = max(1, -min_dc * box_w)
    col_hi = min(8, grid_w - box_w - max_dc * box_w - 3)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    template_top = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
    initial_copies = _copy_positions_from_lattice(lattice, origin, box_h, box_w, template_top)
    if not _all_fit(rel_cells, initial_copies, (grid_h, grid_w), max_col=grid_w - 4):
        return None
    if not _copies_disjoint(rel_cells, initial_copies):
        return None

    template_bbox = (
        template_top[0],
        template_top[1],
        template_top[0] + box_h - 1,
        template_top[1] + box_w - 1,
    )
    markers = _place_marker_cluster(
        rng,
        lattice,
        origin,
        box_h,
        box_w,
        (grid_h, grid_w),
        template_bbox,
    )
    if markers is None:
        return None

    grid = _build_input(rel_cells, (box_h, box_w), lattice, origin, (grid_h, grid_w), template_top, markers)
    if not _validate_generated(grid, rel_cells, lattice, origin, template_top, (box_h, box_w)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
