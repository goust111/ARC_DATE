import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
DYE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_cavities(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cavities: list[dict] = []
    for coords in _components(g == BLACK):
        if _touches_edge(coords, h, w):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(window == BLACK):
            continue
        if not (
            r0 > 0
            and c0 > 0
            and r1 < h - 1
            and c1 < w - 1
            and np.all(g[r0 - 1, c0 : c1 + 1] == GREY)
            and np.all(g[r1 + 1, c0 : c1 + 1] == GREY)
            and np.all(g[r0 : r1 + 1, c0 - 1] == GREY)
            and np.all(g[r0 : r1 + 1, c1 + 1] == GREY)
        ):
            continue
        cavities.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
            }
        )
    cavities.sort(key=lambda cav: cav["bbox"])
    return cavities


def _dyes(grid: np.ndarray) -> list[tuple[int, int, int]]:
    pts = np.argwhere((grid != BLACK) & (grid != GREY))
    return [(int(r), int(c), int(grid[r, c])) for r, c in pts]


def _distance_to_cavity(cavity: dict, r: int, c: int, direction: str) -> int | None:
    r0, c0, r1, c1 = cavity["bbox"]
    if direction == "row":
        if not (r0 <= r <= r1):
            return None
        if c < c0:
            return int(c0 - c)
        if c > c1:
            return int(c - c1)
        return None
    if direction == "col":
        if not (c0 <= c <= c1):
            return None
        if r < r0:
            return int(r0 - r)
        if r > r1:
            return int(r - r1)
        return None
    raise ValueError(direction)


def _pairings_for_cavity(cavity: dict, dyes: list[tuple[int, int, int]]) -> tuple[list[tuple], list[tuple]]:
    row_hits: list[tuple[int, int, int, int]] = []
    col_hits: list[tuple[int, int, int, int]] = []
    for r, c, color in dyes:
        row_dist = _distance_to_cavity(cavity, r, c, "row")
        if row_dist is not None:
            row_hits.append((row_dist, r, c, color))
        col_dist = _distance_to_cavity(cavity, r, c, "col")
        if col_dist is not None:
            col_hits.append((col_dist, r, c, color))
    row_hits.sort()
    col_hits.sort()
    return row_hits, col_hits


def _upper_mask(height: int, width: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    if height == 1 or width == 1:
        return np.ones((height, width), dtype=bool)
    return rows * (width - 1) <= cols * (height - 1)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    dyes = _dyes(g)
    for cavity in _find_cavities(g):
        r0, c0, r1, c1 = cavity["bbox"]
        row_hits, col_hits = _pairings_for_cavity(cavity, dyes)
        total_hits = len(row_hits) + len(col_hits)
        if total_hits == 0:
            continue
        if total_hits == 1:
            color = row_hits[0][3] if row_hits else col_hits[0][3]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            continue

        if row_hits and col_hits:
            upper_color = row_hits[0][3]
            lower_color = col_hits[0][3]
        elif len(row_hits) >= 2:
            upper_color = row_hits[0][3]
            lower_color = row_hits[1][3]
        else:
            upper_color = col_hits[0][3]
            lower_color = col_hits[1][3]

        h, w = cavity["shape"]
        mask = _upper_mask(h, w)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[mask] = upper_color
        patch[~mask] = lower_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inject row- and column-aligned dye colors into hollow grey rings.

    Input structure:
    - The background is black.
    - Several closed grey outline rings appear on the grid; each ring has a one-pixel border and encloses a rectangular black cavity.
    - Scattered single-pixel dye markers of non-grey colors lie outside the rings and do not touch them.

    Rule:
    - Detect each ring by its enclosed rectangular black cavity surrounded by grey.
    - Pair a dye with every ring whose cavity shares the dye's row or whose cavity shares the dye's column.
    - If a ring gets one dye, fill its whole cavity with that dye color; if it gets two dyes, split the cavity by the top-left to bottom-right diagonal, using the row-paired dye above the diagonal and the column-paired dye below it.
    - If both dyes come from the same direction, sort them by distance to the cavity and use the nearer dye for the upper half and the farther dye for the lower half; the grey ring and dye pixels remain unchanged.

    Key insight:
    - The colors are not assigned by nearest object matching: the hidden relation is row/column alignment with a ring's cavity, and double matches resolve by a diagonal cavity split.
    """
    return _transform(np.asarray(grid, dtype=int))


def _build_ring(cavity_h: int, cavity_w: int, rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bulge_side = rng.choice([None, None, "top", "bottom", "left", "right"])
    top_extra = 2 if bulge_side == "top" else 1
    bottom_extra = 2 if bulge_side == "bottom" else 1
    left_extra = 2 if bulge_side == "left" else 1
    right_extra = 2 if bulge_side == "right" else 1

    h = cavity_h + top_extra + bottom_extra
    w = cavity_w + left_extra + right_extra
    patch = np.full((h, w), BLACK, dtype=int)

    r0 = top_extra
    c0 = left_extra
    r1 = r0 + cavity_h - 1
    c1 = c0 + cavity_w - 1
    rt, rb = r0 - 1, r1 + 1
    cl, cr = c0 - 1, c1 + 1

    patch[rt, cl : cr + 1] = GREY
    patch[rb, cl : cr + 1] = GREY
    patch[rt : rb + 1, cl] = GREY
    patch[rt : rb + 1, cr] = GREY

    if bulge_side == "top":
        x1 = rng.randint(cl + 1, cr - 2)
        x2 = rng.randint(x1 + 1, cr - 1)
        if x2 > x1 + 1:
            patch[rt, x1 + 1 : x2] = BLACK
        patch[rt - 1, x1 : x2 + 1] = GREY
    elif bulge_side == "bottom":
        x1 = rng.randint(cl + 1, cr - 2)
        x2 = rng.randint(x1 + 1, cr - 1)
        if x2 > x1 + 1:
            patch[rb, x1 + 1 : x2] = BLACK
        patch[rb + 1, x1 : x2 + 1] = GREY
    elif bulge_side == "left":
        y1 = rng.randint(rt + 1, rb - 2)
        y2 = rng.randint(y1 + 1, rb - 1)
        if y2 > y1 + 1:
            patch[y1 + 1 : y2, cl] = BLACK
        patch[y1 : y2 + 1, cl - 1] = GREY
    elif bulge_side == "right":
        y1 = rng.randint(rt + 1, rb - 2)
        y2 = rng.randint(y1 + 1, rb - 1)
        if y2 > y1 + 1:
            patch[y1 + 1 : y2, cr] = BLACK
        patch[y1 : y2 + 1, cr + 1] = GREY

    return patch, (r0, c0, r1, c1)


def _case_tokens(case_name: str, ring_idx: int) -> list[str]:
    if case_name == "none":
        return []
    if case_name == "single_row":
        return [f"row:{ring_idx}"]
    if case_name == "single_col":
        return [f"col:{ring_idx}"]
    if case_name == "double_mixed":
        return [f"row:{ring_idx}", f"col:{ring_idx}"]
    if case_name == "double_row":
        return [f"row:{ring_idx}", f"row:{ring_idx}"]
    if case_name == "double_col":
        return [f"col:{ring_idx}", f"col:{ring_idx}"]
    raise ValueError(case_name)


def _cell_clear(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BLACK:
        return False
    h, w = grid.shape
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if (nr, nc) != (r, c) and grid[nr, nc] != BLACK:
                return False
    return True


def _cell_matches(cavities: list[dict], r: int, c: int) -> list[tuple[int, str, int]]:
    matches: list[tuple[int, str, int]] = []
    for idx, cavity in enumerate(cavities):
        row_dist = _distance_to_cavity(cavity, r, c, "row")
        if row_dist is not None:
            matches.append((idx, "row", row_dist))
        col_dist = _distance_to_cavity(cavity, r, c, "col")
        if col_dist is not None:
            matches.append((idx, "col", col_dist))
    return sorted(matches)


def _signature(matches: list[tuple[int, str, int]]) -> tuple[tuple[int, str], ...]:
    return tuple((idx, direction) for idx, direction, _ in matches)


def _candidate_cells(
    grid: np.ndarray,
    cavities: list[dict],
    target_sig: tuple[tuple[int, str], ...],
) -> list[tuple[int, int, list[tuple[int, str, int]]]]:
    h, w = grid.shape
    out: list[tuple[int, int, list[tuple[int, str, int]]]] = []
    for r in range(h):
        for c in range(w):
            if not _cell_clear(grid, r, c):
                continue
            matches = _cell_matches(cavities, r, c)
            if _signature(matches) == target_sig:
                out.append((r, c, matches))
    return out


def _pick_color(
    ring_colors: dict[int, set[int]],
    ring_ids: list[int],
    rng: random.Random,
) -> int:
    blocked = set()
    for ring_id in ring_ids:
        blocked |= ring_colors[ring_id]
    choices = [color for color in DYE_COLORS if color not in blocked]
    if not choices:
        choices = list(DYE_COLORS)
    return int(rng.choice(choices))


def _place_dye(
    grid: np.ndarray,
    ring_colors: dict[int, set[int]],
    ring_assoc: dict[int, list[tuple[str, int, int]]],
    r: int,
    c: int,
    matches: list[tuple[int, str, int]],
    rng: random.Random,
) -> None:
    ring_ids = sorted({idx for idx, _, _ in matches})
    color = _pick_color(ring_colors, ring_ids, rng)
    grid[r, c] = color
    for ring_id, direction, dist in matches:
        ring_colors[ring_id].add(color)
        ring_assoc[ring_id].append((direction, dist, color))


def _counts_for_case(case_name: str) -> tuple[int, int]:
    if case_name == "none":
        return (0, 0)
    if case_name == "single_row":
        return (1, 0)
    if case_name == "single_col":
        return (0, 1)
    if case_name == "double_mixed":
        return (1, 1)
    if case_name == "double_row":
        return (2, 0)
    if case_name == "double_col":
        return (0, 2)
    raise ValueError(case_name)


def _choose_cases(n_rings: int, rng: random.Random) -> list[str]:
    cases = ["none", rng.choice(["single_row", "single_col"]), rng.choice(["double_mixed", "double_row", "double_col"])]
    if n_rings == 4:
        cases.append(
            rng.choice(
                [
                    "single_row",
                    "single_col",
                    "double_mixed",
                    "double_row",
                    "double_col",
                ]
            )
        )
    rng.shuffle(cases)
    return cases


def _layout_rings(
    ring_shapes: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    n_rings = len(ring_shapes)
    if n_rings != 3:
        return None

    height = rng.randint(28, 30)
    width = rng.randint(28, 30)
    row_shift = rng.randint(0, height - 28)
    col_shift = rng.randint(0, width - 28)
    row_slots = [1 + row_shift, 10 + row_shift, 19 + row_shift]
    col_slots = [1 + col_shift, 10 + col_shift, 19 + col_shift]

    row_order = list(range(n_rings))
    col_order = list(range(n_rings))
    rng.shuffle(row_order)
    rng.shuffle(col_order)

    tops = [0] * n_rings
    lefts = [0] * n_rings
    for slot_idx, ring_idx in enumerate(row_order):
        patch = ring_shapes[ring_idx][0]
        ph = patch.shape[0]
        tops[ring_idx] = row_slots[slot_idx] + rng.randint(0, 8 - ph)
    for slot_idx, ring_idx in enumerate(col_order):
        patch = ring_shapes[ring_idx][0]
        pw = patch.shape[1]
        lefts[ring_idx] = col_slots[slot_idx] + rng.randint(0, 8 - pw)

    grid = np.full((height, width), BLACK, dtype=int)
    cavities: list[dict] = []
    for idx, (patch, cavity_local) in enumerate(ring_shapes):
        top = tops[idx]
        left = lefts[idx]
        ph, pw = patch.shape
        if np.any(grid[top : top + ph, left : left + pw] != BLACK):
            return None
        grid[top : top + ph, left : left + pw] = np.maximum(grid[top : top + ph, left : left + pw], patch)
        lr0, lc0, lr1, lc1 = cavity_local
        cavities.append(
            {
                "bbox": (top + lr0, left + lc0, top + lr1, left + lc1),
                "shape": (lr1 - lr0 + 1, lc1 - lc0 + 1),
            }
        )
    return grid, cavities


def _place_shared_dyes(
    grid: np.ndarray,
    cavities: list[dict],
    cases: list[str],
    ring_colors: dict[int, set[int]],
    ring_assoc: dict[int, list[tuple[str, int, int]]],
    pending_rows: list[int],
    pending_cols: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    shareable_rows = [ring for ring in pending_rows if cases[ring] != "double_row"]
    shareable_cols = [ring for ring in pending_cols if cases[ring] != "double_col"]
    if not shareable_rows or not shareable_cols:
        return pending_rows, pending_cols

    max_shared = min(2, len(shareable_rows), len(shareable_cols))
    shared_goal = rng.randint(0, max_shared)
    for _ in range(shared_goal):
        pairs = [(rr, cc) for rr in shareable_rows for cc in shareable_cols if rr != cc]
        rng.shuffle(pairs)
        placed = False
        for row_ring, col_ring in pairs:
            target = tuple(sorted(((row_ring, "row"), (col_ring, "col"))))
            candidates = _candidate_cells(grid, cavities, target)
            if not candidates:
                continue
            r, c, matches = rng.choice(candidates)
            _place_dye(grid, ring_colors, ring_assoc, r, c, matches, rng)
            pending_rows.remove(row_ring)
            pending_cols.remove(col_ring)
            shareable_rows.remove(row_ring)
            shareable_cols.remove(col_ring)
            placed = True
            break
        if not placed:
            break
    return pending_rows, pending_cols


def _place_remaining_dyes(
    grid: np.ndarray,
    cavities: list[dict],
    ring_colors: dict[int, set[int]],
    ring_assoc: dict[int, list[tuple[str, int, int]]],
    pending_rows: list[int],
    pending_cols: list[int],
    rng: random.Random,
) -> bool:
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for ring in pending_rows:
        row_counts[ring] = row_counts.get(ring, 0) + 1
    for ring in pending_cols:
        col_counts[ring] = col_counts.get(ring, 0) + 1

    for ring, count in row_counts.items():
        target = ((ring, "row"),)
        candidates = _candidate_cells(grid, cavities, target)
        if count == 1:
            if not candidates:
                return False
            r, c, matches = rng.choice(candidates)
            _place_dye(grid, ring_colors, ring_assoc, r, c, matches, rng)
            continue

        dist_to_candidates: dict[int, list[tuple[int, int, list[tuple[int, str, int]]]]] = {}
        for candidate in candidates:
            dist = candidate[2][0][2]
            dist_to_candidates.setdefault(dist, []).append(candidate)
        if len(dist_to_candidates) < 2:
            return False
        distances = sorted(dist_to_candidates)
        near_dist = rng.choice(distances[:-1])
        far_choices = [dist for dist in distances if dist != near_dist]
        far_dist = rng.choice(far_choices)
        first = rng.choice(dist_to_candidates[near_dist])
        second = rng.choice(dist_to_candidates[far_dist])
        _place_dye(grid, ring_colors, ring_assoc, first[0], first[1], first[2], rng)
        _place_dye(grid, ring_colors, ring_assoc, second[0], second[1], second[2], rng)

    for ring, count in col_counts.items():
        target = ((ring, "col"),)
        candidates = _candidate_cells(grid, cavities, target)
        if count == 1:
            if not candidates:
                return False
            r, c, matches = rng.choice(candidates)
            _place_dye(grid, ring_colors, ring_assoc, r, c, matches, rng)
            continue

        dist_to_candidates: dict[int, list[tuple[int, int, list[tuple[int, str, int]]]]] = {}
        for candidate in candidates:
            dist = candidate[2][0][2]
            dist_to_candidates.setdefault(dist, []).append(candidate)
        if len(dist_to_candidates) < 2:
            return False
        distances = sorted(dist_to_candidates)
        near_dist = rng.choice(distances[:-1])
        far_choices = [dist for dist in distances if dist != near_dist]
        far_dist = rng.choice(far_choices)
        first = rng.choice(dist_to_candidates[near_dist])
        second = rng.choice(dist_to_candidates[far_dist])
        _place_dye(grid, ring_colors, ring_assoc, first[0], first[1], first[2], rng)
        _place_dye(grid, ring_colors, ring_assoc, second[0], second[1], second[2], rng)

    return True


def _validate_generated(
    grid: np.ndarray,
    cavities: list[dict],
    cases: list[str],
) -> bool:
    if max(grid.shape) > 30:
        return False
    found_cavities = _find_cavities(grid)
    if [c["bbox"] for c in found_cavities] != [c["bbox"] for c in cavities]:
        return False

    dyes = _dyes(grid)
    if not dyes:
        return False

    for ring_idx, case_name in enumerate(cases):
        want_rows, want_cols = _counts_for_case(case_name)
        row_hits, col_hits = _pairings_for_cavity(cavities[ring_idx], dyes)
        if len(row_hits) != want_rows or len(col_hits) != want_cols:
            return False
        if want_rows == 2 and row_hits[0][0] == row_hits[1][0]:
            return False
        if want_cols == 2 and col_hits[0][0] == col_hits[1][0]:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rings = 3
    ring_shapes: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for _ in range(n_rings):
        cavity_h = rng.randint(2, 5)
        cavity_w = rng.randint(2, 5)
        ring_shapes.append(_build_ring(cavity_h, cavity_w, rng))

    layout = _layout_rings(ring_shapes, rng)
    if layout is None:
        return None
    grid, cavities = layout
    cases = _choose_cases(n_rings, rng)

    ring_colors = {idx: set() for idx in range(n_rings)}
    ring_assoc = {idx: [] for idx in range(n_rings)}
    pending_rows: list[int] = []
    pending_cols: list[int] = []
    for ring_idx, case_name in enumerate(cases):
        for token in _case_tokens(case_name, ring_idx):
            direction, value = token.split(":")
            if direction == "row":
                pending_rows.append(int(value))
            else:
                pending_cols.append(int(value))

    pending_rows, pending_cols = _place_shared_dyes(
        grid,
        cavities,
        cases,
        ring_colors,
        ring_assoc,
        pending_rows,
        pending_cols,
        rng,
    )
    if not _place_remaining_dyes(grid, cavities, ring_colors, ring_assoc, pending_rows, pending_cols, rng):
        return None

    if not _validate_generated(grid, cavities, cases):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-dye input")
