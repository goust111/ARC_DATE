from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
YELLOW = 4
TEAL = 8

QUADRANTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bboxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _square_from_triplet(
    triplet: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
) -> dict | None:
    pts = {tuple(map(int, rc)) for rc in triplet}
    if len(pts) != 3:
        return None

    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return None

    r0, r1 = rows
    c0, c1 = cols
    span_r = r1 - r0
    span_c = c1 - c0
    if span_r != span_c or span_r < 2 or span_r % 2 != 0:
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not pts <= corners:
        return None

    missing = tuple(corners - pts)
    if len(missing) != 1:
        return None
    missing_corner = tuple(map(int, missing[0]))

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    dr = -1 if missing_corner[0] < center[0] else 1
    dc = -1 if missing_corner[1] < center[1] else 1
    return {
        "corners": corners,
        "present": pts,
        "missing_corner": missing_corner,
        "center": center,
        "direction": (dr, dc),
        "radius": span_r // 2,
        "bbox": (r0, c0, r1, c1),
    }


def _infer_deflectors(grid: np.ndarray) -> list[dict]:
    yellow = [tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)]
    assert yellow and len(yellow) % 3 == 0, "yellow markers must come in groups of three"

    indexed = list(enumerate(yellow))
    candidates: list[dict] = []
    for combo in itertools.combinations(indexed, 3):
        indices = tuple(sorted(i for i, _coord in combo))
        triplet = tuple(coord for _i, coord in combo)
        spec = _square_from_triplet(triplet)
        if spec is None:
            continue
        full_corners = spec["corners"]
        if sum(int(corner in yellow) for corner in full_corners) != 3:
            continue
        spec["indices"] = indices
        candidates.append(spec)

    assert candidates, "expected at least one valid three-corner square"

    by_first: dict[int, list[dict]] = {}
    for cand in candidates:
        for idx in cand["indices"]:
            by_first.setdefault(idx, []).append(cand)

    target_groups = len(yellow) // 3
    solutions: list[list[dict]] = []

    def search(remaining: frozenset[int], chosen: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            if len(chosen) == target_groups:
                solutions.append(chosen.copy())
            return

        first = min(remaining)
        for cand in by_first.get(first, []):
            idxs = set(cand["indices"])
            if not idxs <= remaining:
                continue
            chosen.append(cand)
            search(frozenset(remaining - idxs), chosen)
            chosen.pop()

    search(frozenset(range(len(yellow))), [])
    assert len(solutions) == 1, "yellow pixels must partition into a unique set of three-corner squares"

    centers = [cand["center"] for cand in solutions[0]]
    assert len(centers) == len(set(centers)), "deflector centers must be unique"
    return sorted(solutions[0], key=lambda cand: (cand["center"][0], cand["center"][1]))


def _trace_one(
    shape: tuple[int, int],
    start_col: int,
    deflectors: dict[tuple[int, int], tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = map(int, shape)
    r, c = 0, int(start_col)
    path: list[tuple[int, int]] = []
    hits: list[tuple[int, int]] = []
    steps = 0

    while True:
        path.append((r, c))
        if r == H - 1:
            break

        center = (r + 1, c)
        if center in deflectors:
            dr, dc = deflectors[center]
            nr = r + (1 if dr > 0 else 0)
            nc = c + dc
            hits.append(center)
        else:
            nr, nc = r + 1, c

        assert 0 <= nr < H and 0 <= nc < W, "flow must stay within the grid"
        r, c = nr, nc
        steps += 1
        assert steps <= H * W * 4, "unexpected deflection loop"

    return path, hits


def _render_output(
    grid: np.ndarray,
    deflectors: list[dict],
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    sources = [int(c) for c in np.flatnonzero(grid[0] == TEAL)]
    assert sources, "expected teal source pixels on the top edge"

    deflector_map = {tuple(defl["center"]): tuple(defl["direction"]) for defl in deflectors}
    out = np.array(grid, copy=True)
    paths: dict[int, list[tuple[int, int]]] = {}
    hits: dict[int, list[tuple[int, int]]] = {}

    for source_c in sources:
        path, used = _trace_one(grid.shape, source_c, deflector_map)
        paths[source_c] = path
        hits[source_c] = used
        for r, c in path:
            if int(grid[r, c]) != YELLOW:
                out[r, c] = TEAL

    return out, paths, hits


def _vertical_only_output(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    sources = [int(c) for c in np.flatnonzero(grid[0] == TEAL)]
    for c in sources:
        out[:, c] = np.where(out[:, c] == YELLOW, YELLOW, TEAL)
    return out


def _corner_cells(center: tuple[int, int], radius: int, direction: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = map(int, center)
    k = int(radius)
    corners = {
        (cr - k, cc - k),
        (cr - k, cc + k),
        (cr + k, cc - k),
        (cr + k, cc + k),
    }
    missing = (cr + k * direction[0], cc + k * direction[1])
    corners.remove(missing)
    return corners


def _planned_end_state(
    center: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    cr, cc = map(int, center)
    dr, dc = map(int, direction)
    return (cr - 1 if dr < 0 else cr, cc + dc)


def _valid_source_columns(width: int, n_sources: int) -> list[int] | None:
    candidates = list(range(2, width - 2))
    random.shuffle(candidates)
    chosen: list[int] = []
    for c in candidates:
        if all(abs(c - other) >= 3 for other in chosen):
            chosen.append(c)
            if len(chosen) == n_sources:
                return sorted(chosen)
    return None


def _distribute_counts(total_deflectors: int, n_sources: int) -> list[int]:
    counts = [0] * n_sources
    active = list(range(n_sources))
    random.shuffle(active)
    active = active[: random.randint(2, n_sources)]
    counts[active[0]] = 2
    for idx in active[1:]:
        counts[idx] = 1

    remaining = total_deflectors - sum(counts)
    while remaining > 0:
        options = [i for i in range(n_sources) if counts[i] < 3]
        pick = random.choice(options)
        counts[pick] += 1
        remaining -= 1
    return counts


def _place_centers(
    shape: tuple[int, int],
    source_cols: list[int],
    quadrant_seqs: list[list[tuple[int, int]]],
) -> list[dict] | None:
    H, W = map(int, shape)
    placed: list[dict] = []
    existing_centers: list[tuple[int, int]] = []

    source_order = list(range(len(source_cols)))
    random.shuffle(source_order)
    source_order.sort(key=lambda i: len(quadrant_seqs[i]), reverse=True)

    for source_i in source_order:
        c = int(source_cols[source_i])
        current_r = 0
        seq = quadrant_seqs[source_i]
        for j, direction in enumerate(seq):
            remain = len(seq) - j - 1
            row_min = current_r + 2
            row_max = H - 3 - 2 * remain
            if row_min > row_max:
                return None

            candidate_rows = list(range(row_min, row_max + 1))
            random.shuffle(candidate_rows)
            chosen_center = None

            for cr in candidate_rows:
                center = (cr, c)
                if any(max(abs(cr - er), abs(c - ec)) < 4 for er, ec in existing_centers):
                    continue

                end_r, end_c = _planned_end_state(center, direction)
                if remain > 0 and not (2 <= end_c <= W - 3):
                    continue
                if remain == 0 and not (1 <= end_c <= W - 2):
                    continue

                chosen_center = center
                current_r, c = end_r, end_c
                break

            if chosen_center is None:
                return None

            existing_centers.append(chosen_center)
            placed.append(
                {
                    "center": chosen_center,
                    "direction": tuple(direction),
                    "source_index": int(source_i),
                }
            )

    return placed


def _assign_radii(
    shape: tuple[int, int],
    deflectors: list[dict],
    path_cells: set[tuple[int, int]],
) -> list[dict] | None:
    H, W = map(int, shape)
    ordered = list(deflectors)
    random.shuffle(ordered)
    ordered.sort(
        key=lambda d: min(
            3,
            d["center"][0] - 2,
            H - 2 - d["center"][0],
            d["center"][1] - 1,
            W - 2 - d["center"][1],
        )
    )

    chosen: list[dict] = []
    occupied_corners: set[tuple[int, int]] = set()

    def search(i: int) -> bool:
        if i == len(ordered):
            return True

        base = ordered[i]
        cr, cc = base["center"]
        max_k = min(3, cr - 2, H - 2 - cr, cc - 1, W - 2 - cc)
        if max_k < 1:
            return False

        radii = list(range(1, max_k + 1))
        random.shuffle(radii)
        radii.sort(reverse=True)

        for radius in radii:
            corners = _corner_cells(base["center"], radius, base["direction"])
            if corners & path_cells:
                continue
            if corners & occupied_corners:
                continue

            bbox = (cr - radius, cc - radius, cr + radius, cc + radius)
            if any(not _bboxes_separated(bbox, prev["bbox"], gap=1) for prev in chosen):
                continue

            record = dict(base)
            record["radius"] = int(radius)
            record["corners"] = corners
            record["bbox"] = bbox
            chosen.append(record)
            occupied_corners.update(corners)
            if search(i + 1):
                return True
            chosen.pop()
            occupied_corners.difference_update(corners)

        return False

    if not search(0):
        return None
    return sorted(chosen, key=lambda d: (d["center"][0], d["center"][1]))


def _pairwise_disjoint(paths: dict[int, list[tuple[int, int]]]) -> bool:
    cols = list(paths)
    for i, a in enumerate(cols):
        sa = set(paths[a])
        for b in cols[i + 1 :]:
            if sa & set(paths[b]):
                return False
    return True


def _quadrant_cover(deflectors: list[dict]) -> bool:
    present = {tuple(defl["direction"]) for defl in deflectors}
    return all(quadrant in present for quadrant in QUADRANTS)


def _planned_hits_match(
    source_cols: list[int],
    quadrant_seqs: list[list[tuple[int, int]]],
    deflectors: list[dict],
    actual_hits: dict[int, list[tuple[int, int]]],
) -> bool:
    by_source: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(source_cols))}
    for defl in sorted(deflectors, key=lambda d: (d["center"][0], d["center"][1])):
        by_source[int(defl["source_index"])].append(tuple(defl["center"]))

    for source_i, source_c in enumerate(source_cols):
        if len(quadrant_seqs[source_i]) != len(by_source[source_i]):
            return False
        if actual_hits[int(source_c)] != by_source[source_i]:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw teal flows downward, sidestepping around invisible centers inferred from three-corner yellow squares.

    Input structure:
    - The background is black.
    - Several teal single-pixel sources sit on the top edge.
    - Yellow single pixels form several disjoint groups of three corners of axis-aligned even-sided squares.

    Rule:
    - Recover each unique square from its three visible yellow corners, then infer the square's missing corner and hidden center cell.
    - From every top-edge teal source, trace a teal path downward toward the bottom edge.
    - Whenever the next straight-down step would enter an inferred center cell, replace that step with the center's diagonal neighbor in the missing-corner direction, then continue downward from the new column.
    - Leave the yellow corner markers unchanged.

    Key insight:
    - The working deflectors are never drawn directly: the three visible corners reveal both the hidden center to avoid and the quadrant that tells the flow which way to slip past it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, YELLOW, TEAL}, "expected only black, yellow, and teal"
    assert np.any(g[0] == TEAL), "expected top-edge teal sources"
    assert not np.any(g[1:, :] == TEAL), "input teal pixels must all be top-edge sources"

    deflectors = _infer_deflectors(g)
    out, _paths, _hits = _render_output(g, deflectors)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(16, 24)
    W = random.randint(16, 24)
    n_sources = random.randint(2, 4)
    source_cols = _valid_source_columns(W, n_sources)
    if source_cols is None:
        return None

    total_deflectors = random.choice((4, 4, 5))
    counts = _distribute_counts(total_deflectors, n_sources)

    quadrants = list(QUADRANTS) + [random.choice(QUADRANTS) for _ in range(total_deflectors - 4)]
    random.shuffle(quadrants)
    quadrant_seqs: list[list[tuple[int, int]]] = [[] for _ in range(n_sources)]
    pos = 0
    for source_i, count in enumerate(counts):
        quadrant_seqs[source_i] = quadrants[pos : pos + count]
        pos += count
        random.shuffle(quadrant_seqs[source_i])

    placed = _place_centers((H, W), source_cols, quadrant_seqs)
    if placed is None:
        return None
    if not _quadrant_cover(placed):
        return None

    temp_grid = np.zeros((H, W), dtype=int)
    temp_grid[0, source_cols] = TEAL
    temp_out, paths, hits = _render_output(
        temp_grid,
        [{"center": defl["center"], "direction": defl["direction"]} for defl in placed],
    )
    path_cells = {cell for path in paths.values() for cell in path}

    if not _pairwise_disjoint(paths):
        return None
    if not _planned_hits_match(source_cols, quadrant_seqs, placed, hits):
        return None

    deflectors = _assign_radii((H, W), placed, path_cells)
    if deflectors is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[0, source_cols] = TEAL
    for defl in deflectors:
        for r, c in defl["corners"]:
            grid[r, c] = YELLOW

    inferred = _infer_deflectors(grid)
    if len(inferred) != len(deflectors):
        return None
    if {
        (tuple(d["center"]), tuple(d["direction"]), int(d["radius"]))
        for d in inferred
    } != {
        (tuple(d["center"]), tuple(d["direction"]), int(d["radius"]))
        for d in deflectors
    }:
        return None

    out = main(grid)
    vertical = _vertical_only_output(grid)
    if np.array_equal(out, grid) or np.array_equal(out, vertical):
        return None

    out2, paths2, hits2 = _render_output(grid, inferred)
    if not np.array_equal(out, out2):
        return None
    if not _pairwise_disjoint(paths2):
        return None
    if not _planned_hits_match(source_cols, quadrant_seqs, deflectors, hits2):
        return None

    used_centers = {center for hit_list in hits2.values() for center in hit_list}
    if used_centers != {tuple(defl["center"]) for defl in deflectors}:
        return None
    if any(len(hit_list) == 0 for hit_list in hits2.values()):
        return None
    if not any(len(hit_list) >= 2 for hit_list in hits2.values()):
        return None

    path_cells2 = {cell for path in paths2.values() for cell in path}
    yellow_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)}
    if path_cells2 & yellow_cells:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-deflector ARC input")
