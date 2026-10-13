import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

OUTLINE_COLOR = GREEN
MARKER_COLORS = (BLUE, RED, YELLOW, ORANGE)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 500
SELECTION_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 150


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    assert connectivity in (4, 8)
    deltas = DIRS4 if connectivity == 4 else DIRS8
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": {tuple(map(int, cell)) for cell in coords},
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _boundary_mask(fill_mask: np.ndarray) -> np.ndarray:
    h, w = fill_mask.shape
    out = np.zeros_like(fill_mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not fill_mask[r, c]:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not in_bounds(fill_mask.shape, nr, nc) or not fill_mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _interior_black_cells(grid: np.ndarray, contour_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = np.array([r for r, _ in contour_set], dtype=int)
    cols = np.array([c for _, c in contour_set], dtype=int)
    r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    seen: set[tuple[int, int]] = set()
    q = deque()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if grid[r, c] == BLACK and (r, c) not in seen:
                seen.add((r, c))
                q.append((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if grid[r, c] == BLACK and (r, c) not in seen:
                seen.add((r, c))
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if (
                r0 <= nr <= r1
                and c0 <= nc <= c1
                and grid[nr, nc] == BLACK
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    interior = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r, c] == BLACK and (r, c) not in seen:
                interior.add((r, c))
    return interior


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros_like(grid, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _ray_hits(
    grid: np.ndarray, marker: tuple[int, int], contour_lookup: dict[tuple[int, int], int]
) -> dict[int, int]:
    hits: dict[int, int] = {}
    mr, mc = marker
    for dr, dc in DIRS4:
        r, c = mr, mc
        distance = 0
        while True:
            r += dr
            c += dc
            if not in_bounds(grid.shape, r, c):
                break
            if grid[r, c] == BLACK:
                distance += 1
                continue
            contour_idx = contour_lookup.get((r, c))
            if contour_idx is not None:
                prev = hits.get(contour_idx)
                if prev is None or distance < prev:
                    hits[contour_idx] = distance
            break
    return hits


def _analyze(grid: np.ndarray, *, require_markers: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    comps = _monochrome_components(g)
    contours_raw = [comp for comp in comps if comp["coords"].shape[0] > 1]
    markers_raw = [comp for comp in comps if comp["coords"].shape[0] == 1]
    assert contours_raw, "expected at least one contour"
    if require_markers:
        assert markers_raw, "expected at least one marker"

    contour_lookup: dict[tuple[int, int], int] = {}
    contours: list[dict] = []
    exterior = _exterior_black_mask(g)
    for idx, comp in enumerate(sorted(contours_raw, key=lambda item: item["bbox"])):
        contour_set = comp["coord_set"]
        interior = _interior_black_cells(g, contour_set)
        outside_ring = set()
        for r, c in contour_set:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if in_bounds(g.shape, nr, nc) and g[nr, nc] == BLACK and exterior[nr, nc]:
                    outside_ring.add((nr, nc))
            contour_lookup[(r, c)] = idx
        contours.append(
            {
                "color": comp["color"],
                "coords": comp["coords"],
                "coord_set": contour_set,
                "bbox": comp["bbox"],
                "interior": interior,
                "outside_ring": outside_ring,
            }
        )

    markers = []
    for comp in sorted(markers_raw, key=lambda item: tuple(map(int, item["coords"][0]))):
        pos = tuple(map(int, comp["coords"][0]))
        hits = _ray_hits(g, pos, contour_lookup)
        best_hit_contour = None
        if hits:
            best_hit_contour = min(hits.items(), key=lambda item: (item[1], item[0]))[0]
        euclid_nearest = min(
            range(len(contours)),
            key=lambda idx: min(
                (int(pos[0] - r) ** 2 + int(pos[1] - c) ** 2)
                for r, c in contours[idx]["coord_set"]
            ),
        )
        markers.append(
            {
                "pos": pos,
                "color": int(comp["color"]),
                "hits": hits,
                "best_hit_contour": best_hit_contour,
                "euclid_nearest_contour": euclid_nearest,
            }
        )

    candidates_by_contour: dict[int, list[dict]] = {idx: [] for idx in range(len(contours))}
    for marker_index, marker in enumerate(markers):
        for contour_index, distance in marker["hits"].items():
            candidates_by_contour[contour_index].append(
                {
                    "distance": int(distance),
                    "marker_index": marker_index,
                    "pos": marker["pos"],
                    "color": marker["color"],
                }
            )

    winners: dict[int, dict | None] = {}
    for contour_index, candidates in candidates_by_contour.items():
        ordered = sorted(candidates, key=lambda item: (item["distance"], item["pos"]))
        winners[contour_index] = ordered[0] if ordered else None

    output = np.zeros_like(g)
    for contour_index, contour in enumerate(contours):
        winner = winners[contour_index]
        if winner is None:
            continue
        for r, c in contour["outside_ring"]:
            output[r, c] = winner["color"]

    return {
        "grid": g,
        "contours": contours,
        "markers": markers,
        "candidates_by_contour": candidates_by_contour,
        "winners": winners,
        "output": output,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project border markers inward to claim closed contours, then erase the originals and repaint each contour's outside trace in its winning marker color.

    Input structure:
    - A black background contains 2-3 disjoint closed outline components and 3-4 single-pixel colored markers.
    - Each outline is an 8-connected monochrome loop enclosing black interior cells; markers are outside every loop and not on the outlines.
    - Markers use distinct colors and may align with contours only along the four cardinal directions.

    Rule:
    - Treat every non-singleton foreground component as a contour and every singleton foreground component as a marker.
    - From each marker, cast rays up, down, left, and right; each ray travels through black cells until it either leaves the grid or hits the first non-black cell.
    - If that first hit belongs to a contour, record a claim on that contour with distance equal to the number of black cells crossed; each contour is assigned to the marker with the smallest claim distance, breaking exact ties by marker scan order.
    - Erase all original contours and markers to black, then repaint only the black cells that are 4-neighbors of the original contour and reachable from the grid edge through the original black background; use the winning marker's color for that contour, and leave unclaimed contours erased.

    Key insight:
    - Assignment depends on collision distance along cardinal projections, not on ordinary spatial closeness: a diagonally nearer contour can lose to a farther contour that is actually hittable in-row or in-column.
    """
    details = _analyze(np.asarray(grid, dtype=int))
    return details["output"]


def _make_fill_shape(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(120):
        base_h = int(rng.integers(4, 6))
        base_w = int(rng.integers(4, 7))
        shape = np.zeros((base_h + 2, base_w + 2), dtype=bool)
        shape[1 : 1 + base_h, 1 : 1 + base_w] = True

        bump_count = int(rng.choice([0, 1, 1, 2]))
        for side in rng.permutation(4)[:bump_count]:
            if side in (0, 1):
                max_len = min(3, base_w - 1)
                if max_len < 2:
                    continue
                length = int(rng.integers(2, max_len + 1))
                start = int(rng.integers(1, base_w - length + 2))
                row = 0 if side == 0 else base_h + 1
                shape[row, start : start + length] = True
            else:
                max_len = min(3, base_h - 1)
                if max_len < 2:
                    continue
                length = int(rng.integers(2, max_len + 1))
                start = int(rng.integers(1, base_h - length + 2))
                col = 0 if side == 2 else base_w + 1
                shape[start : start + length, col] = True

        coords = np.argwhere(shape)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        shape = shape[r0 : r1 + 1, c0 : c1 + 1]
        boundary = _boundary_mask(shape)
        if not (10 <= int(boundary.sum()) <= 18):
            continue
        if int((shape & ~boundary).sum()) == 0:
            continue
        if len(_mask_components(boundary, 8)) != 1:
            continue
        return shape
    return None


def _bboxes_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1]


def _place_contours(rng: np.random.Generator, h: int, w: int, count: int) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.zeros((h, w), dtype=int)
    filled = np.zeros((h, w), dtype=bool)
    placed_boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _make_fill_shape(rng)
            if shape is None:
                continue
            sh, sw = shape.shape
            if sh + 2 >= h or sw + 2 >= w:
                continue
            top = int(rng.integers(1, h - sh))
            left = int(rng.integers(1, w - sw))
            box = (top, left, top + sh - 1, left + sw - 1)
            if any(not _bboxes_far_apart(box, other, margin=3) for other in placed_boxes):
                continue
            local_boundary = _boundary_mask(shape)
            fill_coords = np.argwhere(shape) + np.array([top, left], dtype=int)
            boundary_coords = np.argwhere(local_boundary) + np.array([top, left], dtype=int)
            if np.any(grid[boundary_coords[:, 0], boundary_coords[:, 1]] != BLACK):
                continue
            filled[fill_coords[:, 0], fill_coords[:, 1]] = True
            grid[boundary_coords[:, 0], boundary_coords[:, 1]] = OUTLINE_COLOR
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return grid, filled


def _build_candidate_pool(
    base_grid: np.ndarray, filled_mask: np.ndarray
) -> list[dict]:
    details = _analyze(base_grid, require_markers=False)
    contours = details["contours"]
    contour_lookup = {
        cell: contour_index
        for contour_index, contour in enumerate(contours)
        for cell in contour["coord_set"]
    }
    pool = []
    for r in range(base_grid.shape[0]):
        for c in range(base_grid.shape[1]):
            if base_grid[r, c] != BLACK or filled_mask[r, c]:
                continue
            hits = _ray_hits(base_grid, (r, c), contour_lookup)
            if not hits:
                continue
            best_hit = min(hits.items(), key=lambda item: (item[1], item[0]))[0]
            euclid_nearest = min(
                range(len(contours)),
                key=lambda idx: min(
                    (r - rr) ** 2 + (c - cc) ** 2 for rr, cc in contours[idx]["coord_set"]
                ),
            )
            pool.append(
                {
                    "pos": (r, c),
                    "hits": hits,
                    "best_hit_contour": best_hit,
                    "euclid_nearest_contour": euclid_nearest,
                    "mismatch": euclid_nearest != best_hit,
                }
            )
    return pool


def _compatible(candidate: dict, chosen: list[dict]) -> bool:
    r, c = candidate["pos"]
    for other in chosen:
        orow, ocol = other["pos"]
        if r == orow or c == ocol:
            return False
        if max(abs(r - orow), abs(c - ocol)) <= 1:
            return False
    return True


def _choose_markers(
    rng: np.random.Generator, base_grid: np.ndarray, filled_mask: np.ndarray
) -> np.ndarray | None:
    pool = _build_candidate_pool(base_grid, filled_mask)
    contour_count = len(_analyze(base_grid, require_markers=False)["contours"])
    if contour_count not in (2, 3):
        return None

    marker_count = int(rng.choice([3, 4]))
    require_unassigned = contour_count == 3 and rng.random() < 0.55
    active_contours = list(range(contour_count))
    unassigned_contour = None
    if require_unassigned:
        unassigned_contour = int(rng.integers(0, contour_count))
        active_contours = [idx for idx in active_contours if idx != unassigned_contour]

    usable = [
        cand
        for cand in pool
        if unassigned_contour is None or unassigned_contour not in cand["hits"]
    ]
    if len(usable) < marker_count:
        return None
    if not any(cand["mismatch"] for cand in usable):
        return None

    comp_options = [
        idx
        for idx in active_contours
        if sum(1 for cand in usable if idx in cand["hits"]) >= 2
    ]
    if not comp_options:
        return None

    for _ in range(SELECTION_ATTEMPTS):
        chosen: list[dict] = []
        competition_contour = int(rng.choice(comp_options))
        first_pool = [
            cand
            for cand in usable
            if cand["mismatch"] and competition_contour in cand["hits"]
        ]
        if not first_pool:
            first_pool = [cand for cand in usable if cand["mismatch"]]
        if not first_pool:
            continue
        first = random.choice(first_pool)
        chosen.append(first)

        second_pool = [
            cand
            for cand in usable
            if cand is not first and competition_contour in cand["hits"] and _compatible(cand, chosen)
        ]
        if not second_pool:
            continue
        chosen.append(random.choice(second_pool))

        ok = True
        for contour_idx in active_contours:
            if contour_idx == competition_contour:
                continue
            contour_pool = [
                cand
                for cand in usable
                if contour_idx in cand["hits"] and _compatible(cand, chosen)
            ]
            if not contour_pool:
                ok = False
                break
            chosen.append(random.choice(contour_pool))
        if not ok:
            continue

        while len(chosen) < marker_count:
            extra_pool = [cand for cand in usable if cand not in chosen and _compatible(cand, chosen)]
            if not extra_pool:
                ok = False
                break
            chosen.append(random.choice(extra_pool))
        if not ok:
            continue

        grid = base_grid.copy()
        colors = list(rng.choice(np.array(MARKER_COLORS, dtype=int), size=marker_count, replace=False))
        rng.shuffle(colors)
        for cand, color in zip(chosen, colors):
            r, c = cand["pos"]
            grid[r, c] = int(color)

        if _validate_generated_grid(grid, require_unassigned=require_unassigned):
            return grid
    return None


def _validate_generated_grid(grid: np.ndarray, require_unassigned: bool) -> bool:
    try:
        details = _analyze(grid)
    except AssertionError:
        return False

    contours = details["contours"]
    markers = details["markers"]
    winners = details["winners"]
    candidates_by_contour = details["candidates_by_contour"]

    if not (2 <= len(contours) <= 3 and 3 <= len(markers) <= 4):
        return False
    if any(marker["color"] not in MARKER_COLORS for marker in markers):
        return False
    if len({marker["color"] for marker in markers}) != len(markers):
        return False
    if any(contour["color"] != OUTLINE_COLOR for contour in contours):
        return False
    if any(not contour["interior"] for contour in contours):
        return False
    if any(not contour["outside_ring"] for contour in contours):
        return False
    if any(marker["hits"] == {} for marker in markers):
        return False

    for candidates in candidates_by_contour.values():
        distances = [cand["distance"] for cand in candidates]
        if len(distances) != len(set(distances)):
            return False

    if not any(len(candidates) >= 2 for candidates in candidates_by_contour.values()):
        return False

    painted_contours = [idx for idx, winner in winners.items() if winner is not None]
    if require_unassigned:
        if len(contours) != 3 or len(painted_contours) != 2:
            return False
    else:
        if len(painted_contours) != len(contours):
            return False

    winner_markers = {winner["marker_index"] for winner in winners.values() if winner is not None}
    if len(winner_markers) < 2:
        return False

    if not any(
        marker["best_hit_contour"] is not None
        and marker["best_hit_contour"] != marker["euclid_nearest_contour"]
        for marker in markers
    ):
        return False

    claimed_cells: dict[tuple[int, int], int] = {}
    for contour_index, contour in enumerate(contours):
        for cell in contour["outside_ring"]:
            if cell in claimed_cells and claimed_cells[cell] != contour_index:
                return False
            claimed_cells[cell] = contour_index

    output = details["output"]
    if np.count_nonzero(output) < 8:
        return False
    if np.array_equal(output, np.zeros_like(output)):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        h = int(rng.integers(14, 19))
        w = int(rng.integers(14, 19))
        contour_count = int(rng.choice([2, 3]))
        placed = _place_contours(rng, h, w, contour_count)
        if placed is None:
            continue
        base_grid, filled_mask = placed
        grid = _choose_markers(rng, base_grid, filled_mask)
        if grid is None:
            continue
        if _validate_generated_grid(grid, require_unassigned=False) or _validate_generated_grid(
            grid, require_unassigned=True
        ):
            return grid
    raise RuntimeError("failed to generate a valid projected-contour ARC input")
