from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

OUTPUT_SHAPES = ((3, 4), (3, 5), (4, 4), (3, 6), (4, 5), (4, 6), (5, 5))
INPUT_SIDE_RANGE = (14, 18)
PIECE_COUNT_RANGE = (4, 6)
MARKER_COUNT_RANGE = (2, 4)
PIECE_AREA_RANGE = (3, 8)
MAX_GENERATION_ATTEMPTS = 500
SUBSET_ATTEMPTS = 70
SCATTER_ATTEMPTS = 120
MARKER_ATTEMPTS = 220


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    steps = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    return pts - pts.min(axis=0)


def _coords_to_patch(coords: np.ndarray, color: int = 1, bg: int = 0) -> np.ndarray:
    pts = _normalize(coords)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    patch = np.full((h, w), int(bg), dtype=int)
    patch[pts[:, 0], pts[:, 1]] = int(color)
    return patch


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], bytes]:
    patch = _coords_to_patch(coords, color=1, bg=0)
    return tuple(map(int, patch.shape)), patch.astype(np.uint8).tobytes()


def _orientation_variants(coords: np.ndarray) -> list[np.ndarray]:
    patch = _coords_to_patch(coords, color=1, bg=0).astype(bool)
    rots = [
        patch,
        np.fliplr(patch),
        np.rot90(patch, -1),
        np.fliplr(np.rot90(patch, -1)),
        np.rot90(patch, 2),
        np.fliplr(np.rot90(patch, 2)),
        np.rot90(patch, 1),
        np.fliplr(np.rot90(patch, 1)),
    ]
    return [np.argwhere(variant) for variant in rots]


def _compact_shape(area: int) -> tuple[int, int]:
    factors = [(h, area // h) for h in range(1, int(math.isqrt(area)) + 1) if area % h == 0]
    assert factors, "foreground area must factor into a rectangle"
    return max(factors, key=lambda hw: hw[0])


def _piece_order(pieces: list[dict], mode: str = "correct") -> list[int]:
    if mode == "correct":
        return sorted(range(len(pieces)), key=lambda i: (-pieces[i]["area"], pieces[i]["bbox"][0], pieces[i]["bbox"][1]))
    if mode == "colrow":
        return sorted(range(len(pieces)), key=lambda i: (-pieces[i]["area"], pieces[i]["bbox"][1], pieces[i]["bbox"][0]))
    raise ValueError(mode)


def _pack_pieces(
    pieces: list[dict],
    *,
    order_mode: str = "correct",
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[dict], list[int]]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    shape = _compact_shape(total_area) if out_shape is None else tuple(map(int, out_shape))
    assert shape[0] * shape[1] == total_area

    order = _piece_order(pieces, mode=order_mode)
    out = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    placements: list[dict] = []

    for piece_idx in order:
        piece = pieces[piece_idx]
        placed = False
        for orient_idx, orient in enumerate(_orientation_variants(piece["coords"])):
            oh = int(orient[:, 0].max()) + 1
            ow = int(orient[:, 1].max()) + 1
            if oh > shape[0] or ow > shape[1]:
                continue
            for r0 in range(shape[0] - oh + 1):
                for c0 in range(shape[1] - ow + 1):
                    cells = orient + np.array([r0, c0], dtype=int)
                    if np.any(occupied[cells[:, 0], cells[:, 1]]):
                        continue
                    occupied[cells[:, 0], cells[:, 1]] = True
                    out[cells[:, 0], cells[:, 1]] = int(piece["color"])
                    placements.append(
                        {
                            "piece_idx": int(piece_idx),
                            "orient_idx": int(orient_idx),
                            "origin": (int(r0), int(c0)),
                            "coords": cells.copy(),
                        }
                    )
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            raise AssertionError("pieces do not greedily pack into the compact rectangle")

    assert int(occupied.sum()) == total_area == shape[0] * shape[1]
    return out, placements, order


def _min_sq_distance(piece_coords: np.ndarray, marker_rc: tuple[int, int]) -> int:
    diffs = piece_coords - np.array(marker_rc, dtype=int)
    return int(np.min(np.sum(diffs * diffs, axis=1)))


def _piece_owner(piece_coords: np.ndarray, markers: list[dict]) -> tuple[int, dict]:
    scored = []
    for marker in markers:
        scored.append(
            (
                _min_sq_distance(piece_coords, marker["pos"]),
                int(marker["pos"][0]),
                int(marker["pos"][1]),
                int(marker["color"]),
                marker,
            )
        )
    best = min(scored)
    return int(best[3]), best[4]


def _extract_instance(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    markers: list[dict] = []
    seen_colors = set()
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        color = int(g[r, c])
        assert color not in seen_colors, "marker colors must be unique"
        seen_colors.add(color)
        markers.append({"pos": (int(r), int(c)), "color": color})
    assert 2 <= len(markers) <= 4, "expected 2-4 marker pixels"
    markers.sort(key=lambda m: (m["pos"][0], m["pos"][1]))
    for marker in markers:
        r, c = marker["pos"]
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(g.shape, nr, nc):
                assert g[nr, nc] == BLACK, "marker pixels must be isolated and away from fragments"

    pieces: list[dict] = []
    comps = _component_coords(g == GREY, connectivity=4)
    assert 4 <= len(comps) <= 6, "expected 4-6 grey fragments"
    assert len(comps) >= len(markers)
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        color, owner = _piece_owner(coords, markers)
        pieces.append(
            {
                "coords": rel,
                "abs_coords": coords.copy(),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(coords.shape[0]),
                "color": int(color),
                "owner_pos": tuple(owner["pos"]),
            }
        )

    total_area = sum(piece["area"] for piece in pieces)
    shape = _compact_shape(total_area)
    assert shape[0] * shape[1] == total_area
    return pieces, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the grey fragments into a compact rectangle and recolor each fragment by the nearest original marker.

    Input structure:
    - The background is black.
    - There are 4-6 grey 4-connected fragments, each separated from the others.
    - There are 2-4 isolated colored marker pixels with distinct colors, not touching any fragment in the 4-neighborhood.

    Rule:
    - For each grey fragment, find the marker whose pixel has the smallest squared Euclidean distance to any pixel of that fragment; break ties by smaller marker row, then smaller marker column.
    - Sort fragments by decreasing area, breaking equal-area ties by the row then column of the fragment bbox's top-left corner in the input.
    - Greedily pack the fragments into the most compact exact-area rectangle by trying orientations in this order: original, horizontal flip, 90° clockwise, 90° clockwise then flip, 180°, 180° then flip, 270°, 270° then flip; within each orientation, scan candidate placements from top-left to bottom-right and take the first non-overlapping fit.
    - Recolor each packed fragment with its recorded marker color and output only the packed rectangle.

    Key insight:
    Fragment colors come from distances measured in the scattered input layout, but those colors are displayed only after the fragments have been repacked into a new rectangle.
    """
    pieces, _markers = _extract_instance(grid)
    out, _placements, _order = _pack_pieces(pieces, order_mode="correct")
    return out


def _neighbors_in_cells(cell: tuple[int, int], cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1] and (nr, nc) in cells:
            out.append((nr, nc))
    return out


def _is_connected_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nb in _neighbors_in_cells(cell, cells, shape):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _components_from_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            for nb in _neighbors_in_cells(cell, remaining | {cell}, shape):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _capacity_fits(cells: set[tuple[int, int]], sizes: tuple[int, ...], shape: tuple[int, int]) -> bool:
    capacities = tuple(sorted((len(comp) for comp in _components_from_cells(cells, shape)), reverse=True))
    want = tuple(sorted(sizes, reverse=True))

    def dfs(i: int, caps: tuple[int, ...]) -> bool:
        if i == len(want):
            return all(cap == 0 for cap in caps)
        size = want[i]
        tried = set()
        for j, cap in enumerate(caps):
            if cap < size or cap in tried:
                continue
            tried.add(cap)
            nxt = list(caps)
            nxt[j] -= size
            if dfs(i + 1, tuple(sorted(nxt, reverse=True))):
                return True
        return False

    return dfs(0, capacities)


def _sample_connected_subset(cells: set[tuple[int, int]], size: int, shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    comps = [comp for comp in _components_from_cells(cells, shape) if len(comp) >= size]
    random.shuffle(comps)
    for comp in comps:
        for _ in range(18):
            start = random.choice(tuple(comp))
            subset = {start}
            frontier = set(_neighbors_in_cells(start, comp, shape))
            while len(subset) < size and frontier:
                choice = random.choice(tuple(frontier))
                frontier.remove(choice)
                if choice in subset:
                    continue
                subset.add(choice)
                for nb in _neighbors_in_cells(choice, comp, shape):
                    if nb not in subset:
                        frontier.add(nb)
            if len(subset) == size:
                return subset
    return None


def _partition_rectangle(shape: tuple[int, int], sizes: tuple[int, ...]) -> list[np.ndarray] | None:
    all_cells = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    sizes = tuple(sorted(map(int, sizes), reverse=True))

    def rec(cells: set[tuple[int, int]], rest: tuple[int, ...]) -> list[set[tuple[int, int]]] | None:
        if not rest:
            return [] if not cells else None
        if len(rest) == 1:
            if len(cells) == rest[0] and _is_connected_cells(cells, shape):
                return [set(cells)]
            return None

        size = rest[0]
        for _ in range(SUBSET_ATTEMPTS):
            subset = _sample_connected_subset(cells, size, shape)
            if subset is None:
                continue
            remaining = cells - subset
            if not _capacity_fits(remaining, rest[1:], shape):
                continue
            tail = rec(remaining, rest[1:])
            if tail is not None:
                return [subset] + tail
        return None

    parts = rec(all_cells, sizes)
    if parts is None:
        return None
    return [_normalize(np.array(sorted(part), dtype=int)) for part in parts]


def _sample_sizes(area: int, count: int, *, require_repeat: bool, forbid_repeat: bool) -> tuple[int, ...] | None:
    low, high = PIECE_AREA_RANGE
    for _ in range(120):
        sizes = [low] * count
        rem = area - low * count
        while rem > 0:
            choices = [i for i, v in enumerate(sizes) if v < high]
            if not choices:
                break
            idx = random.choice(choices)
            sizes[idx] += 1
            rem -= 1
        if rem != 0:
            continue
        n_unique = len(set(sizes))
        if require_repeat and n_unique == len(sizes):
            continue
        if forbid_repeat and n_unique != len(sizes):
            continue
        if max(sizes) < 4:
            continue
        return tuple(sorted(sizes, reverse=True))
    return None


def _has_distinct_shapes(coords_list: list[np.ndarray]) -> bool:
    keys = [_shape_key(coords) for coords in coords_list]
    return len(keys) == len(set(keys))


def _design_shapes(require_repeat: bool) -> tuple[list[np.ndarray], tuple[int, int]] | None:
    shape = random.choice(OUTPUT_SHAPES)
    area = int(shape[0] * shape[1])
    counts = [n for n in range(PIECE_COUNT_RANGE[0], PIECE_COUNT_RANGE[1] + 1) if PIECE_AREA_RANGE[0] * n <= area <= PIECE_AREA_RANGE[1] * n]
    random.shuffle(counts)

    for count in counts:
        sizes = _sample_sizes(area, count, require_repeat=require_repeat, forbid_repeat=not require_repeat)
        if sizes is None:
            continue
        for _ in range(80):
            parts = _partition_rectangle(shape, sizes)
            if parts is None:
                continue
            if not _has_distinct_shapes(parts):
                continue
            if all(np.array_equal(_coords_to_patch(coords), _coords_to_patch(coords).T) for coords in parts):
                continue
            dummy_pieces = []
            for idx, coords in enumerate(parts):
                r0 = idx * 2
                dummy_pieces.append(
                    {
                        "coords": coords,
                        "area": int(coords.shape[0]),
                        "bbox": (r0, 0, r0, 0),
                        "color": 1,
                    }
                )
            try:
                _pack_pieces(dummy_pieces, order_mode="correct", out_shape=shape)
            except AssertionError:
                continue
            return parts, shape
    return None


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    H, W = occupied.shape
    h, w = patch.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    for dr, dc in np.argwhere(patch == GREY):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if occupied[rr, cc]:
            return False
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def _scatter_pieces(coords_list: list[np.ndarray], *, randomize_orientations: bool) -> tuple[np.ndarray, list[dict]] | None:
    H = random.randint(*INPUT_SIDE_RANGE)
    W = random.randint(*INPUT_SIDE_RANGE)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = sorted(range(len(coords_list)), key=lambda i: (_coords_to_patch(coords_list[i]).size, coords_list[i].shape[0]), reverse=True)
    pieces: list[dict | None] = [None] * len(coords_list)

    for idx in order:
        variants = _orientation_variants(coords_list[idx])
        oriented = random.choice(variants) if randomize_orientations else variants[0]
        patch = _coords_to_patch(oriented, color=GREY, bg=BLACK)
        placed = False
        for _ in range(SCATTER_ATTEMPTS):
            r0 = random.randint(1, H - patch.shape[0] - 1)
            c0 = random.randint(1, W - patch.shape[1] - 1)
            if not _can_place_patch(occupied, patch, r0, c0):
                continue
            grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch == GREY] = GREY
            coords_abs = np.argwhere(patch == GREY) + np.array([r0, c0], dtype=int)
            occupied[coords_abs[:, 0], coords_abs[:, 1]] = True
            bbox = (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1))
            pieces[idx] = {
                "coords": np.argwhere(patch == GREY),
                "abs_coords": coords_abs.copy(),
                "bbox": bbox,
                "area": int(coords_abs.shape[0]),
            }
            placed = True
            break
        if not placed:
            return None

    return grid, [piece for piece in pieces if piece is not None]


def _fragment_forbidden_mask(pieces: list[dict], shape: tuple[int, int]) -> np.ndarray:
    forbidden = np.zeros(shape, dtype=bool)
    for piece in pieces:
        for r, c in piece["abs_coords"]:
            forbidden[int(r), int(c)] = True
            for dr, dc in NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if _in_bounds(shape, nr, nc):
                    forbidden[nr, nc] = True
    return forbidden


def _non_adjacent_markers(positions: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(positions):
        for r2, c2 in positions[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) <= 1:
                return False
    return True


def _choose_random_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _compute_piece_scores(pieces: list[dict], markers: list[dict]) -> list[dict]:
    scored: list[dict] = []
    for piece in pieces:
        distances = []
        for marker in markers:
            d = _min_sq_distance(piece["abs_coords"], marker["pos"])
            distances.append((int(d), int(marker["pos"][0]), int(marker["pos"][1]), int(marker["color"])))
        distances.sort()
        best_d = distances[0][0]
        tied = [entry for entry in distances if entry[0] == best_d]
        winner = tied[0]
        scored.append(
            {
                **piece,
                "color": int(winner[3]),
                "owner_pos": (int(winner[1]), int(winner[2])),
                "best_distance": int(best_d),
                "tie_count": int(len(tied)),
            }
        )
    return scored


def _owners_by_centroid(pieces: list[dict], markers: list[dict]) -> list[tuple[int, tuple[int, int]]]:
    out = []
    for piece in pieces:
        center = piece["abs_coords"].mean(axis=0)
        scored = []
        for marker in markers:
            dr = center[0] - marker["pos"][0]
            dc = center[1] - marker["pos"][1]
            scored.append((float(dr * dr + dc * dc), int(marker["pos"][0]), int(marker["pos"][1]), int(marker["color"])))
        scored.sort()
        best = scored[0]
        out.append((int(best[3]), (int(best[1]), int(best[2]))))
    return out


def _owners_by_manhattan(pieces: list[dict], markers: list[dict]) -> list[tuple[int, tuple[int, int]]]:
    out = []
    for piece in pieces:
        scored = []
        for marker in markers:
            diffs = np.abs(piece["abs_coords"] - np.array(marker["pos"], dtype=int))
            d = int(np.min(diffs.sum(axis=1)))
            scored.append((d, int(marker["pos"][0]), int(marker["pos"][1]), int(marker["color"])))
        scored.sort()
        best = scored[0]
        out.append((int(best[3]), (int(best[1]), int(best[2]))))
    return out


def _has_crossed_same_area_pair(pieces: list[dict]) -> bool:
    for i, a in enumerate(pieces):
        for b in pieces[i + 1 :]:
            if a["area"] != b["area"]:
                continue
            ar, ac = a["bbox"][:2]
            br, bc = b["bbox"][:2]
            if (ar < br and ac > bc) or (ar > br and ac < bc):
                return True
    return False


def _render_input(shape: tuple[int, int], pieces: list[dict], markers: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for piece in pieces:
        grid[piece["abs_coords"][:, 0], piece["abs_coords"][:, 1]] = GREY
    for marker in markers:
        r, c = marker["pos"]
        grid[int(r), int(c)] = int(marker["color"])
    return grid


def _place_markers(
    pieces: list[dict],
    shape: tuple[int, int],
    *,
    require_tie: bool,
    marker_count: int,
) -> list[dict] | None:
    forbidden = _fragment_forbidden_mask(pieces, shape)
    H, W = shape
    candidates = [(r, c) for r in range(H) for c in range(W) if not forbidden[r, c]]
    if len(candidates) < marker_count:
        return None

    colors = random.sample(PALETTE, marker_count)

    if require_tie:
        random.shuffle(pieces)
        for target in pieces:
            groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for cell in candidates:
                groups[_min_sq_distance(target["abs_coords"], cell)].append(cell)
            group_items = list(groups.items())
            random.shuffle(group_items)
            for dist, cells in group_items:
                if len(cells) < 2:
                    continue
                pairs = [(a, b) for i, a in enumerate(cells) for b in cells[i + 1 :] if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1]
                random.shuffle(pairs)
                for a, b in pairs[:30]:
                    lead, follow = sorted((a, b))
                    chosen = [lead, follow]
                    remaining_pool = [cell for cell in candidates if cell not in chosen and all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen)]
                    remaining_pool = [cell for cell in remaining_pool if _min_sq_distance(target["abs_coords"], cell) > dist]
                    extra = _choose_random_positions(remaining_pool, marker_count - 2)
                    if extra is None and marker_count > 2:
                        continue
                    if extra:
                        chosen.extend(extra)
                    if len(chosen) == marker_count and _non_adjacent_markers(chosen):
                        return [{"pos": pos, "color": color} for pos, color in zip(chosen, colors)]
        return None

    for _ in range(MARKER_ATTEMPTS):
        chosen = _choose_random_positions(candidates, marker_count)
        if chosen is None:
            continue
        markers = [{"pos": pos, "color": color} for pos, color in zip(chosen, colors)]
        scored = _compute_piece_scores(pieces, markers)
        if len({piece["color"] for piece in scored}) >= 2:
            return markers
    return None


def _semantic_checks(grid: np.ndarray, pieces: list[dict], markers: list[dict], require_sort: bool, require_metric: bool, require_tie: bool) -> bool:
    scored = _compute_piece_scores(pieces, markers)
    if len({piece["color"] for piece in scored}) < 2:
        return False
    if require_tie and not any(piece["tie_count"] >= 2 for piece in scored):
        return False

    try:
        correct_out, _placements, _order = _pack_pieces(scored, order_mode="correct")
    except AssertionError:
        return False

    extracted_pieces, extracted_markers = _extract_instance(grid)
    scored_sig = sorted((piece["bbox"], piece["color"], piece["owner_pos"]) for piece in scored)
    extracted_sig = sorted((piece["bbox"], piece["color"], piece["owner_pos"]) for piece in extracted_pieces)
    if scored_sig != extracted_sig:
        return False
    marker_sig = sorted((marker["pos"], marker["color"]) for marker in markers)
    extracted_marker_sig = sorted((marker["pos"], marker["color"]) for marker in extracted_markers)
    if marker_sig != extracted_marker_sig:
        return False
    if not np.array_equal(main(grid), correct_out):
        return False

    if require_sort:
        if not _has_crossed_same_area_pair(scored):
            return False
        try:
            wrong_sort_out, _placements, _order = _pack_pieces(scored, order_mode="colrow")
        except AssertionError:
            return False
        if np.array_equal(correct_out, wrong_sort_out):
            return False

    if require_metric:
        correct_owners = [(piece["color"], piece["owner_pos"]) for piece in scored]
        centroid_owners = _owners_by_centroid(scored, markers)
        manhattan_owners = _owners_by_manhattan(scored, markers)
        if centroid_owners == correct_owners and manhattan_owners == correct_owners:
            return False

    if len(markers) == 2 and len({piece["color"] for piece in scored}) == 1:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    require_sort = random.random() < 0.45
    require_metric = random.random() < 0.45
    require_tie = random.random() < 0.25

    design = _design_shapes(require_repeat=require_sort)
    if design is None:
        return None
    coords_list, _out_shape = design

    scattered = _scatter_pieces(coords_list, randomize_orientations=require_metric or require_tie)
    if scattered is None:
        return None
    grid, pieces = scattered

    marker_max = min(MARKER_COUNT_RANGE[1], len(pieces))
    marker_count = random.randint(MARKER_COUNT_RANGE[0], marker_max)
    markers = _place_markers(pieces, grid.shape, require_tie=require_tie, marker_count=marker_count)
    if markers is None:
        return None

    grid = _render_input(grid.shape, pieces, markers)
    if not _semantic_checks(grid, pieces, markers, require_sort=require_sort, require_metric=require_metric, require_tie=require_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
