from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


BACKGROUND = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7)
MIN_GRID = 13
MAX_GRID = 16
MIN_ANCHORS = 2
MAX_ANCHORS = 3
MIN_SOURCES = 4
MAX_SOURCES = 6
MIN_BLOCK_CELL_GAP = 5
MAX_GENERATION_ATTEMPTS = 800


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _anchor_cells(anchor: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _frame_cells(anchor: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    cells = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if not _in_bounds(shape, r, c):
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            cells.append((r, c))
    return cells


def _distance_to_anchor(src: tuple[int, int], anchor: tuple[int, int, int, int]) -> int:
    return min(_chebyshev(src, cell) for cell in _anchor_cells(anchor))


def _nearest_anchor_index(
    src: tuple[int, int], anchors: list[tuple[int, int, int, int]], *, strict: bool
) -> int:
    distances = [_distance_to_anchor(src, anchor) for anchor in anchors]
    best = min(distances)
    winners = [idx for idx, dist in enumerate(distances) if dist == best]
    if strict:
        assert len(winners) == 1, "each source must have a unique nearest anchor"
    return winners[0]


def _destination_for(
    src: tuple[int, int], anchor: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    return min(_frame_cells(anchor, shape), key=lambda cell: (_chebyshev(src, cell), cell[0], cell[1]))


def _l_path(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = src
    dr, dc = dst
    path = [(sr, sc)]
    c_step = _sign(dc - sc)
    if c_step:
        for c in range(sc + c_step, dc + c_step, c_step):
            path.append((sr, c))
    r_step = _sign(dr - sr)
    if r_step:
        for r in range(sr + r_step, dr + r_step, r_step):
            path.append((r, dc))
    return path


def _l_path_vertical_first(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = src
    dr, dc = dst
    path = [(sr, sc)]
    r_step = _sign(dr - sr)
    if r_step:
        for r in range(sr + r_step, dr + r_step, r_step):
            path.append((r, sc))
    c_step = _sign(dc - sc)
    if c_step:
        for c in range(sc + c_step, dc + c_step, c_step):
            path.append((dr, c))
    return path


def _path_segments(src: tuple[int, int], dst: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    sr, sc = src
    dr, dc = dst
    horizontal = {(sr, c) for c in range(min(sc, dc), max(sc, dc) + 1)}
    vertical = {(r, dc) for r in range(min(sr, dr), max(sr, dr) + 1)}
    return horizontal, vertical


def _paths_cross(rec_a: dict, rec_b: dict) -> bool:
    return bool((rec_a["horizontal"] & rec_b["vertical"]) or (rec_a["vertical"] & rec_b["horizontal"]))


def _find_anchors(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g == GREY
    seen = np.zeros_like(mask, dtype=bool)
    anchors: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(cells) == 4 and r1 == r0 + 1 and c1 == c0 + 1, "grey anchors must be solid 2x2 blocks"
            assert all(mask[rr, cc] for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1))
            anchors.append((r0, c0, r1, c1))
    anchors.sort()
    assert MIN_ANCHORS <= len(anchors) <= MAX_ANCHORS, "expected 2-3 grey anchors"
    return anchors


def _source_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    coords = [tuple(map(int, cell)) for cell in np.argwhere((g != BACKGROUND) & (g != GREY))]
    coords.sort()
    return coords


def _build_records(grid: np.ndarray, *, strict: bool) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert Counter(map(int, g.ravel())).most_common(1)[0][0] == BACKGROUND, "background must be black"

    anchors = _find_anchors(g)
    sources = _source_positions(g)
    assert MIN_SOURCES <= len(sources) <= MAX_SOURCES, "expected 4-6 source pixels"

    records = []
    anchor_assignments = Counter()
    all_anchor_cells = {cell for anchor in anchors for cell in _anchor_cells(anchor)}
    all_frame_cells = {cell for anchor in anchors for cell in _frame_cells(anchor, g.shape)}
    source_colors = [int(g[src]) for src in sources]

    if strict:
        assert len(set(source_colors)) == len(source_colors), "source colors must be distinct"

    for src in sources:
        if strict:
            for nr, nc in _neighbors8(*src):
                if _in_bounds(g.shape, nr, nc):
                    assert int(g[nr, nc]) == BACKGROUND, "source pixels must be isolated singletons"
            assert src not in all_frame_cells and src not in all_anchor_cells, "source pixels must start outside anchor frames"

        anchor_idx = _nearest_anchor_index(src, anchors, strict=strict)
        anchor = anchors[anchor_idx]
        dst = _destination_for(src, anchor, g.shape)
        path = _l_path(src, dst)
        horizontal, vertical = _path_segments(src, dst)

        if strict:
            r0, c0, r1, c1 = anchor
            assert src[0] not in (r0, r1) and src[1] not in (c0, c1), "source must not share a row or column with its anchor"
            assert src[0] != dst[0] and src[1] != dst[1], "each route must have one bend"
            assert all(cell not in all_anchor_cells for cell in path), "paths must not enter anchor cells"

        record = {
            "src": src,
            "color": int(g[src]),
            "anchor_idx": anchor_idx,
            "dst": dst,
            "path": path,
            "horizontal": horizontal,
            "vertical": vertical,
            "length": len(path),
        }
        records.append(record)
        anchor_assignments[anchor_idx] += 1

    if strict:
        destinations = [record["dst"] for record in records]
        assert len(set(destinations)) == len(destinations), "destinations must be unique"
        assert len(anchor_assignments) == len(anchors) and all(count >= 1 for count in anchor_assignments.values())

    return {"anchors": anchors, "records": records}


def _render(
    grid: np.ndarray,
    records: list[dict],
    *,
    path_builder,
    reverse_priority: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    anchor_mask = g == GREY
    working = []
    for record in records:
        path = path_builder(record["src"], record["dst"])
        working.append({**record, "path": path, "length": len(path)})

    ordered = sorted(
        working,
        key=lambda record: (record["length"], record["src"][0], record["src"][1], record["dst"][0], record["dst"][1], record["color"]),
        reverse=reverse_priority,
    )
    for record in ordered:
        for r, c in record["path"]:
            if not anchor_mask[r, c]:
                out[r, c] = record["color"]
    out[anchor_mask] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored singleton to the nearest grey anchor frame by drawing its horizontal-first L-shaped trajectory.

    Input structure:
    - The grid has a black background, 2-3 separated grey 2x2 anchor blocks, and 4-6 colored singleton pixels.
    - Each colored pixel is uniquely closest to one anchor under Chebyshev distance and does not share a row or column with that anchor.
    - Every colored pixel starts outside all anchor frames, so its route to the chosen frame requires one bend.

    Rule:
    - For each colored singleton, assign it to the grey 2x2 anchor whose cells are nearest in Chebyshev distance.
    - On that anchor's one-cell outer frame, choose the cell with minimum Chebyshev distance to the singleton; break ties by smaller row, then smaller column.
    - Draw the singleton's path from its start to that destination by moving horizontally first and then vertically, including both endpoints.
    - Draw shorter paths before longer ones so that, at intersections, the longer trajectory overwrites the shorter one while grey anchor cells remain unchanged.

    Key insight:
    - The output is determined by two coupled decisions: nearest-anchor assignment chooses the destination frame, and path-length priority resolves which color survives when horizontal-first L trajectories cross.
    """
    analysis = _build_records(grid, strict=False)
    return _render(np.asarray(grid, dtype=int), analysis["records"], path_builder=_l_path, reverse_priority=False)


def _axis_cell_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _bbox_cell_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(_axis_cell_gap(a[0], a[2], b[0], b[2]), _axis_cell_gap(a[1], a[3], b[1], b[3]))


def _place_anchors(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    top_lefts = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    random.shuffle(top_lefts)
    candidates = [(r, c, r + 1, c + 1) for r, c in top_lefts]

    chosen: list[tuple[int, int, int, int]] = []

    def search(start: int) -> bool:
        if len(chosen) == count:
            return True
        if len(candidates) - start < count - len(chosen):
            return False
        for idx in range(start, len(candidates)):
            anchor = candidates[idx]
            if all(_bbox_cell_gap(anchor, other) >= MIN_BLOCK_CELL_GAP for other in chosen):
                chosen.append(anchor)
                if search(idx + 1):
                    return True
                chosen.pop()
        return False

    if search(0):
        chosen.sort()
        return chosen
    return None


def _enumerate_candidates(
    shape: tuple[int, int], anchors: list[tuple[int, int, int, int]]
) -> list[dict]:
    h, w = shape
    all_anchor_cells = {cell for anchor in anchors for cell in _anchor_cells(anchor)}
    all_frame_cells = {cell for anchor in anchors for cell in _frame_cells(anchor, shape)}
    candidates: list[dict] = []

    for r in range(h):
        for c in range(w):
            src = (r, c)
            if src in all_anchor_cells or src in all_frame_cells:
                continue
            distances = [_distance_to_anchor(src, anchor) for anchor in anchors]
            best = min(distances)
            winners = [idx for idx, dist in enumerate(distances) if dist == best]
            if len(winners) != 1:
                continue
            anchor_idx = winners[0]
            anchor = anchors[anchor_idx]
            r0, c0, r1, c1 = anchor
            if r in (r0, r1) or c in (c0, c1):
                continue
            dst = _destination_for(src, anchor, shape)
            if src[0] == dst[0] or src[1] == dst[1]:
                continue
            path = _l_path(src, dst)
            if any(cell in all_anchor_cells for cell in path):
                continue
            horizontal, vertical = _path_segments(src, dst)
            candidates.append(
                {
                    "src": src,
                    "anchor_idx": anchor_idx,
                    "dst": dst,
                    "path": path,
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "length": len(path),
                }
            )
    random.shuffle(candidates)
    return candidates


def _compatible(candidate: dict, selected: list[dict]) -> bool:
    for other in selected:
        if _chebyshev(candidate["src"], other["src"]) <= 1:
            return False
        if candidate["dst"] == other["dst"]:
            return False
        if candidate["src"] in other["path"] or candidate["dst"] in other["path"]:
            return False
        if other["src"] in candidate["path"] or other["dst"] in candidate["path"]:
            return False
        shared = set(candidate["path"]) & set(other["path"])
        if shared:
            if len(shared) > 1:
                return False
            if candidate["length"] == other["length"]:
                return False
            if not _paths_cross(candidate, other):
                return False
    return True


def _crossing_pairs(records: list[dict]) -> list[tuple[int, int, tuple[int, int]]]:
    pairs = []
    for i, j in combinations(range(len(records)), 2):
        shared = set(records[i]["path"]) & set(records[j]["path"])
        if shared and _paths_cross(records[i], records[j]):
            assert len(shared) == 1
            pairs.append((i, j, next(iter(shared))))
    return pairs


def _wrong_priority_output(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    return _render(grid, records, path_builder=_l_path, reverse_priority=True)


def _vertical_first_output(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    return _render(grid, records, path_builder=_l_path_vertical_first, reverse_priority=False)


def _semantic_validate(grid: np.ndarray) -> None:
    analysis = _build_records(grid, strict=True)
    records = analysis["records"]
    out = main(grid)
    crossings = _crossing_pairs(records)
    assert crossings, "at least one path pair must cross"

    added_path_cells = int(np.count_nonzero((out != grid) & (grid == BACKGROUND)))
    assert added_path_cells >= len(records), "paths must visibly extend beyond the source pixels"

    vertical_first = _vertical_first_output(grid, records)
    assert not np.array_equal(out, vertical_first), "horizontal-first routing must matter"

    wrong_priority = _wrong_priority_output(grid, records)
    assert not np.array_equal(out, wrong_priority), "path-length priority must matter"

    for i, j, cell in crossings:
        longer = records[i] if records[i]["length"] > records[j]["length"] else records[j]
        assert int(out[cell]) == longer["color"], "longer path must overwrite the shorter one at crossings"


def _choose_singletons(
    shape: tuple[int, int], anchors: list[tuple[int, int, int, int]], target_count: int
) -> list[dict] | None:
    candidates = _enumerate_candidates(shape, anchors)
    crossing_pairs = [
        (a, b)
        for a, b in combinations(candidates, 2)
        if a["length"] != b["length"]
        and _compatible(a, [b])
        and _compatible(b, [a])
        and _paths_cross(a, b)
    ]
    random.shuffle(crossing_pairs)

    for first, second in crossing_pairs[:300]:
        selected = [first, second]

        def search(start_idx: int) -> list[dict] | None:
            if len(selected) == target_count:
                covered = {record["anchor_idx"] for record in selected}
                if len(covered) != len(anchors):
                    return None
                return [dict(record) for record in selected]

            covered = {record["anchor_idx"] for record in selected}
            missing = set(range(len(anchors))) - covered

            ordered_indices = list(range(start_idx, len(candidates)))
            random.shuffle(ordered_indices)
            ordered_indices.sort(key=lambda idx: candidates[idx]["anchor_idx"] not in missing)
            for idx in ordered_indices:
                candidate = candidates[idx]
                if candidate in selected:
                    continue
                if not _compatible(candidate, selected):
                    continue
                selected.append(candidate)
                found = search(idx + 1)
                if found is not None:
                    return found
                selected.pop()
            return None

        found = search(0)
        if found is not None:
            return found
    return None


def _grid_from_layout(
    shape: tuple[int, int], anchors: list[tuple[int, int, int, int]], chosen: list[dict]
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for anchor in anchors:
        for r, c in _anchor_cells(anchor):
            grid[r, c] = GREY

    colors = random.sample(SOURCE_COLORS, k=len(chosen))
    random.shuffle(chosen)
    for color, record in zip(colors, chosen):
        r, c = record["src"]
        grid[r, c] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(MIN_GRID, MAX_GRID)
        w = random.randint(MIN_GRID, MAX_GRID)
        anchor_count = 3 if min(h, w) >= 15 and random.random() < 0.45 else 2
        source_count = random.randint(max(MIN_SOURCES, anchor_count + 2), MAX_SOURCES)

        anchors = _place_anchors(h, w, anchor_count)
        if anchors is None:
            continue
        chosen = _choose_singletons((h, w), anchors, source_count)
        if chosen is None:
            continue

        grid = _grid_from_layout((h, w), anchors, chosen)
        try:
            _semantic_validate(grid)
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
