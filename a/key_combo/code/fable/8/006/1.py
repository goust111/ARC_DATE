from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]

BG = int(Color.BLACK)
SEGMENT_COLOR = int(Color.TEAL)
MARKER_COLOR = int(Color.GREY)
RAY_COLOR = int(Color.GREEN)

GEN_RETRIES = 500
SEGMENT_RETRIES = 250


@dataclass(frozen=True)
class CandidateBox:
    box: BBox
    markers: frozenset[int]


def _find_roles(grid: np.ndarray) -> tuple[int, int, int]:
    bg = int(get_bg(grid))
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert len(colors) == 2, "expected exactly two non-background colors in the input"

    def is_segment_color(color: int) -> bool:
        pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(pts) < 2:
            return False
        comps8 = components_mask(grid == color, connectivity=8)
        if len(comps8) != 1:
            return False
        diffs = {c - r for r, c in pts}
        sums = {r + c for r, c in pts}
        if len(diffs) == 1:
            ordered = sorted(pts)
            return all(r2 - r1 == 1 and c2 - c1 == 1 for (r1, c1), (r2, c2) in zip(ordered, ordered[1:], strict=False))
        if len(sums) == 1:
            ordered = sorted(pts, key=lambda rc: (rc[0], -rc[1]))
            return all(r2 - r1 == 1 and c2 - c1 == -1 for (r1, c1), (r2, c2) in zip(ordered, ordered[1:], strict=False))
        return False

    segment_candidates = [color for color in colors if is_segment_color(color)]
    assert len(segment_candidates) == 1, "failed to identify the diagonal segment color"
    segment_color = int(segment_candidates[0])
    marker_color = int(colors[0] if colors[1] == segment_color else colors[1])
    return bg, marker_color, segment_color


def _extract_segment(grid: np.ndarray, color: int) -> tuple[list[Coord], list[tuple[Coord, Coord]]]:
    comps8 = components_mask(grid == color, connectivity=8)
    assert len(comps8) == 1, "expected one diagonal segment"
    coords = [tuple(map(int, rc)) for rc in comps8[0].tolist()]
    assert len(coords) >= 2, "segment must have at least two cells"

    diffs = {c - r for r, c in coords}
    sums = {r + c for r, c in coords}

    rays: list[tuple[Coord, Coord]]
    if len(diffs) == 1:
        coords.sort()
        ordered = coords
        for (r1, c1), (r2, c2) in zip(ordered, ordered[1:], strict=False):
            assert r2 - r1 == 1 and c2 - c1 == 1, "segment must be a contiguous diagonal"
        rays = [(ordered[0], (-1, -1)), (ordered[-1], (1, 1))]
    else:
        assert len(sums) == 1, "segment must lie on one diagonal"
        coords.sort(key=lambda rc: (rc[0], -rc[1]))
        ordered = coords
        for (r1, c1), (r2, c2) in zip(ordered, ordered[1:], strict=False):
            assert r2 - r1 == 1 and c2 - c1 == -1, "segment must be a contiguous diagonal"
        rays = [(ordered[0], (-1, 1)), (ordered[-1], (1, -1))]

    return ordered, rays


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _enumerate_candidates(markers: list[Coord], shape: tuple[int, int]) -> list[CandidateBox]:
    h, w = map(int, shape)
    top_edges = sorted({r + 1 for r, _c in markers if r + 1 < h})
    bottom_edges = sorted({r - 1 for r, _c in markers if r - 1 >= 0})
    left_edges = sorted({c + 1 for _r, c in markers if c + 1 < w})
    right_edges = sorted({c - 1 for _r, c in markers if c - 1 >= 0})

    candidates: dict[tuple[BBox, frozenset[int]], CandidateBox] = {}
    for r0 in top_edges:
        for r1 in bottom_edges:
            if r1 - r0 + 1 < 2:
                continue
            for c0 in left_edges:
                for c1 in right_edges:
                    if c1 - c0 + 1 < 2:
                        continue
                    left = {i for i, (r, c) in enumerate(markers) if c == c0 - 1 and r0 <= r <= r1}
                    right = {i for i, (r, c) in enumerate(markers) if c == c1 + 1 and r0 <= r <= r1}
                    top = {i for i, (r, c) in enumerate(markers) if r == r0 - 1 and c0 <= c <= c1}
                    bottom = {i for i, (r, c) in enumerate(markers) if r == r1 + 1 and c0 <= c <= c1}
                    if not left or not right or not top or not bottom:
                        continue
                    covered = left | right | top | bottom
                    ambiguous = False
                    for idx in covered:
                        memberships = sum(idx in side for side in (left, right, top, bottom))
                        if memberships != 1:
                            ambiguous = True
                            break
                    if ambiguous:
                        continue
                    box = (int(r0), int(c0), int(r1), int(c1))
                    key = (box, frozenset(covered))
                    candidates[key] = CandidateBox(box=box, markers=frozenset(covered))
    assert candidates, "expected at least one rectangle candidate"
    return list(candidates.values())


def _recover_rectangles(markers: list[Coord], shape: tuple[int, int]) -> list[BBox]:
    candidates = _enumerate_candidates(markers, shape)
    marker_to_candidates: dict[int, list[int]] = {i: [] for i in range(len(markers))}
    for idx, cand in enumerate(candidates):
        for marker_idx in cand.markers:
            marker_to_candidates[marker_idx].append(idx)

    for marker_idx, options in marker_to_candidates.items():
        assert options, f"marker {marker_idx} does not belong to any candidate rectangle"

    solutions: list[list[int]] = []

    def search(remaining: set[int], chosen: list[int], chosen_boxes: list[BBox]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return

        def viable(cid: int) -> bool:
            cand = candidates[cid]
            if not cand.markers <= remaining:
                return False
            return not any(_boxes_overlap(cand.box, other) for other in chosen_boxes)

        pivot = min(remaining, key=lambda marker_idx: sum(1 for cid in marker_to_candidates[marker_idx] if viable(cid)))
        options = [cid for cid in marker_to_candidates[pivot] if viable(cid)]
        assert options, "markers do not admit an exact rectangle cover"
        for cid in options:
            cand = candidates[cid]
            search(remaining - set(cand.markers), chosen + [cid], chosen_boxes + [cand.box])

    search(set(range(len(markers))), [], [])
    assert len(solutions) == 1, "control pixels must define a unique set of hidden rectangles"
    boxes = [candidates[cid].box for cid in solutions[0]]
    boxes.sort()
    return boxes


def _blocked_mask(shape: tuple[int, int], boxes: list[BBox]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _box_containing(boxes: list[BBox], cell: Coord) -> BBox | None:
    r, c = cell
    for box in boxes:
        r0, c0, r1, c1 = box
        if r0 <= r <= r1 and c0 <= c <= c1:
            return box
    return None


def _trace_rays(
    shape: tuple[int, int],
    boxes: list[BBox],
    rays: list[tuple[Coord, Coord]],
) -> tuple[set[Coord], list[dict[str, object]]]:
    blocked = _blocked_mask(shape, boxes)
    painted: set[Coord] = set()
    bounces: list[dict[str, object]] = []

    for start, direction in rays:
        r, c = map(int, start)
        dr, dc = map(int, direction)
        seen_states: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, dr, dc)
            assert state not in seen_states, "ray entered a reflection loop"
            seen_states.add(state)

            nr, nc = r + dr, c + dc
            row_hit = in_bounds(shape, nr, c) and bool(blocked[nr, c])
            col_hit = in_bounds(shape, r, nc) and bool(blocked[r, nc])
            corner_hit = in_bounds(shape, nr, nc) and bool(blocked[nr, nc]) and not (row_hit or col_hit)

            if row_hit or col_hit or corner_hit:
                entry: dict[str, object] = {
                    "cell": (r, c),
                    "direction": (dr, dc),
                    "row_hit": bool(row_hit or corner_hit),
                    "col_hit": bool(col_hit or corner_hit),
                    "row_box": _box_containing(boxes, (nr, c)) if row_hit else (_box_containing(boxes, (nr, nc)) if corner_hit else None),
                    "col_box": _box_containing(boxes, (r, nc)) if col_hit else (_box_containing(boxes, (nr, nc)) if corner_hit else None),
                }
                bounces.append(entry)
                if row_hit or corner_hit:
                    dr = -dr
                if col_hit or corner_hit:
                    dc = -dc
                continue

            if not in_bounds(shape, nr, nc):
                break

            r, c = nr, nc
            painted.add((r, c))

    return painted, bounces


def _render_output(grid: np.ndarray, segment_cells: list[Coord], ray_cells: set[Coord], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    segment_set = set(segment_cells)
    for r, c in ray_cells:
        if (r, c) in segment_set:
            continue
        if int(out[r, c]) == bg:
            out[r, c] = RAY_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend two diagonal rays from a short cyan segment and reflect them off marker-defined invisible rectangles.

    Input structure:
    - Black background with one short cyan diagonal segment.
    - Several isolated gray control pixels lie exactly one cell outside the sides of one or more hidden axis-aligned rectangles.
    - Every hidden rectangle has at least one control pixel for each of its four sides, but the rectangles themselves are never drawn.

    Rule:
    - Reconstruct the unique set of hidden rectangles implied by the gray pixels.
    - From the two ends of the cyan segment, launch green rays outward along the segment's diagonal direction.
    - Whenever a ray would enter a hidden rectangle, reflect it by flipping the blocked direction component and continue until it leaves the grid; keep the cyan segment and gray pixels unchanged, and never draw the hidden rectangles themselves.

    Key insight:
    - The gray pixels are not targets to connect: they encode full invisible rectangle walls, so a correct ray path must bounce on unmarked wall spans as well as marked ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, marker_color, segment_color = _find_roles(g)

    segment_cells, rays = _extract_segment(g, segment_color)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == marker_color)]
    assert markers, "expected control markers"
    boxes = _recover_rectangles(markers, g.shape)

    ray_cells, _bounces = _trace_rays(g.shape, boxes, rays)
    return _render_output(g, segment_cells, ray_cells, bg)


def _segment_cells(start: Coord, slope: int, length: int) -> list[Coord]:
    r0, c0 = start
    if slope == 1:
        return [(r0 + k, c0 + k) for k in range(length)]
    return [(r0 + k, c0 - k) for k in range(length)]


def _place_rectangles(shape: tuple[int, int], nrect: int) -> list[BBox] | None:
    h, w = map(int, shape)
    boxes: list[BBox] = []
    for _ in range(nrect):
        placed = False
        for _ in range(200):
            height = random.randint(4, 8)
            width = random.randint(4, 8)
            if height + 6 > h or width + 6 > w:
                continue
            r0 = random.randint(2, h - height - 3)
            c0 = random.randint(2, w - width - 3)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            expanded = (r0 - 2, c0 - 2, r0 + height, c0 + width)
            if any(_boxes_overlap(expanded, (b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2)) for b in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    boxes.sort()
    return boxes


def _sample_side_positions(lo: int, hi: int) -> list[int]:
    pts = random.sample(list(range(lo, hi + 1)), 1)
    pts.sort()
    return pts


def _markers_for_box(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    markers: list[Coord] = []
    for r in _sample_side_positions(r0, r1):
        markers.append((r, c0 - 1))
    for r in _sample_side_positions(r0, r1):
        markers.append((r, c1 + 1))
    for c in _sample_side_positions(c0, c1):
        markers.append((r0 - 1, c))
    for c in _sample_side_positions(c0, c1):
        markers.append((r1 + 1, c))
    markers = sorted(set(markers))
    assert any(c == c0 - 1 for _r, c in markers)
    assert any(c == c1 + 1 for _r, c in markers)
    assert any(r == r0 - 1 for r, _c in markers)
    assert any(r == r1 + 1 for r, _c in markers)
    return markers


def _paint_input(shape: tuple[int, int], boxes: list[BBox], markers_by_box: list[list[Coord]], segment: list[Coord]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for markers in markers_by_box:
        for r, c in markers:
            grid[r, c] = MARKER_COLOR
    for r, c in segment:
        grid[r, c] = SEGMENT_COLOR
    return grid


def _targeted_segment_options(shape: tuple[int, int], boxes: list[BBox], markers_by_box: list[list[Coord]]) -> list[list[Coord]]:
    height, width = map(int, shape)
    marker_set = {m for group in markers_by_box for m in group}
    blocked = _blocked_mask(shape, boxes)
    occupied = marker_set.copy()

    options: list[list[Coord]] = []
    bounce_entries: list[tuple[Coord, Coord]] = []
    for box in boxes:
        r0, c0, r1, c1 = box
        for c in range(c0, c1 + 1):
            if (r0 - 1, c) not in marker_set:
                bounce_entries.extend([((r0 - 1, c), (1, -1)), ((r0 - 1, c), (1, 1))])
            if (r1 + 1, c) not in marker_set:
                bounce_entries.extend([((r1 + 1, c), (-1, -1)), ((r1 + 1, c), (-1, 1))])
        for r in range(r0, r1 + 1):
            if (r, c0 - 1) not in marker_set:
                bounce_entries.extend([((r, c0 - 1), (-1, 1)), ((r, c0 - 1), (1, 1))])
            if (r, c1 + 1) not in marker_set:
                bounce_entries.extend([((r, c1 + 1), (-1, -1)), ((r, c1 + 1), (1, -1))])

    random.shuffle(bounce_entries)
    for bounce_cell, direction in bounce_entries:
        br, bc = bounce_cell
        dr, dc = direction
        distances = list(range(1, 9))
        lengths = [2, 3, 4, 5]
        random.shuffle(distances)
        random.shuffle(lengths)
        for dist in distances:
            endpoint = (br - dr * dist, bc - dc * dist)
            if not in_bounds(shape, endpoint[0], endpoint[1]):
                continue
            path_cells = [(endpoint[0] + dr * step, endpoint[1] + dc * step) for step in range(dist + 1)]
            if any(not in_bounds(shape, r, c) or (r, c) in occupied or blocked[r, c] for r, c in path_cells):
                continue
            for length in lengths:
                segment = [(endpoint[0] - dr * step, endpoint[1] - dc * step) for step in range(length)]
                if any(not in_bounds(shape, r, c) or (r, c) in occupied or blocked[r, c] for r, c in segment):
                    continue
                options.append(segment)
                if len(options) >= SEGMENT_RETRIES:
                    return options
    return options


def _marker_only_boxes(boxes: list[BBox], markers_by_box: list[list[Coord]]) -> list[BBox]:
    local: list[BBox] = []
    for box, markers in zip(boxes, markers_by_box, strict=False):
        r0, c0, r1, c1 = box
        for r, c in markers:
            if c == c0 - 1:
                local.append((r, c0, r, c0))
            elif c == c1 + 1:
                local.append((r, c1, r, c1))
            elif r == r0 - 1:
                local.append((r0, c, r0, c))
            elif r == r1 + 1:
                local.append((r1, c, r1, c))
    return local


def _has_unmarked_side_bounce(bounces: list[dict[str, object]], markers: set[Coord]) -> bool:
    for bounce in bounces:
        cell = bounce["cell"]
        dr, dc = bounce["direction"]
        row_hit = bool(bounce["row_hit"])
        col_hit = bool(bounce["col_hit"])
        if row_hit == col_hit:
            continue
        r, c = cell
        if row_hit:
            box = bounce["row_box"]
            assert box is not None
            r0, _c0, r1, _c1 = box
            probe = (r0 - 1, c) if dr == 1 else (r1 + 1, c)
        else:
            box = bounce["col_box"]
            assert box is not None
            _r0, c0, _r1, c1 = box
            probe = (r, c0 - 1) if dc == 1 else (r, c1 + 1)
        if probe not in markers:
            return True
    return False


def _validate_instance(grid: np.ndarray, boxes: list[BBox], markers_by_box: list[list[Coord]], segment: list[Coord]) -> bool:
    try:
        recovered = _recover_rectangles([m for group in markers_by_box for m in group], grid.shape)
    except AssertionError:
        return False
    if recovered != sorted(boxes):
        return False

    segment_cells, rays = _extract_segment(grid, SEGMENT_COLOR)
    if segment_cells != segment:
        return False

    ray_cells, bounces = _trace_rays(grid.shape, boxes, rays)
    if len(ray_cells) < 6 or not bounces:
        return False

    markers = {m for group in markers_by_box for m in group}
    if not _has_unmarked_side_bounce(bounces, markers):
        return False

    protected = markers | set(segment)
    if ray_cells & protected:
        return False

    true_out = _render_output(grid, segment, ray_cells, BG)
    naive_cells, _ = _trace_rays(grid.shape, _marker_only_boxes(boxes, markers_by_box), rays)
    naive_out = _render_output(grid, segment, naive_cells, BG)
    if np.array_equal(true_out, naive_out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    shape = (height, width)

    nrect = random.choice([1, 1, 2, 2, 2, 3])
    boxes = _place_rectangles(shape, nrect)
    if boxes is None:
        return None

    markers_by_box = [_markers_for_box(box) for box in boxes]
    segment_options = _targeted_segment_options(shape, boxes, markers_by_box)
    if not segment_options:
        return None
    for segment in segment_options:
        grid = _paint_input(shape, boxes, markers_by_box, segment)
        if _validate_instance(grid, boxes, markers_by_box, segment):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phantom-rectangle reflection input")
