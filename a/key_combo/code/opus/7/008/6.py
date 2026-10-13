from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

FRAME_INTERIOR_SIDES = (2, 3, 4)
SOURCE_SIZE_RANGE = (6, 12)
GRID_SIDE_RANGE = (14, 22)
MAX_ATTEMPTS = 600

QUADRANT_OPTIONS = [
    ("NW",),
    ("NE",),
    ("SW",),
    ("SE",),
    ("NW", "NE"),
    ("NE", "SE"),
    ("SW", "SE"),
    ("NW", "SW"),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            comp: set[tuple[int, int]] = set()
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nb in _neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _quadrant_of(r: int, c: int, marker_r: int, marker_c: int) -> str:
    vertical = "N" if r <= marker_r else "S"
    horizontal = "W" if c <= marker_c else "E"
    return vertical + horizontal


def _reflect_cell(r: int, c: int, marker_r: int, marker_c: int) -> tuple[int, int]:
    return 2 * marker_r + 1 - r, 2 * marker_c + 1 - c


def _closure(cells: set[tuple[int, int]], marker_r: int, marker_c: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    row_sum = 2 * marker_r + 1
    col_sum = 2 * marker_c + 1
    for r, c in cells:
        for rr in (r, row_sum - r):
            for cc in (c, col_sum - c):
                out.add((rr, cc))
    return out


def _axis_closure(
    cells: set[tuple[int, int]],
    marker_r: int,
    marker_c: int,
    *,
    horizontal: bool,
    vertical: bool,
) -> set[tuple[int, int]]:
    out = set(cells)
    row_sum = 2 * marker_r + 1
    col_sum = 2 * marker_c + 1
    for r, c in list(cells):
        if horizontal:
            out.add((row_sum - r, c))
        if vertical:
            out.add((r, col_sum - c))
        if horizontal and vertical:
            out.add((row_sum - r, col_sum - c))
    return out


def _find_marker_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREEN)
    assert coords.shape == (4, 2), "expected exactly four green marker pixels"
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    assert r1 == r0 + 1 and c1 == c0 + 1, "marker must be a solid 2x2 square"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN)
    return r0, c0, r1, c1


def _extract_frame_boxes(grid: np.ndarray, *, require_frames: bool) -> list[tuple[int, int, int, int]]:
    blue_comps = _components_of_color(grid, BLUE)
    if require_frames:
        assert 1 <= len(blue_comps) <= 2, "expected one or two blue frames"
    frame_boxes: list[tuple[int, int, int, int]] = []
    for comp in blue_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4, "frame must have non-empty interior"
        expected_area = 2 * h + 2 * w - 4
        assert len(comp) == expected_area, "blue component must be a one-cell-thick rectangle border"
        assert np.all(grid[r0, c0 : c1 + 1] == BLUE)
        assert np.all(grid[r1, c0 : c1 + 1] == BLUE)
        assert np.all(grid[r0 : r1 + 1, c0] == BLUE)
        assert np.all(grid[r0 : r1 + 1, c1] == BLUE)
        assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must stay black in the input"
        frame_boxes.append((r0, c0, r1, c1))
    return frame_boxes


def _frame_forbidden_mask(
    shape: tuple[int, int], frame_boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _solve(grid: np.ndarray, *, require_frames: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker_r, marker_c, marker_r1, marker_c1 = _find_marker_box(g)
    assert marker_r1 == marker_r + 1 and marker_c1 == marker_c + 1

    frame_boxes = _extract_frame_boxes(g, require_frames=require_frames)
    forbidden = _frame_forbidden_mask(g.shape, frame_boxes)
    out = g.copy()
    for r0, c0 in np.argwhere(g == RED):
        r = int(r0)
        c = int(c0)
        for rr, cc in _closure({(r, c)}, marker_r, marker_c):
            if forbidden[rr, cc]:
                continue
            if int(g[rr, cc]) == BG:
                out[rr, cc] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the red figure by four-way mirror reflection around the green 2x2 marker, but suppress any reflected pixel that lands inside a blue hollow frame.

    Input structure:
    - A black background contains one solid green 2x2 marker square near the middle.
    - Red source pixels form one irregular 4-connected shape in one quadrant or across two adjacent quadrants relative to that marker.
    - One or two blue hollow rectangular frames sit in a single target quadrant; their interiors are black in the input.

    Rule:
    - Use the center of the green 2x2 marker as both the horizontal and vertical mirror axis.
    - For every red source pixel, consider its horizontal reflection, vertical reflection, and 180-degree reflection.
    - Keep original red pixels unchanged, and paint a reflected pixel red only when the destination cell is black and not inside any blue frame's border-or-interior region.
    - Leave the green marker, the blue frame borders, and each frame interior unchanged.

    Key insight:
    - The blue hollow rectangles are not objects to fill; each whole framed area acts as a forbidden zone that blocks otherwise valid mirror-completion pixels.
    """
    return _solve(grid, require_frames=True)


def _quadrant_bounds(
    quadrant: str, height: int, width: int, marker_r: int, marker_c: int
) -> tuple[int, int, int, int]:
    if quadrant == "NW":
        return 0, marker_r, 0, marker_c
    if quadrant == "NE":
        return 0, marker_r, marker_c + 1, width - 1
    if quadrant == "SW":
        return marker_r + 1, height - 1, 0, marker_c
    return marker_r + 1, height - 1, marker_c + 1, width - 1


def _is_reflectable(r: int, c: int, height: int, width: int, marker_r: int, marker_c: int) -> bool:
    rr, cc = _reflect_cell(r, c, marker_r, marker_c)
    return 0 <= rr < height and 0 <= cc < width


def _sample_source_cells(
    height: int,
    width: int,
    marker_r: int,
    marker_c: int,
    quadrants: tuple[str, ...],
    count: int,
) -> set[tuple[int, int]] | None:
    allowed = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in {
            (marker_r, marker_c),
            (marker_r, marker_c + 1),
            (marker_r + 1, marker_c),
            (marker_r + 1, marker_c + 1),
        }
        and _quadrant_of(r, c, marker_r, marker_c) in quadrants
        and _is_reflectable(r, c, height, width, marker_r, marker_c)
    }
    if len(allowed) < count:
        return None

    candidate_seeds = []
    for cell in allowed:
        degree = sum(nb in allowed for nb in _neighbors4(*cell))
        if degree >= 2:
            candidate_seeds.append(cell)
    if not candidate_seeds:
        return None

    cells = {random.choice(candidate_seeds)}
    while len(cells) < count:
        frontier = []
        for cell in cells:
            for nb in _neighbors4(*cell):
                if nb in allowed and nb not in cells:
                    frontier.append(nb)
        if not frontier:
            return None
        random.shuffle(frontier)
        scored = sorted(
            frontier,
            key=lambda cell: (
                sum(nb in allowed and nb not in cells for nb in _neighbors4(*cell)),
                random.random(),
            ),
            reverse=True,
        )
        pick_pool = scored[: min(5, len(scored))]
        cells.add(random.choice(pick_pool))

    if not _is_connected(cells):
        return None
    used_quadrants = {_quadrant_of(r, c, marker_r, marker_c) for r, c in cells}
    if used_quadrants != set(quadrants):
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 == r0 or c1 == c0:
        return None

    full = _closure(cells, marker_r, marker_c)
    if full == cells:
        return None
    if _axis_closure(cells, marker_r, marker_c, horizontal=True, vertical=False) == full:
        return None
    if _axis_closure(cells, marker_r, marker_c, horizontal=False, vertical=True) == full:
        return None
    return cells


def _boxes_overlap_or_touch(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], *, margin: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE


def _frame_candidates(
    targets: set[tuple[int, int]],
    quadrant: str,
    height: int,
    width: int,
    marker_r: int,
    marker_c: int,
    occupied: np.ndarray,
) -> list[dict]:
    qr0, qr1, qc0, qc1 = _quadrant_bounds(quadrant, height, width, marker_r, marker_c)
    out: dict[tuple[int, int, int, int], dict] = {}
    for inner_h in FRAME_INTERIOR_SIDES:
        outer_h = inner_h + 2
        if qr1 - qr0 + 1 < outer_h:
            continue
        for inner_w in FRAME_INTERIOR_SIDES:
            outer_w = inner_w + 2
            if qc1 - qc0 + 1 < outer_w:
                continue
            for tr, tc in targets:
                row_start = max(qr0, tr - outer_h + 1)
                row_end = min(tr, qr1 - outer_h + 1)
                col_start = max(qc0, tc - outer_w + 1)
                col_end = min(tc, qc1 - outer_w + 1)
                for r0 in range(row_start, row_end + 1):
                    r1 = r0 + outer_h - 1
                    for c0 in range(col_start, col_end + 1):
                        c1 = c0 + outer_w - 1
                        if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]):
                            continue
                        covered = {(r, c) for r, c in targets if r0 <= r <= r1 and c0 <= c <= c1}
                        if not covered:
                            continue
                        interior = {(r, c) for r, c in covered if r0 < r < r1 and c0 < c < c1}
                        if not interior:
                            continue
                        box = (r0, c0, r1, c1)
                        out[box] = {
                            "box": box,
                            "covered": covered,
                            "interior": interior,
                        }
    return list(out.values())


def _choose_frames_for_quadrant(
    targets: set[tuple[int, int]],
    quadrant: str,
    height: int,
    width: int,
    marker_r: int,
    marker_c: int,
    occupied: np.ndarray,
    desired_count: int,
) -> list[tuple[int, int, int, int]] | None:
    candidates = _frame_candidates(targets, quadrant, height, width, marker_r, marker_c, occupied)
    random.shuffle(candidates)
    if desired_count == 1:
        eligible = [
            item
            for item in candidates
            if 0 < len(item["covered"]) < len(targets)
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda item: (len(item["interior"]), -len(item["covered"]), random.random()), reverse=True)
        return [eligible[0]["box"]]

    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            if _boxes_overlap_or_touch(first["box"], second["box"], margin=1):
                continue
            union = first["covered"] | second["covered"]
            if len(union) >= len(targets):
                continue
            if not (first["covered"] - second["covered"]):
                continue
            if not (second["covered"] - first["covered"]):
                continue
            return [first["box"], second["box"]]
    return None


def _frame_interior_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _validate_instance(
    inp: np.ndarray,
    *,
    source_cells: set[tuple[int, int]],
    marker_r: int,
    marker_c: int,
    blocked_quadrant: str,
) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    source_comps = _components_of_color(inp, RED)
    if len(source_comps) != 1 or source_comps[0] != source_cells:
        return False
    if not (SOURCE_SIZE_RANGE[0] <= len(source_cells) <= SOURCE_SIZE_RANGE[1]):
        return False

    frame_boxes = _extract_frame_boxes(inp, require_frames=True)
    if not (1 <= len(frame_boxes) <= 2):
        return False
    frame_quadrants = {
        _quadrant_of(r0, c0, marker_r, marker_c)
        for r0, c0, _r1, _c1 in frame_boxes
    }
    if frame_quadrants != {blocked_quadrant}:
        return False
    for r0, c0, r1, c1 in frame_boxes:
        if (r1 - r0 - 1) not in FRAME_INTERIOR_SIDES or (c1 - c0 - 1) not in FRAME_INTERIOR_SIDES:
            return False

    full = _closure(source_cells, marker_r, marker_c)
    if _axis_closure(source_cells, marker_r, marker_c, horizontal=True, vertical=False) == full:
        return False
    if _axis_closure(source_cells, marker_r, marker_c, horizontal=False, vertical=True) == full:
        return False

    stripped = inp.copy()
    stripped[stripped == BLUE] = BG
    unblocked = _solve(stripped, require_frames=False)
    diff = {
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if int(unblocked[r, c]) == RED and int(out[r, c]) != RED
    }
    if not diff:
        return False

    frame_union = set().union(*(_frame_interior_cells(box) for box in frame_boxes))
    if not (diff & frame_union):
        return False

    blocked_targets = {cell for cell in diff if _quadrant_of(cell[0], cell[1], marker_r, marker_c) == blocked_quadrant}
    if not blocked_targets:
        return False

    visible_reflections = {
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if int(out[r, c]) == RED and int(inp[r, c]) != RED
    }
    if not {
        cell
        for cell in visible_reflections
        if _quadrant_of(cell[0], cell[1], marker_r, marker_c) == blocked_quadrant
    }:
        return False

    if len({_quadrant_of(r, c, marker_r, marker_c) for r, c in visible_reflections}) < 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    marker_r = random.randint(4, height - 6)
    marker_c = random.randint(4, width - 6)
    source_quadrants = random.choice(QUADRANT_OPTIONS)
    source_count = random.randint(*SOURCE_SIZE_RANGE)
    source_cells = _sample_source_cells(height, width, marker_r, marker_c, source_quadrants, source_count)
    if source_cells is None:
        return None

    full = _closure(source_cells, marker_r, marker_c)
    reflected_only = full - source_cells
    targets_by_quadrant: dict[str, set[tuple[int, int]]] = {q: set() for q in ("NW", "NE", "SW", "SE")}
    for cell in reflected_only:
        targets_by_quadrant[_quadrant_of(cell[0], cell[1], marker_r, marker_c)].add(cell)

    occupied = np.zeros((height, width), dtype=bool)
    for r, c in source_cells:
        occupied[r, c] = True
    occupied[marker_r : marker_r + 2, marker_c : marker_c + 2] = True

    desired_frames = random.choices([1, 2], weights=[3, 2], k=1)[0]
    blocked_candidates = [q for q in ("NW", "NE", "SW", "SE") if q not in set(source_quadrants)]
    random.shuffle(blocked_candidates)

    chosen_frames = None
    chosen_quadrant = None
    for quadrant in blocked_candidates:
        targets = targets_by_quadrant[quadrant]
        if len(targets) < desired_frames + 1:
            continue
        frames = _choose_frames_for_quadrant(
            targets,
            quadrant,
            height,
            width,
            marker_r,
            marker_c,
            occupied,
            desired_frames,
        )
        if frames is not None:
            chosen_frames = frames
            chosen_quadrant = quadrant
            break
    if chosen_frames is None or chosen_quadrant is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[marker_r : marker_r + 2, marker_c : marker_c + 2] = GREEN
    for r, c in source_cells:
        inp[r, c] = RED
    for box in chosen_frames:
        _draw_frame(inp, box)

    if not _validate_instance(
        inp,
        source_cells=source_cells,
        marker_r=marker_r,
        marker_c=marker_c,
        blocked_quadrant=chosen_quadrant,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-frame mirror-blocking ARC input")
