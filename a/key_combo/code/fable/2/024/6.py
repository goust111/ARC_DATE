from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
OVERLAP = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
CORE_RING = {
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
}
LOCAL_LIMIT = 9
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + int(dr), c + int(dc)) for r, c in cells}


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(comp, dtype=int))

    return components


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        comps = _components_from_mask(grid == int(color))
        for coords in comps:
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _hole_components_from_cells(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox_from_cells(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local_occ = {(r - r0, c - c0) for r, c in cells}

    exterior: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if (r, c) in local_occ or (r, c) in exterior:
                continue
            exterior.add((r, c))
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in local_occ and (nr, nc) not in exterior:
                exterior.add((nr, nc))
                queue.append((nr, nc))

    holes: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in local_occ or (r, c) in exterior or (r, c) in seen:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen.add((r, c))
            while queue:
                rr, cc = queue.popleft()
                comp.append((r0 + rr, c0 + cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in local_occ and (nr, nc) not in exterior and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            holes.append(comp)
    return holes


def _single_hole_anchor(coords: np.ndarray) -> tuple[int, int] | None:
    holes = _hole_components_from_cells(_coords_to_set(coords))
    if len(holes) != 1 or len(holes[0]) != 1:
        return None
    return holes[0][0]


def _detect_hole_shapes(grid: np.ndarray, bg: int) -> list[dict]:
    hole_shapes: list[dict] = []
    for component in _find_components(grid, bg):
        anchor = _single_hole_anchor(component["coords"])
        if anchor is None:
            continue
        hole_shapes.append(
            {
                "color": int(component["color"]),
                "coords": component["coords"],
                "bbox": component["bbox"],
                "anchor": anchor,
            }
        )
    return hole_shapes


def _render_difference(
    cells_a: set[tuple[int, int]],
    color_a: int,
    cells_b: set[tuple[int, int]],
    color_b: int,
    bg: int,
) -> np.ndarray:
    union = cells_a | cells_b
    r0, c0, r1, c1 = _bbox_from_cells(union)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in union:
        has_a = (r, c) in cells_a
        has_b = (r, c) in cells_b
        if has_a and has_b:
            out[r - r0, c - c0] = OVERLAP
        elif has_a:
            out[r - r0, c - c0] = int(color_a)
        else:
            out[r - r0, c - c0] = int(color_b)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the two hole-bearing masks by their enclosed background anchors and return the cropped three-color difference map.

    Input structure
    - The grid has a dominant background color and several disconnected monochrome components.
    - Exactly two components are the compared masks: they are non-green and each contains one fully enclosed background hole cell.
    - Optional distractor components contain no enclosed hole and do not participate.

    Rule
    - Detect the two components with a single internal hole cell and use that hole cell as the component's anchor.
    - Translate one component so the two anchor holes coincide, then take the bounding box of the aligned union as the output canvas.
    - Paint overlapping occupied cells green, cells covered by only one mask with that mask's original color, and leave uncovered cells, including the shared hole cell, as background.

    Key insight
    - The registration mark is negative space rather than pigment: only hole-to-hole alignment is stable, and the output is a trinary overlap-versus-difference map instead of a plain union or intersection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    hole_shapes = _detect_hole_shapes(g, bg)
    assert len(hole_shapes) == 2, "expected exactly two hole-bearing components"

    first, second = hole_shapes
    color_a = int(first["color"])
    color_b = int(second["color"])
    assert color_a != OVERLAP and color_b != OVERLAP, "green is reserved for overlap"
    assert color_a != color_b, "the compared masks must use different colors"

    anchor_a = first["anchor"]
    anchor_b = second["anchor"]
    dr = int(anchor_a[0] - anchor_b[0])
    dc = int(anchor_a[1] - anchor_b[1])

    cells_a = _coords_to_set(first["coords"])
    cells_b = _shift_cells(_coords_to_set(second["coords"]), dr, dc)
    return _render_difference(cells_a, color_a, cells_b, color_b, bg)


def _hole_sizes(cells: set[tuple[int, int]]) -> list[int]:
    return sorted(len(comp) for comp in _hole_components_from_cells(cells))


def _valid_tree_add(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    if cell in occupied or abs(r) > LOCAL_LIMIT or abs(c) > LOCAL_LIMIT:
        return False
    return sum((r + dr, c + dc) in occupied for dr, dc in DIRS4) == 1


def _weighted_choice(weighted_items: list[tuple[object, int]]):
    total = sum(weight for _item, weight in weighted_items)
    pick = random.randint(1, total)
    seen = 0
    for item, weight in weighted_items:
        seen += weight
        if pick <= seen:
            return item
    return weighted_items[-1][0]


def _grow_branch(
    occupied: set[tuple[int, int]],
    preferred_dir: tuple[int, int],
    *,
    min_len: int,
    max_len: int,
) -> set[tuple[int, int]] | None:
    root_options: list[tuple[tuple[int, int], int]] = []
    for root in occupied:
        first = (root[0] + preferred_dir[0], root[1] + preferred_dir[1])
        if not _valid_tree_add(first, occupied):
            continue
        if _hole_sizes(set(occupied) | {first}) != [1]:
            continue
        score = 4 + abs(first[0]) + abs(first[1])
        if preferred_dir[0] != 0 and root[0] * preferred_dir[0] >= 0:
            score += 1
        if preferred_dir[1] != 0 and root[1] * preferred_dir[1] >= 0:
            score += 1
        root_options.append((root, score))
    if not root_options:
        return None

    root = _weighted_choice(root_options)
    current = (root[0] + preferred_dir[0], root[1] + preferred_dir[1])
    branch = {current}
    occupied_with_branch = set(occupied) | branch
    target_len = random.randint(min_len, max_len)
    last_dir = preferred_dir

    while len(branch) < target_len:
        options: list[tuple[tuple[int, int], int]] = []
        for dr, dc in DIRS4:
            nxt = (current[0] + dr, current[1] + dc)
            if not _valid_tree_add(nxt, occupied_with_branch):
                continue
            if _hole_sizes(occupied_with_branch | {nxt}) != [1]:
                continue
            score = 1
            if (dr, dc) == last_dir:
                score += 3
            if (dr, dc) == preferred_dir:
                score += 2
            if abs(nxt[0]) + abs(nxt[1]) > abs(current[0]) + abs(current[1]):
                score += 1
            if (preferred_dir[0] and dr == preferred_dir[0]) or (preferred_dir[1] and dc == preferred_dir[1]):
                score += 1
            options.append((nxt, score))
        if not options:
            break
        nxt = _weighted_choice(options)
        last_dir = (nxt[0] - current[0], nxt[1] - current[1])
        current = nxt
        branch.add(current)
        occupied_with_branch.add(current)

    if len(branch) < min_len:
        return None
    return branch


def _add_random_branches(
    occupied: set[tuple[int, int]],
    *,
    count_range: tuple[int, int],
    min_len: int,
    max_len: int,
    attempts_per_branch: int,
) -> set[tuple[int, int]] | None:
    shape = set(occupied)
    branch_count = random.randint(count_range[0], count_range[1])
    for _ in range(branch_count):
        branch = None
        for _ in range(attempts_per_branch):
            candidate = _grow_branch(shape, random.choice(DIRS4), min_len=min_len, max_len=max_len)
            if candidate is not None:
                branch = candidate
                break
        if branch is None:
            return None
        shape |= branch
    return shape


def _generate_shape_pair() -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    common = _add_random_branches(set(CORE_RING), count_range=(2, 4), min_len=2, max_len=5, attempts_per_branch=24)
    if common is None:
        return None

    shape_a = _add_random_branches(common, count_range=(1, 3), min_len=2, max_len=5, attempts_per_branch=24)
    shape_b = _add_random_branches(common, count_range=(1, 3), min_len=2, max_len=5, attempts_per_branch=24)
    if shape_a is None or shape_b is None:
        return None
    if shape_a == shape_b:
        return None
    if _hole_sizes(shape_a) != [1] or _hole_sizes(shape_b) != [1]:
        return None

    overlap = len(shape_a & shape_b)
    only_a = len(shape_a - shape_b)
    only_b = len(shape_b - shape_a)
    if min(overlap, only_a, only_b) < 3:
        return None
    if overlap < 8:
        return None
    if min(len(shape_a), len(shape_b)) < 16:
        return None

    bbox_a = _bbox_from_cells(shape_a)
    bbox_b = _bbox_from_cells(shape_b)
    if bbox_a == bbox_b:
        return None

    union_bbox = _bbox_from_cells(shape_a | shape_b)
    if max(union_bbox[2] - union_bbox[0] + 1, union_bbox[3] - union_bbox[1] + 1) > 24:
        return None

    expected = _render_difference(shape_a, 1, shape_b, 2, BACKGROUND)
    used_colors = set(map(int, np.unique(expected))) - {BACKGROUND}
    if used_colors != {1, 2, OVERLAP}:
        return None

    top_left_shift = (bbox_a[0] - bbox_b[0], bbox_a[1] - bbox_b[1])
    if top_left_shift != (0, 0):
        top_left_out = _render_difference(shape_a, 1, _shift_cells(shape_b, *top_left_shift), 2, BACKGROUND)
        if np.array_equal(top_left_out, expected):
            return None

    center2_shift = (
        (bbox_a[0] + bbox_a[2]) - (bbox_b[0] + bbox_b[2]),
        (bbox_a[1] + bbox_a[3]) - (bbox_b[1] + bbox_b[3]),
    )
    if center2_shift == (0, 0):
        return None
    if center2_shift[0] % 2 == 0 and center2_shift[1] % 2 == 0:
        center_shift = (center2_shift[0] // 2, center2_shift[1] // 2)
        center_out = _render_difference(shape_a, 1, _shift_cells(shape_b, *center_shift), 2, BACKGROUND)
        if np.array_equal(center_out, expected):
            return None

    return shape_a, shape_b


def _grow_distractor() -> set[tuple[int, int]] | None:
    shape = {(0, 0)}
    target_area = random.randint(3, 9)
    while len(shape) < target_area:
        frontier: list[tuple[tuple[int, int], int]] = []
        for r, c in list(shape):
            for dr, dc in DIRS4:
                cell = (r + dr, c + dc)
                if cell in shape or abs(cell[0]) > 4 or abs(cell[1]) > 4:
                    continue
                candidate = set(shape) | {cell}
                if _hole_sizes(candidate):
                    continue
                frontier.append((cell, 1 + abs(cell[0]) + abs(cell[1])))
        if not frontier:
            return None
        shape.add(_weighted_choice(frontier))
    return shape


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _anchor_positions(
    shape: set[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    r0, c0, r1, c1 = _bbox_from_cells(shape)
    positions: list[tuple[int, int]] = []
    for anchor_r in range(-r0, h - r1):
        for anchor_c in range(-c0, w - c1):
            if blocked[anchor_r, anchor_c]:
                continue
            cells = {(anchor_r + r, anchor_c + c) for r, c in shape}
            if all(not blocked[r, c] for r, c in cells):
                positions.append((anchor_r, anchor_c))
    return positions


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: set[tuple[int, int]],
    color: int,
) -> tuple[int, int] | None:
    positions = _anchor_positions(shape, blocked)
    if not positions:
        return None
    anchor_r, anchor_c = random.choice(positions)
    cells = {(anchor_r + r, anchor_c + c) for r, c in shape}
    for r, c in cells:
        grid[r, c] = int(color)
    _mark_blocked(blocked, cells)
    return anchor_r, anchor_c


def _build_expected_from_detected(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return np.array_equal(out, expected)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair = _generate_shape_pair()
    if pair is None:
        return None
    shape_a, shape_b = pair
    colors = random.sample(PALETTE, 2)
    color_a, color_b = colors

    expected = _render_difference(shape_a, color_a, shape_b, color_b, BACKGROUND)
    used_colors = set(map(int, np.unique(expected))) - {BACKGROUND}
    if used_colors != {color_a, color_b, OVERLAP}:
        return None

    distractor_count = random.randint(0, 3)
    distractors: list[set[tuple[int, int]]] = []
    for _ in range(distractor_count):
        distractor = _grow_distractor()
        if distractor is None:
            return None
        distractors.append(distractor)

    union_bbox = _bbox_from_cells(shape_a | shape_b)
    min_h = max(18, union_bbox[2] - union_bbox[0] + 1 + 6)
    min_w = max(18, union_bbox[3] - union_bbox[1] + 1 + 6)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = [(shape_a, color_a), (shape_b, color_b)]
    random.shuffle(order)
    anchors: list[tuple[int, int]] = []
    for shape, color in order:
        anchor = _place_shape(grid, blocked, shape, color)
        if anchor is None:
            return None
        anchors.append(anchor)
    if anchors[0] == anchors[1]:
        return None

    remaining_colors = [color for color in PALETTE if color not in {color_a, color_b}]
    random.shuffle(remaining_colors)
    for distractor, color in zip(distractors, remaining_colors):
        if _place_shape(grid, blocked, distractor, color) is None:
            return None

    hole_shapes = _detect_hole_shapes(grid, BACKGROUND)
    if len(hole_shapes) != 2:
        return None
    if not _build_expected_from_detected(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid hole-anchor registration sample")
