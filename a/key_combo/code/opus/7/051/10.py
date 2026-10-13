import itertools
import random

import numpy as np


BG = 0
FILLER = 6
FRAME_COLORS = (1, 2, 3, 4, 8)
ORTHO = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
COUNT_PERMS = [list(p) for p in itertools.permutations((1, 2, 3))]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _interior_contains(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _frame_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _unique_topo_order(nodes: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for low, high in edges:
        if low == high:
            raise AssertionError("self edge is invalid")
        if high not in above[low]:
            above[low].add(high)
            indegree[high] += 1

    order: list[int] = []
    used: set[int] = set()
    while len(order) < len(nodes):
        avail = [node for node in nodes if node not in used and indegree[node] == 0]
        assert len(avail) == 1, "frame overlaps must define a unique layer order"
        cur = avail[0]
        used.add(cur)
        order.append(cur)
        for nxt in above[cur]:
            indegree[nxt] -= 1
    return order


def _owner_map(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    owners = np.full((h, w), -1, dtype=int)
    ranked = [(_frame_area(box), idx, box) for idx, box in enumerate(boxes)]
    for r in range(h):
        for c in range(w):
            candidates = [(area, idx) for area, idx, box in ranked if _interior_contains(box, r, c)]
            if candidates:
                owners[r, c] = min(candidates)[1]
    return owners


def _is_cross_center(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    return all(int(grid[r + dr, c + dc]) == BG for dr, dc in ORTHO)


def _count_crosses_by_frame(
    grid: np.ndarray,
    owners: np.ndarray,
    frame_count: int,
) -> tuple[list[int], set[tuple[int, int]]]:
    counts = [0] * frame_count
    centers: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _is_cross_center(grid, r, c):
                continue
            owner = int(owners[r, c])
            if owner < 0:
                continue
            cells = [(r + dr, c + dc) for dr, dc in ORTHO]
            if all(int(owners[rr, cc]) == owner for rr, cc in cells):
                counts[owner] += 1
                centers.add((r, c))
    return counts, centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the frame stacking order and repeat each frame color by its owned black-cross count.

    Input structure:
    - The background is black, and each frame color forms one visible fragment of a one-cell-thick hollow rectangle.
    - Multiple colored frames overlap, so some border cells are hidden by higher layers.
    - Inside the frames, black and magenta pixels occupy the interiors; only some black pixels form full 5-cell orthogonal crosses.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels.
    - Use border overlap cells to infer the unique bottom-to-top frame order.
    - Assign every interior cell to the smallest-area frame whose interior contains it, count full black crosses owned by each frame, and output a single column that lists the bottom-to-top frame colors repeated by those counts.

    Key insight:
    - The colored borders reveal the hidden frame geometry and the occlusion order, while the interior black-cross count must be computed on ownership regions defined by the smallest enclosing frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BG, FILLER))
    assert frame_colors, "expected visible colored frames"

    boxes: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []
    for color in frame_colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None, "each frame color must be visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible frame pixels must lie on the reconstructed border"
        boxes.append(box)
        masks.append(mask)

    color_to_index = {color: idx for idx, color in enumerate(frame_colors)}
    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [idx for idx, mask in enumerate(masks) if bool(mask[r, c])]
            if len(covering) < 2:
                continue
            top_color = int(g[r, c])
            assert top_color in color_to_index, "overlap cells must show a frame color"
            top_idx = color_to_index[top_color]
            assert top_idx in covering, "visible overlap color must belong to a covering frame"
            for idx in covering:
                if idx != top_idx:
                    edges.add((idx, top_idx))

    order_idx = _unique_topo_order(list(range(len(frame_colors))), edges)
    owners = _owner_map(g.shape, boxes)
    counts, _ = _count_crosses_by_frame(g, owners, len(frame_colors))
    assert all(count > 0 for count in counts), "each frame must own at least one cross"

    out = [frame_colors[idx] for idx in order_idx for _ in range(counts[idx])]
    return np.array(out, dtype=int).reshape(-1, 1)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _place_frame_stack(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    size_min = 8
    size_h_max = min(14, h - 2)
    size_w_max = min(14, w - 2)
    if size_h_max < size_min or size_w_max < size_min:
        return None

    def random_box() -> tuple[int, int, int, int]:
        bh = random.randint(size_min, size_h_max)
        bw = random.randint(size_min, size_w_max)
        r0 = random.randint(0, h - bh)
        c0 = random.randint(0, w - bw)
        return (r0, c0, r0 + bh - 1, c0 + bw - 1)

    for _ in range(200):
        boxes = [random_box()]
        borders = [_border_cells(boxes[0])]
        while len(boxes) < 3:
            placed = None
            for _ in range(300):
                cand = random_box()
                if _frame_area(cand) in {_frame_area(box) for box in boxes}:
                    continue
                cand_border = _border_cells(cand)
                if len(cand_border & borders[-1]) not in (2, 4):
                    continue
                if any(len(cand_border & border) not in (0, 2, 4) for border in borders[:-1]):
                    continue
                placed = (cand, cand_border)
                break
            if placed is None:
                break
            box, border = placed
            boxes.append(box)
            borders.append(border)
        if len(boxes) == 3:
            return boxes
    return None


def _render_frames(
    shape: tuple[int, int],
    order_colors: list[int],
    order_boxes: list[tuple[int, int, int, int]],
    interior: np.ndarray,
) -> np.ndarray:
    grid = interior.copy()
    for color, box in zip(order_colors, order_boxes):
        mask = _frame_mask(shape, box)
        grid[mask] = int(color)
    return grid


def _visible_boxes_match(grid: np.ndarray, colors: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    for color, box in zip(colors, boxes):
        if _bbox(grid == color) != box:
            return False
    return True


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in ORTHO]


def _blocked_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    for box in boxes:
        blocked |= _frame_mask(shape, box)
    return blocked


def _candidate_centers(
    owners: np.ndarray,
    blocked: np.ndarray,
    frame_idx: int,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = owners.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cells = _plus_cells((r, c))
            if all(
                int(owners[rr, cc]) == frame_idx
                and not bool(occupied[rr, cc])
                and not bool(blocked[rr, cc])
                for rr, cc in cells
            ):
                out.append((r, c))
    random.shuffle(out)
    return out


def _empty_capacity(owners: np.ndarray, blocked: np.ndarray, frame_idx: int) -> int:
    return len(_candidate_centers(owners, blocked, frame_idx, np.zeros(owners.shape, dtype=bool)))


def _place_crosses(
    base: np.ndarray,
    owners: np.ndarray,
    blocked: np.ndarray,
    counts_by_frame: list[int],
) -> set[tuple[int, int]] | None:
    occupied = np.zeros(base.shape, dtype=bool)
    chosen_centers: set[tuple[int, int]] = set()
    frame_order = sorted(range(len(counts_by_frame)), key=lambda idx: counts_by_frame[idx], reverse=True)
    for frame_idx in frame_order:
        need = counts_by_frame[frame_idx]
        placed = 0
        for center in _candidate_centers(owners, blocked, frame_idx, occupied):
            cells = _plus_cells(center)
            if any(bool(occupied[rr, cc]) for rr, cc in cells):
                continue
            for rr, cc in cells:
                base[rr, cc] = BG
                occupied[rr, cc] = True
            chosen_centers.add(center)
            placed += 1
            if placed == need:
                break
        if placed != need:
            return None
    return chosen_centers


def _would_create_unplanned_cross(
    grid: np.ndarray,
    owners: np.ndarray,
    protected: set[tuple[int, int]],
    cell: tuple[int, int],
) -> bool:
    r, c = cell
    prev = int(grid[r, c])
    grid[r, c] = BG
    try:
        candidates = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        for cr, cc in candidates:
            if not _is_cross_center(grid, cr, cc):
                continue
            owner = int(owners[cr, cc])
            if owner < 0:
                continue
            if not all(int(owners[rr, cc2]) == owner for rr, cc2 in _plus_cells((cr, cc))):
                continue
            if (cr, cc) not in protected:
                return True
        return False
    finally:
        grid[r, c] = prev


def _add_noise(
    grid: np.ndarray,
    owners: np.ndarray,
    blocked: np.ndarray,
    protected: set[tuple[int, int]],
) -> None:
    for frame_idx in range(len(np.unique(owners[owners >= 0]))):
        owned = [
            (int(r), int(c))
            for r, c in np.argwhere(owners == frame_idx)
            if int(grid[r, c]) == FILLER and not bool(blocked[r, c])
        ]
        random.shuffle(owned)
        budget = random.randint(1, 4)
        placed = 0
        for cell in owned:
            if _would_create_unplanned_cross(grid, owners, protected, cell):
                continue
            grid[cell] = BG
            placed += 1
            if placed == budget:
                break


def _build_interior(
    shape: tuple[int, int],
    owners: np.ndarray,
    blocked: np.ndarray,
    counts_by_frame: list[int],
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    interior = np.full(shape, BG, dtype=int)
    interior[(owners >= 0) & (~blocked)] = FILLER
    protected = _place_crosses(interior, owners, blocked, counts_by_frame)
    if protected is None:
        return None
    _add_noise(interior, owners, blocked, protected)
    observed_counts, observed_centers = _count_crosses_by_frame(interior, owners, len(counts_by_frame))
    if observed_counts != counts_by_frame or observed_centers != protected:
        return None
    return interior, protected


def _expected_output(order_colors: list[int], color_to_count: dict[int, int]) -> np.ndarray:
    data = [color for color in order_colors for _ in range(color_to_count[color])]
    return np.array(data, dtype=int).reshape(-1, 1)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 22)
    w = random.randint(18, 22)
    boxes_in_order = _place_frame_stack(h, w)
    if boxes_in_order is None:
        return None

    order_colors = random.sample(list(FRAME_COLORS), 3)
    owners = _owner_map((h, w), boxes_in_order)
    blocked = _blocked_mask((h, w), boxes_in_order)
    capacities = [_empty_capacity(owners, blocked, idx) for idx in range(3)]
    feasible_counts = [perm for perm in COUNT_PERMS if all(capacities[idx] >= perm[idx] for idx in range(3))]
    random.shuffle(feasible_counts)

    interior = None
    counts_by_frame = None
    for candidate_counts in feasible_counts:
        interior_result = _build_interior((h, w), owners, blocked, candidate_counts)
        if interior_result is not None:
            interior, _ = interior_result
            counts_by_frame = candidate_counts
            break
    if interior is None or counts_by_frame is None:
        return None

    grid = _render_frames((h, w), order_colors, boxes_in_order, interior)
    if not _visible_boxes_match(grid, order_colors, boxes_in_order):
        return None

    expected = _expected_output(order_colors, {color: count for color, count in zip(order_colors, counts_by_frame)})
    try:
        observed = main(grid)
    except Exception:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    # The written spec conflicts on frame count versus unique cross counts, so the
    # generator uses exactly three frames while keeping the solver general.
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 10")
