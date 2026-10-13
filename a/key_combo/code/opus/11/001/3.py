from __future__ import annotations

import random
from itertools import combinations, permutations

import numpy as np


FRAME_COLORS = (1, 2, 3, 4)
MARKER_COLOR = 8
COUNT_TO_FILL = {0: 0, 1: 7, 2: 6, 3: 9}
MIN_GRID = 20
MAX_GRID = 24
MIN_SIDE = 8
MAX_SIDE = 12
MAX_ATTEMPTS = 600
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _interior_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_bboxes(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in FRAME_COLORS:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"missing frame color {color}"
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
        bboxes[color] = bbox
    return bboxes


def _crossing_edges(
    grid: np.ndarray, bboxes: dict[int, tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    frames = {color: _frame_mask(grid.shape, bbox) for color, bbox in bboxes.items()}
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(FRAME_COLORS, 2):
        overlap = np.argwhere(frames[a] & frames[b])
        assert overlap.size > 0, f"colors {a} and {b} do not cross"
        for r_raw, c_raw in overlap:
            r = int(r_raw)
            c = int(c_raw)
            visible = int(grid[r, c])
            assert visible in (a, b), "triple-covered crossings are not allowed"
            if visible == a:
                edges.add((b, a))
            else:
                edges.add((a, b))
    return edges


def _unique_hamiltonian_path(edges: set[tuple[int, int]]) -> list[int]:
    valid = []
    for order in permutations(FRAME_COLORS):
        if all((int(order[i]), int(order[i + 1])) in edges for i in range(len(order) - 1)):
            valid.append(order)
    assert len(valid) == 1, f"expected a unique Hamiltonian path, got {valid}"
    return [int(x) for x in valid[0]]


def _count_markers_in_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    mask = _interior_mask(grid.shape, bbox) & (grid == MARKER_COLOR)
    return len(_components4(mask))


def _render_output(
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    counts: dict[int, int],
) -> np.ndarray:
    size = max(max(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in bboxes.values())
    out = np.zeros((size, size), dtype=int)

    for layer, color in enumerate(order):
        fill = COUNT_TO_FILL[counts[color]]
        inner_lo = layer + 1
        inner_hi = size - layer - 1
        if fill != 0 and inner_lo < inner_hi:
            out[inner_lo:inner_hi, inner_lo:inner_hi] = fill

    for layer, color in enumerate(order):
        end = size - layer - 1
        out[layer : end + 1, layer] = color
        out[layer : end + 1, end] = color
        out[layer, layer : end + 1] = color
        out[end, layer : end + 1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder overlapping rectangular frames into concentric square rings, then fill each ring band by the original frame's teal-cluster count.

    Input structure:
    - The grid has black background, four one-pixel-thick rectangular frame colors {1,2,3,4}, and teal (8) marker clusters.
    - Each frame is only partially visible because frame perimeters cross and one color is visible at each crossing pixel.
    - Marker clusters lie strictly inside reconstructed frame interiors and each cluster belongs unambiguously to one frame.

    Rule:
    - Infer each full rectangular frame from the bounding box of the visible pixels of its color.
    - At every pairwise frame crossing, add a directed edge from the hidden frame to the visible frame; the unique Hamiltonian path through the four colors gives the outer-to-inner nesting order.
    - Count the 4-connected teal clusters inside each reconstructed frame interior.
    - Redraw the four colors as centered concentric square rings whose outer side length equals the largest original frame side length and whose side lengths decrease by 2 per layer.
    - Fill each band between adjacent rings, and the innermost interior, using the outer ring's cluster count: 0->0, 1->7, 2->6, 3->9.

    Key insight:
    - The same rectangles contribute two independent signals: visible crossings determine the nesting order, while interior teal-cluster counts determine the fill color of each nested band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bboxes = _infer_bboxes(g)
    edges = _crossing_edges(g, bboxes)
    order = _unique_hamiltonian_path(edges)
    counts = {color: _count_markers_in_bbox(g, bboxes[color]) for color in FRAME_COLORS}
    assert all(0 <= counts[color] <= 3 for color in FRAME_COLORS)
    return _render_output(order, bboxes, counts)


def _paint_frames(
    shape: tuple[int, int],
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in order:
        grid[_frame_mask(shape, bboxes[color])] = color
    return grid


def _output_signature(order: list[int], counts: dict[int, int], bboxes: dict[int, tuple[int, int, int, int]]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    size = max(max(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in bboxes.values())
    return tuple(order), tuple(counts[color] for color in FRAME_COLORS), int(size)


def _sample_sorted_distinct(lo: int, hi: int, k: int) -> list[int]:
    vals = random.sample(range(lo, hi + 1), k)
    vals.sort()
    return [int(v) for v in vals]


def _generate_structural_boxes(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(250):
        tops = _sample_sorted_distinct(1, h // 2 - 4, 4)
        lefts = _sample_sorted_distinct(1, w // 2 - 4, 4)

        bottoms: list[int] = []
        top_limit = tops[-1]
        ok = True
        prev_bottom = top_limit
        for i, top in enumerate(tops):
            lo = max(top + MIN_SIDE - 1, top_limit + 1, prev_bottom + 1)
            hi = min(top + MAX_SIDE - 1, h - 2)
            hi -= 3 - i
            if lo > hi:
                ok = False
                break
            bottom = random.randint(lo, hi)
            bottoms.append(int(bottom))
            prev_bottom = int(bottom)
        if not ok:
            continue

        rights: list[int] = []
        used_rights: set[int] = set()
        max_left = lefts[-1]
        for left in lefts:
            lo = max(left + MIN_SIDE - 1, max_left + 1, w // 2 + 1)
            hi = min(left + MAX_SIDE - 1, w - 2)
            candidates = [val for val in range(lo, hi + 1) if val not in used_rights and val not in lefts]
            if not candidates:
                ok = False
                break
            right = int(random.choice(candidates))
            rights.append(right)
            used_rights.add(right)
        if not ok:
            continue

        boxes = [(tops[i], lefts[i], bottoms[i], rights[i]) for i in range(4)]
        if any(max(r1 - r0 + 1, c1 - c0 + 1) < 9 for r0, c0, r1, c1 in boxes):
            continue
        return boxes
    return None


def _exclusive_interiors(
    shape: tuple[int, int], bboxes: dict[int, tuple[int, int, int, int]]
) -> dict[int, np.ndarray]:
    interiors = {color: _interior_mask(shape, bbox) for color, bbox in bboxes.items()}
    frame_union = np.zeros(shape, dtype=bool)
    for bbox in bboxes.values():
        frame_union |= _frame_mask(shape, bbox)
    exclusives: dict[int, np.ndarray] = {}
    for color in FRAME_COLORS:
        others = np.zeros(shape, dtype=bool)
        for other in FRAME_COLORS:
            if other != color:
                others |= interiors[other]
        exclusives[color] = interiors[color] & ~others & ~frame_union
    return exclusives


_CLUSTER_SHAPES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _can_stamp_cluster(
    exclusive: np.ndarray,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor_r: int,
    anchor_c: int,
) -> bool:
    h, w = exclusive.shape
    cells = []
    for dr, dc in shape:
        r = anchor_r + dr
        c = anchor_c + dc
        if not (0 <= r < h and 0 <= c < w and exclusive[r, c] and not occupied[r, c]):
            return False
        cells.append((r, c))
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _stamp_cluster(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor_r: int,
    anchor_c: int,
) -> None:
    for dr, dc in shape:
        r = anchor_r + dr
        c = anchor_c + dc
        grid[r, c] = MARKER_COLOR
        occupied[r, c] = True


def _place_clusters_for_color(
    grid: np.ndarray,
    exclusive: np.ndarray,
    count: int,
) -> bool:
    occupied = np.zeros_like(exclusive, dtype=bool)
    coords = [(int(r), int(c)) for r, c in np.argwhere(exclusive)]
    if count == 0:
        return True
    if len(coords) < count:
        return False

    for _ in range(120):
        probe = grid.copy()
        occupied.fill(False)
        ok = True
        for cluster_index in range(count):
            shapes = list(_CLUSTER_SHAPES)
            random.shuffle(shapes)
            if cluster_index == count - 1:
                shapes.append(((0, 0),))
            placed = False
            for shape in shapes:
                anchors = coords[:]
                random.shuffle(anchors)
                for r, c in anchors:
                    if _can_stamp_cluster(exclusive, occupied, shape, r, c):
                        _stamp_cluster(probe, occupied, shape, r, c)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if ok and len(_components4((probe == MARKER_COLOR) & exclusive)) == count:
            grid[:, :] = probe
            return True
    return False


def _build_expected(
    shape: tuple[int, int],
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    counts: dict[int, int],
) -> np.ndarray | None:
    grid = _paint_frames(shape, order, bboxes)
    exclusives = _exclusive_interiors(shape, bboxes)

    for color in FRAME_COLORS:
        if exclusives[color].sum() < counts[color]:
            return None
    for color in FRAME_COLORS:
        if not _place_clusters_for_color(grid, exclusives[color], counts[color]):
            return None
    return grid


def _validate_input(
    grid: np.ndarray,
    order: list[int],
    counts: dict[int, int],
    bboxes: dict[int, tuple[int, int, int, int]],
) -> bool:
    marker_mask = grid == MARKER_COLOR
    frame_union = np.zeros(grid.shape, dtype=bool)
    interiors = {color: _interior_mask(grid.shape, bboxes[color]) for color in FRAME_COLORS}
    for color in FRAME_COLORS:
        frame_union |= _frame_mask(grid.shape, bboxes[color])
    if np.any(marker_mask & frame_union):
        return False
    owner_count = np.zeros(grid.shape, dtype=int)
    for color in FRAME_COLORS:
        owner_count += interiors[color].astype(int)
    if np.any(marker_mask & (owner_count != 1)):
        return False

    inferred = _infer_bboxes(grid)
    if inferred != bboxes:
        return False

    edges = _crossing_edges(grid, inferred)
    inferred_order = _unique_hamiltonian_path(edges)
    if inferred_order != order:
        return False

    inferred_counts = {color: _count_markers_in_bbox(grid, inferred[color]) for color in FRAME_COLORS}
    if inferred_counts != counts:
        return False

    if sorted(inferred_counts.values()) != [0, 1, 2, 3]:
        return False

    output = main(grid)
    expected = _render_output(order, bboxes, counts)
    if not np.array_equal(output, expected):
        return False

    pair_masks = {color: _frame_mask(grid.shape, inferred[color]) for color in FRAME_COLORS}
    for a, b in combinations(FRAME_COLORS, 2):
        if not (pair_masks[a] & pair_masks[b]).any():
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    boxes = _generate_structural_boxes(h, w)
    if boxes is None:
        return None

    colors = list(FRAME_COLORS)
    order = random.sample(colors, len(colors))
    count_values = random.sample([0, 1, 2, 3], 4)
    counts = {color: int(count_values[i]) for i, color in enumerate(colors)}
    shuffled_boxes = boxes[:]
    random.shuffle(shuffled_boxes)
    bboxes = {color: shuffled_boxes[i] for i, color in enumerate(colors)}

    signature = _output_signature(order, counts, bboxes)
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None

    grid = _build_expected((h, w), order, bboxes, counts)
    if grid is None:
        return None

    if not _validate_input(grid, order, counts, bboxes):
        return None

    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
