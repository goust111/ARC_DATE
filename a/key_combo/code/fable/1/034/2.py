import math
import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAGMENT_COLORS = [3, 4, 5, 6, 7, 8, 9]

FRAME_COUNT_RANGE = (3, 5)
FRAME_SIZE_CHOICES = [5, 7]
GRID_MIN = 20
GRID_MAX = 30
FRAME_GAP = 1
GEN_ATTEMPTS = 400
ANGLE_PERTURB_RANGE = (-0.28, 0.28)
PATH_LEN_RANGE = (2, 28)
MAX_TWO_TURN_TRIES = 24
PROTOTYPE_FILL_RANGE = (0.42, 0.72)
FRAGMENT_KEEP_RANGE = (0.38, 0.72)

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    components = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], list[np.ndarray]]:
    red_components = components_mask(grid == RED, connectivity=4)
    assert len(red_components) >= 3

    frames = []
    for coords in red_components:
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h >= 5

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(border.sum()) == int(coords.shape[0])
        assert np.all(patch[border] == RED)
        assert np.all(patch[~border] != RED)

        interior = patch[1:-1, 1:-1]
        interior_colors = set(map(int, np.unique(interior))) - {int(bg)}
        assert len(interior_colors) == 1
        color = interior_colors.pop()
        mask = interior == color
        assert mask.any()

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": int(color),
                "mask": mask,
            }
        )

    mask_shapes = {frame["mask"].shape for frame in frames}
    assert len(mask_shapes) == 1
    return frames, red_components


def _blue_cycle_edges(grid: np.ndarray, red_components: list[np.ndarray]) -> set[tuple[int, int]]:
    red_id = {}
    for idx, coords in enumerate(red_components):
        for r, c in coords:
            red_id[(int(r), int(c))] = idx

    blue_components = components_mask(grid == BLUE, connectivity=4)
    edges = set()
    for coords in blue_components:
        touched = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS4:
                touched_id = red_id.get((rr + dr, cc + dc))
                if touched_id is not None:
                    touched.add(touched_id)
        assert len(touched) == 2
        a, b = sorted(touched)
        edges.add((a, b))

    n = len(red_components)
    assert len(edges) == n
    for idx in range(n):
        assert sum(idx in edge for edge in edges) == 2
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(frame["center"][0] for frame in frames) / len(frames)
    mean_c = sum(frame["center"][1] for frame in frames) / len(frames)
    angles = [
        math.atan2(frame["center"][0] - mean_r, frame["center"][1] - mean_c)
        for frame in frames
    ]
    assert len({round(angle, 10) for angle in angles}) == len(angles)

    order = sorted(range(len(frames)), key=lambda idx: angles[idx])
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the never-shown prototype by unioning aligned fragments, then fill each frame's missing pixels with the next clockwise fragment color.

    Input structure:
    - The grid contains several hollow red square frames connected into one ring by blue line components.
    - Each frame touches exactly two blue components and encloses one uniquely colored fragment on a same-sized interior canvas.
    - The full prototype is never drawn directly; it is the union of all frame fragments after aligning them by interior position.

    Rule:
    - Detect the red square frames, read each interior fragment mask, and union those masks position-wise to recover the hidden prototype.
    - Order the frames clockwise around the blue ring.
    - Inside every frame, keep the original fragment pixels unchanged and fill only the prototype pixels that are missing there.
    - Color each newly filled pixel with the fragment color from the next clockwise frame; leave the red frames, blue connectors, and background unchanged.

    Key insight:
    - The blue ring specifies only the source of the fill color, while the shape to complete must be reconstructed globally from the union of all partial fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    frames, red_components = _extract_frames(g, bg)
    edges = _blue_cycle_edges(g, red_components)
    order = _clockwise_order(frames, edges)

    prototype = np.zeros_like(frames[0]["mask"], dtype=bool)
    for frame in frames:
        prototype |= frame["mask"]

    out = g.copy()
    for pos, frame_id in enumerate(order):
        next_id = order[(pos + 1) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        fill_mask = prototype & ~frames[frame_id]["mask"]
        interior[fill_mask] = frames[next_id]["color"]
    return out


def _add_frontier(frontier: set[tuple[int, int]], mask: np.ndarray, cell: tuple[int, int]) -> None:
    n = mask.shape[0]
    for dr, dc in NEIGHBORS8:
        nr, nc = cell[0] + dr, cell[1] + dc
        if 0 <= nr < n and 0 <= nc < n and not mask[nr, nc]:
            frontier.add((nr, nc))


def _gen_prototype(interior_size: int, n_frames: int) -> np.ndarray | None:
    total = interior_size * interior_size
    min_area = max(n_frames + 2, int(math.ceil(PROTOTYPE_FILL_RANGE[0] * total)))
    max_area = min(total - 1, int(math.floor(PROTOTYPE_FILL_RANGE[1] * total)))
    if min_area > max_area:
        return None

    target = random.randint(min_area, max_area)
    mask = np.zeros((interior_size, interior_size), dtype=bool)
    start = (random.randrange(interior_size), random.randrange(interior_size))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    _add_frontier(frontier, mask, start)

    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        _add_frontier(frontier, mask, cell)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    if r1 == r0 or c1 == c0:
        return None
    return mask


def _build_fragments(prototype: np.ndarray, n_frames: int) -> list[np.ndarray] | None:
    cells = [tuple(coord) for coord in np.argwhere(prototype)]
    if len(cells) < n_frames + 2:
        return None

    unique_cells = random.sample(cells, n_frames)
    fragments = [np.zeros_like(prototype, dtype=bool) for _ in range(n_frames)]
    for frame_id, cell in enumerate(unique_cells):
        fragments[frame_id][cell] = True

    keep_lo, keep_hi = FRAGMENT_KEEP_RANGE
    cell_set = set(unique_cells)
    for cell in cells:
        if cell in cell_set:
            continue
        while True:
            keep_prob = random.uniform(keep_lo, keep_hi)
            chosen = [idx for idx in range(n_frames) if random.random() < keep_prob]
            if 0 < len(chosen) < n_frames:
                break
        for idx in chosen:
            fragments[idx][cell] = True

    area = int(prototype.sum())
    min_count = max(2, int(math.ceil(area * 0.30)))
    max_count = max(min_count, int(math.floor(area * 0.82)))
    union = np.zeros_like(prototype, dtype=bool)
    for fragment in fragments:
        count = int(fragment.sum())
        if count < min_count or count > max_count:
            return None
        if np.array_equal(fragment, prototype):
            return None
        union |= fragment
    if not np.array_equal(union, prototype):
        return None
    return fragments


def _frame_bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = a
    r0b, c0b, r1b, c1b = b
    return not (
        r1a + FRAME_GAP < r0b
        or r1b + FRAME_GAP < r0a
        or c1a + FRAME_GAP < c0b
        or c1b + FRAME_GAP < c0a
    )


def _choose_grid_size(frame_size: int, n_frames: int) -> tuple[int, int]:
    low = max(GRID_MIN, frame_size + 10 + 2 * n_frames)
    high = GRID_MAX
    if low > high:
        low = high
    return random.randint(low, high), random.randint(low, high)


def _place_frame_bboxes(n_frames: int, frame_size: int, grid_h: int, grid_w: int) -> list[tuple[int, int, int, int]] | None:
    center_r = grid_h / 2.0
    center_c = grid_w / 2.0
    base_angles = [2.0 * math.pi * idx / n_frames for idx in range(n_frames)]
    angles = sorted(angle + random.uniform(*ANGLE_PERTURB_RANGE) for angle in base_angles)

    min_radius = frame_size + 1.0
    max_radius = min(grid_h, grid_w) / 2.0 - frame_size / 2.0 - 1.0
    if max_radius < min_radius:
        return None

    bboxes = []
    for angle in angles:
        radius = random.uniform(min_radius, max_radius)
        top = int(round(center_r + radius * math.sin(angle) - frame_size / 2.0))
        left = int(round(center_c + radius * math.cos(angle) - frame_size / 2.0))
        top = max(0, min(grid_h - frame_size, top))
        left = max(0, min(grid_w - frame_size, left))
        bbox = (top, left, top + frame_size - 1, left + frame_size - 1)
        bboxes.append(bbox)

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if _frame_bboxes_overlap(bboxes[i], bboxes[j]):
                return None
    return bboxes


def _gen_clockwise_order(frame_bboxes: list[tuple[int, int, int, int]]) -> tuple[list[int], list[tuple[float, float]]] | tuple[None, None]:
    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in frame_bboxes]
    mean_r = sum(r for r, _ in centers) / len(centers)
    mean_c = sum(c for _, c in centers) / len(centers)
    angles = [math.atan2(r - mean_r, c - mean_c) for r, c in centers]
    if len({round(angle, 10) for angle in angles}) != len(angles):
        return None, None
    order = [idx for idx, _ in sorted(enumerate(angles), key=lambda item: item[1])]
    return order, centers


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], occupied: set[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    for c in range(c0, c1 + 1):
        occupied.add((r0, c))
        occupied.add((r1, c))
    for r in range(r0, r1 + 1):
        occupied.add((r, c0))
        occupied.add((r, c1))


def _place_fragment(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    fragment: np.ndarray,
    color: int,
    occupied: set[tuple[int, int]],
) -> None:
    r0, c0, _, _ = bbox
    rows, cols = np.nonzero(fragment)
    grid[r0 + 1 + rows, c0 + 1 + cols] = int(color)
    for rr, cc in zip(rows, cols):
        occupied.add((r0 + 1 + int(rr), c0 + 1 + int(cc)))


def _axis_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    step = 1 if er >= sr else -1
    return [(r, sc) for r in range(sr, er + step, step)]


def _one_turn_path(start: tuple[int, int], corner: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    return _axis_path(start, corner) + _axis_path(corner, end)[1:]


def _two_turn_vhv_path(start: tuple[int, int], mid_r: int, end: tuple[int, int]) -> list[tuple[int, int]]:
    _, sc = start
    _, ec = end
    corner1 = (mid_r, sc)
    corner2 = (mid_r, ec)
    return _axis_path(start, corner1) + _axis_path(corner1, corner2)[1:] + _axis_path(corner2, end)[1:]


def _two_turn_hvh_path(start: tuple[int, int], mid_c: int, end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, _ = start
    er, _ = end
    corner1 = (sr, mid_c)
    corner2 = (er, mid_c)
    return _axis_path(start, corner1) + _axis_path(corner1, corner2)[1:] + _axis_path(corner2, end)[1:]


def _path_is_free(path: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return all(cell not in occupied for cell in path)


def _find_path(start: tuple[int, int], end: tuple[int, int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    sr, sc = start
    er, ec = end

    if sr == er or sc == ec:
        direct = _axis_path(start, end)
        if _path_is_free(direct, occupied):
            return direct

    corners = [(sr, ec), (er, sc)]
    random.shuffle(corners)
    for corner in corners:
        path = _one_turn_path(start, corner, end)
        if _path_is_free(path, occupied):
            return path

    for _ in range(MAX_TWO_TURN_TRIES):
        if sr != er and sc != ec:
            mid_r = random.randint(min(sr, er), max(sr, er))
            path = _two_turn_vhv_path(start, mid_r, end)
            if _path_is_free(path, occupied):
                return path

            mid_c = random.randint(min(sc, ec), max(sc, ec))
            path = _two_turn_hvh_path(start, mid_c, end)
            if _path_is_free(path, occupied):
                return path

    return None


def _best_side(frame_bbox: tuple[int, int, int, int], target_r: float, target_c: float) -> str:
    r0, c0, r1, c1 = frame_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = target_r - center_r
    dc = target_c - center_c
    if abs(dr) > abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _pick_attachment(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    occupied: set[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    r0, c0, r1, c1 = frame_bbox
    if side in {"top", "bottom"}:
        values = list(range(c0 + 1, c1))
        random.shuffle(values)
        for c in values:
            border = (r0, c) if side == "top" else (r1, c)
            outside = (r0 - 1, c) if side == "top" else (r1 + 1, c)
            if 0 <= outside[0] < grid_h and 0 <= outside[1] < grid_w and outside not in occupied:
                return border, outside
    else:
        values = list(range(r0 + 1, r1))
        random.shuffle(values)
        for r in values:
            border = (r, c0) if side == "left" else (r, c1)
            outside = (r, c0 - 1) if side == "left" else (r, c1 + 1)
            if 0 <= outside[0] < grid_h and 0 <= outside[1] < grid_w and outside not in occupied:
                return border, outside
    return None, None


def _draw_path(grid: np.ndarray, path: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r, c] = BLUE
        occupied.add((r, c))


def _connect_cycle(
    grid: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    centers: list[tuple[float, float]],
    order: list[int],
    occupied: set[tuple[int, int]],
) -> bool:
    grid_h, grid_w = grid.shape
    n = len(frame_bboxes)
    for pos in range(n):
        left_id = order[pos]
        right_id = order[(pos + 1) % n]

        center_a = centers[left_id]
        center_b = centers[right_id]
        side_a = _best_side(frame_bboxes[left_id], center_b[0], center_b[1])
        side_b = _best_side(frame_bboxes[right_id], center_a[0], center_a[1])

        _border_a, start_a = _pick_attachment(frame_bboxes[left_id], side_a, occupied, grid_h, grid_w)
        _border_b, start_b = _pick_attachment(frame_bboxes[right_id], side_b, occupied, grid_h, grid_w)
        if start_a is None or start_b is None:
            return False

        path = _find_path(start_a, start_b, occupied)
        if path is None or not (PATH_LEN_RANGE[0] <= len(path) <= PATH_LEN_RANGE[1]):
            return False
        if any(not (0 <= r < grid_h and 0 <= c < grid_w) for r, c in path):
            return False
        _draw_path(grid, path, occupied)
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    frame_size = random.choice(FRAME_SIZE_CHOICES)
    interior_size = frame_size - 2
    prototype = _gen_prototype(interior_size, n_frames)
    if prototype is None:
        return None

    fragments = _build_fragments(prototype, n_frames)
    if fragments is None:
        return None

    grid_h, grid_w = _choose_grid_size(frame_size, n_frames)
    frame_bboxes = _place_frame_bboxes(n_frames, frame_size, grid_h, grid_w)
    if frame_bboxes is None:
        return None

    order, centers = _gen_clockwise_order(frame_bboxes)
    if order is None:
        return None

    colors = random.sample(FRAGMENT_COLORS, n_frames)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for bbox, fragment, color in zip(frame_bboxes, fragments, colors):
        _draw_frame(grid, bbox, occupied)
        _place_fragment(grid, bbox, fragment, color, occupied)

    if not _connect_cycle(grid, frame_bboxes, centers, order, occupied):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    frames, _ = _extract_frames(grid, BG)
    prototype_mask = np.zeros_like(frames[0]["mask"], dtype=bool)
    for frame in frames:
        prototype_mask |= frame["mask"]
    for frame in frames:
        if not np.any(prototype_mask & ~frame["mask"]):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
