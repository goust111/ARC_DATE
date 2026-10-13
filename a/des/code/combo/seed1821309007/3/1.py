import numpy as np
from collections import Counter, deque


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
BG = 0
MAX_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _read_legend(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    for c in range(w):
        non_bg = grid[:, c] != bg
        if non_bg.sum() < 4:
            continue
        runs: list[tuple[int, int]] = []
        r = 0
        while r < h:
            if not non_bg[r]:
                r += 1
                continue
            start = r
            while r < h and non_bg[r]:
                r += 1
            runs.append((start, r - 1))
        if len(runs) < 2 or any(end - start + 1 != 2 for start, end in runs):
            continue
        legend = [(int(grid[start, c]), int(grid[start + 1, c])) for start, _ in runs]
        legend_mask = np.zeros_like(grid, dtype=bool)
        for start, _ in runs:
            legend_mask[start : start + 2, c] = True
        return legend, legend_mask
    raise AssertionError("legend not found")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    legend, legend_mask = _read_legend(g, bg)

    object_mask = (g != bg) & ~legend_mask
    object_labels = np.zeros((h, w), dtype=int)
    objects = _components(object_mask)
    for idx, comp in enumerate(objects, start=1):
        for r, c in comp:
            object_labels[r, c] = idx

    holes_by_object: dict[int, list[list[tuple[int, int]]]] = {}
    for comp in _components(g == bg):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        neighbor_ids = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    label = int(object_labels[nr, nc])
                    if label > 0:
                        neighbor_ids.add(label)
        if len(neighbor_ids) == 1:
            owner = next(iter(neighbor_ids))
            holes_by_object.setdefault(owner, []).append(comp)

    out = g.copy()
    for owner, holes in holes_by_object.items():
        hole_count = len(holes)
        assert 1 <= hole_count <= len(legend), "hole count outside legend range"
        fill_color = legend[hole_count - 1][1]
        for hole in holes:
            for r, c in hole:
                out[r, c] = fill_color
    return out


def _solid_shape() -> np.ndarray:
    h = int(RNG.integers(2, 5))
    w = int(RNG.integers(2, 6))
    shape = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3 and RNG.random() < 0.5:
        side = int(RNG.integers(4))
        if side == 0:
            shape[0, int(RNG.integers(w))] = False
        elif side == 1:
            shape[h - 1, int(RNG.integers(w))] = False
        elif side == 2:
            shape[int(RNG.integers(h)), 0] = False
        else:
            shape[int(RNG.integers(h)), w - 1] = False
        if not np.any(shape[0]) and h > 1:
            shape[0, 0] = True
        if not np.any(shape[h - 1]) and h > 1:
            shape[h - 1, 0] = True
        if not np.any(shape[:, 0]) and w > 1:
            shape[0, 0] = True
        if not np.any(shape[:, w - 1]) and w > 1:
            shape[0, w - 1] = True
    return shape


def _holed_shape(hole_count: int) -> np.ndarray:
    inner_h = int(RNG.integers(2, 5))
    h = inner_h + 2
    w = 2 * hole_count + 1
    shape = np.zeros((h, w), dtype=bool)
    shape[0, :] = True
    shape[-1, :] = True
    shape[:, 0] = True
    shape[:, -1] = True
    for wall_c in range(2, w - 1, 2):
        shape[1:-1, wall_c] = True
    if RNG.random() < 0.6:
        if RNG.random() < 0.5:
            side_r = int(RNG.integers(h))
            shape[side_r, 0] = True
        else:
            side_r = int(RNG.integers(h))
            shape[side_r, -1] = True
    return shape


def _shape_for_holes(hole_count: int) -> np.ndarray:
    return _solid_shape() if hole_count == 0 else _holed_shape(hole_count)


def _can_place(reserved: np.ndarray, top: int, left: int, shape: np.ndarray) -> bool:
    h, w = reserved.shape
    sh, sw = shape.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + sh + 1)
    c1 = min(w, left + sw + 1)
    return not reserved[r0:r1, c0:c1].any()


def _mark_reserved(reserved: np.ndarray, top: int, left: int, shape: np.ndarray) -> None:
    h, w = reserved.shape
    sh, sw = shape.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + sh + 1)
    c1 = min(w, left + sw + 1)
    reserved[r0:r1, c0:c1] = True


def _sample_hole_counts(num_pairs: int, object_count: int) -> list[int]:
    counts = [0, num_pairs]
    if object_count >= 3 and num_pairs >= 2:
        counts.append(int(RNG.integers(1, num_pairs)))
    while len(counts) < object_count:
        counts.append(int(RNG.integers(0, num_pairs + 1)))
    RNG.shuffle(counts)
    return counts[:object_count]


def _gen_attempt() -> np.ndarray | None:
    num_pairs = int(RNG.integers(2, 4))
    h = int(RNG.integers(max(12, 3 * num_pairs + 5), 23))
    w = int(RNG.integers(18, 31))
    grid = np.full((h, w), BG, dtype=int)

    legend_height = 3 * num_pairs - 1
    top0 = int(RNG.integers(0, h - legend_height + 1))

    all_colors = list(range(1, 10))
    RNG.shuffle(all_colors)
    value_colors = all_colors[:num_pairs]
    key_pool = [c for c in all_colors[num_pairs:] if c not in value_colors]
    if len(key_pool) < num_pairs:
        key_pool = [c for c in range(1, 10) if c not in value_colors]
    RNG.shuffle(key_pool)
    key_colors = key_pool[:num_pairs]

    for idx in range(num_pairs):
        r = top0 + idx * 3
        grid[r, 0] = key_colors[idx]
        grid[r + 1, 0] = value_colors[idx]

    object_count = int(RNG.integers(max(3, num_pairs + 1), 6))
    hole_counts = _sample_hole_counts(num_pairs, object_count)
    object_colors = [c for c in range(1, 10) if c not in value_colors]
    RNG.shuffle(object_colors)
    if len(object_colors) < object_count:
        return None

    shapes = [_shape_for_holes(k) for k in hole_counts]
    if max(shape.shape[1] for shape in shapes) > w - 4:
        return None

    reserved = np.zeros((h, w), dtype=bool)
    reserved[:, 0] = True
    reserved[:, 1] = True
    reserved[:top0, 0] = True
    reserved[top0 + legend_height :, 0] = True
    reserved[max(0, top0 - 1) : min(h, top0 + legend_height + 1), :2] = True

    order = list(range(object_count))
    RNG.shuffle(order)
    for idx in order:
        shape = shapes[idx]
        sh, sw = shape.shape
        placed = False
        for _ in range(80):
            top = int(RNG.integers(0, h - sh + 1))
            left = int(RNG.integers(3, w - sw + 1))
            if not _can_place(reserved, top, left, shape):
                continue
            color = object_colors[idx]
            patch = grid[top : top + sh, left : left + sw]
            patch[shape] = color
            _mark_reserved(reserved, top, left, shape)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
