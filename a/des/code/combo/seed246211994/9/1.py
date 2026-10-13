import numpy as np


COLOR_CYCLE = (2, 8, 5)  # red, teal, grey
GEN_MAX_ATTEMPTS = 400
GEN_MARGIN_MAX = 5


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _sorted_foreground(coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)

    points = [tuple(map(int, rc)) for rc in coords]
    if bbox_h >= bbox_w:
        return sorted(points, key=lambda rc: (rc[0], rc[1]))
    return sorted(points, key=lambda rc: (rc[1], rc[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor isolated foreground pixels by scan order with a fixed 3-color cycle."""
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    coords = np.argwhere(g != bg)
    assert len(coords) > 0, "input must contain foreground pixels"

    ordered = _sorted_foreground(coords)
    out = g.copy()
    for idx, (r, c) in enumerate(ordered):
        out[r, c] = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
    return out


def _choose_lattice_shape(rng: np.random.Generator, row_major: bool) -> tuple[int, int]:
    if row_major:
        rows = int(rng.integers(4, 9))
        cols = int(rng.integers(2, min(6, rows)))
        while rows <= cols:
            rows = int(rng.integers(4, 9))
            cols = int(rng.integers(2, min(6, rows)))
        return rows, cols

    cols = int(rng.integers(4, 9))
    rows = int(rng.integers(2, min(6, cols)))
    while cols <= rows:
        cols = int(rng.integers(4, 9))
        rows = int(rng.integers(2, min(6, cols)))
    return rows, cols


def _sample_slots(
    rng: np.random.Generator, rows: int, cols: int, count: int
) -> list[tuple[int, int]]:
    all_slots = [(r, c) for r in range(rows) for c in range(cols)]
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    chosen = {
        (0, 0),
        (rows - 1, cols - 1),
        corners[int(rng.integers(1, 3))],  # top-right or bottom-left
    }

    non_corner_slots = [slot for slot in all_slots if slot not in corners]
    if non_corner_slots:
        chosen.add(non_corner_slots[int(rng.integers(len(non_corner_slots)))])

    remaining = [slot for slot in all_slots if slot not in chosen]
    rng.shuffle(remaining)
    for slot in remaining:
        if len(chosen) >= count:
            break
        chosen.add(slot)

    assert len(chosen) == count
    return list(chosen)


def _isolated(coords: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(coords):
        for r2, c2 in coords[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_major = bool(rng.integers(2) == 0)
    rows, cols = _choose_lattice_shape(rng, row_major=row_major)
    bbox_h = 2 * rows - 1
    bbox_w = 2 * cols - 1

    min_points = 5
    max_points = min(rows * cols, 14)
    count = int(rng.integers(min_points, max_points + 1))

    bg_choices = [0, 1, 3, 4, 6, 7, 9]
    bg = int(bg_choices[int(rng.integers(len(bg_choices)))])

    available_colors = [c for c in range(10) if c not in set(COLOR_CYCLE) | {bg}]
    palette_size = int(rng.integers(1, min(5, len(available_colors)) + 1))
    palette = rng.choice(available_colors, size=palette_size, replace=False).tolist()

    top = int(rng.integers(0, GEN_MARGIN_MAX))
    left = int(rng.integers(0, GEN_MARGIN_MAX))
    bottom = int(rng.integers(0, GEN_MARGIN_MAX))
    right = int(rng.integers(0, GEN_MARGIN_MAX))

    grid = np.full((top + bbox_h + bottom, left + bbox_w + right), bg, dtype=int)

    slots = _sample_slots(rng, rows, cols, count)
    coords: list[tuple[int, int]] = []
    for slot_r, slot_c in slots:
        r = top + 2 * slot_r
        c = left + 2 * slot_c
        coords.append((r, c))
        grid[r, c] = int(palette[int(rng.integers(len(palette)))])

    coords_arr = np.array(coords, dtype=int)
    r0, c0 = coords_arr.min(axis=0)
    r1, c1 = coords_arr.max(axis=0)

    if not _isolated(coords):
        return None
    if int(r1 - r0 + 1) != bbox_h or int(c1 - c0 + 1) != bbox_w:
        return None
    if row_major and not (bbox_h > bbox_w):
        return None
    if (not row_major) and not (bbox_w > bbox_h):
        return None
    if len(coords) < len(COLOR_CYCLE):
        return None
    if int(np.sum(grid == bg)) <= len(coords):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
