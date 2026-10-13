import numpy as np


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
DIAG_DOWN = "diag_down"
DIAG_UP = "diag_up"

CANONICAL_DIRECTIONS = (HORIZONTAL, VERTICAL, DIAG_DOWN, DIAG_UP)
DIR_TO_MARKER = {
    HORIZONTAL: 2,  # red
    VERTICAL: 1,  # blue
    DIAG_DOWN: 3,  # green
    DIAG_UP: 4,  # yellow
}
MARKER_TO_DIR = {color: direction for direction, color in DIR_TO_MARKER.items()}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            patch[arr[:, 0] - r0, arr[:, 1] - c0] = color
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                }
            )

    return components


def _classify_direction(component: dict) -> str:
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if width > 2 * height:
        return HORIZONTAL
    if height > 2 * width:
        return VERTICAL

    rel = component["coords"] - np.array([r0, c0], dtype=int)
    down_score = float(np.var(rel[:, 0] - rel[:, 1]))
    up_score = float(np.var(rel[:, 0] + rel[:, 1]))
    if down_score < up_score:
        return DIAG_DOWN
    if up_score < down_score:
        return DIAG_UP

    centered = rel - rel.mean(axis=0, keepdims=True)
    covariance_sign = float(np.sum(centered[:, 0] * centered[:, 1]))
    return DIAG_DOWN if covariance_sign >= 0 else DIAG_UP


def _parse_indicator_order(grid: np.ndarray, bg: int) -> list[str]:
    row = grid[-1]
    directions = []
    c = 0
    while c < row.shape[0]:
        if int(row[c]) == bg:
            c += 1
            continue

        color = int(row[c])
        start = c
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
        run_len = c - start

        assert color in MARKER_TO_DIR, "unexpected indicator color"
        assert run_len == 2, "indicator blocks must be 2x1"
        directions.append(MARKER_TO_DIR[color])

    assert directions, "missing bottom indicator row"
    assert len(set(directions)) == len(directions), "duplicate indicator categories"
    return directions


def _render_output(bg: int, direction_order: list[str], objects: list[dict]) -> np.ndarray:
    assert objects, "expected at least one object"
    object_dirs = {obj["direction"] for obj in objects}
    assert object_dirs <= set(direction_order), "every object direction must appear in the indicator row"

    max_width = max(obj["patch"].shape[1] for obj in objects)
    col_width = max_width + 1
    stack_heights = [
        sum(obj["patch"].shape[0] for obj in objects if obj["direction"] == direction)
        for direction in direction_order
    ]
    out_h = max(stack_heights)
    out_w = len(direction_order) * col_width
    out = np.full((out_h, out_w), bg, dtype=int)

    for idx, direction in enumerate(direction_order):
        col0 = idx * col_width
        cursor = 0
        group = sorted(
            [obj for obj in objects if obj["direction"] == direction],
            key=lambda obj: (obj["bbox"][0], obj["bbox"][1]),
        )
        for obj in group:
            patch = obj["patch"]
            ph, pw = patch.shape
            dst = out[cursor : cursor + ph, col0 : col0 + pw]
            mask = patch != bg
            dst[mask] = patch[mask]
            cursor += ph

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group scattered line-like objects by their skeleton direction and stack each group into a column."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _get_bg(g)
    direction_order = _parse_indicator_order(g, bg)
    objects = _find_components(g[:-1], bg)
    classified = [{**obj, "direction": _classify_direction(obj)} for obj in objects]
    return _render_output(bg, direction_order, classified)


# --------------- generator ---------------

_RNG = np.random.default_rng()
BG_CHOICES = np.array([0, 5, 6, 7, 8, 9], dtype=int)
ALL_COLORS = tuple(range(10))
SCENE_H_RANGE = (9, 17)  # inclusive low, exclusive high
GRID_W_RANGE = (15, 29)  # inclusive low, exclusive high
MAX_GENERATION_ATTEMPTS = 400

DIAG_OPTIONS = [
    (dr, dc)
    for dr in range(1, 5)
    for dc in range(1, 5)
    if 3 <= dr + dc + 1 <= 8 and not ((dc + 1) > 2 * (dr + 1) or (dr + 1) > 2 * (dc + 1))
]


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    return coords - mins


def _flip_coords(coords: np.ndarray, flip_r: bool = False, flip_c: bool = False) -> np.ndarray:
    out = coords.copy()
    if flip_r:
        out[:, 0] = out[:, 0].max() - out[:, 0]
    if flip_c:
        out[:, 1] = out[:, 1].max() - out[:, 1]
    return _normalize_coords(out)


def _monotone_path(dr: int, dc: int) -> np.ndarray:
    moves = np.array([0] * dr + [1] * dc, dtype=int)
    _RNG.shuffle(moves)
    r, c = 0, 0
    coords = [(r, c)]
    for move in moves:
        if int(move) == 0:
            r += 1
        else:
            c += 1
        coords.append((r, c))
    return np.array(coords, dtype=int)


def _sample_shape(direction: str) -> np.ndarray:
    if direction == HORIZONTAL:
        if _RNG.random() < 0.5:
            coords = _monotone_path(0, int(_RNG.integers(2, 8)))
        else:
            coords = _monotone_path(1, int(_RNG.integers(4, 7)))
        return _flip_coords(coords, flip_r=bool(_RNG.integers(0, 2)), flip_c=bool(_RNG.integers(0, 2)))

    if direction == VERTICAL:
        if _RNG.random() < 0.5:
            coords = _monotone_path(int(_RNG.integers(2, 8)), 0)
        else:
            coords = _monotone_path(int(_RNG.integers(4, 7)), 1)
        return _flip_coords(coords, flip_r=bool(_RNG.integers(0, 2)), flip_c=bool(_RNG.integers(0, 2)))

    dr, dc = DIAG_OPTIONS[int(_RNG.integers(0, len(DIAG_OPTIONS)))]
    coords = _monotone_path(dr, dc)
    if direction == DIAG_UP:
        coords = _flip_coords(coords, flip_c=True)
    if _RNG.random() < 0.5:
        coords = _flip_coords(coords, flip_r=True, flip_c=True)
    return coords


def _coords_to_patch(coords: np.ndarray, color: int, bg: int) -> np.ndarray:
    coords = _normalize_coords(coords)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.full((h, w), bg, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _sample_active_directions() -> list[str]:
    n_dirs = int(_RNG.integers(2, 5))
    chosen = sorted(_RNG.choice(len(CANONICAL_DIRECTIONS), size=n_dirs, replace=False))
    return [CANONICAL_DIRECTIONS[idx] for idx in chosen]


def _sample_direction_counts(active_dirs: list[str]) -> dict[str, int]:
    max_objects = min(6, 9 - len(active_dirs))
    counts = {direction: 1 for direction in active_dirs}
    extra_budget = min(2, max_objects - len(active_dirs))
    extras = int(_RNG.integers(0, extra_budget + 1))
    for _ in range(extras):
        direction = active_dirs[int(_RNG.integers(0, len(active_dirs)))]
        counts[direction] += 1
    return counts


def _place_objects(scene_h: int, width: int, bg: int, objects: list[dict]) -> list[dict] | None:
    scene = np.full((scene_h, width), bg, dtype=int)
    blocked = np.zeros((scene_h, width), dtype=bool)
    used_top_rows: dict[str, set[int]] = {direction: set() for direction in CANONICAL_DIRECTIONS}
    placed = []

    order = sorted(
        objects,
        key=lambda obj: (-(obj["patch"].shape[0] * obj["patch"].shape[1]), -max(obj["patch"].shape)),
    )
    for obj in order:
        patch = obj["patch"]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(scene_h - ph + 1):
            if r0 in used_top_rows[obj["direction"]]:
                continue
            for c0 in range(width - pw + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(scene_h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + pw + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = candidates[int(_RNG.integers(0, len(candidates)))]
        mask = patch != bg
        scene[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        blocked[max(0, r0 - 1) : min(scene_h, r0 + ph + 1), max(0, c0 - 1) : min(width, c0 + pw + 1)] = True
        used_top_rows[obj["direction"]].add(r0)
        placed.append(
            {
                **obj,
                "bbox": (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
            }
        )

    placed.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return [{"scene": scene}] + placed


def _build_input(scene_h: int, width: int, bg: int, active_dirs: list[str], scene: np.ndarray) -> np.ndarray:
    grid = np.full((scene_h + 2, width), bg, dtype=int)
    grid[:scene_h, :] = scene

    marker_width = 3 * len(active_dirs) - 1
    start = int(_RNG.integers(0, width - marker_width + 1))
    c = start
    for direction in active_dirs:
        grid[-1, c : c + 2] = DIR_TO_MARKER[direction]
        c += 3
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = int(BG_CHOICES[int(_RNG.integers(0, len(BG_CHOICES)))])
    active_dirs = _sample_active_directions()
    counts = _sample_direction_counts(active_dirs)
    total_objects = sum(counts.values())

    blocked_colors = {bg} | {DIR_TO_MARKER[direction] for direction in active_dirs}
    available_colors = [color for color in ALL_COLORS if color not in blocked_colors]
    if len(available_colors) < total_objects:
        return None

    color_order = list(_RNG.choice(np.array(available_colors, dtype=int), size=total_objects, replace=False))
    objects = []
    color_idx = 0
    for direction in active_dirs:
        for _ in range(counts[direction]):
            color = int(color_order[color_idx])
            color_idx += 1
            coords = _sample_shape(direction)
            patch = _coords_to_patch(coords, color, bg)
            objects.append({"color": color, "direction": direction, "patch": patch})

    marker_width = 3 * len(active_dirs) - 1
    min_width = max(GRID_W_RANGE[0], marker_width + 2)
    if min_width >= GRID_W_RANGE[1]:
        return None
    scene_h = int(_RNG.integers(*SCENE_H_RANGE))
    width = int(_RNG.integers(min_width, GRID_W_RANGE[1]))

    placed = _place_objects(scene_h, width, bg, objects)
    if placed is None:
        return None
    scene = placed[0]["scene"]
    placed_objects = placed[1:]

    expected = _render_output(bg, active_dirs, placed_objects)
    grid = _build_input(scene_h, width, bg, active_dirs, scene)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
