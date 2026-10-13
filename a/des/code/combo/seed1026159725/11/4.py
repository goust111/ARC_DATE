import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(10))
GEN_RETRY_LIMIT = 400
_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox(coords_arr),
                }
            )

    return components


def _crop_component(grid: np.ndarray, component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    color = int(component["color"])
    for r, c in component["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _longest_consecutive_true_1d(arr: np.ndarray) -> int:
    best = 0
    run = 0
    for value in arr:
        if bool(value):
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def _dominant_segment_lengths(mask: np.ndarray) -> tuple[int, int]:
    max_h = 0
    for row in mask:
        max_h = max(max_h, _longest_consecutive_true_1d(row))

    max_v = 0
    for col in mask.T:
        max_v = max(max_v, _longest_consecutive_true_1d(col))

    return int(max_h), int(max_v)


def _centered_top_left(r0: int, c0: int, old_h: int, old_w: int, new_h: int, new_w: int) -> tuple[int, int]:
    return int(r0 + (old_h - new_h) // 2), int(c0 + (old_w - new_w) // 2)


def _paste_non_bg(out: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != bg
    out[top : top + h, left : left + w][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each foreground object when its longest internal line segment is horizontal."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    out = np.full_like(g, bg)
    for component in _find_components(g, bg):
        patch = _crop_component(g, component, bg)
        mask = patch != bg
        max_h, max_v = _dominant_segment_lengths(mask)

        if max_h > max_v:
            patch = np.rot90(patch, k=-1)

        r0, c0, r1, c1 = component["bbox"]
        old_h = int(r1 - r0 + 1)
        old_w = int(c1 - c0 + 1)
        new_h, new_w = map(int, patch.shape)
        top, left = _centered_top_left(r0, c0, old_h, old_w, new_h, new_w)
        _paste_non_bg(out, patch, bg, top, left)

    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _coords_to_patch(coords: np.ndarray, color: int, bg: int) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.full((h, w), bg, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _orientation_of_patch(patch: np.ndarray, bg: int) -> str | None:
    mask = patch != bg
    max_h, max_v = _dominant_segment_lengths(mask)
    if max_h > max_v:
        return "horizontal"
    if max_v > max_h:
        return "vertical"
    return None


def _make_object_patch(orientation: str, color: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        stem_len = int(rng.integers(4, 8))
        arm_slots = np.arange(1, stem_len - 1)
        max_arm_count = min(4, len(arm_slots))
        arm_count = int(rng.integers(1, max_arm_count + 1))
        arm_rows = rng.choice(arm_slots, size=arm_count, replace=False)

        coords: set[tuple[int, int]] = set()
        if orientation == "vertical":
            for r in range(stem_len):
                coords.add((r, 0))
            for row in arm_rows:
                arm_len = int(rng.integers(1, min(4, stem_len - 1)))
                arm_len = min(arm_len, stem_len - 2)
                side = int(rng.choice([-1, 1]))
                for delta in range(1, arm_len + 1):
                    coords.add((int(row), side * delta))
        else:
            for c in range(stem_len):
                coords.add((0, c))
            for col in arm_rows:
                arm_len = int(rng.integers(1, min(4, stem_len - 1)))
                arm_len = min(arm_len, stem_len - 2)
                side = int(rng.choice([-1, 1]))
                for delta in range(1, arm_len + 1):
                    coords.add((side * delta, int(col)))

        norm = _normalize_coords(coords)
        patch = _coords_to_patch(norm, color, bg)
        if _orientation_of_patch(patch, bg) != orientation:
            continue
        return patch

    raise RuntimeError(f"failed to build a {orientation} object")


def _transformed_union_box(patch: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = map(int, patch.shape)
    if _orientation_of_patch(patch, bg) == "horizontal":
        rotated = np.rot90(patch, k=-1)
        rh, rw = map(int, rotated.shape)
        dr, dc = _centered_top_left(0, 0, h, w, rh, rw)
    else:
        rh, rw = h, w
        dr, dc = 0, 0

    top = min(0, int(dr))
    left = min(0, int(dc))
    bottom = max(h, int(dr + rh))
    right = max(w, int(dc + rw))
    return int(top), int(left), int(bottom), int(right)


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    ut, ul, ub, ur = _transformed_union_box(patch, bg)

    min_r = -ut
    max_r = h - ub
    min_c = -ul
    max_c = w - ur
    if min_r > max_r or min_c > max_c:
        return False

    candidates = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        rr0 = max(0, r0 + ut - 1)
        rr1 = min(h, r0 + ub + 1)
        cc0 = max(0, c0 + ul - 1)
        cc1 = min(w, c0 + ur + 1)
        if bool(reserved[rr0:rr1, cc0:cc1].any()):
            continue

        _paste_non_bg(grid, patch, bg, int(r0), int(c0))
        reserved[rr0:rr1, cc0:cc1] = True
        return True

    return False


def _check_grid(grid: np.ndarray, expected_count: int) -> bool:
    bg = get_bg(grid)
    components = _find_components(grid, bg)
    if len(components) != expected_count:
        return False

    orientations = []
    for component in components:
        patch = _crop_component(grid, component, bg)
        orient = _orientation_of_patch(patch, bg)
        if orient is None:
            return False
        orientations.append(orient)

    if "horizontal" not in orientations or "vertical" not in orientations:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if len(_find_components(out, bg)) != expected_count:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(12, 19))
    width = int(rng.integers(12, 19))
    bg = int(rng.choice(COLOR_POOL))
    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    obj_count = int(rng.integers(2, 6))
    colors = [int(c) for c in rng.choice([c for c in COLOR_POOL if c != bg], size=obj_count, replace=False)]

    orientations = ["horizontal", "vertical"]
    while len(orientations) < obj_count:
        orientations.append(str(rng.choice(["horizontal", "vertical"])))
    rng.shuffle(orientations)

    patches = [
        _make_object_patch(orientation, color, bg, rng)
        for orientation, color in zip(orientations, colors)
    ]

    order = list(range(obj_count))
    rng.shuffle(order)
    for idx in order:
        if not _place_patch(grid, reserved, patches[idx], bg, rng):
            return None

    if not _check_grid(grid, obj_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
