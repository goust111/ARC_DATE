import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROW_PATTERNS = {
    "up": np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=int),
    "down": np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
    "left": np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=int),
    "right": np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=int),
}
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (13, 20)
OBJECT_COUNT_RANGE = (2, 5)
MARKER_COUNT_RANGE = (1, 5)
PLACEMENT_BORDER = 1
PLACEMENT_PAD = 1
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_array(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rel = coords - np.array([r0, c0], dtype=int)
    arr[rel[:, 0], rel[:, 1]] = 1
    return arr


def _infer_direction(coords: np.ndarray) -> str:
    arr = _shape_array(coords)
    if arr.shape != (3, 3) or int(arr.sum()) != 7:
        raise AssertionError("arrow component must be a 3x3, 7-cell arrow")
    for direction, pattern in ARROW_PATTERNS.items():
        if np.array_equal(arr, pattern):
            return direction
    raise AssertionError("component shape does not match any arrow orientation")


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    cells = np.argwhere(patch == 1)
    for rr, cc in cells:
        grid[top + int(rr), left + int(cc)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored arrow along its own pointing direction by the number of same-colored isolated markers.
    Same-colored single-pixel markers are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = _components(g == color)
        if not comps:
            continue

        arrow = max(comps, key=lambda comp: int(comp.shape[0]))
        marker_count = sum(1 for comp in comps if int(comp.shape[0]) == 1)
        direction = _infer_direction(arrow)
        dr, dc = DIR_VECTORS[direction]
        shift_r = dr * marker_count
        shift_c = dc * marker_count

        for rr, cc in arrow:
            nr = int(rr) + shift_r
            nc = int(cc) + shift_c
            if not (0 <= nr < h and 0 <= nc < w):
                raise AssertionError("translated arrow moved out of bounds")
            out[nr, nc] = color

    return out


def _clear_with_pad(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _placeable_bounds(length: int, shift: int, border: int) -> tuple[int, int]:
    low = border
    high = length - 3 - border
    return low, high


def _sample_arrow_origin(
    rng: np.random.Generator, h: int, w: int, direction: str, steps: int
) -> tuple[int, int] | None:
    r_low, r_high = _placeable_bounds(h, steps, PLACEMENT_BORDER)
    c_low, c_high = _placeable_bounds(w, steps, PLACEMENT_BORDER)
    if direction == "up":
        r_low += steps
    elif direction == "down":
        r_high -= steps
    elif direction == "left":
        c_low += steps
    else:
        c_high -= steps
    if r_low > r_high or c_low > c_high:
        return None
    r0 = int(rng.integers(r_low, r_high + 1))
    c0 = int(rng.integers(c_low, c_high + 1))
    return r0, c0


def _build_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    object_count = int(rng.integers(*OBJECT_COUNT_RANGE))
    if object_count > len(COLORS):
        return None

    colors = [int(v) for v in rng.choice(COLORS, size=object_count, replace=False)]
    reserved_in = np.zeros((h, w), dtype=bool)
    reserved_out = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for color in colors:
        placed = False
        for _ in range(MAX_ATTEMPTS):
            direction = str(rng.choice(list(DIR_VECTORS.keys())))
            steps = int(rng.integers(*MARKER_COUNT_RANGE))
            origin = _sample_arrow_origin(rng, h, w, direction, steps)
            if origin is None:
                continue
            r0, c0 = origin
            dr, dc = DIR_VECTORS[direction]
            out_r0 = r0 + dr * steps
            out_c0 = c0 + dc * steps

            if not _clear_with_pad(reserved_in, r0, c0, r0 + 2, c0 + 2, PLACEMENT_PAD):
                continue
            if not _clear_with_pad(reserved_out, out_r0, out_c0, out_r0 + 2, out_c0 + 2, PLACEMENT_PAD):
                continue

            _reserve(reserved_in, r0, c0, r0 + 2, c0 + 2, PLACEMENT_PAD)
            _reserve(reserved_out, out_r0, out_c0, out_r0 + 2, out_c0 + 2, PLACEMENT_PAD)
            objects.append(
                {
                    "color": color,
                    "direction": direction,
                    "steps": steps,
                    "in_top": r0,
                    "in_left": c0,
                    "out_top": out_r0,
                    "out_left": out_c0,
                    "markers": [],
                }
            )
            placed = True
            break
        if not placed:
            return None

    for obj in objects:
        steps = int(obj["steps"])
        markers: list[tuple[int, int]] = []
        for _ in range(steps):
            placed = False
            for _ in range(MAX_ATTEMPTS):
                r = int(rng.integers(0, h))
                c = int(rng.integers(0, w))
                if not _clear_with_pad(reserved_in, r, c, r, c, PLACEMENT_PAD):
                    continue
                _reserve(reserved_in, r, c, r, c, PLACEMENT_PAD)
                markers.append((r, c))
                placed = True
                break
            if not placed:
                return None
        obj["markers"] = markers

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        direction = str(obj["direction"])
        _paint_patch(inp, ARROW_PATTERNS[direction], int(obj["in_top"]), int(obj["in_left"]), color)
        _paint_patch(out, ARROW_PATTERNS[direction], int(obj["out_top"]), int(obj["out_left"]), color)
        for r, c in obj["markers"]:
            inp[int(r), int(c)] = color

    if np.any(inp != BG) and len(np.unique(inp[inp != BG])) != object_count:
        return None
    if not np.array_equal(main(inp), out):
        return None

    for color in colors:
        comps = _components(inp == color)
        if sum(1 for comp in comps if int(comp.shape[0]) > 1) != 1:
            return None
        if sum(1 for comp in comps if int(comp.shape[0]) == 1) not in range(1, 5):
            return None

    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        pair = _build_pair(rng)
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid arrow-translation instance")
