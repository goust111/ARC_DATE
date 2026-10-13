import numpy as np


ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 8
GRID_MIN = 20
GRID_MAX = 30
OBJECT_COUNT_RANGE = (3, 5)
PLACEMENT_TRIES = 250
GENERATION_RETRIES = 400

RNG = np.random.default_rng()


def dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def component_masks(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = dominant_color(g)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ORTHO_DIRS:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


def core_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    count = 0
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if (
            0 < rr < mask.shape[0] - 1
            and 0 < cc < mask.shape[1] - 1
            and bool(mask[rr - 1, cc])
            and bool(mask[rr + 1, cc])
            and bool(mask[rr, cc - 1])
            and bool(mask[rr, cc + 1])
        ):
            count += 1
    return int(count)


def shrink_half(mask: np.ndarray) -> np.ndarray:
    new_h = int(mask.shape[0] // 2)
    new_w = int(mask.shape[1] // 2)
    if new_h == 0 or new_w == 0:
        return np.zeros((new_h, new_w), dtype=bool)
    return mask[: new_h * 2 : 2, : new_w * 2 : 2].copy()


def scale_mask(mask: np.ndarray, n_core: int) -> np.ndarray:
    if n_core == 0:
        return shrink_half(mask)
    if n_core == 1:
        return mask.copy()
    return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)


def centered_top_left(bbox: tuple[int, int, int, int], new_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    new_h, new_w = new_shape
    top = (int(r0) + int(r1) - (int(new_h) - 1)) // 2
    left = (int(c0) + int(c1) - (int(new_w) - 1)) // 2
    return int(top), int(left)


def paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    height, width = grid.shape
    for r, c in np.argwhere(mask):
        rr = int(top + int(r))
        cc = int(left + int(c))
        if 0 <= rr < height and 0 <= cc < width:
            grid[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = dominant_color(g)
    out = np.full_like(g, bg)

    for comp in component_masks(g, bg=bg):
        n_core = core_count(comp["mask"])
        scaled = scale_mask(comp["mask"], n_core)
        if scaled.size == 0 or not np.any(scaled):
            continue
        top, left = centered_top_left(comp["bbox"], scaled.shape)
        paint_mask(out, scaled, top, left, comp["color"])

    return out


def normalize_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(int(r - min_r), int(c - min_c)) for r, c in coords}


def coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    coords = normalize_coords(coords)
    height = max(r for r, _ in coords) + 1
    width = max(c for _, c in coords) + 1
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def frontier(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = set()
    for r, c in coords:
        for dr, dc in ORTHO_DIRS:
            nxt = (int(r + dr), int(c + dc))
            if nxt not in coords:
                out.add(nxt)
    return list(out)


def core_count_coords(coords: set[tuple[int, int]]) -> int:
    count = 0
    for r, c in coords:
        if all((int(r + dr), int(c + dc)) in coords for dr, dc in ORTHO_DIRS):
            count += 1
    return int(count)


def random_order(items: list, rng: np.random.Generator) -> list:
    if not items:
        return []
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def generate_zero_core_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(160):
        target = int(rng.integers(4, 13))
        coords = {(0, 0)}
        while len(coords) < target:
            candidates = frontier(coords)
            coords.add(candidates[int(rng.integers(len(candidates)))])
        mask = coords_to_mask(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) > 20 or core_count(mask) != 0:
            continue
        shrunk = shrink_half(mask)
        if shrunk.size == 0 or not np.any(shrunk):
            continue
        return mask
    raise RuntimeError("failed to generate a zero-core shape")


def generate_one_core_shape(rng: np.random.Generator) -> np.ndarray:
    base = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    for _ in range(200):
        target = int(rng.integers(5, 15))
        coords = set(base)
        while len(coords) < target:
            added = False
            for cell in random_order(frontier(coords), rng):
                trial = set(coords)
                trial.add(cell)
                if core_count_coords(trial) == 1:
                    coords = trial
                    added = True
                    break
            if not added:
                break
        mask = coords_to_mask(coords)
        if 5 <= int(mask.sum()) <= 20 and core_count(mask) == 1:
            return mask
    raise RuntimeError("failed to generate a one-core shape")


def generate_multi_core_shape(rng: np.random.Generator) -> np.ndarray:
    rectangles = [
        (3, 4),
        (4, 3),
        (3, 5),
        (5, 3),
        (3, 6),
        (6, 3),
        (4, 4),
        (4, 5),
        (5, 4),
    ]
    for _ in range(160):
        h, w = rectangles[int(rng.integers(len(rectangles)))]
        coords = {(int(r), int(c)) for r in range(h) for c in range(w)}
        extra_steps = int(rng.integers(0, min(5, 21 - len(coords))))
        for _ in range(extra_steps):
            for cell in random_order(frontier(coords), rng):
                trial = set(coords)
                trial.add(cell)
                if len(trial) <= 20 and core_count_coords(trial) >= 2:
                    coords = trial
                    break
        mask = coords_to_mask(coords)
        if 12 <= int(mask.sum()) <= 20 and core_count(mask) >= 2:
            return mask
    raise RuntimeError("failed to generate a multi-core shape")


def absolute_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in np.argwhere(mask)]


def clipped_output_coords(
    scaled_mask: np.ndarray,
    input_shape: tuple[int, int],
    top: int,
    left: int,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    input_h, input_w = input_shape
    out_h, out_w = scaled_mask.shape
    offset_r = (int(input_h) - int(out_h)) // 2
    offset_c = (int(input_w) - int(out_w)) // 2
    grid_h, grid_w = grid_shape
    coords = []
    for r, c in np.argwhere(scaled_mask):
        rr = int(top + offset_r + int(r))
        cc = int(left + offset_c + int(c))
        if 0 <= rr < grid_h and 0 <= cc < grid_w:
            coords.append((rr, cc))
    return coords


def fits_with_clearance(
    coords: list[tuple[int, int]],
    occupied: np.ndarray,
    radius: int,
) -> bool:
    height, width = occupied.shape
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width and occupied[rr, cc]:
                    return False
    return True


def mark_with_clearance(
    coords: list[tuple[int, int]],
    occupied: np.ndarray,
    radius: int,
) -> None:
    height, width = occupied.shape
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    occupied[rr, cc] = True


def make_object(branch: int, color: int, rng: np.random.Generator) -> dict:
    if branch == 0:
        mask = generate_zero_core_shape(rng)
    elif branch == 1:
        mask = generate_one_core_shape(rng)
    else:
        mask = generate_multi_core_shape(rng)
    n_core = core_count(mask)
    scaled = scale_mask(mask, n_core)
    return {
        "branch": int(branch),
        "color": int(color),
        "mask": mask,
        "scaled": scaled,
        "core": int(n_core),
        "input_area": int(mask.sum()),
        "scaled_area": int(scaled.sum()),
    }


def try_place_objects(
    objects: list[dict],
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    grid = np.full((int(grid_h), int(grid_w)), BG_COLOR, dtype=int)
    occupied_in = np.zeros((int(grid_h), int(grid_w)), dtype=bool)
    occupied_out = np.zeros((int(grid_h), int(grid_w)), dtype=bool)

    ordered = sorted(
        objects,
        key=lambda obj: (
            -max(int(obj["scaled_area"]), int(obj["input_area"])),
            -max(obj["scaled"].shape[0], obj["scaled"].shape[1]),
        ),
    )

    for obj in ordered:
        h, w = obj["mask"].shape
        if h > grid_h or w > grid_w:
            return None

        placed = False
        r_max = int(grid_h - h)
        c_max = int(grid_w - w)
        for _ in range(PLACEMENT_TRIES):
            top = int(rng.integers(0, r_max + 1))
            left = int(rng.integers(0, c_max + 1))
            in_coords = absolute_coords(obj["mask"], top, left)
            out_coords = clipped_output_coords(obj["scaled"], obj["mask"].shape, top, left, (grid_h, grid_w))
            if not out_coords:
                continue
            if not fits_with_clearance(in_coords, occupied_in, radius=1):
                continue
            if not fits_with_clearance(out_coords, occupied_out, radius=1):
                continue

            for r, c in in_coords:
                grid[r, c] = int(obj["color"])
            mark_with_clearance(in_coords, occupied_in, radius=1)
            mark_with_clearance(out_coords, occupied_out, radius=1)
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    object_count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    branches = [0, 1, 2]
    while len(branches) < object_count:
        branches.append(int(rng.integers(0, 3)))
    branches = random_order(branches, rng)

    palette = [color for color in range(10) if color != BG_COLOR]
    colors = list(rng.choice(palette, size=object_count, replace=False))
    objects = [make_object(branch, int(color), rng) for branch, color in zip(branches, colors)]

    min_h = max(max(obj["mask"].shape[0], obj["scaled"].shape[0]) for obj in objects) + 4
    min_w = max(max(obj["mask"].shape[1], obj["scaled"].shape[1]) for obj in objects) + 4
    low_h = max(GRID_MIN, int(min_h))
    low_w = max(GRID_MIN, int(min_w))
    if low_h > GRID_MAX or low_w > GRID_MAX:
        return None

    grid_h = int(rng.integers(low_h, GRID_MAX + 1))
    grid_w = int(rng.integers(low_w, GRID_MAX + 1))
    grid = try_place_objects(objects, grid_h, grid_w, rng)
    if grid is None:
        return None

    if dominant_color(grid) != BG_COLOR:
        return None
    components = component_masks(grid, bg=BG_COLOR)
    if len(components) != object_count:
        return None

    seen_branches = sorted(core_count(comp["mask"]) for comp in components)
    if not any(value == 0 for value in seen_branches):
        return None
    if not any(value == 1 for value in seen_branches):
        return None
    if not any(value >= 2 for value in seen_branches):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != BG_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
