import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

MAX_GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 30)  # high exclusive for rng.integers
OBJECT_COUNT_CHOICES = (2, 3, 3, 4)
RING_RADIUS_CHOICES = (3, 4, 5, 6)
EQUAL_RADIUS_PROB = 0.35
PLACEMENT_GAP = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _solid_square_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    area = int(coords.shape[0])
    if area < 9:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    if side_r != side_c or side_r % 2 == 0:
        return None
    if area != side_r * side_r:
        return None
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    if not np.all(patch == int(color)):
        return None
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": int(side_r),
        "center": (int((r0 + r1) // 2), int((c0 + c1) // 2)),
    }


def _ring_offsets(radius: int) -> list[tuple[int, int]]:
    d = int(radius)
    return [
        (dr, dc)
        for dr in range(-d, d + 1)
        for dc in range(-d, d + 1)
        if max(abs(dr), abs(dc)) == d
    ]


def _band_cells(shape: tuple[int, int], center: tuple[int, int], d1: int, d2: int) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    cells: list[tuple[int, int]] = []
    for r in range(cr - d2, cr + d2 + 1):
        for c in range(cc - d2, cc + d2 + 1):
            if 0 <= r < h and 0 <= c < w and d1 <= max(abs(r - cr), abs(c - cc)) <= d2:
                cells.append((r, c))
    return cells


def _paint_band(out: np.ndarray, center: tuple[int, int], d1: int, d2: int, color: int) -> None:
    for r, c in _band_cells(out.shape, center, int(d1), int(d2)):
        out[r, c] = int(color)


def _transform_from_objects(grid: np.ndarray, objects: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    for obj in objects:
        _paint_band(out, obj["center"], int(obj["d1"]), int(obj["d2"]), int(obj["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid odd square by a concentric square band determined by its two same-colored satellites.

    Input structure:
    - A dominant background contains several non-overlapping solid odd-sided squares in distinct colors.
    - Each square color also appears in exactly two isolated satellite pixels away from that square's center.
    - Additional distractor pixels use colors that do not belong to any square.

    Rule:
    - For every square, keep only its center and color; ignore the square's own side length.
    - Find the two same-colored satellite pixels, compute their Chebyshev distances to the square center, sort them as d1 <= d2, erase the original square, and paint every cell whose Chebyshev distance from that center lies in [d1, d2].
    - Satellites are absorbed into the painted band; distractor pixels remain unless a band overwrites them.

    Key insight:
    - The visible square is a decoy: the answer depends only on the square center and on the near/far Chebyshev radii encoded by the two same-colored satellite pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    square_objects: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components_of_color(g, color)
        square = None
        singleton_pixels: list[tuple[int, int]] = []

        for comp in comps:
            info = _solid_square_info(g, color, comp)
            if info is not None:
                assert square is None, "each square color should have only one square component"
                square = info
            else:
                assert len(comp) == 1, "non-square components of a square color must be singleton satellites"
                rr, cc = map(int, comp[0])
                singleton_pixels.append((rr, cc))

        if square is None:
            continue

        assert len(singleton_pixels) == 2, "each square color must have exactly two satellites"
        cr, cc = square["center"]
        dists = sorted(max(abs(r - cr), abs(c - cc)) for r, c in singleton_pixels)
        assert dists[0] >= 1, "satellite must not coincide with the square center"
        square["d1"] = int(dists[0])
        square["d2"] = int(dists[1])
        square_objects.append(square)

    assert square_objects, "expected at least one square object"
    return _transform_from_objects(g, square_objects, bg)


def _bbox_from_center(center: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    cr, cc = center
    d = int(radius)
    return cr - d, cc - d, cr + d, cc + d


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _choose_object_spec() -> dict:
    d2 = int(RNG.choice(RING_RADIUS_CHOICES))
    if RNG.random() < EQUAL_RADIUS_PROB:
        d1 = d2
    else:
        d1 = int(RNG.integers(2, d2 + 1))
    side_choices = [side for side in range(3, 2 * d1, 2)]
    side = int(RNG.choice(side_choices))
    return {"d1": d1, "d2": d2, "side": side}


def _place_objects(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda x: (x["d2"], x["side"]), reverse=True):
        d2 = int(spec["d2"])
        candidates = [
            (r, c)
            for r in range(d2, h - d2)
            for c in range(d2, w - d2)
        ]
        if not candidates:
            return None
        RNG.shuffle(candidates)

        for center in candidates:
            box = _bbox_from_center(center, d2)
            if all(_boxes_separated(box, other, PLACEMENT_GAP) for other in reserved):
                item = dict(spec)
                item["center"] = (int(center[0]), int(center[1]))
                item["bbox"] = box
                placed.append(item)
                reserved.append(box)
                break
        else:
            return None
    return placed


def _square_cells(center: tuple[int, int], side: int) -> list[tuple[int, int]]:
    cr, cc = center
    half = side // 2
    return [(r, c) for r in range(cr - half, cr + half + 1) for c in range(cc - half, cc + half + 1)]


def _choose_satellites(obj: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    cr, cc = obj["center"]
    side_half = obj["side"] // 2

    def choose_one(radius: int, forbid: set[tuple[int, int]]) -> tuple[int, int]:
        offsets = _ring_offsets(int(radius))
        RNG.shuffle(offsets)
        for dr, dc in offsets:
            if max(abs(dr), abs(dc)) <= side_half:
                continue
            cell = (cr + dr, cc + dc)
            if cell not in forbid:
                return cell
        raise RuntimeError("failed to choose a legal satellite")

    first = choose_one(int(obj["d1"]), set())
    second = choose_one(int(obj["d2"]), {first})
    return first, second


def _render_input(h: int, w: int, objects: list[dict], distractors: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in _square_cells(obj["center"], int(obj["side"])):
            grid[r, c] = int(obj["color"])
        for r, c in obj["satellites"]:
            grid[r, c] = int(obj["color"])
    for r, c, color in distractors:
        grid[r, c] = int(color)
    return grid


def _add_distractors(
    h: int,
    w: int,
    objects: list[dict],
    distractor_colors: list[int],
) -> list[tuple[int, int, int]] | None:
    occupied = set()
    ring_cells: list[tuple[int, int]] = []
    safe_cells = {(r, c) for r in range(h) for c in range(w)}

    for obj in objects:
        occupied.update(_square_cells(obj["center"], int(obj["side"])))
        occupied.update(obj["satellites"])
        current_ring = set(_band_cells((h, w), obj["center"], int(obj["d1"]), int(obj["d2"])))
        ring_cells.extend(sorted(current_ring))
        safe_cells.difference_update(current_ring)

    safe_cells.difference_update(occupied)
    ring_candidates = [cell for cell in ring_cells if cell not in occupied]
    ring_candidates = list(dict.fromkeys(ring_candidates))
    RNG.shuffle(ring_candidates)

    safe_candidates = list(safe_cells)
    RNG.shuffle(safe_candidates)

    distractors: list[tuple[int, int, int]] = []
    covered_target = 1 if distractor_colors and RNG.random() < 0.7 and ring_candidates else 0
    if covered_target and len(distractor_colors) >= 2 and RNG.random() < 0.25 and len(ring_candidates) >= 2:
        covered_target = 2

    for color in distractor_colors[:covered_target]:
        if not ring_candidates:
            return None
        r, c = ring_candidates.pop()
        distractors.append((int(r), int(c), int(color)))

    for color in distractor_colors[covered_target:]:
        if not safe_candidates:
            return None
        r, c = safe_candidates.pop()
        distractors.append((int(r), int(c), int(color)))

    return distractors


def _expected_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    expected = inp.copy()
    for obj in objects:
        half = int(obj["side"]) // 2
        cr, cc = obj["center"]
        expected[cr - half : cr + half + 1, cc - half : cc + half + 1] = BG
    for obj in objects:
        _paint_band(expected, obj["center"], int(obj["d1"]), int(obj["d2"]), int(obj["color"]))
    return expected


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(*GRID_SIDE_RANGE))
    w = int(RNG.integers(*GRID_SIDE_RANGE))
    object_count = int(RNG.choice(OBJECT_COUNT_CHOICES))
    max_distractors = min(4, len(PALETTE) - object_count)
    distractor_count = int(RNG.integers(1, max_distractors + 1))

    colors = RNG.choice(PALETTE, size=object_count + distractor_count, replace=False).astype(int).tolist()
    object_colors = colors[:object_count]
    distractor_colors = colors[object_count:]

    specs = [_choose_object_spec() for _ in range(object_count)]
    placed = _place_objects(h, w, specs)
    if placed is None:
        return None

    objects: list[dict] = []
    for color, obj in zip(object_colors, placed):
        item = dict(obj)
        item["color"] = int(color)
        item["satellites"] = _choose_satellites(item)
        objects.append(item)

    distractors = _add_distractors(h, w, objects, distractor_colors)
    if distractors is None:
        return None

    inp = _render_input(h, w, objects, distractors)
    if _get_bg(inp) != BG:
        return None

    expected = _expected_output(inp, objects)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 057/5")
