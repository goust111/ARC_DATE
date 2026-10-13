import numpy as np


BG = 0
COLORS = np.arange(1, 10, dtype=int)
SIZE_VALUES = np.arange(2, 8, dtype=int)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MIN_OBJECTS = 2
MAX_OBJECTS = 5
MAX_BBOX_EXTENSION = 2
IRREGULAR_PROBABILITY = 0.75
MAX_GENERATION_ATTEMPTS = 500


def _parse_runs(values: np.ndarray) -> dict[int, int]:
    runs: dict[int, int] = {}
    i = 0
    while i < len(values):
        color = int(values[i])
        j = i + 1
        while j < len(values) and int(values[j]) == color:
            j += 1
        assert color not in runs
        runs[color] = int(j - i)
        i = j
    return runs


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top ruler for target widths and the left ruler for target heights,
    then replace each interior object with a solid rectangle of that size at the
    object's original bounding-box top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    width_by_color = _parse_runs(g[0, :])
    height_by_color = _parse_runs(g[1:, 0])

    inner = g[1:, 1:]
    components = _find_components(inner)
    out = g.copy()

    for comp in components:
        coords = comp["coords"]
        out[coords[:, 0] + 1, coords[:, 1] + 1] = BG

    for comp in components:
        color = int(comp["color"])
        target_w = int(width_by_color[color])
        target_h = int(height_by_color[color])
        r0, c0, _r1, _c1 = comp["bbox"]
        out[1 + r0 : 1 + r0 + target_h, 1 + c0 : 1 + c0 + target_w] = color

    return out


def _random_monotone_path(rng: np.random.Generator, h: int, w: int) -> set[tuple[int, int]]:
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    r = 0
    c = 0
    cells = {(0, 0)}
    for step in steps:
        if step == 0:
            r += 1
        else:
            c += 1
        cells.add((r, c))
    return cells


def _grow_connected_shape(rng: np.random.Generator, bbox_h: int, bbox_w: int, area: int) -> np.ndarray:
    cells = _random_monotone_path(rng, bbox_h, bbox_w)
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        assert frontier
        choice = tuple(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(choice)

    mask = np.zeros((bbox_h, bbox_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _irregular_bbox_candidates(target_h: int, target_w: int) -> list[tuple[int, int]]:
    area = target_h * target_w
    candidates: list[tuple[int, int]] = []
    for extra_h in range(MAX_BBOX_EXTENSION + 1):
        for extra_w in range(MAX_BBOX_EXTENSION + 1):
            bbox_h = target_h + extra_h
            bbox_w = target_w + extra_w
            if bbox_h == target_h and bbox_w == target_w:
                continue
            if bbox_h * bbox_w <= area:
                continue
            if bbox_h + bbox_w - 1 > area:
                continue
            candidates.append((bbox_h, bbox_w))
    return candidates


def _make_object_mask(rng: np.random.Generator, target_h: int, target_w: int) -> np.ndarray:
    area = target_h * target_w
    irregular = _irregular_bbox_candidates(target_h, target_w)
    if irregular and rng.random() < IRREGULAR_PROBABILITY:
        bbox_h, bbox_w = irregular[int(rng.integers(0, len(irregular)))]
        return _grow_connected_shape(rng, bbox_h, bbox_w, area)
    return np.ones((target_h, target_w), dtype=bool)


def _place_objects(rng: np.random.Generator, inner_h: int, inner_w: int, objects: list[dict]) -> bool:
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1],
        reverse=True,
    )

    for idx in order:
        mask = objects[idx]["mask"]
        bbox_h, bbox_w = mask.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, inner_h - bbox_h + 1):
            for c0 in range(1, inner_w - bbox_w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(inner_h, r0 + bbox_h + 1)
                cc1 = min(inner_w, c0 + bbox_w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        objects[idx]["pos"] = (int(r0), int(c0))
        occupied[r0 : r0 + bbox_h, c0 : c0 + bbox_w] = True

    return True


def _render_input(
    objects: list[dict],
    width_order: list[int],
    height_order: list[int],
    width_by_color: dict[int, int],
    height_by_color: dict[int, int],
    inner_h: int,
    inner_w: int,
) -> np.ndarray:
    grid = np.zeros((inner_h + 1, inner_w + 1), dtype=int)

    c = 0
    for color in width_order:
        seg_w = int(width_by_color[color])
        grid[0, c : c + seg_w] = color
        c += seg_w

    r = 1
    for color in height_order:
        seg_h = int(height_by_color[color])
        grid[r : r + seg_h, 0] = color
        r += seg_h

    for obj in objects:
        r0, c0 = obj["pos"]
        mask = obj["mask"]
        patch = grid[1 + r0 : 1 + r0 + mask.shape[0], 1 + c0 : 1 + c0 + mask.shape[1]]
        patch[mask] = int(obj["color"])

    return grid


def _render_expected(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        r0, c0 = obj["pos"]
        mask = obj["mask"]
        patch = out[1 + r0 : 1 + r0 + mask.shape[0], 1 + c0 : 1 + c0 + mask.shape[1]]
        patch[mask] = BG

    for obj in objects:
        r0, c0 = obj["pos"]
        target_h = int(obj["target_h"])
        target_w = int(obj["target_w"])
        out[1 + r0 : 1 + r0 + target_h, 1 + c0 : 1 + c0 + target_w] = int(obj["color"])

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_objects = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    colors = [int(x) for x in rng.choice(COLORS, size=n_objects, replace=False)]
    widths = [int(x) for x in rng.choice(SIZE_VALUES, size=n_objects, replace=False)]
    heights = [int(x) for x in rng.choice(SIZE_VALUES, size=n_objects, replace=False)]

    width_by_color = {color: width for color, width in zip(colors, widths)}
    height_by_color = {color: height for color, height in zip(colors, heights)}
    inner_w = int(sum(widths) - 1)
    inner_h = int(sum(heights))

    objects: list[dict] = []
    for color in colors:
        target_h = int(height_by_color[color])
        target_w = int(width_by_color[color])
        mask = _make_object_mask(rng, target_h, target_w)
        objects.append(
            {
                "color": int(color),
                "target_h": target_h,
                "target_w": target_w,
                "mask": mask,
            }
        )

    if not _place_objects(rng, inner_h, inner_w, objects):
        return None

    width_order = [int(colors[i]) for i in rng.permutation(n_objects)]
    height_order = [int(colors[i]) for i in rng.permutation(n_objects)]

    inp = _render_input(objects, width_order, height_order, width_by_color, height_by_color, inner_h, inner_w)
    expected = _render_expected(inp, objects)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
