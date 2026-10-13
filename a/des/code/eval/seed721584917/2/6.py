import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = ("up", "down", "left", "right")

# Generator configuration.
GREY = 5
GRID_H_RANGE = (26, 33)
GRID_W_RANGE = (26, 33)
TEMPLATE_COUNT_CHOICES = (2, 3)
SEED_COUNT_RANGE = (4, 6)
EDGE_BUFFER = 6
TEMPLATE_SEPARATION = 4
OUTPUT_SEPARATION = 1
MAX_LATERAL_OFFSET = 2
MAX_EXTRA_TEMPLATE_CELLS = 8
MAX_PLACEMENT_ATTEMPTS = 180
GENERATION_ATTEMPTS = 400
TEMPLATE_DIMS = ((3, 3), (3, 5), (5, 3), (5, 5))

RNG = np.random.default_rng()


def _bg_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat, minlength=max(10, int(flat.max()) + 1))
    return int(np.argmax(counts))


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
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[dict], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _bg_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    template_candidates: list[tuple[int, list[np.ndarray]]] = []
    seeds: list[tuple[int, int, int]] = []

    for color in colors:
        comps = _components(g == color)
        assert comps, "foreground color must appear in at least one component"
        sizes = [int(len(comp)) for comp in comps]
        if all(size == 1 for size in sizes):
            for comp in comps:
                seeds.append((int(comp[0, 0]), int(comp[0, 1]), color))
        elif all(size > 1 for size in sizes):
            template_candidates.append((color, comps))
        else:
            raise AssertionError("a color cannot mix template components and seed singletons")

    assert len(template_candidates) == 1, "expected exactly one template color"
    template_color, comps = template_candidates[0]
    templates: list[dict] = []
    for comp in comps:
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        rel_coords = sorted((int(r - r0), int(c - c0)) for r, c in map(tuple, comp))
        templates.append(
            {
                "color": int(template_color),
                "bbox": bbox,
                "coords": [tuple(map(int, pt)) for pt in comp],
                "rel_coords": rel_coords,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )
    templates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    seeds.sort()
    assert templates and seeds, "expected at least one template and one seed"
    return bg, templates, seeds


def _template_distance(seed: tuple[int, int], template: dict) -> int:
    sr, sc = seed
    return min(max(abs(sr - r), abs(sc - c)) for r, c in template["coords"])


def _choose_template(seed: tuple[int, int], templates: list[dict]) -> dict:
    ranked = []
    for template in templates:
        dist = _template_distance(seed, template)
        r0, c0, _, _ = template["bbox"]
        ranked.append((dist, r0, c0, template))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _template_center(template: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = template["bbox"]
    return (0.5 * (r0 + r1), 0.5 * (c0 + c1))


def _direction_for_seed(seed: tuple[int, int], template: dict) -> str:
    sr, sc = seed
    center_r, center_c = _template_center(template)
    dr = sr - center_r
    dc = sc - center_c
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _stamp_bbox(template: dict, seed: tuple[int, int], direction: str) -> tuple[int, int, int, int]:
    sr, sc = seed
    h = int(template["h"])
    w = int(template["w"])
    if direction == "up":
        r0 = sr - (h - 1)
        c0 = sc - (w // 2)
    elif direction == "down":
        r0 = sr
        c0 = sc - (w // 2)
    elif direction == "left":
        r0 = sr - (h // 2)
        c0 = sc - (w - 1)
    else:
        r0 = sr - (h // 2)
        c0 = sc
    return (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))


def _stamp_shape(out: np.ndarray, bg: int, template: dict, seed: tuple[int, int], color: int) -> None:
    direction = _direction_for_seed(seed, template)
    r0, c0, r1, c1 = _stamp_bbox(template, seed, direction)
    h, w = out.shape
    assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "stamped copy must stay in-bounds"
    for rr, cc in template["rel_coords"]:
        nr = r0 + rr
        nc = c0 + cc
        if out[nr, nc] == bg:
            out[nr, nc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the nearest grey template around each colored seed.

    The template is chosen by minimum Chebyshev distance from the seed to any
    template pixel, with row/column tie-breaks on the template position. The
    seed direction relative to the template bbox center selects which template
    edge midpoint is anchored to the seed.
    """
    g = np.asarray(grid, dtype=int)
    bg, templates, seeds = _analyze_grid(g)

    out = g.copy()
    for sr, sc, color in seeds:
        template = _choose_template((sr, sc), templates)
        _stamp_shape(out, bg, template, (sr, sc), color)
    return out


def _rects_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0 - margin
        or br1 + margin < ar0 - margin
        or ac1 + margin < bc0 - margin
        or bc1 + margin < ac0 - margin
    )


def _random_template_shape() -> tuple[int, int, list[tuple[int, int]]]:
    h, w = TEMPLATE_DIMS[int(RNG.integers(len(TEMPLATE_DIMS)))]
    center_r = h // 2
    center_c = w // 2
    cells = {(center_r, c) for c in range(w)}
    cells.update((r, center_c) for r in range(h))
    extra_count = int(
        RNG.integers(0, min(MAX_EXTRA_TEMPLATE_CELLS, h * w - len(cells)) + 1)
    )
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(extra_count):
        frontier = [
            (r, c)
            for r, c in all_cells
            if (r, c) not in cells
            and any((r + dr, c + dc) in cells for dr, dc in DIRS4)
        ]
        if not frontier:
            break
        cell = frontier[int(RNG.integers(len(frontier)))]
        cells.add(cell)
    return h, w, sorted(cells)


def _place_templates(height: int, width: int, count: int) -> list[dict] | None:
    templates: list[dict] = []
    for _ in range(count):
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            h, w, rel_coords = _random_template_shape()
            row_hi = height - h - EDGE_BUFFER
            col_hi = width - w - EDGE_BUFFER
            if row_hi < EDGE_BUFFER or col_hi < EDGE_BUFFER:
                return None
            r0 = int(RNG.integers(EDGE_BUFFER, row_hi + 1))
            c0 = int(RNG.integers(EDGE_BUFFER, col_hi + 1))
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(
                _rects_too_close(bbox, other["bbox"], TEMPLATE_SEPARATION)
                for other in templates
            ):
                continue
            templates.append(
                {
                    "color": GREY,
                    "bbox": bbox,
                    "coords": [(r0 + rr, c0 + cc) for rr, cc in rel_coords],
                    "rel_coords": rel_coords,
                    "h": h,
                    "w": w,
                }
            )
            placed = True
            break
        if not placed:
            return None
    templates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return templates


def _candidate_seed(
    height: int, width: int, template: dict, direction: str
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = template["bbox"]
    h = int(template["h"])
    w = int(template["w"])
    center_r = r0 + h // 2
    center_c = c0 + w // 2

    if direction == "up":
        max_gap = r0 - (h - 1)
        if max_gap < 2:
            return None
        gap = int(RNG.integers(2, max_gap + 1))
        seed = (r0 - gap, center_c + int(RNG.integers(-MAX_LATERAL_OFFSET, MAX_LATERAL_OFFSET + 1)))
    elif direction == "down":
        max_gap = height - h - r1 - 1
        if max_gap < 2:
            return None
        gap = int(RNG.integers(2, max_gap + 1))
        seed = (r1 + gap, center_c + int(RNG.integers(-MAX_LATERAL_OFFSET, MAX_LATERAL_OFFSET + 1)))
    elif direction == "left":
        max_gap = c0 - (w - 1)
        if max_gap < 2:
            return None
        gap = int(RNG.integers(2, max_gap + 1))
        seed = (center_r + int(RNG.integers(-MAX_LATERAL_OFFSET, MAX_LATERAL_OFFSET + 1)), c0 - gap)
    else:
        max_gap = width - w - c1 - 1
        if max_gap < 2:
            return None
        gap = int(RNG.integers(2, max_gap + 1))
        seed = (center_r + int(RNG.integers(-MAX_LATERAL_OFFSET, MAX_LATERAL_OFFSET + 1)), c1 + gap)

    sr, sc = seed
    if not (0 <= sr < height and 0 <= sc < width):
        return None
    bbox = _stamp_bbox(template, seed, direction)
    br0, bc0, br1, bc1 = bbox
    if not (0 <= br0 <= br1 < height and 0 <= bc0 <= bc1 < width):
        return None
    if _direction_for_seed(seed, template) != direction:
        return None
    return seed, bbox


def _unique_nearest_template_index(
    seed: tuple[int, int], templates: list[dict]
) -> int | None:
    ranked = []
    for idx, template in enumerate(templates):
        dist = _template_distance(seed, template)
        r0, c0, _, _ = template["bbox"]
        ranked.append((dist, r0, c0, idx))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return int(ranked[0][3])


def _seed_specs(template_count: int) -> list[tuple[int, str]]:
    template_dirs = list(RNG.permutation(DIRECTIONS))
    specs = [(template_idx, template_dirs[template_idx]) for template_idx in range(template_count)]

    covered_dirs = {direction for _, direction in specs}
    for direction in DIRECTIONS:
        if direction in covered_dirs:
            continue
        choices = [idx for idx in range(template_count) if (idx, direction) not in specs]
        specs.append((int(choices[int(RNG.integers(len(choices)))]), direction))

    min_count = len(specs)
    target_count = int(RNG.integers(max(4, min_count), SEED_COUNT_RANGE[1] + 1))
    available = [
        (idx, direction)
        for idx in range(template_count)
        for direction in DIRECTIONS
        if (idx, direction) not in specs
    ]
    RNG.shuffle(available)
    specs.extend(available[: max(0, target_count - len(specs))])
    RNG.shuffle(specs)
    return specs


def _build_grid(height: int, width: int, templates: list[dict], seed_specs: list[tuple[int, str]]) -> np.ndarray | None:
    bg_choices = [color for color in range(10) if color != GREY]
    bg = int(bg_choices[int(RNG.integers(len(bg_choices)))])
    seed_colors = [color for color in range(10) if color not in (bg, GREY)]
    if len(seed_specs) > len(seed_colors):
        return None
    chosen_seed_colors = list(RNG.choice(seed_colors, size=len(seed_specs), replace=False))

    reserved = [template["bbox"] for template in templates]
    seed_cells: set[tuple[int, int]] = set()
    placements: list[tuple[tuple[int, int], tuple[int, int, int, int], int, dict]] = []

    ordered_specs = sorted(seed_specs, key=lambda spec: -(templates[spec[0]]["h"] * templates[spec[0]]["w"]))
    for color, (template_idx, direction) in zip(chosen_seed_colors, ordered_specs):
        template = templates[template_idx]
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            candidate = _candidate_seed(height, width, template, direction)
            if candidate is None:
                continue
            seed, out_bbox = candidate
            if seed in seed_cells:
                continue
            if any(_rects_too_close(out_bbox, other, OUTPUT_SEPARATION) for other in reserved):
                continue
            nearest_idx = _unique_nearest_template_index(seed, templates)
            if nearest_idx != template_idx:
                continue
            placements.append((seed, out_bbox, int(color), template))
            reserved.append(out_bbox)
            seed_cells.add(seed)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), bg, dtype=int)
    for template in templates:
        for r, c in template["coords"]:
            grid[r, c] = GREY
    for seed, _bbox, color, _template in placements:
        sr, sc = seed
        grid[sr, sc] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    width = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    template_count = int(TEMPLATE_COUNT_CHOICES[int(RNG.integers(len(TEMPLATE_COUNT_CHOICES)))])
    templates = _place_templates(height, width, template_count)
    if templates is None:
        return None
    seed_specs = _seed_specs(template_count)
    return _build_grid(height, width, templates, seed_specs)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
