import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location
import random
from itertools import product


def _target_and_legend(g: np.ndarray, bg: int) -> tuple[dict, dict[int, int]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    target = [comp for comp in comps if comp["area"] > 2]
    legends = [comp for comp in comps if comp["area"] == 2]
    assert len(target) == 1, "expected one target object larger than a legend domino"
    assert len(target) + len(legends) == len(comps), "all non-target components must be legend dominos"
    assert legends, "expected at least one color-substitution legend"

    target_crop = crop_bbox(g, target[0]["bbox"])
    assert target_crop.size == target[0]["area"], "target object must fill its bbox"

    mapping: dict[int, int] = {}
    for comp in legends:
        r0, c0, r1, c1 = comp["bbox"]
        assert r0 == r1 and c1 == c0 + 1, "legend entries must be horizontal 1x2 dominos"
        replacement = int(g[r0, c0])
        source = int(g[r0, c1])
        assert replacement != source, "legend pair colors must differ"
        assert source not in mapping, "each source color must appear in exactly one legend pair"
        assert np.any(target_crop == source), "legend source color must appear in the target"
        assert not np.any(target_crop == replacement), "legend replacement color must be outside the unrecolored target"
        mapping[source] = replacement

    preserved = set(map(int, target_crop.ravel())) - set(mapping)
    assert len(preserved) == 1, "target must have one un-rewritten base color"
    assert set(mapping).isdisjoint(mapping.values()), "legend substitutions must be non-chaining"
    return target[0], mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the large object and recolor its marked colors using the isolated two-pixel legends.

    Input structure
    - A dominant background separates one solid rectangular multicolor target object
      from one or more isolated horizontal two-pixel legend dominos.
    - In each legend domino, the right color appears in the target and the left
      color is absent from the unrecolored target.

    Rule
    - Output the bbox crop of the target object only.
    - Read every legend domino left-to-right as replacement color, source color.
    - In the crop, replace all pixels of each source color with its legend
      replacement and preserve the one target base color.

    Key insight
    - The small dominos are a color-substitution key; their positions are
      irrelevant once each right-hand color is matched to pixels inside the large
      target object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    target, mapping = _target_and_legend(g, bg)
    crop = crop_bbox(g, target["bbox"])

    out = crop.copy()
    for source, replacement in mapping.items():
        out[crop == source] = replacement
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and retry limits.
GEN_BG, GRID_SHAPE, GENERATION_RETRIES = 0, (13, 13), 300
NON_BG_COLORS = range(1, 10)

# Target rectangle and recolored blob constraints.
LEGEND_COUNTS = (2, 3)
TARGET_HEIGHTS, TARGET_WIDTHS = range(4, 8), range(4, 8)
BLOB_CELL_COUNTS, BASE_FRACTION_BOUNDS = range(3, 6), (0.56, 0.75)
BLOB_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Placement padding.
TARGET_EDGE_MARGIN, LEGEND_PADDING, LEGEND_PADDING_CONNECTIVITY = 2, 1, 8


def _gen_blob_size_options(area: int, n_blobs: int) -> list[tuple[int, ...]]:
    min_base, max_base = BASE_FRACTION_BOUNDS
    return [tuple(sizes) for sizes in product(BLOB_CELL_COUNTS, repeat=n_blobs) if min_base <= (area - sum(sizes)) / area <= max_base]


def _gen_grow_blob(free: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    starts = np.argwhere(free)
    if starts.size == 0:
        return None

    r, c = random.choice(starts).tolist()
    coords = [(int(r), int(c))]
    used = {(int(r), int(c))}
    free[int(r), int(c)] = False

    while len(coords) < size:
        frontier = []
        for rr, cc in coords:
            for dr, dc in BLOB_NEIGHBOR_OFFSETS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and (nr, nc) not in used:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        coords.append((nr, nc))
        used.add((nr, nc))
        free[nr, nc] = False

    return coords


def _gen_build_output(height: int, width: int, base: int, replacements: list[int], sizes: tuple[int, ...]) -> np.ndarray | None:
    out = np.full((height, width), base, dtype=int)
    free = np.ones((height, width), dtype=bool)

    for color, size in zip(replacements, sizes):
        coords = _gen_grow_blob(free, int(size))
        if coords is None:
            return None
        rr, cc = np.array(coords).T
        out[rr, cc] = int(color)

    return out


def _gen_choose_target(n_legends: int) -> tuple[int, int, tuple[int, ...]] | None:
    dims = []
    for height in TARGET_HEIGHTS:
        for width in TARGET_WIDTHS:
            options = _gen_blob_size_options(height * width, n_legends)
            if options:
                dims.append((height, width, options))
    if not dims:
        return None

    height, width, size_options = random.choice(dims)
    return height, width, random.choice(size_options)


def _gen_place_legends(canvas: np.ndarray, replacements: list[int], sources: list[int]) -> bool:
    pairs = list(zip(replacements, sources))
    random.shuffle(pairs)
    for replacement, source in pairs:
        legend = np.array([[replacement, source]], dtype=int)
        try:
            r, c = random_free_location(canvas, legend, transparent=GEN_BG, padding=LEGEND_PADDING, padding_connectivity=LEGEND_PADDING_CONNECTIVITY)
        except ValueError:
            return False
        paste(canvas, legend, r, c)
    return True


def _gen_attempt() -> np.ndarray | None:
    n_legends = random.choice(LEGEND_COUNTS)
    target_choice = _gen_choose_target(n_legends)
    if target_choice is None:
        return None
    height, width, sizes = target_choice

    colors = random.sample(NON_BG_COLORS, 2 * n_legends + 1)
    base = colors[0]
    sources, replacements = colors[1 : 1 + n_legends], colors[1 + n_legends :]
    expected = _gen_build_output(height, width, base, replacements, sizes)
    if expected is None:
        return None

    target = expected.copy()
    for source, replacement in zip(sources, replacements):
        target[expected == replacement] = source

    canvas = np.zeros(GRID_SHAPE, dtype=int)
    max_top = GRID_SHAPE[0] - TARGET_EDGE_MARGIN - height
    max_left = GRID_SHAPE[1] - TARGET_EDGE_MARGIN - width
    paste(canvas, target, random.randint(TARGET_EDGE_MARGIN, max_top), random.randint(TARGET_EDGE_MARGIN, max_left))
    if not _gen_place_legends(canvas, replacements, sources):
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e9b4f6fc input")
