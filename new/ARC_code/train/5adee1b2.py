import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, object_interior
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the left-edge color legend to draw a padded rectangular halo around each matching object while preserving enclosed holes.

    Input structure:
    - The grid has a dominant background color.
    - One or more two-row markers sit on the left edge; each marker is a 2x2 block whose left column is an object color and right column is its halo color.
    - Away from the markers, every source color from the legend appears as one or more 4-connected monochrome objects.

    Rule:
    - Keep the input pixels unchanged.
    - For each source-color object, take its tight bbox expanded by one cell and clipped to the grid.
    - Recolor background pixels in that expanded region to the marker's halo color, except for background holes enclosed by the object.

    Key insight:
    - The small left-edge bars are a legend: they map object colors to the colors used for the surrounding padded bbox.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    h, w = g.shape

    markers: dict[int, int] = {}
    marker_source_cells: dict[int, set[tuple[int, int]]] = {}
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        if c0 != 0 or c1 != 1 or height != 2 or int(comp["area"]) != 4:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub != bg), "legend marker must be a full 2x2 foreground block"
        assert np.all(sub[:, 0] == sub[0, 0]), "legend source column must be monochrome"
        assert np.all(sub[:, 1] == sub[0, 1]), "legend halo column must be monochrome"
        source = int(sub[0, 0])
        halo = int(sub[0, 1])
        assert source != halo, "legend source and halo colors must differ"
        assert source not in markers, "each source color must have exactly one legend marker"

        markers[source] = halo
        marker_source_cells[source] = {tuple(map(int, coord)) for coord in comp["coords"] if int(g[tuple(coord)]) == source}

    assert markers, "expected at least one left-edge legend marker"

    for source, halo in markers.items():
        objects = []
        for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[source], return_format="dict"):
            cells = {tuple(map(int, coord)) for coord in comp["coords"]}
            if cells <= marker_source_cells[source]:
                continue
            objects.append(comp)
        assert objects, "each legend source color must label at least one non-marker object"

        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            pr0 = max(0, int(r0) - 1)
            pc0 = max(0, int(c0) - 1)
            pr1 = min(h - 1, int(r1) + 1)
            pc1 = min(w - 1, int(c1) + 1)

            patch = np.full((pr1 - pr0 + 1, pc1 - pc0 + 1), bg, dtype=int)
            coords = obj["coords"]
            patch[coords[:, 0] - pr0, coords[:, 1] - pc0] = source
            enclosed = object_interior(patch, background=bg)
            fill_mask = (patch == bg) & ~enclosed

            input_region = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
            target = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
            assert np.all(input_region[fill_mask] == bg), "halo cells must be background in the input"
            assert np.all((target[fill_mask] == bg) | (target[fill_mask] == halo)), "halo regions must not conflict"
            target[fill_mask] = halo

    return out

# --------------- generator (att1, Claude) ---------------

# Canvas and color palette.
BACKGROUND = 0
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 26
AVAILABLE_COLORS = tuple(range(1, 10))

# Legend layout.
LEGEND_SOURCE_COL, LEGEND_HALO_COL = 0, 1
LEGEND_MARKER_SIZE, LEGEND_VERTICAL_GAP = 2, 3

# Sprite geometry and acceptance.
SPRITE_SIZE, SPRITE_MIRROR_COLS = 5, 3
SPRITE_AREA_MIN, SPRITE_AREA_MAX = 12, 16
SPRITE_FILL_JITTER, MAX_INTERIOR_HOLES = 0.05, 3
SPRITE_RETRIES = 300

# Object counts and placement.
INSTANCE_MIN, INSTANCE_MAX = 1, 3
OBJECT_MIN_COL, EDGE_CLIP_MARGIN = 2, 1
EDGE_CLIP_PROB, HALO_PADDING = 0.1, 1
PLACEMENT_RETRIES = 200

# Whole-sample retry budget.
GENERATION_RETRIES = 300


def _make_sprite(color):
    """Construct one LR-symmetric 4-connected sprite matching the configured area/hole bounds."""
    target_area = random.randint(SPRITE_AREA_MIN, SPRITE_AREA_MAX)
    for _ in range(SPRITE_RETRIES):
        fill_p = target_area / (SPRITE_SIZE * SPRITE_SIZE) + random.uniform(-SPRITE_FILL_JITTER, SPRITE_FILL_JITTER)
        sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
        for r in range(SPRITE_SIZE):
            for c in range(SPRITE_MIRROR_COLS):
                if random.random() < fill_p:
                    sprite[r, c] = color
                    sprite[r, SPRITE_SIZE - 1 - c] = color
        area = int(np.count_nonzero(sprite))
        if area != target_area:
            continue
        if not is_contiguous(sprite, background=BACKGROUND, connectivity=4):
            continue
        if not (np.any(sprite[0] != BACKGROUND) and np.any(sprite[-1] != BACKGROUND) and
                np.any(sprite[:, 0] != BACKGROUND) and np.any(sprite[:, -1] != BACKGROUND)):
            continue
        padded = np.zeros((SPRITE_SIZE + 2, SPRITE_SIZE + 2), dtype=int)
        padded[1:-1, 1:-1] = sprite
        interior = object_interior(padded, background=BACKGROUND)
        holes = int(np.sum(interior & (padded == BACKGROUND)))
        if holes > MAX_INTERIOR_HOLES:
            continue
        return sprite
    return None


def _boxes_overlap(a, b):
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _draw_legend(grid, src1, halo1, src2, halo2):
    lower_top = grid.shape[0] - LEGEND_MARKER_SIZE
    upper_top = lower_top - LEGEND_VERTICAL_GAP - LEGEND_MARKER_SIZE
    grid[upper_top:upper_top + LEGEND_MARKER_SIZE, LEGEND_SOURCE_COL] = src1
    grid[upper_top:upper_top + LEGEND_MARKER_SIZE, LEGEND_HALO_COL] = halo1
    grid[lower_top:lower_top + LEGEND_MARKER_SIZE, LEGEND_SOURCE_COL] = src2
    grid[lower_top:lower_top + LEGEND_MARKER_SIZE, LEGEND_HALO_COL] = halo2
    return (upper_top, LEGEND_SOURCE_COL, grid.shape[0] - 1, LEGEND_HALO_COL)


def _visible_window(sprite, size, r0, c0):
    """Return the visible grid bbox and matching sprite slice for a possibly clipped sprite."""
    vr0, vc0 = max(0, r0), max(0, c0)
    vr1 = min(size - 1, r0 + SPRITE_SIZE - 1)
    vc1 = min(size - 1, c0 + SPRITE_SIZE - 1)
    visible = sprite[vr0 - r0:vr1 - r0 + 1, vc0 - c0:vc1 - c0 + 1]
    return (vr0, vc0, vr1, vc1), visible


def _padded_bbox(visible_box, size):
    vr0, vc0, vr1, vc1 = visible_box
    return (max(0, vr0 - HALO_PADDING), max(0, vc0 - HALO_PADDING),
            min(size - 1, vr1 + HALO_PADDING), min(size - 1, vc1 + HALO_PADDING))


def _place_objects(size, tasks, legend_box):
    """Choose non-overlapping object origins; returns placement tuples or None on retry exhaustion."""
    placed = []
    placements = []
    for sprite, sc, hc in tasks:
        for _ in range(PLACEMENT_RETRIES):
            if random.random() < EDGE_CLIP_PROB:
                r0 = random.randint(-EDGE_CLIP_MARGIN, size - SPRITE_SIZE + EDGE_CLIP_MARGIN)
                c0 = random.randint(OBJECT_MIN_COL, size - SPRITE_SIZE + EDGE_CLIP_MARGIN)
            else:
                r0 = random.randint(0, size - SPRITE_SIZE)
                c0 = random.randint(OBJECT_MIN_COL, size - SPRITE_SIZE)
            visible_box, visible = _visible_window(sprite, size, r0, c0)
            if np.count_nonzero(visible) == 0:
                continue

            pbox = _padded_bbox(visible_box, size)
            if (pbox[1] < OBJECT_MIN_COL or _boxes_overlap(pbox, legend_box) or
                    any(_boxes_overlap(pbox, eb) for eb in placed)):
                continue

            placed.append(pbox)
            placements.append((sprite, sc, hc, r0, c0))
            break
        else:
            return None
    return placements


def _render_output(grid, placements):
    """Render the paired output and return the generated input with halo-fill cells removed."""
    size = grid.shape[0]
    output = grid.copy()
    halo_mask = np.zeros((size, size), dtype=bool)

    for sprite, sc, hc, r0, c0 in placements:
        visible_box, visible = _visible_window(sprite, size, r0, c0)
        vr0, vc0, vr1, vc1 = visible_box
        output_region = output[vr0:vr1 + 1, vc0:vc1 + 1]
        output_region[visible != BACKGROUND] = sc

        pr0, pc0, pr1, pc1 = _padded_bbox(visible_box, size)
        patch = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=int)
        patch_region = patch[vr0 - pr0:vr1 - pr0 + 1, vc0 - pc0:vc1 - pc0 + 1]
        patch_region[visible != BACKGROUND] = sc

        enclosed = object_interior(patch, background=BACKGROUND)
        fill = (patch == BACKGROUND) & ~enclosed
        output_region = output[pr0:pr1 + 1, pc0:pc1 + 1]
        output_region[fill] = hc
        halo_region = halo_mask[pr0:pr1 + 1, pc0:pc1 + 1]
        halo_region[fill] = True

    inp = output.copy()
    inp[halo_mask] = BACKGROUND
    return inp, output


def _matches_oracle(inp, output):
    try:
        return np.array_equal(main(inp), output)
    except Exception:
        return False


def _gen_attempt():
    sz = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    colors = list(AVAILABLE_COLORS)
    random.shuffle(colors)
    src1, halo1, src2, halo2 = colors[:4]

    sprite1 = _make_sprite(src1)
    sprite2 = _make_sprite(src2)
    if sprite1 is None or sprite2 is None:
        return None

    n1 = random.randint(INSTANCE_MIN, INSTANCE_MAX)
    n2 = random.randint(INSTANCE_MIN, INSTANCE_MAX)
    tasks = [(sprite1, src1, halo1)] * n1 + [(sprite2, src2, halo2)] * n2
    random.shuffle(tasks)

    grid = np.zeros((sz, sz), dtype=int)
    legend_box = _draw_legend(grid, src1, halo1, src2, halo2)

    placements = _place_objects(sz, tasks, legend_box)
    if placements is None:
        return None

    inp, output = _render_output(grid, placements)
    if _matches_oracle(inp, output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
