import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste, random_free_location
import random
from itertools import product


def _component_crop(grid: np.ndarray, coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)


def _corner_l_variants(size: int) -> list[np.ndarray]:
    k = int(size)
    assert k >= 2, "corner motif must be at least 2x2"
    variants: list[np.ndarray] = []
    for top in (True, False):
        for left in (True, False):
            mask = np.zeros((k, k), dtype=bool)
            mask[0 if top else -1, :] = True
            mask[:, 0 if left else -1] = True
            variants.append(mask)
    return variants


def _is_corner_l(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    if h != w or h < 2:
        return False
    return any(np.array_equal(mask, variant) for variant in _corner_l_variants(h))


def _completed_corner_mask(size: int, corner_size: int) -> np.ndarray:
    side = int(size)
    k = int(corner_size)
    assert side % 2 == 1, "completed motif side must be odd"
    assert side >= 2 * k + 1, "corner motifs must leave a center gap"

    mask = np.zeros((side, side), dtype=bool)
    mask[0, :k] = True
    mask[:k, 0] = True
    mask[0, side - k : side] = True
    mask[:k, side - 1] = True
    mask[side - 1, :k] = True
    mask[side - k : side, 0] = True
    mask[side - 1, side - k : side] = True
    mask[side - k : side, side - 1] = True
    return mask


def _completed_color_mask(grid: np.ndarray, color: int) -> np.ndarray:
    comps = components_mask(grid == int(color), connectivity=4)
    assert comps, "each foreground color must have at least one component"
    crops = [_component_crop(grid, coords, color) for coords in comps]

    if len(crops) == 1:
        mask = crops[0]
        h, w = map(int, mask.shape)
        if np.array_equal(mask, mask[::-1, :]) and np.array_equal(mask, mask[:, ::-1]):
            assert h == w and h % 2 == 1, "centered symmetric object must be an odd square"
            return mask.copy()

    corner_sizes = {int(mask.shape[0]) for mask in crops if _is_corner_l(mask)}
    assert len(corner_sizes) == 1, "corner motif color must expose one corner-L size"
    corner_size = corner_sizes.pop()
    assert all(
        mask.shape[0] <= corner_size and mask.shape[1] <= corner_size for mask in crops
    ), "partial corner components must fit within the exposed corner-L size"

    all_coords = np.vstack(comps)
    r0, c0, r1, c1 = bbox_coords(all_coords)
    side = max(r1 - r0 + 1, c1 - c0 + 1, 2 * corner_size + 1)
    if side % 2 == 0:
        side += 1
    return _completed_corner_mask(side, corner_size)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the centered symmetric composition from separated color motifs.

    Input structure:
    - A single background color surrounds several non-background color masks.
    - Each color is either one centered self-symmetric object, or one or more partial corner-L copies of a larger symmetric motif.
    - Corner-L motifs may show only some corners, but at least one full corner copy exposes the motif size.

    Rule:
    - Preserve a self-symmetric single object as its own centered mask.
    - For every corner-L color, complete the motif by placing that L shape in all four corners of the smallest odd square implied by the observed span and the corner size.
    - Center all completed color masks on one square canvas filled with the input background.

    Key insight:
    The scattered colored objects are not arranged by their absolute positions; each color encodes one layer of a common four-way symmetric target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    specs: list[tuple[int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        mask = _completed_color_mask(g, int(color))
        assert mask.shape[0] == mask.shape[1] and mask.shape[0] % 2 == 1
        specs.append((int(color), mask))

    assert specs, "expected at least one foreground color"
    side = max(int(mask.shape[0]) for _color, mask in specs)
    out = np.full((side, side), bg, dtype=int)
    occupied = np.zeros((side, side), dtype=bool)

    for color, mask in specs:
        local_side = int(mask.shape[0])
        offset = (side - local_side) // 2
        placed = np.zeros((side, side), dtype=bool)
        placed[offset : offset + local_side, offset : offset + local_side] = mask
        assert not np.any(occupied & placed), "completed color layers must not overlap"
        out[placed] = color
        occupied |= placed

    return out

# --------------- generator (o5x) ---------------

# Canvas and density bounds.
_GEN_INPUT_ROWS = range(13, 20)
_GEN_INPUT_COLS = range(17, 20)
_GEN_DENSITY_MIN = 0.12
_GEN_DENSITY_MAX = 0.16

# Output layer structure.
_GEN_COLOR_COUNT = 10
_GEN_OUTPUT_SIDES = (7, 11)
_GEN_CENTER_SIDE = 1
_GEN_RING_SIDE = 3
_GEN_CORNER_SIDE_START = 5
_GEN_LAYER_SIDE_STEP = 2
_GEN_CORNER_MIN_K = 2
_GEN_INCLUDE_CENTER_CHOICES = (False, True)

# Corner exposure plan.
_GEN_CORNER_IDS = (0, 1, 2, 3)
_GEN_TOP_CORNERS = {0, 1}
_GEN_LEFT_CORNERS = {0, 2}
_GEN_SHOWN_COUNTS = (1, 2, 4)
_GEN_FULL_CORNER_COUNT = 4
_GEN_ADJACENT_CORNER_PAIRS = ((0, 1), (0, 2), (1, 3), (2, 3))
_GEN_MAX_FRAGMENTS = 2

# Placement and retry limits.
_GEN_PLACEMENT_BORDER = 1
_GEN_PLACEMENT_PADDING = 1
_GEN_PLACEMENT_CONNECTIVITY = 8
_GEN_RETRY_LIMIT = 300


def _gen_corner_mask(side: int, corner_size: int) -> np.ndarray:
    side = int(side)
    k = int(corner_size)
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :k] = True
    mask[:k, 0] = True
    mask[0, side - k : side] = True
    mask[:k, side - 1] = True
    mask[side - 1, :k] = True
    mask[side - k : side, 0] = True
    mask[side - 1, side - k : side] = True
    mask[side - k : side, side - 1] = True
    return mask


def _gen_ring_mask() -> np.ndarray:
    mask = np.ones((_GEN_RING_SIDE, _GEN_RING_SIDE), dtype=bool)
    mask[1, 1] = False
    return mask


def _gen_place_centered(
    out: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int
) -> bool:
    side = int(out.shape[0])
    local_side = int(mask.shape[0])
    offset = (side - local_side) // 2
    view = occupied[offset : offset + local_side, offset : offset + local_side]
    if bool(np.any(view & mask)):
        return False
    out[offset : offset + local_side, offset : offset + local_side][mask] = int(color)
    view[mask] = True
    return True


def _gen_layer_sides(output_side: int) -> list[int]:
    sides = [_GEN_RING_SIDE]
    sides += list(range(_GEN_CORNER_SIDE_START, int(output_side) + 1, _GEN_LAYER_SIDE_STEP))
    if random.choice(_GEN_INCLUDE_CENTER_CHOICES):
        sides = [_GEN_CENTER_SIDE, *sides]
    return sides


def _gen_make_layer(side: int, color: int) -> tuple[np.ndarray, dict]:
    if int(side) == _GEN_CENTER_SIDE:
        mask = np.ones((_GEN_CENTER_SIDE, _GEN_CENTER_SIDE), dtype=bool)
        layer = {"kind": "center", "side": int(side), "k": None, "color": int(color)}
    elif int(side) == _GEN_RING_SIDE:
        mask = _gen_ring_mask()
        layer = {"kind": "ring", "side": int(side), "k": None, "color": int(color)}
    else:
        k = random.randint(_GEN_CORNER_MIN_K, (int(side) - 1) // 2)
        mask = _gen_corner_mask(side, k)
        layer = {"kind": "corner", "side": int(side), "k": k, "color": int(color)}
    return mask, layer


def _gen_build_output():
    output_side = random.choice(_GEN_OUTPUT_SIDES)
    layer_sides = _gen_layer_sides(output_side)

    bg = random.randrange(_GEN_COLOR_COUNT)
    colors = random.sample(
        [c for c in range(_GEN_COLOR_COUNT) if c != bg], len(layer_sides)
    )
    out = np.full((output_side, output_side), bg, dtype=int)
    occupied = np.zeros_like(out, dtype=bool)
    layers = []

    for side, color in zip(layer_sides, colors):
        mask, layer = _gen_make_layer(side, color)
        if not _gen_place_centered(out, occupied, mask, color):
            return None
        layers.append(layer)

    return out, bg, layers


def _gen_corner_options(side: int, corner_size: int):
    l_cells = 2 * int(corner_size) - 1
    options = []
    for shown in _GEN_SHOWN_COUNTS:
        min_frag = 1 if shown == 1 and int(side) > 2 * int(corner_size) + 1 else 0
        max_frag = (
            0
            if shown == _GEN_FULL_CORNER_COUNT
            else min(_GEN_MAX_FRAGMENTS, _GEN_FULL_CORNER_COUNT - shown)
        )
        for fragments in range(min_frag, max_frag + 1):
            options.append(
                {
                    "shown": shown,
                    "fragments": fragments,
                    "fg": shown * l_cells + fragments,
                }
            )
    return options


def _gen_dims_for_fg(fg: int) -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in _GEN_INPUT_ROWS
        for w in _GEN_INPUT_COLS
        if _GEN_DENSITY_MIN <= int(fg) / (h * w) <= _GEN_DENSITY_MAX
    ]


def _gen_choose_plan(layers):
    choices = []
    for layer in layers:
        if layer["kind"] == "center":
            choices.append([{"fg": 1}])
        elif layer["kind"] == "ring":
            choices.append([{"fg": 8}])
        else:
            choices.append(_gen_corner_options(layer["side"], layer["k"]))

    feasible = []
    for picks in product(*choices):
        fg = sum(int(pick["fg"]) for pick in picks)
        dims = _gen_dims_for_fg(fg)
        if dims:
            feasible.append((picks, dims))
    if not feasible:
        return None

    picks, dims = random.choice(feasible)
    h, w = random.choice(dims)
    return list(picks), h, w


def _gen_corner_cells(side: int, corner_size: int, corner: int):
    side = int(side)
    k = int(corner_size)
    top = int(corner) in _GEN_TOP_CORNERS
    left = int(corner) in _GEN_LEFT_CORNERS
    row = 0 if top else side - 1
    col = 0 if left else side - 1
    c_range = range(k) if left else range(side - k, side)
    r_range = range(1, k) if top else range(side - k, side - 1)
    return [(row, c) for c in c_range] + [(r, col) for r in r_range]


def _gen_corner_anchor(side: int, corner: int) -> tuple[int, int]:
    side = int(side)
    row = 0 if int(corner) in _GEN_TOP_CORNERS else side - 1
    col = 0 if int(corner) in _GEN_LEFT_CORNERS else side - 1
    return row, col


def _gen_crop_to_fg(patch: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(patch != int(bg)))
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _gen_selected_corners(shown: int) -> list[int]:
    if int(shown) == _GEN_FULL_CORNER_COUNT:
        return list(_GEN_CORNER_IDS)
    if int(shown) == 2:
        return list(random.choice(_GEN_ADJACENT_CORNER_PAIRS))
    return [random.randrange(_GEN_FULL_CORNER_COUNT)]


def _gen_corner_patch(layer, plan, bg: int) -> np.ndarray:
    side = int(layer["side"])
    k = int(layer["k"])
    color = int(layer["color"])
    patch = np.full((side, side), int(bg), dtype=int)

    selected = _gen_selected_corners(plan["shown"])
    for corner in selected:
        for r, c in _gen_corner_cells(side, k, corner):
            patch[r, c] = color

    available = [corner for corner in _GEN_CORNER_IDS if corner not in selected]
    random.shuffle(available)
    for corner in available[: int(plan["fragments"])]:
        r, c = _gen_corner_anchor(side, corner)
        patch[r, c] = color

    return _gen_crop_to_fg(patch, bg)


def _gen_patch_for_layer(layer, plan, bg: int) -> np.ndarray:
    color = int(layer["color"])
    if layer["kind"] == "center":
        return np.full((_GEN_CENTER_SIDE, _GEN_CENTER_SIDE), color, dtype=int)
    if layer["kind"] == "ring":
        patch = np.full((_GEN_RING_SIDE, _GEN_RING_SIDE), int(bg), dtype=int)
        patch[_gen_ring_mask()] = color
        return patch
    return _gen_corner_patch(layer, plan, bg)


def _gen_scatter_patches(patches, bg: int, h: int, w: int):
    grid = np.full((int(h), int(w)), int(bg), dtype=int)
    ordered = sorted(
        patches,
        key=lambda patch: (
            max(patch.shape),
            int(np.count_nonzero(patch != int(bg))),
            random.random(),
        ),
        reverse=True,
    )
    for patch in ordered:
        try:
            r, c = random_free_location(
                grid,
                patch,
                transparent=int(bg),
                border=_GEN_PLACEMENT_BORDER,
                padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, patch, r, c, transparent=int(bg), overwrite=False)
    return grid


def _gen_attempt() -> np.ndarray | None:
    built = _gen_build_output()
    if built is None:
        return None
    expected, bg, layers = built

    chosen = _gen_choose_plan(layers)
    if chosen is None:
        return None
    plans, h, w = chosen

    patches = [
        _gen_patch_for_layer(layer, plan, bg) for layer, plan in zip(layers, plans)
    ]
    inp = _gen_scatter_patches(patches, bg, h, w)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 4290ef0e input")
