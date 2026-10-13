import numpy as np

from arc_utils import Color, find_components, get_bg
import random


def _is_motif_cell(value: int, bg: int) -> bool:
    return int(value) not in {int(bg), Color.GREY}


def _paint_from_vertical_bar(g: np.ndarray, out: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = bbox
    assert c0 == c1 and r0 < r1
    c = c0
    assert c + 1 < g.shape[1]
    assert _is_motif_cell(g[r1, c + 1], bg)

    width = 0
    while c + 1 + width < g.shape[1] and _is_motif_cell(g[r1, c + 1 + width], bg):
        width += 1
    assert width > 0

    height = 0
    while r1 - height >= r0:
        strip = g[r1 - height, c + 1 : c + 1 + width]
        if not np.all((strip != bg) & (strip != Color.GREY)):
            break
        height += 1
    bar_height = r1 - r0 + 1
    assert height > 0 and bar_height % height == 0

    tile = g[r1 - height + 1 : r1 + 1, c + 1 : c + 1 + width]
    fill = np.tile(tile, (bar_height // height, 1))
    target = g[r0 : r1 + 1, c + 1 : c + 1 + width]
    assert np.all((target == bg) | (target == fill))
    out[r0 : r1 + 1, c + 1 : c + 1 + width] = fill


def _paint_from_horizontal_bar(g: np.ndarray, out: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = bbox
    assert r0 == r1 and c0 < c1
    r = r0
    assert r + 1 < g.shape[0]
    assert _is_motif_cell(g[r + 1, c0], bg)

    height = 0
    while r + 1 + height < g.shape[0] and _is_motif_cell(g[r + 1 + height, c0], bg):
        height += 1
    assert height > 0

    width = 0
    while c0 + width <= c1:
        strip = g[r + 1 : r + 1 + height, c0 + width]
        if not np.all((strip != bg) & (strip != Color.GREY)):
            break
        width += 1
    bar_width = c1 - c0 + 1
    assert width > 0 and bar_width % width == 0

    tile = g[r + 1 : r + 1 + height, c0 : c0 + width]
    fill = np.tile(tile, (1, bar_width // width))
    target = g[r + 1 : r + 1 + height, c0 : c1 + 1]
    assert np.all((target == bg) | (target == fill))
    out[r + 1 : r + 1 + height, c0 : c1 + 1] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat each colored motif from the end of a grey ruler bar to fill the rectangle beside that bar.

    Input structure
    A black-background grid contains one or more straight grey bars.  Each bar is
    either vertical, with a colored rectangular motif attached to its lower-right
    end, or horizontal, with a colored rectangular motif attached below its
    left end.
    Rule
    For a vertical grey bar, repeat the attached motif upward/downward along the
    bar's rows and paint the rectangle immediately to the bar's right.  For a
    horizontal grey bar, repeat the attached motif left/right along the bar's
    columns and paint the rectangle immediately below the bar.
    Key insight
    The grey bar gives the repeat length; the non-grey motif touching one end of
    the bar gives the tile to copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    bars = find_components(g, bg=bg, colors=[Color.GREY], connectivity=4)
    assert bars
    for bar in bars:
        r0, c0, r1, c1 = bar["bbox"]
        assert bar["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)
        if c0 == c1 and r0 < r1:
            _paint_from_vertical_bar(g, out, bar["bbox"], bg)
        elif r0 == r1 and c0 < c1:
            _paint_from_horizontal_bar(g, out, bar["bbox"], bg)
        else:
            raise AssertionError(f"grey component is not a straight bar: {bar['bbox']}")

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GEN_GRID_H, GEN_GRID_W = 10, 10
GEN_GRID_SHAPE = (GEN_GRID_H, GEN_GRID_W)
GEN_BG_COLOR = Color.BLACK
GEN_MOTIF_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Bar and motif structural ranges.
GEN_ORIENTATIONS = ("v", "h")
GEN_VERTICAL = "v"
GEN_BAR_COUNT_MIN, GEN_BAR_COUNT_MAX = 1, 2
GEN_REPEAT_MIN, GEN_REPEAT_MAX = 4, 6
GEN_AXIS_DIM_MIN, GEN_BAR_LENGTH_MAX = 1, 8
GEN_CROSS_DIM_MIN, GEN_CROSS_DIM_MIN_WHEN_AXIS_ONE, GEN_CROSS_DIM_MAX = 1, 2, 6
GEN_MOTIF_COLOR_MIN, GEN_MOTIF_COLOR_MAX = 2, 3

# Placement constraints and retry budget.
GEN_RETRY_LIMIT = 300


def _gen_motif(height: int, width: int) -> np.ndarray:
    area = height * width
    n_colors = random.randint(GEN_MOTIF_COLOR_MIN, min(GEN_MOTIF_COLOR_MAX, area))
    palette = random.sample(GEN_MOTIF_COLORS, n_colors)
    motif = np.array(random.choices(palette, k=area), dtype=int).reshape(height, width)
    for idx, color in zip(random.sample(range(area), n_colors), palette):
        motif.flat[idx] = color
    return motif


def _gen_grow_4(mask: np.ndarray) -> np.ndarray:
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def _gen_touches_grey(bar_mask: np.ndarray, grey_mask: np.ndarray) -> bool:
    return bool(np.any(bar_mask & _gen_grow_4(grey_mask)))


def _gen_choice_masks(
    orient: str,
    length: int,
    motif_h: int,
    motif_w: int,
    r: int,
    c: int,
) -> dict:
    bar_mask = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    fill_mask = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    motif_mask = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    guard_mask = np.zeros(GEN_GRID_SHAPE, dtype=bool)

    if orient == GEN_VERTICAL:
        r1 = r + length
        bar_mask[r:r1, c] = True
        fill_mask[r:r1, c + 1 : c + 1 + motif_w] = True
        motif_mask[r1 - motif_h : r1, c + 1 : c + 1 + motif_w] = True
        if c + 1 + motif_w < GEN_GRID_W:
            guard_mask[r1 - 1, c + 1 + motif_w] = True
        return {
            "bar": bar_mask,
            "fill": fill_mask,
            "motif": motif_mask,
            "input_nonbg": bar_mask | motif_mask,
            "footprint": bar_mask | fill_mask,
            "guard": guard_mask,
            "bar_slice": (slice(r, r1), c),
            "fill_slice": (slice(r, r1), slice(c + 1, c + 1 + motif_w)),
            "motif_slice": (slice(r1 - motif_h, r1), slice(c + 1, c + 1 + motif_w)),
        }

    c1 = c + length
    bar_mask[r, c:c1] = True
    fill_mask[r + 1 : r + 1 + motif_h, c:c1] = True
    motif_mask[r + 1 : r + 1 + motif_h, c : c + motif_w] = True
    if r + 1 + motif_h < GEN_GRID_H:
        guard_mask[r + 1 + motif_h, c] = True
    return {
        "bar": bar_mask,
        "fill": fill_mask,
        "motif": motif_mask,
        "input_nonbg": bar_mask | motif_mask,
        "footprint": bar_mask | fill_mask,
        "guard": guard_mask,
        "bar_slice": (r, slice(c, c1)),
        "fill_slice": (slice(r + 1, r + 1 + motif_h), slice(c, c1)),
        "motif_slice": (slice(r + 1, r + 1 + motif_h), slice(c, c + motif_w)),
    }


def _gen_dimension_options(orient: str):
    """Yield dimensions in the same order as the original nested loops."""
    for repeats in range(GEN_REPEAT_MIN, GEN_REPEAT_MAX + 1):
        for axis_dim in range(GEN_AXIS_DIM_MIN, GEN_BAR_LENGTH_MAX // repeats + 1):
            length = repeats * axis_dim
            cross_min = GEN_CROSS_DIM_MIN_WHEN_AXIS_ONE if axis_dim == 1 else GEN_CROSS_DIM_MIN
            for cross_dim in range(cross_min, GEN_CROSS_DIM_MAX + 1):
                if orient == GEN_VERTICAL:
                    yield length, repeats, axis_dim, cross_dim
                else:
                    yield length, repeats, cross_dim, axis_dim


def _gen_placement_ranges(orient: str, length: int, motif_h: int, motif_w: int) -> tuple[range, range]:
    if orient == GEN_VERTICAL:
        return range(0, GEN_GRID_H + 1 - length), range(0, GEN_GRID_W - motif_w)
    return range(0, GEN_GRID_H - 1 - motif_h), range(0, GEN_GRID_W + 1 - length)


def _gen_placement_is_valid(
    masks: dict,
    occupied: np.ndarray,
    input_nonbg: np.ndarray,
    grey_mask: np.ndarray,
    reserved_guard: np.ndarray,
) -> bool:
    if np.any(masks["footprint"] & occupied):
        return False
    if np.any(masks["input_nonbg"] & reserved_guard):
        return False
    if np.any(masks["guard"] & input_nonbg):
        return False
    return not _gen_touches_grey(masks["bar"], grey_mask)


def _gen_valid_placements(
    orient: str,
    length: int,
    motif_h: int,
    motif_w: int,
    occupied: np.ndarray,
    input_nonbg: np.ndarray,
    grey_mask: np.ndarray,
    reserved_guard: np.ndarray,
) -> list[tuple[int, int, dict]]:
    """Return all placements accepted by the current occupancy masks."""
    placements = []
    row_range, col_range = _gen_placement_ranges(orient, length, motif_h, motif_w)
    for r in row_range:
        for c in col_range:
            masks = _gen_choice_masks(orient, length, motif_h, motif_w, r, c)
            if _gen_placement_is_valid(masks, occupied, input_nonbg, grey_mask, reserved_guard):
                placements.append((r, c, masks))
    return placements


def _gen_valid_choices(
    occupied: np.ndarray,
    input_nonbg: np.ndarray,
    grey_mask: np.ndarray,
    reserved_guard: np.ndarray,
) -> list[dict]:
    choices = []
    for orient in GEN_ORIENTATIONS:
        for length, repeats, motif_h, motif_w in _gen_dimension_options(orient):
            placements = _gen_valid_placements(
                orient, length, motif_h, motif_w, occupied, input_nonbg, grey_mask, reserved_guard
            )
            if placements:
                choices.append(
                    {
                        "orient": orient,
                        "length": length,
                        "repeats": repeats,
                        "motif_h": motif_h,
                        "motif_w": motif_w,
                        "placements": placements,
                    }
                )
    return choices


def _gen_render_structure(out: np.ndarray, choice: dict, masks: dict) -> tuple:
    """Paint one full output structure and return the input-erasure record."""
    motif = _gen_motif(choice["motif_h"], choice["motif_w"])
    if choice["orient"] == GEN_VERTICAL:
        fill = np.tile(motif, (choice["repeats"], 1))
    else:
        fill = np.tile(motif, (1, choice["repeats"]))
    out[masks["bar_slice"]] = Color.GREY
    out[masks["fill_slice"]] = fill
    return masks["fill_slice"], masks["motif_slice"], motif


def _gen_build_input(out: np.ndarray, structures: list[tuple]) -> np.ndarray:
    """Erase fill regions while preserving each original motif."""
    inp = out.copy()
    for fill_slice, motif_slice, motif in structures:
        inp[fill_slice] = GEN_BG_COLOR
        inp[motif_slice] = motif
    return inp


def _gen_attempt() -> np.ndarray | None:
    out = np.zeros(GEN_GRID_SHAPE, dtype=int)
    occupied = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    input_nonbg = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    grey_mask = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    reserved_guard = np.zeros(GEN_GRID_SHAPE, dtype=bool)
    structures = []

    for _ in range(random.randint(GEN_BAR_COUNT_MIN, GEN_BAR_COUNT_MAX)):
        choices = _gen_valid_choices(occupied, input_nonbg, grey_mask, reserved_guard)
        if not choices:
            return None
        choice = random.choice(choices)
        _, _, masks = random.choice(choice["placements"])
        structures.append(_gen_render_structure(out, choice, masks))
        occupied |= masks["footprint"]
        input_nonbg |= masks["input_nonbg"]
        grey_mask |= masks["bar"]
        reserved_guard |= masks["guard"]

    inp = _gen_build_input(out, structures)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9c1e755f input")
