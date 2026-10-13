from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
ROW_MAJOR_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
REF_OVERLAY_ORDER = ((0, 0), (1, 1), (1, 0), (0, 1))

GEN_RETRIES = 500
GEN_SEEN_INPUTS: set[bytes] = set()
GEN_SEEN_OUTPUTS: set[bytes] = set()


def _present_colors(grid: np.ndarray, exclude: tuple[int, ...] = (BLACK,)) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) not in set(exclude))


def _signature(grid: np.ndarray) -> bytes:
    shape = np.array(grid.shape, dtype=np.int16).tobytes()
    return shape + grid.astype(np.int16, copy=False).tobytes()


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    full_rows = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r, :])
        if len(vals) == 1 and int(vals[0]) != BLACK:
            full_rows.append((r, int(vals[0])))

    full_cols = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if len(vals) == 1 and int(vals[0]) != BLACK:
            full_cols.append((c, int(vals[0])))

    candidate_colors = {color for _, color in full_rows} & {color for _, color in full_cols}
    assert len(candidate_colors) == 1, "expected exactly one separator color"
    sep_color = next(iter(candidate_colors))

    sep_rows = [r for r, color in full_rows if color == sep_color]
    sep_cols = [c for c, color in full_cols if color == sep_color]
    assert len(sep_rows) == 1 and len(sep_cols) == 1, "expected one separator row and one separator col"

    sep_r = int(sep_rows[0])
    sep_c = int(sep_cols[0])
    top_h = sep_r
    left_w = sep_c
    bottom_h = grid.shape[0] - sep_r - 1
    right_w = grid.shape[1] - sep_c - 1
    assert top_h == bottom_h and left_w == right_w, "separator must split the grid into four equal quadrants"
    assert top_h > 0 and left_w > 0, "quadrants must be non-empty"
    return sep_color, sep_r, sep_c


def _extract_quadrant_masks(grid: np.ndarray) -> tuple[int, dict[tuple[int, int], tuple[int, np.ndarray]]]:
    sep_color, sep_r, sep_c = _find_separator(grid)
    qh, qw = sep_r, sep_c

    quadrant_data: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    seen_colors: set[int] = set()
    for qr, qc in QUADRANTS:
        r0 = qr * (qh + 1)
        c0 = qc * (qw + 1)
        region = grid[r0 : r0 + qh, c0 : c0 + qw]
        colors = _present_colors(region)
        assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
        color = int(colors[0])
        assert color not in seen_colors, "quadrant colors must be distinct"
        seen_colors.add(color)
        mask = region == color
        assert np.any(mask), "each quadrant must contain a non-empty mask"
        quadrant_data[(qr, qc)] = (color, mask)

    assert len(seen_colors) == 4, "expected four distinct quadrant colors"
    return sep_color, quadrant_data


def _render_layers(
    ordered_layers: list[tuple[int, np.ndarray]],
    shape: tuple[int, int],
    erase_mask: np.ndarray | None = None,
    erase_to_black: bool = True,
    erase_color: int | None = None,
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for color, mask in ordered_layers:
        out[mask] = int(color)
    if erase_mask is not None:
        if erase_to_black:
            out[erase_mask] = BLACK
        else:
            assert erase_color is not None
            out[erase_mask] = int(erase_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the four quadrant masks onto one canvas, but treat the separator-colored one as an erasing stencil.

    Input structure:
    - A single non-black separator row and a single non-black separator column of the same color split the grid into four equal quadrants.
    - Each quadrant has a black background and exactly one monochrome mask color, with four distinct quadrant colors overall.
    - Exactly one quadrant mask uses the same color as the separators.

    Rule:
    - Produce an output canvas the size of one quadrant on a black background.
    - Among the three masks whose colors differ from the separator color, sort them by pixel count from largest to smallest and stamp them onto the shared canvas in that order; later masks overwrite earlier ones.
    - Apply the separator-colored mask last as an etching stencil: every cell it covers becomes black instead of taking its color.

    Key insight:
    - The separator color is not another visible layer. It marks the unique mask that erases, while the visible stack order is determined only by mask area, not by quadrant position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, quadrant_data = _extract_quadrant_masks(g)
    qh, qw = next(iter(quadrant_data.values()))[1].shape

    visible_layers: list[tuple[int, int, np.ndarray]] = []
    erase_mask: np.ndarray | None = None
    for color, mask in quadrant_data.values():
        area = int(mask.sum())
        if color == sep_color:
            assert erase_mask is None, "expected exactly one separator-colored quadrant mask"
            erase_mask = mask
        else:
            visible_layers.append((area, color, mask))

    assert erase_mask is not None, "missing separator-colored erase mask"
    assert len(visible_layers) == 3, "expected exactly three visible masks"
    areas = [area for area, _, _ in visible_layers]
    assert len(set(areas)) == 3, "visible mask areas must be pairwise distinct"

    ordered = [(color, mask) for area, color, mask in sorted(visible_layers, key=lambda item: -item[0])]
    return _render_layers(ordered, (qh, qw), erase_mask=erase_mask, erase_to_black=True)


def _partition_axis(length: int) -> list[tuple[int, int]]:
    cut1 = random.randint(2, length - 4)
    cut2 = random.randint(cut1 + 2, length - 2)
    return [(0, cut1), (cut1, cut2), (cut2, length)]


def _sample_block(zone: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    r0, r1, c0, c1 = zone
    zone_h = r1 - r0
    zone_w = c1 - c0
    max_h = min(zone_h, 2 + int(zone_h >= 4))
    max_w = min(zone_w, 2 + int(zone_w >= 4))
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    rr = random.randint(r0, r1 - h)
    cc = random.randint(c0, c1 - w)

    mask = np.zeros(shape, dtype=bool)
    mask[rr : rr + h, cc : cc + w] = True
    return mask


def _build_feature_blocks(qh: int, qw: int) -> dict[str, np.ndarray]:
    row_bands = _partition_axis(qh)
    col_bands = _partition_axis(qw)
    zones = {
        "u0": (*row_bands[0], *col_bands[0]),
        "all": (*row_bands[0], *col_bands[1]),
        "u1": (*row_bands[0], *col_bands[2]),
        "p01": (*row_bands[1], *col_bands[0]),
        "erase": (*row_bands[1], *col_bands[1]),
        "p12": (*row_bands[1], *col_bands[2]),
        "u2": (*row_bands[2], *col_bands[0]),
        "p02": (*row_bands[2], *col_bands[1]),
        "eraser_only": (*row_bands[2], *col_bands[2]),
    }
    return {name: _sample_block(zone, (qh, qw)) for name, zone in zones.items()}


def _compose_masks(blocks: dict[str, np.ndarray]) -> list[np.ndarray]:
    def combine(names: tuple[str, ...]) -> np.ndarray:
        mask = np.zeros_like(next(iter(blocks.values())), dtype=bool)
        for name in names:
            mask |= blocks[name]
        return mask

    return [
        combine(("all", "p01", "p02", "erase", "u0")),
        combine(("all", "p01", "p12", "erase", "u1")),
        combine(("all", "p02", "p12", "erase", "u2")),
        combine(("erase", "eraser_only")),
    ]


def _assemble_input(
    content_masks: list[np.ndarray],
    erase_mask: np.ndarray,
    content_colors: list[int],
    sep_color: int,
    erase_quadrant: tuple[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, np.ndarray]]]:
    qh, qw = erase_mask.shape
    grid = np.full((2 * qh + 1, 2 * qw + 1), BLACK, dtype=int)
    grid[qh, :] = sep_color
    grid[:, qw] = sep_color

    quadrant_masks: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    remaining_quadrants = [q for q in QUADRANTS if q != erase_quadrant]
    random.shuffle(remaining_quadrants)
    assignments = list(zip(remaining_quadrants, zip(content_colors, content_masks)))
    assignments.append((erase_quadrant, (sep_color, erase_mask)))

    for (qr, qc), (color, mask) in assignments:
        r0 = qr * (qh + 1)
        c0 = qc * (qw + 1)
        grid[r0 : r0 + qh, c0 : c0 + qw][mask] = color
        quadrant_masks[(qr, qc)] = (color, mask)
    return grid, quadrant_masks


def _correct_output_from_quadrants(
    quadrant_masks: dict[tuple[int, int], tuple[int, np.ndarray]],
    sep_color: int,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], int, np.ndarray]], np.ndarray]:
    visible = []
    erase_mask = None
    for quadrant, (color, mask) in quadrant_masks.items():
        if color == sep_color:
            erase_mask = mask
        else:
            visible.append((quadrant, color, mask))
    assert erase_mask is not None

    ordered = sorted(visible, key=lambda item: -int(item[2].sum()))
    out = _render_layers([(color, mask) for _, color, mask in ordered], erase_mask.shape, erase_mask=erase_mask)
    return out, ordered, erase_mask


def _semantic_shortcuts_differ(
    quadrant_masks: dict[tuple[int, int], tuple[int, np.ndarray]],
    sep_color: int,
    expected: np.ndarray,
) -> bool:
    _, correct_order, erase_mask = _correct_output_from_quadrants(quadrant_masks, sep_color)
    base_layers = [(color, mask) for _, color, mask in correct_order]

    no_erase = _render_layers(base_layers, expected.shape)
    if np.array_equal(no_erase, expected):
        return False

    paint_eraser = _render_layers(base_layers, expected.shape, erase_mask=erase_mask, erase_to_black=False, erase_color=sep_color)
    if np.array_equal(paint_eraser, expected):
        return False

    for perm in itertools.permutations(base_layers):
        if list(perm) == base_layers:
            continue
        alt = _render_layers(list(perm), expected.shape, erase_mask=erase_mask)
        if np.array_equal(alt, expected):
            return False

    correct_quadrants = [quadrant for quadrant, _, _ in correct_order]
    row_major_visible = [q for q in ROW_MAJOR_ORDER if q in quadrant_masks and quadrant_masks[q][0] != sep_color]
    ref_visible = [q for q in REF_OVERLAY_ORDER if q in quadrant_masks and quadrant_masks[q][0] != sep_color]
    if correct_quadrants == row_major_visible or correct_quadrants == ref_visible:
        return False

    return True


def _valid_candidate(
    inp: np.ndarray,
    expected: np.ndarray,
    quadrant_masks: dict[tuple[int, int], tuple[int, np.ndarray]],
    sep_color: int,
) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    visible_colors = [color for color, _ in quadrant_masks.values() if color != sep_color]
    for color in visible_colors:
        if not np.any(expected == color):
            return False

    if not np.any(expected == BLACK):
        return False

    if not _semantic_shortcuts_differ(quadrant_masks, sep_color, expected):
        return False

    inp_sig = _signature(inp)
    out_sig = _signature(expected)
    if inp_sig in GEN_SEEN_INPUTS or out_sig in GEN_SEEN_OUTPUTS:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(6, 11)
    qw = random.randint(6, 11)
    blocks = _build_feature_blocks(qh, qw)
    masks = _compose_masks(blocks)

    content_masks = masks[:3]
    erase_mask = masks[3]
    content_areas = [int(mask.sum()) for mask in content_masks]
    if len(set(content_areas)) != 3:
        return None

    sep_color = random.choice(COLOR_POOL)
    content_colors = random.sample([c for c in COLOR_POOL if c != sep_color], 3)
    erase_quadrant = random.choice(QUADRANTS)
    inp, quadrant_masks = _assemble_input(content_masks, erase_mask, content_colors, sep_color, erase_quadrant)

    expected, _, _ = _correct_output_from_quadrants(quadrant_masks, sep_color)
    if not _valid_candidate(inp, expected, quadrant_masks, sep_color):
        return None

    GEN_SEEN_INPUTS.add(_signature(inp))
    GEN_SEEN_OUTPUTS.add(_signature(expected))
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
