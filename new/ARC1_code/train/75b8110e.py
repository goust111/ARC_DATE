from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg


_BACK_TO_FRONT_QUADRANTS = ((0, 0), (1, 1), (1, 0), (0, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four monochrome quadrant masks onto their common half-size canvas.

    Input structure
    The input is a 2x2 arrangement of equal-size regions on a uniform background.
    Each region contains foreground pixels of exactly one color, forming one mask
    at the output scale.

    Rule
    Start with a background output the size of one region. Stamp the masks from
    back to front by quadrant position: top-left, bottom-right, bottom-left,
    then top-right. Later masks overwrite earlier masks where they overlap.

    Key insight
    Folding all four quadrants onto the same coordinates gives the output
    canvas; the quadrant positions encode the visible layer stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width % 2 == 0, "input must split into four equal regions"

    region_h, region_w = height // 2, width // 2
    assert region_h > 0 and region_w > 0, "quadrant regions must be nonempty"

    bg = get_bg(g)
    out = np.full((region_h, region_w), bg, dtype=int)

    seen_colors: set[int] = set()
    for qr, qc in _BACK_TO_FRONT_QUADRANTS:
        region = g[qr * region_h : (qr + 1) * region_h, qc * region_w : (qc + 1) * region_w]
        foreground_colors = colors_present(region, exclude=[bg])
        assert len(foreground_colors) == 1, "each quadrant must have exactly one foreground color"

        color = foreground_colors[0]
        assert color not in seen_colors, "quadrant foreground colors must be distinct"
        seen_colors.add(color)

        out[region != bg] = color

    assert len(seen_colors) == 4, "expected four colored masks"
    return out

# --------------- generator (o5x) ---------------

# Canvas geometry and background.
_GEN_REGION_H, _GEN_REGION_W = 4, 4
_GEN_BG = 0

# Mask density, connectivity, and union-coverage constraints.
_GEN_MIN_MASK_AREA, _GEN_MAX_MASK_AREA = 4, 12
_GEN_MIN_COMPONENTS, _GEN_MAX_COMPONENTS = 1, 4
_GEN_CONNECTIVITY = 4
_GEN_MIN_UNION_COVERAGE = 13
_GEN_COMPONENT_VALUES = tuple(range(_GEN_MIN_COMPONENTS, _GEN_MAX_COMPONENTS + 1))

# Quadrant layout, layer order, fixed observed colors, and retry limit.
_GEN_QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
_GEN_BACK_TO_FRONT = ((0, 0), (1, 1), (1, 0), (0, 1))
_GEN_COLORS = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 9}
_GEN_RETRY_LIMIT = 300

_GEN_RNG = np.random.default_rng()


def _gen_build_mask_options() -> dict[tuple[int, int], list[np.ndarray]]:
    """Group every feasible 4x4 mask by (area, component_count)."""
    options: dict[tuple[int, int], list[np.ndarray]] = {
        (area, comp_count): []
        for area in range(_GEN_MIN_MASK_AREA, _GEN_MAX_MASK_AREA + 1)
        for comp_count in _GEN_COMPONENT_VALUES
    }

    for bits in range(1 << (_GEN_REGION_H * _GEN_REGION_W)):
        area = bits.bit_count()
        if not _GEN_MIN_MASK_AREA <= area <= _GEN_MAX_MASK_AREA:
            continue

        vals = np.array(
            [(bits >> i) & 1 for i in range(_GEN_REGION_H * _GEN_REGION_W)],
            dtype=np.uint8,
        ).reshape(_GEN_REGION_H, _GEN_REGION_W)
        comp_count = len(
            find_components(
                vals,
                bg=_GEN_BG,
                connectivity=_GEN_CONNECTIVITY,
                monochromatic=True,
                return_format="dict",
            )
        )
        if _GEN_MIN_COMPONENTS <= comp_count <= _GEN_MAX_COMPONENTS:
            options[(area, comp_count)].append(vals.astype(bool))

    return options


_GEN_MASK_OPTIONS = _gen_build_mask_options()


def _gen_sample_mask() -> np.ndarray:
    area = int(_GEN_RNG.integers(_GEN_MIN_MASK_AREA, _GEN_MAX_MASK_AREA + 1))
    feasible_comp_counts = [
        comp_count
        for comp_count in _GEN_COMPONENT_VALUES
        if _GEN_MASK_OPTIONS[(area, comp_count)]
    ]
    comp_count = int(_GEN_RNG.choice(feasible_comp_counts))
    bucket = _GEN_MASK_OPTIONS[(area, comp_count)]
    return bucket[int(_GEN_RNG.integers(len(bucket)))].copy()


def _gen_render_output(masks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    out = np.zeros((_GEN_REGION_H, _GEN_REGION_W), dtype=int)
    for quadrant in _GEN_BACK_TO_FRONT:
        out[masks[quadrant]] = _GEN_COLORS[quadrant]
    return out


def _gen_render_input(masks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    inp = np.zeros((_GEN_REGION_H * 2, _GEN_REGION_W * 2), dtype=int)
    for (qr, qc), mask in masks.items():
        region = inp[
            qr * _GEN_REGION_H : (qr + 1) * _GEN_REGION_H,
            qc * _GEN_REGION_W : (qc + 1) * _GEN_REGION_W,
        ]
        region[mask] = _GEN_COLORS[(qr, qc)]
    return inp


def _gen_matches_main(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    masks = {quadrant: _gen_sample_mask() for quadrant in _GEN_QUADRANTS}
    union = np.logical_or.reduce(list(masks.values()))
    if int(union.sum()) < _GEN_MIN_UNION_COVERAGE:
        return None

    out = _gen_render_output(masks)
    inp = _gen_render_input(masks)
    if not _gen_matches_main(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 75b8110e input")
