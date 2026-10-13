import numpy as np

from arc_utils import find_components, get_bg
import random


def _is_solid_rect(g: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return comp["area"] == h * w and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == comp["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each of two aligned rectangular marker objects toward the other, filling half of the gap across the full perpendicular span.

    Input structure
    - A dominant background contains exactly two solid, same-sized monochromatic rectangles.
    - The rectangles share either the same row span or the same column span, with a background gap between their facing edges.

    Rule
    - If the rectangles are side-by-side, color the left half of the inter-object gap across every row with the left object's color, and the right half with the right object's color.
    - If one rectangle is above the other, color the upper half of the gap across every column with the upper object's color, and the lower half with the lower object's color.
    - When the gap length is odd, leave the center row/column as background.

    Key insight
    - The marker objects are endpoints of an axial corridor; their own size identifies the facing edge, while the filled corridor spans the whole grid perpendicular to that axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(comps) == 2
    assert all(_is_solid_rect(g, comp) for comp in comps)

    a, b = comps
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    assert (ar1 - ar0, ac1 - ac0) == (br1 - br0, bc1 - bc0)

    same_rows = (ar0, ar1) == (br0, br1)
    same_cols = (ac0, ac1) == (bc0, bc1)
    assert same_rows != same_cols

    out = g.copy()
    if same_rows:
        left, right = sorted(comps, key=lambda comp: comp["bbox"][1])
        _lr0, _lc0, _lr1, lc1 = left["bbox"]
        _rr0, rc0, _rr1, _rc1 = right["bbox"]
        gap = rc0 - lc1 - 1
        assert gap >= 1
        span = gap // 2
        out[:, lc1 + 1 : lc1 + 1 + span] = left["color"]
        out[:, rc0 - span : rc0] = right["color"]
    else:
        top, bottom = sorted(comps, key=lambda comp: comp["bbox"][0])
        _tr0, _tc0, tr1, _tc1 = top["bbox"]
        br0, _bc0, _br1, _bc1 = bottom["bbox"]
        gap = br0 - tr1 - 1
        assert gap >= 1
        span = gap // 2
        out[tr1 + 1 : tr1 + 1 + span, :] = top["color"]
        out[br0 - span : br0, :] = bottom["color"]
    return out

# --------------- generator (att2) ---------------

# Layout ranges and bounds.
ALIGNMENTS = ("H", "V")
RECT_PERP_BOUNDS = (1, 4)
RECT_AXIAL_VALUES = range(1, 3)
GAP_VALUES = range(6, 17)
AXIAL_BOUNDS = (14, 22)
MARGIN = 1
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300
VALID_AXIAL_GAP_PAIRS = tuple(
    (rect_axial, gap)
    for rect_axial in RECT_AXIAL_VALUES
    for gap in GAP_VALUES
    if AXIAL_BOUNDS[0] <= 2 * rect_axial + gap + 2 * MARGIN <= AXIAL_BOUNDS[1]
)


def _choose_layout() -> tuple[str, int, int, int]:
    alignment = random.choice(ALIGNMENTS)
    rect_perp = random.randint(*RECT_PERP_BOUNDS)
    rect_axial, gap = random.choice(VALID_AXIAL_GAP_PAIRS)
    return alignment, rect_perp, rect_axial, gap


def _choose_colors() -> tuple[int, int, int]:
    bg = random.choice(PALETTE)
    color_a, color_b = random.sample([color for color in PALETTE if color != bg], 2)
    return bg, color_a, color_b


def _render_horizontal(rect_perp: int, rect_axial: int, gap: int, bg: int, color_a: int, color_b: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the horizontal input/output pair; vertical cases are its transpose."""
    perp = rect_perp + 2 * MARGIN
    axial = 2 * rect_axial + gap + 2 * MARGIN
    left_c0 = MARGIN
    left_c1 = left_c0 + rect_axial - 1
    right_c1 = axial - MARGIN - 1
    right_c0 = right_c1 - rect_axial + 1
    row0 = MARGIN
    row1 = row0 + rect_perp - 1

    out = np.full((perp, axial), bg, dtype=int)
    out[row0 : row1 + 1, left_c0 : left_c1 + 1] = color_a
    out[row0 : row1 + 1, right_c0 : right_c1 + 1] = color_b

    span = gap // 2
    out[:, left_c1 + 1 : left_c1 + 1 + span] = color_a
    out[:, right_c0 - span : right_c0] = color_b

    inp = out.copy()
    inp[:, left_c1 + 1 : right_c0] = bg
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    alignment, rect_perp, rect_axial, gap = _choose_layout()
    bg, color_a, color_b = _choose_colors()
    inp, out = _render_horizontal(rect_perp, rect_axial, gap, bg, color_a, color_b)
    if alignment == "V":
        inp = inp.T.copy()
        out = out.T.copy()

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d968ffd4 input")
