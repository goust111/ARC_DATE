# Combo variant of e12f9a14 (index 48)
# combo_spec: D5/0+D11/1+D20/0
# Name: 网格对齐+多缺口+固定两框架

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Keep the original beam physics, but freeze the source geometry into a visible mirrored pair.
#    Example horizontal layout: the left frame opens at its two east corners and the right frame
#    opens at its two west corners. Compared with the original "one frame may contribute one gap",
#    each frame now emits two diagonal beams at once, so the four total beams create two symmetric
#    diagonal crossings above and below the shared row. Those crossings then redirect according to
#    the original collision rules, producing a regular hourglass-like figure rather than an ad hoc
#    multi-source scatter.
# b) Complexity calibration:
#    The change is visually obvious because every sample always has exactly two frames and exactly
#    two gaps per frame. Restricting the layout to one mirrored family prevents the task from
#    exploding into arbitrary four-ray chaos, while still making "track all gaps" necessary.
# c) Assumption audit:
#    The original solver already supports multiple gaps and multiple beams, so the collision engine
#    does not need new physics. What does change is the input contract: we now rely on exactly two
#    frames, mirrored inward-facing corner gaps, and lattice-aligned placement. The wrapper main()
#    checks those assumptions explicitly before delegating to the original beam simulator.
# d) Placement legality:
#    The generator only uses layouts whose 4x4 frames fit fully inside the grid, whose top-left
#    coordinates lie on a visible lattice, and whose major-axis separation has the odd-parity
#    runway needed for geometric diagonal crossings instead of illegal same-cell diagonal crashes.
# e) Visual observability:
#    Across 2-4 examples, a human can see three stable cues without hidden color semantics:
#    always two frames, always two gaps per frame, and the frames sit on one shared lattice row or
#    column with mirrored inward openings. Those cues explain why the output collision points are
#    symmetric and predictable.
# f) Visual cleanliness:
#    No extra markers or decorative holes are added. Only four source beams are present, and the
#    generator rejects any sample with too little painted output, so the result stays structured
#    instead of becoming an unreadable overdraw.
# g) Color inference:
#    Colors keep their original roles: one shared frame color, two distinct block colors, and each
#    beam inherits its own block color. The rule is carried by geometry alone, not by color ids.
#
# New solver assumptions introduced here, and how generate_input() guarantees them:
# - exactly two framed 2x2 blocks are present;
# - each frame has exactly two structural gaps;
# - the two frames lie on a lattice anchored at ALIGN_ORIGIN with one shared step size;
# - the gap pairs are mirrored inward-facing corner pairs, giving two guaranteed symmetric
#   diagonal collisions.


_BASE_PATH = Path("/home/lixinyu/snap/new/arc2_solver_select/e12f9a14.py")
_SPEC = importlib.util.spec_from_file_location("_e12f9a14_base", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_find_blocks_2x2 = _BASE._find_blocks_2x2
_infer_frame_color = _BASE._infer_frame_color
_frame_gaps_for_block = _BASE._frame_gaps_for_block
_base_main = _BASE.main

ALIGN_ORIGIN = 3
LAYOUT_OPTIONS = (
    (4, 2),
    (4, 3),
    (4, 4),
    (6, 2),
    (6, 3),
)

BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}

GEN_RETRIES = 64
MIN_OUTPUT_DELTA = 28

_H_LEFT_GAPS = frozenset({(0, 3), (3, 3)})
_H_RIGHT_GAPS = frozenset({(0, 0), (3, 0)})
_V_TOP_GAPS = frozenset({(3, 0), (3, 3)})
_V_BOTTOM_GAPS = frozenset({(0, 0), (0, 3)})


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: frozenset[tuple[int, int]],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != bg and c != frame_color]
    block_colors = list(rng.choice(palette, size=2, replace=False))
    return bg, frame_color, block_colors


def _layout_choice(
    rng: np.random.Generator,
) -> tuple[int, str, int, int, list[tuple[int, int]], list[frozenset[tuple[int, int]]]]:
    step, span = LAYOUT_OPTIONS[int(rng.integers(len(LAYOUT_OPTIONS)))]
    n = 2 * ALIGN_ORIGIN + 4 + step * span
    orientation = str(rng.choice(["horizontal", "vertical"]))

    if orientation == "horizontal":
        row_idx = int(rng.integers(1, span))
        r = ALIGN_ORIGIN + row_idx * step
        positions = [(r, ALIGN_ORIGIN), (r, ALIGN_ORIGIN + step * span)]
        gaps = [_H_LEFT_GAPS, _H_RIGHT_GAPS]
    else:
        col_idx = int(rng.integers(1, span))
        c = ALIGN_ORIGIN + col_idx * step
        positions = [(ALIGN_ORIGIN, c), (ALIGN_ORIGIN + step * span, c)]
        gaps = [_V_TOP_GAPS, _V_BOTTOM_GAPS]
    return n, orientation, step, span, positions, gaps


def _analyze_variant_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "variant expects a square grid"

    bg = get_bg(g)
    blocks = _find_blocks_2x2(g, bg)
    assert len(blocks) == 2, "variant expects exactly two 2x2 blocks"

    frames: list[dict[str, object]] = []
    for color, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        gaps = sorted(_frame_gaps_for_block(g, bg, frame_col, br, bc))
        assert len(gaps) == 2, "each frame must have exactly two structural gaps"
        r0, c0 = br - 1, bc - 1
        local = frozenset((gr - r0, gc - c0) for gr, gc in gaps)
        frames.append(
            {
                "color": int(color),
                "frame": (r0, c0),
                "gaps_local": local,
            }
        )

    rows = {frame["frame"][0] for frame in frames}
    cols = {frame["frame"][1] for frame in frames}

    orientation: str
    axis_delta: int
    ordered: list[dict[str, object]]
    expected_pairs: tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]

    if len(rows) == 1 and len(cols) == 2:
        ordered = sorted(frames, key=lambda frame: frame["frame"][1])
        orientation = "horizontal"
        axis_delta = ordered[1]["frame"][1] - ordered[0]["frame"][1]
        expected_pairs = (_H_LEFT_GAPS, _H_RIGHT_GAPS)
    elif len(cols) == 1 and len(rows) == 2:
        ordered = sorted(frames, key=lambda frame: frame["frame"][0])
        orientation = "vertical"
        axis_delta = ordered[1]["frame"][0] - ordered[0]["frame"][0]
        expected_pairs = (_V_TOP_GAPS, _V_BOTTOM_GAPS)
    else:
        raise AssertionError("frames must share one lattice row or one lattice column")

    assert ordered[0]["gaps_local"] == expected_pairs[0], "first frame must use the inward mirrored corner pair"
    assert ordered[1]["gaps_local"] == expected_pairs[1], "second frame must use the inward mirrored corner pair"

    matches: list[tuple[int, int]] = []
    coords = [frame["frame"][0] for frame in ordered] + [frame["frame"][1] for frame in ordered]
    for step, span in LAYOUT_OPTIONS:
        if axis_delta != step * span:
            continue
        if any((coord - ALIGN_ORIGIN) % step != 0 for coord in coords):
            continue
        expected_n = 2 * ALIGN_ORIGIN + 4 + step * span
        if g.shape != (expected_n, expected_n):
            continue
        matches.append((step, span))
    assert matches, "frames must sit on the configured lattice"

    step, span = matches[0]
    return {
        "bg": bg,
        "orientation": orientation,
        "step": step,
        "span": span,
        "frames": ordered,
        "collision_count": 2,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four beams from a mirrored pair of lattice-aligned frames and keep the original collision physics.

    Input structure:
    - The grid contains exactly two framed monochrome 2x2 blocks, both using one shared frame color.
    - Each 4x4 frame has exactly two corner gaps rather than one, so each block emits two beams at once.
    - The two frames lie on the same visible lattice row or column, and their gap pairs face inward as a mirror pair.

    Rule:
    - Every gap emits a beam in the standard outward direction for that border position, giving four simultaneous beams.
    - Beam growth and collision handling are unchanged from the source task: beams extend through background, 90-degree
      meetings leave a one-cell hole and turn toward the bisector, and 45-degree meetings merge into a zigzag.
    - The lattice-aligned mirrored placement makes the four beams meet in two predictable symmetric collisions before
      the redirected traces continue toward the boundary.

    Key insight:
    - The new reasoning burden is to recognize the fixed two-frame, two-gap, mirrored source layout. Once those four
      seeds are identified, the rest is the same collision engine as the original task, now producing a regular,
      highly symmetric pattern.
    """
    _analyze_variant_input(np.asarray(grid, dtype=int))
    return _base_main(np.asarray(grid, dtype=int))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg, frame_color, block_colors = _pick_colors(rng)
    n, _orientation, _step, _span, positions, gap_sets = _layout_choice(rng)

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gaps in zip(positions, block_colors, gap_sets):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    try:
        signature = _analyze_variant_input(grid)
        out = main(grid)
    except Exception:
        return None

    if signature["collision_count"] != 2:
        return None
    if np.count_nonzero(out != grid) < MIN_OUTPUT_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused e12f9a14 input")
