from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, crop_bbox, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black target rectangle with a left-right mirror of the colored source motif.

    Input structure:
    - A dominant background surrounds one fully occupied colored rectangular motif.
    - A single black rectangular region is the same shape as the colored motif.

    Rule:
    - Crop the colored motif from its tight bounding box.
    - Reflect that crop horizontally and paste it over the black rectangle.

    Key insight:
    - The black region is not erased by a fill color; it is a destination slot for the mirrored motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.BLACK, "black must be the target color, not the background"

    target_bbox = bbox(g == Color.BLACK)
    assert target_bbox is not None, "expected a black target rectangle"
    tr0, tc0, _, _ = target_bbox
    target = crop_bbox(g, target_bbox)
    assert np.all(target == Color.BLACK), "target bbox must be fully black"

    source_bbox = bbox((g != bg) & (g != Color.BLACK))
    assert source_bbox is not None, "expected a colored source motif"
    source = crop_bbox(g, source_bbox)
    assert np.all((source != bg) & (source != Color.BLACK)), "source bbox must be fully occupied"
    assert source.shape == target.shape, "source motif and target rectangle must have the same shape"

    out = g.copy()
    return paste(out, np.fliplr(source), tr0, tc0)

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SIZE, NON_BLACK_COLORS = 12, tuple(range(1, 10))
# Motif geometry and color balance.
MOTIF_HEIGHT_RANGE, MOTIF_WIDTH_RANGE = (3, 5), (4, 6)
MINORITY_PERCENT_RANGE, PERCENT_DENOMINATOR = (25, 42), 100
# Placement and retry policy.
ROW_GAP, ATTEMPT_LIMIT = 1, 300

def _gen_asymmetric_motif(h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    """Return a full two-color motif with one forced LR-asymmetric witness pair."""
    total = h * w
    lo = (total * MINORITY_PERCENT_RANGE[0] + PERCENT_DENOMINATOR - 1) // PERCENT_DENOMINATOR
    hi = total * MINORITY_PERCENT_RANGE[1] // PERCENT_DENOMINATOR
    minority_count = random.randint(lo, hi)
    minority, majority = random.sample(colors, 2)
    motif = np.full((h, w), majority, dtype=int)
    wr = random.randrange(h)
    wc = random.randrange(w // 2)
    mirror = (wr, w - 1 - wc)
    witness = (wr, wc)
    if random.choice([False, True]):
        witness, mirror = mirror, witness
    motif[witness] = minority
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in (witness, mirror)]
    for r, c in random.sample(candidates, minority_count - 1):
        motif[r, c] = minority
    return motif

def _gen_row_pair(h: int) -> tuple[int, int]:
    """Choose source/destination row starts separated by at least ROW_GAP rows."""
    starts = range(GRID_SIZE + 1 - h)
    pairs = [(sr, dr) for sr in starts for dr in starts if sr + h + ROW_GAP <= dr or dr + h + ROW_GAP <= sr]
    return random.choice(pairs)

def _render_pair(bg: int, motif: np.ndarray, src_pos: tuple[int, int], dst_pos: tuple[int, int]):
    """Build matching input/output grids for one placed source/destination pair."""
    h, w = motif.shape
    src_r, src_c = src_pos
    dst_r, dst_c = dst_pos
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(out, motif, src_r, src_c)
    paste(out, np.fliplr(motif), dst_r, dst_c)
    inp = out.copy()
    inp[dst_r : dst_r + h, dst_c : dst_c + w] = Color.BLACK
    return inp, out

def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(NON_BLACK_COLORS[0], NON_BLACK_COLORS[-1])
    colors = tuple(random.sample([c for c in NON_BLACK_COLORS if c != bg], 2))
    h = random.randint(*MOTIF_HEIGHT_RANGE)
    w = random.randint(*MOTIF_WIDTH_RANGE)
    motif = _gen_asymmetric_motif(h, w, colors)
    src_r, dst_r = _gen_row_pair(h)
    src_c = random.randrange(GRID_SIZE + 1 - w)
    dst_c = random.randrange(GRID_SIZE + 1 - w)
    inp, out = _render_pair(bg, motif, (src_r, src_c), (dst_r, dst_c))
    if np.array_equal(main(inp), out):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {ATTEMPT_LIMIT} attempts")
