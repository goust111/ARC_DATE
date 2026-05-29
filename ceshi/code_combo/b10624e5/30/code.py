# Combo variant of b10624e5 (index 30)
# combo_spec: D2/2+D13/2
# Name: 图案在锚块内部重叠 + 锚块内嵌入微型源图案

from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[5]))
    from arc_utils import bbox_coords, components_mask, get_bg, paste


"""
Design analysis for the D2/2 + D13/2 combo variant
==================================================

a) Concrete behavior and trace
- The quadrant scaffold from the base task stays fixed: one colored divider row and one divider
  column split the 27x27 board into four 13x13 quadrants.
- The fused change moves every meaningful operation into anchor interiors.
- TL no longer has an external motif. Instead it contains a 4x4 red anchor whose 2x2 interior is a
  miniature overlay patch: one interior cell remains red and three interior cells are non-red.
- The copied object is not the whole visible red anchor. The source is only the 2x2 interior patch
  extracted from inside the TL anchor.
- Output behavior: each non-TL quadrant starts from a solid red square anchor; the reflected and
  resized source interior is then written into that anchor's interior, so the target keeps a red
  border while its inside becomes a red-plus-colored overlap pattern.
- Small trace on one sample:
  original task: find the non-bg component touching the TL anchor -> scale the whole component ->
  align transformed red pixels to another red block.
  combo task: find the TL anchor bbox -> slice out its 2x2 interior -> reflect it by quadrant ->
  resize it to the target interior size -> overwrite the target anchor interior.
  The difference is structural: the source is extracted from inside the anchor, and the target copy
  is also placed inside the anchor rather than beside it.

b) Complexity calibration
- The interior source is fixed at 2x2 but uses four visible cells with one red orientation marker
  and three distinct non-red colors. This keeps reflections easy to see.
- Target anchors use sizes 4..6, so target interiors use sizes 2..4. This gives visible scaling
  without creating crowded interiors or degenerate 1-pixel copies.
- At least two different target sizes appear across TR/BL/BR, and at least one is larger than TL,
  so anchor size remains an active control variable in every generated sample.

c) Hidden assumptions from the original solver that must change
- Original assumption: the source is an external connected component containing red. False now;
  the informative pattern is only the TL anchor interior.
- Original assumption: transformed copies are aligned by their red sub-bbox. False now; copies are
  written directly into the target anchor interior, so placement comes from the anchor bbox itself.
- Original assumption: target red pixels should be protected during stamping. False now; overlap is
  the point, so non-red interior pixels must be allowed to overwrite red pixels inside the anchor.
- New combo-specific interaction: because the source interior itself contains red, red now plays two
  roles at once: anchor substrate and copied "background". The solver therefore extracts the full
  interior array, but during copying it interprets red cells as exposed anchor and stamps only the
  non-red foreground cells onto each target interior.

d) Placement legality
- Every anchor is a square fully contained in one quadrant.
- TL anchor size is fixed at 4, so its interior is exactly 2x2 and always writable.
- Non-TL anchor sizes are 4..6, so each target also has a non-empty interior.
- No copied pixels leave the anchor interior, and no object crosses the divider.

e) Visual observability
- In TL, there are no exterior decorations at all; the only non-divider structure is a red frame
  with a colored miniature inside. This makes the D2-style embedded-source change obvious.
- In the outputs, copied interiors keep some red cells instead of becoming fully non-red fills.
  That visible red-under-pattern cue makes the D13-style overlap behavior obvious too.
- Because the 2x2 source is asymmetric under both LR and UD flips, the quadrant reflections can be
  inferred from a few examples.

f) Visual cleanliness
- There is only one overlap layer per target: an interior patch written on top of a red square.
- The red frame stays intact around every copied interior, so anchors remain easy to segment by eye.
- Divider color is excluded from motif colors, so the cross separator stays visually unique.

g) Color-change inferability
- There is no color remapping or numeric color logic.
- The same red/non-red interior pattern is copied directly, only with reflection and nearest-
  neighbor scaling to the target interior size.

Post-edit solver assumptions audited against the generator
---------------------------------------------------------
- `main()` assumes one red component per quadrant with a square bbox; the generator always builds
  TL as a connected framed anchor and TR/BL/BR as solid red squares.
- `main()` assumes the TL anchor interior contains both red and non-red cells; the generator uses
  exactly one red interior cell plus three distinct non-red colors.
- `main()` assumes every non-TL anchor has size at least 4 so its interior exists; the generator
  restricts target sizes to 4..6.
- `main()` assumes writing the full resized interior into the target interior is legal; the
  generator keeps all anchors fully inside their quadrants, so no clipping logic is required.
"""


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 4
SOURCE_INNER_SIZE = SOURCE_ANCHOR_SIZE - 2
TARGET_SIZE_RANGE = (4, 6)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider intersection."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Recover one red-bounded square anchor bbox from each quadrant."""
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "red components used as anchors must have square bboxes"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor component"
        anchors[key] = (int(r0), int(c0), int(side_r))
    assert set(anchors) == set(QUADS), "must detect one anchor in every quadrant"
    return anchors


def _transform_inner(inner: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    """Reflect the source interior by quadrant and resize it to the target interior."""
    motif = inner
    if qr == 1:
        motif = np.fliplr(motif)
    if qb == 1:
        motif = np.flipud(motif)
    inner_size = target_size - 2
    return _resize_nn(motif, inner_size, inner_size)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate an anchor-interior overlap pattern across quadrants using size and reflection cues.

    Input structure:
    - A full row and full column of one non-background color divide the 27x27 grid into four quadrants.
    - Each quadrant contains one red square anchor component; the top-left anchor is a 4x4 framed anchor
      whose 2x2 interior already contains a miniature red-plus-colored overlap pattern.
    - The other three quadrants contain only solid red square anchors in the input.

    Rule:
    - Detect the top-left anchor and extract only its interior 2x2 patch, not the whole anchor area.
    - For each other quadrant, reflect that interior left/right for right-side quadrants and up/down
      for bottom quadrants, then resize it to the interior size of the target anchor.
    - Treat red cells inside the transformed patch as exposed anchor substrate and stamp only the
      transformed non-red cells into the target anchor interior, preserving the outer red border.

    Key insight: the source motif is hidden inside the top-left anchor, and the copied result also lives
    inside target anchors; anchor size determines the interior scaling, while quadrant position determines
    which reflection of the embedded miniature is used.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchors(g, mid_r, mid_c)

    sr0, sc0, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left anchor must be 4x4 in this combo variant"
    src_inner = g[sr0 + 1 : sr0 + sL - 1, sc0 + 1 : sc0 + sL - 1].copy()
    assert src_inner.shape == (SOURCE_INNER_SIZE, SOURCE_INNER_SIZE), "source interior must be 2x2"
    assert np.any(src_inner == RED), "source interior must retain some red anchor color"
    assert np.any((src_inner != RED) & (src_inner != bg)), "source interior must include overlay colors"

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        assert tL >= SOURCE_ANCHOR_SIZE, "target anchors must have writable interiors"
        motif_s = _transform_inner(src_inner, qb, qr, tL)
        target_inner = out[tr0 + 1 : tr0 + tL - 1, tc0 + 1 : tc0 + tL - 1]
        target_inner[motif_s != RED] = motif_s[motif_s != RED]

    return out


def _make_source_inner() -> np.ndarray:
    """Build an asymmetric 2x2 interior with one red cell and three distinct overlay colors."""
    colors = random.sample(COLOR_POOL, 3)
    inner = np.array(
        [
            [colors[0], colors[1]],
            [colors[2], RED],
        ],
        dtype=int,
    )
    assert not np.array_equal(inner, np.fliplr(inner))
    assert not np.array_equal(inner, np.flipud(inner))
    return inner


def _build_source_patch(inner: np.ndarray) -> np.ndarray:
    """Embed the interior miniature inside a framed red 4x4 anchor."""
    patch = np.full((SOURCE_ANCHOR_SIZE, SOURCE_ANCHOR_SIZE), RED, dtype=int)
    patch[1:-1, 1:-1] = inner
    return patch


def _choose_anchor_top_left(qb: int, qr: int, size: int) -> tuple[int, int]:
    """Pick a top-left location for a size x size anchor fully inside its quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    return (
        random.randint(q_r0, q_r1 - size + 1),
        random.randint(q_c0, q_c1 - size + 1),
    )


def _overlay_inner(out: np.ndarray, top: int, left: int, inner: np.ndarray) -> None:
    """Overlay only the non-red foreground cells onto a red anchor interior."""
    target = out[top + 1 : top + 1 + inner.shape[0], left + 1 : left + 1 + inner.shape[1]]
    target[inner != RED] = inner[inner != RED]


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    """Choose target sizes with visible scaling evidence."""
    while True:
        sizes = {
            (0, 0): SOURCE_ANCHOR_SIZE,
            (0, 1): random.randint(*TARGET_SIZE_RANGE),
            (1, 0): random.randint(*TARGET_SIZE_RANGE),
            (1, 1): random.randint(*TARGET_SIZE_RANGE),
        }
        non_tl = [sizes[(0, 1)], sizes[(1, 0)], sizes[(1, 1)]]
        if max(non_tl) == SOURCE_ANCHOR_SIZE:
            continue
        if len(set(non_tl)) == 1:
            continue
        return sizes


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt."""
    src_inner = _make_source_inner()
    source_patch = _build_source_patch(src_inner)
    sizes = _sample_target_sizes()

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    anchors[(0, 0)] = _choose_anchor_top_left(0, 0, SOURCE_ANCHOR_SIZE)
    for key, size in sizes.items():
        if key == (0, 0):
            continue
        anchors[key] = _choose_anchor_top_left(key[0], key[1], size)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, source_patch, *anchors[(0, 0)])

    for key, size in sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
        _overlay_inner(out, ar, ac, _transform_inner(src_inner, key[0], key[1], size))

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, source_patch, *anchors[(0, 0)])
    for key, size in sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), *anchors[key])

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
