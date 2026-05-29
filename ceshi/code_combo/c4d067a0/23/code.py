# Combo variant of c4d067a0 (index 23)
# combo_spec: D1/1+D2/4
# Name: 形状匹配嵌入+对角线标记图例

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis
# ===============
# a) Concrete behavior:
#    - Keep the original coarse layout: a compact legend in the top-left and a repeated structure in the main area.
#    - Replace each explicit legend sprite with only two same-colored singleton markers placed at opposite corners of a small box.
#      Those markers no longer mean "fill the whole rectangle"; instead, they imply a connected staircase path running along that
#      diagonal inside the box. Example: a 3x3 TL-BR marker pair implies a 5-cell staircase shape, not a solid 3x3 square.
#    - Replace the original scaled block projection with carrier panels. Each main-area panel is a 5x5 carrier square whose top-left
#      4x4 work area contains either a bg cavity or a prefilled colored shape. The shape is one of the legend-implied staircase masks,
#      possibly rotated and assigned to a different lattice slot than the legend entry that defined it.
#    - So the old pipeline was: legend dot -> infer scale/offset -> paint corresponding square block. The fused pipeline is:
#      diagonal marker pair -> infer hidden staircase shape -> match that shape up to rotation against panel cavities -> fill cavity.
# b) Complexity calibration:
#    - Using only full rectangles would give very few distinct signatures, so the diagonal pair is converted into a staircase path;
#      that yields a richer but still very regular shape family visible from a few examples.
#    - Each sample uses 3-4 legend entries with distinct rotation-invariant signatures, preventing ambiguous shape-color matches.
#    - Main-area slots are permuted, and at least one slot is rotated, so the task cannot collapse back to same-position copying.
# c) Old implicit assumptions that break:
#    - Legend items are no longer direct colored cells or explicit sprites; `main()` must reconstruct a hidden shape from two markers.
#    - The old gcd/offset projection logic is invalid because main-area placement is intentionally shuffled across slots.
#    - Solid monochrome NxN blocks are gone; the main-area objects are carrier panels with internal non-carrier masks.
#    - Color voting for one global base is no longer meaningful; the solver must identify panels by repeated carrier borders and then
#      classify each exposed mask by shape signature.
# d) Placement legality:
#    - Legend symbols live on a step-6 slot lattice and each symbol fits inside a 4x4 box, so singleton markers never touch.
#    - Panels live on a separate step-6 lattice, each occupying 5x5 with a 4x4 work area, so panels do not overlap.
#    - The generator places the panel area strictly to the right of the legend and checks that every panel fits within the canvas.
# e) Visual observability:
#    - A human can see only two colored endpoints per legend symbol, then compare training outputs to notice that the filled panel
#      shapes are diagonal staircases stretched between those endpoints.
#    - Because slot positions are permuted and some shapes are rotated, the stable explanatory rule is shape reconstruction + matching,
#      not location transfer.
# f) Visual cleanliness:
#    - Legend markers stay confined to the top-left and are grouped by slot spacing rather than scattered across the board.
#    - Panels use one dominant carrier color, so cavities and prefilled shapes are easy to isolate visually.
# g) Color inference:
#    - Marker color is copied directly to every matched cavity; there is no color remapping, ordering by numeric ids, or arithmetic.
# h) Fusion metric to watch for degeneration:
#    - Metric 1: number of legend entries where implied shape area is greater than 2 marker cells. This must always hold, otherwise
#      the diagonal-marker source would degenerate into explicit single-cell clues.
#    - Metric 2: number of panel slots whose cavity signature differs from the legend slot at the same lattice coordinate. This must
#      be positive, otherwise the task would drift back toward coordinate-based placement.
# i) New solver assumptions and generator guarantees:
#    - Assumption: one non-bg color dominates as the panel carrier. Generator enforces a single carrier color over all panels.
#    - Assumption: each legend color appears exactly twice in the legend window and defines one diagonal bounding box. Generator enforces it.
#    - Assumption: each legend-implied staircase signature is unique within a sample. Generator samples only distinct signatures.
#    - Assumption: every real panel is a 5x5 window with carrier last row/last column and a non-empty 4x4 work mask. Generator always renders that.


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: hidden shape / panel ===
WORK_SIZE = 4
PANEL_SIZE = 5
PANEL_STEP = 6

# === CFG: legend slots ===
LEGEND_ORIGIN = (1, 1)
LEGEND_SLOT_STEP = 6
LEGEND_ROWS = (2, 3)
LEGEND_COLS = 2
LEGEND_FILLED = (3, 4)
LEGEND_WINDOW_R = 20
LEGEND_WINDOW_C = 14

# === CFG: placement ===
BASE_R_MIN = 1
BASE_C_MIN = 12
MAX_RETRIES = 600


def _norm_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = np.array(list(coords), dtype=int)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    assert mask.shape == (WORK_SIZE, WORK_SIZE)
    assert mask.any()
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(_norm_coords(np.argwhere(rot)))
    return min(variants)


def _staircase_mask(height: int, width: int, diag: str) -> np.ndarray:
    assert 2 <= height <= WORK_SIZE
    assert 2 <= width <= WORK_SIZE
    assert diag in ("TLBR", "TRBL")
    anchors = []
    for r in range(height):
        c = int(round(r * (width - 1) / (height - 1)))
        if diag == "TRBL":
            c = width - 1 - c
        anchors.append(int(c))
    mask = np.zeros((height, width), dtype=bool)
    mask[0, anchors[0]] = True
    for r in range(height - 1):
        c0 = anchors[r]
        c1 = anchors[r + 1]
        lo, hi = sorted((c0, c1))
        mask[r, lo : hi + 1] = True
        mask[r + 1, c1] = True
    return mask


def _shape_canvas(spec: tuple[int, int, str]) -> np.ndarray:
    height, width, diag = spec
    canvas = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)
    raw = _staircase_mask(height, width, diag)
    canvas[:height, :width] = raw
    return canvas


def _marker_points(spec: tuple[int, int, str]) -> list[tuple[int, int]]:
    height, width, diag = spec
    if diag == "TLBR":
        return [(0, 0), (height - 1, width - 1)]
    return [(0, width - 1), (height - 1, 0)]


def _build_shape_specs() -> list[tuple[int, int, str]]:
    by_sig: dict[tuple[tuple[int, int], ...], tuple[int, int, str]] = {}
    for height in range(2, WORK_SIZE + 1):
        for width in range(2, WORK_SIZE + 1):
            for diag in ("TLBR", "TRBL"):
                spec = (height, width, diag)
                sig = _shape_signature(_shape_canvas(spec))
                if sig not in by_sig:
                    by_sig[sig] = spec
    specs = list(by_sig.values())
    specs.sort(key=lambda spec: (_shape_canvas(spec).sum(), spec[0], spec[1], spec[2]))
    return specs


SHAPE_SPECS = _build_shape_specs()


def _enumerate_slot_patterns() -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for rows in LEGEND_ROWS:
        for total in LEGEND_FILLED:
            def rec(prefix: list[int], remaining: int, r: int) -> None:
                if r == rows:
                    if remaining == 0 and max(prefix) == LEGEND_COLS:
                        patterns.append(tuple(prefix))
                    return
                left = rows - r - 1
                lo = max(1, remaining - left * LEGEND_COLS)
                hi = min(LEGEND_COLS, remaining - left)
                for width in range(lo, hi + 1):
                    rec(prefix + [width], remaining - width, r + 1)
            rec([], total, 0)
    return patterns


SLOT_PATTERNS = _enumerate_slot_patterns()


def _pick_slots() -> list[tuple[int, int]]:
    widths = random.choice(SLOT_PATTERNS)
    return [(r, c) for r, width in enumerate(widths) for c in range(width)]


def _legend_map_from_grid(g: np.ndarray, bg: int, carrier: int) -> dict[tuple[tuple[int, int], ...], int]:
    legend = g[:LEGEND_WINDOW_R, :LEGEND_WINDOW_C]
    markers: list[tuple[int, int, int]] = []
    for pts in components_mask(legend != bg, connectivity=4):
        if len(pts) != 1:
            continue
        r, c = pts[0]
        col = int(legend[r, c])
        if col == carrier:
            continue
        markers.append((int(r), int(c), col))
    assert markers, "legend window must contain singleton markers"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in markers:
        by_color.setdefault(int(col), []).append((int(r), int(c)))

    legend_map: dict[tuple[tuple[int, int], ...], int] = {}
    for col, pts in by_color.items():
        assert len(pts) == 2, "each legend color must have exactly two markers"
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        r0 = min(rows)
        c0 = min(cols)
        height = max(rows) - r0 + 1
        width = max(cols) - c0 + 1
        assert 2 <= height <= WORK_SIZE and 2 <= width <= WORK_SIZE, "legend box must fit the work area"
        rel = {(r - r0, c - c0) for r, c in pts}
        if rel == {(0, 0), (height - 1, width - 1)}:
            diag = "TLBR"
        else:
            assert rel == {(0, width - 1), (height - 1, 0)}, "legend markers must be diagonal corners"
            diag = "TRBL"
        sig = _shape_signature(_shape_canvas((height, width, diag)))
        assert sig not in legend_map, "legend shapes must be unique up to rotation"
        legend_map[sig] = int(col)
    return legend_map


def _panel_masks(g: np.ndarray, carrier: int) -> list[tuple[int, int, np.ndarray]]:
    h, w = g.shape
    panels: list[tuple[int, int, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(h - PANEL_SIZE + 1):
        for c0 in range(w - PANEL_SIZE + 1):
            win = g[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
            if not (np.all(win[PANEL_SIZE - 1, :] == carrier) and np.all(win[:, PANEL_SIZE - 1] == carrier)):
                continue
            shape_mask = win[:WORK_SIZE, :WORK_SIZE] != carrier
            if not shape_mask.any():
                continue
            key = (int(r0), int(c0))
            if key in seen:
                continue
            seen.add(key)
            panels.append((int(r0), int(c0), shape_mask.copy()))
    assert panels, "main area must contain carrier panels"
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a panel array by reconstructing hidden staircase shapes from diagonal legend markers.

    Input structure:
    - `bg` is the dominant background color.
    - The top-left legend contains only singleton marker pairs: each non-carrier color appears exactly twice, and the two pixels sit at opposite corners of a small box.
    - Those marker pairs do not draw the whole symbol directly; they encode a staircase-shaped polyomino inside a 4x4 work area.
    - The main area contains repeated 5x5 carrier panels whose top-left 4x4 work areas expose either bg cavities or already colored versions of those staircase shapes.

    Rule:
    - Group legend markers by color, recover each pair's box size and diagonal direction, and rebuild the implied staircase shape.
    - Scan the carrier panels, extract the exposed shape mask from each work area, and match it to the unique legend shape up to rotation.
    - Fill every matching cavity with the legend color while leaving already correct panels unchanged.

    Key insight:
    - The legend is no longer a coordinate map and it is not an explicit sprite dictionary either; it is a sparse code.
    - Two diagonal endpoints define a hidden shape, and solving the task requires reconstructing that shape and recognizing the same mask after permutation and rotation in the panel field.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)

    vals, cnts = np.unique(g[g != bg], return_counts=True)
    assert len(vals) >= 2, "need legend colors plus a carrier color"
    carrier = int(vals[cnts.argmax()])

    legend_map = _legend_map_from_grid(g, bg, carrier)
    panels = _panel_masks(g, carrier)

    for r0, c0, shape_mask in panels:
        sig = _shape_signature(shape_mask)
        assert sig in legend_map, "every panel mask must match a legend shape"
        col = legend_map[sig]
        work = out[r0 : r0 + WORK_SIZE, c0 : c0 + WORK_SIZE]
        work[shape_mask] = col
        out[r0 : r0 + WORK_SIZE, c0 : c0 + WORK_SIZE] = work
    return out


def _render_markers(target: np.ndarray, top: int, left: int, spec: tuple[int, int, str], color: int) -> None:
    for rr, cc in _marker_points(spec):
        target[top + rr, left + cc] = int(color)


def _render_panel(target: np.ndarray, top: int, left: int, carrier: int, mask: np.ndarray, color: int) -> None:
    target[top : top + PANEL_SIZE, left : left + PANEL_SIZE] = int(carrier)
    work = target[top : top + WORK_SIZE, left : left + WORK_SIZE]
    work[mask] = int(color)
    target[top : top + WORK_SIZE, left : left + WORK_SIZE] = work


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    slots = _pick_slots()
    n = len(slots)
    if n > len(SHAPE_SPECS):
        return None

    bg = random.randrange(NUM_COLORS)
    usable = [c for c in range(NUM_COLORS) if c != bg]
    carrier = random.choice(usable)
    piece_colors = random.sample([c for c in usable if c != carrier], n)
    legend_specs = random.sample(SHAPE_SPECS, n)

    max_r = max(r for r, _ in slots)
    max_c = max(c for _, c in slots)
    legend_right = LEGEND_ORIGIN[1] + max_c * LEGEND_SLOT_STEP + WORK_SIZE - 1
    base_c_min = max(BASE_C_MIN, legend_right + 2)
    r_hi = size - PANEL_SIZE - max_r * PANEL_STEP
    c_hi = size - PANEL_SIZE - max_c * PANEL_STEP
    if r_hi < BASE_R_MIN or c_hi < base_c_min:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(base_c_min, c_hi)

    perm = list(range(n))
    while n > 1 and all(i == p for i, p in enumerate(perm)):
        random.shuffle(perm)

    rotations = [random.randrange(4) for _ in range(n)]
    if all(k == 0 for k in rotations):
        rotations[random.randrange(n)] = random.randint(1, 3)

    out = np.full((size, size), bg, dtype=int)

    for (rr, cc), spec, col in zip(slots, legend_specs, piece_colors):
        top = LEGEND_ORIGIN[0] + rr * LEGEND_SLOT_STEP
        left = LEGEND_ORIGIN[1] + cc * LEGEND_SLOT_STEP
        _render_markers(out, top, left, spec, col)

    slot_masks: list[np.ndarray] = []
    for j, (rr, cc) in enumerate(slots):
        src = perm[j]
        mask = np.rot90(_shape_canvas(legend_specs[src]), rotations[j])
        top = base_r + rr * PANEL_STEP
        left = base_c + cc * PANEL_STEP
        _render_panel(out, top, left, carrier, mask, piece_colors[src])
        slot_masks.append(mask)

    inp = out.copy()
    keep = set(random.sample(range(n), 2))
    for j, (rr, cc) in enumerate(slots):
        if j in keep:
            continue
        top = base_r + rr * PANEL_STEP
        left = base_c + cc * PANEL_STEP
        work = inp[top : top + WORK_SIZE, left : left + WORK_SIZE]
        work[slot_masks[j]] = bg
        inp[top : top + WORK_SIZE, left : left + WORK_SIZE] = work

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
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
