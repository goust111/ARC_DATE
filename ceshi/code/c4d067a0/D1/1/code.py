# Variant of c4d067a0
# Dimension: D1 — 匹配条件：坐标映射方式 (match_cond)
# Change: 图例像素位置通过统一缩放因子+偏移量映射到网格位置，缩放因子由已有色块的间距与图例像素间距的比值确定... → 形状匹配嵌入

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis
# ===============
# a) Concrete behavior:
#    - Keep the original global composition: a small legend in the top-left and a larger regular array in the main area.
#    - Replace each legend "dot" with a connected 3x3 sprite. Each sprite is a unique polyomino-like shape in one color.
#    - Replace each main-area solid block with a 4x4 panel whose top-left 3x3 work area contains a bg cavity shaped like one legend
#      sprite, but the cavity may be rotated and assigned to a different lattice position than in the legend.
#    - Output fills every cavity with the color of its matching legend sprite. Two panels are already filled in the input.
#    - Example of the changed step: originally a legend cell at lattice (1,2) projected to one fixed block coordinate by scale+offset.
#      In the variant, the panel at the corresponding main lattice coordinate may instead contain the rotated cavity of some other
#      legend sprite, so location no longer determines color; the solver must identify the cavity's shape and match it to the legend.
# b) Complexity calibration:
#    - Shapes are unique up to rotation inside each sample, preventing ambiguous matches.
#    - Cavities are shuffled across panel positions and at least one cavity is rotated, so the D1 change is visible.
#    - Colors still repeat across pieces (2-4 colors total), which prevents the task from collapsing into a trivial color lookup.
# c) Old implicit assumptions that break:
#    - `singles` as legend markers: no longer valid because legend items are multi-cell components.
#    - gcd-based legend lattice inference: no longer relevant because there is no scale projection to recover.
#    - solid NxN monochrome blocks as anchors: replaced by panel templates with bg holes, so bbox/area tests for solid squares fail.
#    - color-position voting for one base offset: invalid because pieces are intentionally permuted across positions.
# d) Placement legality:
#    - Legend uses 3x3 boxes on a step-4 lattice, so components do not touch.
#    - Panels use 4x4 boxes on a step-5 lattice, so different panels do not overlap.
#    - Generator rejects layouts where the legend and panel area would collide or exceed the canvas.
#    - Two prefilled panels remain consistent with the legend, and every empty panel still exposes its cavity shape for matching.
# e) Visual observability:
#    - The rule is directly visible: the top-left area shows a catalog of small shapes, and the main area shows the same shapes as holes.
#    - Because cavities are shuffled and rotated, examples cannot be explained by the old coordinate-scaling story.
# f) Visual cleanliness:
#    - Each panel is a compact, repeated 4x4 template; holes stay inside those templates instead of floating in the background.
#    - The legend stays confined to the top-left window, and the main array begins to the right with a clean gap.
# g) Color inference:
#    - Colors are copied directly from matched legend sprites; no hidden ordering, numeric color id logic, or color arithmetic is used.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: one non-bg color dominates the main area as the panel carrier color. Generator enforces this.
#    - Assumption: every legend sprite has a unique rotation-invariant shape signature. Generator samples without repetition.
#    - Assumption: every panel is a 4x4 template with a non-empty cavity in its top-left 3x3 work area. Generator always renders that.


def _norm_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = np.array(list(coords), dtype=int)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    assert mask.shape == (3, 3)
    assert mask.any()
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(_norm_coords(np.argwhere(rot)))
    return min(variants)


def _all_orientations(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    return sorted({_norm_coords(np.argwhere(np.rot90(mask, k))) for k in range(4)})


def _connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _build_shape_library() -> list[np.ndarray]:
    seen: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for area in (3, 4, 5):
        for cells in itertools.combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in cells:
                mask[idx // 3, idx % 3] = True
            if not _connected(mask):
                continue
            if len(_all_orientations(mask)) < 2:
                continue
            sig = _shape_signature(mask)
            if sig not in seen:
                seen[sig] = mask.copy()
    lib = list(seen.values())
    lib.sort(key=lambda m: (_shape_signature(m), int(m.sum())))
    return lib


SHAPE_LIBRARY = _build_shape_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rotated shape cavities in a main-area panel array using a sprite legend in the top-left corner.

    Input structure:
    - bg = most common color.
    - Top-left corner: a staircase-like legend of small monochrome 3x3 sprites, each sprite being a distinct connected shape.
    - Main area: repeated 4x4 carrier panels; inside each panel's top-left 3x3 work area, bg cells form a cavity matching one legend shape, possibly rotated.

    Rule:
    - Match each panel cavity to the unique legend sprite with the same shape up to rotation.
    - Use the legend sprite's color to fill every cell of the matching cavity.
    - Already filled panels stay consistent; empty cavities are completed.

    Key insight: The legend no longer encodes positions to scale and project; it acts as a shape dictionary, and the main task is rotation-invariant shape matching.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    vals, cnts = np.unique(g[g != bg], return_counts=True)
    assert len(vals) >= 2, "need legend colors plus a carrier color"
    carrier = int(vals[cnts.argmax()])

    legend_window_r = 16
    legend_window_c = 12
    legend_map: dict[tuple[tuple[int, int], ...], int] = {}

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        col = int(comp["color"])
        if col == carrier:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if r1 >= legend_window_r or c1 >= legend_window_c or hh > 3 or ww > 3:
            continue
        rel = [(int(r - r0), int(c - c0)) for r, c in comp["coords"]]
        mask = np.zeros((3, 3), dtype=bool)
        for rr, cc in rel:
            mask[rr, cc] = True
        sig = _shape_signature(mask)
        assert sig not in legend_map, "legend shapes must be unique up to rotation"
        legend_map[sig] = col
    assert legend_map, "top-left legend must contain at least one sprite"

    panels: list[tuple[int, int, np.ndarray]] = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            win = g[r0 : r0 + 4, c0 : c0 + 4]
            if not (np.all(win[3, :] == carrier) and np.all(win[:, 3] == carrier)):
                continue
            work = win[:3, :3]
            shape_mask = work != carrier
            if not shape_mask.any():
                continue
            panels.append((int(r0), int(c0), shape_mask))
    assert panels, "main area must contain carrier panels"

    seen_topleft = set()
    for r0, c0, shape_mask in panels:
        if (r0, c0) in seen_topleft:
            continue
        seen_topleft.add((r0, c0))
        sig = _shape_signature(shape_mask)
        assert sig in legend_map, "every cavity must match a legend shape"
        col = legend_map[sig]
        sub = out[r0 : r0 + 3, c0 : c0 + 3]
        sub[shape_mask] = col
        out[r0 : r0 + 3, c0 : c0 + 3] = sub
    return out


# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 4
LEGEND_BOX = 3
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4

# === CFG: panel area ===
PANEL_SIZE = 4
PANEL_STEP = 5
BASE_R_MIN = 1
MAX_RETRIES = 500


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    adjacent = []
    for r, w in enumerate(widths):
        for c in range(w - 1):
            adjacent.append(((r, c), (r, c + 1)))
    return pts, adjacent


def _random_color_counts(total, k):
    counts = [1] * k
    remaining = total - k
    while remaining:
        i = random.randrange(k)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _render_shape(target: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    rr, cc = np.argwhere(mask).T
    target[top + rr, left + cc] = int(color)


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    pts, adjacent_pairs = _pick_staircase()
    n = len(pts)
    if n > len(SHAPE_LIBRARY):
        return None

    bg = random.randrange(NUM_COLORS)
    palette_max = min(PALETTE_SIZE_MAX, n)
    palette_size = random.randint(PALETTE_SIZE_MIN, palette_max)
    usable = [c for c in range(NUM_COLORS) if c != bg]
    carrier = random.choice(usable)
    piece_palette = random.sample([c for c in usable if c != carrier], palette_size)
    piece_counts = _random_color_counts(n, palette_size)
    piece_colors = []
    for col, count in zip(piece_palette, piece_counts):
        piece_colors.extend([col] * count)
    random.shuffle(piece_colors)

    legend_shapes = random.sample(SHAPE_LIBRARY, n)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    legend_right = LEGEND_ORIGIN[1] + max_c * LEGEND_STEP + LEGEND_BOX - 1
    base_c_min = legend_right + 1
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

    for idx, ((rr, cc), mask, col) in enumerate(zip(pts, legend_shapes, piece_colors)):
        lr = LEGEND_ORIGIN[0] + rr * LEGEND_STEP
        lc = LEGEND_ORIGIN[1] + cc * LEGEND_STEP
        _render_shape(out, lr, lc, mask, col)

    slot_shapes = []
    for j, (rr, cc) in enumerate(pts):
        src = perm[j]
        mask = np.rot90(legend_shapes[src], rotations[j])
        top = base_r + rr * PANEL_STEP
        left = base_c + cc * PANEL_STEP
        out[top : top + PANEL_SIZE, left : left + PANEL_SIZE] = carrier
        _render_shape(out, top, left, mask, piece_colors[src])
        slot_shapes.append(mask)

    inp = out.copy()
    if adjacent_pairs:
        keep_pair = random.choice(adjacent_pairs)
        keep = {pts.index(keep_pair[0]), pts.index(keep_pair[1])}
    else:
        keep = set(random.sample(range(n), 2))
    for j, (rr, cc) in enumerate(pts):
        if j in keep:
            continue
        top = base_r + rr * PANEL_STEP
        left = base_c + cc * PANEL_STEP
        mask = slot_shapes[j]
        rr_mask, cc_mask = np.argwhere(mask).T
        inp[top + rr_mask, left + cc_mask] = bg

    if not any(perm[j] != j for j in range(n)):
        return None
    if not any(rotations):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 D1 input")
