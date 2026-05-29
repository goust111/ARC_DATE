# Variant of 7ed72f31
# Dimension: D15 — 符号编码：图例与辅助信息 (symbol_instructions)
# Change: 无显式图例或符号编码。变换规则完全由红色标记的空间几何隐式定义
#   → 左上角新增 3x2 边缘图例；每行是 [形态代码色, 镜像产出色]。
#   红标记的几何形态仍决定使用横轴/竖轴/点反射，但镜像后的像素颜色不再保留原色，
#   而是统一改成图例为该形态指定的颜色。图例本身不参与反射。

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


"""
design_analysis:
  source_task:
    id: dbff022c
    solver_status: solver file not found in the allowed local resources, so this reference was skipped as instructed
  variant_goal:
    keep the original reflection geometry, but add an explicit edge legend that controls the color of the reflected copy
  a_concretization_and_trace:
    input_structure:
      - Top-left 3x2 legend is always present.
      - Each legend row is a 2-cell color pair: [shape-code, reflected-color].
      - Three shape classes exist:
        row: horizontal red line inside a sprite
        col: vertical red line inside a sprite
        point: isolated single red pixel causing point reflection
      - Main scene stays away from the legend via placement padding.
    output_structure:
      - Legend stays unchanged.
      - Original payload and red markers stay unchanged.
      - Only the reflected half is newly painted, and its pixels all use the legend's reflected-color for that marker class.
    trace_example:
      - Suppose legend rows decode to row->8, col->5, point->6.
      - Original task: a blue payload next to a horizontal red line reflects as blue.
      - New task: the same payload reflects across the same line, but the new mirrored pixels are all color 8.
      - Difference: geometry is still inferred from red marker layout, while color now depends on a separate symbol area.
  b_complexity_calibration:
    - The legend has exactly three rows, one per reflection class. This is enough structure to be visible without adding too many cases.
    - Legend row order is shuffled per sample so the solver cannot rely on row position; it must decode the left code color.
    - Every generated grid contains at least one row-axis, one col-axis, and one point-reflection marker, so all legend rows are supported by repeated evidence.
    - Reflected colors are distinct within a grid, making the new parameter visually obvious.
  c_implicit_assumption_audit:
    original_code_dependencies:
      - label all non-bg components:
          old_assumption: every non-bg component belongs to the scene
          variant_issue: the legend is also non-bg
          fix: blank the legend rectangle before component analysis
      - marker geometry determines axis:
          old_assumption: red line vs singleton red remains the only geometry signal
          variant_status: still valid
      - write mirrored pixel with original payload color:
          old_assumption: output color equals source color
          variant_issue: D15 makes output color depend on legend
          fix: write legend-mapped color instead, with fallback to original color if a legend row were malformed
      - scan the whole grid for eligible payload pixels:
          old_assumption: no unrelated same-color pixels fall into the loose bbox filters
          variant_issue: reused colors plus legend make this less safe
          fix: reflect only the actual component pixels, not arbitrary same-color pixels in the bbox
      - isolated red reflects adjacent components in-place color:
          old_assumption: point reflection preserves color
          variant_issue: point reflection must now recolor via legend
          fix: use the point entry from the legend
  d_placement_legality:
    - The output-first generator builds each sprite including its reflected half, so bounds and non-overlap are checked on the final output footprint.
    - The legend is stamped first; placement uses occupancy dilation, so sprites stay separated from the legend and from one another.
    - Input is derived by erasing only the mirror half, leaving all original markers and payload in valid positions for the solver.
  e_visual_observability:
    - The same geometry from the original task remains visible, so the reflection rule is still inferable.
    - The new effect is obvious because mirrored halves change to a class-wide color announced in the legend.
    - Since all three marker classes appear in every sample and the legend row order varies, a human can match geometry classes to legend code colors across 2-3 examples.
  f_visual_cleanliness:
    - The legend is isolated in a dedicated corner block, not mixed into the main scene.
    - Sprite spacing remains at least one dilated cell, preventing merged components and accidental overlap of mirrored output.
    - Each sprite remains monochrome on the input side, which keeps the before/after comparison readable.
  g_color_inference:
    - Only three reflected colors are active per grid, one per marker class.
    - The legend provides the missing condition explicitly: shape-code color on the left, reflected color on the right.
    - Reflected colors are forced distinct, and each class appears at least once, so the mapping is learnable from few examples.
  new_solver_assumptions_to_guarantee_in_generator:
    - legend rectangle is exactly rows 0..2 and cols 0..1
    - left legend cells use a fixed codebook {1: row, 3: col, 4: point}
    - the three legend code colors appear exactly once each
    - all three marker classes are present in the scene
    - output footprints avoid the legend area, so preserving legend pixels during writes is harmless
"""


MARK = 2
LEGEND_H, LEGEND_W = 3, 2
SHAPE_TO_CODE = {"row": 1, "col": 3, "point": 4}
CODE_TO_SHAPE = {v: k for k, v in SHAPE_TO_CODE.items()}
LEGEND_SHAPES = ("row", "col", "point")

# Canvas
GRID_MIN, GRID_MAX = 20, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 4
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 4
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 7
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.45, 0.70

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 5
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_legend(r: int, c: int) -> bool:
    return 0 <= int(r) < LEGEND_H and 0 <= int(c) < LEGEND_W


def _read_legend(grid: np.ndarray, bg: int) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for r in range(min(LEGEND_H, grid.shape[0])):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        shape = CODE_TO_SHAPE.get(left)
        if shape is None or right == int(bg) or right == int(MARK):
            continue
        mapping[shape] = right
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red markers as reflection axes, then recolor only the
    reflected copy according to the top-left legend.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    legend_map = _read_legend(g, bg)

    scene = g.copy()
    scene[:LEGEND_H, :LEGEND_W] = bg

    lab, n = label_mask(scene != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(scene[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    def paint(rr: int, cc: int, color: int) -> None:
        if inb(int(rr), int(cc)) and not _in_legend(int(rr), int(cc)):
            out[int(rr), int(cc)] = int(color)

    # 1) Marker-containing components
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in vals:
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_pixels = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload_pixels:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        for r, c, v in payload_pixels:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            paint(int(rr), int(cc), int(legend_map.get(axis[0], int(v))))

    # 2) Isolated red pixels reflect adjacent 8-connected same-color components
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(scene[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(scene == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    paint(int(rr), int(cc), int(legend_map.get("point", int(v))))

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _build_embedded_sprite(axis_type: str, rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)), max(ph, pw))
    if min_px > max_px:
        return None
    pmask = None
    for _ in range(MASK_RETRIES):
        pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    payload = None
    mirror = None
    red_pixels = None
    side_flip = rng.random() < 0.5

    if axis_type == "row":
        H, W = 2 * ph + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        if side_flip:
            payload[ph + 1 :, :] = pmask
            mirror[:ph, :] = pmask[::-1, :]
        else:
            payload[:ph, :] = pmask
            mirror[ph + 1 :, :] = pmask[::-1, :]
    elif axis_type == "col":
        H, W = ph, 2 * pw + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        red_pixels = [(r, pw) for r in range(ph)]
        if side_flip:
            payload[:, pw + 1 :] = pmask
            mirror[:, :pw] = pmask[:, ::-1]
        else:
            payload[:, :pw] = pmask
            mirror[:, pw + 1 :] = pmask[:, ::-1]
    else:
        return None

    return {"axis_type": axis_type, "payload": payload, "mirror": mirror, "red_pixels": red_pixels}


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {"axis_type": "point", "payload": payload, "mirror": mirror, "red_pixels": [red_pos]}
    return None


def _random_free_location(canvas: np.ndarray, patch_mask: np.ndarray, bg: int, rng: random.Random, *, padding: int = PLACEMENT_PADDING):
    H, W = canvas.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    occ = canvas != int(bg)
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_legend(canvas: np.ndarray, legend_rows: list[tuple[int, int]]) -> None:
    for r, (left, right) in enumerate(legend_rows):
        canvas[r, 0] = int(left)
        canvas[r, 1] = int(right)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["payload"]] = int(spr["payload_color"])
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = int(spr["mirror_color"])
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = int(MARK)


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = int(bg)


def _choose_type_counts(rng: random.Random, total_markers: int) -> dict[str, int] | None:
    counts = {"row": 1, "col": 1, "point": 1}
    while sum(counts.values()) < int(total_markers):
        choices = []
        if counts["row"] + counts["col"] < MAX_EMBEDDED:
            choices.extend(["row", "col"])
        if counts["point"] < MAX_ISOLATED:
            choices.append("point")
        if not choices:
            return None
        counts[rng.choice(choices)] += 1
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    code_colors = set(SHAPE_TO_CODE.values())
    bg_choices = [c for c in range(10) if c not in code_colors and c != MARK]
    bg = rng.choice(bg_choices)

    reflect_choices = [c for c in range(10) if c not in code_colors and c not in {bg, MARK}]
    reflect_colors = rng.sample(reflect_choices, len(LEGEND_SHAPES))
    shape_to_reflect_color = {shape: color for shape, color in zip(LEGEND_SHAPES, reflect_colors)}

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    size_lo = max(GRID_MIN, 14 + 2 * total_markers)
    if size_lo > GRID_MAX:
        return None
    N = rng.randint(size_lo, GRID_MAX)
    counts = _choose_type_counts(rng, total_markers)
    if counts is None:
        return None

    sprites: list[dict] = []
    for _ in range(counts["row"]):
        s = _build_embedded_sprite("row", rng)
        if s is None:
            return None
        sprites.append(s)
    for _ in range(counts["col"]):
        s = _build_embedded_sprite("col", rng)
        if s is None:
            return None
        sprites.append(s)
    for _ in range(counts["point"]):
        s = _build_isolated_sprite(rng)
        if s is None:
            return None
        sprites.append(s)

    payload_palette = [c for c in range(10) if c not in {bg, MARK}]
    n_payload_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(payload_palette), len(sprites)))
    chosen_payload_colors = rng.sample(payload_palette, n_payload_colors)

    used_payload_colors: set[int] = set()
    for spr in sprites:
        mirror_color = int(shape_to_reflect_color[spr["axis_type"]])
        allowed = [c for c in chosen_payload_colors if c != mirror_color]
        if not allowed:
            return None
        spr["mirror_color"] = mirror_color
        spr["payload_color"] = int(rng.choice(allowed))
        used_payload_colors.add(int(spr["payload_color"]))
    if len(used_payload_colors) < PALETTE_COLORS_MIN:
        return None

    legend_rows = [(SHAPE_TO_CODE[shape], shape_to_reflect_color[shape]) for shape in LEGEND_SHAPES]
    rng.shuffle(legend_rows)

    output = np.full((N, N), bg, dtype=int)
    _stamp_legend(output, legend_rows)

    sprites.sort(
        key=lambda spr: int((spr["payload"] | spr["mirror"]).sum()) + len(spr["red_pixels"]),
        reverse=True,
    )
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = int(r0), int(c0)
        _stamp_sprite(output, spr, int(r0), int(c0))

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / float(N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
