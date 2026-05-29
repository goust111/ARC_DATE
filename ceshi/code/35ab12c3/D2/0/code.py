# Variant of 35ab12c3
# Dimension: D2 — 路径方向类型 (path)
# Change: straight_ray + diagonal_path——顶点间通过水平、垂直或严格45度对角线连接，要求dr==0或... → 仅正交路径（禁止对角线）
#
# Design analysis for the D2 variant
# a) Concrete behavior:
#    - Solver side: same-color vertices are still angle-sorted into a cycle/polyline
#      scaffold, but an edge is drawable only when the endpoints share a row or a
#      column. Any diagonal candidate edge that the original task would have drawn
#      is now skipped.
#    - Visible consequence: a 3-vertex right-corner shape such as
#      {(0,0),(0,4),(4,4)} becomes an open L in the output. In the original task,
#      the closing 45-degree edge from (4,4) back to (0,0) would also be drawn.
#    - Distractors are also adapted: a color may now contain diagonal-only pairs
#      that remain unchanged, which gives the human solver direct evidence that
#      diagonal alignment is no longer sufficient.
# b) Complexity calibration:
#    - Keeping only axis-aligned line/rectangle shapes would be too degenerate
#      because those samples also fit the original task. To make the rule visible,
#      the generator includes right-corner triplets whose omitted diagonal closure
#      distinguishes this variant.
#    - A sample is accepted only if it contains at least one such witness shape or
#      a diagonal-only distractor, so the dataset does not collapse back to the
#      original behavior.
# c) Implicit-assumption audit of main():
#    - _segment_cells and the shadow translation logic remain valid because they do
#      not require diagonal movement; they only step by the sign of dr/dc.
#    - The only rule change is the edge-admissibility predicate. Therefore every
#      place that used the old H/V/diag test must now use an orth-only test.
#    - The seed-neighbor search still uses 8-neighborhood, because shadow anchor
#      pixels may still be diagonal from the anchor vertex. This is unchanged.
# d) Placement legality:
#    - Local patches are built from sparse input markers (vertices + one optional
#      shadow marker), then the solver itself generates the filled output patch.
#      Placement checks use the filled patch, so expanded orthogonal segments and
#      translated shadows cannot overlap neighboring objects.
# e) Visual observability:
#    - Corner-triplet groups visibly become L-shapes instead of closed triangles,
#      and diagonal-only distractors stay unchanged. Both cues are easy to see from
#      2-4 training examples without relying on color IDs.
# f) Visual cleanliness:
#    - Shapes are spaced with a 1-cell buffer using output occupancy, and shadows
#      only use non-collinear offsets so they do not smear over their own source
#      edges.
# g) Color inference:
#    - Colors keep the original one-color-per-group policy; no hidden color mapping
#      is introduced.
# h) New solver assumptions and generator guarantees:
#    - main() now assumes every intended drawable shape edge appears as an orthogonal
#      consecutive pair in the angle-sorted cycle. The generator guarantees this by
#      using only axis-aligned lines, rectangles, and square-corner triplets.

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_orth(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0)


def _aligned_orth_delta(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0)


def _aligned_diag45_delta(dr: int, dc: int) -> bool:
    return dr != 0 and abs(dr) == abs(dc)


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_idxs.append(i)
                        break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_orth(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_orth(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
CORNER_SIDE_RANGE = (2, 8)

SHAPE_WEIGHTS = np.array([0.25, 0.30, 0.45], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 6)
DISTRACTOR_RETRIES = 80

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_corner_triplet() -> list[tuple[int, int]]:
    side = int(np.random.randint(*CORNER_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


_SHAPE_BUILDERS = [
    ("line", _build_line, False),
    ("rectangle", _build_rectangle, False),
    ("corner_triplet", _build_corner_triplet, True),
]


def _make_vertices(force_witness: bool = False) -> tuple[list[tuple[int, int]], bool]:
    if force_witness:
        witness_idxs = [i for i, (_, _, witness) in enumerate(_SHAPE_BUILDERS) if witness]
        idx = witness_idxs[int(np.random.randint(len(witness_idxs)))]
    else:
        weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
        idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    _, builder, witness = _SHAPE_BUILDERS[idx]
    return builder(), witness


def _cycle_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cycle = _angle_cycle(vertices)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _drawable_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for a, b in _cycle_edges(vertices):
        if _aligned_orth(a[0], a[1], b[0], b[1]):
            edges.append((a, b))
    return edges


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _drawable_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step_dir(a[0], a[1], b[0], b[1]) for a, b in _drawable_edges(vertices)]


def _diag_only_edge_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _cycle_edges(vertices):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if not _aligned_diag45_delta(dr, dc):
            continue
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    preserve_witness: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    blocked_diag_cells = _diag_only_edge_cells(vertices) - set(vertices) if preserve_witness else set()
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if pos in blocked_diag_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(vertices)
    if shadow is not None:
        _, (dr, dc) = shadow
        coords.extend((r + dr, c + dc) for r, c in vertices)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    force_witness: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    vertices, witness = _make_vertices(force_witness=force_witness)
    shape_cells = _shape_cells(vertices)
    if not shape_cells:
        return None

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, shape_cells, preserve_witness=force_witness)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    return inp, out, witness


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dr = coords[j][0] - coords[i][0]
            dc = coords[j][1] - coords[i][1]
            if _aligned_orth_delta(dr, dc):
                return None
    if count >= 2:
        has_diag_pair = False
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dr = coords[j][0] - coords[i][0]
                dc = coords[j][1] - coords[i][1]
                if _aligned_diag45_delta(dr, dc):
                    has_diag_pair = True
                    break
            if has_diag_pair:
                break
        if not has_diag_pair:
            return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray, bool]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy(), False
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy(), True
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy(), False


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    has_variant_witness = False
    witness_shape_idx = int(np.random.randint(shape_count))

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(
                int(shape_color),
                shadow_map.get(idx),
                force_witness=(idx == witness_shape_idx),
            )
            if patch is None:
                continue
            patch_in, patch_out, witness = patch
            if _try_place(inp, out, patch_in, patch_out):
                has_variant_witness = has_variant_witness or witness
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        patch_in, patch_out, witness = _make_distractor_patch(distractor_color)
        if not _try_place(inp, out, patch_in, patch_out):
            return None
        has_variant_witness = has_variant_witness or witness

    if not has_variant_witness:
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
