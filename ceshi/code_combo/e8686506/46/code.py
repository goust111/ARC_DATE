# Combo variant of e8686506 (index 46)
# combo_spec: D6/2+D11/3
# Name: 边界变色+全局颜色变换
#
# Design analysis
# ---------------
# spec:
#   preserved_core:
#     - input is still a mostly-uniform background with scattered monochrome fragments
#     - the hidden target before postprocessing is still a dense left-right symmetric portrait rectangle
#     - output dimensions, frame locking, and propagation-only packing stay those of the source task
#   fused_transform:
#     - step 1: assemble the original rectangle in native fragment colors
#     - step 2: recolor every cell that 4-touches a differently colored assembled neighbor to fixed grey
#     - step 3: apply one global color substitution to the whole seam-marked rectangle
#     - the global substitution is uniform by color, so it also changes the grey seam cells
#   chosen_mapping:
#     - base fragment colors are restricted to {1,2,3,4}
#     - seam grey is fixed to 5 and is reserved from the input/background
#     - final global recolor is the fixed cycle 1->2, 2->3, 3->4, 4->5, 5->1
#
# a) Concrete behavior and trace
#   - Source flow:
#       input fragments -> assemble dense LR-symmetric rectangle -> return assembled colors
#   - Fused flow:
#       input fragments -> assemble same rectangle -> mark color-contact seams grey -> rotate all colors once
#   - Small trace:
#       base assembly:
#         1 1 2
#         1 3 2
#       after seam step:
#         1 5 5
#         5 5 5
#       after global cycle:
#         2 1 1
#         1 1 1
#     The key interaction is that seam grey does not survive as grey; it becomes 1 in the final output.
#   - Generator impact:
#       scattered pieces must still be cut from the pre-seam base rectangle. If we cut from either postprocess,
#       the input would contain artificial seam/global-transform colors and the task would stop being "assemble first".
#
# b) Complexity calibration
#   - The assembly puzzle stays intact; only the post-assembly rendering changes.
#   - The global recolor is intentionally a single fixed 5-cycle instead of an instance-specific permutation.
#     That keeps the second step inferable from 2-4 examples while still being genuinely different from D6 alone.
#   - The generator forces a visible seam and keeps at least three source colors visible after seam marking, so the
#     composed rule is observable rather than collapsing to "everything almost the same color".
#
# c) Implicit-assumption audit for main()
#   - Piece extraction, type grouping, total-area dimension inference, and frame detection still depend only on the
#     scattered input fragments. That remains valid because the input is cut from the untransformed base rectangle.
#   - _build_requirements() must continue using original fragment colors. Injecting seam grey or globally shifted
#     colors into req would break the source task's anchors because both postprocesses happen strictly after packing.
#   - _contact_seam_mask() relies on the source-task guarantee that different pieces of the same color never 4-touch.
#     The generator preserves that, so color inequality is still a valid proxy for "piece boundary".
#   - The new global recolor assumes the seam output uses only colors 1..5. The generator guarantees base colors
#     are exactly 1..4 and reserves 5 for the seam layer, so the lookup is total on generated instances.
#   - Returning the raw assembled board or even the seam-marked board would be incorrect; main() must apply both
#     postprocesses in the specified order because the second step intentionally consumes the first one's grey cells.
#
# d) Placement legality
#   - Scattering stays source-compatible: the frame fragment group keeps its output-relative geometry inside one
#     reserved bounding box, and all non-frame patches are placed outside it with Chebyshev padding 1.
#   - Because the input is still derived from the base rectangle, no scattered object uses seam grey or final-cycle
#     colors created only by postprocessing, so components never merge due to the added transform logic.
#
# e) Visual observability
#   - A solver can first notice that output seams form a special network that was not present in the input pieces,
#     then also notice that every surviving non-seam color is shifted by the same fixed one-step cycle.
#   - The rule does not depend on hidden ordering by numeric value inside a sample; the only fixed palette fact the
#     human needs is a repeated dataset-level cycle among visible ARC colors.
#
# f) Visual cleanliness
#   - The seam layer is still a single coherent structural marker, and the final color cycle preserves that marker as
#     one output color class instead of introducing many unrelated recolorings.
#   - Background is chosen outside {1,2,3,4,5}, so input/background, source pieces, seam effect, and final output
#     remain visually separated.
#
# g) Color-change inferability
#   - Only five colors participate in the postprocess, and the mapping is always the same 5-cycle.
#   - The generator rejects cases where too few source colors survive seam marking, because those examples would
#     weaken the evidence that the recolor is a true global substitution applied after the seam step.
#
# Post-change solver assumptions that the generator now guarantees
# - Base assembled rectangles use exactly colors {1,2,3,4}; seam postprocessing adds 5, and nothing else appears.
# - The seam step changes at least one cell and leaves at least three of the four base colors visible afterward.
# - The final output differs from both the raw assembly and the seam-only intermediate on every accepted sample.

from collections import Counter
import importlib.util
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


_BASE_COLORS = (1, 2, 3, 4)
_SEAM_COLOR = 5
_FINAL_COLORS = frozenset({1, 2, 3, 4, 5})
_BG_CHOICES = (0, 6, 7, 8, 9)
_GLOBAL_CYCLE = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1}
_GLOBAL_LOOKUP = np.arange(10, dtype=int)
for _src, _dst in _GLOBAL_CYCLE.items():
    _GLOBAL_LOOKUP[_src] = _dst


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    bar_colors: list[tuple[int, int]] = []
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        is_singleton = len(t["coords"]) == 1
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> None:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True
    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def _assemble_base_rectangle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    return _assemble(types, H, W, req, disallow_touch)


def _contact_seam_mask(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(board.shape, dtype=bool)
    diff_ud = board[1:, :] != board[:-1, :]
    diff_lr = board[:, 1:] != board[:, :-1]
    mask[1:, :] |= diff_ud
    mask[:-1, :] |= diff_ud
    mask[:, 1:] |= diff_lr
    mask[:, :-1] |= diff_lr
    return mask


def _apply_seam_transform(board: np.ndarray, seam_color: int = _SEAM_COLOR) -> np.ndarray:
    out = np.array(board, copy=True)
    out[_contact_seam_mask(board)] = seam_color
    return out


def _apply_global_color_cycle(board: np.ndarray) -> np.ndarray:
    present = set(map(int, np.unique(board)))
    assert present <= _FINAL_COLORS, "global recolor expects only seam-stage colors 1..5"
    return _GLOBAL_LOOKUP[board]


def _apply_combo_postprocess(board: np.ndarray) -> np.ndarray:
    seam = _apply_seam_transform(board)
    return _apply_global_color_cycle(seam)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered fragments into the hidden rectangle, then apply two ordered color postprocesses.

    Input structure:
    - The background is the most frequent color; every other connected component is a native fragment of the pre-transform rectangle.
    - One or two frame fragments still remain in output-relative alignment, while the remaining interior fragments are scattered away from that frame box.
    - Only the source colors 1..4 appear on fragments; seam grey and the final recolor effect exist only after assembly.

    Rule:
    - First reconstruct the same dense left-right symmetric portrait rectangle as in the original task by packing all non-background pieces under the mirror and anchor constraints.
    - Next recolor every cell on a 4-neighbor contact between different assembled colors to grey.
    - Finally apply the fixed global color cycle 1->2, 2->3, 3->4, 4->5, 5->1 to the whole rectangle, including those grey seam cells.

    Key insight:
    - The geometric solve is unchanged from the source task; the novelty is that the visible output comes from a strictly ordered two-step postprocess, so the seam layer must be inferred before the final whole-board color substitution.
    """
    base = _assemble_base_rectangle(grid)
    return _apply_combo_postprocess(base).astype(int)


# --------------- generator (o5x) ---------------

# Reuse the D6 generator for stable source-task legality, then recolor the whole instance
# into a fixed 1..4 source palette before applying the combo postprocess.
_MIN_SEAM_CELLS = 6
_MIN_SEAM_STAGE_COLORS = 4
_MIN_SURVIVING_BASE_COLORS = 3
_GEN_MAX_RETRIES = 240
_D6_SOURCE_MOD = None
_D6_SOURCE_PATH = "/home/lixinyu/snap/new/ceshi/code/e8686506/D6/2/code.py"


def _load_d6_source_module():
    global _D6_SOURCE_MOD
    if _D6_SOURCE_MOD is None:
        spec = importlib.util.spec_from_file_location("e8686506_d6_source", _D6_SOURCE_PATH)
        assert spec is not None and spec.loader is not None, "failed to load D6 source task module"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _D6_SOURCE_MOD = mod
    return _D6_SOURCE_MOD


def _recolor_instance_to_fixed_palette(inp: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_colors = sorted(map(int, np.unique(base)))
    assert len(src_colors) == len(_BASE_COLORS), "recoloring expects exactly four base colors"
    dst_colors = list(_BASE_COLORS)
    random.shuffle(dst_colors)
    piece_map = {src: dst for src, dst in zip(src_colors, dst_colors)}
    bg = random.choice(_BG_CHOICES)

    recolored_inp = np.full(inp.shape, bg, dtype=int)
    for src, dst in piece_map.items():
        recolored_inp[inp == src] = dst

    recolored_base = np.empty(base.shape, dtype=int)
    for src, dst in piece_map.items():
        recolored_base[base == src] = dst
    return recolored_inp, recolored_base


def _transformed_output_ok(base_out: np.ndarray, seam_out: np.ndarray, final_out: np.ndarray) -> bool:
    if np.array_equal(base_out, seam_out):
        return False
    if np.array_equal(seam_out, final_out):
        return False
    if np.array_equal(base_out, final_out):
        return False
    seam_cells = int(np.count_nonzero(seam_out == _SEAM_COLOR))
    if seam_cells < _MIN_SEAM_CELLS:
        return False
    seam_colors = set(map(int, np.unique(seam_out)))
    final_colors = set(map(int, np.unique(final_out)))
    if _SEAM_COLOR not in seam_colors:
        return False
    if len(seam_colors) < _MIN_SEAM_STAGE_COLORS:
        return False
    if len(final_colors) < _MIN_SEAM_STAGE_COLORS:
        return False
    surviving_base_colors = {col for col in _BASE_COLORS if np.any(seam_out == col)}
    if len(surviving_base_colors) < _MIN_SURVIVING_BASE_COLORS:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    src = _load_d6_source_module()
    try:
        raw_inp = src.generate_input()
        raw_base = _assemble_base_rectangle(raw_inp)
    except Exception:
        return None
    if len(np.unique(raw_base)) != len(_BASE_COLORS):
        return None

    inp, base_out = _recolor_instance_to_fixed_palette(raw_inp, raw_base)
    seam_out = _apply_seam_transform(base_out)
    final_out = _apply_global_color_cycle(seam_out)
    if not _transformed_output_ok(base_out, seam_out, final_out):
        return None

    try:
        base_check = _assemble_base_rectangle(inp)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(base_check, base_out):
        return None
    if np.array_equal(got, final_out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
