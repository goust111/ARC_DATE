# Combo variant of e8686506 (index 12)
# combo_spec: D7/0+D17/1
# Name: 保持输入尺寸+原位涂色标记

import random
from collections import Counter

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   core_change:
#     - Base task output is a cropped HxW rectangle containing only the solved assembly.
#     - This combo keeps the full input-sized canvas, repaints every original piece cell GREY,
#       and places the solved rectangle into one canonical destination: the centered bbox.
#   concrete_behavior:
#     - All puzzle pieces, including the LR-symmetric frame part/group, appear only in scattered form
#       in the input. The center target bbox is intentionally left empty.
#     - The output has two zones:
#       1. a centered dense rectangle containing the solved pattern;
#       2. grey footprints at every original non-background cell, showing where all pieces came from.
#   traced_example:
#     - Suppose the solved rectangle is 5x5. In the original task, the output is only that 5x5 crop.
#     - In the D17-only reference, the frame may already sit at rows 2..6, cols 1..5, so the 5x5
#       assembly is rebuilt there and only the moved loose pieces become grey.
#     - In this combo, the same frame-shaped piece group can instead be scattered at rows 1..5, cols 10..14,
#       a red 2-cell fragment can be at {(13,2),(13,3)}, and the center 5x5 destination stays empty in input.
#       The output becomes a full 17x17 canvas: the solved 5x5 rectangle is pasted at the centered bbox
#       rows 6..10, cols 6..10, while both the frame cells and the red fragment's old cells are recolored grey.
#     - Difference from base: no crop; difference from D17-only: the frame is no longer the output anchor,
#       because every piece moves to the centered destination.
#   complexity_calibration:
#     - The new placement rule has one small, repeated state: "always center the solved rectangle".
#       This is strong enough to be visible in 2-3 examples and does not introduce many degrees of freedom.
#     - The trail color is fixed GREY for all samples. Humans only need to infer one marker meaning:
#       grey means "a piece used to be here".
#     - The assembly logic itself stays unchanged: LR symmetry, frame-shape constraints, and the
#       no-same-color-4-touch rule still force a unique packing.
#   implicit_assumption_audit:
#     - Piece extraction over all non-background 4-components still holds, because trail cells exist
#       only in the output and never contaminate the input.
#     - Total piece area still determines HxW, since the solved rectangle content is unchanged.
#     - Frame detection by searching for 1-2 LR-symmetric 8-connected parts with bbox HxW still holds
#       even though the frame is moved, because the solver uses only the frame's internal relative layout.
#     - The old D17 rendering assumption "frame cells stay at the final rectangle location" is false here.
#       Rendering must therefore ignore the input frame location and instead compute the centered target bbox.
#     - New generator obligation: the centered target bbox must remain empty in the input, otherwise the
#       grey footprint would be partly overwritten and the D7 placement cue would become muddy.
#   placement_legality:
#     - Generation uses a separate occupancy canvas. The entire centered target bbox is blocked so no
#       input piece overlaps it.
#     - The frame group is placed via a full-bbox blocker patch, which keeps its interior clear and prevents
#       other fragments from appearing inside the frame outline.
#     - Remaining patches are scattered with random_free_location and 8-neighbor padding, so pieces do not
#       overlap or touch.
#   visual_observability:
#     - Across examples, the centered rectangle repeatedly appears in the same canonical location while the
#       grey remnants stay elsewhere. That directly exposes the "same-size + centered placement" rule.
#     - Because the input center is empty and the output center is filled, the D7 placement change is visible
#       without relying on hidden ordering or color-id semantics.
#   neatness_and_inferability:
#     - No region mixes multiple overlays. The centered bbox shows only the final assembly.
#       The scattered source sites show only monochrome grey footprints.
#     - Frame and non-frame origins use the same trail color; this avoids introducing an unnecessary
#       second marker language.
#   new_main_assumptions_checked_against_generator:
#     - main() assumes the centered target HxW bbox fits inside the canvas. The generator fixes the canvas
#       at 17x17, while HxW is always 5x5 or 8x5, so the assumption is guaranteed.
#     - main() assumes a unique frame part/group exists somewhere in the input. The generator constructs
#       the frame from the solved output and scatters it as one preserved HxW patch group.
#     - main() assumes GREY differs from the background and puzzle colors. The generator excludes GREY from
#       both the chosen background and the piece palette.


_TRAIL_COLOR = int(Color.GREY)


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
            req[0, 1 : W - 1] = bar_colors[0][1]
            req[H - 1, 1 : W - 1] = bar_colors[-1][1]

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
            board[r0 + dr, c0 + dc] = col

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
            assert places, "assembly dead end: a piece has no valid placement"
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


def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((h, w), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

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
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break

    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"
    return frame_parts


def _target_top_left(rows: int, cols: int, H: int, W: int) -> tuple[int, int]:
    assert H <= rows and W <= cols, "assembled rectangle must fit inside the same-size output canvas"
    return (rows - H) // 2, (cols - W) // 2


def _compose_output(
    g: np.ndarray,
    bg: int,
    assembled: np.ndarray,
    trail_color: int,
) -> np.ndarray:
    out = np.full_like(g, int(bg))
    moved_mask = g != int(bg)
    out[moved_mask] = int(trail_color)

    H, W = assembled.shape
    tr, tc = _target_top_left(g.shape[0], g.shape[1], H, W)
    out[tr : tr + H, tc : tc + W] = assembled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the symmetric rectangle on the full canvas, centered, and leave grey source footprints behind.

    Input structure:
    - The most common color is background; every other 4-connected component is a puzzle fragment.
    - Among those fragments there is still a special 1-2 piece LR-symmetric frame/group whose bbox equals
      the solved rectangle size, but it is scattered like any other source object rather than already sitting
      at the final destination.

    Rule:
    - Recover the same dense left-right symmetric rectangle as in the base task by packing all fragments into
      the frame-sized HxW assembly.
    - Repaint every original non-background source cell grey, then place the solved HxW rectangle into the
      centered bbox of the original canvas.
    - Return the full input-sized grid, so the centered solution and the grey origin traces are visible together.

    Key insight:
    - The hard part is still the original assembly: the preserved frame layout, mirror symmetry, and no-same-color
      4-touching force a unique rectangle. The combo change is only in rendering: all pieces move to one canonical
      centered destination, and their former cells become a uniform grey witness trail.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert int(_TRAIL_COLOR) != bg, "trail color must differ from background"

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

    frame_parts = _find_frame_parts(g, bg, H, W)
    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch).astype(int)
    return _compose_output(g, bg, sol, _TRAIL_COLOR).astype(int)


# --------------- generator (combo D7/0 + D17/1) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 17
_CANVAS_MAX = 17
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 1200


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        elif move == "diag":
            c = 1 - c
    return cells, start_c, c


def _fill_holes(
    grid: np.ndarray,
    H: int,
    W: int,
    frame1: int,
    frame2: int,
    split: int,
    fill_pool: list[int],
) -> None:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            fc_set = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = fc_set | ({prev_col} if prev_col is not None else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in fc_set] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H

    grid = np.full((H, W), -1, dtype=int)
    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)
    if np.any(grid < 0):
        return None, set()

    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()

    bg_choices = [c for c in range(10) if c != _TRAIL_COLOR]
    bg = random.choice(bg_choices)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c not in {bg, _TRAIL_COLOR}]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _extract_frame_patch(out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    patch = np.full_like(out, bg)
    frame_mask = np.zeros_like(out, dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    patch[frame_mask] = out[frame_mask]
    return patch


def _scatter_combo(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    placer = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    tr, tc = _target_top_left(rows, cols, oh, ow)
    placer[tr : tr + oh, tc : tc + ow] = -1

    frame_patch = _extract_frame_patch(out, bg, fcs)
    frame_block = np.full((oh, ow), -1, dtype=int)
    fr, fc = random_free_location(
        placer,
        frame_block,
        transparent=bg,
        padding=_SCATTER_PADDING,
        padding_connectivity=8,
    )
    paste(placer, frame_block, fr, fc)
    paste(canvas, frame_patch, fr, fc, transparent=bg)

    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            placer,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(placer, patch, r, c, transparent=bg)
        paste(canvas, patch, r, c, transparent=bg)

    expected = np.full((rows, cols), bg, dtype=int)
    expected[canvas != bg] = _TRAIL_COLOR
    expected[tr : tr + oh, tc : tc + ow] = out

    meta = {
        "target_bbox": (tr, tc, tr + oh - 1, tc + ow - 1),
        "frame_bbox": (fr, fc, fr + oh - 1, fc + ow - 1),
        "trail_cell_count": int((canvas != bg).sum()),
        "target_input_nonbg": int((canvas[tr : tr + oh, tc : tc + ow] != bg).sum()),
        "rect_shape": (oh, ow),
        "frame_offset_l1": int(abs(fr - tr) + abs(fc - tc)),
    }
    return canvas, expected, meta


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> tuple[np.ndarray | None, dict | None]:
    out, bg, fcs = _make_output()
    if out is None:
        return None, None
    if not np.array_equal(out, out[:, ::-1]):
        return None, None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None, None

    try:
        inp, expected, meta = _scatter_combo(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None, None

    if meta["target_input_nonbg"] != 0:
        return None, None
    if meta["frame_offset_l1"] <= 0:
        return None, None
    if meta["trail_cell_count"] <= 0:
        return None, None
    if not np.array_equal(got, expected):
        return None, None
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid, _meta = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
