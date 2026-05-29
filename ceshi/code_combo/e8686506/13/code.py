# Combo variant of e8686506 (index 13)
# combo_spec: D7/0+D17/0
# Name: 保持输入尺寸+原位轮廓痕迹

# Design analysis for the D7/0 + D17/0 fusion
# - Concrete behavior:
#   The original task reads a large scatter canvas and outputs only the cropped solved HxW rectangle.
#   The fused task keeps the same input, but the output stays on the original canvas. The anchored
#   frame bbox becomes the destination window where the full symmetric rectangle is painted, while
#   every moved non-frame piece leaves a gray footprint at its source cells.
# - Small flow trace:
#   Suppose the input is 13x13 and the inferred rectangle is 5x5.
#   Original:
#   1. Detect all non-bg pieces.
#   2. Infer the 5x5 LR-symmetric packing.
#   3. Return only that 5x5 block.
#   Fused:
#   1. Perform the exact same inference.
#   2. Paint the solved 5x5 block back into the frame bbox inside the 13x13 canvas.
#   3. Repaint each scattered non-frame piece location in gray on the same 13x13 canvas.
#   The logical solve is unchanged; only the output carrier and the preserved source traces change.
# - Complexity calibration:
#   Gray is fixed to ARC color 5 and reserved for traces only. This keeps the new behavior visually
#   obvious without introducing an extra latent color-mapping rule. The source traces repeat across
#   all moved pieces, so the “move from shadow to destination” idea is supported by multiple objects
#   in each sample instead of a one-off exception.
# - Assumption review against the original main():
#   1. Piece extraction from the input still works because shadows exist only in the output.
#   2. Total non-bg input area still equals the solved rectangle area; the generator never inserts
#      extra colored marks into the input.
#   3. Frame detection still depends on 1-2 LR-symmetric 8-connected parts whose union bbox is HxW.
#      The generator preserves that structure unchanged.
#   4. The old `return sol` assumption is invalid now; the solver must embed `sol` back into a full
#      canvas at the detected frame location.
#   5. A new assumption appears: the gray trace color must not already belong to bg or live pieces.
#      The generator enforces that by reserving color 5.
# - Placement legality:
#   The generator blocks the entire frame bbox before scattering movable pieces, so source traces never
#   overlap the destination rectangle. Chebyshev padding 1 between scattered pieces is kept from the
#   original task, so shadows remain visually separated and unambiguous.
# - Visual observability and neatness:
#   The solved rectangle is concentrated in one anchored region, while the gray traces stay in the
#   scattered region outside it. This makes the fused rule legible from a few examples: the output is
#   same-size, the destination is the pre-existing frame bbox, and the origin of each moved piece is
#   explicitly marked.
# - Anti-degeneration check:
#   `main(inp) == out` alone would not prove the fusion happened, because a degenerate sample could still
#   mimic the original task. The generator therefore requires both `output.shape == input.shape` and
#   `shadow_cells > 0`, and later testing reports those quantities explicitly.

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_SHADOW_COLOR = 5


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
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
                if hh == 1 and ww == W - 2 and len(pts) == ww:
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
        sigs = [_piece_sig(np.asarray(pts, dtype=int)) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in output height"
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
        v = board[r, W - 1 - c]
        return v == -1 or v == col

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
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((h, w), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), np.asarray(pts, dtype=int)))

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

    assert frame_parts, "input must contain 1-2 LR-symmetric frame parts whose union bbox equals output dims"
    return frame_parts


def _frame_origin(frame_parts: list[tuple[int, np.ndarray]]) -> tuple[int, int]:
    all_frame_pts = np.concatenate([pts for _, pts in frame_parts])
    return int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())


def _render_canvas_output(
    shape: tuple[int, int],
    bg: int,
    fr0: int,
    fc0: int,
    sol: np.ndarray,
    shadow_components: list[np.ndarray],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    H, W = sol.shape
    out[fr0 : fr0 + H, fc0 : fc0 + W] = sol
    for pts in shadow_components:
        for r, c in pts:
            out[int(r), int(c)] = _SHADOW_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered fragments inside the anchored rectangle while keeping a visible source trace.

    Input structure:
    - The most common color is background; 1-2 left-right symmetric frame pieces already sit at the future
      assembly bbox and keep their output-relative positions.
    - The remaining colored fragments are scattered elsewhere on the same canvas and represent the movable pieces.

    Rule:
    - Infer the dense left-right symmetric HxW rectangle exactly as in the base task, using symmetry, frame locks,
      and the no-same-color-4-touch condition.
    - Return a full-size canvas, not a cropped rectangle: paint the solved rectangle back into the detected frame bbox.
    - Leave a gray mark at every original location of a moved non-frame piece, so the output shows both destination
      and source.

    Key insight:
    - The hard part is still the original constrained packing problem. Once the frame bbox is identified, the new
      behavior is bookkeeping: frame cells stay as the destination anchor, and every other input component becomes a
      gray “came from here” trace on the preserved input-sized canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}
    all_comps4: list[np.ndarray] = []

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            pts_arr = np.asarray(pts, dtype=int)
            all_comps4.append(pts_arr)
            _, _, coords = _piece_sig(pts_arr)
            pieces.append((int(col), coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.asarray(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "coords": coords,
                "h": int(arr[:, 0].max() + 1),
                "w": int(arr[:, 1].max() + 1),
                "cnt": int(cnt),
            }
        )

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    fr0, fc0 = _frame_origin(frame_parts)
    req = _build_requirements(H, W, comps_by_color)

    frame_cells: set[tuple[int, int]] = set()
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = int(col)
                frame_cells.add((int(r), int(c)))

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)

    shadow_components = []
    for pts in all_comps4:
        if all((int(r), int(c)) in frame_cells for r, c in pts):
            continue
        shadow_components.append(pts)

    return _render_canvas_output(g.shape, int(bg), fr0, fc0, sol.astype(int), shadow_components)


# --------------- generator (o5x) ---------------

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
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 500


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1 and best[0][1] == _WIDTH_REQUIRED and best[0][0] <= _HEIGHT_MAX:
            results.append(best[0])
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


def _fill_holes(grid: np.ndarray, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> None:
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

    _fill_holes(grid, W, frame1, frame2, split, fill_pool)
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

    bg_choices = [c for c in range(10) if c != _SHADOW_COLOR]
    bg = random.choice(bg_choices)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c not in (bg, _SHADOW_COLOR)]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()

    cmap = {a: available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), int(bg), real_fcs


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


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, np.ndarray, int]:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame bbox must fit with a margin")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    shadow_specs: list[tuple[np.ndarray, int, int]] = []
    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, int(r), int(c), transparent=bg)
        shadow_specs.append((patch.copy(), int(r), int(c)))

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    expected = np.full_like(canvas, bg)
    expected[off_r : off_r + oh, off_c : off_c + ow] = out

    shadow_cells = 0
    for patch, r, c in shadow_specs:
        mask = patch != bg
        shadow_cells += int(mask.sum())
        shadow_patch = np.full_like(patch, bg)
        shadow_patch[mask] = _SHADOW_COLOR
        paste(expected, shadow_patch, r, c, transparent=bg)

    return canvas, expected, shadow_cells


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> tuple[np.ndarray | None, dict]:
    out, bg, fcs = _make_output()
    if out is None:
        return None, {}
    if not np.array_equal(out, out[:, ::-1]):
        return None, {}
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, {}
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None, {}
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, {}

    try:
        inp, expected, shadow_cells = _scatter(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None, {}

    if shadow_cells <= 0:
        return None, {}
    if inp.shape != got.shape:
        return None, {}
    if np.array_equal(got, expected):
        return inp, {"shadow_cells": int(shadow_cells), "out_shape": tuple(map(int, got.shape))}
    return None, {}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid, _meta = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
