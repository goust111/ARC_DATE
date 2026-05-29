# Variant of e8686506
# Dimension: D8 — 碎片数量与复杂度 (meta)
# Change: 输入包含一个大框架碎片和多个小碎片（3-6个），总碎片数适中... → 存在干扰碎片（不参与拼合）

import random
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_UTILS = _ROOT.parents[5] / "solvers"
if str(_SOLVER_UTILS) not in sys.path:
    sys.path.append(str(_SOLVER_UTILS))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior:
#    Output stays a dense left-right symmetric rectangle. Input now contains:
#    - one large 4-connected frame fragment whose bbox already equals the output size,
#    - 3-6 genuine fill fragments scattered elsewhere,
#    - 1-2 extra distractor fragments that never appear in the output.
#    Small trace: if the frame bbox is 8x5, the true assembled area is 40. In the old
#    task, summing every non-bg cell also gave 40, so `main()` could infer 8x5 from total
#    area. In this variant, a distractor domino makes the raw area 42, so the old logic
#    would infer the wrong rectangle. The new solver must first read 8x5 from the frame,
#    then discard the 2 extra cells by exclusion.
# b) Complexity calibration:
#    Distractors are only 1-2 pieces, not many, so the task stays close to the original.
#    They use visible standalone colors and compact shapes, so the extra complexity is
#    obvious in a few examples without making subset search explode.
# c) Hidden assumptions broken by the variant:
#    - Original `total_area = sum(all pieces)` is no longer valid.
#    - Original `req` and same-color component counts cannot be built from all pieces,
#      because distractors would inject fake constraints.
#    - Original "every extracted piece participates" assumption is false.
#    The fix is to search for the unique subset that fits the frame-implied rectangle.
# d) Placement legality:
#    Generator blocks the frame bbox, scatters all valid and distractor patches outside
#    it, and uses padding so pieces do not touch or overlap. This preserves the solver's
#    component extraction model.
# e) Visual observability:
#    Across examples, the input visibly has one large symmetric frame plus a handful of
#    small fragments, but 1-2 of those fragments use colors that disappear in the output.
#    A human can infer that not every input fragment is used.
# f) Visual cleanliness:
#    The frame remains grouped in one place, while every other patch is isolated on the
#    background. No overlay, no stacked markers, no local ambiguity.
# g) Color inference:
#    Distractor colors are chosen from colors absent in the output, so their role is
#    learnable from a few examples without relying on invisible color-id ordering.
# h) New solver assumptions introduced here:
#    - The large frame fragment is the only 4-connected piece whose bbox can be the
#      output bbox.
#    - Distractors stay outside the frame bbox.
#    - At most two distractors exist.
#    Generator enforces all three.


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
    comps_by_color: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    The original optional corner/bar/vertical markers are still supported, but only after
    distractors have been excluded from the candidate subset.
    """
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


def _extract_piece_records(g: np.ndarray, bg: int) -> list[dict]:
    pieces = []
    pid = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            pieces.append(
                {
                    "id": pid,
                    "col": int(col),
                    "pts": np.array(pts, dtype=int),
                    "coords": coords,
                    "h": int(h),
                    "w": int(w),
                    "area": int(len(coords)),
                }
            )
            pid += 1
    return pieces


def _group_types(selected: list[dict]) -> list[dict]:
    ctr = Counter((int(p["col"]), p["coords"]) for p in selected)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "coords": coords,
                "h": int(arr[:, 0].max() + 1),
                "w": int(arr[:, 1].max() + 1),
                "cnt": int(cnt),
            }
        )
    return types


def _comps_by_color(selected: list[dict]) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for rec in selected:
        out.setdefault(int(rec["col"]), []).append(rec["pts"])
    return out


def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((h, w), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _frame_candidates(g: np.ndarray, bg: int) -> list[dict]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), np.array(pts, dtype=int)))

    candidates = []
    seen = set()
    for i, (c1, p1) in enumerate(comps8_all):
        groups = [[(c1, p1)]]
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            groups.append([(c1, p1), (c2, p2)])
        for parts in groups:
            union = np.concatenate([p for _, p in parts])
            r0, c0, r1, c1b = bbox_coords(union)
            H, W = r1 - r0 + 1, c1b - c0 + 1
            if H < W:
                continue
            if not _is_lr_symmetric(parts, H, W):
                continue
            cells = frozenset((int(r), int(c)) for _, pts in parts for r, c in pts)
            if cells in seen:
                continue
            seen.add(cells)
            candidates.append({"parts": parts, "cells": cells, "H": int(H), "W": int(W)})
    candidates.sort(key=lambda c: (-(c["H"] * c["W"]), len(c["parts"])))
    return candidates


def _solve_with_subset(
    all_pieces: list[dict],
    selected_ids: set[int],
    frame_cells: frozenset[tuple[int, int]],
    H: int,
    W: int,
) -> np.ndarray:
    selected = [p for p in all_pieces if p["id"] in selected_ids]
    comps = _comps_by_color(selected)
    req = _build_requirements(H, W, comps)

    frame_pts = np.array(list(frame_cells), dtype=int)
    fr0, fc0 = int(frame_pts[:, 0].min()), int(frame_pts[:, 1].min())
    frame_cell_to_color = {}
    for rec in selected:
        for r, c in rec["pts"]:
            key = (int(r), int(c))
            if key in frame_cells:
                frame_cell_to_color[key] = int(rec["col"])
    for (r, c), col in frame_cell_to_color.items():
        req[int(r) - fr0, int(c) - fc0] = int(col)

    color_piece_count = Counter(int(p["col"]) for p in selected)
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    return _assemble(_group_types(selected), H, W, req, disallow_touch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the real fragments into the symmetric rectangle and ignore extra scraps.

    Input structure:
    - The background is the most common color.
    - One large connected frame fragment already shows the target rectangle's bounding box.
    - Several smaller fragments are genuine infill pieces, and 1-2 additional small fragments are distractors.

    Rule:
    - Find the frame fragment (or symmetric frame pair) whose bounding box defines the output size.
    - Keep exactly the subset of remaining fragments that, together with the frame, fills that rectangle.
    - Assemble the kept fragments into a dense left-right mirror-symmetric rectangle; discard the extra fragments.

    Key insight:
    - The frame reveals the target dimensions even when total input area is polluted by distractors.
    - Only one subset of pieces matches that frame-sized rectangle and still satisfies the original symmetry and no-same-color-touching constraints.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    all_pieces = _extract_piece_records(g, bg)
    assert all_pieces, "grid must contain non-background pieces"

    for cand in _frame_candidates(g, bg):
        H, W = int(cand["H"]), int(cand["W"])
        frame_cells = cand["cells"]

        frame_ids = {
            int(rec["id"])
            for rec in all_pieces
            if all((int(r), int(c)) in frame_cells for r, c in rec["pts"])
        }
        if not frame_ids:
            continue

        frame_area = sum(rec["area"] for rec in all_pieces if rec["id"] in frame_ids)
        target_area = H * W
        if frame_area > target_area:
            continue

        scatter = [rec for rec in all_pieces if rec["id"] not in frame_ids]
        extra_area = sum(rec["area"] for rec in scatter) - (target_area - frame_area)
        if extra_area < 0:
            continue

        for drop_n in range(0, min(2, len(scatter)) + 1):
            for drop_group in combinations(scatter, drop_n):
                if sum(rec["area"] for rec in drop_group) != extra_area:
                    continue
                selected_ids = frame_ids | {rec["id"] for rec in scatter if rec not in drop_group}
                if sum(rec["area"] for rec in all_pieces if rec["id"] in selected_ids) != target_area:
                    continue
                try:
                    return _solve_with_subset(all_pieces, selected_ids, frame_cells, H, W).astype(int)
                except Exception:
                    continue

    raise AssertionError("no valid frame-guided subset assembly found")


# --------------- generator ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6

# === CFG: Piece constraints ===
_MAX_CHUNK_SIZE = 6
_VALID_PATCH_MIN = 3
_VALID_PATCH_MAX = 6
_DISTRACTOR_MIN = 1
_DISTRACTOR_MAX = 2

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


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

_DISTRACTOR_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
]


def _zigzag_frame(H: int) -> set[tuple[int, int]]:
    """Generate a 4-connected left-half path. The mirrored full frame is then one large piece."""
    while True:
        cells = set()
        c = random.randint(0, 1)
        touched_inner = False
        for r in range(H):
            cells.add((r, c))
            if c == 1:
                touched_inner = True
            if r == H - 1:
                break
            if random.random() < 0.45:
                cells.add((r, 1 - c))
                c = 1 - c
                touched_inner = touched_inner or (c == 1)
        if touched_inner:
            return cells


def _fill_holes(grid: np.ndarray, W: int, fill_pool: list[int]) -> None:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            opts = [f for f in fill_pool if f != prev_col] or fill_pool
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
    frame_col = palette[0]
    fill_pool = palette[1:]
    if not fill_pool:
        return None, set()

    frame_half = _zigzag_frame(H)
    grid = np.full((H, W), -1, dtype=int)
    for r, c in frame_half:
        grid[r, c] = frame_col
        grid[r, W - 1 - c] = frame_col

    bridge_rows = [r for r in range(H) if grid[r, 1] == frame_col]
    if not bridge_rows:
        return None, set()
    n_bridges = random.randint(1, min(3, len(bridge_rows)))
    for r in random.sample(bridge_rows, n_bridges):
        grid[r, half] = frame_col

    _fill_holes(grid, W, fill_pool)
    if np.any(grid < 0):
        return None, set()
    return grid, {frame_col}


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()

    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    return remapped, bg, real_fcs


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


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    arr = np.array(shape, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _make_distractors(out: np.ndarray, bg: int, used_colors: set[int], count: int) -> list[np.ndarray]:
    available = [c for c in range(10) if c != bg and c not in used_colors]
    if len(available) < count:
        raise ValueError("not enough unused colors for distractors")
    random.shuffle(available)
    distractors = []
    for i in range(count):
        shape = random.choice(_DISTRACTOR_SHAPES)
        distractors.append(_shape_to_patch(shape, available[i], bg))
    return distractors


def _scatter(out: np.ndarray, bg: int, fcs: set[int], distractors: list[np.ndarray]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    for patch in distractors:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _non_frame_piece_count(out: np.ndarray, fcs: set[int]) -> int:
    count = 0
    for col in np.unique(out):
        if int(col) in fcs:
            continue
        count += len(components_mask(out == int(col), connectivity=4))
    return count


def _frame_component_count(inp: np.ndarray) -> int:
    bg = get_bg(inp)
    candidates = _frame_candidates(inp, bg)
    if not candidates:
        return 0
    best = candidates[0]
    return len(best["parts"])


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None

    valid_patch_count = _non_frame_piece_count(out, fcs)
    if not (_VALID_PATCH_MIN <= valid_patch_count <= _VALID_PATCH_MAX):
        return None

    n_distractors = random.randint(_DISTRACTOR_MIN, _DISTRACTOR_MAX)
    try:
        distractors = _make_distractors(out, bg, set(map(int, np.unique(out))), n_distractors)
        inp = _scatter(out, bg, fcs, distractors)
        got = main(inp)
    except Exception:
        return None

    if _frame_component_count(inp) != 1:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D8 input")
