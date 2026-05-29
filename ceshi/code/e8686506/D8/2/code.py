# Variant of e8686506
# Dimension: D8 — 碎片数量与复杂度 (meta)
# Change: 输入包含一个大框架碎片和多个小碎片（3-6个），总碎片数适中... → 碎片面积均等

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior and flow impact
# - Original task: one dominant 8-connected frame piece (or two frame halves) stayed in
#   output-relative position inside the input, so the solver first recovered the rectangle
#   by detecting that frame and only then propagated the smaller fill pieces.
# - Variant: every fragment has the same area (here exactly 5 cells), so there is no
#   visually dominant outline fragment. All pieces are scattered independently, and the
#   solver must reconstruct the whole dense rectangle by jointly fitting all pieces under
#   left-right color symmetry.
# - Small trace:
#   old flow on an 8x5 example:
#     detect big frame -> lock many output cells -> small pieces each have forced slots
#   new flow on a 6x5 example:
#     collect six equal-area pentomino-like pieces -> infer 6x5 from total area 30 ->
#     exact-cover search over all placements while enforcing LR color symmetry ->
#     the unique full tiling is the output.
#
# b) Complexity calibration
# - Keeping width 5 preserves the original "dense portrait rectangle" feel.
# - Restricting outputs to 6x5 gives 6 pieces: enough pieces for a genuine
#   jigsaw, but still small enough for an exact-cover solver.
# - Every piece has area 5, so the removed frame clue is visually obvious: the largest
#   piece is no larger than any other.
#
# c) Implicit assumption audit against the original main()
# - Original `_build_requirements(...)` depended on singleton corners / bars / line pairs.
#   Those structures conflict with equal-area fragments, so they are removed.
# - Original frame detection assumed 1-2 eight-connected components whose union bbox was
#   exactly the output rectangle. That is the key invalidated assumption in this variant.
# - Original `_assemble(...)` relied on frame-locked cells to make propagation sufficient.
#   Without the frame, exact-cover search is needed; otherwise many valid translations of a
#   piece would stay alive too long and propagation would stall.
#
# d) Placement legality
# - Generator uses output-first templates whose labels already tile the target rectangle.
# - Each component is cut as a standalone patch and scattered with `random_free_location`
#   plus 8-neighbor padding, so input pieces never overlap and remain visually separated.
# - Oracle validation always checks `main(input) == intended_output` before accepting a sample.
#
# e) Visual observability
# - Humans can now see that all scattered pieces are similarly sized; the old "frame left in
#   place" cue is absent in every generated sample.
# - The remaining invariant is still readable from 2-3 examples: reassemble all equal-area
#   fragments into a left-right symmetric solid rectangle.
#
# f) Visual cleanliness
# - No stacked markers or extra anchor glyphs are added.
# - Each piece is a single-color connected patch with one-cell padding between patches.
#
# g) Color inference and new solver assumptions
# - Color ids are only used as visible colors, never for sorting or hidden ordering rules.
# - Mirror-related piece orbits share the same color; different orbits use distinct colors.
# - New solver assumptions:
#   1. all non-background components are the intended pieces;
#   2. their total area has a unique closest-to-square portrait factorization;
#   3. the completed board is exactly LR-symmetric by color;
#   4. different pieces of the same color do not 4-touch in the output.
# - The generator enforces all four conditions directly.


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


def _extract_pieces(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pieces = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            pieces.append(
                {
                    "col": int(col),
                    "coords": coords,
                    "h": int(h),
                    "w": int(w),
                    "area": len(coords),
                }
            )
    return bg, pieces


def _infer_dims(total_area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    return best_dims[0]


def _solve_pieces(
    pieces: list[dict],
    H: int,
    W: int,
    *,
    max_solutions: int = 1,
) -> list[np.ndarray]:
    placements: list[list[list[tuple[int, int]]]] = []
    for p in pieces:
        opts = []
        for r0 in range(H - int(p["h"]) + 1):
            for c0 in range(W - int(p["w"]) + 1):
                cells = [(r0 + dr, c0 + dc) for dr, dc in p["coords"]]
                opts.append(cells)
        placements.append(opts)

    board_col = np.full((H, W), -1, dtype=int)
    board_owner = np.full((H, W), -1, dtype=int)
    remaining = set(range(len(pieces)))
    solutions: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    def can_place(pi: int, cells: list[tuple[int, int]]) -> bool:
        col = int(pieces[pi]["col"])
        for r, c in cells:
            if board_owner[r, c] != -1:
                return False
            mc = W - 1 - c
            mirror_col = board_col[r, mc]
            if mirror_col != -1 and mirror_col != col:
                return False
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < H and 0 <= cc < W:
                    owner = board_owner[rr, cc]
                    if owner != -1 and owner != pi and board_col[rr, cc] == col:
                        return False
        return True

    def dfs() -> None:
        if len(solutions) >= max_solutions:
            return
        if not remaining:
            if np.any(board_owner == -1):
                return
            if not np.array_equal(board_col, board_col[:, ::-1]):
                return
            key = tuple(map(int, board_col.ravel()))
            if key not in seen:
                seen.add(key)
                solutions.append(board_col.copy())
            return

        best_pi = None
        best_opts = None
        for pi in list(remaining):
            valid = [cells for cells in placements[pi] if can_place(pi, cells)]
            if not valid:
                return
            if best_opts is None or len(valid) < len(best_opts):
                best_pi = pi
                best_opts = valid
                if len(best_opts) == 1:
                    break

        assert best_pi is not None and best_opts is not None
        remaining.remove(best_pi)
        for cells in best_opts:
            for r, c in cells:
                board_owner[r, c] = best_pi
                board_col[r, c] = int(pieces[best_pi]["col"])
            dfs()
            for r, c in cells:
                board_owner[r, c] = -1
                board_col[r, c] = -1
            if len(solutions) >= max_solutions:
                break
        remaining.add(best_pi)

    dfs()
    return solutions


def _template_ids() -> np.ndarray:
    return np.array(
        [
            [0, 4, 4, 4, 1],
            [0, 0, 4, 1, 1],
            [0, 2, 4, 3, 1],
            [0, 2, 5, 3, 1],
            [2, 2, 5, 3, 3],
            [2, 5, 5, 5, 3],
        ],
        dtype=int,
    )


def _mirror_orbits(piece_ids: np.ndarray) -> list[list[int]]:
    H, W = piece_ids.shape
    partner: dict[int, int] = {}
    for pid in sorted(map(int, np.unique(piece_ids))):
        coords = np.argwhere(piece_ids == pid)
        mapped = {int(piece_ids[int(r), W - 1 - int(c)]) for r, c in coords}
        assert len(mapped) == 1, "template must be closed under LR mirror"
        partner[pid] = next(iter(mapped))
    orbits = []
    seen = set()
    for pid in sorted(partner):
        if pid in seen:
            continue
        mate = partner[pid]
        if mate == pid:
            orbit = [pid]
        else:
            orbit = sorted([pid, mate])
        for x in orbit:
            seen.add(x)
        orbits.append(orbit)
    return orbits


def _build_output() -> tuple[np.ndarray, int]:
    piece_ids = _template_ids()
    if random.random() < 0.5:
        piece_ids = piece_ids[::-1].copy()

    bg = random.randint(0, 9)
    orbits = _mirror_orbits(piece_ids)
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    cmap = {}
    for orbit, col in zip(orbits, colors):
        for pid in orbit:
            cmap[pid] = col

    out = np.zeros_like(piece_ids)
    for pid, col in cmap.items():
        out[piece_ids == pid] = col
    return out.astype(int), bg


def _extract_patches(out: np.ndarray, bg: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    for patch in _extract_patches(out, bg):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _piece_areas(grid: np.ndarray) -> list[int]:
    bg = get_bg(grid)
    areas = []
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            areas.append(len(pts))
    return sorted(areas)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered equal-area fragments into one left-right symmetric solid rectangle.

    Input structure:
    - The background is the most frequent color.
    - Every non-background component is a similarly sized single-color fragment; there is no
      dominant frame component left in place.

    Rule:
    - Extract all non-background 4-connected pieces and sum their area.
    - Infer the portrait output rectangle from the unique closest-to-square factorization.
    - Place every piece without rotation so the rectangle is fully tiled, mirrored colors match
      across the vertical axis, and different pieces of the same color do not 4-touch.

    Key insight:
    - The missing frame clue is replaced by equal-area jigsaw fitting: only one dense tiling can
      satisfy every piece shape together with the global left-right color symmetry.
    """
    g = np.asarray(grid, dtype=int)
    _bg, pieces = _extract_pieces(g)
    assert pieces, "grid must contain non-background pieces"

    areas = [int(p["area"]) for p in pieces]
    assert max(areas) - min(areas) <= 1, "variant expects equal-area or near-equal pieces"

    total_area = sum(areas)
    H, W = _infer_dims(total_area)

    sols = _solve_pieces(pieces, H, W, max_solutions=1)
    assert sols, "no valid symmetric assembly found"
    return sols[0].astype(int)


def _gen_attempt() -> np.ndarray | None:
    out, bg = _build_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None

    areas = []
    for col in np.unique(out):
        for pts in components_mask(out == col, connectivity=4):
            areas.append(len(pts))
    if not areas or len(set(areas)) != 1 or areas[0] != 5:
        return None

    try:
        inp = _scatter(out, bg)
        _bg, pieces = _extract_pieces(inp)
        sols = _solve_pieces(pieces, *out.shape, max_solutions=2)
    except Exception:
        return None

    if len(sols) != 1:
        return None
    if np.array_equal(sols[0], out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D8 input")
