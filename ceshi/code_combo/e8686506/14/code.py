# Combo variant of e8686506 (index 14)
# combo_spec: D8/2+D10/0
# Name: 碎片面积均等+无框架

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and flow impact
# - Original task flow:
#   1. detect one large 8-connected frame piece, or two frame-colored halves;
#   2. read optional corner/bar/line anchors from that frame-friendly layout;
#   3. infer the output rectangle and force most cells immediately;
#   4. propagate the remaining small fill pieces into the holes.
# - Combo flow:
#   1. every non-background component is a small scattered fragment of area 5;
#   2. there is no intact outline fragment and no preserved frame bbox in the input;
#   3. the solver must infer the rectangle only from total area and then fit all pieces;
#   4. the final board is accepted only if it is fully tiled, exactly LR-symmetric by color,
#      and distinct pieces of the same color never 4-touch.
# - Small trace:
#   old 8x5 sample: frame already occupies an 8x5 bbox -> anchors lock edge colors ->
#   interior pieces are forced into the remaining cavities.
#   new 6x5 sample: six pentomino-like pieces are all scattered apart -> total area 30
#   implies 6x5 -> search over placements with mirror-color and no-same-color-touch rules
#   -> exactly one dense symmetric rectangle survives.
#
# b) Complexity calibration
# - The generator fixes all piece areas at 5 cells. This makes the "面积均等" signal visible
#   in every sample and guarantees that no fragment looks like a dominant scaffold.
# - The rectangle stays portrait and narrow (6x5), close to the original task's dense
#   rectangular target, so the puzzle still reads as a reconstruction task rather than as a
#   free placement puzzle.
# - Two closely related tiling templates plus vertical flips provide sample variety without
#   exploding the number of admissible outputs.
#
# c) Implicit assumption audit against the original solver
# - `_build_requirements(...)` relied on singleton corners, width-(W-2) bars, and paired
#   edge lines. Equal-area fragments remove those roles, so the entire requirement-builder
#   must disappear.
# - Original frame detection required one or two 8-connected components whose union bbox
#   already matched the output rectangle. D10/0 explicitly invalidates that premise.
# - Original `_assemble(...)` assumed cell locking by the frame would make pure propagation
#   enough. Once both frame and anchor clues are gone, exact-cover style backtracking is
#   necessary; otherwise many translations of each fragment remain live.
#
# d) Placement legality
# - Output is created first as a valid 6x5 tiling of six 5-cell components.
# - Each component is extracted as its own patch and scattered on a 13-15 square canvas
#   using `random_free_location(..., padding=1, padding_connectivity=8)`, so pieces do not
#   overlap and remain visually separated.
# - Generator acceptance is oracle-based: it only keeps an input if the rewritten `main()`
#   returns exactly the intended output and finds it uniquely.
#
# e) Visual observability
# - Across examples, a human can see two stable facts: there is no large preserved outline,
#   and all fragments are equally sized. That is the intended fusion signal.
# - The output rule remains visually inferable: the scattered fragments must be reassembled
#   into one solid rectangle whose left and right halves mirror by color.
#
# f) Visual cleanliness
# - No extra markers are added.
# - Each input fragment is a single-color patch isolated by at least one background cell in
#   the 8-neighborhood, so the pieces read cleanly as separate jigsaw parts.
#
# g) Color inference and solver assumptions
# - Colors are assigned per LR-mirror orbit of pieces, not by hidden numeric ordering.
# - Different orbits always get different colors; this keeps the symmetry cue strong.
# - New solver assumptions enforced by the generator:
#   1. every non-background 4-connected component is an intended piece;
#   2. all piece areas are equal (here exactly 5);
#   3. total area has a unique closest-to-square portrait factorization;
#   4. the intended assembly is color-LR-symmetric;
#   5. different pieces of the same color do not 4-touch in the output.
#
# Post-rewrite `main()` assumption audit
# - `_extract_pieces(...)` assumes color components stay disconnected in the input; scatter
#   padding guarantees that.
# - `_infer_dims(...)` assumes the chosen total area points to a unique portrait factor pair;
#   the generator fixes area to 30, whose best pair is uniquely 6x5.
# - `_solve_pieces(...)` assumes translation-only placement; generator never rotates or flips
#   pieces after extraction, so that assumption matches the visible puzzle.
# - The same-color adjacency check assumes identical visible colors may correspond to distinct
#   pieces; generator assigns colors by orbit and keeps those pieces separated in the output.


_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300
_PIECE_AREA = 5


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
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait output shape must be unique"
    return best[0]


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
                opts.append([(r0 + dr, c0 + dc) for dr, dc in p["coords"]])
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
    templates = [
        np.array(
            [
                [0, 4, 4, 4, 1],
                [0, 0, 4, 1, 1],
                [0, 2, 4, 3, 1],
                [0, 2, 5, 3, 1],
                [2, 2, 5, 3, 3],
                [2, 5, 5, 5, 3],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 0, 4, 1, 1],
                [0, 4, 4, 4, 1],
                [0, 2, 4, 3, 1],
                [0, 2, 5, 3, 1],
                [2, 2, 5, 3, 3],
                [2, 5, 5, 5, 3],
            ],
            dtype=int,
        ),
    ]
    piece_ids = random.choice(templates).copy()
    if random.random() < 0.5:
        piece_ids = piece_ids[::-1].copy()
    return piece_ids


def _validate_template(piece_ids: np.ndarray) -> None:
    ids = sorted(map(int, np.unique(piece_ids)))
    assert len(ids) == 6, "expected six fragments"
    for pid in ids:
        pts = np.argwhere(piece_ids == pid)
        assert len(pts) == _PIECE_AREA, "all template pieces must have equal area"
        assert len(components_mask(piece_ids == pid, connectivity=4)) == 1, "piece must be 4-connected"
    H, W = piece_ids.shape
    for pid in ids:
        coords = np.argwhere(piece_ids == pid)
        mates = {int(piece_ids[int(r), W - 1 - int(c)]) for r, c in coords}
        assert len(mates) == 1, "template must be closed under LR mirror"


def _mirror_orbits(piece_ids: np.ndarray) -> list[list[int]]:
    H, W = piece_ids.shape
    partner: dict[int, int] = {}
    for pid in sorted(map(int, np.unique(piece_ids))):
        coords = np.argwhere(piece_ids == pid)
        mapped = {int(piece_ids[int(r), W - 1 - int(c)]) for r, c in coords}
        partner[pid] = next(iter(mapped))

    orbits = []
    seen = set()
    for pid in sorted(partner):
        if pid in seen:
            continue
        mate = partner[pid]
        orbit = [pid] if mate == pid else sorted([pid, mate])
        for x in orbit:
            seen.add(x)
        orbits.append(orbit)
    return orbits


def _build_output() -> tuple[np.ndarray, int]:
    piece_ids = _template_ids()
    _validate_template(piece_ids)

    bg = random.randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)

    cmap = {}
    for orbit, col in zip(_mirror_orbits(piece_ids), colors):
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
    _bg, pieces = _extract_pieces(grid)
    return sorted(int(p["area"]) for p in pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered equal-size fragments into one left-right symmetric solid rectangle.

    Input structure:
    - The background is the most frequent color.
    - Every non-background component is a small single-color fragment, and no large outline or
      frame-shaped piece remains parked in output-relative position.

    Rule:
    - Extract every non-background 4-connected piece and sum their area.
    - Infer the portrait rectangle from the unique closest-to-square factorization of that area.
    - Place the pieces without rotation so they tile the rectangle completely, the final colors
      match under left-right mirroring, and different pieces of the same color never 4-touch.

    Key insight:
    - With both size hierarchy and frame skeleton removed, the only reliable route is to fit all
      fragments jointly; the correct dense tiling is the unique one consistent with piece shapes,
      mirror-color symmetry, and same-color separation.
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

    areas = _piece_areas(out)
    if not areas or any(a != _PIECE_AREA for a in areas):
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
    raise RuntimeError("failed to generate a valid e8686506 combo input")
