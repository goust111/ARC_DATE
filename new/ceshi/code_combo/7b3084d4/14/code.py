# Combo variant of 7b3084d4 (index 14)
# combo_spec: D4/0+D8/0+D15/1
# Name: 标记指定角落+多tab+不规则分割

import random
from collections import deque
from itertools import product

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
MARKER_COLOR = 5
ANCHOR_COLOR = 7
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 500
MIN_N = 3
MAX_N = 5

TL, TR, BL, BR = 0, 1, 2, 3
CORNERS = ("tl", "tr", "bl", "br")


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original task: four scattered pieces tile a square, and the grey marker is
#    always normalized to output (0,0).  Here the output is still a tight
#    square, but the marked piece must land in the same corner that the marker
#    occupies inside that piece on input, and the internal seams are staircase
#    profiles with several alternating excursions instead of one straight split
#    plus one tab.  For a 6x6 output, if the input anchor shows its marker on
#    local top-right, then the solver must place that piece so the assembled
#    square has grey at global top-right; the remaining pieces are matched by
#    their irregular multi-tab silhouettes.
# b) Complexity calibration:
#    The generator changes two low-cardinality controls that are visible across
#    samples: output corner in {tl,tr,bl,br} and two irregular profiles, each
#    with 2-3 excursions.  This is enough to make the fusion visually distinct
#    from the seed while avoiding chaotic free-form outlines.
# c) Broken assumptions from the seed:
#    - "one color == one piece" no longer holds for the anchor because the grey
#      marker belongs to the same connected component as the orange anchor.
#    - "marker always targets (0,0)" is false; the solver must infer the target
#      corner from the marker's corner within the input anchor bbox.
#    - "single tab per boundary" is false; several row/column bands shift the
#      split, so shape matching must use the full silhouette rather than one
#      offset parameter.
# d) Placement legality:
#    Pieces are extracted from a full valid output square, then scattered onto a
#    20x20 canvas with `random_free_location(..., padding=1, connectivity=8)`,
#    so components never overlap, never leave the canvas, and never become
#    8-adjacent to each other.
# e) Visual observability:
#    The marker-corner rule is human-visible because the lone grey pixel sits at
#    one of the four anchor bbox corners, and the output places grey in the same
#    global corner.  The multi-tab irregular seams are visible as several
#    staircase bites and protrusions, not as a single centered notch.
# f) Visual cleanliness:
#    Each piece stays single-color except for the grey marker, the output has no
#    layered glyphs, and scattering uses at least one background cell of space.
# g) Inference support:
#    There is no hidden color remapping.  Colors only identify pieces, while the
#    inferable structure comes from corner position plus irregular complementary
#    boundaries.


def _normalize(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(r - min_r, c - min_c, col) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    out: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            abs_cells.append((r, c, int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append({"cells": _normalize(abs_cells), "abs": abs_cells})
    return comps, bg


def _marker_corner(cells: list[tuple[int, int, int]]) -> tuple[str, tuple[int, int]]:
    pts = [(r, c) for r, c, col in cells if col == MARKER_COLOR]
    assert len(pts) == 1, "input must contain exactly one marker pixel"
    mr, mc = pts[0]
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)
    if (mr, mc) == (0, 0):
        return "tl", (mr, mc)
    if (mr, mc) == (0, max_c):
        return "tr", (mr, mc)
    if (mr, mc) == (max_r, 0):
        return "bl", (mr, mc)
    if (mr, mc) == (max_r, max_c):
        return "br", (mr, mc)
    raise AssertionError("marker must be on a corner of the anchor bbox")


def _target_coord(corner: str, H: int, W: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, W - 1
    if corner == "bl":
        return H - 1, 0
    return H - 1, W - 1


def _dims(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return out


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    H, W = board.shape
    for r, c, _ in placement:
        if not (0 <= r < H and 0 <= c < W) or board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    for r, c, col in placement:
        board[r, c] = col if val is None else val


def _all_placements_covering(
    variant: list[tuple[int, int, int]], H: int, W: int, cover: tuple[int, int]
) -> list[list[tuple[int, int, int]]]:
    ph = max(r for r, _, _ in variant) + 1
    pw = max(c for _, c, _ in variant) + 1
    cover_r, cover_c = cover
    out = []
    for tr in range(max(0, cover_r - ph + 1), min(cover_r, H - ph) + 1):
        for tc in range(max(0, cover_c - pw + 1), min(cover_c, W - pw) + 1):
            placed = [(r + tr, c + tc, col) for r, c, col in variant]
            if any(r == cover_r and c == cover_c for r, c, _ in placed):
                out.append(placed)
    return out


def _first_empty(board: np.ndarray) -> tuple[int, int] | None:
    pts = np.argwhere(board == -1)
    if len(pts) == 0:
        return None
    return tuple(map(int, pts[0]))


def _solve_packing(comps: list[dict], H: int, W: int) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    anchor_idx = next((i for i, comp in enumerate(comps) if any(col == MARKER_COLOR for _, _, col in comp["cells"])), None)
    assert anchor_idx is not None, "one piece must contain the marker"

    piece_vars = [_variants(comp["cells"]) for comp in comps]
    target_corner, _ = _marker_corner(comps[anchor_idx]["cells"])
    target_r, target_c = _target_coord(target_corner, H, W)

    remaining = [i for i in range(len(comps)) if i != anchor_idx]
    for var in piece_vars[anchor_idx]:
        marker_pts = [(r, c) for r, c, col in var if col == MARKER_COLOR]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        placed = [(r + target_r - mr, c + target_c - mc, col) for r, c, col in var]
        if not _placeable(board, placed):
            continue
        _apply(board, placed)

        def backtrack(rem: list[int]) -> bool:
            empty = _first_empty(board)
            if empty is None:
                return True

            best_pi = None
            best_options = None
            for pi in rem:
                opts = []
                for variant in piece_vars[pi]:
                    for placement in _all_placements_covering(variant, H, W, empty):
                        if _placeable(board, placement):
                            opts.append(placement)
                if not opts:
                    continue
                if best_options is None or len(opts) < len(best_options):
                    best_pi = pi
                    best_options = opts
                    if len(best_options) == 1:
                        break

            if best_pi is None or best_options is None:
                return False
            next_rem = [x for x in rem if x != best_pi]
            for placement in best_options:
                _apply(board, placement)
                if backtrack(next_rem):
                    return True
                _apply(board, placement, -1)
            return False

        if backtrack(remaining):
            return board
        _apply(board, placed, -1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered interlocking puzzle pieces into one compact square, using the marker corner as the output corner.

    Input structure:
    - The background is the most common color on a 20x20 canvas.
    - There are exactly four separated 8-connected pieces.
    - One piece is the anchor: it contains a single grey marker pixel at one corner of that piece's bounding box.
    - Internal piece boundaries are irregular staircase splits with several alternating tab-like offsets.

    Rule:
    - Extract the four connected pieces from the scattered input.
    - Detect which piece contains the grey marker and read which corner of that anchor bbox the marker occupies.
    - Rotate pieces in 90-degree steps as needed and pack them into the unique tight rectangle that uses every piece cell exactly once.
    - Constrain the anchor so its grey marker lands in the same output corner that it occupied within the input anchor piece.

    Key insight:
    - The grey pixel does not always mean top-left anymore; it names the target corner.
    - The other three pieces are determined by complementary multi-tab irregular silhouettes, so exact shape packing resolves the assembly.
    """
    g = np.asarray(grid, dtype=int)
    comps, _bg = _components(g)
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    area = sum(len(comp["cells"]) for comp in comps)
    for H, W in _dims(area):
        solved = _solve_packing(comps, H, W)
        if solved is not None:
            return solved.astype(int)
    raise AssertionError("packing failed for all candidate dimensions")


def _profile_runs(profile: list[int], base: int) -> int:
    runs = 0
    in_run = False
    for v in profile:
        if v != base and not in_run:
            runs += 1
            in_run = True
        elif v == base:
            in_run = False
    return runs


def _sample_profile(S: int, base: int) -> list[int]:
    max_shift = max(1, min(2, base - 1))
    target_runs = 2 if S < 8 else random.choice([2, 3])
    interior_len = S - 2

    lengths_options = [
        lengths
        for lengths in product((1, 2), repeat=target_runs)
        if sum(lengths) + (target_runs - 1) <= interior_len
    ]
    assert lengths_options, "profile construction requires enough interior cells"

    lengths = list(random.choice(lengths_options))
    extra = interior_len - (sum(lengths) + (target_runs - 1))
    gaps = [0] * (target_runs + 1)
    for i in range(1, target_runs):
        gaps[i] = 1
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    prof = [base] * S
    pos = 1 + gaps[0]
    sign = random.choice([-1, 1])
    for idx, run_len in enumerate(lengths):
        mag = random.randint(1, max_shift)
        val = max(1, min(S - 1, base + sign * mag))
        for off in range(run_len):
            prof[pos + off] = val
        pos += run_len
        if idx + 1 < target_runs:
            pos += gaps[idx + 1]
        sign *= -1

    assert _profile_runs(prof, base) == target_runs
    assert len(set(prof)) >= 2
    return prof


def _build_ownership(N: int) -> tuple[np.ndarray, dict]:
    S = 2 * N
    plain = np.zeros((S, S), dtype=int)
    plain[:N, :N] = TL
    plain[:N, N:] = TR
    plain[N:, :N] = BL
    plain[N:, N:] = BR

    for _ in range(300):
        v = _sample_profile(S, N)
        h = _sample_profile(S, N)
        own = np.zeros((S, S), dtype=int)
        for r in range(S):
            for c in range(S):
                top = r < h[c]
                left = c < v[r]
                if top and left:
                    own[r, c] = TL
                elif top and not left:
                    own[r, c] = TR
                elif not top and left:
                    own[r, c] = BL
                else:
                    own[r, c] = BR
        if any(np.sum(own == q) == 0 for q in (TL, TR, BL, BR)):
            continue
        if not all(is_contiguous((own == q).astype(int), background=0, connectivity=8) for q in (TL, TR, BL, BR)):
            continue
        changed = int(np.sum(own != plain))
        if changed < max(4, N):
            continue
        meta = {
            "vertical_profile": tuple(v),
            "horizontal_profile": tuple(h),
            "vertical_runs": _profile_runs(v, N),
            "horizontal_runs": _profile_runs(h, N),
            "changed_cells": changed,
        }
        return own, meta
    raise RuntimeError("failed to build a valid irregular ownership grid")


def _rotated_marker_coord(S: int, k: int) -> tuple[int, int]:
    mask = np.zeros((S, S), dtype=int)
    mask[0, 0] = 1
    rot = np.rot90(mask, k=k)
    return tuple(map(int, np.argwhere(rot == 1)[0]))


def _corner_name_from_abs(pos: tuple[int, int], S: int) -> str:
    r, c = pos
    if (r, c) == (0, 0):
        return "tl"
    if (r, c) == (0, S - 1):
        return "tr"
    if (r, c) == (S - 1, 0):
        return "bl"
    return "br"


def _extract_piece_from_labels(
    labels: np.ndarray, q: int, color: int, marker_abs: tuple[int, int] | None = None
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(labels == q)
    min_r = int(rows.min())
    min_c = int(cols.min())
    cells = []
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        col = MARKER_COLOR if marker_abs == (r, c) else color
        cells.append((r - min_r, c - min_c, col))
    return _normalize(cells)


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    # Keep the puzzle large enough to visibly show multi-tab irregular seams,
    # while still fitting four scattered pieces on the fixed 20x20 canvas.
    N = random.randint(MIN_N, MAX_N)
    S = 2 * N
    labels, meta = _build_ownership(N)
    k_out = random.randint(0, 3)
    labels = np.rot90(labels, k=k_out)
    marker_abs = _rotated_marker_coord(S, k_out)

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    expected = np.zeros((S, S), dtype=int)
    for q, color in colors.items():
        expected[labels == q] = color
    expected[marker_abs] = MARKER_COLOR

    pieces = [_extract_piece_from_labels(labels, q, colors[q], marker_abs if q == TL else None) for q in (TL, TR, BL, BR)]

    transformed = [pieces[0]]
    transformed.extend(_rotate_cells(piece, random.randint(0, 3)) for piece in pieces[1:])
    canvas = _scatter_patches([_cells_to_patch(piece) for piece in transformed])
    assert canvas is not None, "piece scattering failed"

    meta = dict(meta)
    meta.update(
        {
            "N": N,
            "corner": _corner_name_from_abs(marker_abs, S),
            "rotation_k": k_out,
            "size": S,
        }
    )
    return canvas, expected, meta


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    try:
        canvas, expected, meta = _build_case()
    except Exception:
        return None
    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return canvas, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("generate_input: exhausted attempts")


def _sample_generation_metrics(trials: int = 100) -> dict:
    counts = {corner: 0 for corner in CORNERS}
    changed_cells = []
    run_pairs = []
    for _ in range(trials):
        built = _gen_attempt()
        while built is None:
            built = _gen_attempt()
        _canvas, meta = built
        counts[meta["corner"]] += 1
        changed_cells.append(meta["changed_cells"])
        run_pairs.append((meta["vertical_runs"], meta["horizontal_runs"]))
    return {
        "corner_counts": counts,
        "changed_cells": changed_cells,
        "run_pairs": run_pairs,
    }
