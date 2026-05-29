# Combo variant of e8686506 (index 34)
# combo_spec: D3/0+D10/0+D15/2
# Name: 非矩形目标+无框架+模板提示
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original flow: treat every non-bg component as a real piece, infer a dense portrait rectangle
#      from total area, keep 1-2 large grouped frame pieces in-place, then use LR symmetry plus
#      no-same-color-touching to pack the rest into that rectangle.
#    - Fused flow here: one corner contains a monochrome template showing the outline of the target
#      cross silhouette; that template is not a puzzle piece and must be removed first. Every real
#      piece is then a small scattered fragment. The solver reads the template, reconstructs the
#      cross mask, and packs all fragments into that non-rectangular mask.
#    - Small trace: if the corner template bbox is 9x7, the original code would incorrectly try to
#      fill all 63 cells of that bbox or rely on a preserved frame. The fused code instead erases
#      the template, knows only 39 cells are occupiable (the thick cross), and places fragments only
#      on those 39 cells. The 24 corner cells of the bbox remain background in the output.
# b) Complexity calibration:
#    - D3/0 suggested cross or L, but the base task's retained LR-symmetry objective conflicts with
#      L. The fusion description explicitly asks for a non-rectangular symmetric output, so this
#      implementation keeps the symmetric branch only: thick crosses in a small family of odd sizes.
#    - The template has exactly one visual role: it reveals the occupied silhouette, not colors.
#      The rest of the puzzle still requires fragment assembly because the template is monochrome.
#    - D10/0 is enforced strictly: no grouped frame or anchor fragment remains in the target bbox.
#      All real fragments are small and scattered individually.
# c) Hidden-assumption audit against the original main():
#    - Rectangle factorization from total area is invalid because the target bbox contains bg holes.
#    - Frame detection is invalid because the fused generator never preserves a large in-place frame.
#    - Rectangle-edge heuristics (corner markers, edge bars, paired vertical lines) are invalid
#      because there is no dense rectangle border to anchor.
#    - Treating every non-bg component as a piece is invalid because the corner template is a clue,
#      not inventory.
#    - The preserved assumption is LR color symmetry, so the assembly search still enforces mirrored
#      colors across the vertical axis of the cross.
# d) Placement legality:
#    - The generator reserves the entire template bbox in one corner before scattering any fragment.
#      Every real patch is then placed with a one-cell border and padding, so pieces do not overlap,
#      do not run out of bounds, and do not touch the template.
#    - Because all fragments are extracted from a valid output-first cross, every scattered patch is
#      translation-only and always has at least one legal location in the true solution.
# e) Visual observability:
#    - Across 2-4 examples a human sees the same pattern: a corner monochrome cross-outline hint,
#      many small fragments elsewhere, and an output that is a compact cropped cross rather than a
#      filled rectangle. No hidden ordering uses numeric color ids.
# f) Visual cleanliness:
#    - The template is isolated in a corner. No real fragment is layered inside that corner bbox, and
#      there is no second large anchor object competing with it. The scattered fragments stay apart.
# g) Color and rule inference:
#    - Colors are random but structural: the template color never appears in the output, mirrored
#      fragment groups share colors, and adjacent output pieces never share a color. The human-visible
#      invariant is geometric symmetry, not any color-id arithmetic.
# h) New solver assumptions guaranteed by the generator:
#    - Exactly one corner-anchored template exists, and its bitmap is the boundary of one allowed
#      thick cross.
#    - After removing that template, every remaining non-bg component is a real piece with area <= 6.
#    - The intended output is LR-symmetric, non-rectangular, tightly cropped, and is recoverable by
#      the search implemented in main(); generate_input() validates this by oracle-checking main().

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_OUTSIDE = -2
_EMPTY = -1


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _shape_mask(H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mr, mc = H // 2, W // 2
    for r in range(H):
        for c in range(W):
            if abs(r - mr) <= 1 or abs(c - mc) <= 1:
                mask[r, c] = True
    return mask


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W) or not mask[rr, cc]:
                    out[r, c] = True
                    break
    return out


_OUTPUT_DIMS = [(7, 5), (7, 7), (9, 5), (9, 7)]
_VALID_TEMPLATE_BITMAPS = {
    (H, W): _boundary_mask(_shape_mask(H, W))
    for H, W in _OUTPUT_DIMS
}


def _corner_origins(rows: int, cols: int, h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, cols - w),
        (rows - h, 0),
        (rows - h, cols - w),
    ]


def _bitmap_from_component(pts: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    bmp = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        bmp[int(r) - r0, int(c) - c0] = True
    return bmp, (int(r0), int(c0))


def _find_template(
    g: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    GH, GW = g.shape
    all_non_bg = int(np.count_nonzero(g != bg))
    candidates: list[tuple[np.ndarray, tuple[int, int], np.ndarray]] = []

    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            bmp, (r0, c0) = _bitmap_from_component(pts)
            h, w = bmp.shape
            if (h, w) not in _VALID_TEMPLATE_BITMAPS:
                continue
            if not np.array_equal(bmp, _VALID_TEMPLATE_BITMAPS[(h, w)]):
                continue
            if (r0, c0) not in _corner_origins(GH, GW, h, w):
                continue
            shape_mask = _shape_mask(h, w)
            if all_non_bg - len(pts) != int(shape_mask.sum()):
                continue
            mask = np.zeros_like(g, dtype=bool)
            for r, c in pts:
                mask[int(r), int(c)] = True
            candidates.append((mask, (h, w), shape_mask))

    assert len(candidates) == 1, "input must contain exactly one corner cross-outline template"
    return candidates[0]


def _group_piece_types(g: np.ndarray, bg: int) -> tuple[list[dict], dict[int, int]]:
    pieces = []
    color_piece_count: dict[int, int] = {}
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        color_piece_count[int(col)] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((int(col), coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(
        ctr.items(),
        key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1]),
    ):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})
    return types, color_piece_count


def _assemble(
    types: list[dict],
    shape_mask: np.ndarray,
    disallow_touch: set[int],
) -> np.ndarray:
    H, W = shape_mask.shape
    board = np.full((H, W), _OUTSIDE, dtype=int)
    board[shape_mask] = _EMPTY
    counts = [int(t["cnt"]) for t in types]
    fail_cache: set[tuple[bytes, tuple[int, ...]]] = set()

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = int(board[r, mc])
        return v in (_EMPTY, col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        piece_cells = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W):
                return False
            if board[r, c] != _EMPTY:
                return False
            if not mirror_ok(r, c, col):
                return False
            piece_cells.append((r, c))
        if col in disallow_touch:
            piece_set = set(piece_cells)
            for r, c in piece_cells:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (rr, cc) in piece_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        t = types[ti]
        col = int(t["col"])
        changed = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = _EMPTY

    def forced_propagate() -> list[tuple[int, list[tuple[int, int]]]] | None:
        forced: list[tuple[int, list[tuple[int, int]]]] = []
        while True:
            progress = False
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                places = placements_for(ti)
                if not places:
                    for ti_forced, changed in reversed(forced):
                        unplace(changed)
                        counts[ti_forced] += 1
                    return None
                if len(places) == 1:
                    counts[ti] -= 1
                    changed = place(ti, places[0][0], places[0][1])
                    forced.append((ti, changed))
                    progress = True
            if not progress:
                return forced

    def rollback(forced: list[tuple[int, list[tuple[int, int]]]]) -> None:
        for ti_forced, changed in reversed(forced):
            unplace(changed)
            counts[ti_forced] += 1

    def search() -> bool:
        key = (board.tobytes(), tuple(counts))
        if key in fail_cache:
            return False

        forced = forced_propagate()
        if forced is None:
            fail_cache.add(key)
            return False

        if all(c == 0 for c in counts):
            ok = not np.any(board[shape_mask] == _EMPTY) and np.array_equal(board, board[:, ::-1])
            if not ok:
                rollback(forced)
                fail_cache.add(key)
            return ok

        choices = []
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            if not places:
                rollback(forced)
                fail_cache.add(key)
                return False
            choices.append((len(places), -len(types[ti]["coords"]), ti, places))
        choices.sort()
        _, _, ti, places = choices[0]

        for r0, c0 in places:
            counts[ti] -= 1
            changed = place(ti, r0, c0)
            if search():
                return True
            unplace(changed)
            counts[ti] += 1

        rollback(forced)
        fail_cache.add(key)
        return False

    assert search(), "assembly search failed"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a compact left-right symmetric cross.

    Input structure:
    - The most frequent color is background, and one corner contains a monochrome template showing
      only the outline of the final cross silhouette.
    - Every other non-background component is a small scattered puzzle fragment; no grouped frame
      or anchored boundary piece remains inside the future output bbox.

    Rule:
    - Detect the corner template, infer the exact non-rectangular cross mask from its bitmap, and
      remove the template from the piece inventory.
    - Treat all remaining components as translated puzzle pieces and pack them into that mask so
      every occupied cell is filled exactly once.
    - Preserve the original task's color symmetry: the assembled cross must be LR-symmetric, and
      different pieces of the same color may not become 4-neighbor adjacent.

    Key insight:
    - With the frame removed, the template becomes the only shape cue. It tells you which cells in
      the cropped bbox are real target cells and which corner cells must stay background.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    template_mask, (H, W), shape_mask = _find_template(g, bg)
    piece_grid = g.copy()
    piece_grid[template_mask] = bg

    types, color_piece_count = _group_piece_types(piece_grid, bg)
    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area == int(shape_mask.sum()), "piece area must match the template-implied cross area"

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    solved = _assemble(types, shape_mask, disallow_touch)

    out = np.full((H, W), bg, dtype=int)
    out[shape_mask] = solved[shape_mask]
    return out


# --------------- generator ---------------

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_CANVAS_MIN = 17
_CANVAS_MAX = 21
_SCATTER_PADDING = 1
_COLOR_RETRIES = 120
_GEN_MAX_RETRIES = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _fixed_chunks(H: int, W: int) -> tuple[list[list[tuple[int, int]]], list[int]]:
    mr, mc = H // 2, W // 2
    chunks: list[list[tuple[int, int]]] = []
    group_ids: list[int] = []
    group_id = 0

    def add(chunk: list[tuple[int, int]], gid: int | None = None) -> None:
        nonlocal group_id
        chunks.append(sorted((int(r), int(c)) for r, c in chunk))
        if gid is None:
            group_ids.append(group_id)
            group_id += 1
        else:
            group_ids.append(int(gid))

    # Top stem: self-symmetric U-shape plus a center singleton.
    add([(0, mc - 1), (0, mc), (0, mc + 1), (1, mc - 1), (1, mc + 1)])
    add([(1, mc)])

    # Optional extra stem bars for 9-row crosses.
    if H == 9:
        add([(2, mc - 1), (2, mc), (2, mc + 1)])

    # Horizontal arm center block: three row bars.
    for r in range(mr - 1, mr + 2):
        add([(r, mc - 1), (r, mc), (r, mc + 1)])

    # Side arms: mirrored vertical bars, one color-group per left column.
    for c in range(mc - 1):
        gid = group_id
        left = [(mr - 1, c), (mr, c), (mr + 1, c)]
        right = [(mr - 1, W - 1 - c), (mr, W - 1 - c), (mr + 1, W - 1 - c)]
        add(left, gid)
        add(right, gid)
        group_id += 1

    if H == 9:
        add([(H - 3, mc - 1), (H - 3, mc), (H - 3, mc + 1)])

    # Bottom stem: center singleton plus inverted self-symmetric U-shape.
    add([(H - 2, mc)])
    add([(H - 1, mc - 1), (H - 1, mc), (H - 1, mc + 1), (H - 2, mc - 1), (H - 2, mc + 1)])

    return chunks, group_ids


def _group_adjacency(
    chunks: list[list[tuple[int, int]]],
    group_ids: list[int],
) -> list[set[int]]:
    n_groups = max(group_ids) + 1
    owner: dict[tuple[int, int], int] = {}
    for idx, chunk in enumerate(chunks):
        for cell in chunk:
            owner[cell] = idx

    adj = [set() for _ in range(n_groups)]
    for idx, chunk in enumerate(chunks):
        gi = group_ids[idx]
        for r, c in chunk:
            for nb in _neighbors4(r, c):
                j = owner.get(nb)
                if j is None or j == idx:
                    continue
                gj = group_ids[j]
                if gi != gj:
                    adj[gi].add(gj)
                    adj[gj].add(gi)
    return adj


def _color_groups(
    chunks: list[list[tuple[int, int]]],
    group_ids: list[int],
) -> list[int] | None:
    n_groups = max(group_ids) + 1
    adj = _group_adjacency(chunks, group_ids)
    n_colors = min(_N_COLORS_MAX, n_groups)
    if n_colors < _N_COLORS_MIN:
        return None

    sig_by_group: list[tuple] = [tuple() for _ in range(n_groups)]
    for gi in range(n_groups):
        sigs = []
        for idx, chunk in enumerate(chunks):
            if group_ids[idx] != gi:
                continue
            arr = np.array(chunk, dtype=int)
            r0, c0 = arr.min(axis=0)
            rel = tuple(sorted((int(r) - int(r0), int(c) - int(c0)) for r, c in arr))
            sigs.append(rel)
        sig_by_group[gi] = tuple(sorted(sigs))

    for _ in range(_COLOR_RETRIES):
        palette = list(range(n_colors))
        colors: list[int | None] = [None] * n_groups
        order = sorted(range(n_groups), key=lambda i: len(adj[i]), reverse=True)
        ok = True
        for gi in order:
            forbidden = {colors[gj] for gj in adj[gi] if colors[gj] is not None}
            choices = [col for col in palette if col not in forbidden]
            if not choices:
                ok = False
                break
            same_sig_used = {
                colors[gj]
                for gj in range(n_groups)
                if gj != gi and sig_by_group[gj] == sig_by_group[gi] and colors[gj] is not None
            }
            choices.sort(key=lambda col: ((col in same_sig_used), sum(colors[k] == col for k in range(n_groups))))
            best = [col for col in choices if (col in same_sig_used) == (choices[0] in same_sig_used)]
            colors[gi] = random.choice(best)
        if ok and len(set(int(c) for c in colors if c is not None)) == n_colors:
            return [int(colors[gi]) for gi in group_ids]
    return None


def _build_output() -> tuple[np.ndarray | None, int, dict[str, int] | None]:
    H, W = random.choice(_OUTPUT_DIMS)
    shape_mask = _shape_mask(H, W)
    chunks, group_ids = _fixed_chunks(H, W)
    all_cells = {cell for chunk in chunks for cell in chunk}
    want_cells = {tuple(map(int, rc)) for rc in np.argwhere(shape_mask)}
    if all_cells != want_cells:
        return None, 0, None
    if any(not (1 <= len(chunk) <= _MAX_PIECE_AREA) for chunk in chunks):
        return None, 0, None
    if not (_PIECE_COUNT_MIN <= len(chunks) <= _PIECE_COUNT_MAX):
        return None, 0, None
    piece_colors = _color_groups(chunks, group_ids)
    if piece_colors is None:
        return None, 0, None

    n_colors = len(set(piece_colors))
    bg = random.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < n_colors:
        return None, 0, None
    cmap = {abstract: available[i] for i, abstract in enumerate(sorted(set(piece_colors)))}

    out = np.full((H, W), bg, dtype=int)
    for chunk, abstract_col in zip(chunks, piece_colors):
        real = int(cmap[int(abstract_col)])
        for r, c in chunk:
            out[r, c] = real

    meta = {
        "H": int(H),
        "W": int(W),
        "area": int(shape_mask.sum()),
        "bbox_area": int(H * W),
        "piece_count": int(len(chunks)),
        "n_colors": int(n_colors),
    }
    return out, bg, meta


def _extract_patches(out: np.ndarray, bg: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(set(map(int, np.unique(out))) - {bg}):
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = int(col)
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _template_patch(H: int, W: int, bg: int, col: int) -> np.ndarray:
    patch = np.full((H, W), bg, dtype=int)
    patch[_VALID_TEMPLATE_BITMAPS[(H, W)]] = int(col)
    return patch


def _scatter(out: np.ndarray, bg: int, template_col: int) -> np.ndarray:
    oh, ow = out.shape
    rows_lo = max(_CANVAS_MIN, oh + 8)
    cols_lo = max(_CANVAS_MIN, ow + 10)
    rows = random.randint(rows_lo, _CANVAS_MAX)
    cols = random.randint(cols_lo, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    template = _template_patch(oh, ow, bg, template_col)
    tr, tc = random.choice(_corner_origins(rows, cols, oh, ow))
    canvas[tr : tr + oh, tc : tc + ow] = _EMPTY

    for patch in _extract_patches(out, bg):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            border=1,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[tr : tr + oh, tc : tc + ow] = bg
    paste(canvas, template, tr, tc, transparent=bg)
    return canvas


def _piece_count(out: np.ndarray, bg: int) -> int:
    total = 0
    for col in sorted(set(map(int, np.unique(out))) - {bg}):
        total += len(components_mask(out == col, connectivity=4))
    return total


def _is_valid_output(out: np.ndarray, bg: int) -> bool:
    H, W = out.shape
    if (H, W) not in _VALID_TEMPLATE_BITMAPS:
        return False
    if not np.array_equal(out, out[:, ::-1]):
        return False
    occupied = (out != bg)
    if not np.array_equal(occupied, _shape_mask(H, W)):
        return False
    visible_colors = sorted(set(map(int, np.unique(out))) - {bg})
    if not (_N_COLORS_MIN <= len(visible_colors) <= _N_COLORS_MAX):
        return False
    if not (_PIECE_COUNT_MIN <= _piece_count(out, bg) <= _PIECE_COUNT_MAX):
        return False
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in visible_colors
        for pts in components_mask(out == col, connectivity=4)
    ):
        return False
    return True


def _gen_attempt(return_meta: bool = False):
    out, bg, meta = _build_output()
    if out is None or meta is None:
        return (None, None) if return_meta else None

    H, W = out.shape
    visible_colors = sorted(set(map(int, np.unique(out))) - {bg})
    if not (_N_COLORS_MIN <= len(visible_colors) <= _N_COLORS_MAX):
        return (None, None) if return_meta else None
    if not np.array_equal(out, out[:, ::-1]):
        return (None, None) if return_meta else None
    if _piece_count(out, bg) != meta["piece_count"]:
        return (None, None) if return_meta else None
    if not (_PIECE_COUNT_MIN <= meta["piece_count"] <= _PIECE_COUNT_MAX):
        return (None, None) if return_meta else None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in visible_colors
        for pts in components_mask(out == col, connectivity=4)
    ):
        return (None, None) if return_meta else None

    available = [c for c in range(10) if c not in set(visible_colors) and c != bg]
    if not available:
        return (None, None) if return_meta else None
    template_col = random.choice(available)

    try:
        inp = _scatter(out, bg, template_col)
        got = main(inp)
    except Exception:
        return (None, None) if return_meta else None

    if not _is_valid_output(got, bg):
        return (None, None) if return_meta else None

    meta = dict(meta)
    meta["template_col"] = int(template_col)
    meta["template_area"] = int(_VALID_TEMPLATE_BITMAPS[(H, W)].sum())
    meta["solver_piece_count"] = int(_piece_count(got, bg))
    return (inp, meta) if return_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
