# Combo variant of e8686506 (index 18)
# combo_spec: D3/0+D15/2
# Name: 非矩形目标+模板展示轮廓

# Design analysis for the combo variant
# a) Concrete behavior and step-by-step trace:
#    - Original task flow:
#        treat every non-bg component as a real piece
#        -> infer a dense rectangle from total area
#        -> use LR symmetry plus optional edge anchors to place pieces.
#    - Combo flow:
#        first detect a corner-flush monochrome outline template
#        -> read from that template the exact non-rectangular silhouette (thick cross or thick L)
#        -> erase the template from the piece inventory
#        -> assemble the remaining scattered fragments into that inferred silhouette.
#    - Concrete example:
#        suppose the target is a 7x5 L_bl shape. The input corner contains a 7x5 single-color
#        outline matching the L boundary, while the actual colored fragments of the final L are
#        all scattered elsewhere. The old rectangle-area heuristic would be wrong twice:
#          1. the template is not a piece and would pollute total area;
#          2. the destination is not a filled rectangle.
#        The new solver must therefore do "read template -> infer mask -> pack pieces in mask".
# b) Complexity calibration:
#    - The non-rectangular family is intentionally small and visually stable: one thick cross
#      family plus four thick L orientations. This keeps the rule observable in 2-4 examples.
#    - The template is same-bbox but still small relative to the full canvas, so it visibly acts
#      as a legend rather than another scattered fragment. It gives shape and orientation, while
#      colors still come only from the real pieces.
#    - We do NOT keep the D3 single-variant's preserved boundary anchor among the real pieces;
#      otherwise the template would become redundant and the fusion would degenerate.
# c) Hidden-assumption audit against the original main():
#    - `_factors(total_area)` and closest-to-square rectangle selection are invalid because the
#      destination is no longer a dense rectangle.
#    - LR mirror constraints are invalid because L targets are not LR-symmetric.
#    - Corner/bar/vertical-line anchor heuristics assume rectangle borders and would misfire on
#      cross/L silhouettes, so they are removed entirely.
#    - "all non-bg cells are puzzle pieces" is no longer true because the corner template is a
#      separate symbolic object and must be stripped before counting components or area.
# d) Placement legality:
#    - The generator is output-first. It creates a target cross/L, partitions it into 4-connected
#      chunks, scatters those chunks on a larger background canvas, and reserves a corner bbox for
#      the template so no fragment overlaps or touches it.
#    - No rotations or flips are applied to fragments after extraction; all placement is by
#      translation only, matching the solver's assumptions.
# e) Visual observability:
#    - Across examples, the corner object is always a monochrome outline, flush to one corner,
#      and not reused in the output colors. Humans can see it is a legend, not a piece.
#    - The output visibly preserves the template's silhouette but fills it with scattered colored
#      pieces. This creates the intended reasoning chain: read silhouette first, then assemble.
# f) Visual cleanliness:
#    - Only the template occupies the corner reserved zone. All real pieces are scattered away
#      from it with padding, so there is no ambiguous overlay between hint and puzzle material.
#    - The output is cropped to the silhouette bbox, with background left only outside the cross/L
#      occupancy, so the non-rectangular destination reads clearly.
# g) Color inferability:
#    - The template uses one color that does not appear in the output. Output colors are assigned
#      to adjacency-aware chunks so equal colors never 4-touch. There is no hidden dependence on
#      numeric color ids or ordering.
# h) New solver assumptions guaranteed by the generator:
#    - Exactly one corner-anchored component matches the boundary bitmap of an allowed target mask.
#    - After removing that template, the remaining colored area equals the occupied area of the
#      inferred cross/L silhouette.
#    - Real puzzle pieces do not themselves contain an in-place preserved silhouette boundary cue;
#      the template is the only direct source of the target outline.
#    - Oracle validation keeps only instances that the solver reconstructs exactly.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "solvers" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("could not locate repo root containing solvers/arc_utils.py")


_ROOT = _find_repo_root()
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


def _shape_mask(kind: str, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    if kind == "cross":
        mr, mc = H // 2, W // 2
        mask[:, max(0, mc - 1) : min(W, mc + 2)] = True
        mask[max(0, mr - 1) : min(H, mr + 2), :] = True
        return mask

    base = np.zeros((H, W), dtype=bool)
    base[:, :3] = True
    base[H - 3 :, :] = True
    if kind == "L_bl":
        return base
    if kind == "L_br":
        return base[:, ::-1]
    if kind == "L_tl":
        return base[::-1, :]
    if kind == "L_tr":
        return base[::-1, ::-1]
    raise ValueError(f"unknown shape kind: {kind}")


def _candidate_masks_for_bbox(H: int, W: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    if H >= 5 and W >= 5 and H % 2 == 1 and W % 2 == 1:
        out.append(("cross", _shape_mask("cross", H, W)))
    if H >= 5 and W >= 5:
        for kind in ("L_bl", "L_br", "L_tl", "L_tr"):
            out.append((kind, _shape_mask(kind, H, W)))
    return out


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


def _bitmap_from_pts(pts: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    bmp = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        bmp[int(r) - r0, int(c) - c0] = True
    return bmp, (int(r0), int(c0))


def _is_corner_flush(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> bool:
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == W - 1)
        or (r1 == H - 1 and c0 == 0)
        or (r1 == H - 1 and c1 == W - 1)
    )


def _find_template(
    g: np.ndarray, bg: int
) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int]]:
    GH, GW = g.shape
    all_non_bg = int(np.count_nonzero(g != bg))
    candidates: list[tuple[np.ndarray, np.ndarray, str, tuple[int, int]]] = []

    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            r0, c0, r1, c1 = bbox_coords(pts)
            if not _is_corner_flush(int(r0), int(c0), int(r1), int(c1), GH, GW):
                continue
            bmp, top_left = _bitmap_from_pts(pts)
            th, tw = bmp.shape
            for kind, shape_mask in _candidate_masks_for_bbox(th, tw):
                if not np.array_equal(bmp, _boundary_mask(shape_mask)):
                    continue
                if all_non_bg - len(pts) != int(shape_mask.sum()):
                    continue
                full_mask = np.zeros_like(g, dtype=bool)
                for r, c in pts:
                    full_mask[int(r), int(c)] = True
                candidates.append((full_mask, shape_mask, kind, top_left))

    assert len(candidates) == 1, "input must contain exactly one corner template for a valid cross/L silhouette"
    return candidates[0]


def _extract_piece_types(
    g: np.ndarray, bg: int
) -> tuple[list[dict], dict[int, int], int]:
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
        ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])
    ):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append(
            {"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)}
        )

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    return types, color_piece_count, int(total_area)


def _assemble(
    types: list[dict],
    shape_mask: np.ndarray,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    H, W = shape_mask.shape
    board = np.full((H, W), _OUTSIDE, dtype=int)
    board[shape_mask] = _EMPTY
    counts = [int(t["cnt"]) for t in types]
    fail_cache: set[tuple[bytes, tuple[int, ...]]] = set()

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        piece_cells = {(r0 + dr, c0 + dc) for dr, dc in t["coords"]}
        for r, c in piece_cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if board[r, c] != _EMPTY:
                return False
            if req[r, c] not in (_EMPTY, col):
                return False
        if disallow_touch and col in disallow_touch:
            for r, c in piece_cells:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (rr, cc) in piece_cells:
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

    def search() -> bool:
        key = (board.tobytes(), tuple(counts))
        if key in fail_cache:
            return False

        forced: list[tuple[int, list[tuple[int, int]]]] = []

        def rollback_forced() -> None:
            for ti_forced, changed in reversed(forced):
                unplace(changed)
                counts[ti_forced] += 1

        while True:
            progress = False
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                places = placements_for(ti)
                if not places:
                    rollback_forced()
                    fail_cache.add(key)
                    return False
                if len(places) == 1:
                    counts[ti] -= 1
                    changed = place(ti, places[0][0], places[0][1])
                    forced.append((ti, changed))
                    progress = True
            if not progress:
                break

        if all(c == 0 for c in counts):
            if np.any(board[shape_mask] == _EMPTY):
                rollback_forced()
                fail_cache.add(key)
                return False
            return True

        choices = []
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            if not places:
                rollback_forced()
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

        rollback_forced()
        fail_cache.add(key)
        return False

    assert search(), "assembly search failed"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a compact cross or L silhouette using a corner outline template.

    Input structure:
    - The most common color is background, and one corner contains a monochrome non-rectangular outline template.
    - That template is not a puzzle piece: it only shows the target silhouette's boundary, shape family, and orientation.
    - Every other non-background component is a translated fragment from the final cross/L assembly.

    Rule:
    - Detect the corner template, read its outlined silhouette, and remove it from the fragment inventory.
    - Treat the remaining components as puzzle pieces and infer the occupied output mask from the template.
    - Pack the fragments into that non-rectangular mask so every occupied cell is filled exactly once and same-color copies never 4-touch.

    Key insight:
    - The crucial first move is separating symbol from substance: the corner outline tells you where the final shape exists, while the scattered colored pieces provide only the fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    template_mask, shape_mask, _shape_kind, _top_left = _find_template(g, bg)
    piece_grid = g.copy()
    piece_grid[template_mask] = bg

    types, color_piece_count, total_area = _extract_piece_types(piece_grid, bg)
    assert total_area > 0, "grid must contain non-template puzzle fragments"
    assert total_area == int(shape_mask.sum()), "piece area must match the template-implied silhouette area"

    req = np.full(shape_mask.shape, _OUTSIDE, dtype=int)
    req[shape_mask] = _EMPTY
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}

    solved = _assemble(types, shape_mask, req, disallow_touch)
    out = np.full(shape_mask.shape, bg, dtype=int)
    out[shape_mask] = solved[shape_mask]
    return out


# --------------- generator (combo: template-driven non-rectangular assembly) ---------------

_OUTPUT_DIMS = [(7, 5), (7, 7), (9, 5), (9, 7)]
_OUTPUT_KINDS = ["cross", "L_bl", "L_br", "L_tl", "L_tr"]
_N_OUTPUT_COLORS_MIN = 4
_N_OUTPUT_COLORS_MAX = 6
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_CANVAS_MIN = 17
_CANVAS_MAX = 20
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 500
_PARTITION_RETRIES = 120
_COLOR_RETRIES = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _partition_shape(mask: np.ndarray) -> list[list[tuple[int, int]]] | None:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not cells:
        return []

    size_choices = [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]

    for _ in range(_PARTITION_RETRIES):
        remaining = set(cells)
        chunks: list[list[tuple[int, int]]] = []
        while remaining:
            seed = random.choice(tuple(remaining))
            target = min(random.choice(size_choices), len(remaining))
            chunk = [seed]
            remaining.remove(seed)
            frontier = {nb for nb in _neighbors4(*seed) if nb in remaining}
            while frontier and len(chunk) < target:
                ranked = sorted(
                    frontier,
                    key=lambda rc: (
                        sum(nb in remaining for nb in _neighbors4(*rc)),
                        random.random(),
                    ),
                    reverse=True,
                )
                nxt = ranked[0]
                frontier.remove(nxt)
                if nxt not in remaining:
                    continue
                chunk.append(nxt)
                remaining.remove(nxt)
                for nb in _neighbors4(*nxt):
                    if nb in remaining:
                        frontier.add(nb)
            chunks.append(sorted(chunk))

        if not all(1 <= len(chunk) <= _MAX_PIECE_AREA for chunk in chunks):
            continue
        if not (_PIECE_COUNT_MIN <= len(chunks) <= _PIECE_COUNT_MAX):
            continue
        if sum(len(chunk) == 1 for chunk in chunks) > 4:
            continue
        return chunks
    return None


def _chunk_adjacency(chunks: list[list[tuple[int, int]]]) -> list[set[int]]:
    owner = {}
    for idx, chunk in enumerate(chunks):
        for cell in chunk:
            owner[cell] = idx
    adj = [set() for _ in chunks]
    for idx, chunk in enumerate(chunks):
        for r, c in chunk:
            for nb in _neighbors4(r, c):
                j = owner.get(nb)
                if j is not None and j != idx:
                    adj[idx].add(j)
                    adj[j].add(idx)
    return adj


def _color_chunks(chunks: list[list[tuple[int, int]]]) -> tuple[list[int], int] | tuple[None, None]:
    adj = _chunk_adjacency(chunks)
    max_colors = min(_N_OUTPUT_COLORS_MAX, len(chunks))
    min_colors = min(_N_OUTPUT_COLORS_MIN, max_colors)
    order = sorted(
        range(len(chunks)),
        key=lambda i: (len(adj[i]), len(chunks[i]), random.random()),
        reverse=True,
    )

    for _ in range(_COLOR_RETRIES):
        n_colors = random.randint(min_colors, max_colors)
        palette = list(range(n_colors))
        colors: list[int | None] = [None] * len(chunks)
        usage = {col: 0 for col in palette}
        ok = True

        for idx in order:
            forbidden = {colors[j] for j in adj[idx] if colors[j] is not None}
            choices = [col for col in palette if col not in forbidden]
            if not choices:
                ok = False
                break
            choices.sort(key=lambda col: (usage[col], random.random()))
            chosen = choices[0]
            colors[idx] = chosen
            usage[chosen] += 1

        if not ok:
            continue
        used = {int(c) for c in colors if c is not None}
        if len(used) != n_colors:
            continue
        return [int(c) for c in colors], n_colors
    return None, None


def _build_output() -> tuple[np.ndarray | None, int, np.ndarray | None, dict | None]:
    kind = "cross" if random.random() < 0.65 else random.choice(_OUTPUT_KINDS[1:])
    H, W = random.choice(_OUTPUT_DIMS)
    if kind == "cross" and (H % 2 == 0 or W % 2 == 0):
        return None, 0, None, None

    shape_mask = _shape_mask(kind, H, W)
    chunks = _partition_shape(shape_mask)
    if chunks is None:
        return None, 0, None, None

    chunk_colors, n_colors = _color_chunks(chunks)
    if chunk_colors is None or n_colors is None:
        return None, 0, None, None

    bg = random.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < n_colors + 1:
        return None, 0, None, None

    real_palette = available[:n_colors]
    template_color = int(available[n_colors])

    out = np.full((H, W), bg, dtype=int)
    for abs_col, chunk in zip(chunk_colors, chunks):
        real_col = int(real_palette[abs_col])
        for r, c in chunk:
            out[int(r), int(c)] = real_col

    template = np.full((H, W), bg, dtype=int)
    template[_boundary_mask(shape_mask)] = template_color

    meta = {
        "kind": kind,
        "dims": (H, W),
        "piece_count": len(chunks),
        "template_area": int(np.count_nonzero(template != bg)),
        "occupied_area": int(shape_mask.sum()),
    }
    return out, bg, template, meta


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


def _corner_origins(rows: int, cols: int, h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, cols - w),
        (rows - h, 0),
        (rows - h, cols - w),
    ]


def _scatter(out: np.ndarray, bg: int, template: np.ndarray) -> np.ndarray:
    oh, ow = out.shape
    rows_lo = max(_CANVAS_MIN, oh + 7)
    cols_lo = max(_CANVAS_MIN, ow + 9)
    rows = random.randint(rows_lo, _CANVAS_MAX)
    cols = random.randint(cols_lo, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

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


def _is_valid_variant_output(out: np.ndarray, bg: int) -> bool:
    occupied = out != bg
    comps = components_mask(occupied, connectivity=4)
    if len(comps) != 1:
        return False
    r0, c0, r1, c1 = bbox_coords(comps[0])
    bmp = occupied[r0 : r1 + 1, c0 : c1 + 1]
    H, W = bmp.shape
    return any(np.array_equal(bmp, mask) for _, mask in _candidate_masks_for_bbox(H, W))


def _gen_attempt(return_meta: bool = False):
    out, bg, template, meta = _build_output()
    if out is None or template is None or meta is None:
        return (None, None) if return_meta else None

    visible_colors = sorted(set(map(int, np.unique(out))) - {bg})
    if not (_N_OUTPUT_COLORS_MIN <= len(visible_colors) <= _N_OUTPUT_COLORS_MAX):
        return (None, None) if return_meta else None
    if not (_PIECE_COUNT_MIN <= _piece_count(out, bg) <= _PIECE_COUNT_MAX):
        return (None, None) if return_meta else None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in visible_colors
        for pts in components_mask(out == col, connectivity=4)
    ):
        return (None, None) if return_meta else None

    try:
        inp = _scatter(out, bg, template)
        got = main(inp)
    except Exception:
        return (None, None) if return_meta else None

    if not np.array_equal(got, out):
        return (None, None) if return_meta else None
    if not _is_valid_variant_output(got, bg):
        return (None, None) if return_meta else None
    return (inp, meta) if return_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
