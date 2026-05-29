# Variant of e8686506
# Dimension: D3 — 目标结构：拼合后的形状要求 (dst)
# Change: 拼合后形成一个紧凑的矩形复合体，所有碎片无间隙无重叠地拼成完整矩形... → 拼合成十字形或L形

# Design analysis for the D3 variant
# a) Concrete behavior:
#    The output is no longer a fully filled rectangle. It is a cropped bbox that still contains
#    background cells, while the occupied cells form either:
#      - a thick cross (3-cell-thick vertical bar + 3-cell-thick horizontal bar), or
#      - a thick L polyomino (3-cell-thick upright bar + 3-cell-thick foot), possibly flipped.
#    Input construction still follows "scatter pieces on a large background canvas", but one
#    special boundary piece is kept grouped in-place inside the target bbox. That boundary anchor
#    lets the solver infer which non-rectangular silhouette to assemble. In the original task the
#    grouped anchor only fixed the rectangle bbox; here the grouped anchor fixes the shape family.
#    Example trace: original solver computed output dims from total area, then filled every cell of
#    that rectangle. In this variant the same total area could fit several rectangles, so the
#    solver must first detect the preserved cross/L boundary, derive the exact occupied mask, and
#    only then place the scattered interior fragments into that mask.
# b) Complexity calibration:
#    The variant keeps a small controlled family of target silhouettes: one cross family and one
#    L family with four visual orientations. That keeps the task observable while avoiding too
#    many hidden branches. The boundary anchor is large and distinctive so the non-rectangular
#    dst change is obvious in 2-3 examples.
# c) Hidden-assumption audit against the original main():
#      - `_factors(total_area)` and "closest square portrait rectangle" are invalid, because the
#        target mask is no longer rectangular.
#      - LR symmetry is invalid for L-shapes, so mirror-based placement must be removed.
#      - corner/hbar/vline heuristics assume rectangle edges and would misplace constraints here.
#      - "singletons only on the outer border of the rectangle" is no longer a meaningful rule.
#    Therefore the solver is rewritten around an explicit shape mask inferred from the boundary.
# d) Placement legality:
#    The generator reserves the entire target bbox on the input canvas, scatters only non-frame
#    pieces outside that bbox, and then pastes the boundary anchor back into the reserved region.
#    This guarantees no overlap, no out-of-bounds placement, and a clean silhouette cue for the
#    solver.
# e) Visual observability:
#    The non-rectangular mask is visible twice: as a hollow-looking boundary in the input and as a
#    fully assembled cross/L in the output. No color-id-based ordering is used anywhere; all
#    inference is driven by visible geometry.
# f) Visual cleanliness:
#    Only the boundary anchor stays inside the bbox; all fill fragments are scattered elsewhere.
#    That avoids visual pile-up inside one region and keeps the target silhouette readable.
# g) Color inference:
#    Colors are remapped randomly, but the frame color is never reused for fill chunks. Fill
#    chunks are properly colored so equal-colored chunks never 4-touch, making component structure
#    visible without any hidden palette convention.
# h) New solver assumptions and generator guarantees:
#      - The input contains exactly one preserved boundary anchor whose bitmap matches the boundary
#        of one allowed cross/L silhouette and whose implied filled area equals the total fragment
#        area. `generate_input()` enforces this by output-first generation plus oracle validation.
#      - The frame color is unique to the boundary anchor, so same-color-touch constraints only
#        apply to scattered fill chunks.
#      - The solver assumes pieces are translated only; the generator never rotates or flips any
#        extracted patch after output construction.

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


def _shape_mask(kind: str, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    if kind == "cross":
        mr, mc = H // 2, W // 2
        for r in range(H):
            for c in range(W):
                if abs(r - mr) <= 1 or abs(c - mc) <= 1:
                    mask[r, c] = True
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


def _bitmap_from_parts(parts: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, tuple[int, int]]:
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0, r1, c1 = bbox_coords(all_pts)
    bmp = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return bmp, (r0, c0)


def _candidate_masks(H: int, W: int, total_area: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    if H >= 5 and W >= 5 and H % 2 == 1 and W % 2 == 1:
        cross = _shape_mask("cross", H, W)
        if int(cross.sum()) == total_area:
            out.append(("cross", cross))
    if H >= 5 and W >= 5:
        for kind in ("L_bl", "L_br", "L_tl", "L_tr"):
            mask = _shape_mask(kind, H, W)
            if int(mask.sum()) == total_area:
                out.append((kind, mask))
    return out


def _find_frame_and_shape(
    g: np.ndarray,
    bg: int,
    total_area: int,
) -> tuple[list[tuple[int, np.ndarray]], np.ndarray, tuple[int, int], str]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    candidates = []
    seen = set()
    for i, part1 in enumerate(comps8_all):
        for parts in ([part1],):
            bmp, top_left = _bitmap_from_parts(parts)
            H, W = bmp.shape
            for kind, mask in _candidate_masks(H, W, total_area):
                if np.array_equal(bmp, _boundary_mask(mask)):
                    sig = (tuple(sorted((int(c), tuple(map(tuple, p.tolist()))) for c, p in parts)), kind)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append((parts, mask, top_left, kind))

    assert len(candidates) == 1, "input must contain exactly one preserved cross/L boundary anchor"
    return candidates[0]


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
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W):
                return False
            if board[r, c] != _EMPTY:
                return False
            if req[r, c] not in (_EMPTY, col):
                return False
        if disallow_touch and col in disallow_touch:
            piece_cells = {(r0 + dr, c0 + dc) for dr, dc in t["coords"]}
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
    Assemble scattered colored fragments into a compact non-rectangular cross or L silhouette.

    Input structure:
    - The background is the most frequent color; every other component is a translated puzzle fragment.
    - One special grouped boundary fragment remains inside the future output bbox, visibly outlining a cross or an L.

    Rule:
    - Collect all non-background connected components as puzzle pieces.
    - Detect the preserved boundary anchor, infer from it which thick cross/L mask the output occupies, and lock those boundary cells.
    - Pack every remaining fragment into the inferred mask so all occupied cells are filled exactly once, while different pieces of the same color never 4-touch.

    Key insight:
    - The grouped boundary fragment no longer just tells you a rectangle size; it reveals the exact non-convex target silhouette, and that mask constrains where every scattered interior fragment can go.
    """
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
    assert total_area > 0, "grid must contain non-background fragments"

    frame_parts, shape_mask, (fr0, fc0), _shape_kind = _find_frame_and_shape(g, bg, total_area)

    H, W = shape_mask.shape
    req = np.full((H, W), _OUTSIDE, dtype=int)
    req[shape_mask] = _EMPTY
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = int(col)

    frame_colors = {int(col) for col, _ in frame_parts}
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1 and col not in frame_colors}
    solved = _assemble(types, shape_mask, req, disallow_touch)

    out = np.full((H, W), bg, dtype=int)
    out[shape_mask] = solved[shape_mask]
    return out


# --------------- generator (o5x) ---------------

_OUTPUT_DIMS = [(7, 5), (7, 7), (9, 5), (9, 7), (9, 9)]
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400
_PARTITION_RETRIES = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _partition_interior(mask: np.ndarray) -> list[list[tuple[int, int]]] | None:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not cells:
        return []

    for _ in range(_PARTITION_RETRIES):
        remaining = set(cells)
        chunks: list[list[tuple[int, int]]] = []
        while remaining:
            seed = random.choice(tuple(remaining))
            target = random.choice([1, 1, 1, 2, 2, 3, 4])
            chunk = [seed]
            remaining.remove(seed)
            frontier = {nb for nb in _neighbors4(*seed) if nb in remaining}
            while frontier and len(chunk) < target:
                nxt = random.choice(tuple(frontier))
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
        total_pieces = 1 + len(chunks)
        if _PIECE_COUNT_MIN <= total_pieces <= _PIECE_COUNT_MAX:
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
    max_fill = min(_N_COLORS_MAX - 1, len(chunks))
    for _ in range(80):
        n_fill = random.randint(_N_COLORS_MIN - 1, max_fill)
        palette = list(range(1, n_fill + 1))
        colors = [None] * len(chunks)
        order = sorted(range(len(chunks)), key=lambda i: (len(adj[i]), len(chunks[i])), reverse=True)
        ok = True
        for idx in order:
            forbidden = {colors[j] for j in adj[idx] if colors[j] is not None}
            choices = [col for col in palette if col not in forbidden]
            if not choices:
                ok = False
                break
            colors[idx] = random.choice(choices)
        if ok and len(set(colors)) == n_fill:
            return [int(c) for c in colors], n_fill
    return None, None


def _build_output() -> tuple[np.ndarray | None, int, set[int], dict[str, str] | None]:
    family = "cross" if random.random() < 0.5 else random.choice(["L_bl", "L_br", "L_tl", "L_tr"])
    H, W = random.choice(_OUTPUT_DIMS)
    shape_mask = _shape_mask(family, H, W)
    frame_mask = _boundary_mask(shape_mask)
    chunks = _partition_interior(shape_mask & ~frame_mask)
    if chunks is None:
        return None, 0, set(), None
    chunk_colors, n_fill = _color_chunks(chunks)
    if chunk_colors is None or n_fill is None:
        return None, 0, set(), None

    n_colors = 1 + n_fill
    abstract = np.full((H, W), -1, dtype=int)
    abstract[frame_mask] = 0
    for color, chunk in zip(chunk_colors, chunks):
        for r, c in chunk:
            abstract[r, c] = color
    bg = random.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < n_colors:
        return None, 0, set(), None
    cmap = {a: available[a] for a in range(n_colors)}
    out = np.full((H, W), bg, dtype=int)
    for a, real in cmap.items():
        out[abstract == a] = real
    meta = {"family": family, "dims": f"{H}x{W}"}
    return out, bg, {cmap[0]}, meta


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col == bg or col in frame_colors:
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


def _scatter(out: np.ndarray, bg: int, frame_colors: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape
    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("bbox does not fit canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = _EMPTY
    for patch in _extract_non_frame_patches(out, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.isin(out, list(frame_colors))
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray, bg: int) -> int:
    total = 0
    for col in sorted(set(map(int, np.unique(out))) - {bg}):
        total += len(components_mask(out == col, connectivity=4))
    return total


def _is_valid_variant_output(out: np.ndarray) -> bool:
    for bg in sorted(map(int, np.unique(out))):
        occupied = out != bg
        if not np.any(occupied):
            continue
        comps = components_mask(occupied, connectivity=4)
        if len(comps) != 1:
            continue
        r0, c0, r1, c1 = bbox_coords(comps[0])
        H, W = r1 - r0 + 1, c1 - c0 + 1
        bmp = occupied[r0 : r1 + 1, c0 : c1 + 1]
        for _kind, mask in _candidate_masks(H, W, int(bmp.sum())):
            if np.array_equal(bmp, mask):
                return True
    return False


def _gen_attempt(return_meta: bool = False):
    out, bg, frame_colors, meta = _build_output()
    if out is None or meta is None:
        return None if not return_meta else (None, None)

    visible_colors = sorted(set(map(int, np.unique(out))) - {bg})
    if not (_N_COLORS_MIN <= len(visible_colors) <= _N_COLORS_MAX):
        return None if not return_meta else (None, None)
    pc = _piece_count(out, bg)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None if not return_meta else (None, None)
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in visible_colors
        if col not in frame_colors
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None if not return_meta else (None, None)

    try:
        inp = _scatter(out, bg, frame_colors)
        got = main(inp)
    except Exception:
        return None if not return_meta else (None, None)
    if not _is_valid_variant_output(got):
        return None if not return_meta else (None, None)
    return (inp, meta) if return_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D3 input")
