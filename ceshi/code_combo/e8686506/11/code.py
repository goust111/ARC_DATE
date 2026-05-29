from __future__ import annotations

# Combo variant of e8686506 (index 11)
# combo_spec: D3/0+D10/0
# Name: 拼成非矩形+无框架碎片
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task uses total area -> closest portrait rectangle, then relies on a preserved
#    frame piece plus optional edge markers to lock that rectangle. Here both cues are removed.
#    Example: a total fragment area of 33 would send the original solver to an 11x3 rectangle,
#    but this combo instead allows several cropped non-rectangular LR-symmetric silhouettes with
#    the same area, such as a 7x7 thick cross or a 9x5 thick T. The input now contains only
#    scattered small fragments. The solver must enumerate candidate symmetric silhouettes, then
#    pack the pieces so the occupied cells form one of those masks exactly.
# b) Complexity calibration:
#    To keep the rule visible but not underconstrained, the output family is narrowed to two small
#    symmetric non-convex silhouette: a thick cross. This preserves the D3
#    "non-rectangular destination" change while making the D10 no-frame setting tractable: human
#    solvers can see from training outputs that the target is cropped and non-rectangular, and the
#    solver only has to infer the exact cross outline instead of an arbitrary polygon.
# c) Hidden-assumption audit against the original main():
#      - `_factors(total_area)` no longer identifies the output; multiple candidate masks can share
#        the same area, and none are rectangles.
#      - `_build_requirements(...)` is invalid because corners / edge bars / paired border lines
#        all assume a rectangular frame.
#      - The "1-2 grouped frame parts whose union bbox equals output dims" assumption is removed.
#      - LR symmetry is still retained, but now it constrains a non-rectangular mask instead of a
#        dense rectangle. The solver therefore keeps the symmetry check while replacing frame locks
#        with candidate-mask search plus backtracking.
# d) Placement legality:
#    The generator is output-first. It partitions the chosen silhouette into only small 4-connected
#    chunks, extracts every chunk as its own patch, and scatters all patches on the canvas with
#    Chebyshev padding. No chunk is preserved in-place, so there is no accidental frame anchor.
# e) Visual observability:
#    Across training pairs, outputs visibly become compact cropped crosses/T-shapes with background
#    holes in the bbox, unlike the original full rectangle. Inputs visibly lack any large grouped
#    outline piece; all visible objects are similarly small fragments. The rule is inferable from
#    geometry alone, not from color ids or invisible ordering conventions.
# f) Visual cleanliness:
#    The generator uses 4-6 colors, 9-17 pieces, and small chunk areas so the canvas is busy but
#    readable. Scatter padding prevents accidental touching in the input, and same-color chunks are
#    graph-colored so different pieces of the same color never merge in the output.
# g) Color inference:
#    Colors are remapped randomly after construction. Mirror-related chunks share color, but color
#    itself carries no hidden ordering role; it only exposes component identity and symmetry.
# h) New solver assumptions and generator guarantees:
#      - The occupied output mask is a thick LR-symmetric cross with dimensions drawn from
#        `_LAYOUTS`; `generate_input()` only emits such cases.
#      - All pieces are translated only; no rotation or reflection is applied after extraction.
#      - Same-color pieces may repeat, but different components of the same color never 4-touch in
#        the true output; the generator enforces this and the solver uses it as a placement filter.
# i) Extra audit for the new no-frame solver:
#      - Every candidate loop in `main()` assumes the chosen silhouette mask is LR-symmetric; the
#        candidate table guarantees that.
#      - Every placement test assumes mirror cells must end up with the same color; the generator
#        builds LR-symmetric outputs so this holds.
#      - Backtracking assumes the valid puzzle is small enough for search; the generator rejects any
#        sample that the solver cannot recover quickly and exactly.

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers").is_dir():
        _ROOT = _parent
        break
if _ROOT is None:
    raise RuntimeError("could not locate repository root containing solvers/")
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_OUTSIDE = -2
_EMPTY = -1

_LAYOUTS = [("cross", 7, 5)]

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_MAX_PIECE_AREA = 4
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 500
_PARTITION_RETRIES = 140
_COLOR_RETRIES = 120


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _shape_mask(kind: str, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mid = W // 2
    if kind == "cross":
        mr = H // 2
        for r in range(H):
            for c in range(W):
                if abs(r - mr) <= 1 or abs(c - mid) <= 1:
                    mask[r, c] = True
        return mask
    raise ValueError(f"unknown shape kind: {kind}")


def _candidate_masks(total_area: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for kind, H, W in _LAYOUTS:
        mask = _shape_mask(kind, H, W)
        if int(mask.sum()) == total_area:
            out.append((kind, mask))
    return out


def _build_types(g: np.ndarray, bg: int) -> tuple[list[dict], dict[int, int]]:
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
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})
    return types, color_piece_count


def _assemble(types: list[dict], shape_mask: np.ndarray, disallow_touch: set[int]) -> np.ndarray | None:
    H, W = shape_mask.shape
    board = np.full((H, W), _OUTSIDE, dtype=int)
    board[shape_mask] = _EMPTY
    counts = [int(t["cnt"]) for t in types]
    fail_cache: set[tuple[bytes, tuple[int, ...]]] = set()

    static_places: list[list[tuple[int, int]]] = []
    static_cells: list[list[tuple[tuple[int, int], ...]]] = []
    for t in types:
        t_places = []
        t_cells = []
        for r0 in range(H - int(t["h"]) + 1):
            for c0 in range(W - int(t["w"]) + 1):
                ok = True
                cells = []
                for dr, dc in t["coords"]:
                    r, c = r0 + dr, c0 + dc
                    if not shape_mask[r, c]:
                        ok = False
                        break
                    cells.append((r, c))
                if ok:
                    t_places.append((r0, c0))
                    t_cells.append(tuple(cells))
        static_places.append(t_places)
        static_cells.append(t_cells)

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        if not shape_mask[r, mc]:
            return False
        v = board[r, mc]
        return v in (_EMPTY, col)

    def can_place_cells(ti: int, piece_cells: tuple[tuple[int, int], ...]) -> bool:
        t = types[ti]
        col = int(t["col"])
        piece_set = set(piece_cells)
        for r, c in piece_cells:
            if board[r, c] != _EMPTY:
                return False
            if not mirror_ok(r, c, col):
                return False
        if col in disallow_touch:
            for r, c in piece_cells:
                for rr, cc in _neighbors4(r, c):
                    if (rr, cc) in piece_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def placements_for(ti: int) -> list[int]:
        return [pi for pi, cells in enumerate(static_cells[ti]) if can_place_cells(ti, cells)]

    def place(ti: int, pi: int) -> list[tuple[int, int]]:
        t = types[ti]
        col = int(t["col"])
        changed = []
        for r, c in static_cells[ti][pi]:
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
                    changed = place(ti, places[0])
                    forced.append((ti, changed))
                    progress = True
            if not progress:
                break

        if all(c == 0 for c in counts):
            if np.any(board[shape_mask] == _EMPTY):
                rollback_forced()
                fail_cache.add(key)
                return False
            if not np.array_equal(board[:, ::-1], board):
                rollback_forced()
                fail_cache.add(key)
                return False
            return True

        valid_places: dict[int, list[int]] = {}
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            if not places:
                rollback_forced()
                fail_cache.add(key)
                return False
            valid_places[ti] = places

        best_options = None
        for r in range(H):
            for c in range(W):
                if board[r, c] != _EMPTY:
                    continue
                options = []
                for ti, places in valid_places.items():
                    for pi in places:
                        if (r, c) in static_cells[ti][pi]:
                            options.append((ti, pi))
                if not options:
                    rollback_forced()
                    fail_cache.add(key)
                    return False
                if best_options is None or len(options) < len(best_options):
                    best_options = options
                    if len(best_options) == 1:
                        break
            if best_options is not None and len(best_options) == 1:
                break

        assert best_options is not None
        random.shuffle(best_options)
        best_options.sort(key=lambda item: -len(types[item[0]]["coords"]))

        for ti, pi in best_options:
            counts[ti] -= 1
            changed = place(ti, pi)
            if search():
                return True
            unplace(changed)
            counts[ti] += 1

        rollback_forced()
        fail_cache.add(key)
        return False

    return board.copy() if search() else None


def _solve_for_mask(types: list[dict], bg: int, shape_mask: np.ndarray, disallow_touch: set[int]) -> np.ndarray | None:
    solved = _assemble(types, shape_mask, disallow_touch)
    if solved is None:
        return None
    out = np.full(shape_mask.shape, bg, dtype=int)
    out[shape_mask] = solved[shape_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered small fragments into a cropped left-right symmetric non-rectangular cross.

    Input structure:
    - The background is the most frequent color.
    - Every non-background component is a small translated fragment; unlike the original task, no
      large grouped frame or border scaffold remains in place to reveal the target outline.

    Rule:
    - Collect all non-background fragments and compute their total occupied area.
    - Infer the thick cross mask whose area matches that total.
    - Pack the fragments into that mask so every occupied cell is filled exactly
      once, the final coloring is left-right symmetric, and different components of the same color
      never become 4-neighbor adjacent.

    Key insight:
    - The missing frame means the outline must be inferred indirectly: the output is no longer a
      rectangle at all, so the scattered pieces only lock together when arranged into the cropped
      thick cross while preserving left-right color symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    types, color_piece_count = _build_types(g, bg)
    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "grid must contain non-background fragments"

    candidates = _candidate_masks(total_area)
    assert candidates, "total area does not match the allowed non-rectangular cross silhouette"

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    solutions = []
    seen = set()
    for _kind, shape_mask in candidates:
        out = _solve_for_mask(types, bg, shape_mask, disallow_touch)
        if out is None:
            continue
        key = out.tobytes()
        if key not in seen:
            seen.add(key)
            solutions.append(out)

    assert solutions, "input must fit the allowed non-rectangular symmetric cross assembly"
    return solutions[0].astype(int)


def _partition_connected(cells: set[tuple[int, int]], max_area: int) -> list[list[tuple[int, int]]] | None:
    if not cells:
        return []
    cells = set(cells)
    for _ in range(_PARTITION_RETRIES):
        remaining = set(cells)
        chunks: list[list[tuple[int, int]]] = []
        ok = True
        while remaining:
            seed = random.choice(tuple(remaining))
            target = random.choice([2, 2, 3, 3, 4])
            chunk = [seed]
            remaining.remove(seed)
            frontier = {nb for nb in _neighbors4(*seed) if nb in remaining}
            while frontier and len(chunk) < target:
                frontier_list = sorted(frontier, key=lambda rc: (rc[1], rc[0]))
                nxt = random.choice(frontier_list[: min(3, len(frontier_list))])
                frontier.remove(nxt)
                if nxt not in remaining:
                    continue
                chunk.append(nxt)
                remaining.remove(nxt)
                for nb in _neighbors4(*nxt):
                    if nb in remaining:
                        frontier.add(nb)
            if len(chunk) > max_area:
                ok = False
                break
            chunks.append(sorted(chunk))

        if not ok:
            continue
        sizes = [len(chunk) for chunk in chunks]
        if max(sizes) > max_area:
            continue
        if min(sizes) < 1:
            continue
        if max(sizes) - min(sizes) > 2:
            continue
        return chunks
    return None


def _partition_center(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    mid = W // 2
    rows = [r for r in range(H) if mask[r, mid]]
    chunks = []
    i = 0
    while i < len(rows):
        remaining = len(rows) - i
        size_choices = [s for s in (1, 2, 3) if s <= remaining]
        if remaining == 4:
            size = random.choice([2, 2, 1])
        else:
            size = random.choice(size_choices)
        chunk_rows = rows[i : i + size]
        chunks.append([(r, mid) for r in chunk_rows])
        i += size
    return chunks


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


def _color_groups(group_adj: list[set[int]], n_colors: int) -> list[int] | None:
    order = sorted(range(len(group_adj)), key=lambda i: len(group_adj[i]), reverse=True)
    colors = [None] * len(group_adj)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            used = {int(c) for c in colors if c is not None}
            return len(used) == n_colors
        idx = order[pos]
        forbidden = {colors[j] for j in group_adj[idx] if colors[j] is not None}
        choices = [c for c in range(n_colors) if c not in forbidden]
        random.shuffle(choices)
        for c in choices:
            colors[idx] = c
            if dfs(pos + 1):
                return True
            colors[idx] = None
        return False

    return [int(c) for c in colors] if dfs(0) else None


def _build_output() -> tuple[np.ndarray | None, int, dict[str, object] | None]:
    kind, H, W = random.choice(_LAYOUTS)
    mask = _shape_mask(kind, H, W)
    mid = W // 2

    left_cells = {(int(r), int(c)) for r, c in np.argwhere(mask[:, :mid])}
    left_chunks = _partition_connected(left_cells, _MAX_PIECE_AREA)
    if left_chunks is None:
        return None, 0, None
    center_chunks = _partition_center(mask)

    piece_count = 2 * len(left_chunks) + len(center_chunks)
    if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
        return None, 0, None

    chunks: list[list[tuple[int, int]]] = []
    groups: list[list[int]] = []

    for left_chunk in left_chunks:
        left_id = len(chunks)
        chunks.append(sorted(left_chunk))
        right_chunk = sorted((r, W - 1 - c) for r, c in left_chunk)
        right_id = len(chunks)
        chunks.append(right_chunk)
        groups.append([left_id, right_id])

    for center_chunk in center_chunks:
        cid = len(chunks)
        chunks.append(sorted(center_chunk))
        groups.append([cid])

    sizes = [len(chunk) for chunk in chunks]
    if max(sizes) > _MAX_PIECE_AREA or max(sizes) - min(sizes) > 3:
        return None, 0, None

    chunk_adj = _chunk_adjacency(chunks)
    group_of = {}
    for gid, members in enumerate(groups):
        for midx in members:
            group_of[midx] = gid
    group_adj = [set() for _ in groups]
    for i, nbrs in enumerate(chunk_adj):
        gi = group_of[i]
        for j in nbrs:
            gj = group_of[j]
            if gi != gj:
                group_adj[gi].add(gj)

    max_colors = min(_N_COLORS_MAX, len(groups))
    if max_colors < _N_COLORS_MIN:
        return None, 0, None

    group_colors = None
    n_colors = None
    for _ in range(_COLOR_RETRIES):
        n_try = random.randint(_N_COLORS_MIN, max_colors)
        colored = _color_groups(group_adj, n_try)
        if colored is not None:
            group_colors = colored
            n_colors = n_try
            break
    if group_colors is None or n_colors is None:
        return None, 0, None

    bg = random.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    palette = available[:n_colors]

    out = np.full((H, W), bg, dtype=int)
    for gid, members in enumerate(groups):
        color = palette[group_colors[gid]]
        for midx in members:
            for r, c in chunks[midx]:
                out[r, c] = color

    meta = {
        "kind": kind,
        "dims": (H, W),
        "piece_count": piece_count,
        "max_piece_area": max(sizes),
        "input_has_frame": False,
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
    patches.sort(key=lambda p: (-(p != bg).sum(), p.shape[0] * p.shape[1], random.random()))
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


def _piece_count(out: np.ndarray, bg: int) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in sorted(set(map(int, np.unique(out))) - {bg}))


def _has_large_bbox_component(inp: np.ndarray, bg: int, dims: tuple[int, int]) -> bool:
    H, W = dims
    for col in sorted(set(map(int, np.unique(inp))) - {bg}):
        for pts in components_mask(inp == col, connectivity=8):
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1, c1 - c0 + 1) == (H, W):
                return True
    return False


def _gen_attempt(return_meta: bool = False):
    built = _build_output()
    out, bg, meta = built
    if out is None or meta is None:
        return (None, None) if return_meta else None

    visible_colors = sorted(set(map(int, np.unique(out))) - {bg})
    if not (_N_COLORS_MIN <= len(visible_colors) <= _N_COLORS_MAX):
        return (None, None) if return_meta else None
    if _piece_count(out, bg) != int(meta["piece_count"]):
        return (None, None) if return_meta else None

    try:
        inp = _scatter(out, bg)
        got = main(inp)
    except Exception:
        return (None, None) if return_meta else None

    H, W = meta["dims"]
    if _has_large_bbox_component(inp, bg, (H, W)):
        return (None, None) if return_meta else None
    if not np.array_equal(got, out):
        return (None, None) if return_meta else None
    return (inp, meta) if return_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
