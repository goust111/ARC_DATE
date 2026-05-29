# Combo variant of e8686506 (index 43)
# combo_spec: D4/4+D6/0
# Name: 无对称约束+颜色统一化

from __future__ import annotations

from collections import Counter
import random
import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
except ImportError:
    from solvers.arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# a) Concrete behavior trace:
#    Original: scattered colored pieces -> assemble into a dense LR-symmetric
#    colored rectangle. Symmetry and preserved colors strongly constrain
#    placement. This combo removes both output clues. We still hide one frame
#    piece on the canvas at its true rectangle-relative position, but the final
#    output is not the filled colored rectangle. Instead:
#      1. rebuild the hidden colored rectangle from the pieces,
#      2. discard all interior colors,
#      3. keep only the outer rectangular outline, recolored to the frame color.
#    Small example: if the hidden assembly is a 6x5 asymmetric colored board,
#    the visible output becomes a 6x5 hollow rectangle in one color.
#
# b) Complexity calibration:
#    If the task only asked for a filled monochrome rectangle, total area alone
#    would trivialize it. To keep a visible combo effect without exploding the
#    search space, the generator uses:
#      - fixed narrow width 5 and heights 5..8,
#      - an anchored asymmetric frame spanning the full output bbox,
#      - remaining pieces split into small 1..4 cell chunks,
#      - adjacent chunks forced to have different colors.
#    This preserves a genuine packing problem while keeping solver search small.
#
# c) Assumption audit against the original solver:
#      - `mirror_ok(...)` and the final LR-symmetry assert are invalid now and
#        are removed entirely.
#      - output color requirements inferred from corner bars/lines are also
#        invalid because the final output is unified to one color.
#      - the dimension rule is still safe: total area equals the hidden dense
#        rectangle area, and the generator keeps a unique closest-to-square
#        portrait factor pair.
#      - the new critical assumption is: exactly one anchored frame piece has a
#        bbox equal to the output rectangle dimensions. The generator enforces
#        this explicitly.
#
# d) Placement legality:
#    The generator blocks the frame bbox before scattering other pieces, then
#    uses `random_free_location(..., padding=1)` so pieces never overlap, never
#    leave the canvas, and remain visually separate.
#
# e/f/g) Visual observability and neatness:
#    Across examples, the hidden colored rectangle is often visibly asymmetric,
#    but the output is always a tight monochrome outline in the same color as
#    the anchored frame. This makes the fusion readable to a human: geometry is
#    still needed to infer the rectangle, while color survives only as the
#    chosen outline color cue. Limiting the palette and chunk size prevents the
#    scattered inputs from becoming noisy or visually degenerate.


# --------------- shared solver helpers ---------------

_AREA_MIN = 25
_AREA_MAX = 40
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 8

_MAX_FILL_PIECE_AREA = 4
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 400


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _closest_dims(total_area: int) -> tuple[int, int]:
    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait output shape must be unique"
    return best[0]


def _outline_only(H: int, W: int, bg: int, color: int) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    out[0, :] = int(color)
    out[H - 1, :] = int(color)
    out[:, 0] = int(color)
    out[:, W - 1] = int(color)
    return out


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            pieces.append(
                {
                    "col": int(col),
                    "pts": np.array(pts, dtype=int),
                    "coords": coords,
                    "h": int(h),
                    "w": int(w),
                    "area": len(coords),
                }
            )
    return pieces


def _find_frame_piece(pieces: list[dict], H: int, W: int) -> dict:
    candidates = [p for p in pieces if int(p["h"]) == int(H) and int(p["w"]) == int(W)]
    assert len(candidates) == 1, "there must be exactly one anchored frame piece with full output bbox"
    return candidates[0]


def _group_types(pieces: list[dict], frame_piece: dict) -> tuple[list[dict], set[int]]:
    others = [p for p in pieces if p is not frame_piece]
    ctr = Counter((int(p["col"]), p["coords"]) for p in others)
    color_piece_count = Counter(int(p["col"]) for p in others + [frame_piece])
    types: list[dict] = []
    for (col, coords), cnt in sorted(
        ctr.items(),
        key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1]),
    ):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    return types, disallow_touch


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    frame_piece: dict,
    disallow_touch: set[int],
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)

    for dr, dc in frame_piece["coords"]:
        board[int(dr), int(dc)] = int(frame_piece["col"])

    counts = [int(t["cnt"]) for t in types]
    memo: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W):
                return False
            if board[r, c] != -1:
                return False
        if col in disallow_touch:
            own = {(r0 + dr, c0 + dc) for dr, dc in t["coords"]}
            for r, c in own:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (rr, cc) in own:
                        continue
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        changed = []
        col = int(types[ti]["col"])
        for dr, dc in types[ti]["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def next_empty() -> tuple[int, int] | None:
        for r in range(H):
            for c in range(W):
                if board[r, c] == -1:
                    return r, c
        return None

    def dfs() -> bool:
        pos = next_empty()
        if pos is None:
            return all(c == 0 for c in counts)

        state = (tuple(map(int, board.ravel())), tuple(map(int, counts)))
        if state in memo:
            return False

        r, c = pos
        candidates: list[tuple[int, int, int, int]] = []
        for ti, t in enumerate(types):
            if counts[ti] <= 0:
                continue
            area = len(t["coords"])
            seen = set()
            for dr, dc in t["coords"]:
                r0, c0 = r - dr, c - dc
                key = (r0, c0)
                if key in seen:
                    continue
                seen.add(key)
                if can_place(ti, r0, c0):
                    candidates.append((-area, ti, r0, c0))

        candidates.sort()
        for _neg_area, ti, r0, c0 in candidates:
            counts[ti] -= 1
            changed = place(ti, r0, c0)
            if dfs():
                return True
            unplace(changed)
            counts[ti] += 1

        memo.add(state)
        return False

    assert dfs(), "pieces do not assemble into a full rectangle under the variant rules"
    return board


def _solve_colored_rectangle(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pieces = _extract_pieces(g, bg)
    total_area = sum(int(p["area"]) for p in pieces)
    assert total_area > 0, "grid must contain non-background pieces"
    H, W = _closest_dims(total_area)
    frame_piece = _find_frame_piece(pieces, H, W)
    types, disallow_touch = _group_types(pieces, frame_piece)
    board = _assemble(types, H, W, frame_piece, disallow_touch)
    return board, H, W, int(bg), int(frame_piece["col"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a scattered jigsaw into a rectangle, then discard its interior coloring.

    Input structure:
    - The background is the most common color.
    - One anchored frame fragment already sits at the true output-relative position and spans
      the full output bbox.
    - The remaining colored components are scattered away from that bbox.

    Rule:
    - Reassemble all non-background pieces into the unique closest-to-square portrait rectangle
      whose area matches the total foreground area.
    - The hidden rectangle itself no longer has to be left-right symmetric.
    - After assembly, ignore the interior piece colors and output only the tight rectangular
      outline, using the frame's color on that outline.

    Key insight: The frame still reveals the target bbox, but symmetry and final color layout are
    gone; the meaningful structure is the geometry of filling that rectangle, while the visible
    answer keeps only the final outer contour.
    """
    _board, H, W, bg, frame_color = _solve_colored_rectangle(grid)
    return _outline_only(H, W, bg, frame_color)


# --------------- generator ---------------

_VALID_DIMS = [(h, w) for h in range(5, _HEIGHT_MAX + 1) for w in [_WIDTH_REQUIRED] if _closest_dims(h * w) == (h, w)]
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _build_frame_mask(H: int, W: int) -> np.ndarray | None:
    for _ in range(120):
        cols = [random.randint(0, W - 1)]
        for _r in range(1, H):
            step = random.choice([-1, 0, 1])
            cols.append(max(0, min(W - 1, cols[-1] + step)))
        if min(cols) != 0 or max(cols) != W - 1:
            continue

        mask = np.zeros((H, W), dtype=bool)
        for r in range(H - 1):
            c0, c1 = cols[r], cols[r + 1]
            for c in range(min(c0, c1), max(c0, c1) + 1):
                mask[r, c] = True
            mask[r + 1, c1] = True
        mask[H - 1, cols[-1]] = True

        if mask.sum() < H + 2 or mask.sum() > H + 8:
            continue
        if mask[0].sum() == W or mask[-1].sum() == W:
            continue
        return mask
    return None


def _grow_chunk(open_mask: np.ndarray, target: int) -> set[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(open_mask)]
    seed = random.choice(coords)
    chunk = {seed}
    frontier = [seed]
    while len(chunk) < target and frontier:
        r, c = random.choice(frontier)
        nbrs = []
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < open_mask.shape[0] and 0 <= cc < open_mask.shape[1] and open_mask[rr, cc] and (rr, cc) not in chunk:
                nbrs.append((rr, cc))
        if not nbrs:
            frontier.remove((r, c))
            continue
        nxt = random.choice(nbrs)
        chunk.add(nxt)
        frontier.append(nxt)
    return chunk


def _split_open_area(frame_mask: np.ndarray) -> list[set[tuple[int, int]]] | None:
    open_mask = ~frame_mask.copy()
    pieces: list[set[tuple[int, int]]] = []

    while np.any(open_mask):
        comps = components_mask(open_mask, connectivity=4)
        if not comps:
            break
        comp = max(comps, key=len)
        comp_set = {(int(r), int(c)) for r, c in comp}
        if len(comp_set) <= _MAX_FILL_PIECE_AREA:
            pieces.append(comp_set)
            for r, c in comp_set:
                open_mask[r, c] = False
            continue

        max_target = min(_MAX_FILL_PIECE_AREA, len(comp_set) - 1)
        if max_target < 1:
            return None
        target = random.randint(1, max_target)
        submask = np.zeros_like(open_mask, dtype=bool)
        for r, c in comp_set:
            submask[r, c] = True
        chunk = _grow_chunk(submask, target)
        pieces.append(chunk)
        for r, c in chunk:
            open_mask[r, c] = False

    return pieces


def _adjacency_graph(chunks: list[set[tuple[int, int]]]) -> list[set[int]]:
    owner: dict[tuple[int, int], int] = {}
    for idx, chunk in enumerate(chunks):
        for cell in chunk:
            owner[cell] = idx
    graph = [set() for _ in chunks]
    for idx, chunk in enumerate(chunks):
        for r, c in chunk:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                j = owner.get((rr, cc))
                if j is not None and j != idx:
                    graph[idx].add(j)
    return graph


def _color_chunks(chunks: list[set[tuple[int, int]]], bg: int, frame_color: int) -> dict[int, int] | None:
    graph = _adjacency_graph(chunks)
    pool = [c for c in range(10) if c not in {bg, frame_color}]
    random.shuffle(pool)
    palette = pool[:5]
    if len(palette) < 3:
        return None

    order = sorted(range(len(chunks)), key=lambda i: (-len(graph[i]), -len(chunks[i]), i))
    assign: dict[int, int] = {}
    for idx in order:
        used = {assign[j] for j in graph[idx] if j in assign}
        options = [c for c in palette if c not in used]
        if not options:
            return None
        assign[idx] = random.choice(options)
    return assign


def _build_hidden_output(H: int, W: int) -> tuple[np.ndarray | None, int, int]:
    bg = random.randint(0, 9)
    frame_color = random.choice([c for c in range(10) if c != bg])
    frame_mask = _build_frame_mask(H, W)
    if frame_mask is None:
        return None, 0, 0

    chunks = _split_open_area(frame_mask)
    if not chunks:
        return None, 0, 0
    piece_count = 1 + len(chunks)
    if not (_PIECE_COUNT_MIN <= piece_count <= _PIECE_COUNT_MAX):
        return None, 0, 0

    colors = _color_chunks(chunks, bg, frame_color)
    if colors is None:
        return None, 0, 0

    out = np.full((H, W), bg, dtype=int)
    out[frame_mask] = frame_color
    for idx, chunk in enumerate(chunks):
        col = int(colors[idx])
        for r, c in chunk:
            out[r, c] = col

    return out, bg, frame_color


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_color: int) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for col in sorted(set(map(int, np.unique(out))) - {int(bg), int(frame_color)}):
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, frame_color: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(out, bg, frame_color):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    frame_mask = out == frame_color
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _expected_output(inp: np.ndarray) -> np.ndarray:
    board, H, W, bg, frame_color = _solve_colored_rectangle(inp)
    assert board.shape == (H, W)
    return _outline_only(H, W, bg, frame_color)


def _gen_attempt() -> np.ndarray | None:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    hidden, bg, frame_color = _build_hidden_output(H, W)
    if hidden is None:
        return None
    if np.array_equal(hidden, hidden[:, ::-1]):
        return None
    try:
        inp = _scatter(hidden, bg, frame_color)
        got = main(inp)
        want = _outline_only(H, W, bg, frame_color)
    except Exception:
        return None
    if np.array_equal(got, want):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo e8686506 input")
