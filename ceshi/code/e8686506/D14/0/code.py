# Variant of e8686506
# Dimension: D14 — 碎片颜色的语义作用 (meta)
# Change: 每个碎片为独特的单色，颜色仅用于视觉区分碎片，不参与匹配逻辑... → 颜色编码位置信息

import random
from collections import Counter, deque

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D14 variant
# ==================================
# concreteization_and_trace:
#   - The rectangle-assembly task stays intact: input still shows a preserved LR-symmetric frame
#     plus scattered interior fragments, and output is still a dense LR-symmetric rectangle.
#   - The new visible rule is that piece colors are no longer arbitrary labels. Warm ARC colors
#     (red/yellow/orange) mean the piece belongs in the upper half of the final rectangle, while
#     cool or muted ARC colors (blue/grey/teal) mean the piece belongs in the lower half.
#   - Small trace: in the original task, a 3-cell orange fragment and an identical 3-cell blue
#     fragment could swap vertically if frame constraints allowed it. In the variant, the orange
#     one can only be placed where its centroid stays above the horizontal midline, and the blue
#     one can only be placed below it. This is the core solver change.
# complexity_calibration:
#   - The mapping is intentionally coarse (two bands) rather than per-row. That keeps the rule
#     human-inferable from 2-3 examples while still materially shrinking the placement search.
#   - Only five visible semantic colors are used, and each appears on multiple pieces across the
#     dataset, so the color-position relation is repeated rather than being a one-off coincidence.
# implicit_assumption_audit:
#   - Original `_build_requirements()` assumed four corner singletons may share one exact color.
#     That is no longer reliable because top and bottom corners may belong to different semantic
#     bands. The revised anchor logic therefore only uses upper/lower-compatible singleton pairs
#     and top/bottom bars.
#   - Original `_assemble.can_place()` treated color as exact paint only. The variant adds a
#     centroid-band legality check based on the visible upper/lower color families.
#   - Original `_scatter()` identified frame pieces by color set. That breaks once interior pieces
#     may reuse frame colors, so the generator now tracks the geometric frame mask explicitly.
# placement_legality:
#   - The frame is always generated as exactly two 8-connected parts, warm on top rows and
#     cool/muted on bottom rows, so the solver's 1-2-part frame detection still holds.
#   - Interior chunks are built on the canonical left half plus center column and mirrored; chunk
#     growth is capped by final mirrored area so no produced piece exceeds the allowed size.
# visual_observability_and_tidiness:
#   - The warm-vs-cool/muted split is visible in both input fragments and final assembly without
#     adding extra overlays or textual markers. The old jigsaw outline remains visually clean.
# color_inferability:
#   - No hidden ordering by numeric color id is used in either solver or generator. The only
#     semantic grouping is the visible warm-vs-cool/muted appearance of the ARC palette.

_TOP_COLORS = (2, 4, 7)
_BOTTOM_COLORS = (1, 5, 8)
_BG_CHOICES = (0, 3, 6, 9)


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


def _color_band(col: int) -> str | None:
    if int(col) in _TOP_COLORS:
        return "top"
    if int(col) in _BOTTOM_COLORS:
        return "bottom"
    return None


def _placement_band(coords: tuple[tuple[int, int], ...], r0: int, H: int) -> str:
    mean_r = float(sum(r0 + dr for dr, _ in coords)) / float(len(coords))
    return "top" if mean_r < (H / 2.0) else "bottom"


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell exact-color constraints (req[r,c] == -1 means unconstrained).

    Semantic anchors retained in the variant:
      - An upper-band color with exactly two singleton components marks the top corners.
      - A lower-band color with exactly two singleton components marks the bottom corners.
      - Upper-band full bars of length W-2 mark the top edge; lower-band bars mark the bottom edge.
    """
    req = np.full((H, W), -1, dtype=int)

    warm_singletons = [
        col
        for col, comps in comps_by_color.items()
        if _color_band(col) == "top" and len(comps) == 2 and all(len(p) == 1 for p in comps)
    ]
    cool_singletons = [
        col
        for col, comps in comps_by_color.items()
        if _color_band(col) == "bottom" and len(comps) == 2 and all(len(p) == 1 for p in comps)
    ]
    if len(warm_singletons) == 1:
        col = int(warm_singletons[0])
        req[0, 0] = col
        req[0, W - 1] = col
    if len(cool_singletons) == 1:
        col = int(cool_singletons[0])
        req[H - 1, 0] = col
        req[H - 1, W - 1] = col

    top_bars: list[tuple[int, int]] = []
    bot_bars: list[tuple[int, int]] = []
    if W >= 3:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    band = _color_band(col)
                    if band == "top":
                        top_bars.append((r0, int(col)))
                    elif band == "bottom":
                        bot_bars.append((r0, int(col)))
                    break
        if top_bars:
            top_bars.sort()
            req[0, 1 : W - 1] = top_bars[0][1]
        if bot_bars:
            bot_bars.sort()
            req[H - 1, 1 : W - 1] = bot_bars[-1][1]

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

    def semantic_ok(ti: int, r0: int) -> bool:
        t = types[ti]
        band = _color_band(int(t["col"]))
        if band is None:
            return True
        return _placement_band(t["coords"], r0, H) == band

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        is_singleton = len(t["coords"]) == 1
        if not semantic_ok(ti, r0):
            return False
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a dense left-right symmetric rectangle, with color semantics.

    Input structure:
    - The background is the most common color; every other connected component is a puzzle fragment.
    - A preserved LR-symmetric frame still appears as 1-2 aligned outline pieces, but fragment colors now
      belong to visible upper/lower color families that hint whether the piece belongs in the upper or lower half.

    Rule:
    - Extract all non-background 4-connected fragments and infer the output rectangle size from total area.
    - Preserve the original frame-placement logic and exact LR mirror symmetry.
    - When placing a fragment, also respect the color-position rule: warm-colored pieces assemble above the
      horizontal midpoint, while cool or muted pieces assemble below it; same-color copies still cannot 4-touch.

    Key insight:
    - The outline frame and mirror symmetry still determine the rectangle, but the variant becomes uniquely
      solvable because color now contributes positional information: upper-band fragments fit only in upper slots
      and lower-band fragments fit only in lower slots.
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
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

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
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = int(col)

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (o5x) ---------------

_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

_MAX_CHUNK_AREA = 6
_MAX_OUTPUT_COMPONENT_AREA = 15
_PIECE_COUNT_MIN = 7
_PIECE_COUNT_MAX = 17
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 300


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


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells: set[tuple[int, int]] = set()
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


def _mirrored_cells(cells: list[tuple[int, int]], W: int) -> list[tuple[int, int]]:
    out = set()
    for r, c in cells:
        out.add((int(r), int(c)))
        out.add((int(r), W - 1 - int(c)))
    return sorted(out)


def _mirrored_area(cells: list[tuple[int, int]], W: int) -> int:
    return len(_mirrored_cells(cells, W))


def _split_connected_canonical(pts: np.ndarray, W: int, max_area: int) -> list[list[tuple[int, int]]]:
    remaining = {tuple(map(int, rc)) for rc in pts}
    chunks: list[list[tuple[int, int]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        chunk = [seed]
        frontier = deque([seed])
        while frontier:
            r, c = frontier.popleft()
            nbrs = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ]
            for nb in sorted(nbrs):
                if nb not in remaining:
                    continue
                trial = chunk + [nb]
                if _mirrored_area(trial, W) > max_area:
                    continue
                remaining.remove(nb)
                chunk.append(nb)
                frontier.append(nb)
        chunks.append(sorted(chunk))
    return chunks


def _neighbor_colors(grid: np.ndarray, cells: list[tuple[int, int]]) -> set[int]:
    H, W = grid.shape
    out: set[int] = set()
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W:
                val = int(grid[rr, cc])
                if val != -1:
                    out.add(val)
    return out


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _pick_semantic_color(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    palette: tuple[int, ...],
    usage: Counter,
) -> int:
    avoid = _neighbor_colors(grid, cells)
    options = [c for c in palette if c not in avoid]
    if not options:
        options = list(palette)
    least_used = min(int(usage[c]) for c in options)
    tied = [c for c in options if int(usage[c]) == least_used]
    return int(random.choice(tied))


def _build_output(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    half = W // 2
    split = (H + 1) // 2
    top_frame_color = random.choice(_TOP_COLORS)
    bot_frame_color = random.choice(_BOTTOM_COLORS)
    top_fill_colors = tuple(c for c in _TOP_COLORS if c != top_frame_color)
    bot_fill_colors = tuple(c for c in _BOTTOM_COLORS if c != bot_frame_color)

    while True:
        frame_half, start_c, end_c = _zigzag_frame(H)
        if start_c == 1 and end_c == 1:
            break
    grid = np.full((H, W), -1, dtype=int)
    frame_mask = np.zeros((H, W), dtype=bool)

    for r, c in frame_half:
        color = top_frame_color if r < split else bot_frame_color
        cells = [(r, c), (r, W - 1 - c)]
        _paint_cells(grid, cells, color)
        for rr, cc in cells:
            frame_mask[rr, cc] = True

    grid[0, half] = top_frame_color
    frame_mask[0, half] = True
    grid[H - 1, half] = bot_frame_color
    frame_mask[H - 1, half] = True

    usage: Counter = Counter({int(top_frame_color): 1, int(bot_frame_color): 1})
    left_holes = components_mask(grid[:, : half + 1] == -1, connectivity=4)
    for hole in left_holes:
        for chunk in _split_connected_canonical(hole, W, _MAX_CHUNK_AREA):
            full_cells = _mirrored_cells(chunk, W)
            mean_r = float(sum(r for r, _ in full_cells)) / float(len(full_cells))
            palette = top_fill_colors if mean_r < (H / 2.0) else bot_fill_colors
            color = _pick_semantic_color(grid, full_cells, palette, usage)
            _paint_cells(grid, full_cells, color)
            usage[color] += 1

    assert not np.any(grid == -1), "output build must fill every cell"
    return grid, frame_mask


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_mask: np.ndarray) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            cell_list = [(int(r), int(c)) for r, c in pts]
            if all(bool(frame_mask[r, c]) for r, c in cell_list):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in cell_list:
                patch[r - r0, c - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, frame_mask: np.ndarray) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame bbox does not fit on the scatter canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    for patch in _extract_non_frame_patches(out, bg, frame_mask):
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
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _semantic_score(out: np.ndarray) -> tuple[int, int]:
    good = 0
    total = 0
    H, _W = out.shape
    for col in np.unique(out):
        for pts in components_mask(out == col, connectivity=4):
            coords = [(int(r), int(c)) for r, c in pts]
            band = _color_band(int(col))
            if band is None:
                continue
            mean_r = float(sum(r for r, _ in coords)) / float(len(coords))
            total += 1
            if (band == "top" and mean_r < (H / 2.0)) or (band == "bottom" and mean_r >= (H / 2.0)):
                good += 1
    return good, total


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, frame_mask = _build_output(H, W)
    bg = random.choice(_BG_CHOICES)

    if not np.array_equal(out, out[:, ::-1]):
        return None
    n_colors = len(np.unique(out))
    if not (_N_COLORS_MIN <= n_colors <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_OUTPUT_COMPONENT_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None
    good, total = _semantic_score(out)
    if total == 0 or good != total:
        return None

    try:
        inp = _scatter(out, bg, frame_mask)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _out = result
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D14 input")
