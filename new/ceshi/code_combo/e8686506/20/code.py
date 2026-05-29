# Combo variant of e8686506 (index 20)
# combo_spec: D6/3+D14/0
# Name: 按位置重新着色+颜色编码位置

import random
from collections import Counter, deque

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ---------------
# concreteization_and_trace:
#   - The hidden task skeleton stays the same: the input still contains a preserved LR-symmetric
#     frame plus scattered interior fragments, and the solver still has to recover the dense Hx5
#     rectangle before anything else can happen.
#   - D14/0 is realized by giving source fragments semantic palettes instead of arbitrary labels:
#     warm colors {2,4,7} may only belong to the upper half of the hidden rectangle, while cool/
#     muted colors {1,5,8} may only belong to the lower half. A concrete trace is:
#       original: an orange L and a blue L of the same shape could swap vertically if geometry allowed;
#       fused: the orange L can only land above the midline, and the blue L can only land below it.
#   - D6/3 is then applied after assembly, not during it. Once the hidden source-colored rectangle is
#     recovered, each finished 4-connected piece is recolored by its horizontal zone:
#       outer edge columns {0,4} -> 9, inner flank columns {1,3} -> 6, center column {2} -> 3.
#     Example: two warm source pieces in different top rows but both hugging the outside border both
#     become color 9 in the answer; a cool source piece centered in column 2 becomes color 3.
#   - Difference from the base task: input colors no longer survive to the answer, and they are no
#     longer mere visual IDs either. They first constrain placement, then get discarded and replaced
#     by column-driven output colors.
# complexity_calibration:
#   - The semantic input rule uses only two repeated families (top vs bottom), which is strong enough
#     to matter for placement but still easy to infer from a few examples.
#   - The final recolor uses exactly three repeated target colors, one per folded column zone. This
#     makes the change visually obvious without exploding the number of cases a human must learn.
#   - Source palettes and target palettes are disjoint: input semantics use {1,2,4,5,7,8}, whereas
#     output recoloring uses {3,6,9}. That avoids accidental "identity" samples where the fusion
#     would look like the original task.
# implicit_assumption_audit:
#   - `_build_requirements()` in the base solver assumed corner/bar anchors could be recognized by one
#     exact color family. That remains valid only after being rewritten in semantic form:
#       warm singleton pairs anchor the top corners, cool singleton pairs anchor the bottom corners,
#       and warm/cool full bars anchor top/bottom edges.
#   - The original `_assemble.can_place()` only checked geometric fit, mirror fit, and exact source
#     color requirements. The fused solver adds `semantic_ok()` so a piece's source color family must
#     agree with the row band of the placement centroid.
#   - Frame detection cannot rely on a "frame color set" in the generator, because interior pieces may
#     reuse the same semantic colors as the frame. The generator therefore tracks the geometric frame
#     mask explicitly and scatters only non-frame components outside the blocked bbox.
#   - Grouping by `(source color, shape)` still works: assembly is performed on the hidden source board,
#     so source colors remain meaningful until the final recolor step.
# placement_legality:
#   - Hidden outputs are built first, then scattered. Interior chunks are grown on the canonical
#     left-half-plus-center representation and mirrored, so every produced piece stays in-bounds and
#     the visible rectangle remains perfectly LR-symmetric.
#   - Chunk growth is capped by mirrored area, non-frame patches are placed with the shared
#     `random_free_location(..., padding=1)` helper, and the preserved frame bbox is blocked before
#     scattering, preventing overlap with the outline group.
# visual_observability_and_tidiness:
#   - The input shows clean warm-vs-cool structure without overlays: top-oriented pieces visibly use
#     warm colors and bottom-oriented pieces visibly use cool/muted colors.
#   - The output shows a separate clean rule: pieces are repainted into three column zones. Because the
#     target palette is disjoint from the input palette, the role flip from "clue color" to "result
#     color" is obvious across samples rather than hidden in cosmetic coincidences.
# color_inferability:
#   - No invisible ordering by numeric color id is used in either stage. The only source-color rule is
#     the human-visible warm/cool family split, and the only target-color rule is the human-visible
#     horizontal zone of the assembled piece.
# main_assumptions_check:
#   - `semantic_ok()` assumes every non-background input color belongs to exactly one semantic family;
#     the generator guarantees this by drawing hidden colors only from `_TOP_COLORS` and
#     `_BOTTOM_COLORS`.
#   - `_recolor_by_position()` assumes width 5 so the folded column zones are exactly outer / inner /
#     center; the generator keeps `W == 5`, matching the base task's viable dimensions.
#   - The propagation solver still assumes the instance is pure-propagation solvable. The generator
#     keeps the original output-first oracle check `main(inp) == visible_output` so only compliant
#     samples are emitted.


_TOP_COLORS = (2, 4, 7)
_BOTTOM_COLORS = (1, 5, 8)
_ZONE_COLORS = (9, 6, 3)  # outer edge, inner flank, center
_BG_COLOR = 0


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
    col = int(col)
    if col in _TOP_COLORS:
        return "top"
    if col in _BOTTOM_COLORS:
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
    Build per-cell exact SOURCE-color constraints (req[r,c] == -1 means unconstrained).

    Under the fused rule, source colors encode vertical position bands, so anchor heuristics
    must be interpreted semantically rather than as one arbitrary exact color shared by all edges.
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
                        top_bars.append((int(r0), int(col)))
                    elif band == "bottom":
                        bot_bars.append((int(r0), int(col)))
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
            return False
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
    assert np.array_equal(board, board[:, ::-1]), "assembled hidden board must be LR-symmetric"
    return board


def _piece_zone(pts: np.ndarray, W: int) -> int:
    folded = [min(int(c), W - 1 - int(c)) for _, c in pts]
    counts = np.bincount(folded, minlength=(W // 2) + 1)
    winners = np.flatnonzero(counts == counts.max())
    return int(winners[0])


def _recolor_by_position(hidden: np.ndarray) -> np.ndarray:
    _, W = hidden.shape
    out = np.zeros_like(hidden, dtype=int)
    for col in sorted(map(int, np.unique(hidden))):
        for pts in components_mask(hidden == col, connectivity=4):
            zone = _piece_zone(np.asarray(pts, dtype=int), W)
            target = int(_ZONE_COLORS[zone])
            for r, c in pts:
                out[int(r), int(c)] = target
    assert np.array_equal(out, out[:, ::-1]), "recolored output must stay LR-symmetric"
    return out


def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((h, w), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the hidden rectangle using source-color position clues, then recolor the solved pieces by column zone.

    Input structure:
    - The most common color is background; each remaining component is a scattered fragment from a hidden
      left-right symmetric Hx5 rectangle.
    - Source colors are semantic: warm colors indicate pieces from the upper half, cool/muted colors indicate
      pieces from the lower half. A preserved LR-symmetric frame still appears as 1-2 aligned outline groups.

    Rule:
    - Extract all non-background 4-connected fragments, infer the output dimensions from total area, and recover
      the hidden source-colored rectangle under the original mirror/frame constraints plus the new top/bottom
      color-band placement rule.
    - After assembly, treat each recovered 4-connected source-color component as one finished piece and repaint it
      uniformly by the folded column zone it mostly occupies: outer edge, inner flank, or center.

    Key insight:
    - Input color and output color play different roles. The input palette is only a placement clue for where a
      piece belongs vertically, while the answer palette is determined only after assembly by the piece's final
      horizontal position inside the solved rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(map(int, np.unique(g))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        assert _color_band(col) is not None, "all non-background input colors must encode a top/bottom band"
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
    assert W == 5, "fused recolor rule assumes width-5 output"

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

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
    hidden = _assemble(types, H, W, req, disallow_touch)
    return _recolor_by_position(hidden).astype(int)


# --------------- generator (semantic hidden colors + positional recolor) ---------------

_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

_MAX_CHUNK_AREA = 6
_MAX_OUTPUT_COMPONENT_AREA = 15
_PIECE_COUNT_MIN = 8
_PIECE_COUNT_MAX = 17
_VISIBLE_COMPONENT_MIN = 6

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 1000


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
            nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
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


def _build_hidden_output(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
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
            frame_mask[int(rr), int(cc)] = True

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

    assert not np.any(grid == -1), "hidden output build must fill every cell"
    return grid, frame_mask


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_mask: np.ndarray) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            cells = [(int(r), int(c)) for r, c in pts]
            if all(bool(frame_mask[r, c]) for r, c in cells):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in cells:
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


def _visible_component_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _zone_piece_hist(hidden: np.ndarray) -> list[int]:
    hist = [0] * len(_ZONE_COLORS)
    _, W = hidden.shape
    for col in sorted(map(int, np.unique(hidden))):
        for pts in components_mask(hidden == col, connectivity=4):
            hist[_piece_zone(np.asarray(pts, dtype=int), W)] += 1
    return hist


def _semantic_score(hidden: np.ndarray) -> tuple[int, int]:
    good = 0
    total = 0
    H, _ = hidden.shape
    for col in np.unique(hidden):
        band = _color_band(int(col))
        if band is None:
            continue
        for pts in components_mask(hidden == col, connectivity=4):
            coords = [(int(r), int(c)) for r, c in pts]
            mean_r = float(sum(r for r, _ in coords)) / float(len(coords))
            total += 1
            if (band == "top" and mean_r < (H / 2.0)) or (band == "bottom" and mean_r >= (H / 2.0)):
                good += 1
    return good, total


def _singleton_border_ok(hidden: np.ndarray) -> bool:
    H, W = hidden.shape
    for col in np.unique(hidden):
        for pts in components_mask(hidden == col, connectivity=4):
            if len(pts) != 1:
                continue
            r, c = map(int, pts[0])
            if not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    hidden, frame_mask = _build_hidden_output(H, W)

    if not np.array_equal(hidden, hidden[:, ::-1]):
        return None
    n_colors = len(np.unique(hidden))
    if not (4 <= n_colors <= 6):
        return None
    pc = _piece_count(hidden)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if not _singleton_border_ok(hidden):
        return None
    if any(
        len(pts) > _MAX_OUTPUT_COMPONENT_AREA
        for col in np.unique(hidden)
        for pts in components_mask(hidden == col, connectivity=4)
    ):
        return None
    good, total = _semantic_score(hidden)
    if total == 0 or good != total:
        return None

    visible = _recolor_by_position(hidden)
    zone_hist = _zone_piece_hist(hidden)
    if any(v == 0 for v in zone_hist):
        return None
    if _visible_component_count(visible) < _VISIBLE_COMPONENT_MIN:
        return None

    try:
        inp = _scatter(hidden, _BG_COLOR, frame_mask)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, visible):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
