# Combo variant of 16b78196 (index 8)
# combo_spec: D4/2+D5/2+D15/0
# Name: 颜色排序链+渐变着色+图例
#
# Design analysis
# ===============
# This fusion keeps the original bar/notch/cavity geometry, but replaces the chain-selection logic with
# a three-layer color pipeline:
# 1. border legend colors are symbolic instruction tokens;
# 2. scattered piece colors are sorted by visible brightness to decide chain order;
# 3. after placement, piece colors are shifted forward along a fixed brighter gradient.
#
# a) Concrete behavior and trace
# - Original task:
#   1. find bar notches;
#   2. choose any root piece that geometrically plugs a notch;
#   3. extend the chain with any remaining piece that covers the previous cavity;
#   4. write each placed piece using its original color.
# - Fusion task:
#   1. find a short border legend run on each chain side;
#   2. read its direction: dark->light means "this chain uses source pieces from darker to brighter",
#      light->dark means the reverse;
#   3. place the root and later pieces using the same cavity logic as the base task, but only accept
#      source colors whose brightness order follows that legend direction;
#   4. keep the root color unchanged, recolor depth-1 pieces one step brighter on a fixed movable palette,
#      and recolor depth-2 pieces two steps brighter.
# - Example:
#   a top-side legend [maroon, yellow] means a 2-piece chain with source order dark->light. If the two
#   matching scattered pieces are blue and teal, blue must be the root and teal the continuation. In the
#   output the blue root stays blue, while the teal continuation is recolored one step brighter on the
#   movable palette.
#
# b) Complexity calibration
# - The legend uses only three fixed singleton marker colors: maroon / grey / yellow. Their luminance is
#   visibly ordered, so humans can read the direction without hidden numeric ids.
# - Movable pieces use a separate six-color palette [pink, blue, red, orange, green, teal] ordered by
#   actual ARC display brightness. That keeps the source-color space small enough for repeated evidence.
# - Every generated puzzle has exactly two chains, typically 4-5 total pieces, so both the legend order
#   rule and the recoloring rule recur inside each sample.
#
# c) Hidden-assumption audit against the original main()
# - `pieces_raw = comps[1:]` is invalid now because border singleton legends are no longer pieces.
#   The solver explicitly classifies border area-1 components as legends and excludes them from the
#   movable-piece list.
# - "any remaining piece may extend a chain" is invalid. Extensions now require a brightness monotonicity
#   condition derived from that chain's legend direction.
# - "placed color == source color" is invalid. The output writer transforms colors by chain depth.
# - Undo logic must preserve both bar cells and legend cells, because both are immutable scaffold layers.
#
# d) Placement legality
# - The generator still works output-first. It builds two finished chains, adds legends on the matching
#   border sides, then scatters the source pieces away from the bar and legends.
# - Legends sit on the outer border only; scattered pieces are kept off the border and away from a small
#   legend buffer so the symbolic layer stays visually separate.
# - The output uses only transformed movable colors, while legend markers keep their own fixed marker
#   colors, so the three layers never merge into a single ambiguous object class.
#
# e) Visual observability
# - The ordering cue is human-visible: legend runs are explicitly dark->light or light->dark, and the
#   matching assembled source colors follow the same direction.
# - The transform cue is also visible: non-root pieces become brighter than their input source colors by
#   a consistent step count equal to chain depth.
# - No hidden raw color-id ordering is used. All ranking decisions come from ARC RGB luminance, which is
#   visually observable in the grids themselves.
#
# f) Visual cleanliness
# - Real pieces remain area 8-13; legends are border singletons. This prevents confusion between objects
#   that move and symbols that instruct.
# - Each chain gets its own side-specific legend run whose span lies inside that chain's strip projection,
#   so a legend visually points to one chain rather than to the whole puzzle.
#
# g) Color-change inferability
# - The movable palette is fixed and small, and the recolor rule always moves toward the brighter end.
# - Every sample includes at least two depth-1 pieces, and many include a depth-2 tail, so the gradient
#   rule has repeated evidence rather than a single isolated recoloring.
#
# 9) New assumptions introduced here and generator guarantees
# - Assumption: every border legend run is strictly monotone in brightness.
#   Guarantee: the generator uses only [9,5,4] or its reverse-truncated variants.
# - Assumption: every source color belongs to the movable palette and has enough brighter headroom for its
#   chain depth.
#   Guarantee: source colors are drawn from a fixed movable palette and chosen with depth-safe indices.
# - Assumption: each chain's legend run lies within that chain's strip span.
#   Guarantee: legend spans are sampled inside the chosen strip before scattering pieces.

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import ARC_RGB, components_mask, find_components, get_bg

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIECE_MIN_AREA = 8
PIECE_MAX_AREA = 13
ORDER_MARKERS = [9, 5, 4]  # dark -> mid -> light
MOVABLE_COLOR_BASE = [6, 1, 2, 7, 3, 8]  # sorted by visible brightness, darkest -> brightest


def _brightness(color: int) -> float:
    r, g, b = ARC_RGB[int(color)]
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _pixels(shape, top, left):
    return {(int(top) + lr, int(left) + lc) for lr, lc in shape["rel"]}


def _connected(cells):
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in DIR4
                if (nb := (r + dr, c + dc)) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cmin = min((c for _, c in cells), default=B)
                    cmax = max((c for _, c in cells), default=A)
                    if cmin > A:
                        pref = [nb for nb in frontier if nb[1] < cmin]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cmax < B:
                        pref = [nb for nb in frontier if nb[1] > cmax]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                else:
                    rmin = min((r for r, _ in cells), default=B)
                    rmax = max((r for r, _ in cells), default=A)
                    if rmin > A:
                        pref = [nb for nb in frontier if nb[0] < rmin]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if rmax < B:
                        pref = [nb for nb in frontier if nb[0] > rmax]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue
        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis and span_range:
                    A, B = span_range
                    if must_span_axis == "cols":
                        if rc[1] == A and sum(1 for _, c2 in cells if c2 == A) <= 1:
                            continue
                        if rc[1] == B and sum(1 for _, c2 in cells if c2 == B) <= 1:
                            continue
                    else:
                        if rc[0] == A and sum(1 for r2, _ in cells if r2 == A) <= 1:
                            continue
                        if rc[0] == B and sum(1 for r2, _ in cells if r2 == B) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))
        if len(cells) != target:
            continue
        if must_span_axis and span_range:
            A, B = span_range
            if must_span_axis == "cols":
                if min(c for _, c in cells) != A or max(c for _, c in cells) != B:
                    continue
            else:
                if min(r for r, _ in cells) != A or max(r for r, _ in cells) != B:
                    continue
        if _connected(cells):
            return cells
    return None


def _interval_overlap(a0, a1, b0, b1, pad=0):
    return not (a1 + pad < b0 or b1 + pad < a0)


def _split_runs(vals):
    if not vals:
        return []
    runs = [[vals[0]]]
    for v in vals[1:]:
        if v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    return runs


def _legend_colors(length: int, direction: int) -> list[int]:
    if length == 2:
        colors = [ORDER_MARKERS[0], ORDER_MARKERS[2]]
    else:
        colors = ORDER_MARKERS[:length]
    return colors if direction > 0 else list(reversed(colors))


def _movable_palette(bar_color: int) -> list[int]:
    return [c for c in MOVABLE_COLOR_BASE if c != int(bar_color)]


def _transform_color(palette: list[int], color: int, depth: int) -> int:
    idx = palette.index(int(color))
    return int(palette[idx + int(depth)])


def _rank_window_ok(rank: int, depth: int, length: int, direction: int, palette_size: int) -> bool:
    rank = int(rank)
    depth = int(depth)
    length = int(length)
    direction = int(direction)
    remaining = length - depth - 1
    if rank < 0 or rank + depth >= palette_size:
        return False
    if direction > 0:
        return rank + remaining < palette_size
    return rank - remaining >= 0


def _available_sequences(palette: list[int], length: int, direction: int) -> list[list[int]]:
    seqs = []
    for comb in combinations(range(len(palette)), length):
        seq = list(comb) if direction > 0 else list(reversed(comb))
        if all(seq[depth] + depth < len(palette) for depth in range(length)):
            seqs.append(seq)
    return seqs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    if not comps:
        return g.copy()

    bar = comps[0]
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    palette = _movable_palette(bar_color)
    palette_index = {color: idx for idx, color in enumerate(palette)}
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}

    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        out[r, c] = bar_color

    def on_border(r, c):
        return r == 0 or c == 0 or r == H - 1 or c == W - 1

    legend_cells = {}
    pieces_raw = []
    for comp in comps[1:]:
        area = int(comp["area"])
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if area == 1 and any(on_border(r, c) for r, c in coords):
            r, c = coords[0]
            legend_cells[(r, c)] = int(comp["color"])
        elif PIECE_MIN_AREA <= area <= PIECE_MAX_AREA:
            pieces_raw.append(comp)

    for (r, c), color in legend_cells.items():
        out[r, c] = int(color)

    pieces = []
    for pi, p in enumerate(pieces_raw):
        color = int(p["color"])
        if color not in palette_index:
            continue
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append(
            {
                "id": pi,
                "color": color,
                "rank": int(palette_index[color]),
                "shape": _shape_from_coords(p["coords"], p["bbox"]),
                "orig": (r0, c0),
            }
        )

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    def bar_notches(mask, r0, c0):
        empty = ~mask
        h, w = map(int, empty.shape)
        notches = []
        for pts in components_mask(empty, connectivity=4):
            rr = pts[:, 0]
            cc = pts[:, 1]
            top = int((rr == 0).sum())
            bottom = int((rr == h - 1).sum())
            left = int((cc == 0).sum())
            right = int((cc == w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append(
                {
                    "axis": axis,
                    "side": side,
                    "hole": {(r0 + int(r), c0 + int(c)) for r, c in pts},
                    "mouth_n": mouth_n,
                }
            )
        return notches

    def parse_legends():
        legends = []

        top_cols = sorted(c for (r, c) in legend_cells if r == 0)
        for run in _split_runs(top_cols):
            colors = [legend_cells[(0, c)] for c in run]
            legends.append({"side": "top", "span": (run[0], run[-1]), "colors": colors})

        bottom_cols = sorted(c for (r, c) in legend_cells if r == H - 1)
        for run in _split_runs(bottom_cols):
            colors = [legend_cells[(H - 1, c)] for c in run]
            legends.append({"side": "bottom", "span": (run[0], run[-1]), "colors": colors})

        left_rows = sorted(r for (r, c) in legend_cells if c == 0)
        for run in _split_runs(left_rows):
            colors = [legend_cells[(r, 0)] for r in run]
            legends.append({"side": "left", "span": (run[0], run[-1]), "colors": colors})

        right_rows = sorted(r for (r, c) in legend_cells if c == W - 1)
        for run in _split_runs(right_rows):
            colors = [legend_cells[(r, W - 1)] for r in run]
            legends.append({"side": "right", "span": (run[0], run[-1]), "colors": colors})

        parsed = []
        for legend in legends:
            colors = legend["colors"]
            if not (2 <= len(colors) <= 3):
                continue
            if any(color not in ORDER_MARKERS for color in colors):
                continue
            vals = [_brightness(color) for color in colors]
            if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
                direction = 1
            elif all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
                direction = -1
            else:
                continue
            parsed.append(
                {
                    "side": legend["side"],
                    "span": legend["span"],
                    "colors": colors,
                    "direction": direction,
                    "length": len(colors),
                }
            )
        parsed.sort(key=lambda d: (["top", "bottom", "left", "right"].index(d["side"]), d["span"][0]))
        return parsed

    notches = bar_notches(bar_mask, bar_r0, bar_c0)
    legends = parse_legends()
    if not legends or not pieces:
        return out

    guide_occ = set(legend_cells)
    base_occ = set(bar_occ) | set(guide_occ)

    def placement_pts(shape, top, left, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + shape["h"] > H or left + shape["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or shape["w"] != B - A + 1:
                    return None
            else:
                if top != A or shape["h"] != B - A + 1:
                    return None
        pts = _pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, shape):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def root_candidates(legend):
        lo, hi = legend["span"]
        opts = []
        seen = set()
        for ni, notch in enumerate(notches):
            if notch["side"] != legend["side"]:
                continue
            axis = notch["axis"]
            for pi, piece in enumerate(pieces):
                shape = piece["shape"]
                if not _rank_window_ok(piece["rank"], 0, legend["length"], legend["direction"], len(palette)):
                    continue
                for top, left in hole_positions(notch["hole"], shape):
                    pts = placement_pts(shape, top, left, base_occ, must_cover=notch["hole"])
                    if pts is None:
                        continue
                    if axis == "cols":
                        A, B = int(left), int(left + shape["w"] - 1)
                    else:
                        A, B = int(top), int(top + shape["h"] - 1)
                    if not (A <= lo and hi <= B):
                        continue
                    key = (ni, pi, int(top), int(left), axis, A, B)
                    if key in seen:
                        continue
                    seen.add(key)
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    opts.append(
                        (
                            (int(mv), -int(notch["mouth_n"]), piece["rank"] * legend["direction"]),
                            {
                                "ni": ni,
                                "pi": pi,
                                "top": int(top),
                                "left": int(left),
                                "axis": axis,
                                "A": A,
                                "B": B,
                                "shape": shape,
                                "rank": piece["rank"],
                                "source_color": piece["color"],
                                "legend": legend,
                            },
                        )
                    )
        opts.sort(key=lambda x: x[0])
        return [x[1] for x in opts]

    legend_specs = [{"legend": legend, "roots": root_candidates(legend)} for legend in legends]
    if any(not spec["roots"] for spec in legend_specs):
        return out
    legend_specs.sort(key=lambda d: len(d["roots"]))

    best_out = out.copy()
    best_score = -1

    def maybe_record(out_grid, chains):
        nonlocal best_out, best_score
        placed = sum(chain["next_idx"] for chain in chains)
        if placed > best_score:
            best_score = placed
            best_out = out_grid.copy()

    def gen_moves(chain, out_grid, occ_set, remaining):
        if chain["next_idx"] >= chain["length"]:
            return []
        axis = chain["axis"]
        A, B = int(chain["A"]), int(chain["B"])
        head = chain["head"]
        head_pts = _pixels(head["shape"], head["top"], head["left"])
        head_top, head_left = int(head["top"]), int(head["left"])
        head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])
        depth = int(chain["next_idx"])

        cavity = {
            (r, c)
            for r in range(head_top, head_top + head_h)
            for c in range(head_left, head_left + head_w)
            if out_grid[r, c] == bg
        }
        if not cavity:
            return []

        moves = []
        for pi in remaining:
            piece = pieces[pi]
            rank = int(piece["rank"])
            if not _rank_window_ok(rank, depth, chain["length"], chain["direction"], len(palette)):
                continue
            if chain["direction"] > 0 and rank <= int(chain["prev_rank"]):
                continue
            if chain["direction"] < 0 and rank >= int(chain["prev_rank"]):
                continue
            shape = piece["shape"]
            for top, left in aligned_positions(axis, A, cavity, shape):
                pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                if pts is None:
                    continue
                if not touches(pts, head_pts):
                    continue
                moves.append((pi, int(top), int(left), pts, shape))
        moves.sort(key=lambda m: (pieces[m[0]]["rank"] * chain["direction"], len(m[3])))
        return moves

    def search_extensions(out_grid, occ_set, chains, remaining):
        maybe_record(out_grid, chains)
        done = True
        best_ci = None
        best_moves = None
        for ci, chain in enumerate(chains):
            if chain["next_idx"] >= chain["length"]:
                continue
            done = False
            moves = gen_moves(chain, out_grid, occ_set, remaining)
            if not moves:
                return False
            if best_moves is None or len(moves) < len(best_moves):
                best_ci = ci
                best_moves = moves
        if done:
            return True

        chain = chains[best_ci]
        for pi, top, left, pts, shape in best_moves:
            depth = int(chain["next_idx"])
            color = _transform_color(palette, pieces[pi]["color"], depth)
            for r, c in pts:
                out_grid[r, c] = int(color)
            old_head = chain["head"]
            old_idx = chain["next_idx"]
            old_rank = chain["prev_rank"]
            chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
            chain["next_idx"] += 1
            chain["prev_rank"] = int(pieces[pi]["rank"])
            rem2 = set(remaining)
            rem2.remove(pi)
            if search_extensions(out_grid, occ_set | set(pts), chains, rem2):
                return True
            chain["head"] = old_head
            chain["next_idx"] = old_idx
            chain["prev_rank"] = old_rank
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
            for (r, c), marker in legend_cells.items():
                out_grid[int(r), int(c)] = int(marker)
        return False

    def place_roots(idx, used_notches, out_grid, occ_set, chains, remaining):
        maybe_record(out_grid, chains)
        if idx == len(legend_specs):
            return search_extensions(out_grid, occ_set, chains, remaining)

        for root in legend_specs[idx]["roots"]:
            if root["ni"] in used_notches or root["pi"] not in remaining:
                continue
            pts = _pixels(root["shape"], root["top"], root["left"])
            if pts & occ_set:
                continue
            color = _transform_color(palette, root["source_color"], 0)
            for r, c in pts:
                out_grid[r, c] = int(color)
            chains.append(
                {
                    "axis": root["axis"],
                    "A": root["A"],
                    "B": root["B"],
                    "direction": root["legend"]["direction"],
                    "length": root["legend"]["length"],
                    "next_idx": 1,
                    "prev_rank": int(root["rank"]),
                    "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                }
            )
            used_notches.add(root["ni"])
            rem2 = set(remaining)
            rem2.remove(root["pi"])
            if place_roots(idx + 1, used_notches, out_grid, occ_set | pts, chains, rem2):
                return True
            used_notches.remove(root["ni"])
            chains.pop()
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
            for (r, c), marker in legend_cells.items():
                out_grid[int(r), int(c)] = int(marker)
        return False

    place_roots(0, set(), out.copy(), set(base_occ), [], set(range(len(pieces))))
    return best_out


def _gen_attempt(return_meta: bool = False):
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.choice(MOVABLE_COLOR_BASE)
    palette = _movable_palette(bar_color)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.38, 0.50) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = bar_r0 + thickness - 1
        if bar_r1 >= H:
            bar_r1 = H - 1
            bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
        chain_sides = ["top", "bottom"]
    else:
        center = int(rng.uniform(0.38, 0.50) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W:
            bar_c1 = W - 1
            bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = bar_c0, W - 1 - bar_c1
        chain_sides = ["left", "right"]

    chain_lens = list(rng.choice([[2, 2], [2, 3], [3, 2]]))
    dirs = [1, -1]
    rng.shuffle(dirs)

    if space_before < max(chain_lens) * 2 + 3 or space_after < max(chain_lens) * 2 + 3:
        return None

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}
    occ_set = set(bar_cells)
    notch_holes = []
    all_chain_pieces = []
    chain_meta = []

    for chain_idx, side in enumerate(chain_sides):
        chain_len = int(chain_lens[chain_idx])
        direction = int(dirs[chain_idx])
        strip_width = rng.randint(4, 5)

        placement = None
        for _ in range(100):
            if orientation == "horizontal":
                sc0 = rng.randint(1, W - strip_width - 1)
                sc1 = sc0 + strip_width - 1
                lg0 = rng.randint(sc0, sc1 - chain_len + 1)
                legend_span = (lg0, lg0 + chain_len - 1)
                conflict = any(_interval_overlap(sc0, sc1, meta["strip_span"][0], meta["strip_span"][1], pad=1) for meta in chain_meta)
                if conflict:
                    continue
                placement = {"sA": sc0, "sB": sc1, "legend_span": legend_span}
                break
            else:
                sr0 = rng.randint(1, H - strip_width - 1)
                sr1 = sr0 + strip_width - 1
                lg0 = rng.randint(sr0, sr1 - chain_len + 1)
                legend_span = (lg0, lg0 + chain_len - 1)
                conflict = any(_interval_overlap(sr0, sr1, meta["strip_span"][0], meta["strip_span"][1], pad=1) for meta in chain_meta)
                if conflict:
                    continue
                placement = {"sA": sr0, "sB": sr1, "legend_span": legend_span}
                break
        if placement is None:
            return None

        sA, sB = placement["sA"], placement["sB"]
        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sA, sB + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        seq_options = _available_sequences(palette, chain_len, direction)
        if not seq_options:
            return None
        src_indices = list(rng.choice(seq_options))

        chain_pieces = []
        for piece_idx in range(chain_len):
            piece_area = rng.randint(PIECE_MIN_AREA, PIECE_MAX_AREA)
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sA, sB
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                    else:
                        pc0, pc1 = min(c for _, c in notch), min(W - 1, bar_c1 + chain_dim)
                    pr0, pr1 = sA, sB

                bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                target = max(max(PIECE_MIN_AREA, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < PIECE_MIN_AREA:
                    return None

                avail = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) in notch or (r, c) not in occ_set
                }
                piece_cells = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if piece_cells is None:
                    return None
                if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in piece_cells):
                    return None
            else:
                prev_coords = chain_pieces[-1]["coords"]
                prs = [r for r, _ in prev_coords]
                pcs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prs), max(prs)
                prev_c0, prev_c1 = min(pcs), max(pcs)

                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_span = (
                        (max(0, prev_r0 - chain_dim), prev_r0 - 1)
                        if side == "top"
                        else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(ext_span[0], ext_span[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set}
                        if ext_span[0] <= ext_span[1]
                        else set()
                    )
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_span = (
                        (max(0, prev_c0 - chain_dim), prev_c0 - 1)
                        if side == "left"
                        else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(sA, sB + 1) for c in range(ext_span[0], ext_span[1] + 1) if (r, c) not in occ_set}
                        if ext_span[0] <= ext_span[1]
                        else set()
                    )
                if not cavity:
                    return None

                all_avail = cavity | ext
                target = min(PIECE_MAX_AREA, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > PIECE_MAX_AREA or target < PIECE_MIN_AREA:
                    return None
                is_tail = piece_idx == chain_len - 1
                if is_tail:
                    target = min(PIECE_MAX_AREA, len(all_avail))

                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    target,
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None

                if is_tail:
                    prs2 = [r for r, _ in piece_cells]
                    pcs2 = [c for _, c in piece_cells]
                    t_r0, t_r1 = min(prs2), max(prs2)
                    t_c0, t_c1 = min(pcs2), max(pcs2)
                    if orientation == "horizontal":
                        remaining = {
                            (r, c)
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(sA, sB + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(sA, sB + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None

            if not (PIECE_MIN_AREA <= len(piece_cells) <= PIECE_MAX_AREA):
                return None
            chain_pieces.append(
                {
                    "coords": piece_cells,
                    "depth": piece_idx,
                    "src_color": int(palette[src_indices[piece_idx]]),
                    "direction": direction,
                }
            )
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        chain_meta.append(
            {
                "side": side,
                "direction": direction,
                "strip_span": (sA, sB),
                "legend_span": placement["legend_span"],
                "legend_colors": _legend_colors(chain_len, direction),
                "length": chain_len,
            }
        )

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(500):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))

        d_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_h = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_h | {(r + dr, c + dc) for r, c in all_h for dr, dc in DIR4}
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 4:
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    legend_cells = {}
    legend_buffer = set()
    for meta in chain_meta:
        colors = meta["legend_colors"]
        if orientation == "horizontal":
            row = 0 if meta["side"] == "top" else H - 1
            cols = range(meta["legend_span"][0], meta["legend_span"][1] + 1)
            for color, c in zip(colors, cols):
                legend_cells[(row, c)] = int(color)
        else:
            col = 0 if meta["side"] == "left" else W - 1
            rows = range(meta["legend_span"][0], meta["legend_span"][1] + 1)
            for color, r in zip(colors, rows):
                legend_cells[(r, col)] = int(color)

    for r, c in legend_cells:
        legend_buffer.add((r, c))
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                legend_buffer.add((rr, cc))

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = int(bar_color)
    for (r, c), color in legend_cells.items():
        output[r, c] = int(color)
    for piece in all_chain_pieces:
        out_color = _transform_color(palette, piece["src_color"], piece["depth"])
        for r, c in piece["coords"]:
            output[r, c] = int(out_color)

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)
    for (r, c), color in legend_cells.items():
        input_grid[r, c] = int(color)

    placed_occ = set(bar_cells) | set(legend_buffer)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)

    for piece in pieces_order:
        coords = piece["coords"]
        color = int(piece["src_color"])
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1

        placed = False
        for _ in range(500):
            gap = rng.randint(2, 7)
            if orientation == "horizontal":
                top = bar_r0 - gap - p_h if rng.random() < 0.5 else bar_r1 + gap + 1
                left = rng.randint(1, max(1, W - p_w - 1))
            else:
                left = bar_c0 - gap - p_w if rng.random() < 0.5 else bar_c1 + gap + 1
                top = rng.randint(1, max(1, H - p_h - 1))
            if top < 1 or top + p_h > H - 1 or left < 1 or left + p_w > W - 1:
                continue
            pts = {(top + dr, left + dc) for dr, dc in rel}
            if pts & placed_occ:
                continue
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
                continue
            for r, c in pts:
                input_grid[r, c] = int(color)
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    meta = {
        "orientation": orientation,
        "chain_meta": chain_meta,
        "bar_color": int(bar_color),
        "palette": list(palette),
        "recolored_pieces": sum(1 for piece in all_chain_pieces if piece["depth"] > 0),
    }
    if not np.array_equal(main(input_grid), output):
        return None
    return (input_grid, output, meta) if return_meta else (input_grid, output)


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt(return_meta=False)
        if result is None:
            continue
        inp, _out = result
        return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
