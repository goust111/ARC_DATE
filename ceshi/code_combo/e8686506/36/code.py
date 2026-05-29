# Combo variant of e8686506 (index 36)
# combo_spec: D9/2+D11/0+D14/0
# Name: 按行排列+两阶段拼合+颜色编码位置

import random

import numpy as np

from arc_utils import components_mask, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The original task showed ready-made fragments scattered independently, and the solver could often
#      reason directly about final placement from mirror symmetry plus a preserved frame.
#    - Here every output component is first split into exactly two same-colored scraps. Those scraps are
#      not random on the canvas: warm-colored groups are laid out together on an upper input row, the
#      neutral frame fragments sit on a middle row, and cool-colored groups are laid out on a lower row.
#    - Small trace: two warm 1x3 chambers in the final rectangle are deliberately given different warm
#      colors but identical shapes. In the original logic, shape alone would not decide which warm color
#      goes to the higher chamber. In the combo, the left-to-right order inside the upper input row gives
#      that extra clue: the earlier warm group maps to the higher warm chamber, and the later warm group
#      maps to the lower warm chamber. The same happens for the two cool lower chambers.
# b) Complexity calibration:
#    - The color-position rule is coarse on purpose: only two semantic bands are used, warm=upper and
#      cool=lower. This is strong enough to be visible across examples, but not so specific that each
#      color id would secretly encode one exact row.
#    - Row arrangement matters because each semantic band contains two output holes with identical shapes.
#      If those shapes were all distinct, the input rows would be decorative only. Using repeated shapes
#      inside a band forces the solver to combine band semantics with the ordered row layout.
#    - Every non-background color still appears multiple times in the input, so the two-stage regrouping
#      remains visually obvious rather than collapsing back to the original one-stage assembly.
# c) Old assumptions that break and how the combo replaces them:
#    - Original same-color non-touching is no longer valid. Same-colored scraps must touch after the
#      first-stage reassembly, so the solver now groups by color and enumerates connected unions.
#    - Original frame detection assumed the frame already existed as 1-2 placed outline parts. Here the
#      frame is itself fragmented, so the solver must first reconstruct each color group candidate and
#      only then ask which color can be the full neutral frame.
#    - Original anchor heuristics from singleton corners / bars are no longer the main driver. The combo
#      instead relies on a fixed dense frame-with-holes family, semantic upper/lower color bands, and the
#      left-to-right ordering of color groups inside their input rows.
# d) Placement legality:
#    - The generator is output-first. It builds a dense 5-column symmetric rectangle, splits each of its
#      five color components into two connected fragments, and then places those fragment patches on three
#      non-overlapping horizontal lanes with padding.
#    - Oracle validation remains strict: only an input for which `main(input)` exactly reproduces the
#      intended output is accepted.
# e) Visual observability:
#    - A human can see three independent clues in examples: same-colored scraps imply an intermediate
#      grouping step, warm scraps live on the upper lane while cool scraps live on the lower lane, and
#      within each lane the groups are ordered left-to-right in the same sequence as their final vertical
#      stacking inside that semantic band.
#    - No hidden ordering by numeric color id is used. Only visible ARC color families and visible lane
#      order participate in the rule.
# f) Visual cleanliness:
#    - The output uses one neutral enclosing frame and four clean cavities separated by solid bars, so the
#      holes are easy to distinguish and there is no overdraw or marking clutter.
#    - The input likewise avoids random crowding by using one upper lane, one frame lane, and one lower
#      lane instead of a fully scattered field.
# g) New implicit assumptions guaranteed by the generator:
#    - Exactly one non-semantic color group reconstructs a connected LR-symmetric frame whose bbox is the
#      full output rectangle.
#    - The two warm holes share one normalized shape, and the two cool holes share another normalized
#      shape, so band semantics alone is insufficient and row order is genuinely needed.
#    - Within each semantic band, the mean input x-position of a color group matches the top-to-bottom
#      order of that group's target hole in the output.


_WIDTH = 5
_HEIGHT = 11
_CANVAS_H = 17
_CANVAS_W = 20
_GEN_MAX_RETRIES = 400
_CANDIDATE_LIMIT = 500

_WARM_COLORS = (2, 4, 7)
_COOL_COLORS = (1, 5, 8)
_FRAME_COLORS = (3, 6, 9)
_BG_COLORS = (0,)


def _norm_coords_from_iter(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in coords), 1 + max(c for _, c in coords)


def _coords_to_mask(coords: tuple[tuple[int, int], ...], h: int | None = None, w: int | None = None) -> np.ndarray:
    if h is None or w is None:
        hh, ww = _shape_dims(coords)
        if h is None:
            h = hh
        if w is None:
            w = ww
    mask = np.zeros((int(h), int(w)), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _is_lr_symmetric(coords: tuple[tuple[int, int], ...], h: int | None = None, w: int | None = None) -> bool:
    mask = _coords_to_mask(coords, h, w)
    return np.array_equal(mask, mask[:, ::-1])


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            if h >= w:
                out.append((h, w))
    return out


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = _norm_coords_from_iter(pts)
    h, w = _shape_dims(coords)
    return h, w, coords


def _color_band(col: int) -> str | None:
    if int(col) in _WARM_COLORS:
        return "top"
    if int(col) in _COOL_COLORS:
        return "bottom"
    return None


def _placement_band(coords: tuple[tuple[int, int], ...], H: int) -> str:
    mean_r = float(sum(r for r, _ in coords)) / float(len(coords))
    return "top" if mean_r < (H / 2.0) else "bottom"


def _candidate_unions(piece_shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[tuple[int, int], ...]]:
    pieces = [tuple(shape) for shape in piece_shapes]
    pieces.sort(key=lambda s: (-len(s), s))
    base = set(pieces[0])
    results: set[tuple[tuple[int, int], ...]] = set()
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def rec(union: set[tuple[int, int]], remaining: tuple[int, ...]) -> None:
        if len(results) > _CANDIDATE_LIMIT:
            return
        if not remaining:
            results.add(_norm_coords_from_iter(union))
            return
        idx = remaining[0]
        shape = pieces[idx]
        translations = set()
        for fr, fc in shape:
            for ur, uc in union:
                for dr, dc in dirs4:
                    translations.add((ur + dr - fr, uc + dc - fc))
        for tr, tc in translations:
            placed = {(r + tr, c + tc) for r, c in shape}
            if placed & union:
                continue
            if not any((r + dr, c + dc) in union for r, c in placed for dr, dc in dirs4):
                continue
            merged = union | placed
            norm = _norm_coords_from_iter(merged)
            h, w = _shape_dims(norm)
            if h > _HEIGHT or w > _WIDTH:
                continue
            rec(merged, tuple(j for j in remaining if j != idx))
            if len(results) > _CANDIDATE_LIMIT:
                return

    rec(base, tuple(range(1, len(pieces))))
    out = sorted(results, key=lambda s: (len(s), _shape_dims(s), s))
    assert out, "each color group must yield at least one connected assembly candidate"
    return out


def _hole_specs(frame_coords: tuple[tuple[int, int], ...], h: int, w: int) -> list[dict]:
    mask = _coords_to_mask(frame_coords, h, w)
    holes = components_mask(~mask, connectivity=4)
    specs = []
    for pts in holes:
        norm = _norm_coords_from_iter(pts)
        coords = tuple(sorted((int(r), int(c)) for r, c in pts))
        mean_r = float(sum(r for r, _ in coords)) / float(len(coords))
        specs.append(
            {
                "shape": norm,
                "coords": coords,
                "band": "top" if mean_r < (h / 2.0) else "bottom",
                "top": min(r for r, _ in coords),
            }
        )
    specs.sort(key=lambda spec: (spec["top"], spec["coords"]))
    return specs


def _group_stats(comps_by_color: dict[int, list[np.ndarray]]) -> dict[int, dict]:
    stats = {}
    for col, comps in comps_by_color.items():
        pts = [(int(r), int(c)) for comp in comps for r, c in comp]
        stats[int(col)] = {
            "mean_r": float(sum(r for r, _ in pts)) / float(len(pts)),
            "mean_c": float(sum(c for _, c in pts)) / float(len(pts)),
            "band": _color_band(int(col)),
        }
    return stats


def _solve_from_candidates(
    candidates_by_color: dict[int, list[tuple[tuple[int, int], ...]]],
    stats_by_color: dict[int, dict],
    h: int,
    w: int,
) -> np.ndarray:
    frame_colors = [col for col, stats in stats_by_color.items() if stats["band"] is None]
    assert len(frame_colors) == 1, "combo inputs must contain exactly one neutral frame color"
    frame_color = int(frame_colors[0])

    solutions: list[np.ndarray] = []
    for frame_coords in candidates_by_color[frame_color]:
        if _shape_dims(frame_coords) != (h, w):
            continue
        if not _is_lr_symmetric(frame_coords, h, w):
            continue
        holes = _hole_specs(frame_coords, h, w)
        other_colors = [c for c in sorted(candidates_by_color) if c != frame_color]
        if len(holes) != len(other_colors):
            continue

        band_to_holes = {
            "top": [spec for spec in holes if spec["band"] == "top"],
            "bottom": [spec for spec in holes if spec["band"] == "bottom"],
        }
        band_to_colors = {
            "top": sorted(
                [c for c in other_colors if stats_by_color[c]["band"] == "top"],
                key=lambda c: (stats_by_color[c]["mean_c"], c),
            ),
            "bottom": sorted(
                [c for c in other_colors if stats_by_color[c]["band"] == "bottom"],
                key=lambda c: (stats_by_color[c]["mean_c"], c),
            ),
        }
        if any(len(band_to_holes[band]) != len(band_to_colors[band]) for band in ("top", "bottom")):
            continue

        hole_map: dict[int, tuple[tuple[int, int], ...]] = {}
        ok = True
        for band in ("top", "bottom"):
            ordered_holes = sorted(band_to_holes[band], key=lambda spec: (spec["top"], spec["coords"]))
            ordered_colors = band_to_colors[band]
            for col, spec in zip(ordered_colors, ordered_holes):
                if spec["shape"] not in candidates_by_color[col]:
                    ok = False
                    break
                hole_map[int(col)] = spec["coords"]
            if not ok:
                break
        if not ok:
            continue

        board = np.full((h, w), int(frame_color), dtype=int)
        frame_mask = _coords_to_mask(frame_coords, h, w)
        board[~frame_mask] = -1
        for col, coords in hole_map.items():
            for r, c in coords:
                board[int(r), int(c)] = int(col)
        if np.any(board < 0):
            continue
        if not np.array_equal(board, board[:, ::-1]):
            continue
        solutions.append(board)

    uniq = {}
    for board in solutions:
        uniq[tuple(map(tuple, board.tolist()))] = board
    assert len(uniq) == 1, "combo assembly must yield exactly one rectangle"
    return next(iter(uniq.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a dense left-right symmetric rectangle by first merging same-colored scraps, then placing those re-formed parts with help from semantic colors and row order.

    Input structure:
    - The most frequent color is background.
    - Every non-background color is split into exactly two disconnected fragments.
    - Warm-colored groups sit on an upper input lane, a neutral-colored group forms the fragmented frame lane, and cool-colored groups sit on a lower input lane.

    Rule:
    - Group 4-connected fragments by color and enumerate each color's connected reassembly candidates.
    - Infer the output rectangle size from total foreground area, then identify the neutral color whose assembled candidate is the full left-right symmetric frame.
    - Read the frame's hole shapes, use warm/cool color bands to decide upper versus lower holes, and use the left-to-right order of color groups inside each input lane to assign same-band holes from top to bottom.

    Key insight:
    - The puzzle is no longer solved in one placement pass: color first tells which scraps belong together, then color semantics narrow the vertical band, and finally the ordered row layout resolves which same-band chamber each assembled subcomponent occupies.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[np.ndarray]] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        assert len(comps) == 2, "each color group must contain exactly two fragments"
        comps_by_color[int(col)] = comps
        total_area += sum(len(pts) for pts in comps)

    assert total_area > 0, "grid must contain foreground fragments"
    dims = _factors(total_area)
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait factorization must be unique"
    H, W = best[0]
    assert (H, W) == (_HEIGHT, _WIDTH), "combo generator uses a fixed 11x5 portrait family"

    stats_by_color = _group_stats(comps_by_color)
    top_rows = [stats["mean_r"] for stats in stats_by_color.values() if stats["band"] == "top"]
    mid_rows = [stats["mean_r"] for stats in stats_by_color.values() if stats["band"] is None]
    bot_rows = [stats["mean_r"] for stats in stats_by_color.values() if stats["band"] == "bottom"]
    assert len(top_rows) == 2 and len(mid_rows) == 1 and len(bot_rows) == 2, "must have 2 warm, 1 neutral, 2 cool groups"
    assert max(top_rows) < mid_rows[0] < min(bot_rows), "input rows must be ordered top / frame / bottom"

    candidates_by_color = {}
    for col, comps in comps_by_color.items():
        shapes = [_piece_sig(pts)[2] for pts in comps]
        candidates = _candidate_unions(shapes)
        candidates = [cand for cand in candidates if _shape_dims(cand)[0] <= H and _shape_dims(cand)[1] <= W]
        assert candidates, "a color group has no candidate that fits the output box"
        candidates_by_color[int(col)] = candidates

    sol = _solve_from_candidates(candidates_by_color, stats_by_color, H, W)
    return sol.astype(int)


# --------------- generator ---------------


def _abstract_output() -> tuple[np.ndarray, list[dict]]:
    frame = 0
    hole_ids = [1, 2, 3, 4]
    rows = [
        [frame, frame, frame, frame, frame],
        [frame, hole_ids[0], hole_ids[0], hole_ids[0], frame],
        [frame, frame, frame, frame, frame],
        [frame, hole_ids[1], hole_ids[1], hole_ids[1], frame],
        [frame, frame, frame, frame, frame],
        [frame, frame, hole_ids[2], frame, frame],
        [frame, frame, hole_ids[2], frame, frame],
        [frame, frame, frame, frame, frame],
        [frame, frame, hole_ids[3], frame, frame],
        [frame, frame, hole_ids[3], frame, frame],
        [frame, frame, frame, frame, frame],
    ]
    grid = np.array(rows, dtype=int)
    holes = components_mask(grid != frame, connectivity=4)
    specs = []
    for idx, pts in enumerate(sorted(holes, key=lambda p: int(p[:, 0].min()))):
        norm = _norm_coords_from_iter(pts)
        coords = tuple(sorted((int(r), int(c)) for r, c in pts))
        specs.append({"id": idx + 1, "shape": norm, "coords": coords})
    return grid, specs


def _remap_colors(abstract: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    bg = int(random.choice(_BG_COLORS))
    frame_color = int(random.choice(_FRAME_COLORS))
    warm = list(_WARM_COLORS)
    cool = list(_COOL_COLORS)
    random.shuffle(warm)
    random.shuffle(cool)
    cmap = {
        0: frame_color,
        1: int(warm[0]),
        2: int(warm[1]),
        3: int(cool[0]),
        4: int(cool[1]),
    }
    return np.vectorize(lambda x: cmap[int(x)])(abstract), bg, cmap


def _adjacency_map(cells: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adj = {}
    for cell in cells:
        r, c = cell
        nbrs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if nxt in cells:
                nbrs.append(nxt)
        adj[cell] = nbrs
    return adj


def _partition_component(
    coords: tuple[tuple[int, int], ...],
    k: int,
    *,
    min_size: int = 1,
    tries: int = 150,
) -> list[tuple[tuple[int, int], ...]] | None:
    cells = set(coords)
    adj = _adjacency_map(cells)
    cell_list = list(cells)
    if k == 1:
        return [coords]

    for _ in range(tries):
        seeds = [random.choice(cell_list)]
        while len(seeds) < k:
            choices = [cell for cell in cell_list if cell not in seeds]
            if not choices:
                break
            nxt = max(
                choices,
                key=lambda cell: min(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) for s in seeds),
            )
            seeds.append(nxt)
        if len(seeds) < k:
            continue

        regions = {i: {seed} for i, seed in enumerate(seeds)}
        unassigned = set(cells) - set(seeds)
        ok = True
        while unassigned:
            candidates = []
            for cell in list(unassigned):
                touching = [i for i, region in regions.items() if any(n in region for n in adj[cell])]
                if touching:
                    candidates.append((cell, touching))
            if not candidates:
                ok = False
                break
            cell, touching = random.choice(candidates)
            touching.sort(key=lambda i: (len(regions[i]), random.random()))
            owner_idx = touching[0]
            regions[owner_idx].add(cell)
            unassigned.remove(cell)
        if not ok:
            continue

        parts = [tuple(sorted(region)) for _, region in sorted(regions.items())]
        if min(len(part) for part in parts) < min_size:
            continue
        return [_norm_coords_from_iter(part) for part in parts]
    return None


def _split_output_components(out: np.ndarray) -> list[tuple[int, list[tuple[tuple[int, int], ...]]]] | None:
    groups = []
    for col in sorted(map(int, np.unique(out))):
        comps = components_mask(out == col, connectivity=4)
        assert len(comps) == 1, "each output color must form one connected component"
        norm = _norm_coords_from_iter(comps[0])
        min_size = 2 if len(norm) >= 4 else 1
        parts = _partition_component(norm, 2, min_size=min_size)
        if parts is None:
            return None
        candidates = _candidate_unions(parts)
        if norm not in candidates:
            return None
        groups.append((int(col), parts))
    return groups


def _patch_from_shape(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _lane_pack(
    canvas: np.ndarray,
    patches: list[np.ndarray],
    top: int,
    bg: int,
    left: int = 1,
    gap: int = 1,
) -> None:
    x = int(left)
    for patch in patches:
        h, w = patch.shape
        if x + w > canvas.shape[1] - 1:
            raise ValueError("lane overflow")
        paste(canvas, patch, int(top), int(x), transparent=int(bg))
        x += int(w) + int(gap)


def _color_order_for_output(out: np.ndarray, frame_color: int) -> dict[str, list[int]]:
    top_colors = []
    bot_colors = []
    for col in sorted(map(int, np.unique(out))):
        if col == frame_color:
            continue
        pts = components_mask(out == col, connectivity=4)[0]
        norm = _norm_coords_from_iter(pts)
        mean_r = float(sum(int(r) for r, _ in pts)) / float(len(pts))
        band = "top" if mean_r < (_HEIGHT / 2.0) else "bottom"
        if band == "top":
            top_colors.append((min(r for r, _ in norm), int(col)))
        else:
            bot_colors.append((min(r for r, _ in norm), int(col)))
    top_colors.sort()
    bot_colors.sort()
    return {
        "top": [col for _, col in top_colors],
        "bottom": [col for _, col in bot_colors],
    }


def _arrange_rows(groups: list[tuple[int, list[tuple[tuple[int, int], ...]]]], out: np.ndarray, bg: int) -> np.ndarray:
    canvas = np.full((_CANVAS_H, _CANVAS_W), int(bg), dtype=int)
    by_color = {int(col): parts for col, parts in groups}
    frame_color = next(col for col in by_color if _color_band(col) is None)
    order = _color_order_for_output(out, frame_color)

    def group_patches(color: int) -> list[np.ndarray]:
        parts = list(by_color[int(color)])
        random.shuffle(parts)
        return [_patch_from_shape(shape, int(color), int(bg)) for shape in parts]

    top_patches = []
    for col in order["top"]:
        top_patches.extend(group_patches(int(col)))
    mid_patches = group_patches(int(frame_color))
    bot_patches = []
    for col in order["bottom"]:
        bot_patches.extend(group_patches(int(col)))

    _lane_pack(canvas, top_patches, top=1, bg=bg)
    _lane_pack(canvas, mid_patches, top=3, bg=bg)
    _lane_pack(canvas, bot_patches, top=15, bg=bg)
    return canvas


def _piece_count(grid: np.ndarray, bg: int) -> int:
    return sum(len(components_mask(grid == col, connectivity=4)) for col in np.unique(grid) if int(col) != int(bg))


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = get_bg(inp)
    comps_by_color = {int(col): components_mask(inp == col, connectivity=4) for col in np.unique(inp) if int(col) != int(bg)}
    stats = _group_stats(comps_by_color)
    frame_color = next(col for col, s in stats.items() if s["band"] is None)
    top_order_in = [c for c, _ in sorted(((c, s["mean_c"]) for c, s in stats.items() if s["band"] == "top"), key=lambda x: x[1])]
    bot_order_in = [c for c, _ in sorted(((c, s["mean_c"]) for c, s in stats.items() if s["band"] == "bottom"), key=lambda x: x[1])]
    out_order = _color_order_for_output(out, frame_color)
    top_shapes = []
    bot_shapes = []
    for col in np.unique(out):
        if int(col) == int(frame_color):
            continue
        pts = components_mask(out == col, connectivity=4)[0]
        norm = _norm_coords_from_iter(pts)
        mean_r = float(sum(int(r) for r, _ in pts)) / float(len(pts))
        if mean_r < (out.shape[0] / 2.0):
            top_shapes.append(norm)
        else:
            bot_shapes.append(norm)
    return {
        "top_order_match": int(top_order_in == out_order["top"]),
        "bottom_order_match": int(bot_order_in == out_order["bottom"]),
        "top_same_shape_pair": int(len(top_shapes) == 2 and top_shapes[0] == top_shapes[1]),
        "bottom_same_shape_pair": int(len(bot_shapes) == 2 and bot_shapes[0] == bot_shapes[1]),
        "fragment_group_count": int(all(len(comps) == 2 for comps in comps_by_color.values())),
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    abstract, _hole_specs_abstract = _abstract_output()
    out, bg, _cmap = _remap_colors(abstract)

    if not np.array_equal(out, out[:, ::-1]):
        return None
    if len(np.unique(out)) != 5:
        return None

    groups = _split_output_components(out)
    if groups is None:
        return None

    try:
        inp = _arrange_rows(groups, out, bg)
        got = main(inp)
    except Exception:
        return None

    if not np.array_equal(got, out):
        return None
    metrics = _fusion_metrics(inp, out)
    if not all(metrics.values()):
        return None
    if _piece_count(inp, bg) != 10:
        return None
    return inp, out, metrics


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _out, _metrics = result
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
