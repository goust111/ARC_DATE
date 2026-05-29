# Variant of e8686506
# Dimension: D11 — 多步处理逻辑 (multi_step)
# Change: Step 1: 识别所有非背景碎片; Step 2: 通过边缘匹配将碎片拼合成矩形; Step 3: 紧凑裁剪输出... → 两阶段拼合：先拼子组件再合并

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D11 variant
# a) Concrete behavior and trace:
#    - The output keeps the original high-level look: a dense 5-column portrait rectangle with exact
#      left-right symmetry and a visible enclosing frame.
#    - The crucial change is in the input encoding. Each output color component is first cut into
#      2-3 disconnected fragments of the same color and only those fragments are scattered.
#    - This makes color mean "belongs to the same intermediate subcomponent", not "is already a final
#      piece". Example: originally a frame or cavity-fill region appeared directly as one piece type.
#      In the variant, a wide top chamber might appear as two same-colored scraps; they must be joined
#      first to recover that chamber shape, and only then can it be slotted into the final rectangle.
# b) Complexity calibration:
#    - The frame still constrains the final rectangle, but it is no longer given as a ready-made part.
#      The solver must enumerate candidate assemblies inside each color group before it can identify
#      the frame and its cavities.
#    - To keep the hierarchy visible, every non-background color appears multiple times in the input,
#      and at least one group is often split into 3 pieces instead of 2.
# c) Old assumptions that break under the variant:
#    - Original code grouped identical fragments globally and forbade same-color pieces from 4-touching.
#      That is invalid here because touching same-color fragments is exactly how a group is reassembled.
#    - Original frame detection searched for 1-2 already assembled 8-connected pieces whose union bbox
#      matched the output rectangle. That also fails here because the frame itself is fragmented.
#    - Original requirement-building heuristics for singleton corners/bars are no longer the main driver.
#      The new solver instead uses per-color candidate assemblies and frame-hole matching.
# d) Placement legality:
#    - The generator partitions each output component into connected fragments, extracts each fragment as
#      a patch, and scatters patches with non-overlap and Chebyshev padding 1.
#    - Oracle validation remains output-first: only inputs whose `main(input)` exactly reconstructs the
#      intended output are accepted.
# e) Visual observability:
#    - The two-stage rule is visible without using color ids as hidden metadata: repeated same-colored
#      scraps are plainly visible in the input, and the output shows those colors as single larger regions.
#    - Hole shapes are made distinct by width/height so the second-stage matching is inferable from the
#      examples rather than from arbitrary color ordering.
# f) Visual cleanliness:
#    - The output uses a connected frame with three interior cavities of widths 3 / 1 / 3, producing
#      clean separated regions and avoiding crowded overlays.
# g) New implicit assumptions guaranteed by the generator:
#    - Each non-background color has exactly 2 or 3 input fragments.
#    - Exactly one assembled color candidate is a connected LR-symmetric frame whose bbox is the full
#      output rectangle, and its complement consists of one hole per remaining color.
#    - Distinct holes have distinct normalized shapes, so the frame cavities determine a unique mapping
#      from assembled subcomponents to output positions.


_WIDTH = 5
_HEIGHT_OPTIONS = [9, 10]
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400
_CANDIDATE_LIMIT = 600


def _norm_coords_from_iter(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = _norm_coords_from_iter(pts)
    h = 1 + max(r for r, _ in coords)
    w = 1 + max(c for _, c in coords)
    return h, w, coords


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in coords), 1 + max(c for _, c in coords)


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            if h >= w:
                out.append((h, w))
    return out


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
        ordered = sorted(remaining, key=lambda idx: (-len(pieces[idx]), pieces[idx]))
        for idx in ordered:
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
                if h > 10 or w > _WIDTH:
                    continue
                rec(merged, tuple(j for j in remaining if j != idx))
                if len(results) > _CANDIDATE_LIMIT:
                    return

    rec(base, tuple(range(1, len(pieces))))
    out = sorted(results, key=lambda s: (len(s), _shape_dims(s), s))
    assert out, "each color group must yield at least one connected assembly candidate"
    return out


def _hole_specs(frame_coords: tuple[tuple[int, int], ...], h: int, w: int):
    mask = _coords_to_mask(frame_coords, h, w)
    holes = components_mask(~mask, connectivity=4)
    specs = []
    for pts in holes:
        norm = _norm_coords_from_iter(pts)
        coords = tuple(sorted((int(r), int(c)) for r, c in pts))
        specs.append({"shape": norm, "coords": coords})
    return specs


def _match_holes(
    candidates_by_color: dict[int, list[tuple[tuple[int, int], ...]]],
    frame_color: int,
    hole_specs: list[dict],
) -> dict[int, tuple[tuple[int, int], ...]] | None:
    other_colors = [c for c in sorted(candidates_by_color) if c != frame_color]
    if len(hole_specs) != len(other_colors):
        return None
    hole_shapes = [spec["shape"] for spec in hole_specs]
    if len(set(hole_shapes)) != len(hole_shapes):
        return None

    options = {}
    for col in other_colors:
        matches = [i for i, spec in enumerate(hole_specs) if spec["shape"] in candidates_by_color[col]]
        if not matches:
            return None
        options[col] = matches

    order = sorted(other_colors, key=lambda c: (len(options[c]), len(candidates_by_color[c]), c))
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def rec(i: int) -> bool:
        if i == len(order):
            return True
        col = order[i]
        for hole_idx in options[col]:
            if hole_idx in used:
                continue
            assignment[col] = hole_idx
            used.add(hole_idx)
            if rec(i + 1):
                return True
            used.remove(hole_idx)
            del assignment[col]
        return False

    if not rec(0):
        return None
    return {col: hole_specs[hole_idx]["coords"] for col, hole_idx in assignment.items()}


def _solve_from_candidates(
    candidates_by_color: dict[int, list[tuple[tuple[int, int], ...]]],
    h: int,
    w: int,
) -> np.ndarray:
    solutions: list[np.ndarray] = []
    for frame_color, cand_list in sorted(candidates_by_color.items()):
        for frame_coords in cand_list:
            if _shape_dims(frame_coords) != (h, w):
                continue
            if not _is_lr_symmetric(frame_coords, h, w):
                continue
            holes = _hole_specs(frame_coords, h, w)
            hole_map = _match_holes(candidates_by_color, frame_color, holes)
            if hole_map is None:
                continue
            board = np.full((h, w), frame_color, dtype=int)
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
    assert len(uniq) == 1, "variant assembly must yield exactly one rectangle"
    return next(iter(uniq.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a left-right symmetric rectangle by first re-forming same-color subcomponents, then merging those larger parts.

    Input structure:
    - The majority color is background.
    - Every non-background color appears as 2-3 disconnected fragments; fragments with the same color belong to one larger output subcomponent.
    - One color group reconstructs the enclosing symmetric frame, and the other color groups reconstruct the cavity-filling subcomponents inside it.

    Rule:
    - Collect 4-connected fragments per color and enumerate connected assemblies within each color group.
    - Infer the output rectangle size from total foreground area, then identify the assembled frame candidate whose bounding box is exactly that rectangle.
    - Read the frame's hole shapes and place the other assembled color groups into those holes to obtain the dense cropped rectangle.

    Key insight:
    - Color now encodes an intermediate grouping step: same-colored scraps must be merged locally before the final rectangle can be recognized, and the frame's cavities then force the second-stage placement.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[np.ndarray]] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        assert 2 <= len(comps) <= 3, "each color group must contain 2-3 fragments"
        comps_by_color[int(col)] = comps
        total_area += sum(len(pts) for pts in comps)

    assert total_area > 0, "grid must contain foreground fragments"
    dims = _factors(total_area)
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait factorization must be unique"
    H, W = best[0]
    assert W == _WIDTH and H in _HEIGHT_OPTIONS, "variant outputs use the fixed 5-column portrait family"

    candidates_by_color = {}
    for col, comps in comps_by_color.items():
        shapes = [_piece_sig(pts)[2] for pts in comps]
        candidates = _candidate_unions(shapes)
        candidates = [cand for cand in candidates if _shape_dims(cand)[0] <= H and _shape_dims(cand)[1] <= W]
        assert candidates, "a color group has no candidate that fits the output box"
        candidates_by_color[col] = candidates

    sol = _solve_from_candidates(candidates_by_color, H, W)
    return sol.astype(int)


# --------------- generator (o5x) ---------------


def _abstract_output(h: int) -> np.ndarray:
    total_hole_rows = h - 4
    options = []
    for mid_h in range(2, total_hole_rows - 1):
        rem = total_hole_rows - mid_h
        for top_h in range(1, rem):
            bot_h = rem - top_h
            if bot_h < 1 or top_h == bot_h:
                continue
            options.append((top_h, mid_h, bot_h))
    if not options:
        raise ValueError("no valid chamber heights")
    top_h, mid_h, bot_h = random.choice(options)

    rows = []
    rows.append([0, 0, 0, 0, 0])
    rows.extend([[0, -1, -1, -1, 0] for _ in range(top_h)])
    rows.append([0, 0, 0, 0, 0])
    rows.extend([[0, 0, -1, 0, 0] for _ in range(mid_h)])
    rows.append([0, 0, 0, 0, 0])
    rows.extend([[0, -1, -1, -1, 0] for _ in range(bot_h)])
    rows.append([0, 0, 0, 0, 0])
    out = np.array(rows, dtype=int)
    holes = components_mask(out == -1, connectivity=4)
    assert len(holes) == 3, "template must create exactly three cavities"
    shapes = [_norm_coords_from_iter(pts) for pts in holes]
    assert len(set(shapes)) == 3, "cavities must have distinct shapes"
    for idx, pts in enumerate(holes, start=1):
        for r, c in pts:
            out[int(r), int(c)] = idx
    return out


def _remap_colors(out: np.ndarray) -> tuple[np.ndarray, int]:
    bg = random.randint(0, 9)
    colors = sorted(set(map(int, np.unique(out))))
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    cmap = {col: palette[i] for i, col in enumerate(colors)}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg


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
    tries: int = 120,
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
        sizes = [len(part) for part in parts]
        if min(sizes) < min_size:
            continue
        return [_norm_coords_from_iter(part) for part in parts]
    return None


def _component_target_fragments(area: int, is_frame: bool) -> int:
    if is_frame:
        return 2
    if area >= 6 and random.random() < 0.45:
        return 3
    return 2


def _split_output_components(out: np.ndarray) -> list[tuple[int, list[tuple[tuple[int, int], ...]]]] | None:
    groups = []
    for col in sorted(map(int, np.unique(out))):
        comps = components_mask(out == col, connectivity=4)
        assert len(comps) == 1, "each output color must form exactly one connected component"
        comp = comps[0]
        norm = _norm_coords_from_iter(comp)
        is_frame = _shape_dims(norm) == out.shape
        k = _component_target_fragments(len(norm), is_frame)
        min_size = 2 if len(norm) >= 5 else 1
        parts = _partition_component(norm, k, min_size=min_size)
        if parts is None:
            return None
        candidates = _candidate_unions(parts)
        if norm not in candidates:
            return None
        if len(candidates) > _CANDIDATE_LIMIT:
            return None
        groups.append((col, parts))
    return groups


def _patch_from_shape(shape: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _scatter_groups(groups: list[tuple[int, list[tuple[tuple[int, int], ...]]]], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    patches = []
    for color, pieces in groups:
        for shape in pieces:
            patches.append(_patch_from_shape(shape, color, bg))
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _make_output() -> tuple[np.ndarray | None, int]:
    h = random.choice(_HEIGHT_OPTIONS)
    abstract = _abstract_output(h)
    return _remap_colors(abstract)


def _gen_attempt() -> np.ndarray | None:
    out, bg = _make_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None
    groups = _split_output_components(out)
    if groups is None:
        return None
    try:
        inp = _scatter_groups(groups, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D11 input")
