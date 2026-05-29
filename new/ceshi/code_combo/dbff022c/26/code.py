# Combo variant of dbff022c (index 26)
# combo_spec: D1/3+D4/1+D5/3
# Name: 条件映射 + 嵌套精灵 + 渐进填充

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import components_mask, neighbors4

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# design_analysis:
#   concrete_behavior_and_trace:
#     input_pixels:
#       - one grid edge carries a 6xN or Nx6 palette block.
#       - read from the edge inward, the six palette lines are:
#           large keys / large outer-layer fills / large inner-layer fills /
#           small keys / small outer-layer fills / small inner-layer fills
#       - each relevant sprite is a nested double frame:
#           outer frame color / two-cell-thick bg band / inner frame color / center bg hole
#       - only the band between frames is blank in the input; the center remains blank forever.
#     output_pixels:
#       - choose large-vs-small palette group from the outer frame's visible area.
#       - inside the middle band, cells at distance 1 from the outer frame get the group's
#         first fill color, and cells at distance 2 get the group's second fill color.
#       - the inner frame and the center hole are unchanged.
#     example_trace:
#       - suppose the top palette shows:
#           row0 = [red, blue]
#           row1 = [yellow, grey]
#           row2 = [orange, green]
#           row3 = [red, blue]
#           row4 = [grey, yellow]
#           row5 = [green, orange]
#       - a large red nested sprite uses rows 0-2, so its band becomes:
#           outer band layer = yellow, inner band layer = orange
#       - a small blue nested sprite uses rows 3-5, so its band becomes:
#           outer band layer = yellow, inner band layer = orange
#         but only because that key's small-group column says so; a different column can swap.
#       - the old task would have filled one generic hole with one color.
#         the combo task must first isolate the middle band, then decide large/small, then
#         assign two colors by distance from the outer frame.
#
#   complexity_calibration:
#     - only two size bands exist, with a deliberate area gap between them.
#     - every mapped sprite has exactly two distance layers, so the progressive behavior is visible
#       but not too many palette lines are needed.
#     - palette length stays at 2-3 keys to keep color reasoning human-readable.
#     - both large and small mapped sprites are always present, preventing silent degeneration to
#       a single active group.
#
#   implicit_assumption_review:
#     - the source solver assumed a 2-line palette; this combo needs 6 ordered lines.
#     - the source solver assumed one enclosed bg hole per mapped border; nested sprites violate that,
#       so the solver now filters for bg regions touching both the outer frame and another non-bg color.
#     - the source solver assumed one fill color per border key; now each key maps to a pair of fills,
#       and size decides which pair is active.
#     - progressive filling assumes the target band is connected and every target cell is reachable
#       from a cell adjacent to the outer frame. The generator guarantees this by using clean double frames
#       with a two-cell-thick band.
#
#   placement_legality:
#     - the 6-line palette needs a larger canvas than the source task, so the generator raises both
#       grid dimensions.
#     - sprites are placed with a 1-cell moat from each other and from the palette.
#     - sprite bboxes never touch the outer grid boundary, so the center hole and the band are always
#       proper enclosed background regions.
#
#   visual_observability_and_cleanliness:
#     - the repeated key row in lines 0 and 3 makes the conditional mapping visible to a human.
#     - the untouched center hole is the cue that only the middle band is transformable.
#     - the two-color onion band is the cue that the fill is distance-based rather than uniform.
#     - geometry is kept axis-aligned and symmetric so the grid stays legible instead of noisy.
#
#   inferability_of_color_changes:
#     - the palette itself explains the two layer colors by position: the first fill row in a group
#       is for the outer band layer, the second is for the next inner layer.
#     - fill rows are column-wise different within each group and the large/small groups differ as well,
#       so both the layer rule and the size rule leave visible evidence in 2-4 examples.
#
#   main_assumption_audit:
#     - _best_palette assumes the six edge-inward lines contain no background cells, lines 0 and 3 match
#       exactly, and the palette is color-isolated from sprites. The generator enforces all three.
#     - _collect_target_middle_regions assumes the only enclosed bg component that touches both an outer
#       frame and another non-bg color is the middle band. The generator uses one inner frame per sprite,
#       so this holds.
#     - _fill_progressive_band assumes the band has at most two meaningful distances from the outer frame.
#       The generator fixes the inner frame inset at 3, making the band exactly two layers deep.
#     - size selection uses a midpoint between the smallest and largest mapped outer-frame areas.
#       The generator always includes at least one large and one small mapped sprite with disjoint area bands.
# ---------------------------------------------------------------------------


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    """Boolean mask covering the full 6xN / Nx6 palette footprint."""
    h, w = shape
    pal_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        pal_mask[0:6, start:end] = True
    elif side == "bottom":
        pal_mask[h - 6 : h, start:end] = True
    elif side == "left":
        pal_mask[start:end, 0:6] = True
    else:
        pal_mask[start:end, w - 6 : w] = True
    return pal_mask


def _palette_lines(g: np.ndarray, side: str, start: int, end: int) -> list[np.ndarray]:
    """Return the palette's six lines in edge-inward order."""
    h, w = g.shape
    if side == "top":
        return [g[0, start:end], g[1, start:end], g[2, start:end], g[3, start:end], g[4, start:end], g[5, start:end]]
    if side == "bottom":
        return [g[h - 1, start:end], g[h - 2, start:end], g[h - 3, start:end], g[h - 4, start:end], g[h - 5, start:end], g[h - 6, start:end]]
    if side == "left":
        return [g[start:end, 0], g[start:end, 1], g[start:end, 2], g[start:end, 3], g[start:end, 4], g[start:end, 5]]
    return [g[start:end, w - 1], g[start:end, w - 2], g[start:end, w - 3], g[start:end, w - 4], g[start:end, w - 5], g[start:end, w - 6]]


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 6xN or Nx6 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        pal = _palette_mask(g.shape, side, start, end)
        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def valid_candidate(side: str, start: int, end: int) -> tuple[bool, tuple[int, int, int, int] | None]:
        lines = _palette_lines(g, side, start, end)
        block = np.stack(lines, axis=0)
        if np.any(block == bg):
            return False, None

        large_keys = [int(x) for x in lines[0].tolist()]
        large_outer = [int(x) for x in lines[1].tolist()]
        large_inner = [int(x) for x in lines[2].tolist()]
        small_keys = [int(x) for x in lines[3].tolist()]
        small_outer = [int(x) for x in lines[4].tolist()]
        small_inner = [int(x) for x in lines[5].tolist()]

        if len(large_keys) < 2:
            return False, None
        if len(large_keys) != len(set(large_keys)):
            return False, None
        if large_keys != small_keys:
            return False, None
        for row in (large_outer, large_inner, small_outer, small_inner):
            if len(row) != len(set(row)):
                return False, None
        if not any(a != b for a, b in zip(large_outer, large_inner, strict=False)):
            return False, None
        if not any(a != b for a, b in zip(small_outer, small_inner, strict=False)):
            return False, None
        if not any(
            (lo, li) != (so, si)
            for lo, li, so, si in zip(large_outer, large_inner, small_outer, small_inner, strict=False)
        ):
            return False, None
        if not isolated(side, start, end):
            return False, None

        distinct = len(set(large_keys) | set(large_outer) | set(large_inner) | set(small_outer) | set(small_inner))
        score = (len(large_keys), distinct, -(end - start), block.size)
        return True, score

    def consider_top() -> None:
        nonlocal best
        if h < 6:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                ok, score = valid_candidate("top", c0, c1)
                if ok:
                    cand = (score, "top", c0, c1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 6:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                ok, score = valid_candidate("bottom", c0, c1)
                if ok:
                    cand = (score, "bottom", c0, c1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 6:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                ok, score = valid_candidate("left", r0, r1)
                if ok:
                    cand = (score, "left", r0, r1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 6:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                ok, score = valid_candidate("right", r0, r1)
                if ok:
                    cand = (score, "right", r0, r1)
                    if best is None or cand[0] > best[0]:
                        best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_groups(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]], np.ndarray]:
    """Return (large_group, small_group, palette_mask) where each group maps key -> (layer1, layer2)."""
    side, start, end = _best_palette(g, bg)
    pal_mask = _palette_mask(g.shape, side, start, end)
    lines = _palette_lines(g, side, start, end)

    large_keys = [int(x) for x in lines[0].tolist()]
    large_outer = [int(x) for x in lines[1].tolist()]
    large_inner = [int(x) for x in lines[2].tolist()]
    small_keys = [int(x) for x in lines[3].tolist()]
    small_outer = [int(x) for x in lines[4].tolist()]
    small_inner = [int(x) for x in lines[5].tolist()]

    assert large_keys == small_keys, "_palette_groups: both size groups must reuse the same keys"
    large_group = {
        key: (outer_fill, inner_fill)
        for key, outer_fill, inner_fill in zip(large_keys, large_outer, large_inner, strict=False)
    }
    small_group = {
        key: (outer_fill, inner_fill)
        for key, outer_fill, inner_fill in zip(small_keys, small_outer, small_inner, strict=False)
    }
    return large_group, small_group, pal_mask


def _touches_local_boundary(pts: np.ndarray, h: int, w: int) -> bool:
    return any(int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1 for r, c in pts)


def _collect_target_middle_regions(
    g: np.ndarray,
    bg: int,
    keyset: set[int],
    pal_mask: np.ndarray,
) -> list[tuple[int, np.ndarray, int]]:
    """
    Return [(outer_color, abs_pts, outer_area)] for bg regions that are:
      - inside the bbox of a mapped outer-frame component,
      - adjacent to that outer frame,
      - and adjacent to some other non-bg color as well.

    This isolates the band between the outer and inner frames.
    """
    h, w = g.shape
    targets: list[tuple[int, np.ndarray, int]] = []

    for outer_color in keyset:
        color_mask = (g == int(outer_color)) & (~pal_mask)
        for comp in components_mask(color_mask, connectivity=4):
            r0 = int(comp[:, 0].min())
            c0 = int(comp[:, 1].min())
            r1 = int(comp[:, 0].max())
            c1 = int(comp[:, 1].max())
            local_h = r1 - r0 + 1
            local_w = c1 - c0 + 1
            if local_h < 7 or local_w < 7:
                continue

            local_bg = (g[r0 : r1 + 1, c0 : c1 + 1] == bg) & (~pal_mask[r0 : r1 + 1, c0 : c1 + 1])
            for hole in components_mask(local_bg, connectivity=4):
                if _touches_local_boundary(hole, local_h, local_w):
                    continue

                touches_self = False
                touches_other = False
                abs_pts = hole.copy()
                abs_pts[:, 0] += r0
                abs_pts[:, 1] += c0

                for rr, cc in abs_pts:
                    for nr, nc in neighbors4(int(rr), int(cc)):
                        if 0 <= nr < h and 0 <= nc < w and not pal_mask[nr, nc]:
                            v = int(g[nr, nc])
                            if v == bg:
                                continue
                            if v == int(outer_color):
                                touches_self = True
                            else:
                                touches_other = True
                        if touches_self and touches_other:
                            break
                    if touches_self and touches_other:
                        break

                if touches_self and touches_other:
                    targets.append((int(outer_color), abs_pts, int(comp.shape[0])))

    return targets


def _fill_progressive_band(
    src: np.ndarray,
    out: np.ndarray,
    pts: np.ndarray,
    outer_color: int,
    layer_colors: tuple[int, int],
    pal_mask: np.ndarray,
) -> None:
    """Fill a target middle band by BFS distance from the outer frame."""
    h, w = src.shape
    region = {(int(r), int(c)) for r, c in pts}
    dist: dict[tuple[int, int], int] = {}
    queue: list[tuple[int, int]] = []

    for r, c in region:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc] and int(src[rr, cc]) == int(outer_color):
                dist[(r, c)] = 1
                queue.append((r, c))
                break

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        base = dist[(r, c)]
        for rr, cc in neighbors4(r, c):
            nbr = (int(rr), int(cc))
            if nbr in region and nbr not in dist:
                dist[nbr] = base + 1
                queue.append(nbr)

    outer_fill, inner_fill = map(int, layer_colors)
    for r, c in region:
        d = dist.get((r, c), 2)
        out[r, c] = outer_fill if d <= 1 else inner_fill


def _fill_nested_progressive_regions(
    src: np.ndarray,
    out: np.ndarray,
    bg: int,
    large_group: dict[int, tuple[int, int]],
    small_group: dict[int, tuple[int, int]],
    pal_mask: np.ndarray,
) -> None:
    """Fill only the nested middle bands, choosing the size group before applying the two-layer fill."""
    keyset = set(large_group) & set(small_group)
    targets = _collect_target_middle_regions(src, bg, keyset, pal_mask)
    if not targets:
        return

    areas = [area for _, _, area in targets]
    small_area = min(areas)
    large_area = max(areas)
    pivot = (small_area + large_area) / 2.0

    for outer_color, pts, area in targets:
        if large_area == small_area:
            group = large_group
        else:
            group = large_group if area > pivot else small_group
        _fill_progressive_band(src, out, pts, outer_color, group[outer_color], pal_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the middle band of nested sprites, using a doubled size-conditional legend and a two-step inward gradient.

    Input structure:
    - Background is the dominant outer-border color of the whole grid.
    - One 6xN or Nx6 palette block sits on an edge. Read from the edge inward, it contains:
      large keys / large outer-layer fills / large inner-layer fills / small keys / small outer-layer fills / small inner-layer fills.
    - Sprites are nested double frames: an outer frame, a blank middle band, an inner frame, and a blank center hole.

    Rule:
    - Locate the palette and recover two border-key -> (outer-band-color, inner-band-color) mappings:
      one for larger outer frames and one for smaller outer frames.
    - Among enclosed background regions, target only the middle band that touches both an outer frame and another non-background color.
    - Measure the enclosing outer frame's area to choose the large or small palette group.
    - Within that middle band, color cells by distance from the outer frame: the first layer gets the group's first fill color, and deeper cells get the group's second fill color.

    Key insight:
    - Three signals must be fused: nested structure tells which hole-like region is eligible, outer-frame area selects the active palette group, and distance from the outer frame selects which fill row inside that group is used.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    large_group, small_group, pal_mask = _palette_groups(g, bg)
    assert large_group and small_group, "main: both palette groups must be non-empty"

    out = g.copy()
    _fill_nested_progressive_regions(g, out, bg, large_group, small_group, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions: enlarged for a 6-line palette plus 3-4 large nested sprites.
_GRID_H_MIN, _GRID_H_MAX = 24, 30
_GRID_W_MIN, _GRID_W_MAX = 24, 30

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

# Nested sprite geometry: inset is fixed at 3, making the middle band exactly two layers thick.
_SMALL_SPECS = [(9, 9), (9, 10), (10, 9), (10, 10)]
_LARGE_SPECS = [(11, 11), (11, 12), (12, 11), (12, 12)]

# Generation
_MAX_RETRIES = 500


def _frame_on_canvas(canvas_h: int, canvas_w: int, r0: int, c0: int, box_h: int, box_w: int) -> np.ndarray:
    """Return a 1-cell-thick rectangular frame on a local canvas."""
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _make_nested_sprite(kind: str, rng: random.Random) -> dict[str, np.ndarray]:
    """Create a symmetric nested sprite whose middle band is exactly two layers thick."""
    h, w = rng.choice(_LARGE_SPECS if kind == "large" else _SMALL_SPECS)
    outer_mask = _frame_on_canvas(h, w, 0, 0, h, w)
    inner_mask = _frame_on_canvas(h, w, 3, 3, h - 6, w - 6)

    middle_mask = np.zeros((h, w), dtype=bool)
    middle_mask[1 : h - 1, 1 : w - 1] = True
    middle_mask[3 : h - 3, 3 : w - 3] = False

    layer1 = np.zeros((h, w), dtype=bool)
    for r, c in zip(*np.where(middle_mask), strict=False):
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and outer_mask[rr, cc]:
                layer1[int(r), int(c)] = True
                break
    layer2 = middle_mask & (~layer1)
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[4 : h - 4, 4 : w - 4] = True

    assert layer1.any() and layer2.any(), "_make_nested_sprite: middle band must have two layers"
    return {
        "outer": outer_mask,
        "inner": inner_mask,
        "middle": middle_mask,
        "layer1": layer1,
        "layer2": layer2,
        "center": center_mask,
        "body": outer_mask | inner_mask,
    }


def _deranged(values: list[int], rng: random.Random) -> list[int]:
    """Return a permutation with no fixed points."""
    if len(values) == 2:
        return [values[1], values[0]]
    order = list(values)
    for _ in range(40):
        rng.shuffle(order)
        if all(a != b for a, b in zip(order, values, strict=False)):
            return order
    return values[1:] + values[:1]


def _rotate(values: list[int], shift: int) -> list[int]:
    shift %= len(values)
    return values[shift:] + values[:shift]


def _choose_colors(
    rng: random.Random,
    n: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int], int]:
    """
    Pick repeated border keys and four structured fill rows.

    A small shared fill pool keeps color complexity low while still forcing size-conditional
    and layer-conditional behavior.
    """
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    keys = all_c[:n]
    fills_pool = all_c[n : 2 * n]
    distractor = all_c[2 * n]

    large_outer = fills_pool[:]
    rng.shuffle(large_outer)
    large_inner = _deranged(large_outer, rng)
    small_outer = _rotate(large_outer, rng.randint(1, n - 1))
    small_inner = _deranged(small_outer, rng)

    assert any(a != b for a, b in zip(large_outer, large_inner, strict=False))
    assert any(a != b for a, b in zip(small_outer, small_inner, strict=False))
    assert any(
        (lo, li) != (so, si)
        for lo, li, so, si in zip(large_outer, large_inner, small_outer, small_inner, strict=False)
    )
    return keys, large_outer, large_inner, small_outer, small_inner, distractor


def _build_palette(
    side: str,
    border_keys: list[int],
    large_outer: list[int],
    large_inner: list[int],
    small_outer: list[int],
    small_inner: list[int],
) -> np.ndarray:
    """Construct the 6xN or Nx6 palette block for the chosen edge."""
    edge_lines = [
        np.array(border_keys, dtype=int),
        np.array(large_outer, dtype=int),
        np.array(large_inner, dtype=int),
        np.array(border_keys, dtype=int),
        np.array(small_outer, dtype=int),
        np.array(small_inner, dtype=int),
    ]
    n = len(border_keys)

    if side == "top":
        pal = np.zeros((6, n), dtype=int)
        for i, line in enumerate(edge_lines):
            pal[i] = line
        return pal
    if side == "bottom":
        pal = np.zeros((6, n), dtype=int)
        for i, line in enumerate(edge_lines[::-1]):
            pal[i] = line
        return pal
    if side == "left":
        pal = np.zeros((n, 6), dtype=int)
        for i, line in enumerate(edge_lines):
            pal[:, i] = line
        return pal

    pal = np.zeros((n, 6), dtype=int)
    for i, line in enumerate(edge_lines[::-1]):
        pal[:, i] = line
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place the palette on the chosen edge and reserve a 1-cell moat."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _find_valid_positions(h: int, w: int, occ: np.ndarray, body_mask: np.ndarray) -> list[tuple[int, int]]:
    """All positions where the sprite fits, stays separated, and avoids the outer boundary."""
    sh, sw = body_mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _pick_inner_color(
    rng: random.Random,
    border_keys: list[int],
    outer_color: int,
    fill_pool: set[int],
    force_palette_overlap: bool,
) -> int:
    """
    Prefer another palette key so the unchanged center hole visibly cannot be explained
    by generic palette filling.
    """
    preferred = [c for c in border_keys if c != outer_color]
    if preferred and (force_palette_overlap or rng.random() < 0.85):
        return int(rng.choice(preferred))

    fallback = [c for c in range(1, 10) if c != outer_color and c not in fill_pool]
    return int(rng.choice(fallback))


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    sprite: dict[str, np.ndarray],
    outer_color: int,
    inner_color: int,
    fill_pair: tuple[int, int] | None,
) -> None:
    """Render one nested sprite. Fill colors are applied only when building the output state."""
    view = grid[r0 : r0 + sprite["body"].shape[0], c0 : c0 + sprite["body"].shape[1]]
    view[sprite["outer"]] = int(outer_color)
    view[sprite["inner"]] = int(inner_color)
    if fill_pair is not None:
        view[sprite["layer1"]] = int(fill_pair[0])
        view[sprite["layer2"]] = int(fill_pair[1])


def _sprite_roles(n: int, rng: random.Random) -> list[str]:
    """Mapped sprites always include both size groups; a third mapped sprite may repeat either group."""
    roles = ["large", "small"]
    if n == 3:
        roles.append(rng.choice(["large", "small"]))
    return roles


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    n = rng.randint(_PALETTE_N_MIN, _PALETTE_N_MAX)

    keys, large_outer, large_inner, small_outer, small_inner, distractor_color = _choose_colors(rng, n)
    large_group = {
        key: (a, b) for key, a, b in zip(keys, large_outer, large_inner, strict=False)
    }
    small_group = {
        key: (a, b) for key, a, b in zip(keys, small_outer, small_inner, strict=False)
    }
    pal = _build_palette(side, keys, large_outer, large_inner, small_outer, small_inner)
    fill_pool = set(large_outer) | set(large_inner) | set(small_outer) | set(small_inner)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos: list[tuple[dict[str, np.ndarray], int, int, tuple[int, int] | None, str]] = []
    overlap_used = False
    for key, role in zip(keys, _sprite_roles(n, rng), strict=False):
        sprite = _make_nested_sprite(role, rng)
        fill_pair = large_group[key] if role == "large" else small_group[key]
        inner_color = _pick_inner_color(rng, keys, key, fill_pool, force_palette_overlap=not overlap_used)
        if inner_color in keys and inner_color != key:
            overlap_used = True
        infos.append((sprite, key, inner_color, fill_pair, role))

    distractor_role = rng.choice(["large", "small"])
    distractor_sprite = _make_nested_sprite(distractor_role, rng)
    distractor_inner = _pick_inner_color(rng, keys, distractor_color, fill_pool, force_palette_overlap=False)
    infos.append((distractor_sprite, distractor_color, distractor_inner, None, distractor_role))

    if not overlap_used:
        return None

    infos.sort(key=lambda item: int(item[0]["body"].sum()), reverse=True)

    placed: list[tuple[int, int, dict[str, np.ndarray], tuple[int, int] | None]] = []
    for sprite, outer_color, inner_color, fill_pair, _role in infos:
        positions = _find_valid_positions(h, w, occ, sprite["body"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, sprite, outer_color, inner_color, fill_pair)
        placed.append((r0, c0, sprite, fill_pair))
        sh, sw = sprite["body"].shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, sprite, fill_pair in placed:
        if fill_pair is None:
            continue
        inp[r0 : r0 + sprite["body"].shape[0], c0 : c0 + sprite["body"].shape[1]][sprite["middle"]] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
