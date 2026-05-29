# Combo variant of de809cff (index 38)
# combo_spec: D1/0+D12/1+D18/0
# Name: 十字形补丁+形状编码方向+冲突teal标记

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the fused combo variant
# -------------------------------------------
# a) Concrete behavior trace
# - Cleanup stays the same as the source task: isolated foreground specks in black
#   background are removed, and one-cell boundary contamination is reassigned to
#   its surrounding region before any dent reasoning happens.
# - Dent interpretation changes in two coupled ways:
#   * A 1-cell dent becomes an opposite-color orthogonal cross.
#   * A 2-cell horizontal dent becomes a 3-cell horizontal stripe anchored by the
#     leftmost dent cell; a 2-cell vertical dent becomes a 3-cell vertical stripe
#     anchored by the topmost dent cell.
# - Composition also changes: patches are not written sequentially anymore. Every
#   patch contributes coverage simultaneously; any pixel covered by 2+ patches is
#   teal, even when the overlapping patches want the same opposite color.
# - Small trace:
#   * Original task: two black cells in a horizontal pair would each produce their
#     own 3x3 squares, creating a bulky blob; overlap color would just depend on
#     write order.
#   * This combo: the same two cells are grouped first, produce one thin row mark,
#     and if that row meets a nearby single-dent cross, the shared pixel turns teal.
# - Difference summary:
#   * shape cue: square -> cross / horizontal stripe / vertical stripe
#   * grouping cue: 2-cell dents are solved as one object, not as two singles
#   * collision cue: extra teal can appear away from the original black cells
#
# b) Complexity calibration
# - Only three visible patch classes exist, which is enough for humans to infer the
#   geometry mapping without overloading the task with many cases.
# - The generator guarantees both pair orientations and ordinary singles in the same
#   sample, so the shape-conditioned rule has repeated support.
# - A few intentionally close groups create collision teal, but most other groups
#   stay separated; this keeps the output readable instead of turning it into dense
#   teal clutter.
#
# c) Implicit-assumption audit of main()
# - `_clean_foreground(...)`: unchanged assumption that non-dent corruption is only
#   1-cell noise. The generator preserves that contract.
# - `_dent_groups(...)`: replaces the source task's "every black cell is independent"
#   assumption. Adjacent black cells must now be grouped and classified by shape.
# - `_patch_cells(...)`: the source task assumed every dent writes a 3x3 box. That
#   is invalid here; patch geometry depends on the grouped dent shape.
# - `_compose_output(...)`: sequential overwriting is no longer valid. We must count
#   coverage first, then resolve `cover==1` vs `cover>=2`.
# - Final teal overwrite is still required, because dent cells are always teal even
#   if a conflict also happens there.
#
# d) Placement legality
# - Distinct dent groups may be close enough to make patch overlap, but their black
#   cells are never 4-neighbor adjacent across groups; otherwise the solver would see
#   one larger connected dent component instead of separate objects.
# - The generator verifies that the blackened input recovers exactly the intended
#   dent groups before adding noise, which catches illegal close placements.
#
# e) Visual observability
# - Humans can read the rule from visible structure only:
#   * one black cell -> plus-like mark
#   * horizontal black pair -> horizontal mark
#   * vertical black pair -> vertical mark
#   * non-center teal pixels -> collision marker
# - No ordering by color id or hidden metadata is used.
#
# f) Visual cleanliness
# - Collision motifs are local and sparse. Most groups keep at least the original
#   spacing, so the grid still looks like two clean regions with a modest number of
#   transformed dents rather than a globally entangled overlay.
#
# g) Change inferability
# - Color behavior stays simple: a dent patch still uses the opposite foreground
#   color, while teal has two visible roles: original dent cells and patch conflicts.
# - Because pair dents are present in both orientations and collisions produce extra
#   teal outside dent centers, both parts of the combo are directly observable.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: every transformable multi-cell dent is a 4-connected component
#   of size exactly 2, oriented horizontally or vertically. Generator guarantee: it
#   creates only singles and orthogonal 2-cell pairs.
# - Solver assumption: a pair's stripe is anchored by the top-left-most dent cell.
#   Generator guarantee: horizontal pairs have a valid host cell on the left and
#   vertical pairs have a valid host cell above, so the 3-cell stripe exists.
# - Solver assumption: close groups may overlap in patch space, but the underlying
#   dent grouping remains recoverable. Generator guarantee: it validates exact group
#   recovery on the blackened clean grid before accepting a sample.


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(colors: tuple[int, int], host: int) -> int:
    return int(colors[1]) if int(host) == int(colors[0]) else int(colors[0])


def _has_same_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                if bool(np.all(g[rr : rr + 2, cc : cc + 2] == int(color))):
                    return True
    return False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting dents.

    Two local sweeps are enough for the intended corruption pattern.
    """
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = bg
                    continue

                sub = cur[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
                own = int(np.sum(sub == v))
                other = b if v == a else a
                other_count = int(np.sum(sub == other))
                bg_count = int(np.sum(sub == bg))

                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = v
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = other
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = bg
                elif own <= 2:
                    nxt[r, c] = bg
                else:
                    nxt[r, c] = v
        cur = nxt

    return cur


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _dent_host(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    r: int,
    c: int,
) -> int | None:
    if int(original[r, c]) != int(bg):
        return None
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    host = max(colors, key=lambda color: counts[int(color)])
    other = _other_color(colors, int(host))
    if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
        return int(host)
    return None


def _group_shape(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) == 1:
        return "single"
    if len(cells) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(cells)
    if r0 == r1 and abs(c0 - c1) == 1:
        return "horizontal"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "vertical"
    return None


def _dent_groups(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
) -> list[tuple[list[tuple[int, int]], int, str]]:
    candidates: dict[tuple[int, int], int] = {}
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        host = _dent_host(original, clean, colors, bg, rr, cc)
        if host is not None:
            candidates[(rr, cc)] = int(host)

    groups: list[tuple[list[tuple[int, int]], int, str]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(candidates):
        if start in seen:
            continue
        stack = [start]
        comp: list[tuple[int, int]] = []
        hosts: set[int] = set()
        while stack:
            cell = stack.pop()
            if cell in seen:
                continue
            seen.add(cell)
            comp.append(cell)
            hosts.add(int(candidates[cell]))
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in candidates and (nr, nc) not in seen:
                    stack.append((nr, nc))

        shape = _group_shape(comp)
        if shape is None or len(hosts) != 1:
            continue
        groups.append((sorted(comp), next(iter(hosts)), shape))
    return groups


def _patch_cells(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    dent_shape: str,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if dent_shape == "single":
        r, c = cells[0]
        out: list[tuple[int, int]] = []
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
        return out
    if dent_shape == "horizontal":
        r = int(cells[0][0])
        c0 = min(c for _r, c in cells)
        return [(r, cc) for cc in range(c0 - 1, c0 + 2) if 0 <= cc < w]
    if dent_shape == "vertical":
        c = int(cells[0][1])
        r0 = min(r for r, _c in cells)
        return [(rr, c) for rr in range(r0 - 1, r0 + 2) if 0 <= rr < h]
    raise ValueError(f"unknown dent shape: {dent_shape}")


def _compose_output(
    clean: np.ndarray,
    groups: list[tuple[list[tuple[int, int]], int, str]],
    colors: tuple[int, int],
) -> np.ndarray:
    out = clean.copy()
    h, w = clean.shape
    cover = np.zeros((h, w), dtype=int)
    unique_color = np.full((h, w), -1, dtype=int)

    for cells, host, dent_shape in groups:
        stamp = _other_color(colors, int(host))
        for rr, cc in _patch_cells(clean.shape, cells, dent_shape):
            if cover[rr, cc] == 0:
                unique_color[rr, cc] = int(stamp)
            cover[rr, cc] += 1

    lone = cover == 1
    conflict = cover >= 2
    out[lone] = unique_color[lone]
    out[conflict] = Color.TEAL

    for cells, _host, _shape in groups:
        for r, c in cells:
            out[r, c] = Color.TEAL

    return out


def _compose_without_conflicts(
    clean: np.ndarray,
    groups: list[tuple[list[tuple[int, int]], int, str]],
    colors: tuple[int, int],
) -> np.ndarray:
    out = clean.copy()
    for cells, host, dent_shape in groups:
        stamp = _other_color(colors, int(host))
        for rr, cc in _patch_cells(clean.shape, cells, dent_shape):
            out[rr, cc] = int(stamp)
    for cells, _host, _shape in groups:
        for r, c in cells:
            out[r, c] = Color.TEAL
    return out


def _compose_cellwise_cross(
    clean: np.ndarray,
    groups: list[tuple[list[tuple[int, int]], int, str]],
    colors: tuple[int, int],
) -> np.ndarray:
    pseudo_groups: list[tuple[list[tuple[int, int]], int, str]] = []
    for cells, host, _shape in groups:
        for cell in cells:
            pseudo_groups.append(([cell], int(host), "single"))
    return _compose_output(clean, pseudo_groups, colors)


def _variant_metrics(
    clean: np.ndarray,
    groups: list[tuple[list[tuple[int, int]], int, str]],
    colors: tuple[int, int],
) -> dict[str, int]:
    out = _compose_output(clean, groups, colors)
    no_conflict = _compose_without_conflicts(clean, groups, colors)
    cellwise_cross = _compose_cellwise_cross(clean, groups, colors)

    center_mask = np.zeros(clean.shape, dtype=bool)
    h_pairs = 0
    v_pairs = 0
    singles = 0
    cover = np.zeros(clean.shape, dtype=int)
    for cells, host, dent_shape in groups:
        if dent_shape == "single":
            singles += 1
        elif dent_shape == "horizontal":
            h_pairs += 1
        elif dent_shape == "vertical":
            v_pairs += 1
        for r, c in cells:
            center_mask[r, c] = True
        for rr, cc in _patch_cells(clean.shape, cells, dent_shape):
            cover[rr, cc] += 1

    conflict = cover >= 2
    return {
        "groups": len(groups),
        "singles": singles,
        "horizontal_pairs": h_pairs,
        "vertical_pairs": v_pairs,
        "conflict_pixels": int(np.sum(conflict)),
        "extra_conflict_teal": int(np.sum(conflict & ~center_mask)),
        "shape_delta_vs_cellwise_cross": int(np.sum(out != cellwise_cross)),
        "conflict_delta_vs_overwrite": int(np.sum(out != no_conflict)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert black dents in two colored regions into shape-coded opposite-color marks.

    Input structure:
    - Black background with two large foreground regions in different colors.
    - The regions include black dents plus sparse one-cell foreground noise either
      in the background or along the boundary between the two regions.

    Rule:
    - First clean the foreground regions locally so isolated specks return to black
      and one-cell boundary contamination flips back to its host region.
    - Then group transformable black dents by 4-connectivity: a 1-cell dent becomes
      an opposite-color cross, a 2-cell horizontal dent becomes a 3-cell horizontal
      stripe, and a 2-cell vertical dent becomes a 3-cell vertical stripe.
    - All dent patches are combined simultaneously: pixels covered by exactly one
      patch take that patch color, while pixels covered by two or more patches turn
      teal (8). Every original dent cell is also teal (8).

    Key insight:
    - The output must be inferred from two visible cues at once: dent geometry decides
      patch shape, and extra teal away from the black holes marks where multiple dent
      patches collide.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    groups = _dent_groups(g, clean, colors, bg)
    return _compose_output(clean, groups, colors)


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
PAIR_GROUP_COUNT_MIN, PAIR_GROUP_COUNT_MAX = 2, 4
MIN_COLLISION_EXTRA_TEAL = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 600
MAX_SELECTION_ATTEMPTS = 60


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice([2, 3])
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    chosen_corners = corners[:n_rects]

    if n_rects == 2:
        rect_colors = [color_a, color_b]
        rng.shuffle(rect_colors)
    else:
        first = rng.choice([color_a, color_b])
        mid = color_b if first == color_a else color_a
        rect_colors = [first, mid, first]

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(0, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    grid = np.zeros((N, N), dtype=int)
    for corner, color in zip(chosen_corners, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = color

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        from scipy.ndimage import label

        _, n_components = label(grid == rect_colors[0])
        if n_components < 2:
            return None
    return grid


def _is_safe_dent_cell(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = color
                if int(np.sum(block == color)) == 4:
                    return True
    return False


def _enumerate_single_candidates(clean: np.ndarray, host: int, other: int) -> list[tuple[list[tuple[int, int]], int, str]]:
    h, w = clean.shape
    out: list[tuple[list[tuple[int, int]], int, str]] = []
    for r in range(h):
        for c in range(w):
            if _is_safe_dent_cell(clean, r, c, host, other):
                out.append(([(r, c)], int(host), "single"))
    return out


def _enumerate_pair_candidates(
    clean: np.ndarray,
    host: int,
    other: int,
    dent_shape: str,
) -> list[tuple[list[tuple[int, int]], int, str]]:
    h, w = clean.shape
    out: list[tuple[list[tuple[int, int]], int, str]] = []
    if dent_shape == "horizontal":
        for r in range(h):
            for c in range(1, w - 1):
                cells = [(r, c), (r, c + 1)]
                cap = (r, c - 1)
                if any(int(clean[rr, cc]) != int(host) for rr, cc in cells + [cap]):
                    continue
                if all(_is_safe_dent_cell(clean, rr, cc, host, other) for rr, cc in cells):
                    out.append((cells, int(host), dent_shape))
        return out
    if dent_shape == "vertical":
        for r in range(1, h - 1):
            for c in range(w):
                cells = [(r, c), (r + 1, c)]
                cap = (r - 1, c)
                if any(int(clean[rr, cc]) != int(host) for rr, cc in cells + [cap]):
                    continue
                if all(_is_safe_dent_cell(clean, rr, cc, host, other) for rr, cc in cells):
                    out.append((cells, int(host), dent_shape))
        return out
    raise ValueError(f"unknown dent shape: {dent_shape}")


def _group_signature(group: tuple[list[tuple[int, int]], int, str]) -> tuple[int, str, tuple[tuple[int, int], ...]]:
    cells, host, dent_shape = group
    return int(host), str(dent_shape), tuple(sorted((int(r), int(c)) for r, c in cells))


def _group_set_signature(groups: list[tuple[list[tuple[int, int]], int, str]]) -> tuple[tuple[int, str, tuple[tuple[int, int], ...]], ...]:
    return tuple(sorted(_group_signature(group) for group in groups))


def _render_input_from_groups(clean: np.ndarray, groups: list[tuple[list[tuple[int, int]], int, str]]) -> np.ndarray:
    inp = clean.copy()
    for cells, _host, _shape in groups:
        for r, c in cells:
            inp[r, c] = Color.BLACK
    return inp


def _selection_recoverable(
    clean: np.ndarray,
    groups: list[tuple[list[tuple[int, int]], int, str]],
    colors: tuple[int, int],
) -> bool:
    inp = _render_input_from_groups(clean, groups)
    recovered = _dent_groups(inp, clean, colors, Color.BLACK)
    return _group_set_signature(recovered) == _group_set_signature(groups)


def _cells_4disjoint(
    cells_a: list[tuple[int, int]],
    cells_b: list[tuple[int, int]],
) -> bool:
    for r0, c0 in cells_a:
        for r1, c1 in cells_b:
            if (r0, c0) == (r1, c1):
                return False
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _group_min_cheb(
    group_a: tuple[list[tuple[int, int]], int, str],
    group_b: tuple[list[tuple[int, int]], int, str],
) -> int:
    cells_a, _host_a, _shape_a = group_a
    cells_b, _host_b, _shape_b = group_b
    return min(
        max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1)))
        for r0, c0 in cells_a
        for r1, c1 in cells_b
    )


def _patch_overlap_has_extra_conflict(
    shape: tuple[int, int],
    group_a: tuple[list[tuple[int, int]], int, str],
    group_b: tuple[list[tuple[int, int]], int, str],
) -> bool:
    cells_a, _host_a, dent_shape_a = group_a
    cells_b, _host_b, dent_shape_b = group_b
    overlap = set(_patch_cells(shape, cells_a, dent_shape_a)) & set(_patch_cells(shape, cells_b, dent_shape_b))
    if not overlap:
        return False
    dent_cells = set(cells_a) | set(cells_b)
    return any(cell not in dent_cells for cell in overlap)


def _find_collision_pair(
    clean: np.ndarray,
    colors: tuple[int, int],
    singles: list[tuple[list[tuple[int, int]], int, str]],
    others: list[tuple[list[tuple[int, int]], int, str]],
    selected: list[tuple[list[tuple[int, int]], int, str]],
    remaining_by_color: dict[int, int],
    rng: random.Random,
) -> list[tuple[list[tuple[int, int]], int, str]] | None:
    singles_pool = singles[:]
    others_pool = others[:]
    rng.shuffle(singles_pool)
    rng.shuffle(others_pool)
    for other_group in others_pool:
        other_host = int(other_group[1])
        if remaining_by_color[other_host] <= 0:
            continue
        if any(
            not _cells_4disjoint(other_group[0], existing[0])
            or _group_min_cheb(other_group, existing) < DENT_PATCH_SEPARATION
            for existing in selected
        ):
            continue
        for single_group in singles_pool:
            needed = _Counter([int(other_group[1]), int(single_group[1])])
            if any(remaining_by_color[color] < count for color, count in needed.items()):
                continue
            if _group_signature(single_group) == _group_signature(other_group):
                continue
            if not _cells_4disjoint(single_group[0], other_group[0]):
                continue
            if not _patch_overlap_has_extra_conflict(clean.shape, single_group, other_group):
                continue
            if any(
                not _cells_4disjoint(single_group[0], existing[0])
                or _group_min_cheb(single_group, existing) < DENT_PATCH_SEPARATION
                for existing in selected
            ):
                continue
            tentative = selected + [single_group, other_group]
            if _selection_recoverable(clean, tentative, colors):
                return [single_group, other_group]
    return None


def _pick_far_group(
    clean: np.ndarray,
    colors: tuple[int, int],
    pool: list[tuple[list[tuple[int, int]], int, str]],
    selected: list[tuple[list[tuple[int, int]], int, str]],
    remaining_by_color: dict[int, int],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, str] | None:
    shuffled = pool[:]
    rng.shuffle(shuffled)
    for group in shuffled:
        host = int(group[1])
        if remaining_by_color[host] <= 0:
            continue
        if any(
            not _cells_4disjoint(group[0], existing[0])
            or _group_min_cheb(group, existing) < DENT_PATCH_SEPARATION
            for existing in selected
        ):
            continue
        if _selection_recoverable(clean, selected + [group], colors):
            return group
    return None


def _plan_shape_counts(n_total: int, rng: random.Random) -> dict[str, int]:
    pair_total = rng.randint(PAIR_GROUP_COUNT_MIN, min(PAIR_GROUP_COUNT_MAX, n_total - 4))
    horizontal = rng.randint(1, pair_total - 1)
    vertical = pair_total - horizontal
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "single": n_total - pair_total,
    }


def _select_groups(
    clean: np.ndarray,
    colors: tuple[int, int],
    candidates: dict[str, list[tuple[list[tuple[int, int]], int, str]]],
    n_a: int,
    n_b: int,
    shape_counts: dict[str, int],
    rng: random.Random,
) -> list[tuple[list[tuple[int, int]], int, str]] | None:
    remaining_by_color = {int(colors[0]): int(n_a), int(colors[1]): int(n_b)}
    remaining_by_shape = shape_counts.copy()
    selected: list[tuple[list[tuple[int, int]], int, str]] = []

    horizontal_seed = _find_collision_pair(
        clean,
        colors,
        candidates["single"],
        candidates["horizontal"],
        selected,
        remaining_by_color,
        rng,
    )
    if horizontal_seed is None:
        return None
    for group in horizontal_seed:
        selected.append(group)
        remaining_by_color[int(group[1])] -= 1
        remaining_by_shape[str(group[2])] -= 1

    vertical_seed = _find_collision_pair(
        clean,
        colors,
        candidates["single"],
        candidates["vertical"],
        selected,
        remaining_by_color,
        rng,
    )
    if vertical_seed is None:
        return None
    for group in vertical_seed:
        selected.append(group)
        remaining_by_color[int(group[1])] -= 1
        remaining_by_shape[str(group[2])] -= 1

    fill_order: list[str] = []
    for shape_name in ("horizontal", "vertical", "single"):
        fill_order.extend([shape_name] * remaining_by_shape[shape_name])
    rng.shuffle(fill_order)
    fill_order.sort(key=lambda shape_name: 0 if shape_name != "single" else 1)

    for shape_name in fill_order:
        group = _pick_far_group(clean, colors, candidates[shape_name], selected, remaining_by_color, rng)
        if group is None:
            return None
        selected.append(group)
        remaining_by_color[int(group[1])] -= 1
        remaining_by_shape[shape_name] -= 1

    if any(value != 0 for value in remaining_by_color.values()):
        return None
    if any(value != 0 for value in remaining_by_shape.values()):
        return None
    if not _selection_recoverable(clean, selected, colors):
        return None
    return selected


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = (
            _Counter(nearby_fg).most_common(1)[0][0]
            if nearby_fg
            else rng.choice([color_a, color_b])
        )
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = other
            own_n = int(np.sum(nsub == host))
            other_n = int(np.sum(nsub == other))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (color_a, color_b):
                continue
            other = color_b if v == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((r, c, v, other))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == host))
        other_count = int(np.sum(sub_now == other))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = other
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    colors = (int(color_a), int(color_b))
    candidates = {
        "single": _enumerate_single_candidates(clean, color_a, color_b)
        + _enumerate_single_candidates(clean, color_b, color_a),
        "horizontal": _enumerate_pair_candidates(clean, color_a, color_b, "horizontal")
        + _enumerate_pair_candidates(clean, color_b, color_a, "horizontal"),
        "vertical": _enumerate_pair_candidates(clean, color_a, color_b, "vertical")
        + _enumerate_pair_candidates(clean, color_b, color_a, "vertical"),
    }
    if not candidates["horizontal"] or not candidates["vertical"]:
        return None

    groups: list[tuple[list[tuple[int, int]], int, str]] | None = None
    accepted_groups: list[tuple[list[tuple[int, int]], int, str]] | None = None
    for _ in range(MAX_SELECTION_ATTEMPTS):
        shape_counts = _plan_shape_counts(n_total, rng)
        if shape_counts["single"] < 2:
            continue
        groups = _select_groups(clean, colors, candidates, n_a, n_b, shape_counts, rng)
        if groups is None:
            continue
        metrics = _variant_metrics(clean, groups, colors)
        if metrics["horizontal_pairs"] < 1 or metrics["vertical_pairs"] < 1 or metrics["singles"] < 2:
            continue
        if metrics["extra_conflict_teal"] < MIN_COLLISION_EXTRA_TEAL:
            continue
        if metrics["shape_delta_vs_cellwise_cross"] <= 0:
            continue
        if metrics["conflict_delta_vs_overwrite"] <= 0:
            continue
        accepted_groups = groups
        break
    if accepted_groups is None:
        return None
    groups = accepted_groups

    output = _compose_output(clean, groups, colors)

    inp = _render_input_from_groups(clean, groups)
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
