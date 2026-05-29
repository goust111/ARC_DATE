# Combo variant of abc82100 (index 43)
# combo_spec: D6/0+D9/0+D11/0
# Name: 全局共享模板+单一规则+多色模板

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior
#   - Source task: each source color owns its own local legend and its own teal
#     stencil; stamping is color-matched and monochrome.
#   - This combo collapses the "many rules" axis and keeps only one legend pair
#     and one global template. The visible template is now multicolor:
#       * teal cells are the mutable backbone
#       * non-teal accent cells are fixed paint that must be copied unchanged
#   - Example trace:
#       input may show one red->yellow legend, one large teal cluster with grey
#       and blue diagonal accents, and scattered isolated source pixels in red,
#       green, and orange. In the output every scattered source pixel produces the
#       same motif: the teal backbone turns yellow while the grey/blue accents stay
#       grey/blue. The original legend and global template disappear.
#   - Difference from the source task:
#       * source color no longer selects a rule
#       * there is only one template in the whole grid
#       * copied motifs are not monochrome
#
# b) Complexity calibration
#   - The matching logic is simpler, so the template itself is made richer:
#     4-10 teal cells plus 2-4 accent cells using 2-3 accent colors.
#   - Every generated instance uses at least two scattered source colors. This
#     makes the D9 change observable instead of silently degenerating into
#     "one source color repeated".
#   - There are no bystanders. Under D9 every template-external colored singleton
#     should behave as a source, so preserved bystanders would contradict the rule.
#
# c) Implicit assumption audit
#   - Original main() assumed "source color identifies the rule". That is removed:
#     solver now finds the unique legend pair near the lone teal component and then
#     treats every other non-template colored singleton as a source center.
#   - Original discovery stored only teal coordinates. That breaks under D6 because
#     accent cells would be lost. Solver now extracts every non-background cell
#     within Chebyshev 1 of the teal component as part of the template.
#   - Original preservation logic copied unrelated colored pixels through. That is
#     wrong here because template-external colored pixels are precisely the source
#     markers that must be consumed.
#   - New solver assumption: there is exactly one teal connected component and
#     exactly one colored 4-neighbor pair whose continuation reaches it. Generator
#     enforces this with a single legend and isolation halos.
#
# d) Placement legality
#   - Generator chooses one anchor cell inside the teal component, so the legend is
#     a clean source -> target -> teal chain.
#   - A full Chebyshev-1 halo is reserved around the template region, which makes
#     template extraction unambiguous.
#   - Scattered sources reserve their 4-neighbors so they stay isolated and cannot
#     form fake legends with each other.
#   - Accent cells are diagonal attachments and are rejected if they become
#     4-neighbors of the legend pair, which avoids bogus colored pairs near the
#     template.
#
# e) Visual observability
#   - Human solvers see one explicit example of what changes (teal) and what stays
#     fixed (accent colors).
#   - Repeating the same output motif from multiple different source colors makes it
#     visible that color identity of the scattered sources is irrelevant.
#
# f) Visual cleanliness
#   - Only one legend-template cluster exists, so the input is easy to parse.
#   - Isolation halos keep the global template visually separate from the scattered
#     sources, and the diagonal accents avoid cluttering the legend line.
#
# g) Change inferability
#   - No ordering by numeric color id is used.
#   - The only mutable part is marked structurally by teal, and the target color is
#     shown explicitly by the single adjacent legend target cell.
#   - Accent colors repeat across all stamped copies, giving enough support to infer
#     that they are copied literally rather than recolored.
#
# main() assumption audit after the change
#   - _find_global_template assumes the intended input has exactly one teal
#     component. Generator creates exactly one.
#   - _find_legend assumes the only colored 4-neighbor pair near that component is
#     the true legend pair. Generator isolates sources and keeps accents off the
#     legend's 4-neighborhood.
#   - _extract_template assumes every non-background cell within Chebyshev 1 of the
#     teal component belongs to the template, except the legend pair. Generator
#     keeps all other objects out of that halo.
#   - main() stamps sources in row-major order, so overlapping copies are
#     deterministic. Generator builds expected outputs with the same order.

PATTERN_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

GRID_DIM_MIN = 8
GRID_DIM_MAX = 20

TEAL_MAX_CAP = 10
TEAL_MIN_AREA_DIV = 80
TEAL_MAX_AREA_DIV = 24

ACCENT_MIN = 2
ACCENT_MAX = 4
ACCENT_COLOR_MIN = 2
ACCENT_COLOR_MAX = 3
TEMPLATE_GEN_RETRIES = 100

SOURCE_COLOR_MIN = 2
SOURCE_COLOR_MAX = 4
SCATTERED_MIN = 4
SCATTERED_MAX_CAP = 9
SCATTERED_AREA_DIV = 24

LEGEND_RETRIES = 500
SCATTER_RETRIES_PER_PIXEL = 60
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _find_global_template(grid: np.ndarray, pattern_color: int = PATTERN_COLOR) -> np.ndarray:
    patterns = components_mask(grid == pattern_color, connectivity=8)
    if not patterns:
        raise ValueError("expected one teal template")
    patterns.sort(key=len, reverse=True)
    return np.asarray(patterns[0], dtype=int)


def _find_legend(
    grid: np.ndarray,
    pattern: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
) -> dict:
    h, w = grid.shape
    candidates: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue
            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                ar = tr + dr
                ac = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                    continue
                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue
                key = ((int(r), int(c)), (int(tr), int(tc)))
                if key in seen:
                    continue
                seen.add(key)
                dist = _anchor_distance(pattern, (int(ar), int(ac)))
                if dist <= 1:
                    candidates.append(
                        {
                            "source_pos": (int(r), int(c)),
                            "target_pos": (int(tr), int(tc)),
                            "source_color": int(source_color),
                            "target_color": int(target_color),
                            "anchor": (int(ar), int(ac)),
                            "distance": int(dist),
                        }
                    )

    if not candidates:
        raise ValueError("failed to find legend pair")
    candidates.sort(key=lambda item: (item["distance"], item["source_pos"], item["target_pos"]))
    return candidates[0]


def _extract_template(
    grid: np.ndarray,
    pattern: np.ndarray,
    excluded: set[tuple[int, int]],
    bg: int,
    pattern_color: int = PATTERN_COLOR,
) -> list[tuple[int, int, int]]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    r0 = max(0, int(rows.min()) - 1)
    r1 = min(grid.shape[0] - 1, int(rows.max()) + 1)
    c0 = max(0, int(cols.min()) - 1)
    c1 = min(grid.shape[1] - 1, int(cols.max()) + 1)

    template_cells: list[tuple[int, int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in excluded:
                continue
            color = int(grid[r, c])
            if color == int(bg):
                continue
            if _anchor_distance(pattern, (int(r), int(c))) > 1:
                continue
            template_cells.append((int(r), int(c), int(color)))
    template_cells.sort()
    return template_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn one global multicolor template from a single legend and stamp it everywhere.

    Input structure:
    - Black background with exactly one teal connected component that acts as the
      shared template backbone.
    - Near that component there is one 4-neighbor source->target legend pair; the
      cell after the target points to the template anchor.
    - The template may also contain non-teal accent cells near the teal backbone,
      and every other isolated colored pixel outside the template cluster is a
      source marker, regardless of its color.

    Rule:
    - Extract the full multicolor template around the lone teal component.
    - For each non-template source marker, translate the template so the legend
      anchor lands on that marker.
    - When stamping, teal cells become the legend's target color while non-teal
      accent cells keep their original colors; erase the legend and the original
      global template from the output.

    Key insight: There is no longer a per-color lookup. The lone legend only tells
    you which color the teal part should become; the single shared template is then
    copied from every other isolated colored pixel, even if those source pixels have
    different colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    pattern = _find_global_template(g, pattern_color=PATTERN_COLOR)
    legend = _find_legend(g, pattern=pattern, bg=bg, pattern_color=PATTERN_COLOR)
    template_abs = _extract_template(
        g,
        pattern=pattern,
        excluded={legend["source_pos"], legend["target_pos"]},
        bg=bg,
        pattern_color=PATTERN_COLOR,
    )

    ar, ac = legend["anchor"]
    template = [(int(r - ar), int(c - ac), int(color)) for r, c, color in template_abs]
    template_pixels = {(int(r), int(c)) for r, c, _ in template_abs}
    legend_pixels = {legend["source_pos"], legend["target_pos"]}

    source_positions = [
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in legend_pixels and (int(r), int(c)) not in template_pixels
    ]

    out = np.full_like(g, int(bg))
    for sr, sc in source_positions:
        for rr, cc, color in template:
            nr = int(sr + rr)
            nc = int(sc + cc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = legend["target_color"] if color == PATTERN_COLOR else color
    return out


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS8:
            reserved.add((int(r + dr), int(c + dc)))


def _make_teal_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                pos = (r + dr, c + dc)
                if pos not in coords:
                    frontier.add(pos)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _make_multicolor_template(
    min_teal: int,
    max_teal: int,
    target_color: int,
    accent_palette: list[int],
) -> list[tuple[int, int, int]]:
    for _ in range(TEMPLATE_GEN_RETRIES):
        teal_coords, anchor = _make_teal_stencil(min_teal, max_teal)
        teal_set = set(teal_coords)
        diagonal_candidates: set[tuple[int, int]] = set()
        for r, c in teal_coords:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    pos = (r + dr, c + dc)
                    if pos not in teal_set:
                        diagonal_candidates.add(pos)
        if len(diagonal_candidates) < ACCENT_MIN:
            continue

        accent_count = min(random.randint(ACCENT_MIN, ACCENT_MAX), len(diagonal_candidates))
        needed_colors = min(len(accent_palette), random.randint(ACCENT_COLOR_MIN, ACCENT_COLOR_MAX))
        color_cycle = accent_palette[:]
        random.shuffle(color_cycle)
        color_cycle = color_cycle[:needed_colors]
        if accent_count < len(color_cycle):
            accent_count = len(color_cycle)

        choices = list(diagonal_candidates)
        random.shuffle(choices)
        accents: list[tuple[int, int, int]] = []
        used_positions: list[tuple[int, int]] = []
        for pos in choices:
            if any(_manhattan(pos, other) == 1 for other in used_positions):
                continue
            color = color_cycle[len(accents) % len(color_cycle)]
            accents.append((int(pos[0]), int(pos[1]), int(color)))
            used_positions.append((int(pos[0]), int(pos[1])))
            if len(accents) >= accent_count:
                break
        if len(accents) < ACCENT_MIN:
            continue
        accent_colors = {color for _, _, color in accents}
        if len(accent_colors) < ACCENT_COLOR_MIN:
            continue

        ar, ac = anchor
        rel_template = [(int(r - ar), int(c - ac), PATTERN_COLOR) for r, c in teal_coords]
        rel_template.extend((int(r - ar), int(c - ac), int(color)) for r, c, color in accents)
        rel_template.sort()
        if any(color == target_color for _, _, color in rel_template if color != PATTERN_COLOR):
            continue
        return rel_template
    raise RuntimeError("failed to build multicolor template")


def _absolute_template_cells(
    template: list[tuple[int, int, int]],
    anchor: tuple[int, int],
) -> tuple[set[tuple[int, int]], list[tuple[int, int, int]], set[tuple[int, int]]]:
    ar, ac = anchor
    teal_cells: set[tuple[int, int]] = set()
    accent_cells: list[tuple[int, int, int]] = []
    all_cells: set[tuple[int, int]] = set()
    for rr, cc, color in template:
        nr = int(ar + rr)
        nc = int(ac + cc)
        all_cells.add((nr, nc))
        if color == PATTERN_COLOR:
            teal_cells.add((nr, nc))
        else:
            accent_cells.append((nr, nc, int(color)))
    return teal_cells, accent_cells, all_cells


def _build_problem(area: int) -> dict:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    target_color = colors[0]
    remaining = colors[1:]

    accent_palette_size = random.randint(
        ACCENT_COLOR_MIN,
        min(ACCENT_COLOR_MAX, len(remaining) - SOURCE_COLOR_MIN),
    )
    accent_palette = remaining[:accent_palette_size]
    source_palette = remaining[accent_palette_size:]
    random.shuffle(source_palette)
    source_color_count = random.randint(SOURCE_COLOR_MIN, min(SOURCE_COLOR_MAX, len(source_palette)))
    active_source_palette = source_palette[:source_color_count]
    legend_source_color = random.choice(active_source_palette)

    max_teal = min(TEAL_MAX_CAP, max(4, area // TEAL_MAX_AREA_DIV))
    min_teal = min(max(4, area // TEAL_MIN_AREA_DIV), max_teal)
    template = _make_multicolor_template(min_teal, max_teal, target_color, accent_palette)

    scatter_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    scatter_count = random.randint(SCATTERED_MIN, scatter_max)
    return {
        "target_color": int(target_color),
        "legend_source_color": int(legend_source_color),
        "template": template,
        "source_palette": [int(c) for c in active_source_palette],
        "scatter_count": int(scatter_count),
    }


def _place_legend(problem: dict, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    template = problem["template"]
    for _ in range(LEGEND_RETRIES):
        dr, dc = random.choice(_DIRS4)
        sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
        tr, tc = sr + dr, sc + dc
        ar, ac = tr + dr, tc + dc
        if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
            continue

        teal_cells, accent_cells, all_template_cells = _absolute_template_cells(template, (ar, ac))
        if (ar, ac) not in teal_cells:
            continue
        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_template_cells):
            continue
        if any(
            _manhattan((r, c), (sr, sc)) == 1 or _manhattan((r, c), (tr, tc)) == 1
            for r, c, _ in accent_cells
        ):
            continue
        if any(_chebyshev((sr, sc), cell) <= 1 for cell in teal_cells):
            continue

        all_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | all_template_cells
        if all_cells & occupied or all_cells & reserved:
            continue

        inp[sr, sc] = problem["legend_source_color"]
        inp[tr, tc] = problem["target_color"]
        for r, c in teal_cells:
            inp[r, c] = PATTERN_COLOR
        for r, c, color in accent_cells:
            inp[r, c] = int(color)

        occupied.update(all_cells)
        _reserve_4neighbors(reserved, {(sr, sc), (tr, tc)})
        _reserve_8neighbors(reserved, all_template_cells)
        problem["legend_source_pos"] = (int(sr), int(sc))
        problem["legend_target_pos"] = (int(tr), int(tc))
        problem["legend_anchor"] = (int(ar), int(ac))
        problem["template_abs_cells"] = sorted(all_template_cells)
        return True
    return False


def _place_scattered(problem: dict, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    positions: list[tuple[int, int, int]] = []
    palette = problem["source_palette"]
    required_colors = palette[: min(len(palette), SOURCE_COLOR_MIN)]
    remaining_budget = problem["scatter_count"] * SCATTER_RETRIES_PER_PIXEL

    for i in range(problem["scatter_count"]):
        color = required_colors[i] if i < len(required_colors) else random.choice(palette)
        placed = False
        for _ in range(remaining_budget):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            inp[r, c] = int(color)
            positions.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            placed = True
            break
        if not placed:
            return False

    if len({color for _, _, color in positions}) < SOURCE_COLOR_MIN:
        return False
    problem["scattered"] = sorted(positions)
    return True


def _build_output(problem: dict, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for sr, sc, _ in sorted(problem["scattered"]):
        for rr, cc, color in problem["template"]:
            nr = int(sr + rr)
            nc = int(sc + cc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = problem["target_color"] if color == PATTERN_COLOR else color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)
    w = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)
    problem = _build_problem(h * w)

    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_legend(problem, inp, occupied, reserved):
        return None
    if not _place_scattered(problem, inp, occupied, reserved):
        return None

    expected = _build_output(problem, h, w)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
