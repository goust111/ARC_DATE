# Variant of abc82100
# Dimension: D11 — teal模板的连通性和数量 (meta)
# Change: 每个规则示例附近有一个teal连通分量作为该规则的模板... → 模板由多个不相连的teal片段组成

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


# Design analysis for the D11 variant
# ==================================
# a) Concrete behavior and trace
#    - Original task: one rule legend contains `source -> target -> teal component`.
#      The solver can treat a single 8-connected teal component as the whole stencil.
#    - Variant: the legend still starts with the same adjacent color pair, but the
#      teal stencil is split into 2-4 separate 8-connected fragments packed inside a
#      small 5x5 neighborhood around the anchor. One fragment contains the anchor
#      cell itself; the others are detached by at least one empty cell in every
#      8-neighbor direction, so the union is visibly fragmented.
#    - Example trace:
#      Old logic sees an anchor-side teal cluster like `{(0,0),(0,1)}` and stamps
#      only that cluster. New logic must also collect detached cells such as
#      `{(2,0)}` and `{(-1,2)}` from the same local window, then stamp the union.
#      The change happens exactly inside rule discovery: "pick one component" becomes
#      "group all local teal components belonging to the same legend".
# b) Complexity calibration
#    - Every rule uses at least 2 fragments and at least 4 teal pixels total, so the
#      fragmentation is not a rare edge case and is visible in small training sets.
#    - Fragment count is capped at 4 and all pieces stay in a 5x5 window. This keeps
#      the grouping task nontrivial without turning the legend area into noise.
# c) Implicit assumption audit of the original main()
#    - `patterns = components_mask(...)`: harmless, but the old code assumed each
#      component was directly usable as a full template. That is now false.
#    - Candidate selection by smallest nearby component depended on the old
#      invariant "one component per rule". In the variant, choosing one component
#      would silently erase the detached pieces from every stamp.
#    - The stamping loop itself is still valid once `rule["pattern"]` is redefined
#      as the union of all local fragments.
# d) Placement legality
#    - Generator keeps every legend's template inside a Chebyshev-radius-2 window
#      around its anchor and reserves that window from other objects.
#    - Fragments within one legend are pairwise 8-disconnected, but the whole union
#      remains inside the same local zone so the solver can group them reliably.
#    - Scattered source pixels and bystanders are kept away from legend zones and
#      from each other with the same 4-neighbor safety rule as the original task.
# e) Visual observability
#    - A human now sees "one color pair, then several nearby teal islands" rather
#      than "one color pair, then one teal blob". Across 2-4 examples this is a
#      stable, repeated cue for grouping the detached pieces into one stencil.
# f) Visual cleanliness
#    - The 5x5 local window and low fragment count keep legends compact.
#    - Window reservation avoids teal from different legends blending together.
# g) Color inference
#    - Color behavior is unchanged from the original task: the legend's source and
#      target pixels still define the recoloring pair. No hidden color ordering or
#      color-id arithmetic is introduced.
# h) New solver assumptions and generator guarantees
#    - Solver assumption: every valid legend has an anchor cell occupied by teal and
#      at least two teal components inside the radius-2 window.
#    - Generator guarantee: anchor is always part of one fragment, every rule has
#      2-4 fragments, and no other teal enters that local window.


LOCAL_RADIUS = 2
LOCAL_DIAMETER = 2 * LOCAL_RADIUS + 1


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _max_anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return max(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _collect_local_pattern(
    patterns: list[np.ndarray], anchor: tuple[int, int], radius: int
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    component_ids: list[int] = []
    coords: set[tuple[int, int]] = set()
    anchor_is_teal = False

    for i, pattern in enumerate(patterns):
        if _max_anchor_distance(pattern, anchor) > radius:
            continue
        component_ids.append(int(i))
        for r, c in pattern:
            cell = (int(r), int(c))
            coords.add(cell)
            if cell == anchor:
                anchor_is_teal = True

    if not anchor_is_teal or len(component_ids) < 2:
        return None

    merged = np.asarray(sorted(coords), dtype=int)
    return merged, tuple(component_ids)


def _discover_rules(
    grid: np.ndarray, bg: int, pattern_color: int = 8, local_radius: int = LOCAL_RADIUS
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, ...]]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                    continue

                pattern_info = _collect_local_pattern(patterns, anchor, local_radius)
                if pattern_info is None:
                    continue
                pattern, component_ids = pattern_info

                key = ((int(r), int(c)), (int(tr), int(tc)), component_ids)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": pattern,
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn color-substitution legends whose teal stencil is split across several islands.

    Input structure:
    - The background is black, with isolated colored source pixels scattered around.
    - Each rule legend is a 4-neighbor colored pair source->target, followed in the
      same direction by an anchor cell inside a compact 5x5 teal neighborhood.
    - That local teal neighborhood is not one connected blob: it contains multiple
      disconnected teal fragments that together form the full stencil.

    Rule:
    - For each legend, combine every teal fragment in the anchor's local window into
      one template, then copy that whole fragmented template onto every other pixel
      of the legend's source color.
    - The template is translated so the legend anchor lands on the scattered source
      pixel being replaced, and all stamped pixels are recolored to the legend's
      target color.
    - The legends themselves and all teal pixels disappear in the output.

    Key insight:
    - The adjacent source/target colors still teach the color mapping, but the teal
      stencil must now be reconstructed by grouping several nearby disconnected
      fragments around the same anchor instead of taking a single teal component.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 15, 4: 20}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Fragmented stencil size
STENCIL_TOTAL_MAX_CAP = 9
STENCIL_TOTAL_MIN = 4
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
FRAGMENT_COUNT_MIN = 2
FRAGMENT_COUNT_MAX = 4
FRAGMENT_SIZE_CAP = 3
FRAGMENT_RETRIES = 200

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_LOCAL_CELLS = [
    (r, c)
    for r in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
    for c in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _reserve_window(reserved: set[tuple[int, int]], anchor: tuple[int, int], radius: int) -> None:
    ar, ac = anchor
    for rr in range(ar - radius, ar + radius + 1):
        for cc in range(ac - radius, ac + radius + 1):
            reserved.add((int(rr), int(cc)))


def _far_from_cells(cell: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_chebyshev(cell, other) >= 2 for other in cells)


def _random_partition(total: int, parts: int, piece_cap: int) -> list[int]:
    pieces = [1] * parts
    remaining = total - parts
    while remaining > 0:
        candidates = [i for i, size in enumerate(pieces) if size < piece_cap]
        if not candidates:
            return []
        idx = random.choice(candidates)
        pieces[idx] += 1
        remaining -= 1
    random.shuffle(pieces)
    return pieces


def _grow_fragment(
    start: tuple[int, int], size: int, blocked: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    fragment = {start}
    while len(fragment) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in fragment:
            for dr, dc in _DIRS8:
                nr, nc = r + dr, c + dc
                cell = (int(nr), int(nc))
                if cell in fragment:
                    continue
                if max(abs(cell[0]), abs(cell[1])) > LOCAL_RADIUS:
                    continue
                if not _far_from_cells(cell, blocked):
                    continue
                frontier.add(cell)
        if not frontier:
            return None
        fragment.add(random.choice(sorted(frontier)))
    return fragment


def _make_fragmented_stencil(min_pixels: int, max_pixels: int):
    for _ in range(FRAGMENT_RETRIES):
        total = random.randint(max(FRAGMENT_COUNT_MIN, min_pixels), max_pixels)
        n_fragments = random.randint(FRAGMENT_COUNT_MIN, min(FRAGMENT_COUNT_MAX, total))
        sizes = _random_partition(total, n_fragments, FRAGMENT_SIZE_CAP)
        if not sizes:
            continue

        fragments: list[set[tuple[int, int]]] = []
        first = _grow_fragment((0, 0), sizes[0], blocked=set())
        if first is None:
            continue
        fragments.append(first)
        used = set(first)

        ok = True
        for size in sizes[1:]:
            starts = [cell for cell in _LOCAL_CELLS if cell not in used and _far_from_cells(cell, used)]
            if not starts:
                ok = False
                break
            start = random.choice(starts)
            fragment = _grow_fragment(start, size, blocked=used)
            if fragment is None:
                ok = False
                break
            fragments.append(fragment)
            used.update(fragment)

        if not ok:
            continue

        coords = sorted(used)
        return coords, (0, 0), len(fragments)

    raise RuntimeError("Failed to generate fragmented stencil")


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_TOTAL_MAX_CAP, max(STENCIL_TOTAL_MIN, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_TOTAL_MIN, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor, fragment_count = _make_fragmented_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "fragment_count": fragment_count,
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok or (ar, ac) not in teal_cells:
                continue

            window_cells = {
                (rr, cc)
                for rr in range(ar - LOCAL_RADIUS, ar + LOCAL_RADIUS + 1)
                for cc in range(ac - LOCAL_RADIUS, ac + LOCAL_RADIUS + 1)
            }
            own_cells = {(sr, sc), (tr, tc)} | teal_cells
            if own_cells & (occupied | reserved):
                continue
            if (window_cells - own_cells) & occupied:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(own_cells)
            _reserve_4neighbors(reserved, own_cells)
            _reserve_window(reserved, (ar, ac), LOCAL_RADIUS)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((int(r), int(c), int(bc)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(r, c)])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
