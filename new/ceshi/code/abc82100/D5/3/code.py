# Variant of abc82100
# Dimension: D5 — 模板放置的对齐方式 (transform)
# Change: d.anchor（规则示例中的对齐参考点）对齐到源像素s的位置，模板整体平移... → 模板叠加而非替换

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D5 overlay variant
#
# a) Concrete behavior and trace
# - The original task erased the whole working area and redrew only recolored
#   copies of the teal stencil plus untouched bystanders.
# - This variant keeps every non-legend, non-prototype pixel as an existing
#   canvas layer. Each learned stencil is then translated onto that canvas.
# - Merge rule:
#   * background + template color -> template color
#   * same color + template color -> unchanged
#   * different non-background color + template color -> teal(8) collision mark
# - Small trace:
#   * Suppose a rule says red(2) -> yellow(4) and its stencil includes the
#     anchor cell. A scattered red source at (6,6) remains on the canvas first.
#   * Original logic would overwrite (6,6) with yellow when stamping.
#   * Variant logic overlays yellow onto the existing red at (6,6), so that
#     cell becomes teal(8). Nearby empty stencil cells still become yellow.
#   * If another translated stencil later also reaches one of those cells with a
#     different target color, that cell also becomes teal instead of being
#     silently overwritten.
#
# b) Complexity calibration
# - The collision outcome uses one visible, fixed color (teal) instead of an
#   invisible ordering by color id.
# - Every stencil is forced to contain its anchor, so each scattered source
#   creates at least one visible overlay collision at its own location.
# - The generator also requires extra collisions or footprint overlaps beyond
#   those self-collisions, so the variant does not collapse back to the
#   overwrite behavior of the seed task.
#
# c) Main() lines that depended on the old property
# - `out = full(bg)` assumed the picture should be rebuilt from scratch. That is
#   false now because existing colored pixels participate in the overlay.
# - Skipping all `source_colors` during preservation assumed every scattered
#   source pixel should disappear before stamping. That is false now because the
#   source pixels must remain as the lower layer that templates collide with.
# - Direct assignment `out[...] = target` assumed overwriting was harmless. That
#   is false now because overlap outcome is the whole point of the variant.
#
# d) Placement legality guarantees from the generator
# - Legend triplets and teal prototype components are still separated by a
#   reserved 4-neighbor buffer so rule discovery stays unambiguous.
# - Scattered source pixels are kept off the legend buffer and are prevented from
#   becoming 4-neighbor colored pairs, which avoids accidental extra rules.
# - Each stencil anchor is one of the teal cells, matching the solver's simpler
#   interpretation and guaranteeing the source pixel is covered by its own stamp.
#
# e/f/g) Visual observability and cleanliness
# - The puzzle remains human-readable because there is still a small set of rule
#   examples plus repeated scattered sources.
# - Teal is reused as a collision marker: it is already visually linked to the
#   prototype material, so "teal appears where paint lands on occupied cells" is
#   a compact, observable rule across few examples.
# - The generator keeps bystanders limited and requires nontrivial interaction,
#   while avoiding hidden numeric orderings or arbitrary per-color cases.
# ---------------------------------------------------------------------------


PATTERN_COLOR = 8
COLLISION_COLOR = 8


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

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
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
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
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _merge_color(current: int, incoming: int, bg: int, collision_color: int = COLLISION_COLOR) -> int:
    current = int(current)
    incoming = int(incoming)
    bg = int(bg)
    if current == bg:
        return incoming
    if current == incoming or current == int(collision_color):
        return current
    return int(collision_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-template examples, erase the legend, then overlay
    recolored stencil copies on top of the surviving picture.

    Input structure:
    - Black background with colored source pixels, optional other colored
      bystanders, and teal (8) stencil components used only as rule examples.
    - Each rule example is a 4-neighbor source->target pair whose continued
      direction reaches an anchor touching the corresponding teal component.

    Rule:
    - Remove every legend pair and every teal prototype component from the
      output, but keep all other colored pixels as the starting canvas.
    - For each scattered source pixel, translate the matched teal component so
      its anchor lands on that source pixel, recolor the component to the rule's
      target color, and overlay it cell by cell.
    - Background accepts the new color, equal colors merge harmlessly, and any
      different non-background overlap turns into teal.

    Key insight:
    - The local example still teaches which source color emits which target
      stencil, but now the stencil behaves like paint laid over existing content
      rather than a replacement that clears the canvas first.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = PATTERN_COLOR
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])

    out = np.full_like(g, int(bg))

    # Keep all ordinary non-teal pixels except the legend pair cells themselves.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        if (r, c) in legend_pixels:
            continue
        out[r, c] = int(g[r, c])

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        target_color = int(rule["target_color"])
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
            for nr, nc in zip(rows[ok], cols[ok]):
                nr = int(nr)
                nc = int(nc)
                out[nr, nc] = _merge_color(out[nr, nc], target_color, bg=bg)

    return out


# --------------- generator (overlay variant) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 6, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size and scattered counts
STENCIL_MIN_PIXELS = 3
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 22
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 7
SCATTERED_AREA_DIV = 32
BYSTANDER_MAX = 4

# Placement budgets
LEGEND_RETRIES = 500
SCATTER_CLUSTER_RETRIES = 120
BYSTANDER_RETRIES = 120
OUTER_RETRIES = 400

# Directions
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _has_adjacent_4(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any((r + dr, c + dc) in occupied for dr, dc in _DIRS4)


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    while len(coords) < n:
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                frontier.add((r + dr, c + dc))
        frontier -= coords
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    anchor = random.choice(coords_list)
    return coords_list, anchor


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_MIN_PIXELS, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
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
            if not ok:
                continue
            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue
            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = inp.shape
    for rule in rules:
        positions: list[tuple[int, int]] = []
        placed = False
        for _ in range(SCATTER_CLUSTER_RETRIES):
            radius = random.choice((2, 2, 3))
            cr = random.randint(radius, H - 1 - radius) if H > 2 * radius else random.randint(0, H - 1)
            cc = random.randint(radius, W - 1 - radius) if W > 2 * radius else random.randint(0, W - 1)
            candidates = [
                (r, c)
                for r in range(max(0, cr - radius), min(H, cr + radius + 1))
                for c in range(max(0, cc - radius), min(W, cc + radius + 1))
                if (r, c) not in occupied and (r, c) not in reserved
            ]
            random.shuffle(candidates)
            picks: list[tuple[int, int]] = []
            blocked = set(occupied)
            for r, c in candidates:
                if _has_adjacent_4((r, c), blocked):
                    continue
                picks.append((int(r), int(c)))
                blocked.add((int(r), int(c)))
                if len(picks) >= rule["n_sc"]:
                    break
            if len(picks) >= SCATTERED_MIN:
                positions = picks
                placed = True
                break
        if not placed:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
            occupied.add((int(r), int(c)))
    return True


def _stamp_cells(rel: list[tuple[int, int]], sr: int, sc: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr, cc in rel:
        nr = int(sr + rr)
        nc = int(sc + cc)
        if 0 <= nr < H and 0 <= nc < W:
            cells.append((nr, nc))
    return cells


def _place_bystanders(rules, used_colors: set[int], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    H, W = inp.shape
    footprint_cells: list[tuple[int, int]] = []
    source_cells = {pos for rule in rules for pos in rule["scattered"]}
    for rule in rules:
        for sr, sc in rule["scattered"]:
            footprint_cells.extend(_stamp_cells(rule["rel"], sr, sc, H, W))

    counts = Counter(footprint_cells)
    targeted_candidates = [
        cell
        for cell, _count in counts.most_common()
        if cell not in occupied
        and cell not in reserved
        and not _has_adjacent_4(cell, occupied)
    ]

    placed: list[tuple[int, int, int]] = []
    if targeted_candidates:
        n_targeted = random.randint(1, min(2, len(targeted_candidates), len(bystander_colors)))
        for r, c in targeted_candidates[:n_targeted]:
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))

    remaining = max(0, random.randint(0, BYSTANDER_MAX) - len(placed))
    for _ in range(remaining):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved or _has_adjacent_4((r, c), occupied):
                continue
            if (r, c) in source_cells:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            break

    return placed


def _build_output(rules, bystander_pos, H: int, W: int, bg: int = 0) -> tuple[np.ndarray, dict]:
    output = np.full((H, W), int(bg), dtype=int)
    coverage = Counter()
    source_cells: set[tuple[int, int]] = set()
    bystander_cells = {(int(r), int(c)): int(color) for r, c, color in bystander_pos}

    for rule in rules:
        for sr, sc in rule["scattered"]:
            output[sr, sc] = int(rule["src"])
            source_cells.add((int(sr), int(sc)))
    for (r, c), color in bystander_cells.items():
        output[r, c] = color

    for rule in rules:
        for sr, sc in rule["scattered"]:
            cells = _stamp_cells(rule["rel"], sr, sc, H, W)
            for cell in cells:
                coverage[cell] += 1
                r, c = cell
                output[r, c] = _merge_color(output[r, c], int(rule["tgt"]), bg=bg)

    collision_cells = {(int(r), int(c)) for r, c in np.argwhere(output == COLLISION_COLOR)}
    bystander_hits = sum((r, c) in collision_cells for r, c, _ in bystander_pos)
    metrics = {
        "source_count": len(source_cells),
        "collision_cells": len(collision_cells),
        "extra_collision_cells": len(collision_cells) - len(source_cells),
        "overlap_cells": sum(count >= 2 for count in coverage.values()),
        "bystander_hits": int(bystander_hits),
    }
    return output, metrics


def _build_old_style_output(rules, bystander_pos, H: int, W: int) -> np.ndarray:
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells(rule["rel"], sr, sc, H, W):
                output[nr, nc] = int(rule["tgt"])
    return output


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(rules, used_colors, inp, occupied, reserved)

    output, metrics = _build_output(rules, bystander_pos, H, W)
    old_style = _build_old_style_output(rules, bystander_pos, H, W)
    if metrics["extra_collision_cells"] < 1:
        return None
    if metrics["overlap_cells"] < 1:
        return None
    if np.array_equal(output, old_style):
        return None
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
