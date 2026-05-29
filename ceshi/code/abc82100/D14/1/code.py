# Variant of abc82100
# Dimension: D14 — 模板锚点的定义方式 (meta)
# Change: 锚点由规则示例中源色像素相对于teal模板的位置隐式定义... → 锚点由模板内特殊标记像素指定

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D14 variant
# -----------------------------------
# spec_variant:
#   core_change:
#     original: the legend pair source->target implied an anchor by extending the pair's direction.
#     variant: the nearby template is now a mixed 8-connected component made of teal cells plus
#       exactly one fixed-color marker cell; that marker cell is the only true anchor.
#   concrete_objects:
#     background: black (0)
#     legend_pair: two 4-neighbor non-bg cells source->target, same as the base task
#     template_component:
#       colors: teal(8) cells plus one blue(1) marker cell
#       connectivity: the teal+marker union is 8-connected
#       visibility: the continuation cell after target is always a teal cell, not the marker
#       anchor: the blue marker's position
#     scattered_sources:
#       role: each isolated source-color pixel receives a translated copy of the teal cells only
#       translation: move the template so the blue marker would land on the scattered source pixel
#   behavior_trace:
#     example:
#       legend: source at (5,5), target at (5,6), continuation probe at (5,7)
#       template: probe (5,7) is teal, marker is at (4,7), another teal at (4,8)
#       original_logic: would anchor at (5,7), so the copied shape is shifted one row too low
#       new_logic: must first find the mixed template component, then read the marker at (4,7)
#       difference: the explicit marker changes the stamping offset even though the same legend pair is used
#   complexity_calibration:
#     why_not_simpler: if the marker sat exactly on the continuation cell, old and new behaviors
#       would coincide too often and the variant would visually collapse back to the base task
#     chosen_constraint: generator always chooses a teal continuation cell distinct from the marker
#     color_budget: only one extra dedicated marker color is introduced; source/target colors still
#       come from a small repeated palette so examples remain inferable from 2-4 IO pairs
#   implicit_assumption_review:
#     - old _discover_rules assumed anchor = continuation cell; this is no longer valid and must be
#       replaced by "find component from continuation probe, then read marker inside component"
#     - old pattern extraction used pure teal components; now extraction must use (teal OR marker)
#       components, then discard the marker when stamping
#     - old preserve-loop only excluded teal; now it must also exclude marker-colored legend pixels
#     - unconditional stamping is still safe because output may allow overlaps and clipping exactly
#       as in the base task; only the offset source changes
#   placement_legality:
#     - each template component contains exactly one marker and at least one teal cell
#     - the continuation probe always lands on a teal cell of that component
#     - source/target colors never use the marker color, so marker detection is unambiguous
#     - objects reserve 8-neighborhoods after placement, preventing accidental component merges and
#       keeping mixed templates visually separable
#   visual_observability:
#     - humans can now see a repeated blue dot inside every legend template, and that dot is the
#       place that aligns with the scattered source pixels
#     - because the probe is teal while the anchor is blue, the offset from pair to template is
#       visibly different from the base task's "just continue in the same direction" rule
#   cleanliness:
#     - the marker is only used inside templates and never as a bystander/source/target color
#     - templates do not touch each other even diagonally, avoiding noisy merged shapes
#   color_inference:
#     - color substitution is still taught by local source->target pairs
#     - the marker color does not encode any rule identity; it only identifies anchor position
# assumptions_enforced_by_generator:
#   - every mixed template component has exactly one marker
#   - every rule has a unique source color
#   - every scattered source is isolated from other colored cells by at least one king move


MARKER_COLOR = 1
PATTERN_COLOR = 8


def _discover_pattern_components(
    grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR, marker_color: int = MARKER_COLOR
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    """Return valid mixed template components and a cell->component lookup."""
    mask = (grid == int(pattern_color)) | (grid == int(marker_color))
    comps = components_mask(mask, connectivity=8)
    patterns: list[dict] = []
    cell_to_pattern: dict[tuple[int, int], int] = {}

    for pts in comps:
        teal_cells = []
        marker_cells = []
        for r, c in map(tuple, pts):
            rr = int(r)
            cc = int(c)
            color = int(grid[rr, cc])
            if color == int(pattern_color):
                teal_cells.append((rr, cc))
            elif color == int(marker_color):
                marker_cells.append((rr, cc))
        if len(marker_cells) != 1 or not teal_cells:
            continue

        idx = len(patterns)
        marker = marker_cells[0]
        teal_arr = np.asarray(sorted(teal_cells), dtype=int)
        patterns.append({"marker": marker, "teal": teal_arr})
        for r, c in teal_cells:
            cell_to_pattern[(int(r), int(c))] = idx
        cell_to_pattern[(int(marker[0]), int(marker[1]))] = idx

    return patterns, cell_to_pattern


def _discover_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> list[dict]:
    h, w = grid.shape
    patterns, cell_to_pattern = _discover_pattern_components(grid, bg, pattern_color, marker_color)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(marker_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                pr = tr + dr
                pc = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= pr < h and 0 <= pc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(marker_color)) or target_color == source_color:
                    continue

                pattern_index = cell_to_pattern.get((int(pr), int(pc)))
                if pattern_index is None:
                    continue

                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)

                pattern = patterns[pattern_index]
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "probe_pos": (int(pr), int(pc)),
                        "anchor": (int(pattern["marker"][0]), int(pattern["marker"][1])),
                        "pattern": np.asarray(pattern["teal"], dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-template examples to replace matching source pixels.

    Input structure:
    - Black background with isolated colored pixels and mixed template components.
    - Each template component is mostly teal (8) but contains one blue (1) marker pixel.
    - A rule example is a 4-neighbor source->target color pair whose continued direction
      lands on a teal cell of the nearby template component.

    Rule:
    - For each rule example, find the touched mixed template component and treat its blue
      marker as the template's anchor.
    - Copy only the teal cells of that template onto every other pixel of source.color,
      recoloring the copy to target.color and aligning the blue marker to the source pixel.
    - Remove all legend pairs and all template pixels, including the blue markers.

    Key insight: The source-target pair still teaches the color substitution, but the
    translation offset now comes from the explicit marker embedded inside the template.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Preserve ordinary colored pixels that are not template parts and not consumed as sources.
    keep_mask = (g != int(bg)) & (g != PATTERN_COLOR) & (g != MARKER_COLOR)
    for r, c in map(tuple, np.argwhere(keep_mask)):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        if (rr, cc) in legend_pixels or color in source_colors:
            continue
        out[rr, cc] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            rr = int(sr)
            cc = int(sc)
            if (rr, cc) in legend_pixels:
                continue
            dr = rr - ar
            dc = cc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
PAIR_COLORS = [2, 3, 4, 5, 6, 7, 9]

# Stencil size counts the full mixed component: teal cells + one marker.
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

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


def _reserve_neighbors(reserved: set[tuple[int, int]], cells) -> None:
    """Reserve king-neighborhoods so mixed template components never merge accidentally."""
    for r, c in cells:
        for dr, dc in _DIRS8:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_cells: int, max_cells: int):
    """Generate an 8-connected mixed component. Anchor marker stays at (0,0)."""
    n = random.randint(min_cells, max_cells)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))

    teal_rel = sorted(p for p in coords if p != (0, 0))
    assert teal_rel, "mixed component must contain at least one teal cell"
    touch_rel = random.choice(teal_rel)
    return teal_rel, touch_rel


def _build_rules(num_rules: int, area: int):
    """Assign source/target colors and mixed template geometry."""
    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    source_colors = random.sample(PAIR_COLORS, num_rules)
    rules = []
    used = {MARKER_COLOR}
    for src_c in source_colors:
        target_choices = [c for c in PAIR_COLORS if c != src_c]
        tgt_c = random.choice(target_choices)
        used.update((src_c, tgt_c))
        teal_rel, touch_rel = _make_stencil(min_s, max_s)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "teal_rel": teal_rel,
                "touch_rel": touch_rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    """Place source->target legends and nearby mixed template components."""
    H, W = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            pr, pc = sr + 2 * dr, sc + 2 * dc
            anchor_r = pr - rule["touch_rel"][0]
            anchor_c = pc - rule["touch_rel"][1]
            if not (0 <= tr < H and 0 <= tc < W and 0 <= anchor_r < H and 0 <= anchor_c < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rule["teal_rel"]:
                nr, nc = anchor_r + rr, anchor_c + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok or (pr, pc) not in teal_cells:
                continue

            marker_cell = (anchor_r, anchor_c)
            template_cells = set(teal_cells) | {marker_cell}
            all_cells = {(sr, sc), (tr, tc)} | template_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            inp[marker_cell] = MARKER_COLOR
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    """Place isolated scattered source pixels for each rule."""
    H, W = inp.shape
    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    """Place unrelated bystander pixels whose colors are outside the learned rules."""
    bystander_colors = [c for c in PAIR_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, H, W):
    """Construct expected output by stamping teal cells around each scattered source."""
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["teal_rel"]:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _gen_attempt():
    """Single generation attempt. Returns input grid or None on failure."""
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
