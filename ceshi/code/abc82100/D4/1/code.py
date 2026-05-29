# Variant of abc82100
# Dimension: D4 — teal模板的形状类型 (dst)
# Change: teal模板是由teal像素构成的任意连通图案（菱形、线段等），定义了要复制的形状... → 模板为空心框架

from __future__ import annotations

"""
Design analysis for the D4 hollow-frame variant.

a) Concrete behavior and trace:
   - Each learned teal component is no longer an arbitrary connected blob. It is an
     axis-aligned hollow rectangular frame: only the border pixels are teal and the
     interior stays background.
   - The local legend still uses a source pixel next to a target pixel, followed by
     an anchor cell touching the teal template. The solver still learns
     source_color -> target_color plus a translation anchor.
   - Small trace:
     original: a red source next to a blue target may point to a solid/irregular teal
     component; every loose red pixel turns into that same filled blob in blue.
     variant: the same red/blue legend points to a teal 4x4 border; every loose red
     pixel turns into a blue 4x4 outline, keeping its center empty.
     difference: only the template family changed, so the copied object becomes a
     visibly hollow frame instead of a solid/opaque shape.

b) Complexity calibration:
   - Frames are restricted to a small family of 3x3..5x3 / 4x4 border rectangles.
     This is narrow enough to be learnable, but still gives several distinct sizes.
   - When multiple rules are present, their frame sizes are sampled without replacement
     when possible, so the grid shows repeated evidence that teal means "hollow frame",
     not one accidental tiny square.

c) Implicit assumption audit for main():
   - Rule discovery assumes the legend remains a 4-neighbor source->target pair whose
     continuation reaches exactly one nearby teal component. The generator preserves
     this by isolating legend regions.
   - Stamping assumes each source color belongs to a single rule. The generator keeps
     source colors unique.
   - Unconditional writes during stamping would be order-dependent if outputs overlapped.
     The generator avoids overlapping output frame bounding boxes, so this remains safe.
   - The new frame filter assumes a teal component equals the full border of its bbox.
     The generator guarantees that every teal component has height >= 3, width >= 3,
     and contains exactly those border cells.

d) Placement legality:
   - Legend frames are placed fully in-bounds and separated so teal components cannot
     touch or merge under 8-connectivity.
   - Scattered source pixels are placed only where their translated frame fits fully
     in-bounds; each frame bbox is kept disjoint from others to preserve empty centers.
   - Bystanders are excluded from all reserved frame bboxes so they survive unchanged
     and do not appear inside hollow interiors.

e) Visual observability:
   - The change is visible directly: outputs contain empty rectangular rings rather than
     arbitrary solid blobs.
   - Repetition is structural, not color-id based. The color mapping is still given by
     the legend, while the frame geometry is shown by the teal example itself.

f) Visual cleanliness:
   - Full-frame clipping is avoided in generated samples so the hollow interiors remain
     obvious.
   - Output frame bboxes do not overlap, which prevents confusing tangles of borders.

g) Color inference:
   - Colors behave exactly as in the original task: adjacent legend pixels define the
     source/target mapping. The geometry change does not add hidden color logic.
"""

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _frame_signature(pattern: np.ndarray) -> dict | None:
    """Return normalized hollow-frame metadata, or None if the component is not one."""
    if pattern.size == 0:
        return None

    rows = pattern[:, 0]
    cols = pattern[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    normalized = {(int(r - r0), int(c - c0)) for r, c in pattern}
    expected = set()
    for r in range(height):
        expected.add((r, 0))
        expected.add((r, width - 1))
    for c in range(width):
        expected.add((0, c))
        expected.add((height - 1, c))

    if normalized != expected:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "area": len(expected),
    }


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    raw_patterns = components_mask(grid == pattern_color, connectivity=8)
    patterns: list[tuple[int, np.ndarray]] = []
    for i, pattern in enumerate(raw_patterns):
        if _frame_signature(pattern) is not None:
            patterns.append((i, pattern))

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
                for pattern_index, pattern in patterns:
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(pattern_index), pattern))

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-frame examples to replace matching source pixels.

    Input structure:
    - Black background with isolated colored pixels and teal (8) components.
    - Each relevant teal component is a hollow rectangular frame: only its border
      pixels are teal, and its interior is left as background.
    - A rule example is a 4-neighbor colored pair source->target whose continued
      direction reaches an anchor touching that teal frame.

    Rule:
    - For each learned source_color -> target_color rule, copy the associated teal
      frame onto every other pixel of that source color.
    - The translation aligns the legend's anchor with the scattered source pixel.
    - Recolor the copied frame to the target color, and erase all legend pixels and
      all teal frame pixels from the final output.

    Key insight: the adjacent colored pair still teaches the color substitution, but
    now the nearby teal component is specifically an empty rectangular outline, so the
    solver must replicate a hollow frame rather than a solid blob.
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
MIN_DIM_BY_RULES = {1: 8, 2: 11, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Hollow rectangular frame choices. Area equals border-cell count.
FRAME_DIMS = [(3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3)]
FRAME_ANCHORS = ("tl", "tr", "bl", "br")

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 32

# Bystander pixels
BYSTANDER_MAX = 4

# Placement retry budgets
LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _expanded_cells(cells, h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    r0 = max(0, min(rows) - pad)
    r1 = min(h - 1, max(rows) + pad)
    c0 = max(0, min(cols) - pad)
    c1 = min(w - 1, max(cols) + pad)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _frame_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = set()
    for r in range(height):
        cells.add((r, 0))
        cells.add((r, width - 1))
    for c in range(width):
        cells.add((0, c))
        cells.add((height - 1, c))
    return sorted(cells)


def _corner_anchor(height: int, width: int, kind: str) -> tuple[int, int]:
    if kind == "tl":
        return 0, 0
    if kind == "tr":
        return 0, width - 1
    if kind == "bl":
        return height - 1, 0
    if kind == "br":
        return height - 1, width - 1
    raise ValueError(f"unknown corner anchor: {kind}")


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    dims_pool = FRAME_DIMS[:]
    random.shuffle(dims_pool)
    sc_max = min(
        SCATTERED_MAX_CAP,
        max(SCATTERED_MIN, area // max(1, SCATTERED_AREA_DIV * num_rules)),
    )

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))

        height, width = dims_pool[i % len(dims_pool)]
        anchor_kind = random.choice(FRAME_ANCHORS)
        anchor = _corner_anchor(height, width, anchor_kind)
        cells = _frame_cells(height, width)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in cells]
        rel_rows = [rr for rr, _ in rel]
        rel_cols = [cc for _, cc in rel]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "height": height,
                "width": width,
                "anchor_kind": anchor_kind,
                "bbox_rel": (
                    min(rel_rows),
                    max(rel_rows),
                    min(rel_cols),
                    max(rel_cols),
                ),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _stamp_cells(rule: dict, sr: int, sc: int) -> set[tuple[int, int]]:
    return {(sr + rr, sc + cc) for rr, cc in rule["rel"]}


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    legend_guard: set[tuple[int, int]] = set()

    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & (occupied | reserved | legend_guard):
                continue
            if _frame_signature(np.array(sorted(teal_cells), dtype=int)) is None:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            legend_guard.update(_expanded_cells(all_cells, h, w, pad=1))
            rule["legend_cells"] = all_cells
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    output_blocked: set[tuple[int, int]] = set()

    for rule in rules:
        positions = []
        attempts = rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL
        for _ in range(attempts):
            if len(positions) >= rule["n_sc"]:
                break
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            if (sr, sc) in occupied or (sr, sc) in reserved:
                continue

            stamp_cells = _stamp_cells(rule, sr, sc)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp_cells):
                continue

            blocked = _expanded_cells(stamp_cells, h, w, pad=0)
            if blocked & output_blocked:
                continue

            positions.append((sr, sc))
            occupied.add((sr, sc))
            _reserve_4neighbors(reserved, [(sr, sc)])
            output_blocked.update(blocked)
        if len(positions) < SCATTERED_MIN:
            return False, None
        rule["scattered"] = positions
        for sr, sc in positions:
            inp[sr, sc] = rule["src"]
    return True, output_blocked


def _place_bystanders(used_colors, inp, occupied, reserved, output_blocked):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_blocked:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, h, w):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                output[sr + rr, sc + cc] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None

    ok, output_blocked = _place_scattered(rules, inp, occupied, reserved)
    if not ok or output_blocked is None:
        return None

    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_blocked)
    output = _build_output(rules, bystander_pos, h, w)

    teal_patterns = components_mask(inp == 8, connectivity=8)
    if not teal_patterns or any(_frame_signature(pattern) is None for pattern in teal_patterns):
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
