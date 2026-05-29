# Combo variant of abc82100 (index 39)
# combo_spec: D4/1+D5/0+D6/2
# Name: 中心对齐+空心框架+渐变着色

from __future__ import annotations

"""
Design analysis for the fused combo variant.

a) Concrete behavior and trace:
   - The local legend still has the original source -> target -> anchor chain, and
     the anchor still only identifies which teal component belongs to that rule.
   - The teal component is restricted to an axis-aligned hollow rectangular frame.
     Only border cells exist; the interior stays background.
   - When a scattered source pixel is transformed, the frame is not anchor-aligned.
     Instead, the frame's bbox center is moved onto the scattered source pixel.
   - Coloring is fused with that centering rule: the border cells are recolored by
     Manhattan distance from the frame center. To keep the target color visible on
     a hollow shape whose true center is empty, distances are normalized so the
     nearest border layer becomes bucket 0 (target color), the next layer bucket 1,
     and all farther border layers bucket 2.
   - Small trace:
     original: a source color 6 next to target 9 points to a teal motif; every
     loose 6 becomes that motif translated so its anchor lands on the 6, with all
     stamped pixels recolored uniformly to 9.
     fused: the same local legend points to a 5x5 teal border whose anchor is a
     corner. A loose 6 at (10,10) becomes a centered 5x5 ring around (10,10):
     edge-mid cells get 9, inner-diagonal-adjacent edge cells get 2, and corners
     get 7. Relative to the original, both the placement and the recoloring change.

b) Complexity calibration:
   - Frames are sampled from a small odd-sized family: 3x5, 5x3, or 5x5. This keeps
     the rule family coherent while still varying aspect ratio and size.
   - 3x3 is excluded because it yields only two visible distance layers on a hollow
     frame, which weakens the gradient evidence.
   - Anchors are forced to frame corners, never to the center, so center alignment
     cannot collapse back to the original anchor-aligned behavior.
   - Target colors come from a small hand-authored palette with repeated visible
     gradients, so "lighter outward border" is learnable from 2-4 examples.

c) Implicit assumption audit for main():
   - `_discover_rules` originally accepted any teal component. The fused solver now
     assumes every relevant teal component is exactly the full border of its bbox;
     `_frame_signature` enforces that and the generator constructs only such frames.
   - The original stamping logic used `source - anchor` translation. That no longer
     matches the task; `main()` must compute relative coordinates from the frame
     center instead.
   - Flat recoloring is no longer valid. Each pattern cell now carries a center-
     distance bucket, so `main()` must propagate per-cell colors while stamping.
   - Preserving non-legend pixels still assumes bystanders never share a learned
     source color. The generator keeps bystanders off the active source palette.
   - Overwrites would be ambiguous for multi-color borders. The generator prevents
     centered frame bbox overlap, so stamping order does not matter.

d) Placement legality:
   - Every legend is placed fully in-bounds, with the frame touching the anchor
     corner and not covering the legend's source or target cell.
   - Legend regions are separated by a 1-cell guard so teal frames cannot merge
     under 8-connectivity.
   - Every scattered source is placed where its full centered frame fits in-bounds,
     and centered frame bboxes are kept disjoint. This preserves hollow interiors
     and stops gradients from overwriting one another.
   - Bystanders are forbidden from all centered frame bboxes, so they never appear
     inside the hollow windows of the output rings.

e) Visual observability:
   - The hollow-frame change is visible from the teal legend itself and from the
     empty output interiors.
   - The centering change is visible because the scattered source sits at the visual
     center of the ring rather than on a corner or side of it.
   - The gradient change is visible because the same border shape repeats several
     times per rule with a stable center-outward color ordering.

f) Visual cleanliness:
   - Output frame bboxes do not overlap and are kept fully in-bounds.
   - Bystanders are sparse and avoid active source colors, so they do not imitate
     transform triggers.
   - Restricting the generator to a narrow family of odd hollow frames makes the
     centered "ring with layered border colors" effect read cleanly.

g) Change inferability:
   - The color rule is structural, not based on numeric color ids. The user can see
     that the innermost border layer matches the legend target color and outer border
     layers become lighter according to a small repeated palette.
   - Because every generated frame family has at least three center-distance layers,
     the three-color gradient is repeatedly supported and does not rely on a single
     accidental example.
"""

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


TARGET_GRADIENTS: dict[int, tuple[int, int, int]] = {
    1: (1, 8, 4),
    2: (2, 7, 4),
    3: (3, 8, 4),
    9: (9, 2, 7),
}
TARGET_COLORS = list(TARGET_GRADIENTS.keys())
SOURCE_COLORS = [4, 5, 6, 7]
CLEAN_BYSTANDER_COLORS = [5, 6]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _frame_signature(pattern: np.ndarray) -> dict | None:
    """Return normalized metadata if *pattern* is a hollow rectangular frame."""
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

    center = (height // 2, width // 2)
    dists = [abs(r - center[0]) + abs(c - center[1]) for r, c in expected]
    if len(set(dists)) < 3:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "center": (r0 + center[0], c0 + center[1]),
    }


def _pattern_center(pattern: np.ndarray) -> tuple[int, int]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    return (
        int((int(rows.min()) + int(rows.max())) // 2),
        int((int(cols.min()) + int(cols.max())) // 2),
    )


def _center_distance_buckets(pattern: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    distances = np.abs(pattern[:, 0] - int(center[0])) + np.abs(pattern[:, 1] - int(center[1]))
    normalized = distances - int(distances.min())
    return np.minimum(normalized, 2).astype(int)


def _gradient_color(target_color: int, bucket: int) -> int:
    palette = TARGET_GRADIENTS[int(target_color)]
    return int(palette[min(max(int(bucket), 0), 2)])


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    raw_patterns = components_mask(grid == pattern_color, connectivity=8)
    patterns: list[tuple[int, np.ndarray, tuple[int, int], np.ndarray]] = []
    for i, pattern in enumerate(raw_patterns):
        pattern = np.asarray(pattern, dtype=int)
        sig = _frame_signature(pattern)
        if sig is None:
            continue
        center = _pattern_center(pattern)
        buckets = _center_distance_buckets(pattern, center)
        patterns.append((i, pattern, center, buckets))

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
                candidates: list[tuple[int, int, np.ndarray, tuple[int, int], np.ndarray]] = []
                for pattern_index, pattern, center, buckets in patterns:
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(pattern_index), pattern, center, buckets))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern, center, buckets = candidates[0]
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
                        "center": (int(center[0]), int(center[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                        "distance_buckets": np.asarray(buckets, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-frame examples and stamp centered gradient rings elsewhere.

    Input structure:
    - The grid has a dominant background, isolated colored pixels, and teal (8)
      components used as local rule legends.
    - Each relevant teal component is a hollow rectangular frame: only its border
      cells are teal and its interior remains background.
    - A rule example is still a 4-neighbor source->target pair whose continuation
      reaches an anchor touching that teal frame.

    Rule:
    - For each learned source_color -> target_color mapping, copy the associated
      hollow frame onto every non-legend pixel of that source color.
    - Position the copy by aligning the frame's geometric center (its bbox center)
      to the scattered source pixel, ignoring the legend anchor for placement.
    - Recolor the copied border with a center-outward gradient: the nearest border
      layer to the center uses the target color, the next layer uses a lighter
      intermediate color, and any farther layer uses the lightest outer color.
      Remove all legend pixels and all teal frame pixels from the final output.

    Key insight: the adjacent colored pair still teaches which source color maps to
    which target palette, but the teal example now means "center this hollow frame
    on the source pixel and color its border by distance from the frame center."
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
        cr, cc = rule["center"]
        pattern = rule["pattern"]
        rel_rows = pattern[:, 0] - int(cr)
        rel_cols = pattern[:, 1] - int(cc)
        pattern_colors = np.array(
            [_gradient_color(rule["target_color"], int(bucket)) for bucket in rule["distance_buckets"]],
            dtype=int,
        )
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            rows = rel_rows + sr
            cols = rel_cols + sc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = pattern_colors[ok]

    return out


# --------------- generator (c) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 18, 4: 20}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
FRAME_DIMS = [(3, 5), (5, 3), (5, 5)]
FRAME_ANCHORS = ("tl", "tr", "bl", "br")

SCATTERED_MIN = 3

BYSTANDER_MAX = 2

LEGEND_RETRIES = 800
BYSTANDER_RETRIES = 100
OUTER_RETRIES = 600

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 0) -> set[tuple[int, int]]:
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
    raise ValueError(f"unknown anchor kind: {kind}")


def _build_rules(num_rules: int, area: int):
    target_colors = TARGET_COLORS[:]
    source_colors = SOURCE_COLORS[:]
    dims_pool = FRAME_DIMS[:]
    random.shuffle(target_colors)
    random.shuffle(source_colors)
    random.shuffle(dims_pool)

    rules = []
    used = set()
    active_gradient_colors: set[int] = set()
    for i in range(num_rules):
        src_c = int(source_colors[i])
        tgt_c = int(target_colors[i])
        used.update((src_c, tgt_c))
        active_gradient_colors.update(TARGET_GRADIENTS[tgt_c])

        height, width = dims_pool[i % len(dims_pool)]
        anchor_kind = random.choice(FRAME_ANCHORS)
        anchor = _corner_anchor(height, width, anchor_kind)
        center = (height // 2, width // 2)
        cells = _frame_cells(height, width)
        rel_anchor = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in cells]
        rel_center = [(int(r - center[0]), int(c - center[1])) for r, c in cells]
        center_dists = [abs(r - center[0]) + abs(c - center[1]) for r, c in cells]
        min_dist = min(center_dists)
        center_buckets = [min(dist - min_dist, 2) for dist in center_dists]
        span_rows = [rr for rr, _ in rel_center]
        span_cols = [cc for _, cc in rel_center]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "height": height,
                "width": width,
                "anchor_kind": anchor_kind,
                "rel_anchor": rel_anchor,
                "rel_center": rel_center,
                "center_buckets": center_buckets,
                "center_span": (
                    min(span_rows),
                    max(span_rows),
                    min(span_cols),
                    max(span_cols),
                ),
                "n_sc": SCATTERED_MIN,
            }
        )

    return rules, used, active_gradient_colors


def _stamp_cells(rel: list[tuple[int, int]], r: int, c: int) -> set[tuple[int, int]]:
    return {(int(r) + int(rr), int(c) + int(cc)) for rr, cc in rel}


def _stamp_bbox_cells(rule: dict, sr: int, sc: int, h: int, w: int, pad: int = 0) -> set[tuple[int, int]]:
    min_r, max_r, min_c, max_c = rule["center_span"]
    r0 = max(0, int(sr) + int(min_r) - pad)
    r1 = min(h - 1, int(sr) + int(max_r) + pad)
    c0 = max(0, int(sc) + int(min_c) - pad)
    c1 = min(w - 1, int(sc) + int(max_c) + pad)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _can_fit_full_centered(rule: dict, sr: int, sc: int, h: int, w: int) -> bool:
    min_r, max_r, min_c, max_c = rule["center_span"]
    return (
        0 <= int(sr) + int(min_r)
        and int(sr) + int(max_r) < h
        and 0 <= int(sc) + int(min_c)
        and int(sc) + int(max_c) < w
    )


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    legend_guard: set[tuple[int, int]] = set()

    for rule in rules:
        rel = rule["rel_anchor"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = _stamp_cells(rel, ar, ac)
            if any(not (0 <= r0 < h and 0 <= c0 < w) for r0, c0 in teal_cells):
                continue
            if teal_cells & {(sr, sc), (tr, tc)}:
                continue
            if _frame_signature(np.array(sorted(teal_cells), dtype=int)) is None:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
            if all_cells & (occupied | reserved | legend_guard):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, {(int(sr), int(sc)), (int(tr), int(tc))})
            legend_guard.update(_expanded_cells(all_cells, h, w, pad=1))
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(
    rules,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[bool, set[tuple[int, int]] | None]:
    h, w = inp.shape
    output_blocked: set[tuple[int, int]] = set()

    ordered_rules = sorted(rules, key=lambda rule: (rule["height"] * rule["width"], rule["n_sc"]), reverse=True)
    for rule in ordered_rules:
        positions: list[tuple[int, int]] = []

        while len(positions) < rule["n_sc"]:
            candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
            for sr in range(h):
                for sc in range(w):
                    if (sr, sc) in occupied or (sr, sc) in reserved:
                        continue
                    if not _can_fit_full_centered(rule, sr, sc, h, w):
                        continue

                    stamp_bbox = _stamp_bbox_cells(rule, sr, sc, h, w, pad=0)
                    if stamp_bbox & occupied:
                        continue
                    if stamp_bbox & output_blocked:
                        continue
                    if stamp_bbox & reserved:
                        continue
                    candidates.append((int(sr), int(sc), stamp_bbox))

            if not candidates:
                break

            sr, sc, stamp_bbox = random.choice(candidates)
            positions.append((sr, sc))
            occupied.add((sr, sc))
            _reserve_4neighbors(reserved, [(sr, sc)])
            output_blocked.update(stamp_bbox)

        if len(positions) < SCATTERED_MIN:
            return False, None

        rule["scattered"] = positions
        for sr, sc in positions:
            inp[sr, sc] = rule["src"]

    return True, output_blocked


def _place_bystanders(
    used_colors: set[int],
    active_gradient_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_blocked: set[tuple[int, int]],
):
    bystander_colors = [
        c
        for c in CLEAN_BYSTANDER_COLORS
        if c not in used_colors and c not in active_gradient_colors
    ]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_blocked:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int, *, align: str, color_mode: str) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)

    for rule in rules:
        if align == "center":
            rel = rule["rel_center"]
        elif align == "anchor":
            rel = rule["rel_anchor"]
        else:
            raise ValueError(f"unknown align mode: {align}")

        if color_mode == "gradient":
            colors = [_gradient_color(rule["tgt"], bucket) for bucket in rule["center_buckets"]]
        elif color_mode == "flat":
            colors = [int(rule["tgt"])] * len(rel)
        else:
            raise ValueError(f"unknown color mode: {color_mode}")

        for sr, sc in rule["scattered"]:
            for (rr, cc), color in zip(rel, colors):
                nr = int(sr) + int(rr)
                nc = int(sc) + int(cc)
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = int(color)

    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors, active_gradient_colors = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None

    ok, output_blocked = _place_scattered(rules, inp, occupied, reserved)
    if not ok or output_blocked is None:
        return None

    bystander_pos = _place_bystanders(
        used_colors,
        active_gradient_colors,
        inp,
        occupied,
        reserved,
        output_blocked,
    )

    output = _build_output(rules, bystander_pos, h, w, align="center", color_mode="gradient")
    anchor_baseline = _build_output(rules, bystander_pos, h, w, align="anchor", color_mode="gradient")
    flat_baseline = _build_output(rules, bystander_pos, h, w, align="center", color_mode="flat")

    if np.array_equal(output, anchor_baseline):
        return None
    if np.array_equal(output, flat_baseline):
        return None

    teal_patterns = components_mask(inp == 8, connectivity=8)
    if not teal_patterns or any(_frame_signature(np.asarray(pattern, dtype=int)) is None for pattern in teal_patterns):
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
