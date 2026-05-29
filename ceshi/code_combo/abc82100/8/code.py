# Combo variant of abc82100 (index 8)
# combo_spec: D3/1+D4/1
# Name: 对角标记源+空心框架模板

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


"""
Design analysis for the D3/1 + D4/1 combo variant.

a) Concrete behavior and trace:
   - The legend mechanism stays local: a colored source pixel next to a colored target pixel,
     followed in the same cardinal direction by a teal component, still teaches one
     source_color -> target_color rule.
   - The two modified dimensions now cooperate instead of acting independently. Away from the
     legend, a source occurrence is not a single pixel but a same-color diagonal marker pair.
     Those two markers define the opposite corners of an axis-aligned rectangle. Meanwhile the
     teal template is not an arbitrary shape but a hollow rectangular frame.
   - Because both objects are rectangles, the natural fused rule is: recover the source rectangle
     from the diagonal pair, then draw the learned frame so that it exactly occupies that
     rectangle boundary.
   - Small trace: if a legend says "3 becomes 2" and its teal component is a 4x5 hollow frame,
     then two stray 3-pixels placed at opposite corners of an empty 4x5 box produce one 2-colored
     4x5 outline in the output. In the original task, each stray 3-pixel would have stamped the
     teal component around itself; in the combo, the pair jointly determines one whole rectangle.

b) Complexity calibration:
   - Frames are restricted to a small family of sizes (3x3 up to 5x3 / 3x5 / 4x4). This keeps
     the geometry learnable while still making the hollow-frame change visually obvious.
   - Each source pair must match the learned frame dimensions exactly. That prevents an overly
     permissive "clip any frame into any box" interpretation and makes the fusion visible.
   - Source rectangles are isolated with a one-cell halo, so diagonal pairing is readable and
     accidental alternative pairings are suppressed.

c) Implicit assumption audit for main():
   - The original main() assumed every non-legend source-colored pixel was an independent source.
     That breaks: markers must be consumed in pairs.
   - The original main() also assumed translation was anchored on one source pixel. That breaks
     too: the relevant geometry is now the full bounding rectangle of a pair, not a single point.
   - The D4 change adds a new assumption: a usable teal component must be a full rectangular
     border of its bbox. The solver filters components accordingly, and the generator guarantees it.
   - Once these assumptions change together, the anchor is used only to associate the correct
     legend pair with the correct teal frame. Placement over non-legend sources comes from the
     recovered rectangle dimensions, not from anchor translation.

d) Placement legality enforced by the generator:
   - Every non-legend source rectangle contains exactly two colored cells: the diagonal markers.
   - Each source rectangle has the same height/width as that rule's legend frame.
   - Rectangles, legends, teal components, and bystanders are separated by reserved halos so
     nothing overlaps and no false legend can be discovered.
   - Legend teal components are fully in-bounds and isolated under 8-connectivity.

e) Visual observability:
   - The input shows two strong cues humans can use together: same-color diagonal corner pairs and
     teal empty rectangles in the legend.
   - The output then replaces those corner pairs with colored hollow rectangles of the same size.
     This is visibly different from both the base task and a generic rectangle-filling task.

f) Visual cleanliness:
   - Output frame bboxes are kept disjoint, so hollow interiors remain legible.
   - Bystanders are excluded from source rectangles and from legend neighborhoods, preventing
     clutter inside empty boxes or near the teaching examples.

g) Color inference:
   - Color logic is still simple and local: the adjacent legend pair determines the mapping.
     The diagonal markers use only the source color, and the produced frame uses only the target
     color. No extra hidden color categories are introduced.

h) Generator/solver agreement:
   - The solver assumes valid sources are the only same-color diagonal pairs whose empty bbox
     matches the learned frame dimensions. The generator constructs exactly that family and
     validates each sample by re-running main().
"""


GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 7, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

FRAME_DIMS = [(3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3)]
FRAME_ANCHORS = ("tl", "tr", "bl", "br")

BYSTANDER_MAX = 4

LEGEND_RETRIES = 700
SOURCE_LAYOUT_RETRIES = 50
PAIR_RETRIES_PER_OBJECT = 100
BYSTANDER_RETRIES = 100
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _frame_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = set()
    for r in range(int(height)):
        cells.add((r, 0))
        cells.add((r, int(width) - 1))
    for c in range(int(width)):
        cells.add((0, c))
        cells.add((int(height) - 1, c))
    return sorted(cells)


def _frame_signature(pattern: np.ndarray) -> dict | None:
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
    expected = set(_frame_cells(height, width))
    if normalized != expected:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "normalized": np.asarray(sorted(expected), dtype=int),
    }


def _corner_anchor(height: int, width: int, kind: str) -> tuple[int, int]:
    if kind == "tl":
        return 0, 0
    if kind == "tr":
        return 0, int(width) - 1
    if kind == "bl":
        return int(height) - 1, 0
    if kind == "br":
        return int(height) - 1, int(width) - 1
    raise ValueError(f"unknown anchor kind: {kind}")


def _box_cells(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 0) -> set[tuple[int, int]]:
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(int(h) - 1, int(r1) + int(pad))
    cc1 = min(int(w) - 1, int(c1) + int(pad))
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _bbox_frame_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0 + rr, c0 + cc) for rr, cc in _frame_cells(r1 - r0 + 1, c1 - c0 + 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    raw_patterns = components_mask(grid == pattern_color, connectivity=8)
    patterns: list[tuple[int, np.ndarray, dict]] = []
    for idx, pattern in enumerate(raw_patterns):
        meta = _frame_signature(np.asarray(pattern, dtype=int))
        if meta is not None:
            patterns.append((int(idx), np.asarray(pattern, dtype=int), meta))

    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray, dict]] = []
                for pattern_index, pattern, meta in patterns:
                    dist = _anchor_distance(pattern, anchor)
                    if dist == 0:
                        candidates.append((int(dist), int(pattern_index), pattern, meta))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[3]["height"] * item[3]["width"]))
                _dist, pattern_index, pattern, meta = candidates[0]
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
                        "height": int(meta["height"]),
                        "width": int(meta["width"]),
                    }
                )

    return rules


def _pair_area(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (abs(int(a[0]) - int(b[0])) + 1) * (abs(int(a[1]) - int(b[1])) + 1)


def _find_source_pairs(
    grid: np.ndarray,
    color: int,
    *,
    bg: int,
    legend_pixels: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[dict]:
    points = sorted(
        (
            (int(r), int(c))
            for r, c in np.argwhere(grid == int(color))
            if (int(r), int(c)) not in legend_pixels
        )
    )
    if len(points) < 2:
        return []

    n = len(points)
    candidates: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            a = points[i]
            b = points[j]
            if a[0] == b[0] or a[1] == b[1]:
                continue

            r0 = min(a[0], b[0])
            c0 = min(a[1], b[1])
            r1 = max(a[0], b[0])
            c1 = max(a[1], b[1])
            if r1 - r0 + 1 != int(height) or c1 - c0 + 1 != int(width):
                continue

            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(np.count_nonzero(window != int(bg))) != 2:
                continue

            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "bbox": (r0, c0, r1, c1),
                    "markers": (a, b),
                    "area": _pair_area(a, b),
                }
            )

    by_point: list[list[dict]] = [[] for _ in range(n)]
    for cand in candidates:
        by_point[cand["i"]].append(cand)
        by_point[cand["j"]].append(cand)

    full_mask = (1 << n) - 1
    memo: dict[int, tuple[int, int, list[dict]]] = {}

    def search(mask: int) -> tuple[int, int, list[dict]]:
        if mask == full_mask:
            return (0, 0, [])
        if mask in memo:
            return memo[mask]

        first = next(idx for idx in range(n) if not (mask & (1 << idx)))
        best = search(mask | (1 << first))

        for cand in by_point[first]:
            i = cand["i"]
            j = cand["j"]
            if mask & (1 << i) or mask & (1 << j):
                continue
            count, area, chosen = search(mask | (1 << i) | (1 << j))
            proposal = (count + 1, area + int(cand["area"]), [cand] + chosen)
            if proposal[0] > best[0] or (proposal[0] == best[0] and proposal[1] < best[1]):
                best = proposal

        memo[mask] = best
        return best

    pair_count, _total_area, chosen = search(0)
    if pair_count == 0:
        return []

    chosen.sort(key=lambda item: item["bbox"])
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-frame examples to turn diagonal marker pairs into hollow rectangles.

    Input structure:
    - Black background with teal (8) components and local colored rule examples.
    - Each useful teal component is a hollow axis-aligned rectangular frame rather than an
      arbitrary connected blob.
    - Away from the legends, each source occurrence is a same-color diagonal marker pair whose
      two pixels are the opposite corners of an otherwise empty rectangle.

    Rule:
    - Learn source_color -> target_color from the adjacent legend pair, and learn the frame size
      from the nearby teal component.
    - For every non-legend diagonal pair of that source color, recover its bounding rectangle.
      If that rectangle matches the learned frame dimensions, paint only its border in the target
      color and erase the source markers.
    - Remove all legend pixels and all teal legend frames; preserve ordinary bystanders.

    Key insight: the legend still teaches the color substitution, but the source object is now a
    hidden rectangle encoded by two diagonal corners, and the teal template tells you that the
    output should be the corresponding hollow rectangular border.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8

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
        pairs = _find_source_pairs(
            g,
            rule["source_color"],
            bg=bg,
            legend_pixels=legend_pixels,
            height=rule["height"],
            width=rule["width"],
        )
        for pair in pairs:
            for rr, cc in _bbox_frame_cells(pair["bbox"]):
                out[rr, cc] = rule["target_color"]

    return out


def _choose_pair_count(num_rules: int, area: int) -> int:
    if num_rules == 1:
        return 3 if area >= 260 and random.random() < 0.5 else 2
    if num_rules == 2:
        return 2 if area >= 220 else 1
    return 1


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    dims_pool = FRAME_DIMS[:]
    random.shuffle(dims_pool)

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        height, width = dims_pool[i % len(dims_pool)]
        anchor_kind = random.choice(FRAME_ANCHORS)
        anchor = _corner_anchor(height, width, anchor_kind)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in _frame_cells(height, width)]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "height": height,
                "width": width,
                "rel": rel,
                "n_sources": _choose_pair_count(num_rules, area),
            }
        )
    return rules, used


def _place_source_pairs(rules, inp, occupied, reserved) -> bool:
    h, w = inp.shape
    placement_order = sorted(rules, key=lambda rule: (-(rule["height"] * rule["width"]), -rule["n_sources"]))

    for rule in placement_order:
        placed_records: list[dict] = []
        success = False
        for _ in range(SOURCE_LAYOUT_RETRIES):
            temp_records: list[dict] = []
            temp_occupied = set(occupied)
            temp_reserved = set(reserved)
            ok = True

            for _obj in range(rule["n_sources"]):
                candidates: list[dict] = []
                for r0 in range(h - rule["height"] + 1):
                    for c0 in range(w - rule["width"] + 1):
                        r1 = r0 + rule["height"] - 1
                        c1 = c0 + rule["width"] - 1
                        blocked = _box_cells(r0, c0, r1, c1, h, w, pad=1)
                        if blocked & (temp_occupied | temp_reserved):
                            continue
                        for diag in ("TLBR", "TRBL"):
                            markers = ((r0, c0), (r1, c1)) if diag == "TLBR" else ((r0, c1), (r1, c0))
                            candidates.append(
                                {
                                    "bbox": (r0, c0, r1, c1),
                                    "markers": markers,
                                    "blocked": blocked,
                                }
                            )

                if not candidates:
                    ok = False
                    break

                choice = random.choice(candidates[: min(len(candidates), PAIR_RETRIES_PER_OBJECT)])
                temp_records.append({"bbox": choice["bbox"], "markers": choice["markers"]})
                temp_occupied.update(choice["markers"])
                temp_reserved.update(choice["blocked"])

            if not ok:
                continue

            scratch = inp.copy()
            for rec in temp_records:
                for r, c in rec["markers"]:
                    scratch[r, c] = rule["src"]

            recovered = _find_source_pairs(
                scratch,
                rule["src"],
                bg=0,
                legend_pixels=set(),
                height=rule["height"],
                width=rule["width"],
            )
            expected = {tuple(rec["bbox"]) for rec in temp_records}
            actual = {tuple(rec["bbox"]) for rec in recovered}
            if actual != expected:
                continue

            placed_records = temp_records
            occupied.update(temp_occupied)
            reserved.update(temp_reserved)
            success = True
            break

        if not success:
            return False

        rule["sources"] = placed_records
        for rec in placed_records:
            for r, c in rec["markers"]:
                inp[r, c] = rule["src"]

    return True


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    legend_pixels: set[tuple[int, int]] = set()

    for rule in rules:
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
            for rr, cc in rule["rel"]:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & (occupied | reserved):
                continue
            if _frame_signature(np.asarray(sorted(teal_cells), dtype=int)) is None:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            reserved.update(_box_cells(min(r for r, _ in all_cells), min(c for _, c in all_cells),
                                       max(r for r, _ in all_cells), max(c for _, c in all_cells), h, w, pad=1))
            _reserve_4neighbors(reserved, all_cells)
            legend_pixels.add((sr, sc))
            legend_pixels.add((tr, tc))
            placed = True
            break
        if not placed:
            return None

    return legend_pixels


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for source in rule["sources"]:
            for rr, cc in _bbox_frame_cells(source["bbox"]):
                output[rr, cc] = rule["tgt"]
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

    if not _place_source_pairs(rules, inp, occupied, reserved):
        return None
    legend_pixels = _place_legends(rules, inp, occupied, reserved)
    if legend_pixels is None:
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, h, w)

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
