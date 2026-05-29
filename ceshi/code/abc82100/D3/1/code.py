# Variant of abc82100
# Dimension: D3 — 源对象类型 (src)
# Change: 散落的单个彩色像素（不属于任何规则示例的非背景像素）... → 对角标记像素对作为源

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D3 variant
# a) Concrete behavior:
#    - The local legend rule is unchanged: a colored source pixel, its 4-neighbor target pixel,
#      and a nearby teal component still define source_color -> target_color plus the stencil.
#    - What changes is the "thing to transform". Away from the legend, each source occurrence is
#      no longer one pixel. It is an isolated pair of same-color diagonal corner markers. Those
#      two pixels define an empty axis-aligned rectangle. The stencil is placed once at the
#      midpoint of that rectangle, and only cells inside the rectangle are kept.
#    - Small trace: originally a rule with source color 3 and anchor-relative stencil would stamp
#      around every stray 3-pixel directly. In the variant, two 3-pixels at opposite corners of
#      a 5x5 empty box produce one stamp centered in that box. The difference is that the solver
#      must first pair the diagonal markers and recover the hidden center.
# b) Complexity calibration:
#    - The variant must visibly differ from the original, so source rectangles are at least 3x3.
#      This guarantees the stamp center is not equal to either marker pixel.
#    - To keep the task readable, the rectangle interior is empty in the input and source pairs are
#      separated by a one-cell halo. That makes the diagonal pairing a real cue rather than noise.
#    - Stencils are kept small but nontrivial (>= 3 cells) so the copied shape is recognizable.
# c) Hidden assumptions from the original solver that break here:
#    - Iterating over every source-colored pixel as an independent source is no longer valid.
#    - Translating the stencil by "source pixel - anchor" is no longer valid; the new translation
#      point is the midpoint of a recovered rectangle.
#    - Preserving every non-legend pixel whose color is not special is still valid, but source
#      marker pixels themselves must be consumed and removed from the output.
# d) Placement legality enforced by the generator:
#    - Each source rectangle contains exactly two non-background cells: its corner markers.
#    - Rectangles, legends, teal stencils, and bystanders do not overlap.
#    - Rectangles are large enough to contain the entire translated stencil, so the solver's
#      rectangle clipping is logically present but usually not visually messy.
# e) Visual observability:
#    - Humans can compare the legend stencil with output shapes and notice that each output shape
#      appears once between a same-color diagonal pair, not on top of either marker.
#    - Both TL-BR and TR-BL diagonal orientations may appear, so the rule is "pair the corners",
#      not "use a fixed corner type".
# f) Visual cleanliness:
#    - A full rectangle plus one-cell border is reserved for each source pair. This avoids stacked
#      clutter and prevents accidental fake legends or ambiguous pairings.
# g) Color inference:
#    - Color logic stays exactly as in the original task: the local legend teaches the mapping from
#      source color to target color, and the source rectangle carries only the source color.
# h) Generator/solver agreement:
#    - The solver assumes valid source pairs are the only same-color diagonal corner pairs whose
#      bounding boxes are empty except for the two markers. The generator explicitly guarantees it.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
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


def _pair_area(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (abs(int(a[0]) - int(b[0])) + 1) * (abs(int(a[1]) - int(b[1])) + 1)


def _bbox_of_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    return (
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[0]), int(b[0])),
        max(int(a[1]), int(b[1])),
    )


def _find_source_pairs(
    grid: np.ndarray,
    color: int,
    *,
    bg: int,
    legend_pixels: set[tuple[int, int]],
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
            if abs(a[0] - b[0]) < 2 or abs(a[1] - b[1]) < 2:
                continue
            if (a[0] + b[0]) % 2 or (a[1] + b[1]) % 2:
                continue

            r0, c0, r1, c1 = _bbox_of_pair(a, b)
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(np.count_nonzero(window != int(bg))) != 2:
                continue

            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "bbox": (r0, c0, r1, c1),
                    "center": ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2),
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

    chosen.sort(key=lambda item: (item["center"][0], item["center"][1], item["bbox"]))
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-pattern examples to stamp templates inside source rectangles.

    Input structure:
    - Black background with teal (8) stencil components and local colored rule examples.
    - Each rule example is the same 4-neighbor source->target pair as in the base task, with a
      continued anchor touching the teal stencil.
    - Away from the legends, source occurrences are isolated same-color diagonal marker pairs that
      mark opposite corners of an otherwise empty rectangle.

    Rule:
    - Learn source_color -> target_color together with the nearby teal stencil from each legend.
    - For every non-legend diagonal pair of that source color, recover its rectangle midpoint and
      translate the stencil so the learned anchor lands on that midpoint.
    - Recolor the translated stencil to the rule's target color, keep only cells that lie inside
      the rectangle and inside the grid, and erase all legend / teal / marker pixels.

    Key insight: The legend still teaches the color substitution and stencil shape; the new work is
    that a source is no longer a single pixel, but a diagonal corner pair whose midpoint determines
    where the stencil should appear.
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

    # Preserve ordinary colored pixels that are not source markers and not part of a legend.
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
        source_pairs = _find_source_pairs(g, rule["source_color"], bg=bg, legend_pixels=legend_pixels)
        for pair in source_pairs:
            cr, cc = pair["center"]
            r0, c0, r1, c1 = pair["bbox"]
            dr = cr - ar
            dc = cc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (
                (0 <= rows)
                & (rows < h)
                & (0 <= cols)
                & (cols < w)
                & (r0 <= rows)
                & (rows <= r1)
                & (c0 <= cols)
                & (cols <= c1)
            )
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (D3) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 7, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 5
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 28
STENCIL_MIN_PIXELS = 3

BYSTANDER_MAX = 4

LEGEND_RETRIES = 500
SOURCE_PAIR_RETRIES = 40
PAIR_RETRIES_PER_OBJECT = 60
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 350

PAIR_MARGIN_MAX = 0

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _box_cells(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 0) -> set[tuple[int, int]]:
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(h - 1, int(r1) + int(pad))
    cc1 = min(w - 1, int(c1) + int(pad))
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _neighbor_cells4(cells) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in _DIRS4:
            out.add((int(r) + dr, int(c) + dc))
    return out


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    allowed = {(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p in allowed and p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, (0, 0)


def _pattern_spans(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    max_up = max(-rr for rr, _ in rel)
    max_down = max(rr for rr, _ in rel)
    max_left = max(-cc for _, cc in rel)
    max_right = max(cc for _, cc in rel)
    return max(1, max_up), max(1, max_down), max(1, max_left), max(1, max_right)


def _choose_pair_count(num_rules: int, area: int) -> int:
    if num_rules == 1:
        if area >= 260:
            return random.randint(2, 3)
        return 2
    if num_rules == 2:
        if area >= 260:
            return 2
        return 1
    return 1


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    _ = area
    max_s = STENCIL_MAX_CAP
    min_s = STENCIL_MIN_PIXELS

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        up, down, left, right = _pattern_spans(rel)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "spans": (up, down, left, right),
                "n_sources": _choose_pair_count(num_rules, area),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    legend_pixels: set[tuple[int, int]] = set()
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
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            legend_pixels.add((sr, sc))
            legend_pixels.add((tr, tc))
            rule["anchor"] = (ar, ac)
            placed = True
            break
        if not placed:
            return None
    return legend_pixels


def _place_source_pairs(rules, inp, occupied, reserved, legend_pixels) -> bool:
    h, w = inp.shape
    placement_order = sorted(
        rules,
        key=lambda rule: (
            -(rule["spans"][0] + rule["spans"][1]),
            -(rule["spans"][2] + rule["spans"][3]),
            -int(rule["n_sources"]),
        ),
    )

    for rule in placement_order:
        placed_records: list[dict] = []
        success = False
        for _ in range(SOURCE_PAIR_RETRIES):
            temp_records: list[dict] = []
            temp_occupied = set(occupied)
            temp_reserved = set(reserved)
            ok = True

            for _obj in range(rule["n_sources"]):
                up, down, left, right = rule["spans"]
                candidates: list[dict] = []
                for add_h in range(PAIR_MARGIN_MAX + 1):
                    for add_w in range(PAIR_MARGIN_MAX + 1):
                        top = up + add_h
                        bottom = down + add_h
                        left_span = left + add_w
                        right_span = right + add_w
                        r0_max = h - (top + bottom + 1)
                        c0_max = w - (left_span + right_span + 1)
                        if r0_max < 0 or c0_max < 0:
                            continue
                        for r0 in range(r0_max + 1):
                            for c0 in range(c0_max + 1):
                                center = (r0 + top, c0 + left_span)
                                r1 = center[0] + bottom
                                c1 = center[1] + right_span
                                box = _box_cells(r0, c0, r1, c1, h, w, pad=0)
                                for diag in ("TLBR", "TRBL"):
                                    if diag == "TLBR":
                                        markers = ((r0, c0), (r1, c1))
                                    else:
                                        markers = ((r0, c1), (r1, c0))
                                    blocked = box | _neighbor_cells4(markers)
                                    if blocked & (temp_occupied | temp_reserved):
                                        continue
                                    candidates.append(
                                        {
                                            "bbox": (r0, c0, r1, c1),
                                            "center": center,
                                            "markers": markers,
                                            "blocked": blocked,
                                            "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                                        }
                                    )

                if not candidates:
                    ok = False
                    break

                candidates.sort(key=lambda item: (item["area"], item["bbox"]))
                shortlist = candidates[: min(len(candidates), PAIR_RETRIES_PER_OBJECT)]
                choice = random.choice(shortlist)
                temp_records.append(
                    {
                        "bbox": choice["bbox"],
                        "center": choice["center"],
                        "markers": choice["markers"],
                    }
                )
                temp_occupied.update(choice["markers"])
                temp_reserved.update(choice["blocked"])

            if not ok:
                continue

            scratch = inp.copy()
            for rec in temp_records:
                for r, c in rec["markers"]:
                    scratch[r, c] = rule["src"]

            recovered = _find_source_pairs(scratch, rule["src"], bg=0, legend_pixels=legend_pixels)
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
        ar, ac = rule["anchor"]
        pattern = np.asarray([(ar + rr, ac + cc) for rr, cc in rule["rel"]], dtype=int)
        for source in rule["sources"]:
            cr, cc = source["center"]
            r0, c0, r1, c1 = source["bbox"]
            dr = cr - ar
            dc = cc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (
                (0 <= rows)
                & (rows < h)
                & (0 <= cols)
                & (cols < w)
                & (r0 <= rows)
                & (rows <= r1)
                & (c0 <= cols)
                & (cols <= c1)
            )
            output[rows[ok], cols[ok]] = rule["tgt"]
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

    if not _place_source_pairs(rules, inp, occupied, reserved, set()):
        return None
    legend_pixels = _place_legends(rules, inp, occupied, reserved)
    if legend_pixels is None:
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, h, w)
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
