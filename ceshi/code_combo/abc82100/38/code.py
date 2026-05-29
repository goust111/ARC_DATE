# Combo variant of abc82100 (index 38)
# combo_spec: D3/1+D4/0
# Name: 矩形模板+对角标记源

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# a) Concrete behavior:
#    - The legend is still local: a source-color pixel sits next to a target-color pixel, and the
#      continued direction points to a nearby teal template. The difference is that the teal object
#      is now always a filled rectangle or square, not an arbitrary component.
#    - Away from the legend, a "source" is no longer a single pixel. It is a same-color diagonal
#      corner pair whose bbox is otherwise empty. That pair defines an axis-aligned rectangle.
#    - Small trace: if a legend shows source 3 -> target 2 plus a teal 5x3 rectangle, and the grid
#      later contains two 3-pixels at opposite corners of a 3x5 empty box, the output is not two
#      target pixels. Instead, the solver centers the learned 5x3 rectangle on the marker-box
#      center, clips it by the marker box, and recolors the surviving cells to 2. Here that yields
#      a centered 3x3 or 3x5 rectangle depending on the box size.
# b) Complexity calibration:
#    - The teal template is restricted to a small set of filled odd-sized rectangles. This makes the
#      rectangular nature legible while still leaving multiple aspect ratios for humans to compare.
#    - Source boxes are also odd-sized and at least 3x3, so the recovered center is a real cell and
#      the diagonal markers visibly define a rectangle rather than a near-adjacent pair.
#    - At least one source box per generated instance is smaller than its learned template on one
#      axis, so clipping is not accidental background detail; the size relation affects the answer.
# c) Hidden assumptions from the original solver that break here:
#    - Iterating over every source-colored pixel independently is wrong because source objects are
#      diagonal marker pairs.
#    - Translating the teal component by the legend anchor is no longer the useful geometry. Since
#      both source and destination objects are rectangles, the robust invariant is bbox-center
#      alignment between the source rectangle and the teal rectangle.
#    - Copying an arbitrary teal component pixel-for-pixel is also no longer the intended rule; the
#      solver should explicitly recognize the legend teal object as a filled rectangle.
# d) Placement legality enforced by the generator:
#    - Every source pair is the only non-background content inside its bbox, and a one-cell halo is
#      reserved around that bbox to avoid ambiguous pairings or fake legends.
#    - Legend source/target pixels, their anchor gap, and the teal rectangle are kept disjoint from
#      source boxes and from one another.
#    - All learned rectangles have odd height and width, so center alignment is well-defined.
# e) Visual observability:
#    - A human can see two rectangle cues at once: the teal legend template reveals the copied size,
#      and the diagonal source markers reveal a destination rectangle whose center and clipping window
#      matter. The output keeps the template's rectangular look, so the transfer is visible.
#    - Because clipping is guaranteed to happen in generated samples, the source rectangle size is
#      not a cosmetic cue; it changes the produced rectangle.
# f) Visual cleanliness:
#    - Source boxes never overlap and keep their interiors empty in the input, which prevents teal
#      rectangles, legends, and bystanders from visually mixing with the marker cue.
#    - Only a few template aspect ratios are used, so the examples look intentional instead of noisy.
# g) Color inference:
#    - Color logic stays local and structural: the legend teaches source_color -> target_color, while
#      the diagonal pair contributes only the source color and a rectangle frame.
#    - Bystanders use colors outside all rule colors, so they remain easy to classify as untouched.
# h) Generator/solver agreement:
#    - The solver assumes legend teal components are filled odd rectangles and non-legend source
#      pixels of a rule decompose into unambiguous diagonal pairs. The generator constructs exactly
#      that regime and validates each candidate with the solver before accepting it.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _rectangle_signature(pattern: np.ndarray) -> dict | None:
    pattern = np.asarray(pattern, dtype=int)
    r0 = int(pattern[:, 0].min())
    c0 = int(pattern[:, 1].min())
    r1 = int(pattern[:, 0].max())
    c1 = int(pattern[:, 1].max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if len(pattern) != height * width:
        return None
    if (height % 2 == 0) or (width % 2 == 0):
        return None
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return {
        "bbox": (r0, c0, r1, c1),
        "center": center,
        "height": height,
        "width": width,
        "half_h": height // 2,
        "half_w": width // 2,
    }


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    pattern_meta: list[dict | None] = [_rectangle_signature(pattern) for pattern in patterns]

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
                candidates: list[tuple[int, int, int, dict]] = []
                for i, pattern in enumerate(patterns):
                    rect = pattern_meta[i]
                    if rect is None:
                        continue
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(rect["height"] * rect["width"]), int(i), rect))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[1]))
                _dist, _area, pattern_index, rect = candidates[0]
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
                        "template_height": int(rect["height"]),
                        "template_width": int(rect["width"]),
                        "template_center": tuple(rect["center"]),
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


def _paint_centered_rectangle(
    out: np.ndarray,
    *,
    center: tuple[int, int],
    height: int,
    width: int,
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    h, w = out.shape
    cr, cc = center
    half_h = height // 2
    half_w = width // 2
    rr0 = max(0, int(cr) - half_h, int(bbox[0]))
    rr1 = min(h - 1, int(cr) + half_h, int(bbox[2]))
    cc0 = max(0, int(cc) - half_w, int(bbox[1]))
    cc1 = min(w - 1, int(cc) + half_w, int(bbox[3]))
    if rr0 <= rr1 and cc0 <= cc1:
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn rectangular legend templates and stamp them inside diagonal-marker boxes.

    Input structure:
    - Black background with local colored source->target examples and teal (8) template blocks.
    - Each legend example is a 4-neighbor colored pair whose continued direction reaches an anchor
      cell touching a teal filled rectangle or square.
    - Away from the legends, each source occurrence is a same-color diagonal marker pair defining
      an otherwise empty axis-aligned rectangle.

    Rule:
    - From each legend, learn source_color -> target_color together with the size of the nearby teal
      rectangle.
    - For every matching diagonal marker pair, recover its bbox and center, center the learned
      rectangle on that source-box center, and recolor it to the legend target color.
    - Keep only the part of the centered rectangle that lies inside the source bbox and inside the
      grid, then erase all legend pixels, all teal pixels, and all source markers.

    Key insight: The legend pair identifies which color uses which teal rectangle, but the real
    transfer is rectangle-to-rectangle: the source markers define a destination box, and the teal
    template contributes a rectangular size that is center-aligned and then clipped by that box.
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
        source_pairs = _find_source_pairs(g, rule["source_color"], bg=bg, legend_pixels=legend_pixels)
        for pair in source_pairs:
            _paint_centered_rectangle(
                out,
                center=pair["center"],
                height=int(rule["template_height"]),
                width=int(rule["template_width"]),
                bbox=tuple(pair["bbox"]),
                color=int(rule["target_color"]),
            )

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 9, 2: 12, 3: 15, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

TEMPLATE_DIM_OPTIONS = [(3, 5), (5, 3), (5, 5)]
TEMPLATE_DIM_WEIGHTS = [3, 3, 2]

BYSTANDER_MAX = 4

LEGEND_RETRIES = 400
SOURCE_PAIR_RETRIES = 50
PAIR_RETRIES_PER_OBJECT = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _template_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _choose_source_count(num_rules: int, area: int) -> int:
    if num_rules == 1:
        return 3 if area >= 220 else 2
    if num_rules == 2:
        return 2 if area >= 220 else 1
    return 1


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        template_h, template_w = random.choices(TEMPLATE_DIM_OPTIONS, weights=TEMPLATE_DIM_WEIGHTS, k=1)[0]
        half_h = template_h // 2
        half_w = template_w // 2
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "template_h": template_h,
                "template_w": template_w,
                "half_h": half_h,
                "half_w": half_w,
                "half_h_choices": sorted({max(1, half_h - 1), half_h, half_h + 1}),
                "half_w_choices": sorted({max(1, half_w - 1), half_w, half_w + 1}),
                "n_sources": _choose_source_count(num_rules, area),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    legend_pixels: set[tuple[int, int]] = set()
    side_defs = ["left", "right", "top", "bottom"]

    for rule in rules:
        placed = False
        sides = side_defs[:]
        random.shuffle(sides)
        for _ in range(LEGEND_RETRIES):
            template_h = int(rule["template_h"])
            template_w = int(rule["template_w"])
            top = random.randint(0, h - template_h)
            left = random.randint(0, w - template_w)
            teal_cells = _template_cells(top, left, template_h, template_w)
            side_name = random.choice(sides)
            if side_name == "left":
                anchor = (top + template_h // 2, left - 1)
                tr, tc = anchor[0], anchor[1] - 1
                sr, sc = anchor[0], anchor[1] - 2
            elif side_name == "right":
                anchor = (top + template_h // 2, left + template_w)
                tr, tc = anchor[0], anchor[1] + 1
                sr, sc = anchor[0], anchor[1] + 2
            elif side_name == "top":
                anchor = (top - 1, left + template_w // 2)
                tr, tc = anchor[0] - 1, anchor[1]
                sr, sc = anchor[0] - 2, anchor[1]
            else:
                anchor = (top + template_h, left + template_w // 2)
                tr, tc = anchor[0] + 1, anchor[1]
                sr, sc = anchor[0] + 2, anchor[1]

            ar, ac = anchor
            if not (0 <= ar < h and 0 <= ac < w and 0 <= tr < h and 0 <= tc < w and 0 <= sr < h and 0 <= sc < w):
                continue

            all_cells = teal_cells | {anchor, (tr, tc), (sr, sc)}
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            reserved.update(_box_cells(top, left, top + template_h - 1, left + template_w - 1, h, w, pad=1))
            _reserve_4neighbors(reserved, {(sr, sc), (tr, tc), anchor})
            legend_pixels.add((sr, sc))
            legend_pixels.add((tr, tc))
            rule["anchor"] = anchor
            rule["legend_side"] = side_name
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
            -(int(rule["template_h"]) * int(rule["template_w"])),
            -int(rule["n_sources"]),
        ),
    )

    any_clipped = False

    for rule in placement_order:
        need_clipped = True
        placed_records: list[dict] = []
        success = False
        for _ in range(SOURCE_PAIR_RETRIES):
            temp_records: list[dict] = []
            temp_occupied = set(occupied)
            temp_reserved = set(reserved)
            local_clipped = False
            ok = True

            for obj_idx in range(int(rule["n_sources"])):
                candidates: list[dict] = []
                for half_h in rule["half_h_choices"]:
                    for half_w in rule["half_w_choices"]:
                        box_h = 2 * int(half_h) + 1
                        box_w = 2 * int(half_w) + 1
                        is_clipped = box_h < int(rule["template_h"]) or box_w < int(rule["template_w"])
                        if need_clipped and not local_clipped and obj_idx == int(rule["n_sources"]) - 1 and not is_clipped:
                            continue
                        for r0 in range(0, h - box_h + 1):
                            r1 = r0 + box_h - 1
                            center_r = r0 + int(half_h)
                            for c0 in range(0, w - box_w + 1):
                                c1 = c0 + box_w - 1
                                center_c = c0 + int(half_w)
                                box = _box_cells(r0, c0, r1, c1, h, w, pad=0)
                                halo = _box_cells(r0, c0, r1, c1, h, w, pad=1)
                                for diag in ("TLBR", "TRBL"):
                                    markers = ((r0, c0), (r1, c1)) if diag == "TLBR" else ((r0, c1), (r1, c0))
                                    blocked = halo | _neighbor_cells4(markers)
                                    if blocked & (temp_occupied | temp_reserved):
                                        continue
                                    candidates.append(
                                        {
                                            "bbox": (r0, c0, r1, c1),
                                            "center": (center_r, center_c),
                                            "markers": markers,
                                            "blocked": blocked,
                                            "area": box_h * box_w,
                                            "is_clipped": is_clipped,
                                            "diag": diag,
                                        }
                                    )

                if not candidates:
                    ok = False
                    break

                if need_clipped and not local_clipped:
                    clipped_candidates = [cand for cand in candidates if cand["is_clipped"]]
                    if clipped_candidates:
                        candidates = clipped_candidates

                candidates.sort(key=lambda item: (item["area"], item["bbox"], item["diag"]))
                shortlist = candidates[: min(len(candidates), PAIR_RETRIES_PER_OBJECT)]
                choice = random.choice(shortlist)
                temp_records.append(
                    {
                        "bbox": choice["bbox"],
                        "center": choice["center"],
                        "markers": choice["markers"],
                        "is_clipped": choice["is_clipped"],
                    }
                )
                temp_occupied.update(choice["markers"])
                temp_reserved.update(choice["blocked"])
                local_clipped = local_clipped or bool(choice["is_clipped"])

            if not ok or (need_clipped and not local_clipped):
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
            any_clipped = any_clipped or local_clipped
            break

        if not success:
            return False

        rule["sources"] = placed_records
        for rec in placed_records:
            for r, c in rec["markers"]:
                inp[r, c] = rule["src"]

    return any_clipped


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
            _paint_centered_rectangle(
                output,
                center=tuple(source["center"]),
                height=int(rule["template_h"]),
                width=int(rule["template_w"]),
                bbox=tuple(source["bbox"]),
                color=int(rule["tgt"]),
            )
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

    has_clipped = _place_source_pairs(rules, inp, occupied, reserved, set())
    if not has_clipped:
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
