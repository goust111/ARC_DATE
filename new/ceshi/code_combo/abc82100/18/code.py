# Combo variant of abc82100 (index 18)
# combo_spec: D8/1+D15/2
# Name: 输出扩展容纳模板+网格自动扩展

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for combo D8/1 + D15/2
# a) Concrete behavior and trace:
#    The base task learns rules from local source->target legends with a nearby
#    teal stencil, then stamps that stencil onto every scattered source pixel of
#    the same color. Originally those translated stencil cells were clipped back
#    into the input frame. In this combo, clipping is replaced by automatic canvas
#    growth on every needed side. Example: on a 6x6 input, suppose one rule has
#    relative stencil cells [(-1, 0), (0, 0), (1, 1)] and another has
#    [(0, -1), (0, 0), (1, 0)]. If one scattered source sits at (0, 4) and the
#    other at (5, 0), the original solver would lose cells at (-1, 4), (1, 5),
#    and (5, -1). The combo solver instead keeps them all, so the final canvas
#    spans rows -1..6 and cols -1..5, then applies one global shift.
# b) Complexity calibration:
#    The local legend semantics stay unchanged, so humans still infer the task
#    from repeated source-color examples. The visible novelty is that outputs can
#    grow upward/downward/leftward/rightward. To prevent a degenerate dataset that
#    behaves like the original task, the generator forces overflow in at least one
#    vertical direction and one horizontal direction whenever possible.
# c) Hidden assumptions in the original solver:
#    - `out = np.full_like(g, bg)` assumed output size equals input size.
#    - The `ok = ... in bounds` mask assumed out-of-frame stencil cells should be
#      discarded rather than preserved.
#    - Preserved bystanders were written at their original coordinates, assuming
#      no global coordinate shift would ever be needed.
#    In the combo these assumptions fail together: once any translated stencil
#    crosses a boundary, the whole rendering must happen on an expanded canvas
#    whose bounds are the union of the original input rectangle and all stamped
#    coordinates.
# d) Placement legality:
#    Legends and teal components are still placed fully inside the input so rule
#    discovery remains stable. Only implied output stamps may cross the boundary.
#    Scattered sources and bystanders remain isolated with the same reservation
#    scheme, which avoids accidental extra source-target legends.
# e) Visual observability:
#    A human can still read "this color maps to that color using this teal shape"
#    directly from the legends. The combo effect is visible because the output is
#    larger than the input and the added border can appear on both axes.
# f) Visual cleanliness:
#    No new marker type is introduced. Expansion only adds black margin around the
#    original frame, and the generator keeps legends separated from scattered
#    pixels so the explanatory structure is still legible.
# g) Change inferability:
#    Color mapping remains fully explicit in the legends, so the only inferred
#    novelty is spatial: do not crop translated templates. Requiring both a row
#    side and a column side overflow keeps the fusion visually distinct from the
#    base task while avoiding too many free cases.
# Solver-side assumptions introduced by the combo:
#    - The original input rectangle must remain embedded in the output even when
#      no final colored cell lands on some of its border rows/cols.
#    - Each learned stencil is non-empty and each source color belongs to exactly
#      one rule; the generator preserves those conditions.
#    - Forced overflow samples still keep legends fully in-bounds so `_discover_rules`
#      does not need extra disambiguation logic.


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_VERT_DIRS = ("top", "bottom")
_HORZ_DIRS = ("left", "right")


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

            for dr, dc in _DIRS4:
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


def _render_expanded(
    h: int,
    w: int,
    bg: int,
    preserved_cells: list[tuple[int, int, int]],
    stamped_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    min_r = 0
    min_c = 0
    max_r = h - 1
    max_c = w - 1

    for r, c, _color in stamped_cells:
        min_r = min(min_r, int(r))
        min_c = min(min_c, int(c))
        max_r = max(max_r, int(r))
        max_c = max(max_c, int(c))

    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    row_shift = -int(min_r)
    col_shift = -int(min_c)

    for r, c, color in preserved_cells:
        out[int(r) + row_shift, int(c) + col_shift] = int(color)
    for r, c, color in stamped_cells:
        out[int(r) + row_shift, int(c) + col_shift] = int(color)
    return out


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in rel]
    cols = [int(c) for _r, c in rel]
    return min(rows), max(rows), min(cols), max(cols)


def _overflow_dirs(bounds: tuple[int, int, int, int], sr: int, sc: int, h: int, w: int) -> set[str]:
    min_rr, max_rr, min_cc, max_cc = bounds
    dirs: set[str] = set()
    if int(sr) + int(min_rr) < 0:
        dirs.add("top")
    if int(sr) + int(max_rr) >= h:
        dirs.add("bottom")
    if int(sc) + int(min_cc) < 0:
        dirs.add("left")
    if int(sc) + int(max_cc) >= w:
        dirs.add("right")
    return dirs


def _overflow_candidates(rule: dict, h: int, w: int) -> list[tuple[int, int, frozenset[str]]]:
    bounds = rule["bounds"]
    candidates: list[tuple[int, int, frozenset[str]]] = []
    for r in range(h):
        for c in range(w):
            dirs = _overflow_dirs(bounds, r, c, h, w)
            if dirs:
                candidates.append((int(r), int(c), frozenset(dirs)))
    return candidates


def _choose_required_dirs(rules: list[dict], h: int, w: int) -> set[str]:
    available: set[str] = set()
    for rule in rules:
        for _r, _c, dirs in _overflow_candidates(rule, h, w):
            available.update(dirs)

    vertical = [d for d in _VERT_DIRS if d in available]
    horizontal = [d for d in _HORZ_DIRS if d in available]
    if not vertical or not horizontal:
        return set()

    required = {random.choice(vertical), random.choice(horizontal)}
    remaining = list(available - required)
    if remaining and random.random() < 0.6:
        required.add(random.choice(remaining))
    return required


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                pos = (int(r) + dr, int(c) + dc)
                if pos not in coords:
                    frontier.add(pos)
        coords.add(random.choice(list(frontier)))

    coords_list = sorted(coords)
    best_anchors: list[tuple[int, int]] = []
    best_score = -1
    for ar, ac in coords_list:
        rel = [(int(r - ar), int(c - ac)) for r, c in coords_list]
        min_rr, max_rr, min_cc, max_cc = _rel_bounds(rel)
        score = int(min_rr < 0) + int(max_rr > 0) + int(min_cc < 0) + int(max_cc > 0)
        if score > best_score:
            best_score = score
            best_anchors = [(int(ar), int(ac))]
        elif score == best_score:
            best_anchors.append((int(ar), int(ac)))
    anchor = random.choice(best_anchors)
    return coords_list, anchor


def _build_rules(num_rules: int, area: int):
    colors = [1, 2, 3, 4, 5, 6, 7, 9]
    random.shuffle(colors)

    max_s = min(12, max(3, area // 25))
    min_s = min(max(3, area // 100), max_s)
    sc_max = min(8, max(3, area // 30))

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((int(src_c), int(tgt_c)))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "bounds": _rel_bounds(rel),
                "n_sc": random.randint(3, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(500):
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
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_forced_overflows(required_dirs, rules, inp, occupied, reserved):
    h, w = inp.shape
    uncovered = set(required_dirs)
    forced_by_rule: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(rules))}

    while uncovered:
        best = None
        best_key = None
        for idx, rule in enumerate(rules):
            candidates = _overflow_candidates(rule, h, w)
            random.shuffle(candidates)
            for r, c, dirs in candidates:
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                cover = uncovered & set(dirs)
                if not cover:
                    continue
                key = (len(cover), len(dirs), -abs(r - h // 2) - abs(c - w // 2))
                if best_key is None or key > best_key:
                    best = (idx, int(r), int(c), set(dirs))
                    best_key = key
        if best is None:
            return None

        idx, r, c, dirs = best
        forced_by_rule[idx].append((int(r), int(c)))
        occupied.add((int(r), int(c)))
        _reserve_4neighbors(reserved, [(int(r), int(c))])
        uncovered -= dirs

    return forced_by_rule


def _place_scattered(rules, inp, occupied, reserved, required_dirs):
    h, w = inp.shape
    forced_by_rule = _place_forced_overflows(required_dirs, rules, inp, occupied, reserved)
    if forced_by_rule is None:
        return False

    for idx, rule in enumerate(rules):
        positions = list(forced_by_rule[idx])
        for _ in range(rule["n_sc"] * 50):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])

        if len(positions) < 3:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in [1, 2, 3, 4, 5, 6, 7, 9] if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    count = random.randint(0, min(5, len(bystander_colors) * 2))
    placed = []
    for _ in range(count):
        for _ in range(50):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, h, w):
    preserved_cells = [(int(r), int(c), int(color)) for r, c, color in bystander_pos]
    stamped_cells: list[tuple[int, int, int]] = []
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                stamped_cells.append((int(sr) + int(rr), int(sc) + int(cc), int(rule["tgt"])))
    return _render_expanded(h, w, 0, preserved_cells, stamped_cells)


def _expansion_signature(rules, h: int, w: int) -> tuple[bool, set[str], tuple[int, int]]:
    min_r = 0
    min_c = 0
    max_r = h - 1
    max_c = w - 1
    for rule in rules:
        for sr, sc in rule["scattered"]:
            min_rr, max_rr, min_cc, max_cc = rule["bounds"]
            min_r = min(min_r, int(sr) + int(min_rr))
            max_r = max(max_r, int(sr) + int(max_rr))
            min_c = min(min_c, int(sc) + int(min_cc))
            max_c = max(max_c, int(sc) + int(max_cc))

    dirs = set()
    if min_r < 0:
        dirs.add("top")
    if max_r >= h:
        dirs.add("bottom")
    if min_c < 0:
        dirs.add("left")
    if max_c >= w:
        dirs.add("right")
    return (min_r < 0 or min_c < 0 or max_r >= h or max_c >= w), dirs, (
        int(max_r - min_r + 1),
        int(max_c - min_c + 1),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target stencil rules, then render them on an auto-expanded output canvas.

    Input structure:
    - The grid has a black background, several isolated colored pixels, and one teal
      component per rule that serves as the stencil template.
    - Each rule is shown by two 4-neighbor colored cells in a line; continuing one
      more step in the same direction reaches an anchor touching that teal stencil.

    Rule:
    - For every learned rule, find the other pixels of the same source color and
      translate the rule's teal component so the rule anchor lands on that source.
    - Recolor the translated stencil to the rule's target color and erase all legend
      pixels together with every original teal template pixel.
    - If any translated stencil would leave the input frame, expand the output grid
      on the needed side or sides instead of discarding those cells.

    Key insight: The legends still define both the source->target color mapping and
    the template shape; the extra step is to compute one global canvas large enough
    to contain the entire original frame plus every translated template cell.
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

    preserved_cells: list[tuple[int, int, int]] = []
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        preserved_cells.append((int(r), int(c), int(color)))

    stamped_cells: list[tuple[int, int, int]] = []
    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = int(sr) - int(ar)
            dc = int(sc) - int(ac)
            for pr, pc in pattern:
                stamped_cells.append((int(pr) + dr, int(pc) + dc, int(rule["target_color"])))

    return _render_expanded(h, w, int(bg), preserved_cells, stamped_cells)


def _gen_attempt():
    num_rules = random.choice([1, 1, 2, 2, 3, 3, 4])
    min_dim = {1: 5, 2: 10, 3: 14, 4: 18}[num_rules]
    h = random.randint(min_dim, 20)
    w = random.randint(min_dim, 20)

    rules, used_colors = _build_rules(num_rules, h * w)
    required_dirs = _choose_required_dirs(rules, h, w)
    if len(required_dirs) < 2:
        return None

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved, required_dirs):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    expanded, dirs, out_shape = _expansion_signature(rules, h, w)
    if not expanded:
        return None
    if not ({*dirs} & set(_VERT_DIRS) and {*dirs} & set(_HORZ_DIRS)):
        return None

    output = _build_output(rules, bystander_pos, h, w)
    if output.shape == inp.shape:
        return None
    if output.shape != out_shape:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
