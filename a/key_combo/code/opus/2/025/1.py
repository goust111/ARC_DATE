import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
PATTERN_COLORS = (BLUE, RED, GREEN)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _pattern_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != ORANGE)


def _mirror_col(axis: int, c: int) -> int:
    return 2 * int(axis) - int(c)


def _axis_scores(grid: np.ndarray) -> list[tuple[int, int, int, float]]:
    g = np.asarray(grid, dtype=int)
    mask = _pattern_mask(g)
    box = _bbox(mask)
    if box is None:
        return [(0, 0, 0, 0.0)] * g.shape[1]
    center = (box[1] + box[3]) / 2.0
    coords = np.argwhere(mask)
    scores: list[tuple[int, int, int, float]] = []
    for axis in range(g.shape[1]):
        matches = 0
        mismatches = 0
        paired = 0
        for r, c in coords:
            mc = _mirror_col(axis, int(c))
            if 0 <= mc < g.shape[1] and mask[int(r), int(mc)]:
                paired += 1
                if int(g[int(r), int(c)]) == int(g[int(r), int(mc)]):
                    matches += 1
                else:
                    mismatches += 1
        scores.append((int(matches), -int(mismatches), int(paired), -abs(axis - center)))
    return scores


def _infer_axis(grid: np.ndarray) -> int:
    scores = _axis_scores(grid)
    return int(max(range(len(scores)), key=lambda axis: scores[axis]))


def _ray_candidates(
    grid: np.ndarray, marker: tuple[int, int], box: tuple[int, int, int, int]
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = box
    mr, mc = marker
    candidates: list[dict] = []
    for dr, dc in DIRS:
        rr = mr + dr
        cc = mc + dc
        gap = 0
        while 0 <= rr < h and 0 <= cc < w:
            val = int(g[rr, cc])
            if val == BLACK:
                gap += 1
                rr += dr
                cc += dc
                continue
            if (
                val != ORANGE
                and gap > 0
                and r0 <= rr <= r1
                and c0 <= cc <= c1
                and (rr in (r0, r1) or cc in (c0, c1))
            ):
                candidates.append(
                    {
                        "marker": (int(mr), int(mc)),
                        "direction": (int(dr), int(dc)),
                        "distance": int(gap),
                        "hit": (int(rr), int(cc)),
                    }
                )
            break
    return candidates


def _match_markers(grid: np.ndarray) -> tuple[tuple[int, int, int, int] | None, list[dict]]:
    g = np.asarray(grid, dtype=int)
    box = _bbox(_pattern_mask(g))
    if box is None:
        return None, []
    markers = sorted(map(tuple, np.argwhere(g == ORANGE)))
    matches: list[dict] = []
    for marker in markers:
        candidates = _ray_candidates(g, marker, box)
        if not candidates:
            continue
        candidates.sort(
            key=lambda cand: (
                cand["distance"],
                DIRS.index(cand["direction"]),
                cand["hit"][0],
                cand["hit"][1],
            )
        )
        matches.append(candidates[0])
    return box, matches


def _paint_ray(out: np.ndarray, marker: tuple[int, int], hit: tuple[int, int]) -> None:
    mr, mc = marker
    hr, hc = hit
    dr = 0 if hr == mr else (1 if hr > mr else -1)
    dc = 0 if hc == mc else (1 if hc > mc else -1)
    rr = mr + dr
    cc = mc + dc
    while (rr, cc) != (hr, hc):
        out[int(rr), int(cc)] = ORANGE
        rr += dr
        cc += dc


def _repair_from_hit(
    base: np.ndarray,
    out: np.ndarray,
    box: tuple[int, int, int, int],
    axis: int,
    hit: tuple[int, int],
    distance: int,
) -> None:
    hr, hc = hit
    r0, c0, r1, c1 = box
    start_c = _mirror_col(axis, hc)
    if not (c0 <= start_c <= c1):
        return
    step = 1 if start_c > axis else -1 if start_c < axis else 0
    if step == 0:
        return
    cur_c = int(start_c)
    for _ in range(int(distance)):
        if not (r0 <= hr <= r1 and c0 <= cur_c <= c1):
            break
        if int(out[int(hr), int(cur_c)]) != BLACK:
            break
        src_c = _mirror_col(axis, cur_c)
        src_val = int(base[int(hr), int(src_c)])
        if src_val in (BLACK, ORANGE):
            break
        out[int(hr), int(cur_c)] = int(src_val)
        cur_c += step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each orange sightline and use its gap length to control partial symmetric repair.

    Input structure:
    - Black background with a central multicolor pattern using blue/red/green pixels.
    - The pattern is close to left-right symmetric around an unknown vertical axis, but several boundary runs were erased to black.
    - Orange singleton markers sit outside the pattern bbox with a straight black sightline to one boundary pixel.

    Rule:
    - Infer the vertical symmetry axis from the surviving non-black, non-orange pixels by maximizing mirrored color matches.
    - For each orange marker, find the shortest valid cardinal sightline to a pattern boundary pixel and fill only the black gap cells on that ray with orange.
    - Let the ray gap length be d; mirror the hit boundary pixel across the inferred axis, then restore up to d consecutive black pixels from that mirrored position outward, copying colors from their symmetric counterparts.

    Key insight:
    - The markers do not request full completion: each ray's visible spacing encodes exactly how deep the opposite-side symmetric repair should go.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    box, matches = _match_markers(g)
    if box is None:
        return out
    axis = _infer_axis(g)
    for match in matches:
        _paint_ray(out, match["marker"], match["hit"])
    for match in matches:
        _repair_from_hit(g, out, box, axis, match["hit"], match["distance"])
    return out


def _make_pattern(height: int, width: int) -> np.ndarray:
    axis = width // 2
    for _ in range(200):
        left = np.random.choice(PATTERN_COLORS, size=(height, axis), replace=True)
        center = np.random.choice(PATTERN_COLORS, size=(height, 1), replace=True)
        pattern = np.zeros((height, width), dtype=int)
        pattern[:, :axis] = left
        pattern[:, axis : axis + 1] = center
        pattern[:, axis + 1 :] = left[:, ::-1]
        if len(set(map(int, pattern.ravel()))) == len(PATTERN_COLORS):
            return pattern
    raise RuntimeError("failed to build symmetric pattern")


def _available_length(side: str, width: int, source_col: int) -> int:
    if side == "left":
        return int(source_col + 1)
    return int(width - source_col)


def _build_single_spec(
    side: str, row_local: int, width: int, distance: int, need_partial: bool
) -> tuple[list[dict], bool] | None:
    axis = width // 2
    if side == "left":
        cols = list(range(1, axis))
        cols.sort(key=lambda c: abs(c - axis))
    else:
        cols = list(range(axis + 1, width - 1))
        cols.sort(key=lambda c: abs(c - axis))
    random.shuffle(cols)
    cols.sort(key=lambda c: abs(c - axis))

    usable = []
    for source_col in cols:
        max_len = min(5, _available_length(side, width, source_col))
        if max_len >= 2:
            usable.append((source_col, max_len))
    if not usable:
        return None

    partial_choices = [(c, m) for c, m in usable if m > distance]
    if need_partial and partial_choices:
        source_col, max_len = random.choice(partial_choices)
        length = random.randint(max(2, distance + 1), max_len)
    else:
        source_col, max_len = random.choice(usable)
        length = random.randint(2, max_len)

    dest_start = _mirror_col(axis, source_col)
    step = 1 if dest_start > axis else -1
    spec = {
        "row_local": int(row_local),
        "side": side,
        "source_col": int(source_col),
        "dest_start": int(dest_start),
        "step": int(step),
        "distance": int(distance),
        "length": int(length),
    }
    return [spec], bool(length > distance)


def _build_pair_specs(
    side: str, row_local: int, width: int, distances: list[int], need_partial: bool
) -> tuple[list[dict], bool] | None:
    axis = width // 2
    d_small, d_large = sorted(map(int, distances))
    near_col = axis - 1 if side == "left" else axis + 1
    edge_col = 1 if side == "left" else width - 2

    near_max = min(3, min(5, _available_length(side, width, near_col)))
    if near_max < 2:
        return None
    if need_partial and near_max > d_small:
        near_len = random.randint(max(2, d_small + 1), near_max)
    else:
        near_len = random.randint(2, near_max)
    edge_len = 2

    specs = []
    for source_col, distance, length in (
        (near_col, d_small, near_len),
        (edge_col, d_large, edge_len),
    ):
        dest_start = _mirror_col(axis, source_col)
        step = 1 if dest_start > axis else -1
        specs.append(
            {
                "row_local": int(row_local),
                "side": side,
                "source_col": int(source_col),
                "dest_start": int(dest_start),
                "step": int(step),
                "distance": int(distance),
                "length": int(length),
            }
        )
    return specs, bool(near_len > d_small)


def _build_row_specs(
    side: str, row_local: int, width: int, distances: list[int], need_partial: bool
) -> tuple[list[dict], bool] | None:
    if len(distances) == 1:
        return _build_single_spec(side, row_local, width, int(distances[0]), need_partial)
    if len(distances) == 2:
        return _build_pair_specs(side, row_local, width, distances, need_partial)
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    pattern_h = random.randint(7, 9)
    pattern_w = 11
    top_pad = random.randint(5, 6)
    bottom_pad = random.randint(5, 6)
    left_pad = random.randint(2, 4)
    right_pad = random.randint(2, 4)
    n_markers = random.randint(2, 4)

    if n_markers == 2:
        top_count, bottom_count = 1, 1
    elif n_markers == 3:
        top_count, bottom_count = random.choice([(1, 2), (2, 1)])
    else:
        top_count, bottom_count = 2, 2

    top_side = random.choice(["left", "right"])
    bottom_side = "right" if top_side == "left" else "left"

    max_distance = min(top_pad - 1, bottom_pad - 1)
    if max_distance < n_markers:
        return None
    distances = [1]
    distances.extend(random.sample(list(range(2, max_distance + 1)), n_markers - 1))

    partial_row = random.choice(["top", "bottom"])
    remaining = [d for d in distances if d != 1]
    if partial_row == "top":
        top_distances = [1]
        if top_count > 1:
            chosen = random.sample(remaining, top_count - 1)
            top_distances.extend(chosen)
            for d in chosen:
                remaining.remove(d)
        bottom_distances = remaining[:]
    else:
        bottom_distances = [1]
        if bottom_count > 1:
            chosen = random.sample(remaining, bottom_count - 1)
            bottom_distances.extend(chosen)
            for d in chosen:
                remaining.remove(d)
        top_distances = remaining[:]
    random.shuffle(top_distances)
    random.shuffle(bottom_distances)

    top_built = _build_row_specs(top_side, 0, pattern_w, top_distances, partial_row == "top")
    bottom_built = _build_row_specs(
        bottom_side, pattern_h - 1, pattern_w, bottom_distances, partial_row == "bottom"
    )
    if top_built is None or bottom_built is None:
        return None
    top_specs, top_partial = top_built
    bottom_specs, bottom_partial = bottom_built
    specs = top_specs + bottom_specs
    if not (top_partial or bottom_partial):
        return None

    pattern = _make_pattern(pattern_h, pattern_w)
    grid_h = top_pad + pattern_h + bottom_pad
    grid_w = left_pad + pattern_w + right_pad
    r0 = top_pad
    c0 = left_pad
    full_grid = np.zeros((grid_h, grid_w), dtype=int)
    full_grid[r0 : r0 + pattern_h, c0 : c0 + pattern_w] = pattern
    inp = full_grid.copy()

    marker_map: dict[tuple[int, int], dict] = {}
    for spec in specs:
        row_g = r0 + spec["row_local"]
        hit_col_g = c0 + spec["source_col"]
        if spec["row_local"] == 0:
            marker = (row_g - spec["distance"] - 1, hit_col_g)
        else:
            marker = (row_g + spec["distance"] + 1, hit_col_g)
        if not (0 <= marker[0] < grid_h and 0 <= marker[1] < grid_w):
            return None
        spec["marker"] = (int(marker[0]), int(marker[1]))
        spec["hit"] = (int(row_g), int(hit_col_g))
        marker_map[spec["marker"]] = spec

        cells = []
        for k in range(spec["length"]):
            col_local = spec["dest_start"] + spec["step"] * k
            if not (0 <= col_local < pattern_w):
                return None
            cells.append((row_g, c0 + col_local))
        spec["missing_cells"] = cells

    if len(marker_map) != len(specs):
        return None

    for spec in specs:
        for rr, cc in spec["missing_cells"]:
            inp[int(rr), int(cc)] = BLACK
        mr, mc = spec["marker"]
        if inp[int(mr), int(mc)] != BLACK:
            return None
        inp[int(mr), int(mc)] = ORANGE

    expected = inp.copy()
    axis_global = c0 + pattern_w // 2
    for spec in specs:
        _paint_ray(expected, spec["marker"], spec["hit"])
    for spec in specs:
        row_g = int(spec["hit"][0])
        for k in range(min(spec["distance"], spec["length"])):
            dest_col = c0 + spec["dest_start"] + spec["step"] * k
            src_col = _mirror_col(axis_global, dest_col)
            expected[row_g, dest_col] = full_grid[row_g, src_col]

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    scores = _axis_scores(inp)
    match_counts = [score[0] for score in scores]
    if match_counts.count(max(match_counts)) != 1:
        return None
    if _infer_axis(inp) != axis_global:
        return None

    box, matches = _match_markers(inp)
    if box != (r0, c0, r0 + pattern_h - 1, c0 + pattern_w - 1):
        return None
    if len(matches) != len(specs):
        return None
    if len({spec["distance"] for spec in specs}) != len(specs):
        return None

    for match in matches:
        spec = marker_map.get(match["marker"])
        if spec is None:
            return None
        candidates = _ray_candidates(inp, match["marker"], box)
        if not candidates:
            return None
        best_dist = min(cand["distance"] for cand in candidates)
        if sum(cand["distance"] == best_dist for cand in candidates) != 1:
            return None
        if match["distance"] != spec["distance"] or match["hit"] != spec["hit"]:
            return None

    restored = int(np.sum((_pattern_mask(expected)) & (inp == BLACK)))
    remaining_missing = int(np.sum((_pattern_mask(full_grid)) & (expected == BLACK)))
    added_ray = int(np.sum((expected == ORANGE) & (inp == BLACK)))
    if restored <= 0 or remaining_missing <= 0 or added_ray <= 0:
        return None

    shortcut = inp.copy()
    for spec in specs:
        _paint_ray(shortcut, spec["marker"], spec["hit"])
        for rr, cc in spec["missing_cells"]:
            src_col = _mirror_col(axis_global, cc)
            shortcut[int(rr), int(cc)] = full_grid[int(rr), int(src_col)]
    if np.array_equal(shortcut, expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_case()
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError("failed to generate a valid input")
