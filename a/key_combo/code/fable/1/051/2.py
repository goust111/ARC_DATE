from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    """Minimal ARC palette ids, kept consistent with arc_utils.py."""

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common cell value."""

    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    """Return sorted colors in the grid, optionally excluding some values."""

    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array, inclusive."""

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox of a boolean mask."""

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


Axis = tuple[str, int]


def _find_line_candidates(grid: np.ndarray, bg: int) -> tuple[int, list[Axis], np.ndarray]:
    """Detect the shared candidate-line color and all full-row/full-column axes."""

    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    specs: list[tuple[int, int, int, list[int], list[int]]] = []

    for color in colors_present(g, exclude={bg}):
        rows = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        cols = [c for c in range(w) if bool(np.all(g[:, c] == color))]
        count = len(rows) + len(cols)
        if count < 2:
            continue
        line_mask = np.zeros_like(g, dtype=bool)
        if rows:
            line_mask[rows, :] = True
        if cols:
            line_mask[:, cols] = True
        extra = int(np.sum(g == color)) - int(line_mask.sum())
        specs.append((extra, -count, int(color), rows, cols))

    assert specs, "expected at least two candidate full lines"
    specs.sort()
    extra, _neg_count, line_color, rows, cols = specs[0]
    assert extra == 0, "candidate line color should not appear in the motif"

    candidates: list[Axis] = [("h", int(r)) for r in rows] + [("v", int(c)) for c in cols]
    line_mask = np.zeros_like(g, dtype=bool)
    for axis in candidates:
        if axis[0] == "h":
            line_mask[axis[1], :] = True
        else:
            line_mask[:, axis[1]] = True
    return int(line_color), candidates, line_mask


def _evaluate_axis(grid: np.ndarray, bg: int, line_mask: np.ndarray, axis: Axis) -> dict:
    """Score a candidate axis by occupancy symmetry and color-bijection consistency."""

    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    motif = (g != bg) & (~line_mask)
    orient, idx = axis

    pair_count = 0
    single_count = 0
    l_to_r: dict[int, int] = {}
    r_to_l: dict[int, int] = {}
    consistent = True
    pending: list[tuple[int | None, int | None, int, bool]] = []

    if orient == "v":
        max_d = max(int(idx), int(w - 1 - idx))
        for d in range(1, max_d + 1):
            lc = idx - d
            rc = idx + d
            left_in = lc >= 0
            right_in = rc < w
            for r in range(h):
                left_occ = bool(left_in and motif[r, lc])
                right_occ = bool(right_in and motif[r, rc])
                if left_occ and right_occ:
                    pair_count += 1
                    left_color = int(g[r, lc])
                    right_color = int(g[r, rc])
                    if left_color in l_to_r and l_to_r[left_color] != right_color:
                        consistent = False
                    if right_color in r_to_l and r_to_l[right_color] != left_color:
                        consistent = False
                    l_to_r[left_color] = right_color
                    r_to_l[right_color] = left_color
                elif left_occ:
                    single_count += 1
                    pending.append((r, rc if right_in else None, int(g[r, lc]), True))
                elif right_occ:
                    single_count += 1
                    pending.append((r, lc if left_in else None, int(g[r, rc]), False))
    else:
        max_d = max(int(idx), int(h - 1 - idx))
        for d in range(1, max_d + 1):
            tr = idx - d
            br = idx + d
            top_in = tr >= 0
            bottom_in = br < h
            for c in range(w):
                top_occ = bool(top_in and motif[tr, c])
                bottom_occ = bool(bottom_in and motif[br, c])
                if top_occ and bottom_occ:
                    pair_count += 1
                    top_color = int(g[tr, c])
                    bottom_color = int(g[br, c])
                    if top_color in l_to_r and l_to_r[top_color] != bottom_color:
                        consistent = False
                    if bottom_color in r_to_l and r_to_l[bottom_color] != top_color:
                        consistent = False
                    l_to_r[top_color] = bottom_color
                    r_to_l[bottom_color] = top_color
                elif top_occ:
                    single_count += 1
                    pending.append((br if bottom_in else None, c, int(g[tr, c]), True))
                elif bottom_occ:
                    single_count += 1
                    pending.append((tr if top_in else None, c, int(g[br, c]), False))

    fillable = consistent
    fills: list[tuple[int, int, int]] = []
    for maybe_r, maybe_c, source_color, forward in pending:
        if maybe_r is None or maybe_c is None:
            fillable = False
            continue
        if int(g[maybe_r, maybe_c]) != bg:
            fillable = False
            continue
        mapping = l_to_r if forward else r_to_l
        if source_color not in mapping:
            fillable = False
            continue
        fills.append((int(maybe_r), int(maybe_c), int(mapping[source_color])))

    orbit_total = pair_count + single_count
    quality = pair_count / orbit_total if orbit_total else 0.0
    mapping_size = len(l_to_r)
    valid = bool(
        consistent
        and fillable
        and mapping_size >= 2
        and pair_count >= 6
        and orbit_total >= 8
    )
    return {
        "axis": axis,
        "pairs": int(pair_count),
        "singles": int(single_count),
        "quality": float(quality),
        "mapping_size": int(mapping_size),
        "mapping": dict(l_to_r),
        "inverse_mapping": dict(r_to_l),
        "consistent": bool(consistent),
        "fillable": bool(fillable),
        "fills": fills,
        "valid": valid,
    }


def _axis_rank(stats: dict) -> tuple[int, int, int, int, int]:
    """Ranking tuple used to pick the unique true axis."""

    return (
        int(bool(stats["valid"])),
        int(round(stats["quality"] * 1000)),
        int(stats["pairs"]),
        -int(stats["singles"]),
        int(stats["mapping_size"]),
    )


def _select_true_axis(grid: np.ndarray) -> tuple[int, list[Axis], np.ndarray, Axis, dict[Axis, dict]]:
    """Find the unique true axis among all candidate full lines."""

    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    line_color, candidates, line_mask = _find_line_candidates(g, bg)
    stats_by_axis = {axis: _evaluate_axis(g, bg, line_mask, axis) for axis in candidates}
    ordered = sorted(candidates, key=lambda axis: (_axis_rank(stats_by_axis[axis]), axis[0], axis[1]), reverse=True)
    true_axis = ordered[0]
    assert stats_by_axis[true_axis]["valid"], "expected one valid symmetric axis"
    if len(ordered) > 1:
        assert _axis_rank(stats_by_axis[ordered[0]]) > _axis_rank(stats_by_axis[ordered[1]]), "true axis must be unique"
    return line_color, candidates, line_mask, true_axis, stats_by_axis


def _apply_rule_from_stats(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    candidates: list[Axis],
    true_axis: Axis,
    stats_by_axis: dict[Axis, dict],
) -> np.ndarray:
    """Erase decoy lines, keep the true axis, and fill single-sided missing motif cells."""

    out = np.asarray(grid, dtype=int).copy()

    for axis in candidates:
        if axis == true_axis:
            continue
        if axis[0] == "h":
            out[axis[1], :] = bg
            if true_axis[0] == "v":
                out[axis[1], true_axis[1]] = line_color
        else:
            out[:, axis[1]] = bg
            if true_axis[0] == "h":
                out[true_axis[1], axis[1]] = line_color

    if true_axis[0] == "h":
        out[true_axis[1], :] = line_color
    else:
        out[:, true_axis[1]] = line_color

    for r, c, color in stats_by_axis[true_axis]["fills"]:
        out[r, c] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the unique shape-symmetric full line, infer the cross-axis color permutation, and complete one-sided holes.

    Input structure:
    - A black background contains one multicolor foreground motif plus several same-color
      full candidate lines, each spanning an entire row or an entire column.
    - The candidate-line rows or columns contain no motif pixels beyond the line color itself.
    - Exactly one candidate line is the true mirror axis of the motif's occupancy shape.

    Rule:
    - Test each candidate line on occupancy only, ignoring colors, and locate the unique axis
      whose two sides are mirror-matched except for a few single-sided missing cells.
    - From intact mirrored pixel pairs across that axis, infer a consistent bijection from one
      side's colors to the other side's colors.
    - Erase every decoy full line to background, keep the true axis, and fill each mirrored
      missing background cell using the inferred color permutation.

    Key insight:
    - The real symmetry is hidden at the shape level: colors differ across the axis by a learned
      permutation, so the solver must ignore color while finding the axis and only then use color
      correspondence to restore the missing cells.
    """

    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    line_color, candidates, _line_mask, true_axis, stats_by_axis = _select_true_axis(g)
    return _apply_rule_from_stats(g, bg, line_color, candidates, true_axis, stats_by_axis)


GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (18, 28)
MOTIF_HEIGHT_RANGE = (7, 13)
MOTIF_HALF_WIDTH_RANGE = (4, 7)


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int = 2) -> list[int] | None:
    """Pick `count` distinct integer positions separated by at least `gap`."""

    if count == 0:
        return []
    vals = list(range(lo, hi + 1))
    random.shuffle(vals)
    chosen: list[int] = []
    for v in vals:
        if all(abs(v - u) >= gap for u in chosen):
            chosen.append(v)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _grow_left_mask(height: int, half_width: int) -> np.ndarray | None:
    """Grow an irregular connected half-shape with enough structure to reject shortcuts."""

    area = height * half_width
    target_lo = max(14, area // 3)
    target_hi = max(target_lo, min(area - 1, (2 * area) // 3))

    for _ in range(80):
        target = random.randint(target_lo, target_hi)
        mask = np.zeros((height, half_width), dtype=bool)
        start = (random.randrange(height), random.randrange(half_width))
        mask[start] = True

        while int(mask.sum()) < target:
            occ = np.argwhere(mask)
            sr, sc = occ[random.randrange(len(occ))]
            dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            nr, nc = int(sr + dr), int(sc + dc)
            if 0 <= nr < height and 0 <= nc < half_width:
                mask[nr, nc] = True

        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        row_sums = mask.sum(axis=1)
        col_sums = mask.sum(axis=0)
        if box_h < max(5, height - 3):
            continue
        if box_w < min(3, half_width):
            continue
        if len(set(map(int, row_sums))) < 3:
            continue
        if len(set(map(int, col_sums))) < 2:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray | None:
    """Assign blocky color regions on the left half by nearest-seed Voronoi partitioning."""

    coords = np.argwhere(mask)
    k = len(palette)
    if len(coords) < 3 * k:
        return None

    for _ in range(40):
        seed_idx = random.sample(range(len(coords)), k)
        seeds = coords[seed_idx]
        patch = np.zeros(mask.shape, dtype=int)
        counts = Counter()
        for rr, cc in coords:
            best = min(
                range(k),
                key=lambda i: (abs(int(rr) - int(seeds[i, 0])) + abs(int(cc) - int(seeds[i, 1])), i),
            )
            color = int(palette[best])
            patch[int(rr), int(cc)] = color
            counts[color] += 1
        if min(counts.values()) >= 3:
            return patch
    return None


def _choose_missing_orbits(left_patch: np.ndarray) -> list[tuple[int, int, str]] | None:
    """Remove a few mirrored orbit halves while keeping every color inferable from intact pairs."""

    occupied = [(int(r), int(c), int(left_patch[r, c])) for r, c in np.argwhere(left_patch > 0)]
    counts = Counter(color for _r, _c, color in occupied)
    max_missing = min(4, max(1, len(occupied) // 6))
    want = random.randint(1, max_missing)

    if want >= 2 and random.random() < 0.7:
        sides = ["left", "right"] + [random.choice(["left", "right"]) for _ in range(want - 2)]
    else:
        sides = [random.choice(["left", "right"]) for _ in range(want)]
    random.shuffle(sides)

    used: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int, str]] = []
    for side in sides:
        random.shuffle(occupied)
        picked = False
        for r, c, color in occupied:
            if (r, c) in used:
                continue
            if counts[color] <= 1:
                continue
            used.add((r, c))
            counts[color] -= 1
            chosen.append((r, c, side))
            picked = True
            break
        if not picked:
            return None
    return chosen


def _build_expected_from_truth(
    grid: np.ndarray,
    axis: Axis,
    line_color: int,
    candidates: list[Axis],
    left_to_right: dict[int, int],
    missing: list[tuple[int, int, str]],
    top: int,
) -> np.ndarray:
    """Construct the target output directly from planted ground truth."""

    out = np.asarray(grid, dtype=int).copy()
    bg = Color.BLACK
    orient, idx = axis
    inverse = {right: left for left, right in left_to_right.items()}

    for cand in candidates:
        if cand == axis:
            continue
        if cand[0] == "h":
            out[cand[1], :] = bg
            if axis[0] == "v":
                out[cand[1], idx] = line_color
        else:
            out[:, cand[1]] = bg
            if axis[0] == "h":
                out[idx, cand[1]] = line_color

    if orient == "v":
        out[:, idx] = line_color
        for r_local, d_local, side in missing:
            r = top + r_local
            lc = idx - (d_local + 1)
            rc = idx + (d_local + 1)
            if side == "left":
                out[r, lc] = inverse[int(grid[r, rc])]
            else:
                out[r, rc] = left_to_right[int(grid[r, lc])]
    else:
        out[idx, :] = line_color
        for d_local, c_local, side in missing:
            tr = idx - (d_local + 1)
            br = idx + (d_local + 1)
            c = top + c_local
            if side == "left":
                out[tr, c] = inverse[int(grid[br, c])]
            else:
                out[br, c] = left_to_right[int(grid[tr, c])]

    return out


def _transpose_truth(
    grid: np.ndarray,
    expected: np.ndarray,
    axis: Axis,
    candidates: list[Axis],
    missing: list[tuple[int, int, str]],
    top: int,
) -> tuple[np.ndarray, np.ndarray, Axis, list[Axis], list[tuple[int, int, str]], int]:
    """Rotate the canonical vertical construction into a horizontal one via transpose."""

    assert axis[0] == "v"
    transposed_grid = grid.T.copy()
    transposed_expected = expected.T.copy()
    transposed_axis = ("h", axis[1])
    transposed_candidates = [("h", pos) for _orient, pos in candidates]
    transposed_missing = [(d_local, r_local, side) for r_local, d_local, side in missing]
    return transposed_grid, transposed_expected, transposed_axis, transposed_candidates, transposed_missing, top


def _generate_vertical_instance() -> tuple[np.ndarray, np.ndarray, Axis, list[Axis], int]:
    """Generate one vertical-axis scene plus its independently constructed output."""

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    half_width = random.randint(*MOTIF_HALF_WIDTH_RANGE)
    motif_height = random.randint(*MOTIF_HEIGHT_RANGE)
    motif_height = min(motif_height, height - 4)
    if motif_height < MOTIF_HEIGHT_RANGE[0]:
        raise RuntimeError("motif height collapsed below minimum")

    axis_col_lo = half_width + 4
    axis_col_hi = width - half_width - 5
    if axis_col_lo > axis_col_hi:
        raise RuntimeError("canvas too small for a central axis and decoys")
    axis_col = random.randint(axis_col_lo, axis_col_hi)
    top = random.randint(2, height - motif_height - 2)

    extra_side = random.choice(["left", "right", None])
    left_need = 2 if extra_side == "left" else 1
    right_need = 2 if extra_side == "right" else 1
    left_cols = _sample_spaced_positions(1, axis_col - half_width - 2, left_need)
    right_cols = _sample_spaced_positions(axis_col + half_width + 2, width - 2, right_need)
    if left_cols is None or right_cols is None:
        raise RuntimeError("failed to place decoy columns")

    line_color = random.choice(Color.NOT_BLACK)
    remaining = [c for c in Color.NOT_BLACK if c != line_color]
    mask = _grow_left_mask(motif_height, half_width)
    if mask is None:
        raise RuntimeError("failed to grow left mask")

    area = int(mask.sum())
    color_count = random.randint(2, min(4, max(2, area // 5)))
    chosen_colors = random.sample(remaining, 2 * color_count)
    left_palette = chosen_colors[:color_count]
    right_palette = chosen_colors[color_count:]
    random.shuffle(right_palette)
    left_to_right = {int(left): int(right) for left, right in zip(left_palette, right_palette)}

    left_patch = _colorize_mask(mask, left_palette)
    if left_patch is None:
        raise RuntimeError("failed to colorize mask")

    missing = _choose_missing_orbits(left_patch)
    if missing is None:
        raise RuntimeError("failed to choose removable orbits")
    missing_lookup = {(int(r), int(c)): side for r, c, side in missing}

    grid = np.full((height, width), Color.BLACK, dtype=int)
    candidates: list[Axis] = [("v", col) for col in left_cols + [axis_col] + right_cols]
    for _orient, col in candidates:
        grid[:, col] = line_color

    for r_local, d_local in np.argwhere(left_patch > 0):
        left_color = int(left_patch[r_local, d_local])
        right_color = int(left_to_right[left_color])
        row = top + int(r_local)
        lc = axis_col - (int(d_local) + 1)
        rc = axis_col + (int(d_local) + 1)
        side = missing_lookup.get((int(r_local), int(d_local)))
        if side != "left":
            grid[row, lc] = left_color
        if side != "right":
            grid[row, rc] = right_color

    expected = _build_expected_from_truth(
        grid=grid,
        axis=("v", axis_col),
        line_color=line_color,
        candidates=candidates,
        left_to_right=left_to_right,
        missing=missing,
        top=top,
    )
    return grid, expected, ("v", axis_col), candidates, len(missing)


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    planted_axis: Axis,
    planted_candidates: list[Axis],
    missing_count: int,
) -> bool:
    """Check semantics, uniqueness, and non-degeneracy beyond a plain closed loop."""

    g = np.asarray(grid, dtype=int)
    out = np.asarray(expected, dtype=int)
    if g.ndim != 2 or out.ndim != 2 or max(g.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(g, out):
        return False

    bg = get_bg(g)
    try:
        line_color, candidates, line_mask, true_axis, stats_by_axis = _select_true_axis(g)
    except AssertionError:
        return False

    if candidates != planted_candidates:
        return False
    if true_axis != planted_axis:
        return False
    if len(candidates) < 3:
        return False

    true_stats = stats_by_axis[true_axis]
    if true_stats["singles"] != missing_count:
        return False
    if len(true_stats["fills"]) != missing_count:
        return False
    if true_stats["quality"] < 0.75:
        return False
    if true_stats["mapping_size"] < 2:
        return False

    ranks = sorted((_axis_rank(stats), axis) for axis, stats in stats_by_axis.items())
    if len(ranks) >= 2 and not (ranks[-1][0] > ranks[-2][0]):
        return False

    for axis, stats in stats_by_axis.items():
        if axis == true_axis:
            continue
        if stats["quality"] >= 0.6:
            return False
        if stats["pairs"] >= true_stats["pairs"]:
            return False

    motif_box = bbox((g != bg) & (~line_mask))
    if motif_box is None:
        return False
    r0, c0, r1, c1 = motif_box
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 8:
        return False

    line_cells = np.zeros_like(g, dtype=bool)
    for axis in candidates:
        if axis[0] == "h":
            line_cells[axis[1], :] = True
        else:
            line_cells[:, axis[1]] = True
    added_fill = bool(np.any((g == bg) & (out != bg) & (~line_cells)))
    removed_decoy = any(
        np.all(out[row, :] == bg) if orient == "h" and axis != true_axis else False
        for axis, (orient, row) in zip(candidates, candidates)
    ) or any(
        np.all(out[:, col] == bg) if orient == "v" and axis != true_axis else False
        for axis, (orient, col) in zip(candidates, candidates)
    )
    if not added_fill or not removed_decoy:
        return False

    try:
        solved = main(g)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    """Generate varied nondegenerate inputs for the unique-axis color-permutation puzzle."""

    for _ in range(GEN_RETRIES):
        try:
            grid, expected, axis, candidates, missing_count = _generate_vertical_instance()
        except RuntimeError:
            continue

        if random.random() < 0.5:
            grid, expected, axis, candidates, _unused_missing, _unused_top = _transpose_truth(
                grid=grid,
                expected=expected,
                axis=axis,
                candidates=candidates,
                missing=[],
                top=0,
            )
        if _validate_instance(grid, expected, axis, candidates, missing_count):
            return grid

    raise RuntimeError("failed to generate a valid unique-axis permutation-symmetry input")
