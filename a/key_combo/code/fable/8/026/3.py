from __future__ import annotations

import random
from collections import Counter

import numpy as np


ORTHO4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_MAX_ATTEMPTS = 600
PALETTE = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _component_dfs(
    grid: np.ndarray,
    start: tuple[int, int],
    seen: np.ndarray,
) -> dict:
    color = int(grid[start])
    stack = [start]
    coords = []
    seen[start] = True
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in ORTHO4:
            nr, nc = r + dr, c + dc
            if _inside(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                stack.append((nr, nc))
    arr = np.array(coords, dtype=int)
    return {"color": color, "coords": arr, "bbox": _bbox(arr), "area": int(arr.shape[0])}


def _mono_components(grid: np.ndarray, colors: set[int] | None = None) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    allowed = None if colors is None else set(map(int, colors))
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if seen[r, c]:
                continue
            seen[r, c] = True
            if allowed is not None and color not in allowed:
                continue
            if allowed is None:
                comps.append(_component_dfs(grid, (r, c), seen=np.zeros_like(seen)))
            else:
                seen[r, c] = False
                comps.append(_component_dfs(grid, (r, c), seen))
    if allowed is None:
        # The generic branch above is not used by this task; keep behavior simple.
        raise ValueError("colors must be provided")
    return comps


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            comps.append(_component_dfs(grid, (r, c), seen))
    return comps


def _solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 > r0) and (c1 > c0) and int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _reflect_diff(rc: tuple[int, int], d: int) -> tuple[int, int]:
    r, c = rc
    return c + d, r - d


def _reflect_sum(rc: tuple[int, int], s: int) -> tuple[int, int]:
    r, c = rc
    return s - c, s - r


def _orbit_for(rc: tuple[int, int], d: int, s: int, shape: tuple[int, int]) -> tuple[tuple[int, int], ...] | None:
    pts = {
        tuple(map(int, rc)),
        tuple(map(int, _reflect_diff(rc, d))),
        tuple(map(int, _reflect_sum(rc, s))),
    }
    pts.add(tuple(map(int, _reflect_sum(_reflect_diff(rc, d), s))))
    if any(not _inside(shape, r, c) for r, c in pts):
        return None
    return tuple(sorted(pts))


def _fit_one_axis(
    grid: np.ndarray,
    hidden: np.ndarray,
    bg: int,
    axis_kind: str,
) -> list[int]:
    h, w = map(int, grid.shape)
    colored = np.argwhere((grid != bg) & (~hidden))
    if axis_kind == "diff":
        values = range(-(w - 1), h)
        reflect = _reflect_diff
    else:
        values = range(h + w - 1)
        reflect = _reflect_sum

    valid = []
    for value in values:
        ok = True
        support = 0
        for r, c in colored:
            rr, cc = reflect((int(r), int(c)), int(value))
            if not _inside(grid.shape, rr, cc):
                ok = False
                break
            if hidden[rr, cc]:
                continue
            if int(grid[rr, cc]) != int(grid[r, c]):
                ok = False
                break
            if (rr, cc) != (int(r), int(c)):
                support += 1
        if ok and support > 0:
            valid.append((int(value), int(support)))
    if not valid:
        return []
    best = max(score for _value, score in valid)
    return [value for value, score in valid if score == best]


def _fit_axes_for_rect(grid: np.ndarray, rect: tuple[int, int, int, int], bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    hidden = np.zeros_like(grid, dtype=bool)
    hidden[r0 : r1 + 1, c0 : c1 + 1] = True

    ds = _fit_one_axis(grid, hidden, bg, "diff")
    ss = _fit_one_axis(grid, hidden, bg, "sum")
    valid = []
    for d in ds:
        for s in ss:
            ok = True
            for r, c in np.argwhere((grid != bg) & (~hidden)):
                orbit = _orbit_for((int(r), int(c)), d, s, grid.shape)
                if orbit is None:
                    ok = False
                    break
                color = int(grid[r, c])
                for rr, cc in orbit:
                    if hidden[rr, cc]:
                        continue
                    if int(grid[rr, cc]) != color:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid.append((int(d), int(s)))
    return valid


def _reconstruct_rect(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    d: int,
    s: int,
) -> np.ndarray | None:
    r0, c0, r1, c1 = rect
    hidden = np.zeros_like(grid, dtype=bool)
    hidden[r0 : r1 + 1, c0 : c1 + 1] = True
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = _orbit_for((r, c), d, s, grid.shape)
            if orbit is None:
                return None
            visible_colors = {int(grid[rr, cc]) for rr, cc in orbit if not hidden[rr, cc]}
            if len(visible_colors) != 1:
                return None
            out[r - r0, c - c0] = next(iter(visible_colors))
    return out


def _solve_with_details(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    candidates = []
    for comp in _foreground_components(g, bg):
        if not _solid_rect(comp):
            continue
        color = int(comp["color"])
        if int(np.count_nonzero(g == color)) <= int(comp["area"]):
            continue
        rect = tuple(map(int, comp["bbox"]))
        axes = _fit_axes_for_rect(g, rect, bg)
        if len(axes) != 1:
            continue
        recovered = _reconstruct_rect(g, rect, axes[0][0], axes[0][1])
        if recovered is None:
            continue
        candidates.append((rect, axes[0], recovered))

    assert len(candidates) == 1, "expected exactly one rectangle/axis explanation"
    return candidates[0]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the original contents hidden by the solid rectangle using two inferred shifted diagonal symmetries.

    Input structure:
    - A square colored pattern sits on a background and is symmetric under two diagonal transpose reflections:
      one around a shifted `r-c=d` axis and one around a shifted `r+c=s` axis.
    - A single monochrome axis-aligned rectangle occludes part of that pattern, and the occluder color also
      appears elsewhere in disconnected pattern pixels.
    - The visible pattern often looks almost left-right symmetric, but some visible pixels contradict that decoy.

    Rule:
    - Detect the solid rectangular foreground component that is the occluder.
    - Infer the two diagonal-axis offsets from the visible non-occluded pixels by finding the unique pair of
      reflections that preserves every visible color orbit.
    - For each hidden cell, collect its full orbit under the two reflections and fill the cell with the unique
      visible color found elsewhere in that orbit.

    Key insight:
    - A single mirror is not enough: hidden cells may map to other hidden cells under either diagonal reflection,
      so reconstruction must use the complete four-point orbit generated by both shifted diagonal axes.
    """
    _rect, _axes, recovered = _solve_with_details(input_grid)
    return recovered


def _build_orbits(size: int, d: int, s: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = _orbit_for((r, c), d, s, (size, size))
            if orbit is None:
                continue
            orbits.append(orbit)
            seen.update(orbit)
    return orbits


def _seed_orbits(size: int, cr: int, cc: int) -> list[tuple[tuple[int, int], ...]] | None:
    reach = min(cr, cc, size - 1 - cr, size - 1 - cc) - 1
    if reach < 4:
        return None
    max_x = random.randint(3, reach)
    max_y = random.randint(3, reach)
    seeds = []
    for x in range(max_x + 1):
        for y in range(x, max_y + 1):
            if x <= 2 or y <= 2 or y >= max_y - 1 or random.random() < 0.92:
                seeds.append((x, y))
    if len(seeds) < 12:
        return None

    orbits = []
    seen = set()
    for x, y in seeds:
        rel = {(x, y), (y, x), (-y, -x), (-x, -y)}
        orbit = tuple(sorted((cr + rx, cc + ry) for rx, ry in rel))
        if any(not _inside((size, size), r, c) for r, c in orbit):
            return None
        if orbit in seen:
            continue
        seen.add(orbit)
        orbits.append(orbit)
    return orbits if len(orbits) >= 12 else None


def _orbit_id_map(orbits: list[tuple[tuple[int, int], ...]]) -> dict[tuple[int, int], int]:
    out = {}
    for idx, orbit in enumerate(orbits):
        for rc in orbit:
            out[rc] = idx
    return out


def _assign_pattern_colors(
    size: int,
    orbits: list[tuple[tuple[int, int], ...]],
    orbit_index: dict[tuple[int, int], int],
) -> np.ndarray:
    palette = random.sample(PALETTE, random.randint(4, 6))
    grid = np.zeros((size, size), dtype=int)
    colored: dict[int, int] = {}

    pair_reps = []
    seen = set()
    for idx, orbit in enumerate(orbits):
        if idx in seen:
            continue
        mirrors = {(r, size - 1 - c) for r, c in orbit}
        if all(rc in orbit_index for rc in mirrors):
            partner_ids = {orbit_index[rc] for rc in mirrors}
            if len(partner_ids) == 1:
                partner = next(iter(partner_ids))
                if partner not in seen:
                    pair_reps.append((idx, partner))
                    seen.add(idx)
                    seen.add(partner)

    def pick_color(forbidden: set[int]) -> int:
        choices = [c for c in palette if c not in forbidden]
        return random.choice(choices if choices else palette)

    for a, b in pair_reps:
        left_cells = set(orbits[a])
        right_cells = set(orbits[b])
        forbid_a = {
            int(grid[nr, nc])
            for r, c in left_cells
            for dr, dc in ORTHO4
            if _inside(grid.shape, (nr := r + dr), (nc := c + dc)) and int(grid[nr, nc]) != 0
        }
        color_a = pick_color(forbid_a)
        if a == b:
            color_b = color_a
        elif random.random() < 0.82:
            color_b = color_a
        else:
            color_b = random.choice([c for c in palette if c != color_a])
        colored[a] = color_a
        colored[b] = color_b

    for idx, orbit in enumerate(orbits):
        if idx not in colored:
            forbid = {
                int(grid[nr, nc])
                for r, c in orbit
                for dr, dc in ORTHO4
                if _inside(grid.shape, (nr := r + dr), (nc := c + dc)) and int(grid[nr, nc]) != 0
            }
            colored[idx] = pick_color(forbid)

    for idx, orbit in enumerate(orbits):
        color = int(colored[idx])
        for r, c in orbit:
            grid[r, c] = color
    return grid


def _rect_metrics(
    rect: tuple[int, int, int, int],
    d: int,
    s: int,
    shape: tuple[int, int],
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = rect
    hidden = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    diff_hidden = 0
    sum_hidden = 0
    composite_useful = 0
    for rc in sorted(hidden):
        orbit = _orbit_for(rc, d, s, shape)
        if orbit is None:
            return -1, -1, -1
        dr = _reflect_diff(rc, d)
        sr = _reflect_sum(rc, s)
        visible = [pt for pt in orbit if pt not in hidden]
        if not visible:
            return -1, -1, -1
        if dr in hidden:
            diff_hidden += 1
        if sr in hidden:
            sum_hidden += 1
        if (dr in hidden or sr in hidden) and len(visible) >= 2:
            composite_useful += 1
    return diff_hidden, sum_hidden, composite_useful


def _visible_lr_mismatches(grid: np.ndarray, rect: tuple[int, int, int, int], bg: int) -> int:
    n = grid.shape[1]
    r0, c0, r1, c1 = rect
    hidden = np.zeros_like(grid, dtype=bool)
    hidden[r0 : r1 + 1, c0 : c1 + 1] = True
    mismatches = 0
    checked = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            mc = n - 1 - c
            if (r, c) in checked or not _inside(grid.shape, r, mc):
                continue
            checked.add((r, c))
            checked.add((r, mc))
            if hidden[r, c] or hidden[r, mc]:
                continue
            if int(grid[r, c]) == bg or int(grid[r, mc]) == bg:
                continue
            if int(grid[r, c]) != int(grid[r, mc]):
                mismatches += 1
    return mismatches


def _find_valid_rectangles(base: np.ndarray, d: int, s: int) -> list[tuple[int, int, int, int]]:
    size = base.shape[0]
    valid = []
    want_diff = random.random() < 0.5
    for h in range(2, 6):
        for w in range(2, 6):
            if h * w < 6:
                continue
            for r0 in range(1, size - h):
                for c0 in range(1, size - w):
                    rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                    cells = [(r, c) for r in range(rect[0], rect[2] + 1) for c in range(rect[1], rect[3] + 1)]
                    if any(int(base[r, c]) == 0 for r, c in cells):
                        continue
                    colors = {int(base[r, c]) for r, c in cells}
                    if len(colors) < 2:
                        continue
                    diff_hidden, sum_hidden, composite_useful = _rect_metrics(rect, d, s, base.shape)
                    if composite_useful < 2:
                        continue
                    if want_diff and diff_hidden < 1:
                        continue
                    if (not want_diff) and sum_hidden < 1:
                        continue
                    valid.append(rect)
    random.shuffle(valid)
    return valid


def _near_intersection_rectangles(
    base: np.ndarray,
    cr: int,
    cc: int,
    d: int,
    s: int,
) -> list[tuple[int, int, int, int]]:
    size = base.shape[0]
    rects = []
    want_diff = random.random() < 0.5
    for top in range(max(1, cr - 4), min(cr + 1, size - 2)):
        for left in range(max(1, cc - 4), min(cc + 1, size - 2)):
            for bottom in range(max(top + 1, cr), min(size - 2, cr + 4)):
                for right in range(max(left + 1, cc), min(size - 2, cc + 4)):
                    rect = (top, left, bottom, right)
                    cells = [
                        (r, c)
                        for r in range(rect[0], rect[2] + 1)
                        for c in range(rect[1], rect[3] + 1)
                    ]
                    if any(int(base[r, c]) == 0 for r, c in cells):
                        continue
                    if len({int(base[r, c]) for r, c in cells}) < 2:
                        continue
                    diff_hidden, sum_hidden, composite_useful = _rect_metrics(rect, d, s, base.shape)
                    if composite_useful < 2:
                        continue
                    if want_diff and diff_hidden < 1:
                        continue
                    if (not want_diff) and sum_hidden < 1:
                        continue
                    rects.append((composite_useful, max(diff_hidden, sum_hidden), rect))
    rects.sort(reverse=True)
    return [rect for _fo, _hc, rect in rects]


def _pick_occluder_color(base: np.ndarray, rect: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = rect
    border_neighbors = set()
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if not _inside(base.shape, r, c):
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if abs(r - min(max(r, r0), r1)) + abs(c - min(max(c, c0), c1)) == 1:
                border_neighbors.add(int(base[r, c]))
    candidates = [
        color
        for color in sorted(set(map(int, base.ravel())) - {0})
        if color not in border_neighbors and int(np.count_nonzero(base == color)) >= 4
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _occlude(base: np.ndarray, rect: tuple[int, int, int, int], occ_color: int) -> np.ndarray:
    grid = base.copy()
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(occ_color)
    return grid


def _flip_lr_rect(rect: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return int(r0), int(width - 1 - c1), int(r1), int(width - 1 - c0)


def _axes_decoy_ok(grid: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    bg = _get_bg(grid)
    mismatches = _visible_lr_mismatches(grid, rect, bg)
    visible_fg = int(np.count_nonzero(grid != bg)) - (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    return mismatches >= 1 and visible_fg >= 20


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(15, 23)
    cr = random.randint(4, size - 5)
    cc_choices = [c for c in range(4, size - 4) if c != cr and c != size - 1 - cr]
    if not cc_choices:
        return None
    cc = random.choice(cc_choices)
    d = cr - cc
    s = cr + cc
    if d == 0 or s == size - 1:
        return None

    orbits = _seed_orbits(size, cr, cc)
    if orbits is None:
        return None
    orbit_index = _orbit_id_map(orbits)
    base = _assign_pattern_colors(size, orbits, orbit_index)

    rects = _near_intersection_rectangles(base, cr, cc, d, s)
    if not rects:
        rects = _find_valid_rectangles(base, d, s)[:30]
    for rect in rects[:40]:
        occ_color = _pick_occluder_color(base, rect)
        if occ_color is None:
            continue
        trial_base = base
        trial_rect = rect
        trial_grid = _occlude(trial_base, trial_rect, occ_color)
        expected = trial_base[trial_rect[0] : trial_rect[2] + 1, trial_rect[1] : trial_rect[3] + 1]
        if random.random() < 0.5:
            trial_base = np.fliplr(trial_base)
            trial_grid = np.fliplr(trial_grid)
            trial_rect = _flip_lr_rect(trial_rect, size)
            expected = np.fliplr(expected)
        if not _axes_decoy_ok(trial_grid, trial_rect):
            continue
        try:
            found_rect, _found_axes, recovered = _solve_with_details(trial_grid)
        except AssertionError:
            continue
        if found_rect != trial_rect:
            continue
        if not np.array_equal(recovered, expected):
            continue
        return trial_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid shifted-diagonal occlusion instance")
