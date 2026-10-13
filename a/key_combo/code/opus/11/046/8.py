from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = (1, 2, 3, 4)
GREY = 5
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 800
_FRAGMENT_CACHE: dict[tuple[int, int], list[dict]] = {}
_CANDIDATE_CACHE: dict[tuple[int, int, int], tuple[list[dict], list[dict]]] = {}


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _diagonal_cells(shape: tuple[int, int], diag_sum: int) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for r in range(h):
        c = diag_sum - r
        if 0 <= c < w:
            cells.append((r, c))
    return cells


def _field_color(diag_sum: int, phase: int) -> int:
    return PALETTE[(diag_sum + phase) % len(PALETTE)]


def _components_of_grey(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _merge_fragments(fragments: list[dict]) -> list[dict]:
    groups = [{"fragments": [frag], "coords": list(frag["coords"]), "bbox": frag["bbox"]} for frag in fragments]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                a = groups[i]["bbox"]
                b = groups[j]["bbox"]
                same_row_band = _spans_overlap(a[0], a[2], b[0], b[2])
                same_col_band = _spans_overlap(a[1], a[3], b[1], b[3])
                if same_row_band or same_col_band:
                    coords = groups[i]["coords"] + groups[j]["coords"]
                    groups[i] = {
                        "fragments": groups[i]["fragments"] + groups[j]["fragments"],
                        "coords": coords,
                        "bbox": _bbox_from_coords(coords),
                    }
                    del groups[j]
                    changed = True
                    break
    return groups


def _group_authority(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int | None, Counter]:
    g = np.asarray(grid, dtype=int)
    coord_set = set(coords)
    counts: Counter = Counter()
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(g.shape, nr, nc) and (nr, nc) not in coord_set:
                color = int(g[nr, nc])
                if color != GREY:
                    counts[color] += 1
    if not counts:
        return None, counts
    items = counts.most_common()
    if len(items) == 1 or items[0][1] > items[1][1]:
        return int(items[0][0]), counts
    return None, counts


def _group_diag_range(coords: list[tuple[int, int]]) -> tuple[int, int]:
    sums = [r + c for r, c in coords]
    return min(sums), max(sums)


def _diagonal_info(grid: np.ndarray, diag_sum: int) -> dict:
    coords = _diagonal_cells(grid.shape, diag_sum)
    values = [int(grid[r, c]) for r, c in coords]
    non_grey = [v for v in values if v != GREY]
    counts = Counter(non_grey)
    resolved = None
    if counts:
        items = counts.most_common()
        if len(items) == 1 or items[0][1] > items[1][1]:
            resolved = int(items[0][0])
    return {
        "coords": coords,
        "values": values,
        "non_grey_counts": counts,
        "resolved_visible": resolved,
    }


def _resolve_diagonal_colors(grid: np.ndarray) -> tuple[dict[int, int], list[dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    fragments = _components_of_grey(g)
    groups = _merge_fragments(fragments)

    for group in groups:
        authority, counts = _group_authority(g, group["coords"])
        group["authority"] = authority
        group["authority_counts"] = counts
        group["diag_range"] = _group_diag_range(group["coords"])

    diag_colors: dict[int, int] = {}
    unresolved: list[int] = []
    for diag_sum in range(g.shape[0] + g.shape[1] - 1):
        info = _diagonal_info(g, diag_sum)
        if info["resolved_visible"] is not None:
            diag_colors[diag_sum] = info["resolved_visible"]
            continue

        covering = {
            int(group["authority"])
            for group in groups
            if group["authority"] is not None and group["diag_range"][0] <= diag_sum <= group["diag_range"][1]
        }
        if len(covering) == 1:
            diag_colors[diag_sum] = next(iter(covering))
        else:
            unresolved.append(diag_sum)
    return diag_colors, groups, unresolved


def main(grid: np.ndarray) -> np.ndarray:
    """Repair ambiguous anti-diagonals by using merged grey fragments as arbitration witnesses.

    Input structure:
    - The whole grid is an anti-diagonal color field using colors 1, 2, 3, and 4, plus scattered grey fragments of color 5.
    - Each anti-diagonal is indexed by `row + col`; most anti-diagonals already show one consistent color, but a few conflicted anti-diagonals visibly mix two colors.
    - Grey cells form 4-connected fragments; disconnected fragments that overlap in row span or column span belong to one merged witness group.

    Rule:
    - Keep the already consistent anti-diagonals as they are.
    - Merge grey fragments transitively by overlapping row span or overlapping column span, then read each merged group's authority color as the unique dominant non-grey color touching that group.
    - For every conflicted anti-diagonal, use the authority color of the merged group whose anti-diagonal span covers that index, repaint the whole anti-diagonal to that color, and also recolor all grey cells by the resolved color of their own anti-diagonal.

    Key insight:
    - The anti-diagonal field alone is insufficient on 50-50 conflict lines; the decisive information is encoded in how separate grey fragments must first be merged into larger witness groups whose spans tell you which conflicted diagonals they govern.
    """
    g = np.asarray(grid, dtype=int)
    diag_colors, groups, unresolved = _resolve_diagonal_colors(g)
    assert not unresolved, f"unresolved anti-diagonals: {unresolved}"
    assert groups, "expected at least one grey fragment group"

    out = g.copy()
    for diag_sum, color in diag_colors.items():
        coords = _diagonal_cells(g.shape, diag_sum)
        if any(int(g[r, c]) == GREY for r, c in coords) or len({int(g[r, c]) for r, c in coords if int(g[r, c]) != GREY}) > 1:
            for r, c in coords:
                out[r, c] = int(color)
        else:
            for r, c in coords:
                if int(out[r, c]) == GREY:
                    out[r, c] = int(color)
    return out


def _enumerate_fragments(shape: tuple[int, int]) -> list[dict]:
    if shape in _FRAGMENT_CACHE:
        return _FRAGMENT_CACHE[shape]

    h, w = shape
    fragments: list[dict] = []

    for length in (2, 3, 4):
        for c0 in range(w - length + 1):
            coords = tuple((0, c0 + dc) for dc in range(length))
            fragments.append({"side": "top", "coords": coords})
            coords = tuple((h - 1, c0 + dc) for dc in range(length))
            fragments.append({"side": "bottom", "coords": coords})
        for r0 in range(h - length + 1):
            coords = tuple((r0 + dr, 0) for dr in range(length))
            fragments.append({"side": "left", "coords": coords})
            coords = tuple((r0 + dr, w - 1) for dr in range(length))
            fragments.append({"side": "right", "coords": coords})

    for frag in fragments:
        frag["bbox"] = _bbox_from_coords(list(frag["coords"]))
        frag["diag_range"] = _group_diag_range(list(frag["coords"]))
    _FRAGMENT_CACHE[shape] = fragments
    return fragments


def _candidate_groups(solution: np.ndarray, phase: int) -> tuple[list[dict], list[dict]]:
    cache_key = (solution.shape[0], solution.shape[1], phase)
    if cache_key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[cache_key]

    fragments = _enumerate_fragments(solution.shape)
    singles: list[dict] = []
    multis: list[dict] = []

    for frag in fragments:
        authority, counts = _group_authority(solution, list(frag["coords"]))
        if authority is None:
            continue
        allowed_ds = []
        d0, d1 = frag["diag_range"]
        for diag_sum in range(d0, d1 + 1):
            if _field_color(diag_sum, phase) != authority:
                continue
            if len(_diagonal_cells(solution.shape, diag_sum)) < 4:
                continue
            allowed_ds.append(diag_sum)
        if allowed_ds:
            singles.append(
                {
                    "coords": list(frag["coords"]),
                    "fragments": [frag],
                    "bbox": frag["bbox"],
                    "diag_range": frag["diag_range"],
                    "authority": authority,
                    "authority_counts": counts,
                    "allowed_ds": allowed_ds,
                    "side": frag["side"],
                }
            )

    by_side: dict[str, list[dict]] = {side: [] for side in ("top", "bottom", "left", "right")}
    for frag in fragments:
        by_side[frag["side"]].append(frag)

    for side, side_frags in by_side.items():
        for count in (2, 3):
            for combo in _iter_combinations(side_frags, count):
                coord_list = [coord for frag in combo for coord in frag["coords"]]
                if len(set(coord_list)) != len(coord_list):
                    continue
                authority, counts = _group_authority(solution, coord_list)
                if authority is None:
                    continue
                bbox = _bbox_from_coords(coord_list)
                diag_range = _group_diag_range(coord_list)
                fragment_ranges = [frag["diag_range"] for frag in combo]
                allowed_ds = []
                for diag_sum in range(diag_range[0], diag_range[1] + 1):
                    if _field_color(diag_sum, phase) != authority:
                        continue
                    if len(_diagonal_cells(solution.shape, diag_sum)) < 4:
                        continue
                    if any(r0 <= diag_sum <= r1 for r0, r1 in fragment_ranges):
                        continue
                    allowed_ds.append(diag_sum)
                if not allowed_ds:
                    continue
                multis.append(
                    {
                        "coords": coord_list,
                        "fragments": list(combo),
                        "bbox": bbox,
                        "diag_range": diag_range,
                        "authority": authority,
                        "authority_counts": counts,
                        "allowed_ds": allowed_ds,
                        "side": side,
                    }
                )
    _CANDIDATE_CACHE[cache_key] = (singles, multis)
    return singles, multis


def _iter_combinations(items: list[dict], count: int):
    n = len(items)
    if count == 2:
        for i in range(n):
            for j in range(i + 1, n):
                yield (items[i], items[j])
    elif count == 3:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    yield (items[i], items[j], items[k])


def _groups_separated(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    row_overlap = _spans_overlap(ar0, ar1, br0, br1)
    col_overlap = _spans_overlap(ac0, ac1, bc0, bc1)
    return not row_overlap and not col_overlap


def _visible_count_after_greys(shape: tuple[int, int], diag_sum: int, grey_cells: set[tuple[int, int]]) -> int:
    return sum((r, c) not in grey_cells for r, c in _diagonal_cells(shape, diag_sum))


def _assign_conflicts(groups: list[dict], shape: tuple[int, int], grey_cells: set[tuple[int, int]]) -> dict[int, int] | None:
    allowed_lists: list[list[int]] = []
    for i, group in enumerate(groups):
        allowed = []
        for diag_sum in group["allowed_ds"]:
            if _visible_count_after_greys(shape, diag_sum, grey_cells) < 4:
                continue
            if _visible_count_after_greys(shape, diag_sum, grey_cells) % 2 != 0:
                continue
            if any(
                other["diag_range"][0] <= diag_sum <= other["diag_range"][1]
                for j, other in enumerate(groups)
                if j != i
            ):
                continue
            allowed.append(diag_sum)
        if not allowed:
            return None
        random.shuffle(allowed)
        allowed_lists.append(allowed)

    order = sorted(range(len(groups)), key=lambda idx: len(allowed_lists[idx]))
    chosen: dict[int, int] = {}
    used_ds: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for diag_sum in allowed_lists[idx]:
            if diag_sum in used_ds:
                continue
            chosen[idx] = diag_sum
            used_ds.add(diag_sum)
            if backtrack(pos + 1):
                return True
            used_ds.remove(diag_sum)
            del chosen[idx]
        return False

    if not backtrack(0):
        return None
    return {diag_sum: idx for idx, diag_sum in chosen.items()}


def _paint_conflicts(inp: np.ndarray, solution: np.ndarray, conflict_to_group: dict[int, int], phase: int) -> dict[int, int]:
    alt_colors: dict[int, int] = {}
    for diag_sum in conflict_to_group:
        correct = _field_color(diag_sum, phase)
        choices = [color for color in PALETTE if color != correct]
        alt = random.choice(choices)
        cells = [(r, c) for r, c in _diagonal_cells(inp.shape, diag_sum) if int(inp[r, c]) != GREY]
        assert len(cells) >= 4 and len(cells) % 2 == 0
        pick = set(random.sample(range(len(cells)), len(cells) // 2))
        for idx, (r, c) in enumerate(cells):
            inp[r, c] = alt if idx in pick else correct
        alt_colors[diag_sum] = alt
    return alt_colors


def _validate_instance(inp: np.ndarray, solution: np.ndarray, conflict_to_group: dict[int, int], raw_groups: list[dict]) -> bool:
    diag_colors, merged_groups, unresolved = _resolve_diagonal_colors(inp)
    if unresolved:
        return False
    if not np.array_equal(main(inp), solution):
        return False

    if len(merged_groups) != len(raw_groups):
        return False

    for diag_sum, group_idx in conflict_to_group.items():
        info = _diagonal_info(inp, diag_sum)
        counts = info["non_grey_counts"]
        if len(counts) != 2:
            return False
        values = sorted(counts.values())
        if values[0] != values[1]:
            return False
        if diag_colors.get(diag_sum) != int(solution[_diagonal_cells(solution.shape, diag_sum)[0]]):
            return False
        covering = [
            group
            for group in merged_groups
            if group["diag_range"][0] <= diag_sum <= group["diag_range"][1]
        ]
        if len({group["authority"] for group in covering}) != 1:
            return False
        if next(iter({group["authority"] for group in covering})) != diag_colors[diag_sum]:
            return False
        target_group = raw_groups[group_idx]
        if len(target_group["fragments"]) > 1:
            if any(
                frag["diag_range"][0] <= diag_sum <= frag["diag_range"][1]
                for frag in target_group["fragments"]
            ):
                return False

    grey_cells = set(map(tuple, np.argwhere(inp == GREY)))
    for diag_sum in range(inp.shape[0] + inp.shape[1] - 1):
        cells = _diagonal_cells(inp.shape, diag_sum)
        non_grey = [int(inp[r, c]) for r, c in cells if int(inp[r, c]) != GREY]
        if not non_grey:
            return False
        if diag_sum not in conflict_to_group and len(set(non_grey)) != 1:
            return False
        if any((r, c) in grey_cells for r, c in cells) and diag_sum not in diag_colors:
            return False
    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    phase = random.randint(0, 3)
    solution = np.fromfunction(lambda r, c: ((r + c + phase) % 4) + 1, (h, w), dtype=int)

    singles, multis = _candidate_groups(solution, phase)
    if not singles or not multis:
        return None

    num_conflicts = random.randint(2, 4)
    multi_size = 2 if num_conflicts == 4 else random.choice((2, 3))
    multis = [group for group in multis if len(group["fragments"]) == multi_size]
    if not multis:
        return None

    random.shuffle(multis)
    random.shuffle(singles)

    for multi in multis[:120]:
        side_order = ["top", "bottom", "left", "right"]
        random.shuffle(side_order)
        chosen_sides = random.sample(side_order, num_conflicts)
        if multi["side"] not in chosen_sides:
            chosen_sides[0] = multi["side"]
        groups = [multi]
        needed = [side for side in chosen_sides if side != multi["side"]]

        pools = {side: [group for group in singles if group["side"] == side] for side in needed}
        if any(not pools[side] for side in needed):
            continue

        for side in pools:
            random.shuffle(pools[side])

        ok = True
        for side in needed:
            picked = None
            for candidate in pools[side]:
                if all(_groups_separated(candidate, group) for group in groups):
                    picked = candidate
                    break
            if picked is None:
                ok = False
                break
            groups.append(picked)
        if not ok:
            continue

        grey_cells = {coord for group in groups for coord in group["coords"]}
        conflict_to_group = _assign_conflicts(groups, solution.shape, grey_cells)
        if conflict_to_group is None:
            continue

        inp = solution.copy()
        for r, c in grey_cells:
            inp[r, c] = GREY
        _paint_conflicts(inp, solution, conflict_to_group, phase)

        if _validate_instance(inp, solution, conflict_to_group, groups):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anti-diagonal arbitration puzzle")
