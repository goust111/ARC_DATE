from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PALETTE = (1, 2, 4, 6, 7, 8, 9)
INPUT_COLORS = tuple(color for color in PALETTE if color != GREEN and color != GREY)

GEN_ATTEMPTS = 400
ROW_BAND_MIN = 5
ROW_BAND_MAX = 7
COL_BAND_MIN = 5
COL_BAND_MAX = 7
LOWER_GAP = 1
COL_GAP_CHOICES = (1, 2)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox_from_coords(arr), "area": int(arr.shape[0])})
    return components


def _merge_fragment_groups(components: list[dict]) -> list[dict]:
    groups = [{"coords": comp["coords"].copy(), "bbox": comp["bbox"]} for comp in components]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    coords = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": coords, "bbox": _bbox_from_coords(coords)}
                    del groups[j]
                    changed = True
                    break
    return groups


def _divider_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full-width grey divider row"
    return int(rows[0])


def _foreground_colors(grid: np.ndarray) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY))
    assert len(colors) == 2, "expected exactly two non-black, non-grey colors"
    return colors


def _expected_from_groups(
    grid: np.ndarray,
    divider: int,
    target_color: int,
    merged_target_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[divider, :] = BLACK
    for r0, c0, r1, c1 in merged_target_bboxes:
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == BLACK] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Choose the color with the larger below-vs-above growth, then complete only its lower fragmented rectangles with green fill.

    Input structure:
    - A black background grid contains one full-width grey horizontal divider.
    - Exactly two other foreground colors appear above and below that divider.
    - In the lower region, each color forms one or more disconnected 4-connected fragments that belong to incomplete rectangles recoverable by row/column-span merging.

    Rule:
    - Count, for each foreground color, how many pixels lie above the divider and how many lie below it.
    - Compute below_count minus above_count and select the unique color with the larger growth.
    - In the lower region only, find that color's 4-connected fragments and merge fragments transitively whenever their bounding boxes overlap in row span or column span.
    - For each merged group, fill every black cell inside the group's bounding box with green, keep all original colored pixels unchanged, and erase the grey divider to black.

    Key insight:
    - The counting comparison is a gate: only the color that grows more below the divider earns rectangle completion, so solving requires both the region-count selection step and the span-based fragment merge step.
    """
    g = np.asarray(grid, dtype=int)
    divider = _divider_row(g)
    colors = _foreground_colors(g)

    scores: list[tuple[int, int]] = []
    for color in colors:
        above = int(np.count_nonzero(g[:divider] == color))
        below = int(np.count_nonzero(g[divider + 1 :] == color))
        scores.append((below - above, int(color)))

    best_growth = max(score for score, _ in scores)
    winners = [color for score, color in scores if score == best_growth]
    assert len(winners) == 1, "growth winner must be unique"
    target = int(winners[0])

    lower_mask = g[divider + 1 :] == target
    fragments = _find_components(lower_mask)
    shifted = []
    for comp in fragments:
        coords = comp["coords"].copy()
        coords[:, 0] += divider + 1
        shifted.append({"coords": coords, "bbox": _bbox_from_coords(coords), "area": comp["area"]})

    merged = _merge_fragment_groups(shifted)
    merged_bboxes = [tuple(group["bbox"]) for group in merged]
    return _expected_from_groups(g, divider, target, merged_bboxes)


def _build_vhv_mask(h: int, w: int) -> np.ndarray:
    len1_max = min(3, h - 3)
    assert len1_max >= 2
    len1 = int(np.random.randint(2, len1_max + 1))

    len2_max = min(5, w - 2)
    assert len2_max >= 2
    len2 = int(np.random.randint(2, len2_max + 1))

    len3_max = min(3, h - len1 - 1)
    assert len3_max >= 2
    len3 = int(np.random.randint(2, len3_max + 1))

    mask = np.zeros((h, w), dtype=bool)
    mask[:len1, 0] = True
    mask[len1 - 1, w - len2 : w] = True
    mask[h - len3 :, w - 1] = True
    return mask


def _build_group_mask(h: int, w: int) -> np.ndarray:
    if np.random.rand() < 0.5:
        return _build_vhv_mask(h, w)
    return _build_vhv_mask(w, h).T


def _choose_bands(total: int, low: int, high: int, gap_choices: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    choices: list[tuple[int, int, int, int, int]] = []
    for first in range(low, high + 1):
        for second in range(low, high + 1):
            for gap in gap_choices:
                if first + second + gap > total:
                    continue
                slack = total - first - second - gap
                for lead in range(slack + 1):
                    trail = slack - lead
                    start1 = lead
                    start2 = lead + first + gap
                    choices.append((start1, first, start2, second, trail))
    assert choices
    return choices[int(np.random.randint(len(choices)))]


def _sample_slots(lower_h: int, width: int) -> dict[str, tuple[int, int, int, int]]:
    r0, h0, r1, h1, _ = _choose_bands(lower_h, ROW_BAND_MIN, ROW_BAND_MAX, (LOWER_GAP,))
    c0, w0, c1, w1, _ = _choose_bands(width, COL_BAND_MIN, COL_BAND_MAX, COL_GAP_CHOICES)
    return {
        "TL": (r0, c0, h0, w0),
        "TR": (r0, c1, h0, w1),
        "BL": (r1, c0, h1, w0),
        "BR": (r1, c1, h1, w1),
    }


def _branch_slot_assignment() -> dict[int, list[str]]:
    branch = int(np.random.randint(4))
    if branch == 0:
        return {0: ["TL"], 1: ["BR"]}
    if branch == 1:
        return {0: ["TL", "BR"], 1: ["TR"]}
    if branch == 2:
        return {0: ["TR"], 1: ["TL", "BR"]}
    return {0: ["TL", "BR"], 1: ["TR", "BL"]}


def _place_upper_pixels(grid: np.ndarray, color: int, count: int, occupied: set[tuple[int, int]], upper_h: int) -> None:
    cells = [(r, c) for r in range(upper_h) for c in range(grid.shape[1]) if (r, c) not in occupied]
    assert len(cells) >= count
    chosen_idx = np.random.choice(len(cells), size=count, replace=False)
    for idx in np.atleast_1d(chosen_idx):
        r, c = cells[int(idx)]
        grid[r, c] = color
        occupied.add((int(r), int(c)))


def _collect_group_info(
    divider: int,
    local_slot: tuple[int, int, int, int],
    color: int,
) -> dict:
    lr0, lc0, h, w = local_slot
    mask = _build_group_mask(h, w)
    coords_local = np.argwhere(mask)
    coords_global = coords_local.copy()
    coords_global[:, 0] += divider + 1 + lr0
    coords_global[:, 1] += lc0
    bbox = _bbox_from_coords(coords_global)
    return {
        "color": int(color),
        "slot": local_slot,
        "mask": mask,
        "coords": coords_global,
        "bbox": bbox,
        "area": int(coords_global.shape[0]),
    }


def _render_case(
    grid: np.ndarray,
    divider: int,
    group_infos: list[dict],
) -> None:
    for info in group_infos:
        color = int(info["color"])
        for r, c in info["coords"]:
            grid[int(r), int(c)] = color


def _merged_bboxes_for_color(grid: np.ndarray, divider: int, color: int) -> list[tuple[int, int, int, int]]:
    lower_mask = grid[divider + 1 :] == color
    comps = _find_components(lower_mask)
    shifted = []
    for comp in comps:
        coords = comp["coords"].copy()
        coords[:, 0] += divider + 1
        shifted.append({"coords": coords, "bbox": _bbox_from_coords(coords)})
    return [tuple(group["bbox"]) for group in _merge_fragment_groups(shifted)]


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    divider: int,
    colors: tuple[int, int],
    winner: int,
    loser: int,
    intended_bboxes: dict[int, list[tuple[int, int, int, int]]],
) -> bool:
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    if np.any(grid[divider, :] != GREY):
        return False
    if not np.all(expected[divider, :] == BLACK):
        return False

    growths = {}
    for color in colors:
        above = int(np.count_nonzero(grid[:divider] == color))
        below = int(np.count_nonzero(grid[divider + 1 :] == color))
        growths[int(color)] = below - above
    if not (growths[winner] > 0 and growths[loser] <= 0 and growths[winner] > growths[loser]):
        return False

    actual_winner_bboxes = sorted(_merged_bboxes_for_color(grid, divider, winner))
    actual_loser_bboxes = sorted(_merged_bboxes_for_color(grid, divider, loser))
    if actual_winner_bboxes != sorted(intended_bboxes[winner]):
        return False
    if actual_loser_bboxes != sorted(intended_bboxes[loser]):
        return False

    winner_fill = expected == GREEN
    if int(np.count_nonzero(winner_fill)) == 0:
        return False

    loser_variant = _expected_from_groups(grid, divider, loser, intended_bboxes[loser])
    both_variant = _expected_from_groups(
        _expected_from_groups(grid, divider, winner, intended_bboxes[winner]),
        divider,
        loser,
        intended_bboxes[loser],
    )
    erase_only = np.array(grid, copy=True)
    erase_only[divider, :] = BLACK
    if np.array_equal(loser_variant, expected):
        return False
    if np.array_equal(both_variant, expected):
        return False
    if np.array_equal(erase_only, expected):
        return False

    if len(intended_bboxes[winner]) > 1:
        union_bbox = _bbox_from_coords(np.argwhere(grid[divider + 1 :] == winner) + np.array([divider + 1, 0]))
        global_variant = _expected_from_groups(grid, divider, winner, [union_bbox])
        if np.array_equal(global_variant, expected):
            return False

    for color in colors:
        lower_mask = grid[divider + 1 :] == color
        comps = _find_components(lower_mask)
        if len(comps) < len(intended_bboxes[color]) * 2:
            return False
        if any(not (2 <= comp["area"] <= 5) for comp in comps):
            return False
        for bbox in intended_bboxes[color]:
            r0, c0, r1, c1 = bbox
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(np.count_nonzero(region == BLACK)) == 0:
                return False
            if color == winner and int(np.count_nonzero(expected[r0 : r1 + 1, c0 : c1 + 1] == GREEN)) == 0:
                return False
    return True


def _attempt_generate() -> np.ndarray | None:
    upper_h = int(np.random.randint(4, 7))
    lower_h = int(np.random.randint(11, 14))
    width = int(np.random.randint(16, 21))
    height = upper_h + 1 + lower_h
    divider = upper_h

    colors = tuple(int(c) for c in np.random.choice(np.array(INPUT_COLORS, dtype=int), size=2, replace=False))
    winner_index = int(np.random.randint(2))
    winner = colors[winner_index]
    loser = colors[1 - winner_index]

    slots = _sample_slots(lower_h, width)
    assignment = _branch_slot_assignment()
    if winner_index == 1:
        assignment = {1: assignment[0], 0: assignment[1]}

    grid = np.zeros((height, width), dtype=int)
    grid[divider, :] = GREY

    group_infos: list[dict] = []
    intended_bboxes: dict[int, list[tuple[int, int, int, int]]] = {colors[0]: [], colors[1]: []}
    for idx, color in enumerate(colors):
        for slot_name in assignment[idx]:
            info = _collect_group_info(divider, slots[slot_name], int(color))
            group_infos.append(info)
            intended_bboxes[int(color)].append(tuple(info["bbox"]))

    _render_case(grid, divider, group_infos)

    below_counts = {color: sum(info["area"] for info in group_infos if info["color"] == color) for color in colors}
    winner_above = int(np.random.randint(max(0, below_counts[winner] - 3), below_counts[winner]))
    loser_above = int(np.random.randint(below_counts[loser], below_counts[loser] + 4))

    occupied_upper: set[tuple[int, int]] = set()
    _place_upper_pixels(grid, winner, winner_above, occupied_upper, upper_h)
    _place_upper_pixels(grid, loser, loser_above, occupied_upper, upper_h)

    expected = _expected_from_groups(grid, divider, winner, intended_bboxes[winner])
    if not _semantic_checks(grid, expected, divider, colors, winner, loser, intended_bboxes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
