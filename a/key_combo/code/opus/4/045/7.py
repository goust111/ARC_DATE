from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg, neighbors4


GREY = 5
ORANGE = 7
MARKER_COLORS = (1, 2, 3)
DIR_RIGHT = (0, 1)
DIR_LEFT = (0, -1)
DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
RECT_MIN_H = 9
RECT_MAX_H = 11
RECT_MIN_W = 9
RECT_MAX_W = 10
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _has_hole(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _mask_bbox(mask)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    bg_comps = components_mask(~cropped, connectivity=4)
    for comp in bg_comps:
        if all(
            0 < int(r) < cropped.shape[0] - 1 and 0 < int(c) < cropped.shape[1] - 1
            for r, c in comp
        ):
            return True
    return False


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    start_coords = np.argwhere(g == ORANGE)
    assert start_coords.shape[0] == 1
    start = tuple(map(int, start_coords[0]))
    assert start[1] == 0

    mask = g == GREY
    mask_comps = components_mask(mask, connectivity=4)
    assert len(mask_comps) == 1
    assert not _has_hole(mask)

    assert _in_bounds(g.shape, start[0], start[1] + 1)
    assert mask[start[0], start[1] + 1]

    marker_comps = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color in (GREY, ORANGE):
            continue
        marker_comps.append(comp)

    assert 3 <= len(marker_comps) <= 5
    markers = []
    for comp in sorted(marker_comps, key=lambda item: tuple(map(int, item["coords"][0]))):
        assert int(comp["area"]) == 1
        color = int(comp["color"])
        assert color in MARKER_COLORS
        r, c = map(int, comp["coords"][0])
        mask_neighbors = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if _in_bounds(g.shape, nr, nc) and mask[nr, nc]
        ]
        assert len(mask_neighbors) == 1
        markers.append(
            {
                "pos": (r, c),
                "color": color,
                "neighbor": mask_neighbors[0],
            }
        )

    assert 2 <= len({marker["color"] for marker in markers}) <= 3
    return g, mask, start, markers


def _turn_direction(
    mask: np.ndarray, cell: tuple[int, int], direction: tuple[int, int]
) -> tuple[tuple[int, int], int]:
    r, c = cell
    if direction in (DIR_LEFT, DIR_RIGHT):
        candidates = (DIR_UP, DIR_DOWN)
    else:
        candidates = (DIR_LEFT, DIR_RIGHT)

    valid = []
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            valid.append((dr, dc))

    assert valid
    if direction in (DIR_LEFT, DIR_RIGHT):
        return valid[0], len(valid)
    assert len(valid) == 1
    return valid[0], 1


def _trace_ray(
    mask: np.ndarray, start: tuple[int, int], markers: list[dict]
) -> tuple[np.ndarray, list[int], list[int]]:
    marker_by_pos = {tuple(marker["pos"]): idx for idx, marker in enumerate(markers)}
    cut = np.zeros_like(mask, dtype=bool)
    used_markers: set[int] = set()
    turn_indices: list[int] = []
    branch_sizes: list[int] = []
    direction = DIR_RIGHT
    cell = (start[0], start[1] + 1)
    assert _in_bounds(mask.shape, cell[0], cell[1]) and mask[cell]
    seen_states: set[tuple[int, int, int, int, tuple[int, ...]]] = set()

    while True:
        state = (cell[0], cell[1], direction[0], direction[1], tuple(sorted(used_markers)))
        assert state not in seen_states
        seen_states.add(state)
        cut[cell] = True

        adj_unused = []
        for nr, nc in neighbors4(cell[0], cell[1]):
            marker_idx = marker_by_pos.get((nr, nc))
            if marker_idx is not None and marker_idx not in used_markers:
                adj_unused.append(marker_idx)
        assert len(adj_unused) <= 1

        if adj_unused:
            marker_idx = adj_unused[0]
            used_markers.add(marker_idx)
            turn_indices.append(marker_idx)
            direction, branch_count = _turn_direction(mask, cell, direction)
            branch_sizes.append(int(branch_count))

        nr = cell[0] + direction[0]
        nc = cell[1] + direction[1]
        if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
            break
        cell = (nr, nc)

    return cut, turn_indices, branch_sizes


def _contact_counts(coords: np.ndarray, marker_map: dict[tuple[int, int], int]) -> Counter:
    counts: Counter = Counter()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            color = marker_map.get((nr, nc))
            if color is not None:
                counts[int(color)] += 1
    return counts


def _choose_fragment_color(counts: Counter, marker_freq: Counter) -> int:
    if not counts:
        return GREY
    if len(counts) == 1:
        return int(next(iter(counts)))

    max_contact = max(counts.values())
    candidates = [int(color) for color, value in counts.items() if value == max_contact]
    if len(candidates) == 1:
        return candidates[0]

    min_freq = min(int(marker_freq[color]) for color in candidates)
    candidates = [color for color in candidates if int(marker_freq[color]) == min_freq]
    assert len(candidates) == 1, "generator must avoid unresolved contact/frequency ties"
    return candidates[0]


def _analyze(grid: np.ndarray) -> dict:
    g, mask, start, markers = _parse_grid(grid)
    cut, turn_indices, branch_sizes = _trace_ray(mask, start, markers)
    remaining = mask & ~cut
    fragment_coords = components_mask(remaining, connectivity=4)
    marker_map = {tuple(marker["pos"]): int(marker["color"]) for marker in markers}
    marker_freq = Counter(int(marker["color"]) for marker in markers)

    fragments = []
    for coords in fragment_coords:
        counts = _contact_counts(coords, marker_map)
        color = _choose_fragment_color(counts, marker_freq)
        rs = coords[:, 0]
        cs = coords[:, 1]
        fragments.append(
            {
                "coords": coords,
                "bbox": (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())),
                "contact_counts": counts,
                "color": int(color),
            }
        )

    out = np.zeros_like(g)
    for frag in fragments:
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = int(frag["color"])

    return {
        "input": g,
        "mask": mask,
        "start": start,
        "markers": markers,
        "cut": cut,
        "turn_indices": turn_indices,
        "branch_sizes": branch_sizes,
        "fragments": fragments,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the neutral grey mask with a turning ray, then recolor each resulting fragment by adjacent marker contact.

    Input structure:
    - The background is black, with one 4-connected grey mask region and one orange start pixel on the left edge whose right neighbor is mask.
    - Three to five isolated red/green/blue marker pixels sit just outside the mask boundary, each touching exactly one mask cell.
    - The mask is connected, hole-free, and wide enough for a horizontal ray to enter from the start and later turn inside the mask.

    Rule:
    - Launch a ray from the orange pixel into the grey mask, moving right and marking every visited mask cell as cut.
    - Whenever the current ray cell is 4-adjacent to an unused marker, consume that marker as a one-time turn signal and continue in a perpendicular direction that stays inside the mask, preferring up when both vertical choices are possible.
    - Stop when the next step would leave the mask, erase the full cut path to black, then split the remaining grey mask into 4-connected fragments.
    - Recolor each fragment by the marker colors touching it in 4-neighborhood: one touching color wins directly; multiple colors use largest contact count, with ties broken by the globally rarer marker color; fragments touching no markers stay grey.
    - Erase all markers and the orange start pixel to black.

    Key insight:
    - The visible answer depends on both stages together: marker placements determine where the ray bends, which changes the fragment topology before adjacency-based recoloring can even be evaluated.
    """
    return _analyze(grid)["output"]


def _sample_nonadjacent(seq: list[int], count: int) -> list[int] | None:
    seq = sorted(seq)
    combos: list[list[int]] = []

    def dfs(idx: int, picked: list[int]) -> None:
        if len(picked) == count:
            combos.append(picked[:])
            return
        if idx >= len(seq):
            return
        for j in range(idx, len(seq)):
            value = seq[j]
            if picked and abs(value - picked[-1]) <= 1:
                continue
            picked.append(value)
            dfs(j + 1, picked)
            picked.pop()

    dfs(0, [])
    if not combos:
        return None
    return random.choice(combos)


def _marker_layout(
    scenario: str,
    r0: int,
    r1: int,
    c1: int,
    r_up: int,
    r_mid: int,
    notch_start: int,
) -> list[tuple[tuple[int, int], str]]:
    markers = [((r_mid, c1 + 1), "turn1"), ((r_up - 1, c1), "turn2")]
    top_col = random.randint(1, notch_start - 1)

    if scenario == "single":
        middle_rows = list(range(r_up + 1, r_mid))
        row = random.choice(middle_rows)
        markers.extend([((r0 - 1, top_col), "top"), ((row, 0), "middle")])
    elif scenario == "tie":
        rows = _sample_nonadjacent(list(range(r_up + 1, r_mid)), 2)
        if rows is None:
            raise AssertionError("tie scenario needs two separated middle rows")
        markers.extend(
            [
                ((r0 - 1, top_col), "top"),
                ((rows[0], 0), "middle_a"),
                ((rows[1], 0), "middle_b"),
            ]
        )
    else:
        rows = _sample_nonadjacent(list(range(r_up + 1, r_mid)), 3)
        if rows is None:
            raise AssertionError("count scenario needs three separated middle rows")
        markers.extend(
            [
                ((rows[0], 0), "middle_a1"),
                ((rows[1], 0), "middle_a2"),
                ((rows[2], 0), "middle_b"),
            ]
        )
    return markers


def _assign_marker_colors(scenario: str) -> dict[str, int]:
    a, b, c = random.sample(MARKER_COLORS, 3)
    if scenario == "single":
        if random.random() < 0.5:
            return {"turn1": a, "turn2": b, "top": c, "middle": a}
        return {"turn1": a, "turn2": b, "top": c, "middle": c}
    if scenario == "tie":
        return {
            "turn1": a,
            "turn2": c,
            "top": a,
            "middle_a": a,
            "middle_b": b,
        }
    return {
        "turn1": c,
        "turn2": b,
        "middle_a1": a,
        "middle_a2": a,
        "middle_b": b,
    }


def _build_candidate() -> np.ndarray | None:
    scenario = random.choices(
        ["single", "tie", "count"], weights=[2, 2, 3], k=1
    )[0]
    rect_h = random.randint(RECT_MIN_H, RECT_MAX_H)
    rect_w = random.randint(RECT_MIN_W, RECT_MAX_W)
    h = rect_h + random.randint(3, 5)
    w = rect_w + random.randint(3, 5)
    r0 = random.randint(1, h - rect_h - 1)
    c0 = 1
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    notch_depth = random.randint(2, min(3, rect_h - 6))
    r_up = r0 + notch_depth
    middle_gap = 6 if scenario == "count" else 4
    if r_up + middle_gap > r1 - 1:
        return None
    r_mid = random.randint(r_up + middle_gap, r1 - 1)

    notch_start_min = c0 + 4
    notch_start_max = c1 - 2
    if notch_start_min > notch_start_max:
        return None
    notch_start = random.randint(notch_start_min, notch_start_max)

    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask[r0:r_up, notch_start : c1 + 1] = False
    area = int(mask.sum())
    if not (50 <= area <= 80):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[mask] = GREY
    grid[r_mid, 0] = ORANGE

    try:
        marker_specs = _marker_layout(scenario, r0, r1, c1, r_up, r_mid, notch_start)
    except AssertionError:
        return None
    marker_colors = _assign_marker_colors(scenario)
    used_positions = {spec[0] for spec in marker_specs}
    if len(used_positions) != len(marker_specs):
        return None

    for pos, role in marker_specs:
        r, c = pos
        if not _in_bounds(grid.shape, r, c):
            return None
        if grid[r, c] != 0:
            return None
        grid[r, c] = int(marker_colors[role])

    try:
        analysis = _analyze(grid)
    except AssertionError:
        return None

    if analysis["output"].shape != grid.shape:
        return None
    if len(analysis["turn_indices"]) != 2:
        return None
    if analysis["branch_sizes"][0] != 2:
        return None
    if len(analysis["fragments"]) != 3:
        return None
    if not any(frag["color"] == GREY for frag in analysis["fragments"]):
        return None
    if np.array_equal(analysis["output"], (analysis["mask"] & ~analysis["cut"]).astype(int) * GREY):
        return None

    if scenario == "tie":
        if not any(
            len(frag["contact_counts"]) >= 2
            and len({v for v in frag["contact_counts"].values()}) == 1
            for frag in analysis["fragments"]
        ):
            return None
    if scenario == "count":
        if not any(
            frag["contact_counts"]
            and max(frag["contact_counts"].values()) >= 2
            and len(frag["contact_counts"]) >= 2
            for frag in analysis["fragments"]
        ):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
