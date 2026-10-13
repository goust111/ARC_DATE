from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 500


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == YELLOW)]
    assert len(rows) == 1, "expected one yellow separator row"
    return int(rows[0])


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    assert sep * 2 + 1 == g.shape[0], "separator must split two equal-height panels"
    top = g[:sep, :].copy()
    bottom = g[sep + 1 :, :].copy()
    assert top.shape == bottom.shape
    assert np.array_equal(top == GREY, bottom == GREY), "panels must share the same obstacle layout"
    return top, bottom


def _drop_pixel(occupied: np.ndarray, start: tuple[int, int]) -> tuple[tuple[int, int], int, tuple[str, ...]]:
    h, w = occupied.shape
    r, c = map(int, start)
    steps = 0
    flags: list[str] = []

    while True:
        if r + 1 < h and not occupied[r + 1, c]:
            r += 1
            steps += 1
            continue

        if r + 1 >= h:
            break

        left_open = c - 1 >= 0 and not occupied[r + 1, c - 1]
        right_open = c + 1 < w and not occupied[r + 1, c + 1]
        if left_open and right_open:
            r += 1
            c -= 1
            steps += 1
            flags.append("both_open_left")
            continue
        if left_open:
            r += 1
            c -= 1
            steps += 1
            flags.append("left_only")
            continue
        if right_open:
            r += 1
            c += 1
            steps += 1
            flags.append("right_only")
            continue
        break

    return (int(r), int(c)), int(steps), tuple(flags)


def _simulate_panel(panel: np.ndarray, particle_color: int) -> tuple[np.ndarray, list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(panel, dtype=int)
    h, w = g.shape
    occupied = g == GREY
    settled = np.where(occupied, GREY, BLACK)
    sources = sorted((int(r), int(c)) for r, c in np.argwhere(g == particle_color))
    infos: list[dict] = []
    steps_by_final: dict[tuple[int, int], int] = {}

    for source in sources:
        final, steps, flags = _drop_pixel(occupied, source)
        assert not occupied[final], "source landed on an occupied cell"
        occupied[final] = True
        settled[final] = particle_color
        assert source not in steps_by_final, "source positions must be unique"
        steps_by_final[final] = steps
        infos.append(
            {
                "source": source,
                "final": final,
                "steps": int(steps),
                "flags": flags,
            }
        )

    assert settled.shape == (h, w)
    return settled, infos, steps_by_final


def _merge_panels(
    top_steps: dict[tuple[int, int], int],
    bottom_steps: dict[tuple[int, int], int],
    obstacle_mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros(obstacle_mask.shape, dtype=int)
    out[obstacle_mask] = GREY

    cells = set(top_steps) | set(bottom_steps)
    for cell in cells:
        if obstacle_mask[cell]:
            continue
        has_top = cell in top_steps
        has_bottom = cell in bottom_steps
        if has_top and not has_bottom:
            out[cell] = RED
        elif has_bottom and not has_top:
            out[cell] = BLUE
        else:
            out[cell] = RED if top_steps[cell] >= bottom_steps[cell] else BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Independently settle the two separator-stacked panels, then overlay them using fall distance to break collisions.

    Input structure:
    - A single yellow row splits the grid into two equal-height panels of the same width.
    - The upper panel contains grey obstacles plus red source pixels; the lower panel contains the same grey obstacles plus blue source pixels.
    - Obstacles occupy the same coordinates in both panels, while all other cells are black background.

    Rule:
    - Simulate each panel independently. Process same-color source pixels in original top-to-bottom, left-to-right order.
    - A pixel falls straight down through black cells; if the cell below is blocked, it moves to left-down when open, otherwise to right-down when only that is open, and when both diagonals are open it chooses left-down. Each one-cell move counts as one fall step. A pixel stops when no move is possible.
    - Remove the separator and align the two settled panels cell by cell. Grey obstacle cells stay grey. A cell occupied by only one settled pixel keeps that pixel's color. A red-blue collision is resolved by larger recorded fall distance, with red winning ties.

    Key insight:
    - The decisive information is not only where each panel's pixels end up, but how far each pixel had to travel to get there.
    """
    top, bottom = _split_input(grid)
    top_settled, _top_infos, top_steps = _simulate_panel(top, RED)
    bottom_settled, _bottom_infos, bottom_steps = _simulate_panel(bottom, BLUE)
    obstacle_mask = top == GREY
    assert np.array_equal(top_settled == GREY, obstacle_mask)
    assert np.array_equal(bottom_settled == GREY, obstacle_mask)
    return _merge_panels(top_steps, bottom_steps, obstacle_mask)


def _ascending_profile(length: int, low: int, high: int) -> list[int]:
    diff = int(high - low)
    assert 0 <= diff <= length
    incs = [1] * diff + [0] * (length - diff)
    random.shuffle(incs)
    vals = [int(low)]
    cur = int(low)
    for inc in incs:
        cur += int(inc)
        vals.append(int(cur))
    return vals


def _make_basin_profile(panel_h: int) -> list[int]:
    width = random.randint(5, 7)
    floor = random.randint(0, 1)
    min_col = random.randint(1, width - 2)
    left_max = min(panel_h - 2, floor + min_col)
    right_max = min(panel_h - 2, floor + (width - 1 - min_col))
    left_rim = random.randint(floor + 1, left_max)
    right_rim = random.randint(floor + 1, right_max)

    left = list(reversed(_ascending_profile(min_col, floor, left_rim)))
    right = _ascending_profile(width - 1 - min_col, floor, right_rim)
    return left + right[1:]


def _build_profile(panel_h: int) -> list[int] | None:
    n_basins = random.randint(2, 3)
    parts: list[int] = []
    for i in range(n_basins):
        if i:
            ridge_w = random.randint(1, 2)
            ridge_h = random.randint(max(3, panel_h - 3), panel_h - 2)
            parts.extend([ridge_h] * ridge_w)
        parts.extend(_make_basin_profile(panel_h))

    left_pad = [random.randint(0, 2) for _ in range(random.randint(0, 1))]
    right_pad = [random.randint(0, 2) for _ in range(random.randint(0, 1))]
    profile = left_pad + parts + right_pad
    return profile if 14 <= len(profile) <= 24 else None


def _panel_from_profile(panel_h: int, profile: list[int]) -> np.ndarray:
    panel = np.zeros((panel_h, len(profile)), dtype=int)
    for c, height in enumerate(profile):
        if height > 0:
            panel[panel_h - height :, c] = GREY
    return panel


def _single_candidate(obstacle_panel: np.ndarray, start: tuple[int, int]) -> dict:
    occupied = obstacle_panel == GREY
    final, steps, flags = _drop_pixel(occupied, start)
    return {
        "source": (int(start[0]), int(start[1])),
        "final": final,
        "steps": int(steps),
        "flags": flags,
    }


def _pair_score(a: dict, b: dict) -> tuple[int, int, int, int, int]:
    same_row_diff = int(a["source"][0] == b["source"][0] and a["steps"] != b["steps"])
    reversal = int(
        (a["source"][0] < b["source"][0] and a["steps"] < b["steps"])
        or (a["source"][0] > b["source"][0] and a["steps"] > b["steps"])
    )
    both_open = int("both_open_left" in a["flags"] or "both_open_left" in b["flags"])
    right_only = int("right_only" in a["flags"] or "right_only" in b["flags"])
    diagonal = int(bool(a["flags"]) or bool(b["flags"]))
    return (
        4 * same_row_diff + 3 * reversal + 2 * both_open + right_only + diagonal,
        abs(a["steps"] - b["steps"]),
        same_row_diff,
        reversal,
        both_open,
    )


def _candidate_groups(obstacle_panel: np.ndarray) -> tuple[list[dict], list[dict]]:
    h, w = obstacle_panel.shape
    start_rows = range(0, max(2, h // 2 + 1))
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    all_candidates: list[dict] = []

    for r in start_rows:
        for c in range(w):
            if obstacle_panel[r, c] != BLACK:
                continue
            cand = _single_candidate(obstacle_panel, (r, c))
            if cand["final"][0] < h // 2:
                continue
            all_candidates.append(cand)
            groups[cand["final"]].append(cand)

    options: list[dict] = []
    for final, group in groups.items():
        pair_options = []
        for a, b in combinations(group, 2):
            if a["steps"] == b["steps"]:
                continue
            pair_options.append({"a": a, "b": b, "score": _pair_score(a, b)})
        if not pair_options:
            continue
        pair_options.sort(key=lambda item: item["score"], reverse=True)
        options.append({"final": final, "pairs": pair_options})

    options.sort(key=lambda item: item["pairs"][0]["score"], reverse=True)
    return options, all_candidates


def _choose_conflicts(group_options: list[dict]) -> list[dict] | None:
    if len(group_options) < 2:
        return None

    best: tuple[int, list[dict]] | None = None
    top_groups = group_options[:8]
    for g1, g2 in combinations(top_groups, 2):
        if abs(g1["final"][1] - g2["final"][1]) < 4:
            continue
        for pair1, pair2 in product(g1["pairs"][:4], g2["pairs"][:4]):
            pair_specs = []
            seen_starts = set()
            total_score = 0
            for idx, pair in enumerate((pair1, pair2)):
                a = pair["a"]
                b = pair["b"]
                red_cand, blue_cand = (a, b) if idx == 0 else (b, a)
                if red_cand["source"] in seen_starts or blue_cand["source"] in seen_starts:
                    break
                seen_starts.add(red_cand["source"])
                seen_starts.add(blue_cand["source"])
                pair_specs.append({"red": red_cand, "blue": blue_cand})
                total_score += pair["score"][0] * 10 + pair["score"][1]
            else:
                nontrivial_conflict = any(
                    spec["red"]["source"][1] != spec["blue"]["source"][1]
                    and (spec["red"]["flags"] or spec["blue"]["flags"])
                    for spec in pair_specs
                )
                diagonal = any(spec["red"]["flags"] or spec["blue"]["flags"] for spec in pair_specs)
                if not nontrivial_conflict or not diagonal:
                    continue
                score = total_score + 20 * int(
                    any("both_open_left" in spec["red"]["flags"] or "both_open_left" in spec["blue"]["flags"] for spec in pair_specs)
                )
                if best is None or score > best[0]:
                    best = (score, pair_specs)
    return None if best is None else best[1]


def _add_branch_extra(
    chosen: dict[str, list[tuple[int, int]]],
    all_candidates: list[dict],
    required_flag: str,
) -> bool:
    used_starts = set(chosen["red"]) | set(chosen["blue"])
    used_finals = set(chosen["red_finals"]) | set(chosen["blue_finals"])
    for cand in all_candidates:
        if required_flag not in cand["flags"]:
            continue
        if cand["source"] in used_starts or cand["final"] in used_finals:
            continue
        if any(abs(cand["final"][1] - fc[1]) < 3 for fc in used_finals):
            continue
        target = "red" if len(chosen["red"]) <= len(chosen["blue"]) else "blue"
        chosen[target].append(cand["source"])
        chosen[f"{target}_finals"].append(cand["final"])
        return True
    return False


def _assemble_input(obstacle_panel: np.ndarray, red_starts: list[tuple[int, int]], blue_starts: list[tuple[int, int]]) -> np.ndarray:
    top = obstacle_panel.copy()
    bottom = obstacle_panel.copy()
    for r, c in red_starts:
        assert top[r, c] == BLACK
        top[r, c] = RED
    for r, c in blue_starts:
        assert bottom[r, c] == BLACK
        bottom[r, c] = BLUE
    separator = np.full((1, obstacle_panel.shape[1]), YELLOW, dtype=int)
    return np.vstack([top, separator, bottom])


def _semantic_checks(grid: np.ndarray, pair_specs: list[dict], require_both_open: bool) -> bool:
    try:
        top, bottom = _split_input(grid)
        _out = main(grid)
        _top_settled, top_infos, top_steps = _simulate_panel(top, RED)
        _bottom_settled, bottom_infos, bottom_steps = _simulate_panel(bottom, BLUE)
    except Exception:
        return False

    top_by_source = {info["source"]: info for info in top_infos}
    bottom_by_source = {info["source"]: info for info in bottom_infos}

    if len(set(info["final"] for info in top_infos)) != len(top_infos):
        return False
    if len(set(info["final"] for info in bottom_infos)) != len(bottom_infos):
        return False
    if set(top_by_source) & {info["final"] for info in top_infos}:
        return False
    if set(bottom_by_source) & {info["final"] for info in bottom_infos}:
        return False

    conflicts = set(top_steps) & set(bottom_steps)
    if len(conflicts) < 2:
        return False
    if any(top_steps[cell] == bottom_steps[cell] for cell in conflicts):
        return False

    nontrivial_conflict = False
    any_diagonal = False
    any_both_open = False
    red_wins = 0
    blue_wins = 0
    for spec in pair_specs:
        red_info = top_by_source.get(spec["red"]["source"])
        blue_info = bottom_by_source.get(spec["blue"]["source"])
        if red_info is None or blue_info is None:
            return False
        if red_info["final"] != blue_info["final"]:
            return False
        if red_info["steps"] == blue_info["steps"]:
            return False
        if red_info["steps"] > blue_info["steps"]:
            red_wins += 1
        else:
            blue_wins += 1
        any_diagonal = any_diagonal or bool(red_info["flags"]) or bool(blue_info["flags"])
        any_both_open = any_both_open or ("both_open_left" in red_info["flags"]) or ("both_open_left" in blue_info["flags"])
        nontrivial_conflict = nontrivial_conflict or (
            red_info["source"][1] != blue_info["source"][1]
            and (red_info["flags"] or blue_info["flags"])
        )

    if red_wins == 0 or blue_wins == 0:
        return False
    if not nontrivial_conflict or not any_diagonal:
        return False
    if require_both_open and not any_both_open:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(5, 7)
    profile = _build_profile(panel_h)
    if profile is None:
        return None
    obstacle_panel = _panel_from_profile(panel_h, profile)
    group_options, all_candidates = _candidate_groups(obstacle_panel)
    pair_specs = _choose_conflicts(group_options)
    if pair_specs is None:
        return None

    chosen = {
        "red": [spec["red"]["source"] for spec in pair_specs],
        "blue": [spec["blue"]["source"] for spec in pair_specs],
        "red_finals": [spec["red"]["final"] for spec in pair_specs],
        "blue_finals": [spec["blue"]["final"] for spec in pair_specs],
    }

    require_both_open = random.random() < 0.7
    if require_both_open:
        has_both_open = any(
            "both_open_left" in spec["red"]["flags"] or "both_open_left" in spec["blue"]["flags"]
            for spec in pair_specs
        )
        if not has_both_open and not _add_branch_extra(chosen, all_candidates, "both_open_left"):
            return None

    if random.random() < 0.5:
        _add_branch_extra(chosen, all_candidates, "right_only")

    grid = _assemble_input(obstacle_panel, chosen["red"], chosen["blue"])
    return grid if _semantic_checks(grid, pair_specs, require_both_open) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
