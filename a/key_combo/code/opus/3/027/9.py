from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _fallback_bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | None = None,
    min_area: int = 1,
    return_format: str = "dict",
):
    if bg is None:
        bg = _fallback_get_bg(grid)
    if not monochromatic or return_format != "dict":
        raise NotImplementedError("fallback find_components only supports monochromatic dict mode")

    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    h, w = grid.shape
    if colors is None:
        colors_iter = sorted(set(map(int, grid.ravel())) - {int(bg)})
    else:
        colors_iter = [int(color) for color in colors if int(color) != int(bg)]

    out = []
    for color in colors_iter:
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in steps:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                if len(cells) < int(min_area):
                    continue
                coords = np.array(cells, dtype=int)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _fallback_bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
    return out


try:
    if not _ARC_UTILS_PATH.exists():
        raise FileNotFoundError(_ARC_UTILS_PATH)
    _arc_utils = SourceFileLoader("_arc_utils_027_9", str(_ARC_UTILS_PATH)).load_module()
    get_bg = _arc_utils.get_bg
    find_components = _arc_utils.find_components
except Exception:
    get_bg = _fallback_get_bg
    find_components = _fallback_find_components


BG = 0
TARGET = 5
OBSTACLE_COLORS = (1, 2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
SCENARIOS = ("two_color", "three_color_left", "three_color_right")
_GEN_CALLS = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _ray_votes(grid: np.ndarray, r: int, c: int) -> list[int]:
    votes: list[int] = []
    h, w = grid.shape
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        while 0 <= rr < h and 0 <= cc < w:
            value = int(grid[rr, cc])
            if value == BG or value == TARGET:
                rr += dr
                cc += dc
                continue
            assert value in OBSTACLE_COLORS, f"unexpected ray obstacle color {value}"
            votes.append(value)
            break
    return votes


def _majority_color(votes: list[int]) -> int:
    counts = Counter(votes)
    assert counts, "each target must receive at least one vote"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "target votes must have a unique majority"
    return int(winners[0])


def _segment_info(component: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    length = max(h, w)
    assert (h == 1) ^ (w == 1), "obstacles must be one-cell-thick line segments"
    assert int(component["area"]) == length, "segments must be solid with no gaps"
    assert 3 <= length <= 6, "segment length must be within 3..6"
    return r0, c0, r1, c1, length


def _classify_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    targets: list[tuple[int, int]] = []
    obstacle_counts: Counter[int] = Counter()
    obstacle_owner = {}
    obstacle_cells = set()

    for idx, component in enumerate(components):
        color = int(component["color"])
        coords = np.asarray(component["coords"], dtype=int)
        if color == TARGET:
            assert int(component["area"]) == 1, "grey targets must be isolated single pixels"
            r, c = map(int, coords[0])
            targets.append((r, c))
            continue

        assert color in OBSTACLE_COLORS, f"unexpected foreground color {color}"
        _segment_info(component)
        obstacle_counts[color] += 1
        for r, c in coords:
            cell = (int(r), int(c))
            obstacle_cells.add(cell)
            obstacle_owner[cell] = idx

    assert 2 <= len(obstacle_counts) <= 3, "input must use 2 or 3 obstacle colors"
    assert all(1 <= count <= 3 for count in obstacle_counts.values()), "each obstacle color needs 1..3 segments"

    return {
        "targets": targets,
        "obstacle_cells": obstacle_cells,
        "obstacle_owner": obstacle_owner,
        "obstacle_counts": obstacle_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey target by the majority color of the first obstacle segments hit along its four cardinal rays.

    Input structure:
    - The background is black.
    - Non-grey foreground cells form separate horizontal or vertical one-cell-thick obstacle segments in 2 or 3 colors.
    - Grey cells are isolated single-pixel targets and are not 4-neighbor-adjacent to any obstacle segment.

    Rule:
    - Identify every grey singleton as a target pixel.
    - For each target, cast rays upward, downward, leftward, and rightward; black and grey cells are transparent, and the first non-black non-grey cell hit in a direction contributes one vote of its color.
    - If a direction reaches the border without hitting an obstacle, it contributes no vote; the target's new color is the unique majority among all collected votes.
    - Recolor only the grey target pixels; keep the background and all obstacle segments unchanged.

    Key insight:
    - The rays are invisible probes, not drawing operations: line-of-sight to the nearest obstacle in each direction determines the vote set, so nearby targets in the same open region can still recolor differently.
    """
    g = np.asarray(grid, dtype=int)
    scene = _classify_scene(g)
    out = g.copy()

    for r, c in scene["targets"]:
        votes = _ray_votes(g, r, c)
        assert len(votes) >= 2, "each target must see obstacles in at least two directions"
        out[r, c] = _majority_color(votes)

    return out


def _paint_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    grid[int(row), int(c0) : int(c1) + 1] = int(color)


def _paint_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    grid[int(r0) : int(r1) + 1, int(col)] = int(color)


def _scenario_assignment(scenario: str) -> dict[str, int]:
    if scenario == "two_color":
        a, b = random.sample(OBSTACLE_COLORS, 2)
        return {"UL": a, "DL": a, "LV": a, "UR": b, "DR": b, "RV": b}
    if scenario == "three_color_left":
        a, b, c = random.sample(OBSTACLE_COLORS, 3)
        return {"UL": a, "DL": a, "LV": c, "UR": b, "DR": b, "RV": b}
    if scenario == "three_color_right":
        a, b, c = random.sample(OBSTACLE_COLORS, 3)
        return {"UL": a, "DL": a, "LV": a, "UR": b, "DR": b, "RV": c}
    raise ValueError(f"unknown scenario {scenario}")


def _random_dihedral(grid: np.ndarray) -> np.ndarray:
    transformed = np.asarray(grid, dtype=int)
    k = random.randrange(4)
    if k:
        transformed = np.rot90(transformed, k)
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _passable_region_ids(grid: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    passable = (grid == BG) | (grid == TARGET)
    seen = np.zeros((h, w), dtype=bool)
    region_of: dict[tuple[int, int], int] = {}
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                region_of[(int(rr), int(cc))] = next_id
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            next_id += 1
    return region_of


def _validate_generated_grid(grid: np.ndarray) -> None:
    scene = _classify_scene(grid)
    targets = scene["targets"]
    obstacle_cells = scene["obstacle_cells"]
    obstacle_owner = scene["obstacle_owner"]

    if len(targets) < 4:
        raise AssertionError("generator should place several targets")

    for r, c in targets:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in obstacle_cells:
                raise AssertionError("targets must not touch obstacles in the 4-neighborhood")
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == TARGET:
                if (nr, nc) != (r, c):
                    raise AssertionError("grey targets should stay visually separated")

    for (r, c), owner in obstacle_owner.items():
        for nr, nc in _neighbors8(r, c):
            other = obstacle_owner.get((nr, nc))
            if other is not None and other != owner:
                raise AssertionError("obstacle segments should not touch even diagonally")

    winners = []
    hit_counts = []
    mixed_four_hit = False
    for r, c in targets:
        votes = _ray_votes(grid, r, c)
        if len(votes) < 2:
            raise AssertionError("every target must receive at least two votes")
        winner = _majority_color(votes)
        winners.append(winner)
        hit_counts.append(len(votes))
        counts = Counter(votes)
        if len(votes) == 4 and len(counts) >= 2 and max(counts.values()) >= 2:
            mixed_four_hit = True

    expected = {target: winner for target, winner in zip(targets, winners)}

    if len(set(winners)) < 2:
        raise AssertionError("all targets voting to one color makes region shortcuts too plausible")
    if 2 not in hit_counts:
        raise AssertionError("generator must include a two-hit target branch")
    if not mixed_four_hit:
        raise AssertionError("generator must include a four-hit mixed-color target")

    region_of = _passable_region_ids(grid)
    found_same_region_disagreement = False
    for i, (r0, c0) in enumerate(targets):
        for j in range(i + 1, len(targets)):
            r1, c1 = targets[j]
            if region_of[(r0, c0)] == region_of[(r1, c1)] and winners[i] != winners[j]:
                found_same_region_disagreement = True
                break
        if found_same_region_disagreement:
            break
    if not found_same_region_disagreement:
        raise AssertionError("need two targets in the same open region with different winners")

    out = main(grid)
    assert out.shape == grid.shape
    for r, c in targets:
        if int(out[r, c]) != expected[(r, c)]:
            raise AssertionError("main() disagrees with direct ray-vote evaluation")
    changed = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == TARGET:
                changed += int(out[r, c] != grid[r, c])
            elif int(out[r, c]) != int(grid[r, c]):
                raise AssertionError("only target pixels may change")
    if changed != len(targets):
        raise AssertionError("every grey target should recolor")


def _build_scaffold(scenario: str) -> np.ndarray:
    gap = 3
    side_gap = 1
    left_len = random.randint(3, 6)
    right_len = random.randint(3, 6)
    vert_len = random.randint(3, 6)
    extra_bottom = random.randint(2, 4)
    extra_right = random.randint(2, 4)

    top_row = random.randint(1, 2)
    left_col = random.randint(1, 2)
    row_lo = top_row + gap
    row_hi = row_lo + vert_len - 1
    bottom_row = row_hi + gap

    left_start = left_col + gap
    left_end = left_start + left_len - 1
    right_start = left_end + side_gap + 1
    right_end = right_start + right_len - 1
    right_col = right_end + gap

    h = bottom_row + extra_bottom + 1
    w = right_col + extra_right + 1
    grid = np.zeros((h, w), dtype=int)

    colors = _scenario_assignment(scenario)
    _paint_h(grid, top_row, left_start, left_end, colors["UL"])
    _paint_h(grid, top_row, right_start, right_end, colors["UR"])
    _paint_h(grid, bottom_row, left_start, left_end, colors["DL"])
    _paint_h(grid, bottom_row, right_start, right_end, colors["DR"])
    _paint_v(grid, left_col, row_lo, row_hi, colors["LV"])
    _paint_v(grid, right_col, row_lo, row_hi, colors["RV"])

    inner_rows = list(range(row_lo + 1, row_hi))
    assert inner_rows, "vertical span must leave at least one safe interior row"
    top_partial = row_lo - 1
    bottom_partial = row_hi + 1
    left_cols = list(range(left_start, left_end + 1))
    right_cols = list(range(right_start, right_end + 1))

    targets = [
        (random.choice(inner_rows), random.choice(left_cols)),
        (random.choice(inner_rows), random.choice(right_cols)),
        (top_partial, random.choice(left_cols)),
        (bottom_partial, random.choice(right_cols)),
    ]

    extras = [
        (bottom_partial, random.choice(left_cols)),
        (top_partial, random.choice(right_cols)),
        (random.choice(inner_rows), random.choice(left_cols)),
        (random.choice(inner_rows), random.choice(right_cols)),
    ]
    random.shuffle(extras)
    for candidate in extras[: random.randint(0, 1)]:
        if all(max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 2 for r, c in targets):
            targets.append(candidate)

    for r, c in targets:
        grid[int(r), int(c)] = TARGET

    return grid


def _gen_attempt(scenario: str) -> np.ndarray | None:
    grid = _build_scaffold(scenario)
    grid = _random_dihedral(grid)
    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    scenario = SCENARIOS[_GEN_CALLS % len(SCENARIOS)]
    _GEN_CALLS += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle-9 input after {GEN_ATTEMPTS} attempts")
