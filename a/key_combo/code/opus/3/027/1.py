from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


def _ensure_arc_utils() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = candidate.parent
            if str(utils_dir) not in sys.path:
                sys.path.append(str(utils_dir))
            return
    raise ImportError("could not locate new/solvers/arc_utils.py from target module path")


_ensure_arc_utils()

from arc_utils import Color, components_mask  # noqa: E402


Box = tuple[int, int, int, int]


GRID_H_RANGE = (12, 17)
GRID_W_RANGE = (12, 17)
GEN_ATTEMPTS = 400
MIN_REGION_SPAN = 2
MAX_BRANCHES_PER_SIDE = 3
EXTRA_ROLE_WEIGHTS = {
    "pair": 0.45,
    "green_only": 0.15,
    "yellow_only": 0.15,
    "empty": 0.25,
}


def _sign(delta: int) -> int:
    return int((delta > 0) - (delta < 0))


def _bbox_from_coords(coords: np.ndarray) -> Box:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _open_regions(grid: np.ndarray) -> list[np.ndarray]:
    return components_mask(grid != Color.GREY, connectivity=4)


def _region_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    pts = [tuple(map(int, rc)) for rc in coords]
    greens = [pt for pt in pts if int(grid[pt]) == Color.GREEN]
    yellows = [pt for pt in pts if int(grid[pt]) == Color.YELLOW]
    return {
        "coords": pts,
        "bbox": _bbox_from_coords(coords),
        "greens": greens,
        "yellows": yellows,
        "touches_boundary": any(
            r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in pts
        ),
    }


def _aligned_delta(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    assert (dr, dc) != (0, 0)
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "marker and target must share a straight ARC line"
    return dr, dc


def _line_kind(step: tuple[int, int]) -> str:
    if step[0] == 0:
        return "horizontal"
    if step[1] == 0:
        return "vertical"
    return "diagonal"


def _pair_distance(src: tuple[int, int], dst: tuple[int, int]) -> int:
    dr, dc = _aligned_delta(src, dst)
    return max(abs(dr), abs(dc))


def _region_role(info: dict) -> str:
    if info["greens"] and info["yellows"]:
        return "pair"
    if info["greens"]:
        return "green_only"
    if info["yellows"]:
        return "yellow_only"
    return "empty"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each green marker one step toward the yellow target that shares its wall-separated open region.

    Input structure:
    - Black cells are background, grey cells are single-pixel-thick wall segments, green cells are markers, and yellow cells are targets.
    - The grey wall layout partitions the non-wall cells into several 4-connected open regions; each region contains at most one green and at most one yellow.
    - Any region that contains both colors places them on the same row, same column, or a 45-degree diagonal.

    Rule:
    - Flood-fill all non-grey cells with 4-connectivity to recover the wall-separated regions.
    - For each region containing exactly one green and one yellow, compute the normalized direction from green to yellow and move the green one cell along that direction.
    - Regions containing only one of the two markers, or neither, remain unchanged; walls and yellow targets always stay fixed.

    Key insight:
    - Pairing is defined by shared open-region membership, not by geometric proximity across walls, so a green can move away from a visually nearer yellow in another region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _open_regions(g):
        info = _region_info(g, comp)
        assert len(info["greens"]) <= 1 and len(info["yellows"]) <= 1
        if len(info["greens"]) != 1 or len(info["yellows"]) != 1:
            continue

        green = info["greens"][0]
        yellow = info["yellows"][0]
        dr, dc = _aligned_delta(green, yellow)
        step = (_sign(dr), _sign(dc))
        nr = int(green[0] + step[0])
        nc = int(green[1] + step[1])
        assert int(g[nr, nc]) == Color.BLACK, "green must advance into background"

        out[green] = Color.BLACK
        out[nr, nc] = Color.GREEN

    return out


def _bounded_partition(total: int, parts: int, lo: int) -> list[int] | None:
    if total < parts * lo:
        return None
    vals = [lo] * parts
    extra = total - parts * lo
    while extra > 0:
        idx = random.randrange(parts)
        vals[idx] += 1
        extra -= 1
    random.shuffle(vals)
    return vals


def _branch_rows(height: int, count: int) -> list[int] | None:
    region_heights = _bounded_partition(height - count, count + 1, MIN_REGION_SPAN)
    if region_heights is None:
        return None
    rows = []
    consumed = 0
    for idx, size in enumerate(region_heights[:-1]):
        consumed += size
        rows.append(consumed + idx)
    return rows


def _vertical_layout() -> tuple[np.ndarray, list[dict], int] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    if width < 12:
        return None

    max_left = min(MAX_BRANCHES_PER_SIDE, (height - MIN_REGION_SPAN) // (MIN_REGION_SPAN + 1))
    if max_left < 1:
        return None

    for _ in range(60):
        left_branches = random.randint(1, max_left)
        right_branches = random.randint(1, max_left)
        if left_branches + right_branches < 3:
            continue

        left_rows = _branch_rows(height, left_branches)
        right_rows = _branch_rows(height, right_branches)
        if left_rows is None or right_rows is None:
            continue

        trunk_col = random.randint(5, width - 6)
        grid = np.full((height, width), Color.BLACK, dtype=int)
        grid[:, trunk_col] = Color.GREY
        for r in left_rows:
            grid[r, :trunk_col] = Color.GREY
        for r in right_rows:
            grid[r, trunk_col + 1 :] = Color.GREY

        regions: list[dict] = []
        prev = -1
        for row in left_rows + [height]:
            bbox = (prev + 1, 0, row - 1, trunk_col - 1)
            regions.append({"side": "L", "bbox": bbox})
            prev = row
        prev = -1
        for row in right_rows + [height]:
            bbox = (prev + 1, trunk_col + 1, row - 1, width - 1)
            regions.append({"side": "R", "bbox": bbox})
            prev = row
        return grid, regions, trunk_col
    return None


def _cells_in_box(box: Box) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _pair_candidates(box: Box) -> list[dict]:
    cells = _cells_in_box(box)
    out = []
    for green in cells:
        for yellow in cells:
            if green == yellow:
                continue
            dr = yellow[0] - green[0]
            dc = yellow[1] - green[1]
            if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                continue
            dist = max(abs(dr), abs(dc))
            if dist < 2:
                continue
            step = (_sign(dr), _sign(dc))
            out.append(
                {
                    "green": green,
                    "yellow": yellow,
                    "dist": dist,
                    "step": step,
                    "kind": _line_kind(step),
                }
            )
    return out


def _choose_pair(
    box: Box,
    *,
    forbidden_dists: set[int] | None = None,
    preferred_kinds: set[str] | None = None,
) -> dict | None:
    candidates = _pair_candidates(box)
    if not candidates:
        return None
    random.shuffle(candidates)

    if forbidden_dists:
        filtered = [cand for cand in candidates if cand["dist"] not in forbidden_dists]
        if filtered:
            candidates = filtered

    if preferred_kinds:
        filtered = [cand for cand in candidates if cand["kind"] in preferred_kinds]
        if filtered:
            candidates = filtered

    return random.choice(candidates)


def _region_for_row(regions: list[dict], row: int) -> dict | None:
    for region in regions:
        r0, _c0, r1, _c1 = region["bbox"]
        if r0 <= row <= r1:
            return region
    return None


def _place_single(grid: np.ndarray, box: Box, color: int) -> tuple[int, int]:
    cell = random.choice(_cells_in_box(box))
    grid[cell] = int(color)
    return cell


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _validate_generated_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.any(g == Color.GREY):
        return False

    infos = [_region_info(g, comp) for comp in _open_regions(g)]
    if len(infos) < 5:
        return False
    if not all(info["touches_boundary"] for info in infos):
        return False
    if any(len(info["greens"]) > 1 or len(info["yellows"]) > 1 for info in infos):
        return False

    roles = [_region_role(info) for info in infos]
    if roles.count("pair") < 2:
        return False
    if "green_only" not in roles or "yellow_only" not in roles or "empty" not in roles:
        return False

    pair_infos = [info for info in infos if _region_role(info) == "pair"]
    pair_distances = set()
    for info in pair_infos:
        green = info["greens"][0]
        yellow = info["yellows"][0]
        dr, dc = _aligned_delta(green, yellow)
        step = (_sign(dr), _sign(dc))
        dest = (green[0] + step[0], green[1] + step[1])
        if int(g[dest]) != Color.BLACK:
            return False
        pair_distances.add(max(abs(dr), abs(dc)))
    if len(pair_distances) < 2:
        return False

    all_yellows = [info["yellows"][0] for info in infos if info["yellows"]]
    has_trap = False
    for info in pair_infos:
        green = info["greens"][0]
        own = info["yellows"][0]
        own_step = (_sign(own[0] - green[0]), _sign(own[1] - green[1]))
        own_manhattan = abs(own[0] - green[0]) + abs(own[1] - green[1])
        own_chebyshev = max(abs(own[0] - green[0]), abs(own[1] - green[1]))
        for other in all_yellows:
            if other == own:
                continue
            other_step = (_sign(other[0] - green[0]), _sign(other[1] - green[1]))
            other_manhattan = abs(other[0] - green[0]) + abs(other[1] - green[1])
            other_chebyshev = max(abs(other[0] - green[0]), abs(other[1] - green[1]))
            if (
                other_manhattan < own_manhattan
                and other_chebyshev < own_chebyshev
                and other_step != own_step
            ):
                has_trap = True
                break
        if has_trap:
            break
    if not has_trap:
        return False

    out = main(g)
    if np.any(out[g == Color.GREY] != Color.GREY):
        return False
    if np.any(out[g == Color.YELLOW] != Color.YELLOW):
        return False

    diff = np.argwhere(out != g)
    if int(diff.shape[0]) != 2 * len(pair_infos):
        return False
    for info in pair_infos:
        green = info["greens"][0]
        yellow = info["yellows"][0]
        step = (_sign(yellow[0] - green[0]), _sign(yellow[1] - green[1]))
        dest = (green[0] + step[0], green[1] + step[1])
        if int(out[green]) != Color.BLACK or int(out[dest]) != Color.GREEN:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    built = _vertical_layout()
    if built is None:
        return None
    grid, regions, trunk_col = built

    left_regions = [region for region in regions if region["side"] == "L"]
    right_regions = [region for region in regions if region["side"] == "R"]
    trap_side = random.choice(["L", "R"])
    trap_regions = left_regions if trap_side == "L" else right_regions
    opposite_regions = right_regions if trap_side == "L" else left_regions

    trap_options: list[tuple[dict, int, dict]] = []
    for region in trap_regions:
        tr0, _tc0, tr1, _tc1 = region["bbox"]
        for row in range(tr0, tr1 + 1):
            lure_region = _region_for_row(opposite_regions, row)
            if lure_region is not None:
                trap_options.append((region, row, lure_region))
    if not trap_options:
        return None
    trap_region, trap_row, lure_region = random.choice(trap_options)

    assignments: dict[int, str] = {}
    trap_idx = regions.index(trap_region)
    lure_idx = regions.index(lure_region)
    assignments[trap_idx] = "pair"
    assignments[lure_idx] = "yellow_only"

    if trap_side == "L":
        green = (trap_row, trunk_col - 1)
        yellow = (trap_row, random.randint(trap_region["bbox"][1], trunk_col - 4))
        lure = (trap_row, trunk_col + 1)
    else:
        green = (trap_row, trunk_col + 1)
        yellow = (trap_row, random.randint(trunk_col + 4, trap_region["bbox"][3]))
        lure = (trap_row, trunk_col - 1)
    grid[green] = Color.GREEN
    grid[yellow] = Color.YELLOW
    grid[lure] = Color.YELLOW
    trap_distance = _pair_distance(green, yellow)

    remaining = [idx for idx in range(len(regions)) if idx not in assignments]
    if len(remaining) < 3:
        return None

    preferred_regions = [
        idx
        for idx in remaining
        if any(cand["kind"] != "horizontal" for cand in _pair_candidates(regions[idx]["bbox"]))
    ]
    pair2_pool = preferred_regions or [idx for idx in remaining if _pair_candidates(regions[idx]["bbox"])]
    if not pair2_pool:
        return None
    pair2_idx = random.choice(pair2_pool)
    assignments[pair2_idx] = "pair"
    remaining.remove(pair2_idx)

    green_only_idx = random.choice(remaining)
    assignments[green_only_idx] = "green_only"
    remaining.remove(green_only_idx)

    empty_idx = random.choice(remaining)
    assignments[empty_idx] = "empty"
    remaining.remove(empty_idx)

    extra_roles = list(EXTRA_ROLE_WEIGHTS)
    extra_weights = [EXTRA_ROLE_WEIGHTS[role] for role in extra_roles]
    for idx in remaining:
        assignments[idx] = random.choices(extra_roles, weights=extra_weights, k=1)[0]

    used_distances = {trap_distance}
    for idx, role in assignments.items():
        box = regions[idx]["bbox"]
        if idx in (trap_idx, lure_idx):
            continue
        if role == "pair":
            preferred = {"vertical", "diagonal"} if idx == pair2_idx else None
            placed = _choose_pair(box, forbidden_dists=used_distances, preferred_kinds=preferred)
            if placed is None:
                return None
            grid[placed["green"]] = Color.GREEN
            grid[placed["yellow"]] = Color.YELLOW
            used_distances.add(int(placed["dist"]))
        elif role == "green_only":
            _place_single(grid, box, Color.GREEN)
        elif role == "yellow_only":
            _place_single(grid, box, Color.YELLOW)
        elif role != "empty":
            return None

    grid = _apply_random_symmetry(grid)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid region-pairing slide input")
