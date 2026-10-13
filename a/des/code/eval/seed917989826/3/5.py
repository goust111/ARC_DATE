import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

DIR_FROM_POS = {
    (0, 1): (-1, 0, "v"),
    (2, 1): (1, 0, "v"),
    (1, 0): (0, -1, "h"),
    (1, 2): (0, 1, "h"),
}
POS_FROM_DIR = {
    (-1, 0): (0, 1),
    (1, 0): (2, 1),
    (0, -1): (1, 0),
    (0, 1): (1, 2),
}


@dataclass(frozen=True)
class Seed:
    block_r: int
    block_c: int
    dr: int
    dc: int
    orientation: str
    color: int
    order: tuple[int, int, int, int]


def _mode(values) -> int:
    return Counter(int(v) for v in np.asarray(values).ravel()).most_common(1)[0][0]


def _line_color_candidates(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    if axis == 0:
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] == grid[r, 0]):
                groups.setdefault(int(grid[r, 0]), []).append(r)
    else:
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] == grid[0, c]):
                groups.setdefault(int(grid[0, c]), []).append(c)
    return groups


def _block_starts(separator_mask: np.ndarray) -> list[int]:
    starts: list[int] = []
    i = 0
    n = len(separator_mask)
    while i < n:
        if separator_mask[i]:
            if i + 1 < n:
                assert not separator_mask[i + 1], "separator lines must be 1 cell wide"
            i += 1
            continue
        start = i
        while i < n and not separator_mask[i]:
            i += 1
        assert i - start == 3, "each block span must be 3 cells"
        starts.append(start)
    assert starts, "at least one block row/col is required"
    return starts


def _detect_structure(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    row_groups = _line_color_candidates(grid, axis=0)
    col_groups = _line_color_candidates(grid, axis=1)
    common = set(row_groups) & set(col_groups)
    assert common, "failed to detect separator color"
    grey = max(common, key=lambda color: (len(row_groups[color]) + len(col_groups[color]), color))

    sep_rows = np.array([np.all(grid[r, :] == grey) for r in range(grid.shape[0])], dtype=bool)
    sep_cols = np.array([np.all(grid[:, c] == grey) for c in range(grid.shape[1])], dtype=bool)
    row_starts = _block_starts(sep_rows)
    col_starts = _block_starts(sep_cols)
    return grey, row_starts, col_starts


def _extract_seeds(grid: np.ndarray) -> tuple[int, list[int], list[int], int, list[Seed]]:
    grey, row_starts, col_starts = _detect_structure(grid)

    block_values = []
    for r0 in row_starts:
        for c0 in col_starts:
            block_values.extend(int(v) for v in grid[r0 : r0 + 3, c0 : c0 + 3].ravel())
    bg = _mode(block_values)
    assert bg != grey, "background and separator colors must differ"

    seeds: list[Seed] = []
    for br, r0 in enumerate(row_starts):
        for bc, c0 in enumerate(col_starts):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            nz = [(rr, cc, int(patch[rr, cc])) for rr in range(3) for cc in range(3) if int(patch[rr, cc]) != bg]
            if not nz:
                continue
            assert len(nz) == 1, "seed blocks must contain exactly one non-background pixel"
            rr, cc, color = nz[0]
            assert color != grey, "seed color cannot match separator color"
            assert (rr, cc) in DIR_FROM_POS, "seed must sit on a block-edge midpoint"
            dr, dc, orientation = DIR_FROM_POS[(rr, cc)]
            seeds.append(
                Seed(
                    block_r=br,
                    block_c=bc,
                    dr=dr,
                    dc=dc,
                    orientation=orientation,
                    color=color,
                    order=(br, bc, rr, cc),
                )
            )
    assert seeds, "at least one seed is required"
    return grey, row_starts, col_starts, bg, seeds


def _full_patch(color: int) -> np.ndarray:
    return np.full((3, 3), int(color), dtype=int)


def _split_patch(horizontal_color: int, vertical_color: int) -> np.ndarray:
    patch = np.full((3, 3), int(horizontal_color), dtype=int)
    patch[2, :] = int(vertical_color)
    return patch


def _simulate_growth(block_rows: int, block_cols: int, seeds: list[Seed]) -> tuple[dict[tuple[int, int], np.ndarray], dict]:
    occupied: dict[tuple[int, int], tuple[str, object]] = {}
    for idx, seed in enumerate(seeds):
        occupied[(seed.block_r, seed.block_c)] = ("source", idx)

    fills: dict[tuple[int, int], np.ndarray] = {}
    active = {idx: 1 for idx in range(len(seeds))}
    blocked_events: list[dict] = []
    split_blocks: list[tuple[int, int]] = []
    same_axis_conflicts: list[tuple[int, int]] = []
    complex_conflicts: list[tuple[int, int]] = []
    successful_steps = [0] * len(seeds)

    while active:
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx, distance in active.items():
            seed = seeds[idx]
            br = seed.block_r + seed.dr * distance
            bc = seed.block_c + seed.dc * distance
            if not (0 <= br < block_rows and 0 <= bc < block_cols):
                continue
            target = (br, bc)
            if target in occupied:
                kind, blocker = occupied[target]
                blocked_events.append(
                    {
                        "seed": idx,
                        "target": target,
                        "distance": distance,
                        "kind": kind,
                        "blocker": blocker,
                    }
                )
                continue
            proposals.setdefault(target, []).append(idx)

        if not proposals:
            break

        next_active: dict[int, int] = {}
        for target, ids in proposals.items():
            ids = sorted(ids, key=lambda idx: seeds[idx].order)
            if len(ids) == 1:
                idx = ids[0]
                fills[target] = _full_patch(seeds[idx].color)
                occupied[target] = ("fill", idx)
                successful_steps[idx] += 1
                next_active[idx] = active[idx] + 1
                continue

            h_ids = [idx for idx in ids if seeds[idx].orientation == "h"]
            v_ids = [idx for idx in ids if seeds[idx].orientation == "v"]
            if h_ids and v_ids:
                if len(h_ids) > 1 or len(v_ids) > 1:
                    complex_conflicts.append(target)
                h_idx = h_ids[0]
                v_idx = v_ids[0]
                fills[target] = _split_patch(seeds[h_idx].color, seeds[v_idx].color)
                occupied[target] = ("split", (h_idx, v_idx))
                split_blocks.append(target)
            else:
                same_axis_conflicts.append(target)
                winner = ids[0]
                fills[target] = _full_patch(seeds[winner].color)
                occupied[target] = ("tie", winner)

        active = next_active

    info = {
        "blocked_events": blocked_events,
        "split_blocks": split_blocks,
        "same_axis_conflicts": same_axis_conflicts,
        "complex_conflicts": complex_conflicts,
        "successful_steps": successful_steps,
        "occupied": occupied,
    }
    return fills, info


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    grey, row_starts, col_starts, _bg, seeds = _extract_seeds(g)
    assert grey is not None

    fills, _info = _simulate_growth(len(row_starts), len(col_starts), seeds)

    out = g.copy()
    for (br, bc), patch in fills.items():
        r0 = row_starts[br]
        c0 = col_starts[bc]
        out[r0 : r0 + 3, c0 : c0 + 3] = patch
    return out


def _empty_grid(block_rows: int, block_cols: int, bg: int = BLACK, grey: int = GREY) -> np.ndarray:
    h = block_rows * 3 + (block_rows - 1)
    w = block_cols * 3 + (block_cols - 1)
    grid = np.full((h, w), int(bg), dtype=int)
    for r in range(3, h, 4):
        grid[r, :] = int(grey)
    for c in range(3, w, 4):
        grid[:, c] = int(grey)
    return grid


def _place_seed(grid: np.ndarray, seed: Seed) -> None:
    rr, cc = POS_FROM_DIR[(seed.dr, seed.dc)]
    r0 = seed.block_r * 4
    c0 = seed.block_c * 4
    grid[r0 + rr, c0 + cc] = int(seed.color)


def _render_input(block_rows: int, block_cols: int, seeds: list[Seed]) -> np.ndarray:
    grid = _empty_grid(block_rows, block_cols)
    for seed in seeds:
        _place_seed(grid, seed)
    return grid


def _new_seed(src_r: int, src_c: int, dr: int, dc: int, color: int) -> Seed:
    rr, cc = POS_FROM_DIR[(dr, dc)]
    orientation = "h" if dc != 0 else "v"
    return Seed(src_r, src_c, dr, dc, orientation, int(color), (src_r, src_c, rr, cc))


def _scenario_tie(rng: random.Random, block_rows: int, block_cols: int, colors: list[int]) -> tuple[list[Seed], dict]:
    for _ in range(200):
        tr = rng.randrange(1, block_rows - 1)
        tc = rng.randrange(1, block_cols - 1)
        h_dir = rng.choice([(0, 1), (0, -1)])
        v_dir = rng.choice([(1, 0), (-1, 0)])
        max_h = tc if h_dir == (0, 1) else (block_cols - 1 - tc)
        max_v = tr if v_dir == (1, 0) else (block_rows - 1 - tr)
        max_d = min(max_h, max_v)
        if max_d < 1:
            continue
        d = rng.randint(1, max_d)
        h_src = (tr - h_dir[0] * d, tc - h_dir[1] * d)
        v_src = (tr - v_dir[0] * d, tc - v_dir[1] * d)
        if h_src == v_src:
            continue
        seeds = [
            _new_seed(h_src[0], h_src[1], h_dir[0], h_dir[1], colors[0]),
            _new_seed(v_src[0], v_src[1], v_dir[0], v_dir[1], colors[1]),
        ]
        return seeds, {"mode": "tie", "target": (tr, tc)}
    raise RuntimeError("failed to build tie scenario")


def _scenario_block(rng: random.Random, block_rows: int, block_cols: int, colors: list[int]) -> tuple[list[Seed], dict]:
    for _ in range(300):
        tr = rng.randrange(1, block_rows - 1)
        tc = rng.randrange(1, block_cols - 1)
        early_dir = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        late_dir = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        if (early_dir[0] == 0) == (late_dir[0] == 0):
            continue

        max_early = tc if early_dir == (0, 1) else (
            block_cols - 1 - tc if early_dir == (0, -1) else (
                tr if early_dir == (1, 0) else (block_rows - 1 - tr)
            )
        )
        max_late = tc if late_dir == (0, 1) else (
            block_cols - 1 - tc if late_dir == (0, -1) else (
                tr if late_dir == (1, 0) else (block_rows - 1 - tr)
            )
        )
        if max_early < 1 or max_late < 2:
            continue

        early_d = rng.randint(1, max_early)
        late_min = early_d + 1
        if late_min > max_late:
            continue
        late_d = rng.randint(late_min, max_late)

        early_src = (tr - early_dir[0] * early_d, tc - early_dir[1] * early_d)
        late_src = (tr - late_dir[0] * late_d, tc - late_dir[1] * late_d)
        if early_src == late_src:
            continue

        seeds = [
            _new_seed(early_src[0], early_src[1], early_dir[0], early_dir[1], colors[0]),
            _new_seed(late_src[0], late_src[1], late_dir[0], late_dir[1], colors[1]),
        ]
        return seeds, {"mode": "block", "target": (tr, tc), "late_seed": 1}
    raise RuntimeError("failed to build block scenario")


def _seed_has_room(block_rows: int, block_cols: int, block_r: int, block_c: int, dr: int, dc: int) -> bool:
    nr = block_r + dr
    nc = block_c + dc
    return 0 <= nr < block_rows and 0 <= nc < block_cols


def _scenario_open(rng: random.Random, block_rows: int, block_cols: int, colors: list[int]) -> tuple[list[Seed], dict]:
    desired = rng.randint(2, min(4, len(colors)))
    for _ in range(500):
        used_blocks: set[tuple[int, int]] = set()
        seeds: list[Seed] = []
        shuffled_colors = colors[:]
        rng.shuffle(shuffled_colors)
        for color in shuffled_colors:
            if len(seeds) >= desired:
                break
            for _ in range(40):
                block_r = rng.randrange(block_rows)
                block_c = rng.randrange(block_cols)
                if (block_r, block_c) in used_blocks:
                    continue
                dr, dc = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                if not _seed_has_room(block_rows, block_cols, block_r, block_c, dr, dc):
                    continue
                candidate = _new_seed(block_r, block_c, dr, dc, color)
                trial = seeds + [candidate]
                fills, info = _simulate_growth(block_rows, block_cols, trial)
                if info["split_blocks"] or info["same_axis_conflicts"] or info["complex_conflicts"] or info["blocked_events"]:
                    continue
                if not fills:
                    continue
                seeds = trial
                used_blocks.add((block_r, block_c))
                break
        if len(seeds) >= 2:
            return seeds, {"mode": "open"}
    raise RuntimeError("failed to build open scenario")


def _maybe_add_extra_seeds(
    rng: random.Random,
    block_rows: int,
    block_cols: int,
    seeds: list[Seed],
    scenario: dict,
    colors: list[int],
) -> list[Seed]:
    available_colors = [color for color in colors if color not in {seed.color for seed in seeds}]
    extras_goal = rng.randint(0, min(2, len(available_colors)))
    if extras_goal == 0:
        return seeds

    result = seeds[:]
    used_blocks = {(seed.block_r, seed.block_c) for seed in seeds}
    for color in available_colors:
        if len(result) >= len(seeds) + extras_goal:
            break
        accepted = False
        for _ in range(80):
            block_r = rng.randrange(block_rows)
            block_c = rng.randrange(block_cols)
            if (block_r, block_c) in used_blocks:
                continue
            dr, dc = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            if not _seed_has_room(block_rows, block_cols, block_r, block_c, dr, dc):
                continue
            candidate = _new_seed(block_r, block_c, dr, dc, color)
            trial = result + [candidate]
            if _valid_seed_set(block_rows, block_cols, trial, scenario):
                result = trial
                used_blocks.add((block_r, block_c))
                accepted = True
                break
        if not accepted and rng.random() < 0.5:
            break
    return result


def _valid_seed_set(block_rows: int, block_cols: int, seeds: list[Seed], scenario: dict) -> bool:
    if len({(seed.block_r, seed.block_c) for seed in seeds}) != len(seeds):
        return False
    if len({seed.color for seed in seeds}) != len(seeds):
        return False

    fills, info = _simulate_growth(block_rows, block_cols, seeds)
    if not fills:
        return False
    if info["same_axis_conflicts"] or info["complex_conflicts"]:
        return False

    mode = scenario["mode"]
    if mode == "tie":
        return scenario["target"] in info["split_blocks"]
    if mode == "block":
        return any(
            event["seed"] == scenario["late_seed"]
            and event["target"] == scenario["target"]
            and event["kind"] in {"fill", "split", "tie"}
            for event in info["blocked_events"]
        )
    if mode == "open":
        return not info["split_blocks"] and not info["blocked_events"]
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["tie", "block", "open"])
    min_size = 4 if mode in {"tie", "block"} else 3
    block_rows = rng.randint(min_size, 7)
    block_cols = rng.randint(min_size, 7)

    colors = SEED_COLORS[:]
    rng.shuffle(colors)

    try:
        if mode == "tie":
            seeds, scenario = _scenario_tie(rng, block_rows, block_cols, colors)
        elif mode == "block":
            seeds, scenario = _scenario_block(rng, block_rows, block_cols, colors)
        else:
            seeds, scenario = _scenario_open(rng, block_rows, block_cols, colors)
    except RuntimeError:
        return None

    seeds = _maybe_add_extra_seeds(rng, block_rows, block_cols, seeds, scenario, colors)
    if not _valid_seed_set(block_rows, block_cols, seeds, scenario):
        return None

    grid = _render_input(block_rows, block_cols, seeds)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if len(seeds) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
