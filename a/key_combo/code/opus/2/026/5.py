import random
from collections import deque
from functools import lru_cache

import numpy as np


GRID_SIZE = 9
CENTER = (4, 4)
CENTER_COLOR = 5
CLUE_COLORS = (1, 2)
ALL_PERIODS = ((2, 2), (2, 3), (3, 2), (3, 3))
GEN_PERIODS = ((2, 3), (3, 2), (3, 3))
MIN_CLUES = 8
MAX_CLUES = 12
MAX_GENERATION_ATTEMPTS = 400
MAX_PATTERN_ATTEMPTS = 200
MAX_CLUE_ATTEMPTS = 300

ALL_CELLS = [
    (r, c)
    for r in range(GRID_SIZE)
    for c in range(GRID_SIZE)
    if (r, c) != CENTER
]
CELL_TO_INDEX = {cell: idx for idx, cell in enumerate(ALL_CELLS)}


def _symmetry_cell(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return GRID_SIZE - 1 - r, GRID_SIZE - 1 - c


def _build_adjacency(pr: int, pc: int, *, use_periodic: bool, use_symmetry: bool) -> list[list[int]]:
    adj = [[] for _ in ALL_CELLS]
    for idx, (r, c) in enumerate(ALL_CELLS):
        neighbors: set[int] = set()
        if use_periodic:
            for dr in (-pr, pr):
                rr = r + dr
                if 0 <= rr < GRID_SIZE and (rr, c) != CENTER:
                    neighbors.add(CELL_TO_INDEX[(rr, c)])
            for dc in (-pc, pc):
                cc = c + dc
                if 0 <= cc < GRID_SIZE and (r, cc) != CENTER:
                    neighbors.add(CELL_TO_INDEX[(r, cc)])
        if use_symmetry:
            sr, sc = _symmetry_cell((r, c))
            if (sr, sc) != CENTER:
                neighbors.add(CELL_TO_INDEX[(sr, sc)])
        adj[idx] = sorted(neighbors)
    return adj


def _connected_components(adj: list[list[int]]) -> tuple[list[tuple[int, ...]], list[int]]:
    comp_ids = [-1] * len(adj)
    comps: list[tuple[int, ...]] = []
    for start in range(len(adj)):
        if comp_ids[start] != -1:
            continue
        queue = deque([start])
        comp_ids[start] = len(comps)
        comp: list[int] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if comp_ids[nxt] != -1:
                    continue
                comp_ids[nxt] = comp_ids[start]
                queue.append(nxt)
        comps.append(tuple(sorted(comp)))
    return comps, comp_ids


@lru_cache(maxsize=None)
def _candidate_data(pr: int, pc: int) -> dict:
    periodic_adj = _build_adjacency(pr, pc, use_periodic=True, use_symmetry=False)
    symmetry_adj = _build_adjacency(pr, pc, use_periodic=False, use_symmetry=True)
    joint_adj = _build_adjacency(pr, pc, use_periodic=True, use_symmetry=True)

    periodic_components, periodic_ids = _connected_components(periodic_adj)
    symmetry_components, symmetry_ids = _connected_components(symmetry_adj)
    joint_components, joint_ids = _connected_components(joint_adj)

    joint_periodic_groups: list[tuple[tuple[int, ...], ...]] = []
    paired_joint_ids: list[int] = []
    for comp in joint_components:
        groups: dict[int, list[int]] = {}
        for idx in comp:
            groups.setdefault(periodic_ids[idx], []).append(idx)
        ordered = tuple(
            tuple(sorted(group))
            for _cid, group in sorted(groups.items(), key=lambda item: min(item[1]))
        )
        joint_periodic_groups.append(ordered)
        if len(ordered) == 2:
            paired_joint_ids.append(len(joint_periodic_groups) - 1)

    return {
        "periods": (pr, pc),
        "periodic_components": periodic_components,
        "periodic_ids": periodic_ids,
        "symmetry_components": symmetry_components,
        "symmetry_ids": symmetry_ids,
        "joint_components": joint_components,
        "joint_ids": joint_ids,
        "joint_periodic_groups": tuple(joint_periodic_groups),
        "paired_joint_ids": tuple(paired_joint_ids),
    }


CANDIDATE_DATA = {periods: _candidate_data(*periods) for periods in ALL_PERIODS}


def _extract_clues(grid: np.ndarray) -> dict[int, int]:
    clues: dict[int, int] = {}
    for idx, (r, c) in enumerate(ALL_CELLS):
        color = int(grid[r, c])
        if color in CLUE_COLORS:
            clues[idx] = color
        else:
            assert color == 0, "only black blanks are allowed away from the center"
    return clues


def _render_completion(component_colors: list[int], data: dict) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    out[CENTER] = CENTER_COLOR
    for color, comp in zip(component_colors, data["joint_components"]):
        for idx in comp:
            r, c = ALL_CELLS[idx]
            out[r, c] = int(color)
    return out


def _solve_candidate(grid: np.ndarray, periods: tuple[int, int]) -> dict:
    data = CANDIDATE_DATA[periods]
    clues = _extract_clues(grid)
    component_colors: list[int] = []
    for comp in data["joint_components"]:
        colors = {clues[idx] for idx in comp if idx in clues}
        if len(colors) > 1:
            return {"periods": periods, "full": False, "conflict": True, "output": None}
        if not colors:
            return {"periods": periods, "full": False, "conflict": False, "output": None}
        component_colors.append(colors.pop())
    return {
        "periods": periods,
        "full": True,
        "conflict": False,
        "output": _render_completion(component_colors, data),
    }


def _closure_from_seeds(components: tuple[tuple[int, ...], ...] | list[tuple[int, ...]], comp_ids: list[int], seeds: set[int]) -> set[int]:
    covered: set[int] = set()
    active = {comp_ids[idx] for idx in seeds}
    for comp_id in active:
        covered.update(components[comp_id])
    return covered


def _completion_periods(grid: np.ndarray) -> list[tuple[int, int]]:
    return [
        periods
        for periods in ALL_PERIODS
        if _solve_candidate(grid, periods)["full"]
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sparse 9x9 red/blue pattern so that periodicity and central point symmetry hold at once.

    Input structure:
    - The grid is exactly 9x9 with black blanks, one grey center pivot at (4,4), and sparse blue/red clue pixels.
    - Every non-grey colored cell belongs to a hidden full pattern that uses only blue and red.
    - Away from the grey pivot, the hidden pattern obeys one row period and one column period from {2, 3}, plus point symmetry around the center.

    Rule:
    - Preserve the grey center and ignore it when propagating the red/blue equality constraints.
    - Test each candidate period pair in {2,3} x {2,3}; for that pair, connect cells that match by row-period shifts, column-period shifts, or central reflection.
    - Keep the unique pair whose joint connected components are all seeded without color conflicts, then fill every cell in each component with its forced clue color.

    Key insight:
    - Periodic residue classes and symmetric pairs form one joint equality graph, and only one graph is fully colorable from the sparse clues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 grid"
    assert int(g[CENTER]) == CENTER_COLOR, "expected a grey pivot at the center"
    assert int(np.count_nonzero(g == CENTER_COLOR)) == 1, "expected exactly one grey pivot"

    candidates = []
    for periods in ALL_PERIODS:
        result = _solve_candidate(g, periods)
        if result["full"]:
            candidates.append(result)

    assert len(candidates) == 1, "expected a unique jointly consistent period pair"
    return candidates[0]["output"]


def _sample_full_grid(true_periods: tuple[int, int]) -> np.ndarray | None:
    data = CANDIDATE_DATA[true_periods]
    comp_count = len(data["joint_components"])
    for _ in range(MAX_PATTERN_ATTEMPTS):
        component_colors = [random.choice(CLUE_COLORS) for _ in range(comp_count)]
        if len(set(component_colors)) != 2:
            continue
        full = _render_completion(component_colors, data)
        if _completion_periods(full) == [true_periods]:
            return full
    return None


def _make_input(full_grid: np.ndarray, clue_indices: set[int]) -> np.ndarray:
    inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    inp[CENTER] = CENTER_COLOR
    for idx in clue_indices:
        r, c = ALL_CELLS[idx]
        inp[r, c] = int(full_grid[r, c])
    return inp


def _semantic_checks(inp: np.ndarray, full_grid: np.ndarray, true_periods: tuple[int, int], clue_indices: set[int]) -> bool:
    if int(np.count_nonzero(inp == CENTER_COLOR)) != 1 or int(inp[CENTER]) != CENTER_COLOR:
        return False

    clue_values = [int(inp[r, c]) for r, c in ALL_CELLS if int(inp[r, c]) in CLUE_COLORS]
    if not (MIN_CLUES <= len(clue_values) <= MAX_CLUES):
        return False
    if set(clue_values) != set(CLUE_COLORS):
        return False

    results = [_solve_candidate(inp, periods) for periods in ALL_PERIODS]
    full_results = [result for result in results if result["full"]]
    if len(full_results) != 1 or full_results[0]["periods"] != true_periods:
        return False
    if not np.array_equal(full_results[0]["output"], full_grid):
        return False

    data = CANDIDATE_DATA[true_periods]
    seeds = set(clue_indices)
    period_closure = _closure_from_seeds(data["periodic_components"], data["periodic_ids"], seeds)
    symmetry_closure = _closure_from_seeds(data["symmetry_components"], data["symmetry_ids"], seeds)
    joint_closure = _closure_from_seeds(data["joint_components"], data["joint_ids"], seeds)
    universe = set(range(len(ALL_CELLS)))

    if joint_closure != universe:
        return False
    if period_closure == universe or symmetry_closure == universe:
        return False
    if not (period_closure - (symmetry_closure | seeds)):
        return False
    if not (symmetry_closure - (period_closure | seeds)):
        return False
    if not (joint_closure - (period_closure | symmetry_closure | seeds)):
        return False
    return True


def _sample_clues(full_grid: np.ndarray, true_periods: tuple[int, int]) -> np.ndarray | None:
    data = CANDIDATE_DATA[true_periods]
    full_colors = np.array([int(full_grid[r, c]) for r, c in ALL_CELLS], dtype=int)

    for _ in range(MAX_CLUE_ATTEMPTS):
        target = random.randint(MIN_CLUES, MAX_CLUES)
        bridge_comp = random.choice(data["paired_joint_ids"])
        bridge_groups = data["joint_periodic_groups"][bridge_comp]
        open_group = random.randrange(len(bridge_groups))
        blocked = set(bridge_groups[1 - open_group])

        chosen: set[int] = set()
        for comp_id, groups in enumerate(data["joint_periodic_groups"]):
            group = bridge_groups[open_group] if comp_id == bridge_comp else random.choice(groups)
            chosen.add(random.choice(group))

        next_color = random.choice(CLUE_COLORS)
        while len(chosen) < target:
            pool = [
                idx
                for idx in range(len(ALL_CELLS))
                if idx not in chosen and idx not in blocked and int(full_colors[idx]) == next_color
            ]
            if not pool:
                pool = [idx for idx in range(len(ALL_CELLS)) if idx not in chosen and idx not in blocked]
            if not pool:
                break
            chosen.add(random.choice(pool))
            next_color = CLUE_COLORS[0] if next_color == CLUE_COLORS[1] else CLUE_COLORS[1]

        inp = _make_input(full_grid, chosen)
        if _semantic_checks(inp, full_grid, true_periods, chosen):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        true_periods = random.choice(GEN_PERIODS)
        full_grid = _sample_full_grid(true_periods)
        if full_grid is None:
            continue
        inp = _sample_clues(full_grid, true_periods)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid periodic-plus-symmetric ARC input")
