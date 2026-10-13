from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

RED_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
BLUE_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_SOLUTIONS = 2
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _targets(cell: tuple[int, int], color: int) -> tuple[tuple[int, int], ...]:
    r, c = cell
    dirs = RED_DIRS if color == RED else BLUE_DIRS
    return tuple((r + dr, c + dc) for dr, dc in dirs)


def _base_domains(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    shape = g.shape
    domains: dict[tuple[int, int], tuple[int, ...]] = {}
    for r, c in np.argwhere(g == BLACK):
        cell = (int(r), int(c))
        options: list[int] = []
        if all(
            (not _in_bounds(shape, rr, cc)) or int(g[rr, cc]) != ORANGE
            for rr, cc in _targets(cell, RED)
        ):
            options.append(RED)
        if all(
            (not _in_bounds(shape, rr, cc)) or int(g[rr, cc]) != YELLOW
            for rr, cc in _targets(cell, BLUE)
        ):
            options.append(BLUE)
        domains[cell] = tuple(options)
    return domains


def _compute_hits(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    assign: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    yellow_hits = np.zeros(shape, dtype=np.uint8)
    orange_hits = np.zeros(shape, dtype=np.uint8)
    for cell, color in assign.items():
        hit_map = yellow_hits if color == RED else orange_hits
        for rr, cc in _targets(cell, color):
            if _in_bounds(shape, rr, cc) and not bool(grey_mask[rr, cc]):
                hit_map[rr, cc] += 1
    return yellow_hits, orange_hits


def _visible_from_centers(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    assign: dict[tuple[int, int], int],
) -> np.ndarray | None:
    out = np.full(shape, BLACK, dtype=int)
    out[grey_mask] = GREY
    center_cells = set(assign)
    for cell, color in assign.items():
        trace_color = YELLOW if color == RED else ORANGE
        for rr, cc in _targets(cell, color):
            if not _in_bounds(shape, rr, cc):
                continue
            if bool(grey_mask[rr, cc]) or (rr, cc) in center_cells:
                continue
            prev = int(out[rr, cc])
            if prev == BLACK:
                out[rr, cc] = trace_color
            elif prev != trace_color:
                return None
    return out


def _build_parent_maps(
    grid: np.ndarray,
    domains: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[dict[tuple[int, int], tuple[tuple[int, int], ...]], dict[tuple[int, int], tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    shape = g.shape
    yellow_parents: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    orange_parents: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for r, c in np.argwhere(g == YELLOW):
        cell = (int(r), int(c))
        parents = []
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            parent = (int(rr), int(cc))
            if _in_bounds(shape, rr, cc) and RED in domains.get(parent, ()):
                parents.append(parent)
        yellow_parents[cell] = tuple(parents)
    for r, c in np.argwhere(g == ORANGE):
        cell = (int(r), int(c))
        parents = []
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            parent = (int(rr), int(cc))
            if _in_bounds(shape, rr, cc) and BLUE in domains.get(parent, ()):
                parents.append(parent)
        orange_parents[cell] = tuple(parents)
    return yellow_parents, orange_parents


def _can_place(
    grid: np.ndarray,
    domains: dict[tuple[int, int], tuple[int, ...]],
    assign: dict[tuple[int, int], int],
    cell: tuple[int, int],
    color: int,
) -> bool:
    def support_count(target_cell: tuple[int, int], target_color: int) -> int:
        wanted = YELLOW if target_color == RED else ORANGE
        total = 0
        for rr, cc in _targets(target_cell, target_color):
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == wanted:
                total += 1
        return total

    if int(grid[cell]) != BLACK:
        return False
    current = assign.get(cell)
    if current is not None:
        return current == color
    if color not in domains.get(cell, ()):
        return False
    for rr, cc in _targets(cell, color):
        if not _in_bounds(grid.shape, rr, cc):
            continue
        observed = int(grid[rr, cc])
        if observed == GREY:
            continue
        if color == RED and observed == ORANGE:
            return False
        if color == BLUE and observed == YELLOW:
            return False
        if observed == BLACK and (rr, cc) not in assign:
            blocker_opts = domains.get((rr, cc), ())
            if not blocker_opts:
                return False
            if not any(support_count((rr, cc), blocker_color) > 0 for blocker_color in blocker_opts):
                return False
    return True


def _solution_signature(assign: dict[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((r, c, color) for (r, c), color in assign.items()))


def _solve_all(grid: np.ndarray, limit: int = MAX_SOLUTIONS) -> list[dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed = {BLACK, YELLOW, GREY, ORANGE}
    assert set(map(int, np.unique(g))) <= allowed, "input may only contain black, grey, yellow, and orange"

    shape = g.shape
    black_mask = g == BLACK
    yellow_mask = g == YELLOW
    orange_mask = g == ORANGE
    grey_mask = g == GREY
    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(yellow_mask)]
    orange_cells = [tuple(map(int, rc)) for rc in np.argwhere(orange_mask)]
    domains = _base_domains(g)
    yellow_parents, orange_parents = _build_parent_maps(g, domains)

    for cell in yellow_cells:
        if not yellow_parents[cell]:
            return []
    for cell in orange_cells:
        if not orange_parents[cell]:
            return []

    seen: set[tuple[tuple[int, int, int], ...]] = set()
    solutions: list[dict[tuple[int, int], int]] = []

    def normalize(assign: dict[tuple[int, int], int]):
        work = dict(assign)
        while True:
            yellow_hits, orange_hits = _compute_hits(shape, grey_mask, work)
            if np.any(orange_hits[yellow_mask]) or np.any(yellow_hits[orange_mask]):
                return None

            forced: dict[tuple[int, int], int] = {}

            for r, c in np.argwhere(black_mask & ((yellow_hits > 0) | (orange_hits > 0))):
                cell = (int(r), int(c))
                if cell in work:
                    continue
                opts = [
                    color
                    for color in domains.get(cell, ())
                    if _can_place(g, domains, work, cell, color)
                ]
                if not opts:
                    return None
                if len(opts) == 1:
                    color = opts[0]
                    prev = forced.get(cell)
                    if prev is not None and prev != color:
                        return None
                    forced[cell] = color

            for cell in yellow_cells:
                if yellow_hits[cell] > 0:
                    continue
                opts = [
                    parent
                    for parent in yellow_parents[cell]
                    if _can_place(g, domains, work, parent, RED)
                ]
                if not opts:
                    return None
                if len(opts) == 1:
                    parent = opts[0]
                    prev = forced.get(parent)
                    if prev is not None and prev != RED:
                        return None
                    forced[parent] = RED

            for cell in orange_cells:
                if orange_hits[cell] > 0:
                    continue
                opts = [
                    parent
                    for parent in orange_parents[cell]
                    if _can_place(g, domains, work, parent, BLUE)
                ]
                if not opts:
                    return None
                if len(opts) == 1:
                    parent = opts[0]
                    prev = forced.get(parent)
                    if prev is not None and prev != BLUE:
                        return None
                    forced[parent] = BLUE

            if not forced:
                must_centers = [
                    (int(r), int(c))
                    for r, c in np.argwhere(black_mask & ((yellow_hits > 0) | (orange_hits > 0)))
                    if (int(r), int(c)) not in work
                ]
                return work, yellow_hits, orange_hits, must_centers

            for cell, color in forced.items():
                prev = work.get(cell)
                if prev is not None and prev != color:
                    return None
                work[cell] = color

    def branch_score_for_cell(
        assign: dict[tuple[int, int], int],
        cell: tuple[int, int],
        color: int,
    ) -> tuple[int, int]:
        trace_gain = 0
        new_black_hits = 0
        for rr, cc in _targets(cell, color):
            if not _in_bounds(shape, rr, cc):
                continue
            observed = int(g[rr, cc])
            if observed == GREY:
                continue
            if color == RED and observed == YELLOW:
                trace_gain += 1
            if color == BLUE and observed == ORANGE:
                trace_gain += 1
            if observed == BLACK and (rr, cc) not in assign:
                new_black_hits += 1
        return (-trace_gain, new_black_hits)

    def dfs(assign: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= limit:
            return

        normalized = normalize(assign)
        if normalized is None:
            return
        work, yellow_hits, orange_hits, must_centers = normalized
        sig = _solution_signature(work)
        if sig in seen:
            return
        seen.add(sig)

        tasks: list[tuple[str, tuple[int, int], tuple]] = []
        for cell in yellow_cells:
            if yellow_hits[cell] > 0:
                continue
            opts = tuple(
                parent for parent in yellow_parents[cell] if _can_place(g, domains, work, parent, RED)
            )
            if not opts:
                return
            tasks.append(("yellow", cell, opts))
        for cell in orange_cells:
            if orange_hits[cell] > 0:
                continue
            opts = tuple(
                parent for parent in orange_parents[cell] if _can_place(g, domains, work, parent, BLUE)
            )
            if not opts:
                return
            tasks.append(("orange", cell, opts))
        for cell in must_centers:
            opts = tuple(color for color in domains.get(cell, ()) if _can_place(g, domains, work, cell, color))
            if not opts:
                return
            tasks.append(("must", cell, opts))

        if not tasks:
            visible = _visible_from_centers(shape, grey_mask, work)
            if visible is not None and np.array_equal(visible, g):
                solutions.append(work)
            return

        task = min(tasks, key=lambda item: (len(item[2]), 0 if item[0] != "must" else 1))
        kind, cell, options = task

        if kind == "yellow":
            ordered = sorted(options, key=lambda p: branch_score_for_cell(work, p, RED))
            for parent in ordered:
                nxt = dict(work)
                nxt[parent] = RED
                dfs(nxt)
                if len(solutions) >= limit:
                    return
            return

        if kind == "orange":
            ordered = sorted(options, key=lambda p: branch_score_for_cell(work, p, BLUE))
            for parent in ordered:
                nxt = dict(work)
                nxt[parent] = BLUE
                dfs(nxt)
                if len(solutions) >= limit:
                    return
            return

        ordered_colors = sorted(
            options,
            key=lambda color: branch_score_for_cell(work, cell, color),
        )
        for color in ordered_colors:
            nxt = dict(work)
            nxt[cell] = color
            dfs(nxt)
            if len(solutions) >= limit:
                return

    dfs({})
    return solutions


def _solve_unique(grid: np.ndarray) -> dict[tuple[int, int], int]:
    solutions = _solve_all(grid, limit=MAX_SOLUTIONS)
    assert len(solutions) == 1, "expected exactly one center configuration"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the erased red and blue stamp centers from the remaining yellow and orange traces.

    Input structure:
    - A black background contains visible grey occluders plus yellow and orange one-pixel traces.
    - Hidden red centers emitted to the four diagonal neighbors, and hidden blue centers emitted to the four orthogonal neighbors.
    - Emissions only survived on cells that were black background at emission time; the original centers were then erased back to black.

    Rule:
    - Infer the unique set of red and blue center cells whose replay reproduces every yellow and orange trace exactly.
    - A missing trace is valid only when that direction is blocked by the grid edge, a grey occluder, or another center.
    - Return the input grid with the erased red and blue centers restored in place; keep all traces and grey cells unchanged.

    Key insight:
    - The task is a local causal inversion: every visible trace demands at least one matching hidden emitter, and every absent trace around a chosen center must itself be explained by a blocker.
    """
    g = np.asarray(grid, dtype=int)
    assign = _solve_unique(g)
    out = g.copy()
    for (r, c), color in assign.items():
        out[r, c] = color
    return out


def _profile_center(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    centers: dict[tuple[int, int], int],
    cell: tuple[int, int],
    color: int,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    visible: list[tuple[int, int]] = []
    blocked = {"edge": 0, "grey": 0, "center": 0}
    center_cells = set(centers)
    for rr, cc in _targets(cell, color):
        if not _in_bounds(shape, rr, cc):
            blocked["edge"] += 1
        elif bool(grey_mask[rr, cc]):
            blocked["grey"] += 1
        elif (rr, cc) in center_cells:
            blocked["center"] += 1
        else:
            visible.append((rr, cc))
    return visible, blocked


def _trace_multiplicity(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    centers: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    yellow_sources: dict[tuple[int, int], int] = defaultdict(int)
    orange_sources: dict[tuple[int, int], int] = defaultdict(int)
    center_cells = set(centers)
    for cell, color in centers.items():
        target_map = yellow_sources if color == RED else orange_sources
        for rr, cc in _targets(cell, color):
            if not _in_bounds(shape, rr, cc):
                continue
            if bool(grey_mask[rr, cc]) or (rr, cc) in center_cells:
                continue
            target_map[(rr, cc)] += 1
    return yellow_sources, orange_sources


def _cell_is_free(
    cell: tuple[int, int],
    grey_mask: np.ndarray,
    centers: dict[tuple[int, int], int],
) -> bool:
    return not bool(grey_mask[cell]) and cell not in centers


def _simulate_output(centers: dict[tuple[int, int], int], shape: tuple[int, int], grey_mask: np.ndarray) -> np.ndarray | None:
    visible = _visible_from_centers(shape, grey_mask, centers)
    if visible is None:
        return None
    out = visible.copy()
    for (r, c), color in centers.items():
        out[r, c] = color
    return out


def _single_or_pair_extra_solution_exists(
    grid: np.ndarray,
    solved_assign: dict[tuple[int, int], int],
) -> bool:
    g = np.asarray(grid, dtype=int)
    shape = g.shape
    grey_mask = g == GREY
    domains = _base_domains(g)
    base_visible = _visible_from_centers(shape, grey_mask, solved_assign)
    if base_visible is None or not np.array_equal(base_visible, g):
        return True

    black_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == BLACK)]
    candidate_cells = []
    for cell in black_cells:
        if cell in solved_assign:
            continue
        if cell[0] in (0, shape[0] - 1) or cell[1] in (0, shape[1] - 1):
            candidate_cells.append(cell)
            continue
        for rr, cc in _targets(cell, RED) + _targets(cell, BLUE):
            if _in_bounds(shape, rr, cc) and int(g[rr, cc]) != BLACK:
                candidate_cells.append(cell)
                break
    candidate_cells = list(dict.fromkeys(candidate_cells))

    for cell in candidate_cells:
        for color in (RED, BLUE):
            if color not in domains.get(cell, ()):
                continue
            extra = dict(solved_assign)
            extra[cell] = color
            visible = _visible_from_centers(shape, grey_mask, extra)
            if visible is not None and np.array_equal(visible, g):
                return True

    pair_budget = min(12, len(candidate_cells))
    probe_cells = candidate_cells[:pair_budget]
    for i, a in enumerate(probe_cells):
        for b in probe_cells[i + 1 :]:
            for color_a in (RED, BLUE):
                for color_b in (RED, BLUE):
                    if color_a not in domains.get(a, ()) or color_b not in domains.get(b, ()):
                        continue
                    extra = dict(solved_assign)
                    extra[a] = color_a
                    extra[b] = color_b
                    visible = _visible_from_centers(shape, grey_mask, extra)
                    if visible is not None and np.array_equal(visible, g):
                        return True
    return False


def _add_center_if_valid(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    centers: dict[tuple[int, int], int],
    cell: tuple[int, int],
    color: int,
) -> bool:
    if not _cell_is_free(cell, grey_mask, centers):
        return False

    trial = dict(centers)
    trial[cell] = color
    visible = _visible_from_centers(shape, grey_mask, trial)
    if visible is None:
        return False

    yellow_mult, orange_mult = _trace_multiplicity(shape, grey_mask, trial)
    if any(count > 1 for count in yellow_mult.values()):
        return False
    if any(count > 1 for count in orange_mult.values()):
        return False

    for other_cell, other_color in trial.items():
        visible_targets, _blocked = _profile_center(shape, grey_mask, trial, other_cell, other_color)
        if not visible_targets:
            return False

    centers.clear()
    centers.update(trial)
    return True


def _claim_reserved(reserved: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> None:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = max(0, min(rows) - pad)
    c0 = max(0, min(cols) - pad)
    r1 = min(reserved.shape[0] - 1, max(rows) + pad)
    c1 = min(reserved.shape[1] - 1, max(cols) + pad)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _motif_footprint(
    shape: tuple[int, int],
    motif_centers: dict[tuple[int, int], int],
    motif_greys: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells = set(motif_centers) | set(motif_greys)
    for cell, color in motif_centers.items():
        for rr, cc in _targets(cell, color):
            if _in_bounds(shape, rr, cc):
                cells.add((rr, cc))
    return cells


def _try_place_motif(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    reserved: np.ndarray,
    centers: dict[tuple[int, int], int],
    motif_centers: dict[tuple[int, int], int],
    motif_greys: set[tuple[int, int]],
) -> bool:
    footprint = _motif_footprint(shape, motif_centers, motif_greys)
    if not footprint:
        return False
    if any(bool(reserved[cell]) for cell in footprint):
        return False
    if any(cell in centers or bool(grey_mask[cell]) for cell in motif_centers):
        return False
    if any(cell in centers or bool(grey_mask[cell]) for cell in motif_greys):
        return False

    trial_grey = grey_mask.copy()
    for cell in motif_greys:
        trial_grey[cell] = True
    trial_centers = dict(centers)
    for cell, color in motif_centers.items():
        if not _add_center_if_valid(shape, trial_grey, trial_centers, cell, color):
            return False

    grey_mask[:, :] = trial_grey
    centers.clear()
    centers.update(trial_centers)
    _claim_reserved(reserved, footprint, pad=1)
    return True


def _sample_puzzle() -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    h = random.randint(10, 16)
    w = random.randint(10, 16)
    shape = (h, w)
    grey_mask = np.zeros(shape, dtype=bool)
    reserved = np.zeros(shape, dtype=bool)
    centers: dict[tuple[int, int], int] = {}

    def place_edge_single(color: int) -> bool:
        choices: list[tuple[int, int]] = []
        for c in range(1, w - 1):
            choices.extend([(0, c), (h - 1, c)])
        for r in range(1, h - 1):
            choices.extend([(r, 0), (r, w - 1)])
        random.shuffle(choices)
        for cell in choices:
            if _try_place_motif(shape, grey_mask, reserved, centers, {cell: color}, set()):
                return True
        return False

    def place_grey_single(color: int) -> bool:
        dirs = RED_DIRS if color == RED else BLUE_DIRS
        positions = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
        random.shuffle(positions)
        dir_choices = list(dirs)
        for cell in positions:
            random.shuffle(dir_choices)
            for dr, dc in dir_choices:
                blocker = (cell[0] + dr, cell[1] + dc)
                if not _in_bounds(shape, blocker[0], blocker[1]):
                    continue
                if _try_place_motif(shape, grey_mask, reserved, centers, {cell: color}, {blocker}):
                    return True
        return False

    def place_center_pair() -> bool:
        pair_modes = ["blue_pair", "red_pair"]
        random.shuffle(pair_modes)
        for mode in pair_modes:
            if mode == "blue_pair":
                positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
                random.shuffle(positions)
                dir_choices = list(BLUE_DIRS)
                for a in positions:
                    random.shuffle(dir_choices)
                    for dr, dc in dir_choices:
                        b = (a[0] + dr, a[1] + dc)
                        if not _in_bounds(shape, b[0], b[1]):
                            continue
                        motif = {a: BLUE, b: BLUE}
                        if _try_place_motif(shape, grey_mask, reserved, centers, motif, set()):
                            return True
            else:
                positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
                random.shuffle(positions)
                dir_choices = list(RED_DIRS)
                for a in positions:
                    random.shuffle(dir_choices)
                    for dr, dc in dir_choices:
                        b = (a[0] + dr, a[1] + dc)
                        if not _in_bounds(shape, b[0], b[1]):
                            continue
                        second_color = random.choice((RED, BLUE))
                        motif = {a: RED, b: second_color}
                        if _try_place_motif(shape, grey_mask, reserved, centers, motif, set()):
                            return True
        return False

    edge_color = random.choice((RED, BLUE))
    grey_color = BLUE if edge_color == RED else RED
    if random.random() < 0.35:
        grey_color = edge_color

    if not place_edge_single(edge_color):
        return None
    if not place_grey_single(grey_color):
        return None
    if not place_center_pair():
        return None

    if random.random() < 0.55:
        extra_color = random.choice((RED, BLUE))
        if random.random() < 0.5:
            place_edge_single(extra_color)
        else:
            place_grey_single(extra_color)

    placed_red = sum(color == RED for color in centers.values())
    placed_blue = sum(color == BLUE for color in centers.values())
    if len(centers) < 4 or placed_red == 0 or placed_blue == 0:
        return None

    visible = _visible_from_centers(shape, grey_mask, centers)
    if visible is None:
        return None

    yellow_mult, orange_mult = _trace_multiplicity(shape, grey_mask, centers)
    if any(count > 1 for count in yellow_mult.values()) or any(count > 1 for count in orange_mult.values()):
        return None

    profiles = [_profile_center(shape, grey_mask, centers, cell, color) for cell, color in centers.items()]
    if any(len(visible_targets) == 0 for visible_targets, _blocked in profiles):
        return None

    block_reasons = {"edge": 0, "grey": 0, "center": 0}
    for _visible_targets, blocked in profiles:
        for key, value in blocked.items():
            block_reasons[key] += int(value > 0)

    if block_reasons["edge"] == 0:
        return None
    if block_reasons["grey"] == 0:
        return None
    if block_reasons["center"] == 0:
        return None

    if len(yellow_mult) < 2 or len(orange_mult) < 2:
        return None

    solved = _solve_all(visible, limit=MAX_SOLUTIONS)
    if len(solved) != 1:
        return None
    if solved[0] != centers:
        return None
    if _single_or_pair_extra_solution_exists(visible, solved[0]):
        return None

    return visible, centers


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _sample_puzzle()
        if sample is None:
            continue
        visible, _centers = sample
        return visible
    raise RuntimeError("failed to generate a unique stamp-inversion input")
