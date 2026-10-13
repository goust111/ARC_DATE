import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS: Sequence[Tuple[str, int, int]] = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)
DIR_INDEX = {name: (dr, dc) for name, dr, dc in DIRS}
AXIS_HIDDEN_PATTERNS = [frozenset({"U", "D"}), frozenset({"L", "R"})]
MIXED_PATTERNS = [
    frozenset(),
    frozenset({"U"}),
    frozenset({"D"}),
    frozenset({"L"}),
    frozenset({"R"}),
    frozenset({"U", "L"}),
    frozenset({"U", "R"}),
    frozenset({"D", "L"}),
    frozenset({"D", "R"}),
]
EXTRA_PATTERNS = MIXED_PATTERNS
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 220


def _is_border(h: int, w: int, r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _distance_to_border(h: int, w: int, r: int, c: int, direction: str) -> int:
    if direction == "U":
        return r
    if direction == "D":
        return h - 1 - r
    if direction == "L":
        return c
    if direction == "R":
        return w - 1 - c
    raise ValueError(direction)


def _border_residues(grid: np.ndarray) -> Dict[int, Set[Tuple[int, int]]]:
    h, w = grid.shape
    residues: Dict[int, Set[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if not _is_border(h, w, r, c):
                continue
            color = int(grid[r, c])
            if color in (BG, WALL):
                continue
            residues.setdefault(color, set()).add((r, c))
    return residues


def _candidate_matches(grid: np.ndarray, r: int, c: int, residues: Set[Tuple[int, int]]) -> bool:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != BG:
        return False

    expected: Set[Tuple[int, int]] = set()
    for _name, dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        blocked = False
        while 0 <= rr < h and 0 <= cc < w:
            if int(grid[rr, cc]) == WALL:
                blocked = True
                break
            rr += dr
            cc += dc
        if blocked:
            continue
        endpoint = (rr - dr, cc - dc)
        expected.add(endpoint)
    return expected == residues


def _find_candidates_for_color(grid: np.ndarray, color: int) -> List[Tuple[int, int]]:
    residues_by_color = _border_residues(grid)
    residues = residues_by_color.get(int(color), set())
    if not residues:
        return []
    h, w = grid.shape
    candidates: List[Tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if _candidate_matches(grid, r, c, residues):
                candidates.append((r, c))
    return candidates


def _reconstruct_emitters(grid: np.ndarray) -> Dict[int, Tuple[int, int]]:
    emitters: Dict[int, Tuple[int, int]] = {}
    for color in sorted(_border_residues(grid)):
        candidates = _find_candidates_for_color(grid, color)
        assert len(candidates) == 1, f"color {color} has {len(candidates)} candidates"
        emitters[int(color)] = candidates[0]
    return emitters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each hidden emitter from colored border residues and redraw its cross rays.

    Input structure:
    - The background is black (0), with some internal gray (5) wall cells.
    - The only colored pixels are isolated residues on the outer border; each residue color belongs to one hidden emitter.
    - Nothing else is drawn: emitters and their rays have been erased from the interior.

    Rule:
    - For each residue color, find the unique interior cell whose four cardinal directions are all explained as follows:
      a direction either reaches the border without hitting a wall and ends at exactly one residue of that color, or is blocked by a wall before the border.
    - Every residue of that color must be explained by those unobstructed directions, and missing residues must therefore be justified by blocking walls.
    - Paint the emitter at that cell in its color, then extend its rays up/down/left/right until the border or until the cell just before a wall; keep walls and original residues unchanged.

    Key insight:
    - The border shows only the surviving projections of a deleted cross, so absent residues are not noise: together with wall positions they uniquely determine blocked directions and can even recover an entire hidden coordinate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    emitters = _reconstruct_emitters(g)

    out = g.copy()
    for color, (r, c) in emitters.items():
        out[r, c] = int(color)
        for _name, dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != WALL:
                out[rr, cc] = int(color)
                rr += dr
                cc += dc
    return out


def _make_blocker_segment(h: int, w: int, wr: int, wc: int, dr: int, dc: int) -> Set[Tuple[int, int]]:
    _ = (h, w, dr, dc)
    return {(wr, wc)}


def _propose_emitter(
    h: int,
    w: int,
    color: int,
    blocked_dirs: Set[str],
) -> Optional[dict]:
    r = random.randint(2, h - 3)
    c = random.randint(2, w - 3)

    for direction in blocked_dirs:
        if _distance_to_border(h, w, r, c, direction) < 2:
            return None

    color_cells: Set[Tuple[int, int]] = {(r, c)}
    wall_cells: Set[Tuple[int, int]] = set()
    residues: Set[Tuple[int, int]] = set()
    open_dirs: Set[str] = set()

    for direction, dr, dc in DIRS:
        dist = _distance_to_border(h, w, r, c, direction)
        if direction in blocked_dirs:
            min_step = 1
            max_step = dist - 1
            if max_step < min_step:
                return None
            if max_step >= 2 and random.random() < 0.7:
                wall_step = random.randint(2, max_step)
            else:
                wall_step = random.randint(min_step, max_step)
            for step in range(1, wall_step):
                color_cells.add((r + dr * step, c + dc * step))
            wr = r + dr * wall_step
            wc = c + dc * wall_step
            wall_cells |= _make_blocker_segment(h, w, wr, wc, dr, dc)
        else:
            open_dirs.add(direction)
            for step in range(1, dist + 1):
                color_cells.add((r + dr * step, c + dc * step))
            residues.add((r + dr * dist, c + dc * dist))

    if not open_dirs:
        return None

    return {
        "color": int(color),
        "pos": (r, c),
        "blocked": set(blocked_dirs),
        "open": open_dirs,
        "color_cells": color_cells,
        "wall_cells": wall_cells,
        "residues": residues,
    }


def _residue_isolated(residues: Iterable[Tuple[int, int]], h: int, w: int) -> bool:
    points = list(residues)
    for i, (r0, c0) in enumerate(points):
        for j in range(i + 1, len(points)):
            r1, c1 = points[j]
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
            if r0 == r1 and abs(c0 - c1) == 1 and (r0 in (0, h - 1)):
                return False
            if c0 == c1 and abs(r0 - r1) == 1 and (c0 in (0, w - 1)):
                return False
    return True


def _fits_scene(
    spec: dict,
    occupied_color: Set[Tuple[int, int]],
    occupied_wall: Set[Tuple[int, int]],
    border_points: Set[Tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    all_cells = spec["color_cells"] | spec["wall_cells"]
    if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
        return False
    if spec["color_cells"] & occupied_color:
        return False
    if spec["color_cells"] & occupied_wall:
        return False
    if spec["wall_cells"] & occupied_color:
        return False
    if spec["wall_cells"] & occupied_wall:
        return False
    if any(_is_border(h, w, r, c) for r, c in spec["wall_cells"]):
        return False

    for endpoint in spec["residues"]:
        if endpoint in border_points:
            return False
        rr, cc = endpoint
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) in border_points:
                return False
    return _residue_isolated(border_points | spec["residues"], h, w)


def _build_grids(h: int, w: int, specs: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    output = np.full((h, w), BG, dtype=int)
    for spec in specs:
        for r, c in spec["wall_cells"]:
            output[r, c] = WALL
    for spec in specs:
        for r, c in spec["color_cells"]:
            output[r, c] = int(spec["color"])

    inp = np.full((h, w), BG, dtype=int)
    inp[output == WALL] = WALL
    for spec in specs:
        for r, c in spec["residues"]:
            inp[r, c] = int(spec["color"])
    return inp, output


def _choose_patterns(count: int) -> List[Set[str]]:
    patterns: List[Set[str]] = [set(random.choice(MIXED_PATTERNS))]
    while len(patterns) < count:
        patterns.append(set(random.choice(EXTRA_PATTERNS)))
    patterns[-1] = set(random.choice(AXIS_HIDDEN_PATTERNS))
    return patterns


def _wall_independent_choices(inp: np.ndarray, spec: dict) -> int:
    r, c = spec["pos"]
    residues = spec["residues"]
    h, w = inp.shape
    if residues == {(r, 0), (r, w - 1)}:
        return sum(int(inp[r, cc]) == BG for cc in range(1, w - 1))
    if residues == {(0, c), (h - 1, c)}:
        return sum(int(inp[rr, c]) == BG for rr in range(1, h - 1))
    return 0


def _semantic_checks(inp: np.ndarray, out: np.ndarray, specs: Sequence[dict]) -> bool:
    try:
        if not np.array_equal(main(inp), out):
            return False
    except AssertionError:
        return False

    residues = _border_residues(inp)
    if set(residues) != {int(spec["color"]) for spec in specs}:
        return False

    seen_axis_hidden = False
    seen_mixed = False
    for spec in specs:
        color = int(spec["color"])
        candidates = _find_candidates_for_color(inp, color)
        if candidates != [tuple(spec["pos"])]:
            return False

        blocked = spec["blocked"]
        open_dirs = spec["open"]
        if blocked in ({"U", "D"}, {"L", "R"}):
            seen_axis_hidden = True
            if _wall_independent_choices(inp, spec) < 3:
                return False
        if any(d in open_dirs for d in ("U", "D")) and any(d in open_dirs for d in ("L", "R")):
            seen_mixed = True

    if not seen_axis_hidden or not seen_mixed:
        return False
    return True


def _generate_example() -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(12, 20)
        w = random.randint(12, 20)
        count = random.randint(2, 3)
        colors = random.sample(PALETTE, count)
        patterns = _choose_patterns(count)

        specs: List[dict] = []
        occupied_color: Set[Tuple[int, int]] = set()
        occupied_wall: Set[Tuple[int, int]] = set()
        border_points: Set[Tuple[int, int]] = set()

        failed = False
        for color, pattern in zip(colors, patterns):
            placed = False
            for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
                spec = _propose_emitter(h, w, color, pattern)
                if spec is None:
                    continue
                if not _fits_scene(spec, occupied_color, occupied_wall, border_points, h, w):
                    continue
                specs.append(spec)
                occupied_color |= spec["color_cells"]
                occupied_wall |= spec["wall_cells"]
                border_points |= spec["residues"]
                placed = True
                break
            if not placed:
                failed = True
                break

        if failed:
            continue

        inp, out = _build_grids(h, w, specs)
        if _semantic_checks(inp, out, specs):
            return inp, out
    raise RuntimeError("failed to generate a valid residue-emitter scene")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, out = _generate_example()
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input failed")
