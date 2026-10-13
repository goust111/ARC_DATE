from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
EMITTER = 4
PRISM = 3
ABSORBER = 6
ORANGE = 7
AZURE = 8
RED = 2
RAY_COLORS = (ORANGE, AZURE, RED)

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

GENERATE_RETRIES = 300
PLACEMENT_RETRIES = 40
DECOR_RETRIES = 60


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_prisms(grid: np.ndarray) -> dict[tuple[int, int], dict[str, int | str]]:
    prism_at: dict[tuple[int, int], dict[str, int | str]] = {}
    for comp in _components4(grid == PRISM):
        assert len(comp) == 2, "each prism must be a 2-cell component"
        rows = sorted(r for r, _ in comp)
        cols = sorted(c for _, c in comp)
        if rows[0] == rows[1] and cols[1] == cols[0] + 1:
            info: dict[str, int | str] = {"orient": "H", "r0": rows[0], "c0": cols[0]}
        elif cols[0] == cols[1] and rows[1] == rows[0] + 1:
            info = {"orient": "V", "r0": rows[0], "c0": cols[0]}
        else:
            raise AssertionError("each prism must be a 1x2 or 2x1 rectangle")
        for cell in comp:
            prism_at[cell] = info
    return prism_at


def _compatible_prism(orient: str, direction: str) -> bool:
    return (orient == "H" and direction in ("U", "D")) or (orient == "V" and direction in ("L", "R"))


def _split_children(info: dict[str, int | str], direction: str) -> list[tuple[int, int, str]]:
    r0 = int(info["r0"])
    c0 = int(info["c0"])
    orient = str(info["orient"])
    if orient == "H":
        if direction == "U":
            return [(r0 - 1, c0, "L"), (r0 - 1, c0 + 1, "R")]
        assert direction == "D"
        return [(r0 + 1, c0, "L"), (r0 + 1, c0 + 1, "R")]
    assert orient == "V"
    if direction == "L":
        return [(r0, c0 - 1, "U"), (r0 + 1, c0 - 1, "D")]
    assert direction == "R"
    return [(r0, c0 + 1, "U"), (r0 + 1, c0 + 1, "D")]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit three generations of cardinal light rays, splitting them through 2-cell prisms and stopping them at absorbers.

    Input structure:
    - The background is black.
    - Exactly one yellow singleton pixel is the emitter.
    - Green 2-cell components are prisms, each being either a horizontal 1x2 bar or a vertical 2x1 bar.
    - Magenta pixels are absorbers; they stay fixed and block any incoming ray.

    Rule:
    - From the emitter, cast four orange rays upward, downward, leftward, and rightward.
    - A ray paints every background cell it traverses until it would enter a prism, an absorber, or the grid exterior.
    - When an orange or azure ray hits a compatible prism, it stops before the prism and splits from the prism's far side into two perpendicular child rays; the child color advances orange->azure->red.
    - Red rays never split: on reaching any prism they simply stop before it. No ray paints the emitter, a prism cell, or an absorber cell.

    Key insight:
    - Each prism is an oriented 2-cell splitter: horizontal prisms only branch vertical arrivals into left/right exits, and vertical prisms only branch horizontal arrivals into up/down exits.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all(np.isin(g, [BG, EMITTER, PRISM, ABSORBER])), "input should contain only task colors"

    emitter_cells = np.argwhere(g == EMITTER)
    assert emitter_cells.shape == (1, 2), "expected exactly one emitter"
    emitter_r, emitter_c = map(int, emitter_cells[0])

    prism_at = _parse_prisms(g)
    h, w = g.shape
    out = g.copy()

    q: deque[tuple[int, int, str, int]] = deque()
    for direction, (dr, dc) in DIRS.items():
        start_r = emitter_r + dr
        start_c = emitter_c + dc
        if 0 <= start_r < h and 0 <= start_c < w:
            q.append((start_r, start_c, direction, 0))

    seen: set[tuple[int, int, str, int]] = set()
    while q:
        r, c, direction, generation = q.popleft()
        state = (r, c, direction, generation)
        if state in seen:
            continue
        seen.add(state)

        dr, dc = DIRS[direction]
        path: list[tuple[int, int]] = []
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            value = int(g[rr, cc])
            if value == BG:
                path.append((rr, cc))
                rr += dr
                cc += dc
                continue
            if value == ABSORBER or value == EMITTER:
                break
            assert value == PRISM
            info = prism_at[(rr, cc)]
            if generation < 2 and _compatible_prism(str(info["orient"]), direction):
                for child_r, child_c, child_dir in _split_children(info, direction):
                    if 0 <= child_r < h and 0 <= child_c < w:
                        q.append((child_r, child_c, child_dir, generation + 1))
            break

        beam_color = RAY_COLORS[generation]
        for pr, pc in path:
            out[pr, pc] = beam_color

    return out


class _Scene:
    def __init__(self, h: int, w: int, emitter: tuple[int, int]):
        self.h = h
        self.w = w
        self.grid = np.full((h, w), BG, dtype=int)
        self.grid[emitter] = EMITTER
        self.emitter = emitter
        self.occupied: set[tuple[int, int]] = {emitter}
        self.reserved: set[tuple[int, int]] = set()
        self.green_cells: set[tuple[int, int]] = set()

    def clone(self) -> "_Scene":
        other = _Scene.__new__(_Scene)
        other.h = self.h
        other.w = self.w
        other.grid = self.grid.copy()
        other.emitter = self.emitter
        other.occupied = set(self.occupied)
        other.reserved = set(self.reserved)
        other.green_cells = set(self.green_cells)
        return other


def _advance(cell: tuple[int, int], direction: str, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return cell[0] + dr * steps, cell[1] + dc * steps


def _in_bounds(scene: _Scene, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < scene.h and 0 <= c < scene.w


def _line_to_boundary(scene: _Scene, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    line: list[tuple[int, int]] = []
    r, c = start
    dr, dc = DIRS[direction]
    while 0 <= r < scene.h and 0 <= c < scene.w:
        line.append((r, c))
        r += dr
        c += dc
    return line


def _path_is_clear(scene: _Scene, cells: list[tuple[int, int]]) -> bool:
    return all(cell not in scene.occupied and cell not in scene.reserved for cell in cells)


def _reserve_path(scene: _Scene, cells: list[tuple[int, int]]) -> None:
    for cell in cells:
        scene.reserved.add(cell)


def _place_absorber(scene: _Scene, cell: tuple[int, int]) -> bool:
    if cell in scene.occupied or cell in scene.reserved:
        return False
    scene.occupied.add(cell)
    scene.grid[cell] = ABSORBER
    return True


def _prism_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _place_prism(scene: _Scene, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for cell in cells:
        if not _in_bounds(scene, cell):
            return False
        if cell in scene.occupied or cell in scene.reserved:
            return False
        for nbr in _prism_neighbors(cell):
            if nbr in scene.green_cells and nbr not in cell_set:
                return False
    for cell in cells:
        scene.occupied.add(cell)
        scene.green_cells.add(cell)
        scene.grid[cell] = PRISM
    return True


def _split_candidate(
    scene: _Scene,
    hit_cell: tuple[int, int],
    direction: str,
    lane_index: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]] | None:
    r, c = hit_cell
    if direction in ("U", "D"):
        left = c - lane_index
        cells = [(r, left), (r, left + 1)]
        exits_row = r - 1 if direction == "U" else r + 1
        children = [(exits_row, left, "L"), (exits_row, left + 1, "R")]
    else:
        top = r - lane_index
        cells = [(top, c), (top + 1, c)]
        exits_col = c - 1 if direction == "L" else c + 1
        children = [(top, exits_col, "U"), (top + 1, exits_col, "D")]

    if not all(_in_bounds(scene, cell) for cell in cells):
        return None
    return cells, children


def _build_beam(scene: _Scene, start: tuple[int, int], direction: str, generation: int, spec: dict[str, object]) -> _Scene | None:
    if not _in_bounds(scene, start):
        return None

    line = _line_to_boundary(scene, start, direction)
    if not line:
        return None

    kind = str(spec["kind"])
    if kind == "edge":
        if not _path_is_clear(scene, line):
            return None
        out = scene.clone()
        _reserve_path(out, line)
        return out

    candidates: list[tuple[str, list[tuple[int, int]], object]] = []

    if kind == "absorber":
        for path_len in range(1, len(line)):
            path = line[:path_len]
            stop_cell = line[path_len]
            if _path_is_clear(scene, path) and stop_cell not in scene.occupied and stop_cell not in scene.reserved:
                candidates.append(("absorber", path, stop_cell))

    elif kind in {"split", "prism_stop"}:
        if kind == "split" and generation >= 2:
            return None
        for path_len in range(1, len(line)):
            path = line[:path_len]
            if not _path_is_clear(scene, path):
                continue
            hit_cell = line[path_len]
            for lane_index in random.sample([0, 1], k=2):
                cand = _split_candidate(scene, hit_cell, direction, lane_index)
                if cand is None:
                    continue
                prism_cells, children = cand
                if kind == "split":
                    if not all(_in_bounds(scene, (r, c)) for r, c, _ in children):
                        continue
                    if any((r, c) in scene.occupied or (r, c) in scene.reserved for r, c, _ in children):
                        continue
                candidates.append((kind, path, (prism_cells, children)))
    else:
        raise AssertionError(f"unknown beam kind: {kind}")

    random.shuffle(candidates)
    for cand_kind, path, payload in candidates:
        out = scene.clone()
        _reserve_path(out, path)
        if cand_kind == "absorber":
            if _place_absorber(out, payload):
                return out
            continue

        prism_cells, children = payload
        if not _place_prism(out, prism_cells):
            continue
        if cand_kind == "prism_stop":
            return out

        child_specs = list(spec["children"])
        built = out
        success = True
        for child_spec, (child_r, child_c, child_dir) in zip(child_specs, children):
            built = _build_beam(built, (child_r, child_c), child_dir, generation + 1, child_spec)
            if built is None:
                success = False
                break
        if success:
            return built

    return None


def _terminal_spec() -> dict[str, object]:
    return {"kind": random.choice(["edge", "absorber"])}


def _sample_config() -> dict[str, object]:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    margin = 5
    emitter = (random.randint(margin, h - margin - 1), random.randint(margin, w - margin - 1))

    dirs = list(DIRS.keys())
    random.shuffle(dirs)
    chain_dir, absorber_dir, edge_dir, extra_dir = dirs

    second_split_idx = random.randrange(2)
    red_stop_idx = random.randrange(2)

    red_children = [_terminal_spec(), _terminal_spec()]
    red_children[red_stop_idx] = {"kind": "prism_stop"}
    azure_children = [_terminal_spec(), _terminal_spec()]
    azure_children[second_split_idx] = {"kind": "split", "children": red_children}
    chain_spec = {"kind": "split", "children": azure_children}

    extra_kind = random.choice(["edge", "absorber", "split"])
    if extra_kind == "split":
        extra_children = [_terminal_spec(), _terminal_spec()]
        extra_spec = {"kind": "split", "children": extra_children}
    else:
        extra_spec = {"kind": extra_kind}

    plan = [
        (chain_dir, chain_spec),
        (extra_dir, extra_spec),
        (absorber_dir, {"kind": "absorber"}),
        (edge_dir, {"kind": "edge"}),
    ]
    plan.sort(key=lambda item: 0 if str(item[1]["kind"]) == "split" else 1)

    return {
        "h": h,
        "w": w,
        "emitter": emitter,
        "plan": plan,
        "decoy_absorbers": random.randint(0, 2),
        "decoy_prisms": random.randint(0, 2),
    }


def _add_decoys(scene: _Scene, absorber_count: int, prism_count: int) -> _Scene:
    out = scene.clone()
    for _ in range(absorber_count):
        for _try in range(DECOR_RETRIES):
            cell = (random.randrange(out.h), random.randrange(out.w))
            if _place_absorber(out, cell):
                break

    for _ in range(prism_count):
        for _try in range(DECOR_RETRIES):
            if random.random() < 0.5:
                r = random.randrange(out.h)
                c = random.randrange(out.w - 1)
                cells = [(r, c), (r, c + 1)]
            else:
                r = random.randrange(out.h - 1)
                c = random.randrange(out.w)
                cells = [(r, c), (r + 1, c)]
            if _place_prism(out, cells):
                break
    return out


def _gen_attempt(config: dict[str, object]) -> np.ndarray | None:
    base = _Scene(int(config["h"]), int(config["w"]), tuple(config["emitter"]))

    built: _Scene | None = base
    for direction, spec in config["plan"]:
        start = _advance(base.emitter, str(direction))
        built = _build_beam(built, start, str(direction), 0, spec)
        if built is None:
            return None

    built = _add_decoys(built, int(config["decoy_absorbers"]), int(config["decoy_prisms"]))
    inp = built.grid

    try:
        out = main(inp)
    except AssertionError:
        return None

    if np.array_equal(inp, out):
        return None
    if not {ORANGE, AZURE, RED}.issubset(set(map(int, np.unique(out)))):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        config = _sample_config()
        for _try in range(PLACEMENT_RETRIES):
            grid = _gen_attempt(config)
            if grid is not None:
                return grid
    raise RuntimeError("generate_input failed to place a valid light-and-prism scene")
