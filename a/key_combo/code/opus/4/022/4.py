from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(TOOLBOX_DIR))

from arc_utils import Color, find_components, in_bounds

BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREY = Color.GREY

DIAGONALS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

DISTRACTOR_SHAPES = [
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
]

MAX_GEN_ATTEMPTS = 600


def _ordered_blue_objects(grid: np.ndarray) -> list[dict]:
    objects = find_components(
        grid,
        bg=BG,
        colors=[BLUE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 4 <= len(objects) <= 6, "expected 4-6 blue objects"
    top_rows = [int(obj["bbox"][0]) for obj in objects]
    assert len(set(top_rows)) == len(top_rows), "blue objects must have distinct top bbox rows"
    return sorted(objects, key=lambda obj: int(obj["bbox"][0]))


def _shifted_blue_cells(grid: np.ndarray) -> list[np.ndarray]:
    ordered = _ordered_blue_objects(grid)
    h, w = grid.shape
    shifted_objects: list[np.ndarray] = []
    occupied = np.zeros((h, w), dtype=bool)
    for rank, obj in enumerate(ordered, start=1):
        dc = -1 if rank % 2 == 1 else 1
        coords = np.asarray(obj["coords"], dtype=int)
        shifted = coords + np.array([0, dc], dtype=int)
        assert np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < w)), "shift would leave grid"
        assert not np.any(occupied[shifted[:, 0], shifted[:, 1]]), "shifted objects overlap"
        occupied[shifted[:, 0], shifted[:, 1]] = True
        shifted_objects.append(shifted)
    return shifted_objects


def _ray_paths(seed: tuple[int, int], blockers: set[tuple[int, int]], shape: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    rays: dict[str, list[tuple[int, int]]] = {}
    for name, (dr, dc) in DIAGONALS.items():
        r, c = seed
        path: list[tuple[int, int]] = []
        while True:
            nr, nc = r + dr, c + dc
            if not in_bounds(shape, nr, nc):
                break
            if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                break
            if (nr, nc) in blockers:
                break
            path.append((nr, nc))
            r, c = nr, nc
        rays[name] = path
    return rays


def _build_output(
    grid: np.ndarray,
    *,
    shifted_objects: list[np.ndarray],
    blocker_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full_like(grid, BG)
    for coords in shifted_objects:
        out[coords[:, 0], coords[:, 1]] = BLUE

    seed_cells = np.argwhere(grid == RED)
    assert seed_cells.shape == (1, 2), "expected exactly one red seed"
    seed = tuple(map(int, seed_cells[0]))
    out[seed] = RED

    for path in _ray_paths(seed, blocker_cells, grid.shape).values():
        for r, c in path:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift blue objects in alternating top-to-bottom order, then redraw diagonal red rays using their shifted positions as blockers.

    Input structure:
    - Black background with 4-6 blue 4-connected objects whose top bbox rows are all distinct.
    - Exactly one isolated red seed pixel sits away from the outer border.
    - Two or three isolated grey singleton pixels may appear as visual decoys.

    Rule:
    - Sort the blue objects by increasing top bbox row.
    - Shift ranks 1,3,5,... one column left and ranks 2,4,6,... one column right.
    - Erase every grey pixel.
    - From the red seed, extend red rays one cell at a time along NE, NW, SE, and SW; stop before a shifted blue pixel or before stepping onto the outer border.
    - Keep only the shifted blue objects, the red seed, and the painted red ray cells.

    Key insight:
    - The real blockers are the blue objects after the alternating shift; grey pixels only mimic blockers visually and must be ignored and erased.
    """
    g = np.asarray(grid, dtype=int)
    shifted_objects = _shifted_blue_cells(g)
    blocker_cells = {
        (int(r), int(c))
        for coords in shifted_objects
        for r, c in coords
    }
    return _build_output(g, shifted_objects=shifted_objects, blocker_cells=blocker_cells)


def _neighbor4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbor8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _shift_value(rank: int) -> int:
    return -1 if rank % 2 == 1 else 1


def _blocker_target_col(seed_c: int, row: int, seed_r: int, dc_diag: int) -> int:
    return seed_c + abs(row - seed_r) * dc_diag


def _is_on_seed_diagonal(seed: tuple[int, int], cell: tuple[int, int]) -> bool:
    dr = cell[0] - seed[0]
    dc = cell[1] - seed[1]
    return dr != 0 and abs(dr) == abs(dc)


def _cells_with_offset(cells: set[tuple[int, int]], dc: int) -> set[tuple[int, int]]:
    return {(r, c + dc) for r, c in cells}


def _touches_existing(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if cells & occupied:
        return True
    for cell in cells:
        for nb in _neighbor4(cell):
            if nb in occupied:
                return True
    return False


def _touches_seed(seed: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return any(max(abs(r - seed[0]), abs(c - seed[1])) <= 1 for r, c in cells)


def _candidate_blockers(rows: list[int], seed: tuple[int, int], width: int) -> dict[int, list[dict]]:
    seed_r, seed_c = seed
    out: dict[int, list[dict]] = {}
    for rank, row in enumerate(rows, start=1):
        dc_shift = _shift_value(rank)
        if abs(row - seed_r) < 2:
            continue
        vertical_name = "N" if row < seed_r else "S"
        for dc_diag, horiz_name in [(-1, "W"), (1, "E")]:
            target_c = _blocker_target_col(seed_c, row, seed_r, dc_diag)
            if not (1 <= target_c <= width - 2):
                continue
            max_len = target_c if dc_shift == 1 else (width - 1 - target_c)
            if max_len < 3:
                continue
            out.setdefault(rank, []).append(
                {
                    "rank": rank,
                    "row": row,
                    "dir_name": vertical_name + horiz_name,
                    "dc_diag": dc_diag,
                    "target_c": target_c,
                    "max_len": min(5, max_len),
                    "distance": abs(row - seed_r),
                }
            )
    return out


def _pick_blocker_specs(rng: random.Random, rows: list[int], seed: tuple[int, int], width: int) -> list[dict] | None:
    candidates = _candidate_blockers(rows, seed, width)
    ranks = list(candidates)
    if len(ranks) < 2:
        return None
    rank_parity = {rank: rank % 2 for rank in ranks}
    sizes = [3, 2, 2]
    for desired in sizes:
        for _ in range(160):
            chosen_ranks = rng.sample(ranks, k=min(desired, len(ranks)))
            if len({rank_parity[r] for r in chosen_ranks}) < 2:
                continue
            specs = []
            used_dirs: set[str] = set()
            good = True
            for rank in chosen_ranks:
                options = candidates[rank][:]
                rng.shuffle(options)
                pick = next((opt for opt in options if opt["dir_name"] not in used_dirs), None)
                if pick is None:
                    good = False
                    break
                specs.append(pick)
                used_dirs.add(pick["dir_name"])
            if not good:
                continue
            if not any(spec["distance"] >= 3 for spec in specs):
                continue
            if len(used_dirs) >= 2:
                return specs
    return None


def _blocker_cells_for_spec(spec: dict) -> set[tuple[int, int]]:
    rank = int(spec["rank"])
    row = int(spec["row"])
    target_c = int(spec["target_c"])
    length = int(spec["length"])
    dc_shift = _shift_value(rank)
    if dc_shift == -1:
        return {(row, c) for c in range(target_c + 1, target_c + length + 1)}
    return {(row, c) for c in range(target_c - length, target_c)}


def _random_distractor_cells(
    rng: random.Random,
    *,
    row: int,
    rank: int,
    shape: tuple[int, int],
    seed: tuple[int, int],
    input_occupied: set[tuple[int, int]],
    shifted_occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = shape
    dc_shift = _shift_value(rank)
    templates = DISTRACTOR_SHAPES[:]
    rng.shuffle(templates)
    for rel in templates:
        rel_rows = [r for r, _ in rel]
        rel_cols = [c for _, c in rel]
        height = max(rel_rows) + 1
        width = max(rel_cols) + 1
        if row + height > h:
            continue
        cols = list(range(0, w - width + 1))
        rng.shuffle(cols)
        for c0 in cols:
            cells = {(row + dr, c0 + dc) for dr, dc in rel}
            shifted = _cells_with_offset(cells, dc_shift)
            if any(not in_bounds(shape, r, c) for r, c in shifted):
                continue
            if _touches_existing(cells, input_occupied):
                continue
            if _touches_existing(shifted, shifted_occupied):
                continue
            if _touches_seed(seed, cells):
                continue
            if any(_is_on_seed_diagonal(seed, cell) for cell in cells):
                continue
            if any(_is_on_seed_diagonal(seed, cell) for cell in shifted):
                continue
            return cells
    return None


def _rays_from_output(out: np.ndarray, seed: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    rays = {}
    for name, (dr, dc) in DIAGONALS.items():
        path = []
        r, c = seed
        while True:
            r += dr
            c += dc
            if not in_bounds(out.shape, r, c):
                break
            if out[r, c] != RED:
                break
            path.append((r, c))
        rays[name] = path
    return rays


def _wrong_output_original_blockers(grid: np.ndarray) -> np.ndarray:
    shifted_objects = _shifted_blue_cells(grid)
    original_objects = _ordered_blue_objects(grid)
    blocker_cells = {
        (int(r), int(c))
        for obj in original_objects
        for r, c in np.asarray(obj["coords"], dtype=int)
    }
    return _build_output(grid, shifted_objects=shifted_objects, blocker_cells=blocker_cells)


def _wrong_output_grey_blockers(grid: np.ndarray) -> np.ndarray:
    shifted_objects = _shifted_blue_cells(grid)
    blocker_cells = {
        (int(r), int(c))
        for coords in shifted_objects
        for r, c in coords
    }
    grey_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    return _build_output(grid, shifted_objects=shifted_objects, blocker_cells=blocker_cells | grey_cells)


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    red_cells = np.argwhere(grid == RED)
    grey_cells = np.argwhere(grid == GREY)
    if red_cells.shape != (1, 2) or not (2 <= len(grey_cells) <= 3):
        return False

    seed = tuple(map(int, red_cells[0]))
    if seed[0] in {0, grid.shape[0] - 1} or seed[1] in {0, grid.shape[1] - 1}:
        return False

    try:
        ordered = _ordered_blue_objects(grid)
        shifted = _shifted_blue_cells(grid)
    except AssertionError:
        return False

    if any(not (3 <= len(obj["coords"]) <= 8) for obj in ordered):
        return False

    input_blue = {(int(r), int(c)) for obj in ordered for r, c in np.asarray(obj["coords"], dtype=int)}
    for cell in grey_cells:
        if any(max(abs(int(cell[0]) - r), abs(int(cell[1]) - c)) <= 1 for r, c in input_blue):
            return False
    if any(max(abs(seed[0] - r), abs(seed[1] - c)) <= 1 for r, c in input_blue):
        return False

    out = main(grid)
    wrong_original = _wrong_output_original_blockers(grid)
    wrong_grey = _wrong_output_grey_blockers(grid)
    if np.array_equal(out, wrong_original) or np.array_equal(out, wrong_grey):
        return False

    rays = _rays_from_output(out, seed)
    ray_lengths = [len(path) for path in rays.values()]
    if max(ray_lengths) < 1:
        return False
    if not any(length >= 2 for length in ray_lengths):
        return False

    shifted_cells = {(int(r), int(c)) for coords in shifted for r, c in coords}
    blocked = 0
    open_dirs = 0
    for name, path in rays.items():
        dr, dc = DIAGONALS[name]
        terminal = (seed[0] + dr * (len(path) + 1), seed[1] + dc * (len(path) + 1))
        if in_bounds(grid.shape, *terminal) and terminal[0] not in {0, grid.shape[0] - 1} and terminal[1] not in {0, grid.shape[1] - 1}:
            if terminal in shifted_cells:
                blocked += 1
        else:
            open_dirs += 1
    if blocked < 2 or open_dirs < 1:
        return False

    red_path_cells = {cell for path in rays.values() for cell in path}
    if not any(tuple(map(int, cell)) in red_path_cells for cell in grey_cells):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 16)
    w = rng.randint(13, 16)
    seed = (rng.randint(3, h - 4), rng.randint(3, w - 4))
    object_count = rng.randint(4, 6)

    candidate_rows = [r for r in range(1, h - 1) if r not in {seed[0] - 1, seed[0], seed[0] + 1}]
    if len(candidate_rows) < object_count:
        return None
    rows = sorted(rng.sample(candidate_rows, object_count))

    blocker_specs = _pick_blocker_specs(rng, rows, seed, w)
    if blocker_specs is None:
        return None

    blockers_by_rank = {int(spec["rank"]): dict(spec) for spec in blocker_specs}
    for spec in blockers_by_rank.values():
        spec["length"] = rng.randint(3, int(spec["max_len"]))

    input_occupied: set[tuple[int, int]] = set()
    shifted_occupied: set[tuple[int, int]] = set()
    object_cells_by_rank: dict[int, set[tuple[int, int]]] = {}

    for rank, row in enumerate(rows, start=1):
        if rank in blockers_by_rank:
            cells = _blocker_cells_for_spec(blockers_by_rank[rank])
        else:
            cells = _random_distractor_cells(
                rng,
                row=row,
                rank=rank,
                shape=(h, w),
                seed=seed,
                input_occupied=input_occupied,
                shifted_occupied=shifted_occupied,
            )
            if cells is None:
                return None

        shifted = _cells_with_offset(cells, _shift_value(rank))
        if any(not in_bounds((h, w), r, c) for r, c in cells | shifted):
            return None
        if _touches_existing(cells, input_occupied) or _touches_existing(shifted, shifted_occupied):
            return None
        if _touches_seed(seed, cells):
            return None
        if len(cells) < 3 or len(cells) > 8:
            return None

        input_occupied |= cells
        shifted_occupied |= shifted
        object_cells_by_rank[rank] = cells

    grid = np.zeros((h, w), dtype=int)
    for cells in object_cells_by_rank.values():
        for r, c in cells:
            grid[r, c] = BLUE
    grid[seed] = RED

    provisional = main(grid)
    rays = _rays_from_output(provisional, seed)
    path_cells = [cell for path in rays.values() for cell in path]
    if len(path_cells) < 2:
        return None

    grey_total = rng.randint(2, 3)
    path_candidates = [cell for cell in path_cells if grid[cell] == BG and not _touches_seed(seed, {cell})]
    rng.shuffle(path_candidates)
    if not path_candidates:
        return None

    grey_cells: list[tuple[int, int]] = [path_candidates[0]]
    free_cells = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if grid[r, c] == BG and (r, c) != seed
    ]
    rng.shuffle(free_cells)
    for cell in free_cells:
        if len(grey_cells) >= grey_total:
            break
        if cell in grey_cells:
            continue
        if any(max(abs(cell[0] - b[0]), abs(cell[1] - b[1])) <= 1 for b in input_occupied):
            continue
        if max(abs(cell[0] - seed[0]), abs(cell[1] - seed[1])) <= 1:
            continue
        grey_cells.append(cell)
    if len(grey_cells) != grey_total:
        return None

    for cell in grey_cells:
        grid[cell] = GREY

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
