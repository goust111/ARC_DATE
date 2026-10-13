from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREY = int(Color.GREY)
TEAL = int(Color.TEAL)

K_SQUARE = "square"
K_HBAR = "hbar"
K_VBAR = "vbar"
KIND_ORDER = {K_SQUARE: 0, K_HBAR: 1, K_VBAR: 2}

OFFSETS = {
    K_SQUARE: ((0, 0), (0, 1), (1, 0), (1, 1)),
    K_HBAR: ((0, 0), (0, 1), (0, 2)),
    K_VBAR: ((0, 0), (1, 0), (2, 0)),
}
OUTPUT_COLORS = {
    K_SQUARE: TEAL,
    K_HBAR: BLUE,
    K_VBAR: RED,
}
INPUT_MARGIN = 3
GRID_ROWS = (15, 24)
GRID_COLS = (15, 24)
SQUARE_COUNT = (1, 3)
HBAR_COUNT = (1, 3)
VBAR_COUNT = (1, 3)
CORE_EXTRA_MAX = 2
GEN_ATTEMPTS = 800


def _sorted_piece_keys(pieces: list[dict]) -> list[tuple[str, int, int]]:
    return sorted(
        (str(piece["kind"]), int(piece["top"]), int(piece["left"])) for piece in pieces
    )


def _piece_cells(kind: str, top: int, left: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(top) + dr, int(left) + dc) for dr, dc in OFFSETS[kind])


def _piece_patch(kind: str, color: int) -> np.ndarray:
    offsets = OFFSETS[kind]
    height = max(dr for dr, _dc in offsets) + 1
    width = max(dc for _dr, dc in offsets) + 1
    patch = np.full((height, width), BG, dtype=int)
    for dr, dc in offsets:
        patch[dr, dc] = int(color)
    return patch


def _enumerate_candidates(mask: np.ndarray) -> list[dict]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cell_index = {coord: idx for idx, coord in enumerate(coords)}
    height, width = map(int, mask.shape)
    candidates: list[dict] = []

    for kind in (K_SQUARE, K_HBAR, K_VBAR):
        offsets = OFFSETS[kind]
        max_r = max(dr for dr, _dc in offsets)
        max_c = max(dc for _dr, dc in offsets)
        for top in range(height - max_r):
            for left in range(width - max_c):
                cells = tuple((top + dr, left + dc) for dr, dc in offsets)
                if not all(mask[r, c] for r, c in cells):
                    continue
                bit = 0
                for cell in cells:
                    bit |= 1 << cell_index[cell]
                candidates.append({"kind": kind, "top": int(top), "left": int(left), "bit": bit})

    candidates.sort(key=lambda cand: (cand["top"], cand["left"], KIND_ORDER[str(cand["kind"])]))
    return candidates


def _cover_solutions(mask: np.ndarray, limit: int = 2) -> list[list[dict]]:
    candidates = _enumerate_candidates(mask)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cell_to_candidates: list[list[int]] = [[] for _ in cells]

    for cand_idx, cand in enumerate(candidates):
        bit = int(cand["bit"])
        for cell_idx in range(len(cells)):
            if bit & (1 << cell_idx):
                cell_to_candidates[cell_idx].append(cand_idx)

    full_cover = (1 << len(cells)) - 1
    solutions: list[list[dict]] = []

    def search(covered: int, chosen: list[int]) -> None:
        if len(solutions) >= int(limit):
            return
        if covered == full_cover:
            solutions.append([candidates[idx] for idx in chosen])
            return

        best_options: list[int] | None = None
        for cell_idx in range(len(cells)):
            if covered & (1 << cell_idx):
                continue
            options = [
                cand_idx
                for cand_idx in cell_to_candidates[cell_idx]
                if int(candidates[cand_idx]["bit"]) & covered == 0
            ]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_options is not None
        for cand_idx in best_options:
            chosen.append(cand_idx)
            search(covered | int(candidates[cand_idx]["bit"]), chosen)
            chosen.pop()

    if cells:
        search(0, [])
    return solutions


def _decompose_component(mask: np.ndarray) -> list[dict]:
    solutions = _cover_solutions(mask, limit=2)
    assert len(solutions) == 1, "each grey component must have exactly one primitive decomposition"
    solution = solutions[0]
    solution.sort(key=lambda cand: (cand["top"], cand["left"], KIND_ORDER[str(cand["kind"])]))
    return solution


def _decompose_grid(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BG,
        colors=[GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected at least one grey component"

    pieces: list[dict] = []
    for component in sorted(components, key=lambda comp: tuple(map(int, comp["bbox"]))):
        bbox = tuple(map(int, component["bbox"]))
        r0, c0, _r1, _c1 = bbox
        patch = crop_bbox(grid, bbox)
        mask = patch == GREY
        local_pieces = _decompose_component(mask)
        covered = sum(len(OFFSETS[str(piece["kind"])]) for piece in local_pieces)
        assert covered == int(component["area"])
        for piece in local_pieces:
            pieces.append(
                {
                    "kind": str(piece["kind"]),
                    "top": int(r0 + int(piece["top"])),
                    "left": int(c0 + int(piece["left"])),
                }
            )

    pieces.sort(key=lambda piece: (piece["top"], piece["left"], KIND_ORDER[piece["kind"]]))
    return pieces


def _output_piece_cells(piece: dict) -> tuple[tuple[int, int], ...]:
    kind = str(piece["kind"])
    top = int(piece["top"])
    left = int(piece["left"])
    if kind == K_SQUARE:
        return _piece_cells(kind, top, left)
    if kind == K_HBAR:
        return _piece_cells(kind, top, 0)
    if kind == K_VBAR:
        return _piece_cells(kind, 0, left)
    raise ValueError(f"unknown primitive kind {kind}")


def _expected_output(shape: tuple[int, int], pieces: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for piece in pieces:
        cells = _output_piece_cells(piece)
        rows = np.array([r for r, _c in cells], dtype=int)
        cols = np.array([c for _r, c in cells], dtype=int)
        assert not bool(np.any(occupied[rows, cols])), "primitive destinations must not overlap"
        occupied[rows, cols] = True
        out[rows, cols] = int(OUTPUT_COLORS[str(piece["kind"])])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each grey composite block into primitive squares and bars, then move each primitive by its shape class.

    Input structure:
    - The background is black.
    - Foreground cells are grey and form one or more 4-connected composite blocks.
    - Every grey block can be partitioned in exactly one way into 2x2 squares, 1x3 horizontal bars, and 3x1 vertical bars.

    Rule:
    - Recover that unique primitive decomposition.
    - Recolor every 2x2 square teal in place.
    - Slide every 1x3 horizontal bar along its row to the left edge and recolor it blue.
    - Slide every 3x1 vertical bar along its column to the top edge and recolor it red; vacated cells become background.

    Key insight:
    - Connected components are only camouflage: the true action units are the hidden primitive tiles, and each tile's shape alone determines both its destination and its new color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, GREY}, "expected only black background and grey foreground"

    pieces = _decompose_grid(g)
    return _expected_output(g.shape, pieces)


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = map(int, cell)
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = map(int, cell)
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _touches4(cells: tuple[tuple[int, int], ...], others: set[tuple[int, int]]) -> bool:
    return any(neighbor in others for cell in cells for neighbor in _neighbors4(cell))


def _touches8(cells: tuple[tuple[int, int], ...], others: set[tuple[int, int]]) -> bool:
    return any(neighbor in others for cell in cells for neighbor in _neighbors8(cell))


def _candidate_positions(kind: str, shape: tuple[int, int], used_h_rows: set[int], used_v_cols: set[int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    offsets = OFFSETS[kind]
    max_r = max(dr for dr, _dc in offsets)
    max_c = max(dc for _dr, dc in offsets)
    positions = []
    for top in range(INPUT_MARGIN, height - max_r):
        for left in range(INPUT_MARGIN, width - max_c):
            if kind == K_HBAR and top in used_h_rows:
                continue
            if kind == K_VBAR and left in used_v_cols:
                continue
            positions.append((int(top), int(left)))
    random.shuffle(positions)
    return positions


def _place_piece(
    grid: np.ndarray,
    pieces: list[dict],
    components: list[set[tuple[int, int]]],
    kind: str,
    target_component: int | None,
    used_h_rows: set[int],
    used_v_cols: set[int],
) -> bool:
    occupied = {cell for component in components for cell in component}
    for top, left in _candidate_positions(kind, grid.shape, used_h_rows, used_v_cols):
        cells = _piece_cells(kind, top, left)
        if any(cell in occupied for cell in cells):
            continue
        if target_component is None:
            if _touches8(cells, occupied):
                continue
        else:
            target_cells = components[target_component]
            if not _touches4(cells, target_cells):
                continue
            other_cells = occupied - target_cells
            if _touches8(cells, other_cells):
                continue

        paste(grid, _piece_patch(kind, GREY), top, left, transparent=BG)
        piece = {"kind": kind, "top": int(top), "left": int(left)}
        pieces.append(piece)
        if target_component is None:
            components.append(set(cells))
        else:
            components[target_component].update(cells)
        if kind == K_HBAR:
            used_h_rows.add(int(top))
        elif kind == K_VBAR:
            used_v_cols.add(int(left))
        return True
    return False


def _connected_component_piece_counts(grid: np.ndarray, pieces: list[dict]) -> list[dict]:
    comp_map = {}
    for idx, component in enumerate(
        find_components(
            grid,
            bg=BG,
            colors=[GREY],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
    ):
        for r, c in np.asarray(component["coords"], dtype=int):
            comp_map[(int(r), int(c))] = idx

    summary: list[dict] = []
    for component_idx in sorted(set(comp_map.values())):
        members = []
        kinds = set()
        for piece in pieces:
            cells = _piece_cells(str(piece["kind"]), int(piece["top"]), int(piece["left"]))
            owner = comp_map[cells[0]]
            if owner == component_idx:
                members.append(piece)
                kinds.add(str(piece["kind"]))
        summary.append({"count": len(members), "kinds": kinds})
    return summary


def _build_case() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    grid = np.full((height, width), BG, dtype=int)

    n_square = random.randint(*SQUARE_COUNT)
    n_hbar = random.randint(*HBAR_COUNT)
    n_vbar = random.randint(*VBAR_COUNT)
    all_kinds = [K_SQUARE] * n_square + [K_HBAR] * n_hbar + [K_VBAR] * n_vbar
    random.shuffle(all_kinds)

    core = [K_SQUARE, K_HBAR, K_VBAR]
    leftovers = all_kinds.copy()
    for kind in core:
        leftovers.remove(kind)
    random.shuffle(leftovers)
    core_size = 3 + random.randint(0, min(CORE_EXTRA_MAX, len(leftovers)))
    core.extend(leftovers[: core_size - 3])
    remaining = leftovers[core_size - 3 :]

    pieces: list[dict] = []
    components: list[set[tuple[int, int]]] = []
    used_h_rows: set[int] = set()
    used_v_cols: set[int] = set()

    first_kind = core[0]
    if not _place_piece(grid, pieces, components, first_kind, None, used_h_rows, used_v_cols):
        return None
    core_component = 0
    for kind in core[1:]:
        if not _place_piece(grid, pieces, components, kind, core_component, used_h_rows, used_v_cols):
            return None

    random.shuffle(remaining)
    while remaining:
        group_size = 2 if len(remaining) >= 2 and random.random() < 0.5 else 1
        group = [remaining.pop() for _ in range(group_size)]
        if not _place_piece(grid, pieces, components, group[0], None, used_h_rows, used_v_cols):
            return None
        component_idx = len(components) - 1
        for kind in group[1:]:
            if not _place_piece(grid, pieces, components, kind, component_idx, used_h_rows, used_v_cols):
                return None

    component_summary = _connected_component_piece_counts(grid, pieces)
    if not any(item["count"] >= 2 and len(item["kinds"]) >= 2 for item in component_summary):
        return None

    try:
        recovered = _decompose_grid(grid)
    except AssertionError:
        return None
    if _sorted_piece_keys(recovered) != _sorted_piece_keys(pieces):
        return None

    out = _expected_output(grid.shape, pieces)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid, out, pieces


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, _out, _pieces = built
        return grid
    raise RuntimeError(f"failed to generate a valid puzzle instance after {GEN_ATTEMPTS} attempts")
