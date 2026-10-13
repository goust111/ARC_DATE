from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask


BLACK = Color.BLACK
GREY = Color.GREY
ORANGE = Color.ORANGE
ENDPOINT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON)

GEN_RETRIES = 500


def _regular_spacing(indices: list[int]) -> bool:
    if len(indices) < 2:
        return False
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    return len(set(gaps)) == 1 and gaps[0] in (4, 5)


def _infer_separators(grid: np.ndarray, axis: int) -> list[int]:
    counts = np.sum(grid == GREY, axis=axis)
    best: tuple[int, list[int]] | None = None
    for k in (5, 6):
        if grid.shape[1 - axis] < k:
            continue
        idx = sorted(int(v) for v in np.argsort(counts)[-k:])
        if not _regular_spacing(idx):
            continue
        score = int(sum(int(counts[i]) for i in idx))
        if best is None or score > best[0]:
            best = (score, idx)
    assert best is not None, "failed to identify separator lines"
    return best[1]


def _bands_from_separators(seps: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    for a, b in zip(seps, seps[1:]):
        assert b - a > 1, "separator lines must enclose non-empty cells"
        bands.append((a + 1, b - 1))
    return bands


def _grid_bands(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = _infer_separators(grid, axis=1)
    col_seps = _infer_separators(grid, axis=0)
    row_bands = _bands_from_separators(row_seps)
    col_bands = _bands_from_separators(col_seps)
    assert len(row_bands) == len(col_bands)
    assert len(row_bands) in (4, 5)
    return row_seps, col_seps, row_bands, col_bands


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= value <= hi:
            return idx
    raise AssertionError("coordinate must lie inside a cell band")


def _endpoint_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in ENDPOINT_COLORS:
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        assert coords.shape[0] == 2, "each endpoint color must appear exactly twice"
        pairs[int(color)] = (tuple(map(int, coords[0])), tuple(map(int, coords[1])))
    assert 2 <= len(pairs) <= 3, "expected two or three endpoint pairs"
    return pairs


def _obstacle_bboxes_from_visible_orange(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = components_mask(grid == ORANGE, connectivity=4)
    assert 1 <= len(comps) <= 2, "expected one or two orange obstacle components"
    bboxes = []
    for comp in comps:
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        bboxes.append((int(r0), int(c0), int(r1), int(c1)))
    return sorted(bboxes)


def _mask_from_bboxes(shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment endpoints must share a row or column")


def _clear_path(coords: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    return all(not blocked[r, c] for r, c in coords[1:-1])


def _choose_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    blocked: np.ndarray,
    *,
    allow_turns: bool,
) -> tuple[str, list[tuple[int, int]]] | None:
    if p1[0] == p2[0]:
        straight = _segment(p1, p2)
        if _clear_path(straight, blocked):
            return "straight", straight
    if p1[1] == p2[1]:
        straight = _segment(p1, p2)
        if _clear_path(straight, blocked):
            return "straight", straight
    if not allow_turns:
        return None

    corner_hv = (p1[0], p2[1])
    hv = _segment(p1, corner_hv) + _segment(corner_hv, p2)[1:]
    if _clear_path(hv, blocked):
        return "hv", hv

    corner_vh = (p2[0], p1[1])
    vh = _segment(p1, corner_vh) + _segment(corner_vh, p2)[1:]
    if _clear_path(vh, blocked):
        return "vh", vh
    return None


def _render_solution(
    grid: np.ndarray,
    *,
    use_bbox_obstacles: bool,
    allow_turns: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _row_seps, _col_seps, row_bands, col_bands = _grid_bands(g)

    pairs = _endpoint_pairs(g)
    for p1, p2 in pairs.values():
        same_row_band = _band_index(p1[0], row_bands) == _band_index(p2[0], row_bands)
        same_col_band = _band_index(p1[1], col_bands) == _band_index(p2[1], col_bands)
        assert same_row_band or same_col_band, "paired endpoints must share a cell-row or cell-column"

    if use_bbox_obstacles:
        obstacle_bboxes = _obstacle_bboxes_from_visible_orange(g)
        blocked = _mask_from_bboxes(g.shape, obstacle_bboxes)
    else:
        blocked = g == ORANGE

    out = g.copy()
    for color in sorted(pairs):
        p1, p2 = pairs[color]
        choice = _choose_path(p1, p2, blocked, allow_turns=allow_turns)
        assert choice is not None, "every pair must have a valid prescribed path"
        _mode, coords = choice
        for r, c in coords:
            if out[r, c] == BLACK:
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color endpoint pixels across a separator grid while routing around each orange rectangle's full implied footprint.

    Input structure
    - A black background is divided into a 4x4 or 5x5 lattice by grey separator rows and columns.
    - Two or three non-grey colors appear as endpoint pixels, with each color appearing exactly twice in cells that share a lattice row or lattice column.
    - One or two orange rectangles may span multiple cells; endpoint pixels can hide a few orange cells, but each visible orange component still determines one rectangular footprint.

    Rule
    - Identify the separator lattice and pair the two occurrences of each endpoint color.
    - For each visible orange 4-connected component, take its axis-aligned bbox and treat that whole rectangle as blocked, even where endpoint pixels hide some orange cells.
    - Connect each endpoint pair in color order: use a straight row segment when both points share a row and that segment avoids blocked footprint cells, otherwise use a straight column segment when they share a column and that segment avoids the footprint.
    - If no straight segment is available, try the two L-shapes in the stated order: horizontal-then-vertical, else vertical-then-horizontal; recolor only black cells along the chosen path and leave separators, obstacles, and endpoints unchanged.

    Key insight
    - The orange obstacle is not just its visible orange pixels: the entire bbox of each visible orange component is impassable, and that hidden rectangular footprint determines which of the two orthogonal detours is legal.
    """
    return _render_solution(grid, use_bbox_obstacles=True, allow_turns=True)


def _build_lattice(n_cells: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h = n_cells * cell_h + (n_cells + 1)
    w = n_cells * cell_w + (n_cells + 1)
    grid = np.zeros((h, w), dtype=int)
    row_seps = [i * (cell_h + 1) for i in range(n_cells + 1)]
    col_seps = [i * (cell_w + 1) for i in range(n_cells + 1)]
    grid[row_seps, :] = GREY
    grid[:, col_seps] = GREY
    row_bands = _bands_from_separators(row_seps)
    col_bands = _bands_from_separators(col_seps)
    return grid, row_seps, col_seps, row_bands, col_bands


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_obstacles(
    grid: np.ndarray,
    row_seps: list[int],
    col_seps: list[int],
) -> list[tuple[int, int, int, int]] | None:
    h, w = map(int, grid.shape)
    count = 1 if random.random() < 0.58 else 2
    want_crossing = random.random() < 0.78
    bboxes: list[tuple[int, int, int, int]] = []

    for idx in range(count):
        for _ in range(120):
            hh = random.randint(3, min(6, h - 4))
            ww = random.randint(3, min(6, w - 4))
            r0 = random.randint(1, h - hh - 2)
            c0 = random.randint(1, w - ww - 2)
            bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)
            if any(_bbox_overlap(bbox, other, pad=1) for other in bboxes):
                continue
            if idx == 0 and want_crossing:
                crosses = any(r0 < sep < bbox[2] for sep in row_seps[1:-1]) or any(c0 < sep < bbox[3] for sep in col_seps[1:-1])
                if not crosses:
                    continue
            bboxes.append(bbox)
            break
        else:
            return None

    for r0, c0, r1, c1 in bboxes:
        grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    return sorted(bboxes)


def _obstacle_border_mask(shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _cell_coords(
    row_band: tuple[int, int],
    col_band: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, r1 = row_band
    c0, c1 = col_band
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _aligned_cell_pairs(n_cells: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for r in range(n_cells):
        for c1 in range(n_cells):
            for c2 in range(c1 + 1, n_cells):
                pairs.append(((r, c1), (r, c2)))
    for c in range(n_cells):
        for r1 in range(n_cells):
            for r2 in range(r1 + 1, n_cells):
                pairs.append(((r1, c), (r2, c)))
    random.shuffle(pairs)
    return pairs


def _placement_candidates(
    grid: np.ndarray,
    row_band: tuple[int, int],
    col_band: tuple[int, int],
    occupied_endpoints: set[tuple[int, int]],
    border_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    regular: list[tuple[int, int]] = []
    overlay: list[tuple[int, int]] = []
    for coord in _cell_coords(row_band, col_band):
        if coord in occupied_endpoints:
            continue
        r, c = coord
        if grid[r, c] == BLACK:
            regular.append(coord)
        elif grid[r, c] == ORANGE and border_mask[r, c]:
            overlay.append(coord)
    random.shuffle(regular)
    random.shuffle(overlay)
    return regular, overlay


def _find_pair_placement(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    blocked: np.ndarray,
    border_mask: np.ndarray,
    occupied_endpoints: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    role: str,
    must_overlay: bool,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], bool] | None:
    for (r1, c1), (r2, c2) in _aligned_cell_pairs(len(row_bands)):
        regular1, overlay1 = _placement_candidates(grid, row_bands[r1], col_bands[c1], occupied_endpoints, border_mask)
        regular2, overlay2 = _placement_candidates(grid, row_bands[r2], col_bands[c2], occupied_endpoints, border_mask)
        if must_overlay:
            pair_sources = [(overlay1, regular2), (regular1, overlay2)]
        else:
            pair_sources = [(regular1, regular2)]

        for left_pool, right_pool in pair_sources:
            if not left_pool or not right_pool:
                continue
            for p1 in left_pool:
                for p2 in right_pool:
                    if p1 == p2:
                        continue
                    choice = _choose_path(p1, p2, blocked, allow_turns=True)
                    if choice is None:
                        continue
                    mode, coords = choice
                    if mode != role:
                        continue
                    if role == "straight" and not (p1[0] == p2[0] or p1[1] == p2[1]):
                        continue
                    if role in {"hv", "vh"} and (p1[0] == p2[0] or p1[1] == p2[1]):
                        continue
                    inner = coords[1:-1]
                    if any(coord in occupied_endpoints for coord in inner):
                        continue
                    if any(grid[r, c] not in (BLACK, GREY) for r, c in inner):
                        continue
                    fill_coords = [coord for coord in coords if grid[coord] == BLACK]
                    if not fill_coords:
                        continue
                    if any(coord in reserved_fill for coord in fill_coords):
                        continue
                    used_overlay = grid[p1] == ORANGE or grid[p2] == ORANGE
                    if must_overlay and not used_overlay:
                        continue
                    return p1, p2, fill_coords, used_overlay
    return None


def _solve_without_turns(grid: np.ndarray) -> np.ndarray:
    return _render_solution(grid, use_bbox_obstacles=True, allow_turns=False)


def _solve_with_visible_orange_only(grid: np.ndarray) -> np.ndarray:
    return _render_solution(grid, use_bbox_obstacles=False, allow_turns=True)


def _validate_generated_case(grid: np.ndarray, meta: dict) -> None:
    row_seps, col_seps, row_bands, col_bands = _grid_bands(grid)
    assert row_seps == meta["row_seps"]
    assert col_seps == meta["col_seps"]

    visible_bboxes = _obstacle_bboxes_from_visible_orange(grid)
    assert visible_bboxes == meta["bboxes"], "visible orange components must preserve each obstacle bbox"

    pairs = _endpoint_pairs(grid)
    assert sorted(pairs) == sorted(spec["color"] for spec in meta["pairs"])
    blocked = _mask_from_bboxes(grid.shape, visible_bboxes)
    turn_count = 0
    for spec in meta["pairs"]:
        color = spec["color"]
        p1, p2 = pairs[color]
        expected = {spec["p1"], spec["p2"]}
        assert {p1, p2} == expected
        same_row_band = _band_index(p1[0], row_bands) == _band_index(p2[0], row_bands)
        same_col_band = _band_index(p1[1], col_bands) == _band_index(p2[1], col_bands)
        assert same_row_band or same_col_band
        mode, coords = _choose_path(p1, p2, blocked, allow_turns=True)  # type: ignore[misc]
        assert mode == spec["role"]
        assert any(grid[r, c] == BLACK for r, c in coords)
        if mode != "straight":
            turn_count += 1

    oracle = _render_solution(grid, use_bbox_obstacles=True, allow_turns=True)
    assert np.array_equal(main(grid), oracle)
    if meta["requires_turn"]:
        assert turn_count >= 1
        assert not np.array_equal(_solve_without_turns(grid), oracle)
    if meta["requires_hidden_bbox"]:
        assert any(spec["overlay"] for spec in meta["pairs"])
        assert not np.array_equal(_solve_with_visible_orange_only(grid), oracle)


def _generate_case() -> tuple[np.ndarray, dict] | None:
    n_cells = random.choice([4, 5])
    cell_h = random.choice([3, 4])
    cell_w = random.choice([3, 4])
    grid, row_seps, col_seps, row_bands, col_bands = _build_lattice(n_cells, cell_h, cell_w)

    bboxes = _sample_obstacles(grid, row_seps, col_seps)
    if bboxes is None:
        return None
    blocked = _mask_from_bboxes(grid.shape, bboxes)
    border_mask = _obstacle_border_mask(grid.shape, bboxes)

    pair_count = random.choice([2, 2, 3])
    requires_hidden_bbox = random.random() < 0.6
    requires_turn = random.random() < 0.75

    pair_specs: list[dict] = []
    if requires_hidden_bbox:
        pair_specs.append({"role": "straight", "must_overlay": True})
    if requires_turn:
        pair_specs.append({"role": random.choice(["hv", "vh"]), "must_overlay": False})
    while len(pair_specs) < pair_count:
        pair_specs.append({"role": random.choice(["straight", "hv", "vh"]), "must_overlay": False})
    pair_specs = pair_specs[:pair_count]

    colors = random.sample(ENDPOINT_COLORS, len(pair_specs))
    occupied_endpoints: set[tuple[int, int]] = set()
    reserved_fill: set[tuple[int, int]] = set()
    placed_pairs: list[dict] = []

    for color, spec in zip(colors, pair_specs):
        found = _find_pair_placement(
            grid,
            row_bands,
            col_bands,
            blocked,
            border_mask,
            occupied_endpoints,
            reserved_fill,
            role=spec["role"],
            must_overlay=spec["must_overlay"],
        )
        if found is None:
            return None
        p1, p2, fill_coords, overlay = found
        grid[p1] = int(color)
        grid[p2] = int(color)
        occupied_endpoints.update([p1, p2])
        reserved_fill.update(fill_coords)
        placed_pairs.append(
            {
                "color": int(color),
                "p1": p1,
                "p2": p2,
                "role": spec["role"],
                "overlay": bool(overlay),
            }
        )

    meta = {
        "row_seps": row_seps,
        "col_seps": col_seps,
        "bboxes": bboxes,
        "pairs": placed_pairs,
        "requires_hidden_bbox": requires_hidden_bbox,
        "requires_turn": requires_turn,
    }

    try:
        _validate_generated_case(grid, meta)
    except AssertionError:
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        generated = _generate_case()
        if generated is not None:
            grid, _meta = generated
            return grid
    raise RuntimeError("failed to generate a valid obstacle-rerouting ARC input")
