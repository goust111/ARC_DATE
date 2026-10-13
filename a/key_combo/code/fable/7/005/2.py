from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray, *, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        dirs = DIRS8
    else:
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _depth_map(side: int) -> np.ndarray:
    rr, cc = np.indices((side, side))
    return np.minimum.reduce([rr, cc, side - 1 - rr, side - 1 - cc])


def _extract_legend_sequence(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    side = patch.shape[0]
    assert patch.shape[0] == patch.shape[1] and side >= 3, "legend must be a square"
    assert not np.any(patch == BLACK), "legend must fill its square bbox"

    depth = _depth_map(side)
    n_layers = int(depth.max()) + 1
    sequence: list[int] = []
    used: set[int] = set()
    for d in range(n_layers):
        vals = np.unique(patch[depth == d])
        assert len(vals) == 1, "each legend layer must have one color"
        color = int(vals[0])
        assert color != BLACK and color not in used, "legend layer colors must be distinct"
        sequence.append(color)
        used.add(color)
    return sequence


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert _get_bg(g) == BLACK, "expected a black canvas"

    comps = _find_components(g != BLACK, connectivity=8)
    big = [coords for coords in comps if len(coords) > 1]
    singletons = [coords for coords in comps if len(coords) == 1]
    assert len(big) == 1, "expected exactly one nontrivial legend component"
    assert len(singletons) == 2 and len(comps) == 3, "expected exactly two isolated seeds"

    legend_coords = big[0]
    legend_bbox = _bbox_coords(legend_coords)
    r0, c0, r1, c1 = legend_bbox
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1, "legend bbox must be square"
    assert len(legend_coords) == side * side, "legend must occupy its full square bbox"
    sequence = _extract_legend_sequence(g, legend_bbox)
    assert len(sequence) >= 2, "outer and inner legend colors must differ"

    seed_specs = []
    for coords in singletons:
        r, c = map(int, coords[0])
        seed_specs.append(((r, c), int(g[r, c])))

    outer_candidates = [pos for pos, color in seed_specs if color == sequence[0]]
    inner_candidates = [pos for pos, color in seed_specs if color == sequence[-1]]
    assert len(outer_candidates) == 1 and len(inner_candidates) == 1, "seed colors must match outermost and innermost layers"
    outer_seed = outer_candidates[0]
    inner_seed = inner_candidates[0]
    assert outer_seed != inner_seed

    return {
        "sequence": sequence,
        "legend_bbox": legend_bbox,
        "outer_seed": outer_seed,
        "inner_seed": inner_seed,
    }


def _distance_field(shape: tuple[int, int], seed: tuple[int, int], *, metric: str = "chebyshev") -> np.ndarray:
    rr, cc = np.indices(shape)
    sr, sc = seed
    dr = np.abs(rr - int(sr))
    dc = np.abs(cc - int(sc))
    if metric == "chebyshev":
        return np.maximum(dr, dc)
    if metric == "manhattan":
        return dr + dc
    raise ValueError(f"unknown metric: {metric}")


def _render_competition(
    shape: tuple[int, int],
    outer_seed: tuple[int, int],
    inner_seed: tuple[int, int],
    sequence: list[int],
    *,
    metric: str = "chebyshev",
    keep_ties_black: bool = True,
    reverse_inner: bool = True,
) -> np.ndarray:
    seq_outer = np.asarray(sequence, dtype=int)
    seq_inner = seq_outer[::-1] if reverse_inner else seq_outer
    period = len(sequence)

    d_outer = _distance_field(shape, outer_seed, metric=metric)
    d_inner = _distance_field(shape, inner_seed, metric=metric)
    out = np.zeros(shape, dtype=int)

    outer_mask = d_outer < d_inner
    inner_mask = d_inner < d_outer
    tie_mask = ~(outer_mask | inner_mask)

    out[outer_mask] = seq_outer[d_outer[outer_mask] % period]
    out[inner_mask] = seq_inner[d_inner[inner_mask] % period]
    if not keep_ties_black:
        out[tie_mask] = seq_outer[d_outer[tie_mask] % period]
    return out


def _paint_legend(grid: np.ndarray, top: int, left: int, sequence: list[int], even_side: bool) -> tuple[int, int, int, int]:
    layers = len(sequence)
    side = 2 * layers - 1 + int(even_side)
    depth = _depth_map(side)
    patch = np.zeros((side, side), dtype=int)
    for d, color in enumerate(sequence):
        patch[depth == d] = int(color)
    grid[top : top + side, left : left + side] = patch
    return int(top), int(left), int(top + side - 1), int(left + side - 1)


def _isolated_black_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h, r + 2)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 2)
            if not np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
                cells.append((r, c))
    return cells


def _mode_ok(a: tuple[int, int], b: tuple[int, int], mode: str) -> bool:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if max(dr, dc) < 3:
        return False
    if mode == "row":
        return dr == 0
    if mode == "col":
        return dc == 0
    if mode == "diag":
        return dr == dc
    if mode == "general":
        return dr != 0 and dc != 0 and dr != dc
    raise ValueError(mode)


def _place_seeds(grid: np.ndarray, outer_color: int, inner_color: int) -> tuple[tuple[int, int], tuple[int, int]]:
    modes = ["row", "col", "diag", "general"]
    for _ in range(200):
        mode = random.choice(modes)
        first_choices = _isolated_black_cells(grid)
        if len(first_choices) < 2:
            break
        outer_seed = random.choice(first_choices)
        grid[outer_seed] = outer_color
        second_choices = [cell for cell in _isolated_black_cells(grid) if _mode_ok(outer_seed, cell, mode)]
        if second_choices:
            inner_seed = random.choice(second_choices)
            grid[inner_seed] = inner_color
            return outer_seed, inner_seed
        grid[outer_seed] = BLACK
    raise RuntimeError("failed to place seeds")


def _build_case() -> tuple[np.ndarray, dict]:
    for _ in range(500):
        h = random.randint(12, 30)
        w = random.randint(12, 30)
        max_layers = min(6, h // 2, w // 2)
        if max_layers < 2:
            continue

        layers = random.randint(2, max_layers)
        even_side = bool(random.getrandbits(1))
        side = 2 * layers - 1 + int(even_side)
        if side > h or side > w:
            continue

        sequence = random.sample(PALETTE, layers)
        grid = np.zeros((h, w), dtype=int)
        top = random.randint(0, h - side)
        left = random.randint(0, w - side)
        legend_bbox = _paint_legend(grid, top, left, sequence, even_side)
        try:
            outer_seed, inner_seed = _place_seeds(grid, sequence[0], sequence[-1])
        except RuntimeError:
            continue

        spec = {
            "sequence": tuple(sequence),
            "legend_bbox": legend_bbox,
            "outer_seed": outer_seed,
            "inner_seed": inner_seed,
        }
        if _validate_case(grid, spec):
            return grid, spec
    raise RuntimeError("failed to generate a valid puzzle instance")


def _validate_case(grid: np.ndarray, spec: dict) -> bool:
    try:
        parsed = _parse_puzzle(grid)
    except Exception:
        return False

    if tuple(parsed["sequence"]) != tuple(spec["sequence"]):
        return False
    if parsed["legend_bbox"] != spec["legend_bbox"]:
        return False
    if parsed["outer_seed"] != spec["outer_seed"] or parsed["inner_seed"] != spec["inner_seed"]:
        return False

    expected = _render_competition(
        grid.shape,
        spec["outer_seed"],
        spec["inner_seed"],
        list(spec["sequence"]),
    )
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    d_outer = _distance_field(grid.shape, spec["outer_seed"], metric="chebyshev")
    d_inner = _distance_field(grid.shape, spec["inner_seed"], metric="chebyshev")
    if not np.any(d_outer < d_inner):
        return False
    if not np.any(d_inner < d_outer):
        return False
    if not np.any(d_outer == d_inner):
        return False

    if int(max(d_outer.max(), d_inner.max())) < len(spec["sequence"]):
        return False

    wrong_same_direction = _render_competition(
        grid.shape,
        spec["outer_seed"],
        spec["inner_seed"],
        list(spec["sequence"]),
        reverse_inner=False,
    )
    wrong_manhattan = _render_competition(
        grid.shape,
        spec["outer_seed"],
        spec["inner_seed"],
        list(spec["sequence"]),
        metric="manhattan",
    )
    wrong_no_ties = _render_competition(
        grid.shape,
        spec["outer_seed"],
        spec["inner_seed"],
        list(spec["sequence"]),
        keep_ties_black=False,
    )
    if np.array_equal(expected, wrong_same_direction):
        return False
    if np.array_equal(expected, wrong_manhattan):
        return False
    if np.array_equal(expected, wrong_no_ties):
        return False
    if np.array_equal(expected, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    grid, _spec = _build_case()
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the nested square legend as a bidirectional color program for two competing Chebyshev wavefronts.

    Input structure:
    - The canvas is black except for one filled square made of one-pixel concentric color layers and two isolated seed pixels.
    - The legend square occupies its whole bbox; each inward layer has a distinct color.
    - One seed matches the outermost legend color and the other matches the innermost legend color.

    Rule:
    - Read the legend colors from outer to inner as one cyclic sequence.
    - Remove the legend itself from consideration and evaluate every canvas cell only by its Chebyshev distances to the two seeds.
    - Cells strictly closer to the outer-color seed use the sequence in outer-to-inner order, indexed by that distance.
    - Cells strictly closer to the inner-color seed use the same sequence in reverse order, indexed by that distance.
    - Cells equidistant from both seeds stay black.

    Key insight:
    - The concentric square is not an object to move or copy; it is a color program executed from opposite ends by two seeds, and the black crack is the equal-distance frontier between their wavefronts.
    """
    parsed = _parse_puzzle(np.asarray(grid, dtype=int))
    return _render_competition(
        tuple(map(int, grid.shape)),
        parsed["outer_seed"],
        parsed["inner_seed"],
        parsed["sequence"],
    )
