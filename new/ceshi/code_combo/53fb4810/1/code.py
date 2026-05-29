# Combo variant of 53fb4810 (index 1)
# combo_spec: D1/0+D3/1
# Name: 对角线渐变射线

from __future__ import annotations

import numpy as np
from arc_utils import get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant D1/0 + D3/1
#
# a) Concrete behavior trace
#    Original 53fb4810 uses side-attached marker tiles and extends them along
#    one cardinal axis with an unchanged periodic pattern. The fused variant
#    changes both the path geometry and the repetition rule:
#    - The marker is now a fixed 2x2 corner tile missing the corner nearest the
#      '+' interior, so it visually sits in one of the four diagonal quadrants.
#    - Repetitions advance by one cell in both row and column, i.e. along a
#      45-degree ray.
#    - Each copied corner tile is recolored by one step of a global 3-color
#      cycle learned from the single long clue ray already present in the input.
#    Example for a SE ray with seed colors [A,B,C] on the three occupied cells:
#      step 0: {(0,1)=A,(1,0)=B,(1,1)=C}
#      step 1: shift (+1,+1), recolor via A->B, B->C, C->A
#      step 2: shift again, recolor again
#    Because the missing inner corner makes consecutive copies disjoint, the
#    result is a clean staircase-like diagonal band rather than an overpainted
#    blur.
#
# b) Complexity calibration
#    The diagonal cue must be obvious and the color-cycle cue must be learnable.
#    To avoid ambiguous weak clues, every grid contains exactly one long ray
#    whose seed uses all three cycle colors on the three occupied L-cells. Other
#    seeds may reuse a subset of those colors, but the clue ray always exposes
#    the full mapping in repeated form.
#
# c) Hidden assumptions audited against the original code
#    - Original direction inference assumed 4-neighbor side attachment. That
#      breaks here because the marker lives in a diagonal corner pocket, so the
#      solver must look at the four canonical corner boxes around each '+'.
#    - Original seed extraction used the connected component bbox directly. That
#      fails for the clue ray because the bbox spans the whole diagonal band; the
#      new solver reconstructs the seed from the '+' bbox and the chosen corner.
#    - Original extension assumed axis-aligned period by width/height. That is
#      no longer valid: the new copy index advances by (+/-1,+/-1) per step.
#    - The color-cycle rule adds a new global assumption: one elongated clue ray
#      exists and consecutive diagonal copies preserve shape while permuting
#      colors consistently. The generator guarantees this.
#
# d) Placement legality
#    The generator remains output-first. It places identical '+' anchors with a
#    one-cell border margin, reserves all four 2x2 corner boxes around each '+',
#    and rejects any ray that would hit an existing ray, a '+', or another
#    anchor's reserved corner boxes. This keeps every seed box interpretable.
#
# e) Visual observability
#    A human can read the rule from 2-4 examples because the clue ray shows both
#    control variables at once: it departs from a specific corner of the '+'
#    along a diagonal and its three visible seed colors rotate copy-by-copy.
#    The rule never depends on numeric color ids; only repeated visual
#    correspondences matter.
#
# f) Visual cleanliness
#    Consecutive copies do not overlap because the occupied seed pattern is the
#    3-cell L missing the inner corner. Rays are therefore crisp staircase bands.
#    Reserving all corner boxes prevents other rays from contaminating the
#    neighborhood that the solver uses to decide which diagonal is active.
#
# g) Change inferability
#    The generator uses exactly three cycle colors shared by all markers in one
#    grid. The long clue ray contains all three in its seed, so the solver never
#    has to guess an unseen mapping. A diagonal-only ablation and a static-color
#    ablation both differ from the combo output on generated samples, which is
#    checked after implementation.
# ---------------------------------------------------------------------------


_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-bg color with the most large connected components."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    if not color_map:
        return np.array(tile, dtype=int, copy=True)
    src = np.array(tile, dtype=int, copy=False)
    out = src.copy()
    for a, b in color_map.items():
        out[src == int(a)] = int(b)
    return out


def _inner_corner_local(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return 1, 1
    if (dr, dc) == (-1, 1):
        return 1, 0
    if (dr, dc) == (1, -1):
        return 0, 1
    if (dr, dc) == (1, 1):
        return 0, 0
    raise ValueError("direction must be diagonal")


def _occupied_offsets(direction: tuple[int, int]) -> list[tuple[int, int]]:
    mi, mj = _inner_corner_local(direction)
    return [(i, j) for i in range(2) for j in range(2) if (i, j) != (mi, mj)]


def _seed_bbox_from_plus(
    plus_coords: np.ndarray,
    direction: tuple[int, int],
) -> tuple[int, int, int, int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    if (dr, dc) == (-1, -1):
        return int(pr0 - 1), int(pc0 - 1), int(pr0), int(pc0)
    if (dr, dc) == (-1, 1):
        return int(pr0 - 1), int(pc1), int(pr0), int(pc1 + 1)
    if (dr, dc) == (1, -1):
        return int(pr1), int(pc0 - 1), int(pr1 + 1), int(pc0)
    if (dr, dc) == (1, 1):
        return int(pr1), int(pc1), int(pr1 + 1), int(pc1 + 1)
    raise ValueError("direction must be diagonal")


def _bbox_in_bounds(
    bbox: tuple[int, int, int, int],
    H: int,
    W: int,
) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W


def _copy_cells(
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    step: int,
) -> list[tuple[int, int, int, int]]:
    r0, c0, _r1, _c1 = map(int, seed_bbox)
    dr, dc = map(int, direction)
    return [
        (int(r0 + i + step * dr), int(c0 + j + step * dc), int(i), int(j))
        for i, j in _occupied_offsets(direction)
    ]


def _extract_visible_copies(
    grid: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    *,
    bg: int,
    plus_col: int,
) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    copies: list[list[tuple[int, int]]] = []
    step = 0
    while True:
        cells = _copy_cells(seed_bbox, direction, step)
        if any(not (0 <= r < H and 0 <= c < W) for r, c, _i, _j in cells):
            break
        vals = [int(grid[r, c]) for r, c, _i, _j in cells]
        if any(v == int(bg) or v == int(plus_col) for v in vals):
            break
        copies.append([(int(r), int(c)) for r, c, _i, _j in cells])
        step += 1
    return copies


def _infer_color_map_from_copies(
    grid: np.ndarray,
    copies: list[list[tuple[int, int]]],
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    evidence = 0
    for cells_a, cells_b in zip(copies, copies[1:]):
        for (ra, ca), (rb, cb) in zip(cells_a, cells_b):
            va = int(grid[ra, ca])
            vb = int(grid[rb, cb])
            evidence += 1
            prev = mapping.get(int(va))
            if prev is not None and int(prev) != int(vb):
                return None
            mapping[int(va)] = int(vb)
    if evidence == 0 or len(mapping) < 3:
        return None
    return mapping


def _find_marker_infos(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    pn: int,
    *,
    bg: int,
    plus_col: int,
) -> list[dict]:
    H, W = grid.shape
    infos: list[dict] = []
    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        candidates: list[dict] = []
        for direction in _DIAGONALS:
            seed_bbox = _seed_bbox_from_plus(plus_coords, direction)
            if not _bbox_in_bounds(seed_bbox, H, W):
                continue
            r0, c0, _r1, _c1 = seed_bbox
            mi, mj = _inner_corner_local(direction)
            if int(grid[r0 + mi, c0 + mj]) != int(bg):
                continue
            vals = [int(grid[r0 + i, c0 + j]) for i, j in _occupied_offsets(direction)]
            if any(v == int(bg) or v == int(plus_col) for v in vals):
                continue
            tile = np.full((2, 2), int(bg), dtype=int)
            for i, j in _occupied_offsets(direction):
                tile[i, j] = int(grid[r0 + i, c0 + j])
            copies = _extract_visible_copies(
                grid,
                seed_bbox,
                direction,
                bg=int(bg),
                plus_col=int(plus_col),
            )
            candidates.append(
                {
                    "plus_id": int(pk),
                    "direction": tuple(map(int, direction)),
                    "seed_bbox": tuple(map(int, seed_bbox)),
                    "tile": tile,
                    "copies": copies,
                }
            )

        if len(candidates) == 1:
            infos.append(candidates[0])
    return infos


def extend_diagonal_color_cycle(
    out: np.ndarray,
    seed_tile: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
    color_map: dict[int, int] | None = None,
) -> None:
    """Repeat the 3-cell corner tile along a diagonal, recoloring each copy."""
    protect = set() if protect is None else set(map(int, protect))
    color_map = {} if color_map is None else {int(k): int(v) for k, v in color_map.items()}
    H, W = out.shape
    cur = np.array(seed_tile, dtype=int, copy=True)
    step = 1

    while True:
        cur = _apply_color_map(cur, color_map)
        cells = _copy_cells(seed_bbox, direction, step)
        if any(not (0 <= r < H and 0 <= c < W) for r, c, _i, _j in cells):
            break
        for r, c, i, j in cells:
            v = int(cur[i, j])
            if v == int(bg):
                continue
            if int(out[r, c]) in protect:
                continue
            out[r, c] = int(v)
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend corner markers away from '+' anchors along diagonal rays, while cycling colors.

    Input structure:
    - The most common color is background.
    - One dominant non-background color forms several identical '+'-like anchors.
    - Each anchor has exactly one 3-cell L-shaped marker tucked into one corner of its bbox; exactly one marker is already expanded into a long diagonal staircase.

    Rule:
    - The active corner around the '+' determines which of the four 45-degree directions the ray follows.
    - The long example ray reveals a global 3-color permutation: each next diagonal copy keeps the same L-shape but recolors every occupied cell by one step of that cycle.
    - Recreate that diagonal, recolored repetition for every compact seed until the grid boundary, never overwriting '+' pixels.

    Key insight:
    The task is not just "copy this marker diagonally" and not just "cycle colors"; the single elongated clue jointly exposes both the corner-selected diagonal direction and the whole-tile recoloring rule that every other seed must follow.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    infos = _find_marker_infos(g, plus_lab, pn, bg=int(bg), plus_col=int(plus_col))
    if not infos:
        return out

    color_map: dict[int, int] = {}
    for info in sorted(infos, key=lambda x: len(x["copies"]), reverse=True):
        if len(info["copies"]) < 2:
            continue
        inferred = _infer_color_map_from_copies(g, info["copies"])
        if inferred:
            color_map = inferred
            break

    for info in infos:
        extend_diagonal_color_cycle(
            out,
            info["tile"],
            info["seed_bbox"],
            info["direction"],
            bg=int(bg),
            protect={int(plus_col)},
            color_map=color_map,
        )

    return out


# --------------- generator ---------------

_H_RANGE = (14, 29)
_W_RANGE = (14, 24)
# np.random.Generator.integers uses an exclusive upper bound, so this yields 2..4 anchors.
_N_PLUS_RANGE = (2, 5)
_MIN_EXTRA_COPIES = 3
_CLUE_MIN_EXTRA_COPIES = 5
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 600
_CYCLE_LEN = 3

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
]


def _build_seed_tile(
    direction: tuple[int, int],
    cycle_colors: list[int],
    bg: int,
    rng: np.random.Generator,
    *,
    force_clue: bool = False,
) -> np.ndarray:
    tile = np.full((2, 2), int(bg), dtype=int)
    occ = _occupied_offsets(direction)
    if force_clue:
        vals = list(map(int, cycle_colors))
        rng.shuffle(vals)
    else:
        n_colors = int(rng.integers(1, len(cycle_colors) + 1))
        chosen = list(map(int, rng.choice(cycle_colors, size=n_colors, replace=False)))
        vals = chosen + list(map(int, rng.choice(chosen, size=len(occ) - n_colors, replace=True)))
        rng.shuffle(vals)
    for (i, j), v in zip(occ, vals):
        tile[i, j] = int(v)
    return tile


def _project_diagonal_cycle(
    H: int,
    W: int,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile: np.ndarray,
    color_map: dict[int, int],
) -> tuple[list[tuple[int, int, int]], list[list[tuple[int, int]]]]:
    cells: list[tuple[int, int, int]] = []
    copies: list[list[tuple[int, int]]] = []
    cur = np.array(tile, dtype=int, copy=True)
    step = 0
    while True:
        copy_info = _copy_cells(seed_bbox, direction, step)
        if any(not (0 <= r < H and 0 <= c < W) for r, c, _i, _j in copy_info):
            break
        copy_cells: list[tuple[int, int]] = []
        for r, c, i, j in copy_info:
            cells.append((int(r), int(c), int(cur[i, j])))
            copy_cells.append((int(r), int(c)))
        copies.append(copy_cells)
        cur = _apply_color_map(cur, color_map)
        step += 1
    return cells, copies


def _all_corner_guard_cells(
    plus_bbox: tuple[int, int, int, int],
) -> set[tuple[int, int]]:
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    boxes = [
        (pr0 - 1, pc0 - 1, pr0, pc0),
        (pr0 - 1, pc1, pr0, pc1 + 1),
        (pr1, pc0 - 1, pr1 + 1, pc0),
        (pr1, pc1, pr1 + 1, pc1 + 1),
    ]
    cells: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in boxes:
        for r in range(int(r0), int(r1) + 1):
            for c in range(int(c0), int(c1) + 1):
                cells.add((int(r), int(c)))
    return cells


def _mark_plus_halo(plus_occ: np.ndarray, plus_cells: list[tuple[int, int]]) -> None:
    H, W = plus_occ.shape
    for r, c in plus_cells:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if 0 <= rr < H and 0 <= cc < W:
                    plus_occ[rr, cc] = 1


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    cycle_colors: list[int],
    color_map: dict[int, int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    guard: np.ndarray,
    out: np.ndarray,
    bg: int,
    plus_color: int,
    *,
    force_clue: bool = False,
    forced_direction: tuple[int, int] | None = None,
) -> dict | None:
    ph, pw = plus_shape.shape
    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]
        if any(int(occ[r, c]) != 0 or int(plus_occ[r, c]) != 0 for r, c in plus_cells):
            continue

        corner_guard = _all_corner_guard_cells(plus_bbox)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in corner_guard):
            continue
        if any(int(occ[r, c]) != 0 or int(guard[r, c]) != 0 for r, c in corner_guard):
            continue

        dirs = [tuple(map(int, forced_direction))] if forced_direction is not None else list(_DIAGONALS)
        if forced_direction is None:
            rng.shuffle(dirs)

        for direction in dirs:
            seed_bbox = _seed_bbox_from_plus(np.argwhere(plus_shape == 1) + np.array([r0, c0]), direction)
            tile = _build_seed_tile(
                direction,
                cycle_colors,
                int(bg),
                rng,
                force_clue=bool(force_clue),
            )
            cells, copies = _project_diagonal_cycle(H, W, seed_bbox, direction, tile, color_map)
            extra = int(len(copies) - 1)
            need = _CLUE_MIN_EXTRA_COPIES if force_clue else _MIN_EXTRA_COPIES
            if extra < need:
                continue
            if any(int(occ[r, c]) != 0 or int(guard[r, c]) != 0 for r, c, _v in cells):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
            _mark_plus_halo(plus_occ, plus_cells)
            for r, c in corner_guard:
                guard[r, c] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            return {
                "direction": tuple(map(int, direction)),
                "seed_bbox": tuple(map(int, seed_bbox)),
                "tile": np.array(tile, dtype=int, copy=True),
                "copies": copies,
                "extra_copies": int(extra),
            }
    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    keep_idx: int,
) -> np.ndarray:
    inp = np.array(out, dtype=int, copy=True)
    for i, info in enumerate(pluses):
        if int(i) == int(keep_idx):
            continue
        for copy_cells in info["copies"][1:]:
            for r, c in copy_cells:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [int(c) for c in range(10) if int(c) != int(bg)]
    plus_color = int(rng.choice(others))
    cycle_colors = list(map(int, rng.choice([c for c in others if c != plus_color], size=_CYCLE_LEN, replace=False)))
    rng.shuffle(cycle_colors)
    color_map = {
        int(cycle_colors[i]): int(cycle_colors[(i + 1) % _CYCLE_LEN])
        for i in range(_CYCLE_LEN)
    }

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)
    guard = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []

    clue = _place_plus_and_ray(
        rng,
        plus_shape,
        cycle_colors,
        color_map,
        H,
        W,
        occ,
        plus_occ,
        guard,
        out,
        bg,
        plus_color,
        force_clue=True,
        forced_direction=_DIAGONALS[int(rng.integers(0, len(_DIAGONALS)))],
    )
    if clue is None:
        return None
    pluses.append(clue)

    for _ in range(N - 1):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            cycle_colors,
            color_map,
            H,
            W,
            occ,
            plus_occ,
            guard,
            out,
            bg,
            plus_color,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, keep_idx=0)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
