from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
GREY = 5
SPRITE_COLORS = (2, 3, 4, 6)
QUADRANTS = ("TL", "TR", "BL", "BR")
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}

SYMMETRIC_MASKS = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

ASYMMETRIC_MASKS = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _horizontal_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def _quadrant_of(r: int, c: int, mid_r: int, mid_c: int) -> str:
    return ("T" if r < mid_r else "B") + ("L" if c < mid_c else "R")


def _quadrant_slices(shape: tuple[int, int], quadrant: str) -> tuple[slice, slice]:
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2
    r0 = 0 if quadrant[0] == "T" else mid_r
    r1 = mid_r if quadrant[0] == "T" else h
    c0 = 0 if quadrant[1] == "L" else mid_c
    c1 = mid_c if quadrant[1] == "L" else w
    return slice(r0, r1), slice(c0, c1)


def _quadrant_ranges(shape: tuple[int, int], quadrant: str) -> tuple[range, range]:
    rs, cs = _quadrant_slices(shape, quadrant)
    return range(rs.start, rs.stop), range(cs.start, cs.stop)


def _quadrant_relation(src: str, dst: str) -> str:
    if src == dst:
        return "same"
    if src[0] == dst[0]:
        return "E" if src[1] == "L" else "W"
    if src[1] == dst[1]:
        return "S" if src[0] == "T" else "N"
    raise ValueError(f"unreachable control quadrant: {src} -> {dst}")


def _reachable_targets(quadrant: str) -> list[str]:
    out = [quadrant]
    if quadrant[0] == "T":
        out.append("B" + quadrant[1])
    else:
        out.append("T" + quadrant[1])
    if quadrant[1] == "L":
        out.append(quadrant[0] + "R")
    else:
        out.append(quadrant[0] + "L")
    return out


def _bbox_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mid_r = h // 2
    mid_c = w // 2

    if not set(map(int, np.unique(g))).issubset({BACKGROUND, GREY, *SPRITE_COLORS}):
        raise AssertionError("unexpected colors")

    objects: list[dict] = []
    used_controls: set[tuple[int, int]] = set()
    for color in SPRITE_COLORS:
        for cells in _components(g == color):
            controls = {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREY
            }
            if len(controls) != 1:
                raise AssertionError("each sprite must have exactly one grey control")
            control = next(iter(controls))
            used_controls.add(control)
            mask = _crop_mask(cells)
            obj_quadrants = {_quadrant_of(r, c, mid_r, mid_c) for r, c in cells}
            if len(obj_quadrants) != 1:
                raise AssertionError("sprite crosses a global midline")
            objects.append(
                {
                    "color": int(color),
                    "cells": sorted(cells),
                    "bbox": _bbox(cells),
                    "control": control,
                    "mask": mask,
                    "symmetric": _horizontal_symmetric(mask),
                    "object_quadrant": next(iter(obj_quadrants)),
                    "control_quadrant": _quadrant_of(control[0], control[1], mid_r, mid_c),
                }
            )

    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    if grey_cells != used_controls:
        raise AssertionError("every grey pixel must be used as exactly one control")

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _solve_variant(
    grid: np.ndarray,
    *,
    use_control_quadrant: bool,
    protect_fixed: bool,
    clear_all_symmetric_first: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = _parse_scene(g)
    out = g.copy()
    fixed = np.zeros_like(g, dtype=bool)

    asymmetric = [obj for obj in objects if not obj["symmetric"]]
    symmetric = [obj for obj in objects if obj["symmetric"]]

    for obj in asymmetric:
        for r, c in obj["cells"]:
            out[r, c] = GREY
            fixed[r, c] = True
        cr, cc = obj["control"]
        out[cr, cc] = obj["color"]
        fixed[cr, cc] = True

    if clear_all_symmetric_first:
        for obj in symmetric:
            for r, c in obj["cells"]:
                out[r, c] = BACKGROUND
            cr, cc = obj["control"]
            out[cr, cc] = BACKGROUND

    for obj in symmetric:
        if not clear_all_symmetric_first:
            for r, c in obj["cells"]:
                out[r, c] = BACKGROUND
            cr, cc = obj["control"]
            out[cr, cc] = BACKGROUND

        target = obj["control_quadrant"] if use_control_quadrant else obj["object_quadrant"]
        rs, cs = _quadrant_slices(g.shape, target)
        patch = out[rs, cs]
        paintable = np.ones_like(patch, dtype=bool)
        if protect_fixed:
            paintable &= ~fixed[rs, cs]
        patch[paintable] = obj["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each sprite's horizontal symmetry and either fill the control-selected quadrant or swap the sprite color with its grey control.

    Input structure:
    - A black background contains several small 4-connected single-color sprites in distinct colors from {2,3,4,6}.
    - Each sprite has exactly one grey control pixel that is 4-neighbor adjacent to it.
    - Every sprite stays entirely on one side of the global row and column midlines, although its control may sit across one midline.

    Rule:
    - Split the grid into four quadrants using row midpoint floor(H/2) and column midpoint floor(W/2), with the midpoint row/column belonging to the bottom/right halves.
    - For each sprite, test whether its tight-bbox mask is mirror-symmetric across that bbox's horizontal midline.
    - First process every non-symmetric sprite by recoloring the whole sprite grey and recoloring its control pixel to the sprite's original color.
    - Then process the symmetric sprites in reading order of sprite bbox top-left corners: erase that sprite and its control, find the control pixel's global quadrant, and fill that quadrant with the sprite color without overwriting any already-finished non-symmetric swap pixels.

    Key insight:
    - The grey control pixel has two different meanings, and the symmetry test is the gate that decides whether grey acts as a swap partner or as a quadrant selector.
    """
    return _solve_variant(
        grid,
        use_control_quadrant=True,
        protect_fixed=True,
        clear_all_symmetric_first=False,
    )


def _random_symmetric_mask() -> np.ndarray:
    return np.fliplr(random.choice(SYMMETRIC_MASKS)).copy() if random.random() < 0.5 else random.choice(SYMMETRIC_MASKS).copy()


def _random_asymmetric_mask() -> np.ndarray:
    for _ in range(20):
        mask = random.choice(ASYMMETRIC_MASKS).copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if not _horizontal_symmetric(mask):
            return mask
    raise RuntimeError("failed to sample asymmetric mask")


def _candidate_placements(
    shape: tuple[int, int], mask: np.ndarray, object_quadrant: str, control_quadrant: str
) -> list[dict]:
    h, w = shape
    mh, mw = mask.shape
    row_range, col_range = _quadrant_ranges(shape, object_quadrant)
    rel = _quadrant_relation(object_quadrant, control_quadrant)
    pixels = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    candidates: list[dict] = []

    for r0 in range(row_range.start, row_range.stop - mh + 1):
        for c0 in range(col_range.start, col_range.stop - mw + 1):
            cells = {(r0 + r, c0 + c) for r, c in pixels}
            controls: set[tuple[int, int]] = set()
            for r, c in cells:
                for direction, (dr, dc) in DIRS.items():
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in cells or not _in_bounds(shape, nr, nc):
                        continue
                    if rel != "same" and direction != rel:
                        continue
                    if _quadrant_of(nr, nc, h // 2, w // 2) != control_quadrant:
                        continue
                    controls.add((nr, nc))
            for control in controls:
                pair_cells = set(cells)
                pair_cells.add(control)
                candidates.append(
                    {
                        "cells": tuple(sorted(cells)),
                        "control": control,
                        "bbox": _bbox(pair_cells),
                        "all_cells": pair_cells,
                    }
                )

    random.shuffle(candidates)
    return candidates


def _placement_compatible(candidate: dict, placed: list[dict]) -> bool:
    new_cells = candidate["all_cells"]
    for other in placed:
        if not _bbox_disjoint(candidate["bbox"], other["bbox"]):
            return False
        other_cells = other["all_cells"]
        if new_cells & other_cells:
            return False
        for r, c in new_cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in other_cells:
                    return False
    return True


def _place_specs(specs: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    indexed_candidates = [
        (idx, _candidate_placements(shape, spec["mask"], spec["object_quadrant"], spec["control_quadrant"]))
        for idx, spec in enumerate(specs)
    ]
    if any(not candidates for _, candidates in indexed_candidates):
        return None

    indexed_candidates.sort(key=lambda pair: len(pair[1]))
    chosen: dict[int, dict] = {}
    placed: list[dict] = []

    def backtrack(pos: int) -> bool:
        if pos == len(indexed_candidates):
            return True
        idx, candidates = indexed_candidates[pos]
        ordered = candidates[:]
        random.shuffle(ordered)
        for candidate in ordered:
            if not _placement_compatible(candidate, placed):
                continue
            placed.append(candidate)
            chosen[idx] = candidate
            if backtrack(pos + 1):
                return True
            placed.pop()
            del chosen[idx]
        return False

    if not backtrack(0):
        return None
    return [chosen[i] for i in range(len(specs))]


def _render_grid(shape: tuple[int, int], specs: list[dict], placements: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec, placement in zip(specs, placements):
        for r, c in placement["cells"]:
            grid[r, c] = spec["color"]
        cr, cc = placement["control"]
        grid[cr, cc] = GREY
    return grid


def _semantics_ok(grid: np.ndarray) -> bool:
    try:
        objects = _parse_scene(grid)
    except AssertionError:
        return False

    if len(objects) not in (3, 4):
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    if any(not 3 <= len(obj["cells"]) <= 6 for obj in objects):
        return False

    symmetric = [obj for obj in objects if obj["symmetric"]]
    asymmetric = [obj for obj in objects if not obj["symmetric"]]
    if len(symmetric) != 2 or len(asymmetric) < 1:
        return False

    sym_targets = [obj["control_quadrant"] for obj in symmetric]
    if len(set(sym_targets)) != len(sym_targets):
        return False
    if any(obj["object_quadrant"] == obj["control_quadrant"] for obj in symmetric):
        return False
    if not any(obj["object_quadrant"] in set(sym_targets) for obj in asymmetric):
        return False

    correct = main(grid)
    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, _solve_variant(grid, use_control_quadrant=False, protect_fixed=True, clear_all_symmetric_first=False)):
        return False
    if np.array_equal(correct, _solve_variant(grid, use_control_quadrant=True, protect_fixed=False, clear_all_symmetric_first=False)):
        return False
    if np.array_equal(correct, _solve_variant(grid, use_control_quadrant=True, protect_fixed=True, clear_all_symmetric_first=True)):
        return False
    return True


def _random_asym_target(quadrant: str) -> str:
    options = _reachable_targets(quadrant)
    weights = [4 if q == quadrant else 3 for q in options]
    return random.choices(options, weights=weights, k=1)[0]


def _make_specs(object_count: int) -> list[dict]:
    pair_kind = random.choice(("top_row", "bottom_row", "left_col", "right_col"))
    if pair_kind == "top_row":
        symmetric_layout = [("TL", "TR"), ("TR", "TL")]
    elif pair_kind == "bottom_row":
        symmetric_layout = [("BL", "BR"), ("BR", "BL")]
    elif pair_kind == "left_col":
        symmetric_layout = [("TL", "BL"), ("BL", "TL")]
    else:
        symmetric_layout = [("TR", "BR"), ("BR", "TR")]

    colors = random.sample(list(SPRITE_COLORS), object_count)
    specs = [
        {
            "kind": "sym",
            "mask": _random_symmetric_mask(),
            "color": colors[idx],
            "object_quadrant": src,
            "control_quadrant": dst,
        }
        for idx, (src, dst) in enumerate(symmetric_layout)
    ]

    fill_quadrants = [dst for _, dst in symmetric_layout]
    asym_quadrants = [random.choice(fill_quadrants)]
    if object_count == 4:
        remaining = [q for q in QUADRANTS if q not in asym_quadrants]
        asym_quadrants.append(random.choice(remaining))

    for idx, quadrant in enumerate(asym_quadrants, start=2):
        specs.append(
            {
                "kind": "asym",
                "mask": _random_asymmetric_mask(),
                "color": colors[idx],
                "object_quadrant": quadrant,
                "control_quadrant": _random_asym_target(quadrant),
            }
        )

    random.shuffle(specs[2:])
    return specs


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(14, 19), random.randint(14, 19))
    specs = _make_specs(random.choice((3, 4)))
    placements = _place_specs(specs, shape)
    if placements is None:
        return None
    grid = _render_grid(shape, specs, placements)
    if not _semantics_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
