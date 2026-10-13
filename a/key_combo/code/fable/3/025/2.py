from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(cells, dtype=int))
    return out


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    holes = 0
    for sr in range(h):
        for sc in range(w):
            if not empty[sr, sc] or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            stack = [(sr, sc)]
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _edge_markers(g: np.ndarray) -> list[tuple[int, int, int, tuple[int, int], str]]:
    h, w = g.shape
    markers: list[tuple[int, int, int, tuple[int, int], str]] = []
    boundary = []
    boundary.extend((0, c) for c in range(w))
    boundary.extend((h - 1, c) for c in range(w))
    boundary.extend((r, 0) for r in range(1, h - 1))
    boundary.extend((r, w - 1) for r in range(1, h - 1))
    for r, c in boundary:
        color = int(g[r, c])
        if color in (BG, GREY):
            continue
        on_vertical_edge = c == 0 or c == w - 1
        on_horizontal_edge = r == 0 or r == h - 1
        assert on_vertical_edge ^ on_horizontal_edge, "corner markers have ambiguous direction"
        if r == 0:
            markers.append((r, c, color, (-1, 0), "col"))
        elif r == h - 1:
            markers.append((r, c, color, (1, 0), "col"))
        elif c == 0:
            markers.append((r, c, color, (0, -1), "row"))
        else:
            markers.append((r, c, color, (0, 1), "row"))
    return markers


def _paired_moves(g: np.ndarray):
    comps = _components(g == GREY)
    row_sets = [set(map(int, coords[:, 0])) for coords in comps]
    col_sets = [set(map(int, coords[:, 1])) for coords in comps]
    moves = []
    used_components: set[int] = set()

    for mr, mc, color, direction, axis in _edge_markers(g):
        if axis == "col":
            candidates = [i for i, cols in enumerate(col_sets) if mc in cols]
        else:
            candidates = [i for i, rows in enumerate(row_sets) if mr in rows]
        assert len(candidates) == 1, "each boundary marker line must cross exactly one grey sprite"
        idx = candidates[0]
        assert idx not in used_components, "a grey sprite must be locked by at most one marker"
        used_components.add(idx)

        coords = comps[idx]
        mask = _component_mask(coords)
        k = _hole_count(mask)
        moves.append((idx, coords, int(k), int(color), direction, (mr, mc)))
    return comps, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each marked grey sprite toward its boundary marker by its own hole count.

    Input structure:
    - A black background contains separated grey (5) sprites, each with zero or more enclosed background holes.
    - Non-grey colored marker pixels sit on the outer border, never in the corners.
    - Each marker's row or column crosses exactly one grey sprite; some grey sprites cross no marker line.

    Rule:
    - Pair every boundary marker with the unique grey sprite crossed by that marker's row or column.
    - Count that sprite's enclosed holes inside its tight bounding box.
    - Erase all grey sprites, then translate each paired sprite toward its marker by that hole count and recolor it to the marker color.
    - Keep the marker pixels fixed on the border; unpaired grey distractor sprites disappear.

    Key insight:
    - The direction comes only from the boundary marker, while the distance comes only from the sprite topology, so a marked zero-hole sprite recolors in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    h, w = g.shape
    comps, moves = _paired_moves(g)
    assert moves, "expected at least one boundary marker"

    out = g.copy()
    for coords in comps:
        out[coords[:, 0], coords[:, 1]] = BG

    target_mask = np.zeros(g.shape, dtype=bool)
    marker_cells = {(r, c) for r, c, _color, _direction, _axis in _edge_markers(g)}
    for _idx, coords, k, color, direction, _marker in moves:
        dr, dc = direction
        rr = coords[:, 0] + dr * k
        cc = coords[:, 1] + dc * k
        assert np.all((0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)), "translated sprite must fit"
        shifted = list(zip(map(int, rr), map(int, cc)))
        assert not any(cell in marker_cells for cell in shifted), "translated sprite must not overwrite markers"
        assert not np.any(target_mask[rr, cc]), "translated sprites must not overlap"
        target_mask[rr, cc] = True
        out[rr, cc] = color
    return out


def _sprite_with_holes(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        h = rng.randint(2, 5)
        w = rng.randint(2, 6)
        return np.ones((h, w), dtype=bool)

    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    h = top + 1 + bottom
    w = left + (2 * hole_count - 1) + right
    mask = np.ones((h, w), dtype=bool)
    hole_row = top
    for i in range(hole_count):
        mask[hole_row, left + 2 * i] = False
    if rng.random() < 0.5:
        mask = mask.T
    return mask


def _can_place(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = occ.shape
    mh, mw = mask.shape
    if r0 < 1 or c0 < 1 or r0 + mh > h - 1 or c0 + mw > w - 1:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + mh + pad)
    cc1 = min(w, c0 + mw + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_specs(h: int, w: int, specs: list[dict], rng: random.Random) -> bool:
    occ = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: -int(specs[i]["mask"].sum()))
    for idx in order:
        mask = specs[idx]["mask"]
        mh, mw = mask.shape
        placed = False
        for _ in range(250):
            r0 = rng.randint(1, h - mh - 1)
            c0 = rng.randint(1, w - mw - 1)
            if not _can_place(occ, mask, r0, c0):
                continue
            coords = np.argwhere(mask) + np.array([r0, c0])
            occ[coords[:, 0], coords[:, 1]] = True
            specs[idx]["coords"] = coords.astype(int)
            specs[idx]["bbox"] = _bbox(coords)
            placed = True
            break
        if not placed:
            return False
    return True


def _marker_candidates(
    spec: dict,
    specs: list[dict],
    marker_cells: set[tuple[int, int]],
    target_mask: np.ndarray,
) -> list[tuple[int, int, tuple[int, int], str]]:
    h, w = target_mask.shape
    idx = int(spec["index"])
    coords = spec["coords"]
    hole_count = int(spec["holes"])
    own = np.zeros((h, w), dtype=bool)
    own[coords[:, 0], coords[:, 1]] = True
    other_grey = np.zeros((h, w), dtype=bool)
    for other in specs:
        if int(other["index"]) == idx:
            continue
        oc = other["coords"]
        other_grey[oc[:, 0], oc[:, 1]] = True

    row_sets = [set(map(int, s["coords"][:, 0])) for s in specs]
    col_sets = [set(map(int, s["coords"][:, 1])) for s in specs]
    rows = sorted(row_sets[idx])
    cols = sorted(col_sets[idx])
    choices: list[tuple[int, int, tuple[int, int], str]] = []
    edges = [
        ("top", (-1, 0), "col", cols),
        ("bottom", (1, 0), "col", cols),
        ("left", (0, -1), "row", rows),
        ("right", (0, 1), "row", rows),
    ]
    for edge, direction, axis, values in edges:
        dr, dc = direction
        shifted_r = coords[:, 0] + dr * hole_count
        shifted_c = coords[:, 1] + dc * hole_count
        if not np.all((1 <= shifted_r) & (shifted_r < h - 1) & (1 <= shifted_c) & (shifted_c < w - 1)):
            continue
        if np.any(target_mask[shifted_r, shifted_c]):
            continue
        if np.any(other_grey[shifted_r, shifted_c]):
            continue
        if edge == "top":
            marker_line = [(0, int(c)) for c in values]
        elif edge == "bottom":
            marker_line = [(h - 1, int(c)) for c in values]
        elif edge == "left":
            marker_line = [(int(r), 0) for r in values]
        else:
            marker_line = [(int(r), w - 1) for r in values]
        for value, marker in zip(values, marker_line):
            if marker in marker_cells:
                continue
            if axis == "col":
                if any(j != idx and value in col_sets[j] for j in range(len(specs))):
                    continue
            else:
                if any(j != idx and value in row_sets[j] for j in range(len(specs))):
                    continue
            choices.append((marker[0], marker[1], direction, axis))
    return choices


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        coords = spec["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for spec in specs:
        if not spec.get("locked", False):
            continue
        coords = spec["coords"]
        dr, dc = spec["direction"]
        k = int(spec["holes"])
        rr = coords[:, 0] + dr * k
        cc = coords[:, 1] + dc * k
        out[rr, cc] = int(spec["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    locked_count = rng.randint(2, 4)
    distractor_count = rng.randint(1, 3)

    locked_holes = [0, rng.randint(1, 3)]
    while len(locked_holes) < locked_count:
        locked_holes.append(rng.randint(0, 4))
    rng.shuffle(locked_holes)

    specs: list[dict] = []
    for i, holes in enumerate(locked_holes):
        specs.append(
            {
                "index": i,
                "locked": True,
                "holes": int(holes),
                "mask": _sprite_with_holes(int(holes), rng),
            }
        )
    for _ in range(distractor_count):
        holes = rng.randint(0, 4)
        specs.append(
            {
                "index": len(specs),
                "locked": False,
                "holes": int(holes),
                "mask": _sprite_with_holes(int(holes), rng),
            }
        )

    if not _place_specs(h, w, specs, rng):
        return None

    marker_cells: set[tuple[int, int]] = set()
    target_mask = np.zeros((h, w), dtype=bool)
    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    used_dirs: set[tuple[int, int]] = set()

    for spec in [s for s in specs if s["locked"]]:
        candidates = _marker_candidates(spec, specs, marker_cells, target_mask)
        rng.shuffle(candidates)
        if not candidates:
            return None
        mr, mc, direction, axis = candidates[0]
        spec["marker"] = (mr, mc)
        spec["direction"] = direction
        spec["axis"] = axis
        spec["color"] = colors.pop()
        marker_cells.add((mr, mc))
        used_dirs.add(direction)
        coords = spec["coords"]
        rr = coords[:, 0] + direction[0] * int(spec["holes"])
        cc = coords[:, 1] + direction[1] * int(spec["holes"])
        target_mask[rr, cc] = True

    if len(used_dirs) < 2:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        coords = spec["coords"]
        grid[coords[:, 0], coords[:, 1]] = GREY
    for spec in specs:
        if spec.get("locked", False):
            mr, mc = spec["marker"]
            grid[mr, mc] = int(spec["color"])

    comps, moves = _paired_moves(grid)
    if len(comps) != len(specs) or len(moves) != locked_count:
        return None

    def coord_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((int(r), int(c)) for r, c in coords))

    spec_by_coords = {coord_key(spec["coords"]): spec for spec in specs}
    if len(spec_by_coords) != len(specs):
        return None

    seen_zero = False
    seen_positive = False
    paired_spec_indexes: set[int] = set()
    for _idx, coords, k, _color, direction, _marker in moves:
        spec = spec_by_coords.get(coord_key(coords))
        if spec is None:
            return None
        paired_spec_indexes.add(int(spec["index"]))
        if not spec["locked"] or k != int(spec["holes"]) or direction != spec["direction"]:
            return None
        if k == 0:
            seen_zero = True
        else:
            seen_positive = True

    if not seen_zero or not seen_positive:
        return None
    if all(int(spec["index"]) in paired_spec_indexes for spec in specs):
        return None

    expected = _expected_from_specs(grid, specs)
    for spec in specs:
        if spec["locked"]:
            continue
        coords = spec["coords"]
        if np.any(expected[coords[:, 0], coords[:, 1]] != BG):
            return None
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-pull input")
