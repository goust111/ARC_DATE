from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
OBJECT = 2
LINE = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400
SHAPE_CHAR = "#"


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze_components(grid: np.ndarray, *, enforce_input_contract: bool) -> tuple[int, int, list[dict], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(_most_common_color(g))
    colors = sorted(set(map(int, np.unique(g))))
    fg_colors = [color for color in colors if color != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    comps = []
    for coords in _components(g == fg):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "bbox center must be integral"
        comps.append(
            {
                "coords": coords,
                "bbox": box,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "area": int(coords.shape[0]),
            }
        )

    assert comps, "expected at least one foreground object"

    rows = sorted({comp["center"][0] for comp in comps})
    cols = sorted({comp["center"][1] for comp in comps})

    if enforce_input_contract:
        assert bg == BACKGROUND, "background must be black"
        assert fg == OBJECT, "foreground must be red"
        assert LINE not in colors, "grid-line color must be absent from the input"
        assert 6 <= len(comps) <= 9, "expected 6-9 foreground objects"
        assert len(rows) == 3 and len(cols) == 3, "expected exactly three unique center rows and columns"
        for comp in comps:
            assert 6 <= comp["area"] <= 12, "generator must stay in the intended area range"

    return bg, fg, comps, rows, cols


def _erode_coords(coords: np.ndarray) -> np.ndarray:
    cells = {tuple(map(int, rc)) for rc in coords}
    kept = [
        (r, c)
        for r, c in cells
        if all((nr, nc) in cells for nr, nc in _neighbors4(r, c))
    ]
    assert kept, "shrink would erase the object"
    return np.array(sorted(kept), dtype=int)


def _dilate_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    expanded = {tuple(map(int, rc)) for rc in coords}
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return np.array(sorted(expanded), dtype=int)


def _paint_coords(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def _transform_objects(
    grid: np.ndarray,
    *,
    draw_lines: bool,
    op_overrides: dict[int, str] | None = None,
    enforce_input_contract: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, fg, comps, rows, cols = _analyze_components(g, enforce_input_contract=enforce_input_contract)
    del bg
    assert len(rows) == 3 and len(cols) == 3, "the puzzle expects exactly three center rows and columns"

    row_rank = {value: idx for idx, value in enumerate(rows)}
    col_rank = {value: idx for idx, value in enumerate(cols)}

    out = np.full_like(g, BACKGROUND)
    occupied = np.zeros(g.shape, dtype=bool)
    overrides = {} if op_overrides is None else dict(op_overrides)

    for idx, comp in enumerate(comps):
        cr, cc = comp["center"]
        parity = (row_rank[cr] + col_rank[cc]) % 2
        op = overrides.get(idx)
        if op is None:
            op = "shrink" if parity == 0 else "dilate"
        if op == "shrink":
            transformed = _erode_coords(comp["coords"])
        else:
            assert op == "dilate"
            transformed = _dilate_coords(comp["coords"], g.shape)
        assert not np.any(occupied[transformed[:, 0], transformed[:, 1]]), "generated objects must not overlap"
        occupied[transformed[:, 0], transformed[:, 1]] = True
        _paint_coords(out, transformed, fg)

    if draw_lines:
        for r in rows:
            out[int(r), :] = LINE
        for c in cols:
            out[:, int(c)] = LINE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw object-derived gray grid lines, then shrink or dilate each red object according to its center-rank parity.

    Input structure:
    - A black background contains 6-9 red 4-connected solid objects.
    - Each object's bounding-box center is integral, and the set of object centers uses exactly three distinct rows and three distinct columns.
    - Objects are well separated so one-layer shrinking stays nonempty and one-layer dilation stays unambiguous.

    Rule:
    - Find every red connected component and compute its bounding-box center.
    - Sort the distinct center rows and distinct center columns; draw full gray horizontal and vertical lines through those coordinates.
    - For each object, use the rank of its center row and center column within those sorted lists.
    - If row-rank + column-rank is even, shrink the object by one 4-neighborhood layer; if it is odd, dilate it by one 4-neighborhood layer.
    - Keep the red color for transformed objects, then let gray grid lines overwrite any red pixels that lie on the lines.

    Key insight:
    The objects do double duty: their own bounding-box centers create the output lattice, and the parity of each center's rank inside that lattice decides whether the same object contracts or expands.
    """
    return _transform_objects(
        np.asarray(grid, dtype=int),
        draw_lines=True,
        enforce_input_contract=False,
    )


def _shape_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    coords = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == SHAPE_CHAR]
    arr = np.array(coords, dtype=int)
    assert arr.size > 0
    return arr


def _normalize_shape(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, _normalize_shape(coords).tolist()))


def _shape_orientations(coords: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(coords, dtype=int)
    variants = []
    seen = set()
    for rot in range(4):
        cur = base.copy()
        for _ in range(rot):
            cur = np.stack([cur[:, 1], -cur[:, 0]], axis=1)
        for flip in (False, True):
            variant = cur.copy()
            if flip:
                variant[:, 1] *= -1
            variant = _normalize_shape(variant)
            key = _shape_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _shape_info(coords: np.ndarray) -> dict:
    norm = _normalize_shape(coords)
    box = _bbox(norm)
    h = box[2] - box[0] + 1
    w = box[3] - box[1] + 1
    center = (h // 2, w // 2)
    eroded = _erode_coords(norm)
    shrink_visible = any(int(r) != center[0] and int(c) != center[1] for r, c in eroded)
    return {
        "coords": norm,
        "h": h,
        "w": w,
        "center": center,
        "area": int(norm.shape[0]),
        "shrink_visible": shrink_visible,
    }


def _build_shape_library() -> list[dict]:
    base_shapes = [
        ("square3", ("###", "###", "###")),
        ("wide_bar", (".###.", "#####", ".###.")),
        ("tall_bar", (".#.", "###", "###", "###", ".#.")),
        ("blob_a", ("###..", "#####", ".##..", "..#..", "..#..")),
        ("blob_b", ("..##.", "#####", "..##.", "..##.", "..#..")),
    ]
    shapes = []
    for name, rows in base_shapes:
        for orient in _shape_orientations(_shape_from_rows(rows)):
            info = _shape_info(orient)
            assert 6 <= info["area"] <= 12
            info["name"] = name
            shapes.append(info)
    unique = []
    seen = set()
    for info in shapes:
        key = _shape_key(info["coords"])
        if key not in seen:
            seen.add(key)
            unique.append(info)
    return unique


SHAPE_LIBRARY = _build_shape_library()
VISIBLE_SHRINK_SHAPES = [shape for shape in SHAPE_LIBRARY if shape["shrink_visible"]]
OCCLUDED_SHRINK_SHAPES = [shape for shape in SHAPE_LIBRARY if not shape["shrink_visible"]]


def _place_shape(grid: np.ndarray, shape: dict, center_r: int, center_c: int) -> None:
    top = int(center_r) - int(shape["center"][0])
    left = int(center_c) - int(shape["center"][1])
    coords = shape["coords"].copy()
    coords[:, 0] += top
    coords[:, 1] += left
    assert coords[:, 0].min() >= 0 and coords[:, 1].min() >= 0
    assert coords[:, 0].max() < grid.shape[0] and coords[:, 1].max() < grid.shape[1]
    assert np.all(grid[coords[:, 0], coords[:, 1]] == BACKGROUND)
    grid[coords[:, 0], coords[:, 1]] = OBJECT


def _pick_centers(length: int, rng: np.random.Generator) -> list[int]:
    triples = []
    for a in range(4, length - 4):
        for b in range(a + 8, min(length - 4, a + 10) + 1):
            for c in range(b + 8, min(length - 4, b + 10) + 1):
                triples.append((a, b, c))
    assert triples
    choice = triples[int(rng.integers(0, len(triples)))]
    return list(map(int, choice))


def _pick_positions(rng: np.random.Generator) -> list[tuple[int, int]]:
    all_positions = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(200):
        count = int(rng.integers(6, 10))
        chosen_idx = rng.choice(len(all_positions), size=count, replace=False)
        chosen = [all_positions[int(i)] for i in chosen_idx]
        even = [pos for pos in chosen if (pos[0] + pos[1]) % 2 == 0]
        odd = [pos for pos in chosen if (pos[0] + pos[1]) % 2 == 1]
        if len(even) >= 2 and len(odd) >= 2:
            return sorted(chosen)
    raise RuntimeError("failed to choose occupied lattice cells")


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    row_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
    col_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
    if a[2] < b[0]:
        row_gap = b[0] - a[2] - 1
    elif b[2] < a[0]:
        row_gap = a[0] - b[2] - 1
    else:
        row_gap = 0
    if a[3] < b[1]:
        col_gap = b[1] - a[3] - 1
    elif b[3] < a[1]:
        col_gap = a[1] - b[3] - 1
    else:
        col_gap = 0
    return int(row_gap), int(col_gap)


def _validate_generator_sample(grid: np.ndarray) -> bool:
    try:
        bg, fg, comps, rows, cols = _analyze_components(grid, enforce_input_contract=True)
        assert bg == BACKGROUND and fg == OBJECT
        row_rank = {value: idx for idx, value in enumerate(rows)}
        col_rank = {value: idx for idx, value in enumerate(cols)}

        positions = {(row_rank[comp["center"][0]], col_rank[comp["center"][1]]) for comp in comps}
        assert len(positions) == len(comps), "at most one object per implicit lattice cell"

        even_count = sum((r + c) % 2 == 0 for r, c in positions)
        odd_count = len(positions) - even_count
        assert even_count >= 2 and odd_count >= 2

        line_hits = 0
        for comp in comps:
            cr, cc = comp["center"]
            coords = {tuple(map(int, rc)) for rc in comp["coords"]}
            assert any(r == cr for r, _c in coords)
            assert any(c == cc for _r, c in coords)
            eroded = _erode_coords(comp["coords"])
            dilated = _dilate_coords(comp["coords"], grid.shape)
            line_hits += sum(r == cr or c == cc for r, c in coords)
            other_rows = set(rows) - {cr}
            other_cols = set(cols) - {cc}
            assert not any(r in other_rows or c in other_cols for r, c in map(tuple, dilated.tolist()))
            assert eroded.shape[0] >= 1

        assert line_hits > 0

        for i, comp_a in enumerate(comps):
            bbox_a = comp_a["bbox"]
            dilated_a = {tuple(map(int, rc)) for rc in _dilate_coords(comp_a["coords"], grid.shape)}
            for j in range(i + 1, len(comps)):
                comp_b = comps[j]
                row_gap, col_gap = _bbox_gap(bbox_a, comp_b["bbox"])
                assert row_gap >= 3 or col_gap >= 3, "objects must have a clear bbox gap"
                dilated_b = {tuple(map(int, rc)) for rc in _dilate_coords(comp_b["coords"], grid.shape)}
                assert dilated_a.isdisjoint(dilated_b), "one-layer dilation must stay separate"
                for r, c in dilated_a:
                    assert all((nr, nc) not in dilated_b for nr, nc in _neighbors4(r, c)), "dilated objects must not touch"

        out = main(grid)
        no_lines = _transform_objects(grid, draw_lines=False, enforce_input_contract=False)
        all_shrink = _transform_objects(
            grid,
            draw_lines=True,
            op_overrides={idx: "shrink" for idx in range(len(comps))},
            enforce_input_contract=False,
        )
        all_dilate = _transform_objects(
            grid,
            draw_lines=True,
            op_overrides={idx: "dilate" for idx in range(len(comps))},
            enforce_input_contract=False,
        )
        lines_only = np.zeros_like(grid)
        for r in rows:
            lines_only[r, :] = LINE
        for c in cols:
            lines_only[:, c] = LINE

        assert not np.array_equal(out, no_lines)
        assert not np.array_equal(out, all_shrink)
        assert not np.array_equal(out, all_dilate)
        assert not np.array_equal(out, lines_only)

        visible_shrink = False
        visible_dilate = False
        for idx, comp in enumerate(comps):
            cr, cc = comp["center"]
            parity = (row_rank[cr] + col_rank[cc]) % 2
            alt_op = "dilate" if parity == 0 else "shrink"
            alt = _transform_objects(
                grid,
                draw_lines=True,
                op_overrides={idx: alt_op},
                enforce_input_contract=False,
            )
            assert not np.array_equal(out, alt), "every object's branch must matter"

            changed = np.argwhere(out != grid)
            assert changed.size > 0
            if parity == 0:
                eroded = {tuple(map(int, rc)) for rc in _erode_coords(comp["coords"])}
                visible_shrink |= any(r not in rows and c not in cols for r, c in eroded)
            else:
                comp_cells = {tuple(map(int, rc)) for rc in comp["coords"]}
                dilated = {tuple(map(int, rc)) for rc in _dilate_coords(comp["coords"], grid.shape)}
                visible_dilate |= any(
                    (r, c) not in comp_cells and r not in rows and c not in cols for r, c in dilated
                )

        assert visible_shrink, "at least one shrunken object should remain visible after line overwrite"
        assert visible_dilate, "at least one dilated object should add visible pixels"
        assert set(map(int, np.unique(out))) <= {BACKGROUND, OBJECT, LINE}
        return True
    except AssertionError:
        return False


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(24, 31))
    w = int(rng.integers(24, 31))
    row_centers = _pick_centers(h, rng)
    col_centers = _pick_centers(w, rng)
    positions = _pick_positions(rng)

    even_positions = [pos for pos in positions if (pos[0] + pos[1]) % 2 == 0]
    visible_even = even_positions[int(rng.integers(0, len(even_positions)))]
    occluded_even_choices = [pos for pos in even_positions if pos != visible_even]
    occluded_even = None
    if occluded_even_choices:
        occluded_even = occluded_even_choices[int(rng.integers(0, len(occluded_even_choices)))]

    assignments: dict[tuple[int, int], dict] = {}
    for pos in positions:
        if pos == visible_even:
            pool = VISIBLE_SHRINK_SHAPES
        elif occluded_even is not None and pos == occluded_even:
            pool = OCCLUDED_SHRINK_SHAPES
        else:
            pool = SHAPE_LIBRARY
        assignments[pos] = pool[int(rng.integers(0, len(pool)))]

    grid = np.zeros((h, w), dtype=int)
    for rr, cc in positions:
        shape = assignments[(rr, cc)]
        _place_shape(grid, shape, row_centers[rr], col_centers[cc])

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_candidate(rng)
        if _validate_generator_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
