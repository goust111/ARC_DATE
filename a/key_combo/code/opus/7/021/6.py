from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
OBJECT_COLOR = 9
MARKER_COLORS = (1, 2, 3, 4)
CELL_SIZE = 3
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _full_divider_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == int(color))]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == int(color))]
    return rows, cols


def _non_divider_runs(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    dividers = set(map(int, divider_indices))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(length)):
        if idx in dividers:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, int(length) - 1))
    return runs


def _region_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _classify_cell(region: np.ndarray, bg: int, divider: int) -> dict:
    fg_mask = (region != int(bg)) & (region != int(divider))
    count = int(fg_mask.sum())
    if count == 0:
        return {"kind": "empty"}

    colors = _colors_present(region[fg_mask], exclude=None)
    assert len(colors) == 1, "cell contents must be monochrome"
    color = int(colors[0])

    if count == 1:
        (rr, cc) = map(int, np.argwhere(fg_mask)[0])
        return {"kind": "marker", "color": color, "pixel": (rr, cc)}

    bbox = _region_bbox(fg_mask)
    assert bbox is not None
    r0, c0, r1, c1 = bbox
    sub = region[r0 : r1 + 1, c0 : c1 + 1]
    assert sub.shape == (3, 3), "objects must have a 3x3 bounding box"
    assert color == OBJECT_COLOR, "objects must use the shared source color"

    if count == 9 and np.all(sub == OBJECT_COLOR):
        return {"kind": "solid", "bbox": bbox}

    border_mask = np.ones((3, 3), dtype=bool)
    border_mask[1, 1] = False
    if count == 8 and np.all(sub[border_mask] == OBJECT_COLOR) and int(sub[1, 1]) == int(bg):
        return {"kind": "hollow", "bbox": bbox}

    raise AssertionError("unrecognized cell contents")


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    divider_candidates = []
    for color in _colors_present(g, exclude={bg}):
        row_divs, col_divs = _full_divider_indices(g, int(color))
        if row_divs or col_divs:
            divider_candidates.append(int(color))
    assert divider_candidates == [DIVIDER], "expected grey full divider lines"

    row_dividers, col_dividers = _full_divider_indices(g, DIVIDER)
    row_runs = _non_divider_runs(g.shape[0], row_dividers)
    col_runs = _non_divider_runs(g.shape[1], col_dividers)
    assert row_runs and col_runs
    assert len({r1 - r0 + 1 for r0, r1 in row_runs}) == 1, "cell heights must be uniform"
    assert len({c1 - c0 + 1 for c0, c1 in col_runs}) == 1, "cell widths must be uniform"

    markers: dict[tuple[int, int], int] = {}
    objects: list[dict] = []
    empties = 0
    for ri, (r0, r1) in enumerate(row_runs):
        for ci, (c0, c1) in enumerate(col_runs):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            info = _classify_cell(cell, bg, DIVIDER)
            info.update({"cell": (ri, ci), "cell_bounds": (r0, r1, c0, c1)})
            if info["kind"] == "marker":
                markers[(ri, ci)] = int(info["color"])
            elif info["kind"] in {"solid", "hollow"}:
                objects.append(info)
            else:
                empties += 1

    assert markers, "expected at least one marker cell"
    assert objects, "expected at least one object cell"
    assert empties > 0, "keep some empty cells for visual separation"

    return {
        "grid": g,
        "bg": bg,
        "row_runs": row_runs,
        "col_runs": col_runs,
        "markers": markers,
        "objects": objects,
    }


def _paint_object(
    out: np.ndarray,
    cell_bounds: tuple[int, int, int, int],
    local_bbox: tuple[int, int, int, int],
    kind: str,
    color: int,
    bg: int,
) -> None:
    cell_r0, _cell_r1, cell_c0, _cell_c1 = cell_bounds
    r0, c0, r1, c1 = local_bbox
    ar0 = int(cell_r0) + int(r0)
    ac0 = int(cell_c0) + int(c0)
    ar1 = int(cell_r0) + int(r1)
    ac1 = int(cell_c0) + int(c1)
    if kind == "solid":
        out[ar0 : ar1 + 1, ac0 : ac1 + 1] = int(color)
        return

    assert kind == "hollow"
    out[ar0 : ar1 + 1, ac0 : ac1 + 1] = int(bg)
    out[ar0, ac0 : ac1 + 1] = int(color)
    out[ar1, ac0 : ac1 + 1] = int(color)
    out[ar0 : ar1 + 1, ac0] = int(color)
    out[ar0 : ar1 + 1, ac1] = int(color)


def _render_with_modes(
    analysis: dict,
    *,
    use_row: bool,
    use_col: bool,
    hollow_as_hollow: bool,
    solid_as_solid: bool,
) -> np.ndarray:
    out = analysis["grid"].copy()
    bg = int(analysis["bg"])
    markers = analysis["markers"]
    marker_rows = {cell[0] for cell in markers}
    marker_cols = {cell[1] for cell in markers}
    row_color = {ri: color for (ri, _ci), color in markers.items()}
    col_color = {ci: color for (_ri, ci), color in markers.items()}

    for obj in analysis["objects"]:
        ri, ci = obj["cell"]
        if ri in marker_rows:
            if not use_row:
                continue
            color = int(row_color[ri])
        else:
            assert ci in marker_cols
            if not use_col:
                continue
            color = int(col_color[ci])

        kind = obj["kind"]
        effective_kind = kind
        if kind == "hollow" and not hollow_as_hollow:
            effective_kind = "solid"
        if kind == "solid" and not solid_as_solid:
            effective_kind = "hollow"
        _paint_object(out, obj["cell_bounds"], obj["bbox"], effective_kind, color, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor aligned 3x3 objects inside a divider table, using frame-vs-solid type to choose how repainting is applied.

    Input structure:
    - The background is black, and full grey separator lines partition the grid into a regular table of cells.
    - Some cells contain a centered 3x3 maroon object: either a hollow 3x3 frame with a black center or a solid 3x3 block.
    - Some cells contain exactly one colored marker pixel, and the remaining cells are empty.

    Rule:
    - Split the grid into table cells using the full grey divider lines.
    - Classify each occupied cell as marker, hollow frame, solid block, or empty.
    - For every object cell, find the unique marker cell that shares its table row or its table column.
    - Repaint the object with that marker color: solid blocks become fully filled, while hollow frames repaint only their 3x3 border and keep the center black.
    - Preserve divider lines, marker pixels, and empty cells.

    Key insight:
    - The marker color is determined by row/column alignment, but the visible repainting pattern is determined independently by whether the aligned object is hollow or solid.
    """
    analysis = _analyze_grid(grid)
    markers = analysis["markers"]
    marker_rows = {cell[0] for cell in markers}
    marker_cols = {cell[1] for cell in markers}

    row_color = {}
    col_color = {}
    for (ri, ci), color in markers.items():
        assert ri not in row_color and ci not in col_color, "marker rows and columns must be unique"
        row_color[ri] = int(color)
        col_color[ci] = int(color)

    out = analysis["grid"].copy()
    bg = int(analysis["bg"])

    for obj in analysis["objects"]:
        ri, ci = obj["cell"]
        aligned = []
        if ri in row_color:
            aligned.append(int(row_color[ri]))
        if ci in col_color:
            aligned.append(int(col_color[ci]))
        assert len(aligned) == 1, "each object must have exactly one aligned marker"

        assert (ri in marker_rows) ^ (ci in marker_cols), "objects must align by exactly one axis"
        _paint_object(out, obj["cell_bounds"], obj["bbox"], obj["kind"], aligned[0], bg)

    return out


def _grid_shape(n_rows: int, n_cols: int, cell_size: int = CELL_SIZE) -> tuple[int, int]:
    return n_rows * cell_size + (n_rows - 1), n_cols * cell_size + (n_cols - 1)


def _cell_origin(index: int, cell_size: int = CELL_SIZE) -> int:
    return int(index) * (int(cell_size) + 1)


def _draw_dividers(grid: np.ndarray, n_rows: int, n_cols: int, cell_size: int = CELL_SIZE) -> None:
    for ri in range(1, n_rows):
        grid[ri * (cell_size + 1) - 1, :] = DIVIDER
    for ci in range(1, n_cols):
        grid[:, ci * (cell_size + 1) - 1] = DIVIDER


def _marker_layout(n_rows: int, n_cols: int, k: int) -> list[tuple[int, int]]:
    rows = sorted(random.sample(range(n_rows), k))
    cols = sorted(random.sample(range(n_cols), k))
    permuted_cols = random.sample(cols, len(cols))
    return [(rows[i], permuted_cols[i]) for i in range(k)]


def _valid_slots(
    n_rows: int,
    n_cols: int,
    markers: list[tuple[int, int]],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    marker_rows = {r for r, _c in markers}
    marker_cols = {c for _r, c in markers}
    row_slots: dict[int, list[tuple[int, int]]] = {}
    col_slots: dict[int, list[tuple[int, int]]] = {}
    for idx, (mr, mc) in enumerate(markers):
        row_slots[idx] = [(mr, c) for c in range(n_cols) if c not in marker_cols]
        col_slots[idx] = [(r, mc) for r in range(n_rows) if r not in marker_rows]
    assert all(row_slots[idx] and col_slots[idx] for idx in range(len(markers)))
    return row_slots, col_slots


def _render_input(
    n_rows: int,
    n_cols: int,
    markers: list[tuple[int, int, int]],
    objects: list[dict],
    cell_size: int = CELL_SIZE,
) -> np.ndarray:
    h, w = _grid_shape(n_rows, n_cols, cell_size)
    grid = np.full((h, w), BG, dtype=int)
    _draw_dividers(grid, n_rows, n_cols, cell_size)

    for ri, ci, color in markers:
        r0 = _cell_origin(ri, cell_size)
        c0 = _cell_origin(ci, cell_size)
        grid[r0 + cell_size // 2, c0 + cell_size // 2] = int(color)

    for obj in objects:
        ri, ci = obj["cell"]
        r0 = _cell_origin(ri, cell_size)
        c0 = _cell_origin(ci, cell_size)
        if obj["kind"] == "solid":
            grid[r0 : r0 + 3, c0 : c0 + 3] = OBJECT_COLOR
        else:
            grid[r0 : r0 + 3, c0 : c0 + 3] = BG
            grid[r0, c0 : c0 + 3] = OBJECT_COLOR
            grid[r0 + 2, c0 : c0 + 3] = OBJECT_COLOR
            grid[r0 : r0 + 3, c0] = OBJECT_COLOR
            grid[r0 : r0 + 3, c0 + 2] = OBJECT_COLOR

    return grid


def _expected_output(
    inp: np.ndarray,
    markers: list[tuple[int, int, int]],
    objects: list[dict],
) -> np.ndarray:
    out = np.array(inp, copy=True)
    marker_color = {(ri, ci): color for ri, ci, color in markers}
    marker_rows = {ri for ri, _ci, _color in markers}
    marker_cols = {ci for _ri, ci, _color in markers}
    row_color = {ri: color for ri, _ci, color in markers}
    col_color = {ci: color for _ri, ci, color in markers}

    for obj in objects:
        ri, ci = obj["cell"]
        assert (ri in marker_rows) ^ (ci in marker_cols)
        if ri in row_color:
            color = int(row_color[ri])
        else:
            color = int(col_color[ci])
        r0 = _cell_origin(ri)
        c0 = _cell_origin(ci)
        if obj["kind"] == "solid":
            out[r0 : r0 + 3, c0 : c0 + 3] = color
        else:
            out[r0 : r0 + 3, c0 : c0 + 3] = BG
            out[r0, c0 : c0 + 3] = color
            out[r0 + 2, c0 : c0 + 3] = color
            out[r0 : r0 + 3, c0] = color
            out[r0 : r0 + 3, c0 + 2] = color

    for ri, ci, color in markers:
        r0 = _cell_origin(ri)
        c0 = _cell_origin(ci)
        out[r0 + 1, c0 + 1] = int(color)
        assert marker_color[(ri, ci)] == int(color)
    return out


def _choose_objects(
    n_rows: int,
    n_cols: int,
    markers: list[tuple[int, int, int]],
) -> list[dict] | None:
    bare_markers = [(ri, ci) for ri, ci, _color in markers]
    row_slots, col_slots = _valid_slots(n_rows, n_cols, bare_markers)
    k = len(markers)
    focus = random.randrange(k)

    chosen: dict[tuple[int, int], dict] = {}
    row_cell = random.choice(row_slots[focus])
    col_cell = random.choice(col_slots[focus])
    pair_types = ["hollow", "solid"]
    random.shuffle(pair_types)
    chosen[row_cell] = {"cell": row_cell, "kind": pair_types[0], "marker_idx": focus, "axis": "row"}
    chosen[col_cell] = {"cell": col_cell, "kind": pair_types[1], "marker_idx": focus, "axis": "col"}

    for idx in range(k):
        if idx == focus:
            continue
        candidates = [(cell, "row") for cell in row_slots[idx]] + [(cell, "col") for cell in col_slots[idx]]
        random.shuffle(candidates)
        for cell, axis in candidates:
            if cell not in chosen:
                chosen[cell] = {
                    "cell": cell,
                    "kind": random.choice(["hollow", "solid"]),
                    "marker_idx": idx,
                    "axis": axis,
                }
                break

    extras = []
    for idx in range(k):
        extras.extend((cell, idx, "row") for cell in row_slots[idx] if cell not in chosen)
        extras.extend((cell, idx, "col") for cell in col_slots[idx] if cell not in chosen)
    random.shuffle(extras)

    if min(n_rows, n_cols) == 3:
        target_count = 3
    else:
        target_count = random.randint(max(4, k + 1), min(6, len(chosen) + len(extras)))

    while len(chosen) < target_count and extras:
        cell, idx, axis = extras.pop()
        if cell in chosen:
            continue
        chosen[cell] = {
            "cell": cell,
            "kind": random.choice(["hollow", "solid"]),
            "marker_idx": idx,
            "axis": axis,
        }

    objects = list(chosen.values())
    marker_to_types: dict[int, set[str]] = {}
    for obj in objects:
        marker_to_types.setdefault(int(obj["marker_idx"]), set()).add(str(obj["kind"]))

    if not any(types == {"hollow", "solid"} for types in marker_to_types.values()):
        return None
    if {obj["axis"] for obj in objects} != {"row", "col"}:
        return None
    if {obj["kind"] for obj in objects} != {"hollow", "solid"}:
        return None

    return objects


def _validate_generated(
    inp: np.ndarray,
    markers: list[tuple[int, int, int]],
    objects: list[dict],
    expected: np.ndarray,
) -> bool:
    try:
        analysis = _analyze_grid(inp)
    except Exception:
        return False

    parsed_markers = {
        (ri, ci, color)
        for (ri, ci), color in analysis["markers"].items()
    }
    if parsed_markers != set(markers):
        return False

    parsed_objects = {(obj["cell"], obj["kind"]) for obj in analysis["objects"]}
    target_objects = {(tuple(obj["cell"]), str(obj["kind"])) for obj in objects}
    if parsed_objects != target_objects:
        return False

    marker_rows = [ri for ri, _ci, _color in markers]
    marker_cols = [ci for _ri, ci, _color in markers]
    if len(marker_rows) != len(set(marker_rows)) or len(marker_cols) != len(set(marker_cols)):
        return False

    marker_color_by_idx = {idx: markers[idx][2] for idx in range(len(markers))}
    per_marker_types: dict[int, set[str]] = {}
    output_colors = set()
    for obj in objects:
        ri, ci = obj["cell"]
        row_hits = sum(1 for mr, _mc, _color in markers if mr == ri)
        col_hits = sum(1 for _mr, mc, _color in markers if mc == ci)
        if row_hits + col_hits != 1:
            return False
        per_marker_types.setdefault(int(obj["marker_idx"]), set()).add(str(obj["kind"]))
        output_colors.add(int(marker_color_by_idx[int(obj["marker_idx"])]))

    if {obj["axis"] for obj in objects} != {"row", "col"}:
        return False
    if {obj["kind"] for obj in objects} != {"hollow", "solid"}:
        return False
    if len(output_colors) < 2:
        return False
    if not any(types == {"hollow", "solid"} for types in per_marker_types.values()):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, inp):
        return False

    row_only = _render_with_modes(analysis, use_row=True, use_col=False, hollow_as_hollow=True, solid_as_solid=True)
    col_only = _render_with_modes(analysis, use_row=False, use_col=True, hollow_as_hollow=True, solid_as_solid=True)
    all_solid = _render_with_modes(analysis, use_row=True, use_col=True, hollow_as_hollow=False, solid_as_solid=True)
    all_hollow = _render_with_modes(analysis, use_row=True, use_col=True, hollow_as_hollow=True, solid_as_solid=False)
    if np.array_equal(row_only, expected):
        return False
    if np.array_equal(col_only, expected):
        return False
    if np.array_equal(all_solid, expected):
        return False
    if np.array_equal(all_hollow, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choice([3, 4])
    n_cols = random.choice([3, 4])
    k = 2 if min(n_rows, n_cols) == 3 else random.choice([2, 3])
    if k >= min(n_rows, n_cols):
        return None

    marker_positions = _marker_layout(n_rows, n_cols, k)
    marker_colors = random.sample(MARKER_COLORS, k)
    markers = [(ri, ci, marker_colors[idx]) for idx, (ri, ci) in enumerate(marker_positions)]
    objects = _choose_objects(n_rows, n_cols, markers)
    if objects is None:
        return None

    inp = _render_input(n_rows, n_cols, markers, objects)
    expected = _expected_output(inp, markers, objects)
    if not _validate_generated(inp, markers, objects, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
