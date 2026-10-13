from __future__ import annotations

import numpy as np


LEFT_PANEL = 1
RIGHT_PANEL = 2
MARKER_COLORS = (3, 4, 6, 7)
GREY = 5
GUIDE = 8

GEN_ATTEMPTS = 600
HEIGHT_RANGE = (15, 22)
WIDTH_CHOICES = (15, 17, 19, 21)
RECT_SIZE_CHOICES = (3, 4, 5)
MARKER_COUNT_CHOICES = (3, 4)


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _find_divider_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(candidates) != 1:
        raise ValueError("expected exactly one full-height grey divider")
    return int(candidates[0])


def _find_center_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = grid == GREY
    grey_counts = grey_mask.sum(axis=1)
    rect_rows = [int(r) for r in np.where(grey_counts > 1)[0]]
    if len(rect_rows) < 3:
        raise ValueError("grey rectangle rows not found")
    if any(b - a != 1 for a, b in zip(rect_rows, rect_rows[1:])):
        raise ValueError("grey rectangle rows must be contiguous")

    full_cols = [c for c in range(grid.shape[1]) if np.all(grey_mask[rect_rows, c])]
    runs = [(c0, c1) for c0, c1 in _contiguous_runs(full_cols) if c1 - c0 + 1 >= 3]
    if len(runs) != 1:
        raise ValueError("expected exactly one solid grey rectangle column run")

    r0, r1 = rect_rows[0], rect_rows[-1]
    c0, c1 = runs[0]
    if not np.all(grey_mask[r0 : r1 + 1, c0 : c1 + 1]):
        raise ValueError("grey rectangle is not solid")
    return r0, c0, r1, c1


def _path_options(
    marker: tuple[int, int], rect: tuple[int, int, int, int]
) -> list[tuple[str, list[tuple[int, int]]]]:
    r, c = marker
    r0, c0, r1, c1 = rect
    options: list[tuple[str, list[tuple[int, int]]]] = []

    if c < c0 and r0 <= r <= r1:
        options.append(("horizontal", [(r, cc) for cc in range(c + 1, c0)]))
    if c > c1 and r0 <= r <= r1:
        options.append(("horizontal", [(r, cc) for cc in range(c1 + 1, c)]))
    if r < r0 and c0 <= c <= c1:
        options.append(("vertical", [(rr, c) for rr in range(r + 1, r0)]))
    if r > r1 and c0 <= c <= c1:
        options.append(("vertical", [(rr, c) for rr in range(r1 + 1, r)]))

    # Allow corner-adjacent alignments so crossings can occur next to the rectangle.
    if c < c0 and r in {r0 - 1, r1 + 1}:
        options.append(("horizontal", [(r, cc) for cc in range(c + 1, c0)]))
    if c > c1 and r in {r0 - 1, r1 + 1}:
        options.append(("horizontal", [(r, cc) for cc in range(c1 + 1, c)]))
    if r < r0 and c in {c0 - 1, c1 + 1}:
        options.append(("vertical", [(rr, c) for rr in range(r + 1, r0)]))
    if r > r1 and c in {c0 - 1, c1 + 1}:
        options.append(("vertical", [(rr, c) for rr in range(r1 + 1, r)]))

    deduped: list[tuple[str, list[tuple[int, int]]]] = []
    seen = set()
    for orientation, cells in options:
        key = (orientation, tuple(cells))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((orientation, cells))
    return deduped


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        raise ValueError("grid must be a 2D ARC grid")

    divider_col = _find_divider_col(g)
    rect = _find_center_rect(g)
    r0, c0, r1, c1 = rect

    if int(g[-1, 0]) != GUIDE or int(np.count_nonzero(g == GUIDE)) != 1:
        raise ValueError("expected a unique bottom-left guide pixel")

    allowed = {LEFT_PANEL, RIGHT_PANEL, GREY, GUIDE, *MARKER_COLORS}
    if not set(np.unique(g)).issubset(allowed):
        raise ValueError("unexpected colors present")

    markers: list[dict] = []
    coords_seen: set[tuple[int, int]] = set()
    for color in MARKER_COLORS:
        for r, c in np.argwhere(g == color):
            coord = (int(r), int(c))
            if coord == (g.shape[0] - 1, 0):
                raise ValueError("guide pixel cannot be a marker")
            options = _path_options(coord, rect)
            if len(options) != 1:
                raise ValueError("each marker must have exactly one valid alignment")
            orientation, raw_path = options[0]
            if not raw_path:
                raise ValueError("markers must be at least one cell away from the target edge")
            line_color = RIGHT_PANEL if c < divider_col else LEFT_PANEL
            path = [cell for cell in raw_path if int(g[cell]) in {LEFT_PANEL, RIGHT_PANEL}]
            markers.append(
                {
                    "coord": coord,
                    "color": color,
                    "orientation": orientation,
                    "path": path,
                    "raw_path": raw_path,
                    "line_color": line_color,
                    "panel": "left" if c < divider_col else "right",
                }
            )
            coords_seen.add(coord)

    if len(markers) < 3 or len(markers) > len(MARKER_COLORS):
        raise ValueError("expected 3-4 markers")
    if len({m["color"] for m in markers}) != len(markers):
        raise ValueError("marker colors must be distinct")

    occupancy: dict[tuple[int, int], set[int]] = {}
    for idx, marker in enumerate(markers):
        for cell in marker["path"]:
            if cell in coords_seen or cell == (g.shape[0] - 1, 0):
                raise ValueError("line path cannot pass through a marker or the guide")
            if int(g[cell]) == GREY:
                raise ValueError("visible line cells cannot overwrite grey structure")
            occupancy.setdefault(cell, set()).add(idx)

    crossings = {cell for cell, owners in occupancy.items() if len(owners) >= 2}
    if not crossings:
        raise ValueError("at least one line crossing is required")
    if not any(m["orientation"] == "horizontal" for m in markers):
        raise ValueError("need a horizontal line")
    if not any(m["orientation"] == "vertical" for m in markers):
        raise ValueError("need a vertical line")
    if {m["panel"] for m in markers} != {"left", "right"}:
        raise ValueError("need markers on both panels")

    return {
        "grid": g,
        "divider_col": divider_col,
        "rect": rect,
        "markers": markers,
        "occupancy": occupancy,
        "crossings": crossings,
    }


def _render_solution(scene: dict) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()

    for marker in scene["markers"]:
        for cell in marker["path"]:
            out[cell] = marker["line_color"]

    for cell in scene["crossings"]:
        out[cell] = GUIDE

    out[-1, 0] = LEFT_PANEL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw opposite-panel guide lines to the center block, recolor line crossings with the corner guide, then erase the guide.

    Input structure:
    - The grid is split into a blue left panel and a red right panel by one full-height grey divider column.
    - A solid grey rectangle sits near the center, and 3-4 isolated colored markers lie around it.
    - The bottom-left corner contains a single teal guide pixel, while the divider and rectangle are also grey.

    Rule:
    - Recover the central grey rectangle and treat every non-grey marker color as an isolated source pixel.
    - Classify each marker by panel membership: markers left of the divider belong to the blue panel, markers right of it belong to the red panel.
    - Extend each marker straight toward the rectangle along its unique row/column alignment, stopping at the cell immediately before the grey structure; keep the marker itself unchanged.
    - Paint each line with the opposite panel base color, then recolor any cell traversed by at least two different lines with the bottom-left guide color.
    - Finally erase the bottom-left guide pixel back to the blue panel color while preserving the divider, the rectangle, and the markers.

    Key insight:
    - The added line color depends on spatial panel membership rather than the marker's own color, and the guide pixel matters only at line intersections before it disappears.
    """
    return _render_solution(_analyze_scene(np.asarray(grid, dtype=int)))


def _base_grid(height: int, width: int) -> tuple[np.ndarray, int]:
    divider_col = width // 2
    grid = np.full((height, width), RIGHT_PANEL, dtype=int)
    grid[:, :divider_col] = LEFT_PANEL
    grid[:, divider_col] = GREY
    return grid, divider_col


def _choose_rect(height: int, width: int, divider_col: int) -> tuple[int, int, int, int]:
    rect_h = int(np.random.choice(RECT_SIZE_CHOICES))
    rect_w = int(np.random.choice(RECT_SIZE_CHOICES))

    row_center = height // 2 + int(np.random.randint(-1, 2))
    r0 = max(2, min(row_center - rect_h // 2, height - rect_h - 3))
    r1 = r0 + rect_h - 1

    mode = str(np.random.choice(["span", "left", "right"]))
    if mode == "span" and rect_w >= 3:
        left_width = int(np.random.randint(1, rect_w))
        c0 = divider_col - left_width
        c1 = c0 + rect_w - 1
        if c0 >= 2 and c1 <= width - 3 and c0 <= divider_col - 1 and c1 >= divider_col + 1:
            return r0, c0, r1, c1

    if mode == "left":
        max_c0 = divider_col - rect_w - 1
        if max_c0 >= 2:
            c0 = int(np.random.randint(2, max_c0 + 1))
            return r0, c0, r1, c0 + rect_w - 1

    min_c0 = divider_col + 1
    max_c0 = width - rect_w - 2
    if max_c0 >= min_c0:
        c0 = int(np.random.randint(min_c0, max_c0 + 1))
        return r0, c0, r1, c0 + rect_w - 1

    raise ValueError("could not place rectangle")


def _place_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY


def _corner_specs(rect: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = rect
    return [
        {
            "name": "top_left",
            "cross": (r0 - 1, c0 - 1),
            "horizontal_side": "left",
            "vertical_side": "top",
        },
        {
            "name": "top_right",
            "cross": (r0 - 1, c1 + 1),
            "horizontal_side": "right",
            "vertical_side": "top",
        },
        {
            "name": "bottom_left",
            "cross": (r1 + 1, c0 - 1),
            "horizontal_side": "left",
            "vertical_side": "bottom",
        },
        {
            "name": "bottom_right",
            "cross": (r1 + 1, c1 + 1),
            "horizontal_side": "right",
            "vertical_side": "bottom",
        },
    ]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_cells_for_candidate(
    rect: tuple[int, int, int, int], coord: tuple[int, int]
) -> tuple[str, list[tuple[int, int]]] | None:
    options = _path_options(coord, rect)
    if len(options) != 1 or not options[0][1]:
        return None
    return options[0]


def _candidate_position_sets(
    rect: tuple[int, int, int, int], shape: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    height, width = shape
    positions = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
    }

    for r in range(r0 - 1, r1 + 2):
        if not (0 <= r < height):
            continue
        for c in range(1, c0 - 1):
            if _line_cells_for_candidate(rect, (r, c)) is not None:
                positions["left"].append((r, c))
        for c in range(c1 + 2, width - 1):
            if _line_cells_for_candidate(rect, (r, c)) is not None:
                positions["right"].append((r, c))

    for c in range(c0 - 1, c1 + 2):
        if not (0 <= c < width):
            continue
        for r in range(1, r0 - 1):
            if _line_cells_for_candidate(rect, (r, c)) is not None:
                positions["top"].append((r, c))
        for r in range(r1 + 2, height - 1):
            if _line_cells_for_candidate(rect, (r, c)) is not None:
                positions["bottom"].append((r, c))

    return positions


def _path_signature(
    grid: np.ndarray, rect: tuple[int, int, int, int], coord: tuple[int, int], divider_col: int
) -> dict | None:
    result = _line_cells_for_candidate(rect, coord)
    if result is None:
        return None
    orientation, raw_path = result
    if any(int(grid[cell]) not in {LEFT_PANEL, RIGHT_PANEL} for cell in raw_path):
        return None
    return {
        "coord": coord,
        "orientation": orientation,
        "path": raw_path,
        "panel": "left" if coord[1] < divider_col else "right",
    }


def _select_cross_pair(
    grid: np.ndarray, rect: tuple[int, int, int, int], divider_col: int
) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]]]:
    candidates_by_side = _candidate_position_sets(rect, grid.shape)
    corner_specs = _corner_specs(rect)
    np.random.shuffle(corner_specs)

    for spec in corner_specs:
        cross = spec["cross"]
        if not _in_bounds(grid.shape, cross):
            continue
        if int(grid[cross]) not in {LEFT_PANEL, RIGHT_PANEL}:
            continue

        horiz_candidates = candidates_by_side[spec["horizontal_side"]]
        vert_candidates = candidates_by_side[spec["vertical_side"]]
        np.random.shuffle(horiz_candidates)
        np.random.shuffle(vert_candidates)

        for h_coord in horiz_candidates:
            if cross not in _line_cells_for_candidate(rect, h_coord)[1]:
                continue
            h_sig = _path_signature(grid, rect, h_coord, divider_col)
            if h_sig is None:
                continue
            for v_coord in vert_candidates:
                if cross not in _line_cells_for_candidate(rect, v_coord)[1]:
                    continue
                if h_coord == v_coord:
                    continue
                v_sig = _path_signature(grid, rect, v_coord, divider_col)
                if v_sig is None:
                    continue

                markers = [h_sig, v_sig]
                coords = {h_coord, v_coord}
                path_cells = set(h_sig["path"]) | set(v_sig["path"])
                if coords & path_cells:
                    continue
                return markers, coords, path_cells

    raise ValueError("failed to place a crossing pair")


def _extra_candidate_pool(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    divider_col: int,
    used_coords: set[tuple[int, int]],
    used_path_cells: set[tuple[int, int]],
) -> list[dict]:
    pool = []
    candidates = _candidate_position_sets(rect, grid.shape)
    for side in ("left", "right", "top", "bottom"):
        for coord in candidates[side]:
            if coord in used_coords or coord == (grid.shape[0] - 1, 0):
                continue
            sig = _path_signature(grid, rect, coord, divider_col)
            if sig is None:
                continue
            path_set = set(sig["path"])
            if path_set & used_coords:
                continue
            if path_set & used_path_cells:
                continue
            pool.append(sig)
    np.random.shuffle(pool)
    return pool


def _choose_markers(
    grid: np.ndarray, rect: tuple[int, int, int, int], divider_col: int
) -> list[dict]:
    markers, used_coords, used_path_cells = _select_cross_pair(grid, rect, divider_col)
    target_count = int(np.random.choice(MARKER_COUNT_CHOICES))

    needs_left = not any(m["panel"] == "left" for m in markers)
    needs_right = not any(m["panel"] == "right" for m in markers)
    pool = _extra_candidate_pool(grid, rect, divider_col, used_coords, used_path_cells)

    chosen = list(markers)
    for candidate in pool:
        if len(chosen) >= target_count and not needs_left and not needs_right:
            break
        if needs_left and candidate["panel"] != "left":
            continue
        if needs_right and candidate["panel"] != "right":
            continue
        chosen.append(candidate)
        used_coords.add(candidate["coord"])
        used_path_cells.update(candidate["path"])
        needs_left = needs_left and candidate["panel"] != "left"
        needs_right = needs_right and candidate["panel"] != "right"

    if needs_left or needs_right or len(chosen) < 3:
        for candidate in pool:
            if candidate["coord"] in used_coords:
                continue
            chosen.append(candidate)
            used_coords.add(candidate["coord"])
            used_path_cells.update(candidate["path"])
            if len(chosen) >= max(3, target_count):
                break

    if len(chosen) < 3 or len(chosen) > 4:
        raise ValueError("invalid marker count")
    if {m["panel"] for m in chosen} != {"left", "right"}:
        raise ValueError("markers must cover both panels")
    return chosen


def _paint_markers(grid: np.ndarray, markers: list[dict]) -> None:
    colors = list(np.random.choice(MARKER_COLORS, size=len(markers), replace=False))
    np.random.shuffle(colors)
    for color, marker in zip(colors, markers):
        r, c = marker["coord"]
        grid[r, c] = int(color)


def _semantic_checks(grid: np.ndarray) -> None:
    scene = _analyze_scene(grid)
    out = _render_solution(scene)

    if int(out[-1, 0]) != LEFT_PANEL:
        raise ValueError("guide was not erased correctly")

    rect = scene["rect"]
    r0, c0, r1, c1 = rect
    if not np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
        raise ValueError("rectangle must remain unchanged")
    if not np.array_equal(grid[:, scene["divider_col"]], out[:, scene["divider_col"]]):
        raise ValueError("divider must remain unchanged")

    marker_coords = [m["coord"] for m in scene["markers"]]
    for marker in scene["markers"]:
        coord = marker["coord"]
        if int(out[coord]) != int(grid[coord]):
            raise ValueError("marker pixels must stay unchanged")
        for cell in marker["path"]:
            expected = GUIDE if cell in scene["crossings"] else marker["line_color"]
            if int(out[cell]) != expected:
                raise ValueError("line coloring is inconsistent")

    if len(set(r for r, _ in marker_coords)) == 1 or len(set(c for _, c in marker_coords)) == 1:
        raise ValueError("markers cannot all share one row or one column")
    if len(scene["crossings"]) < 1:
        raise ValueError("crossings missing after analysis")


def _gen_attempt() -> np.ndarray:
    height = int(np.random.randint(*HEIGHT_RANGE))
    width = int(np.random.choice(WIDTH_CHOICES))
    grid, divider_col = _base_grid(height, width)
    rect = _choose_rect(height, width, divider_col)
    _place_rectangle(grid, rect)
    markers = _choose_markers(grid, rect, divider_col)
    _paint_markers(grid, markers)
    grid[-1, 0] = GUIDE
    _semantic_checks(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _gen_attempt()
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
