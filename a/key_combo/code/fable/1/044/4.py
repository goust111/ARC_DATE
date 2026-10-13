from __future__ import annotations

import random

import numpy as np


QUADRANTS = ("tl", "tr", "bl", "br")
CORNER_TO_OFFSET = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = [int(v) for v in np.unique(np.asarray(grid, dtype=int))]
    if exclude is None:
        return colors
    return [c for c in colors if c not in exclude]


def _find_palette_blocks(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2].reshape(-1)
            if np.any(block == bg):
                continue
            if len(set(map(int, block))) == 4:
                candidates.append((r, c))
    return candidates


def _palette_color_map(grid: np.ndarray, palette_rc: tuple[int, int]) -> dict[str, int]:
    r, c = palette_rc
    return {
        "tl": int(grid[r, c]),
        "tr": int(grid[r, c + 1]),
        "bl": int(grid[r + 1, c]),
        "br": int(grid[r + 1, c + 1]),
    }


def _candidate_groups(coords: set[tuple[int, int]]) -> list[dict]:
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    seen: set[tuple[tuple[int, int], ...]] = set()
    groups: list[dict] = []

    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            side = int(r1 - r0)
            for c0 in cols:
                c1 = c0 + side
                corners = {
                    "tl": (r0, c0),
                    "tr": (r0, c1),
                    "bl": (r1, c0),
                    "br": (r1, c1),
                }
                present_names = [name for name, pos in corners.items() if pos in coords]
                if len(present_names) != 3:
                    continue
                missing_kind = next(name for name in QUADRANTS if name not in present_names)
                present = tuple(sorted(corners[name] for name in present_names))
                if present in seen:
                    continue
                seen.add(present)
                groups.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "side": side,
                        "present": present,
                        "present_set": set(present),
                        "missing": corners[missing_kind],
                        "missing_kind": missing_kind,
                    }
                )
    return groups


def _recover_three_corner_groups(coords: np.ndarray | list[tuple[int, int]]) -> list[dict]:
    coord_set = {tuple(map(int, pt)) for pt in coords}
    assert coord_set, "expected at least one marker pixel"
    assert len(coord_set) % 3 == 0, "marker pixels must split into 3-corner groups"

    candidates = _candidate_groups(coord_set)
    assert candidates, "could not form any 3-corner square candidates"

    cover_map: dict[tuple[int, int], list[int]] = {coord: [] for coord in coord_set}
    for idx, candidate in enumerate(candidates):
        for coord in candidate["present"]:
            cover_map[coord].append(idx)
    assert all(cover_map.values()), "some marker pixels do not belong to any square candidate"

    target_group_count = len(coord_set) // 3
    ordered_coords = sorted(coord_set)
    solutions: list[list[int]] = []
    chosen: list[int] = []
    used: set[tuple[int, int]] = set()

    def dfs() -> None:
        if len(solutions) > 1:
            return
        if len(chosen) > target_group_count:
            return
        if len(used) == len(coord_set):
            if len(chosen) == target_group_count:
                solutions.append(list(chosen))
            return

        best_coord: tuple[int, int] | None = None
        best_options: list[int] | None = None
        for coord in ordered_coords:
            if coord in used:
                continue
            options = [
                idx
                for idx in cover_map[coord]
                if candidates[idx]["present_set"].isdisjoint(used)
            ]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_coord = coord
                best_options = options

        assert best_coord is not None and best_options is not None
        for idx in best_options:
            chosen.append(idx)
            new_cells = tuple(candidates[idx]["present"])
            used.update(new_cells)
            dfs()
            for cell in new_cells:
                used.remove(cell)
            chosen.pop()

    dfs()
    assert len(solutions) == 1, "marker pixels must partition into one unique set of square groups"

    groups = [candidates[idx] for idx in solutions[0]]
    groups.sort(key=lambda item: item["bbox"])
    return groups


def _find_marker_groups(
    grid: np.ndarray,
    bg: int,
    palette_rc: tuple[int, int],
) -> tuple[int, list[dict]]:
    pr, pc = palette_rc
    palette_cells = {(pr, pc), (pr, pc + 1), (pr + 1, pc), (pr + 1, pc + 1)}
    matches: list[tuple[int, list[dict]]] = []

    for color in _colors_present(grid, exclude={bg}):
        coords = [
            (int(r), int(c))
            for r, c in np.argwhere(grid == color)
            if (int(r), int(c)) not in palette_cells
        ]
        if not coords or len(coords) % 3 != 0:
            continue
        try:
            groups = _recover_three_corner_groups(coords)
        except AssertionError:
            continue
        matches.append((int(color), groups))

    assert len(matches) == 1, "expected exactly one marker color forming 3-corner square groups"
    return matches[0]


def _group_quadrant(group: dict, palette_rc: tuple[int, int]) -> str:
    pr, pc = palette_rc
    palette_center = (pr + 0.5, pc + 0.5)
    r0, c0, r1, c1 = group["bbox"]
    group_center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    assert group_center[0] != palette_center[0]
    assert group_center[1] != palette_center[1]
    return ("t" if group_center[0] < palette_center[0] else "b") + (
        "l" if group_center[1] < palette_center[1] else "r"
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each three-corner square marker with the palette color from the marker group's global quadrant.

    Input structure:
    - The grid has one dominant background color and exactly one solid 2x2 palette block whose four cells are all different colors.
    - Elsewhere, one repeated marker color appears only as singleton pixels.
    - Those singleton pixels partition into several groups of three, and each group is exactly three corners of one axis-aligned square.

    Rule:
    - Keep the palette block and all existing marker pixels unchanged.
    - For each 3-pixel square-corner group, infer the missing fourth corner position from the square geometry.
    - Determine the whole group's quadrant relative to the center of the palette block, then copy the palette color from that same quadrant into the missing corner.

    Key insight:
    - The missing-corner direction decides only where to draw; the added color is looked up from the central 2x2 palette by the group's macro quadrant, not by the missing corner's own direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    palette_blocks = _find_palette_blocks(g, bg)
    assert len(palette_blocks) == 1, "expected exactly one 2x2 palette block"
    palette_rc = palette_blocks[0]
    palette_colors = _palette_color_map(g, palette_rc)
    _marker_color, groups = _find_marker_groups(g, bg, palette_rc)

    out = g.copy()
    for group in groups:
        quadrant = _group_quadrant(group, palette_rc)
        mr, mc = group["missing"]
        out[mr, mc] = palette_colors[quadrant]
    return out


GEN_MAX_ATTEMPTS = 500
GEN_GROUP_PADDING = 2
GEN_PALETTE_CLEARANCE = 3
GEN_MARKER_COLOR = 5
GEN_SIDE_RANGE = (2, 7)
GEN_GROUP_COUNT_RANGE = (4, 7)
GEN_GRID_RANGE = (20, 31)


def _palette_array(color_map: dict[str, int]) -> np.ndarray:
    return np.array(
        [
            [color_map["tl"], color_map["tr"]],
            [color_map["bl"], color_map["br"]],
        ],
        dtype=int,
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _random_deranged_assignment() -> dict[str, str]:
    while True:
        missing_kinds = list(QUADRANTS)
        random.shuffle(missing_kinds)
        mismatch_count = sum(
            quadrant != missing_kind
            for quadrant, missing_kind in zip(QUADRANTS, missing_kinds)
        )
        if mismatch_count >= 2:
            return dict(zip(QUADRANTS, missing_kinds))


def _sample_palette_colors(bg: int) -> dict[str, int]:
    usable = [c for c in range(10) if c not in {bg, GEN_MARKER_COLOR}]
    chosen = random.sample(usable, 4)
    random.shuffle(chosen)
    return dict(zip(QUADRANTS, chosen))


def _quadrant_ranges(
    quadrant: str,
    side: int,
    palette_rc: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[range, range]:
    pr, pc = palette_rc
    h, w = shape

    if quadrant[0] == "t":
        row_start = 1
        row_stop = pr - GEN_PALETTE_CLEARANCE - side + 1
    else:
        row_start = pr + 2 + GEN_PALETTE_CLEARANCE
        row_stop = h - side - 1

    if quadrant[1] == "l":
        col_start = 1
        col_stop = pc - GEN_PALETTE_CLEARANCE - side + 1
    else:
        col_start = pc + 2 + GEN_PALETTE_CLEARANCE
        col_stop = w - side - 1

    return range(row_start, row_stop), range(col_start, col_stop)


def _place_group(
    grid: np.ndarray,
    blocked: np.ndarray,
    palette_rc: tuple[int, int],
    quadrant: str,
    side: int,
    missing_kind: str,
) -> dict | None:
    row_range, col_range = _quadrant_ranges(quadrant, side, palette_rc, grid.shape)
    positions = [(r0, c0) for r0 in row_range for c0 in col_range]
    random.shuffle(positions)

    for r0, c0 in positions:
        r1, c1 = r0 + side, c0 + side
        er0, ec0, er1, ec1 = _expand_bbox((r0, c0, r1, c1), GEN_GROUP_PADDING, grid.shape)
        if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue

        corners = {
            "tl": (r0, c0),
            "tr": (r0, c1),
            "bl": (r1, c0),
            "br": (r1, c1),
        }
        present = [name for name in QUADRANTS if name != missing_kind]
        for name in present:
            rr, cc = corners[name]
            grid[rr, cc] = GEN_MARKER_COLOR

        blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return {
            "bbox": (r0, c0, r1, c1),
            "quadrant": quadrant,
            "side": side,
            "missing_kind": missing_kind,
            "missing": corners[missing_kind],
            "present": tuple(sorted(corners[name] for name in present)),
        }

    return None


def _intended_signature(groups: list[dict]) -> set[tuple]:
    return {
        (
            tuple(group["present"]),
            tuple(group["missing"]),
            str(group["missing_kind"]),
        )
        for group in groups
    }


def _build_expected_output(grid: np.ndarray, palette_colors: dict[str, int], groups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for group in groups:
        rr, cc = group["missing"]
        out[rr, cc] = palette_colors[group["quadrant"]]
    return out


def _sample_specs() -> list[tuple[str, int, str]]:
    assignment = _random_deranged_assignment()
    specs = [
        (quadrant, random.randrange(*GEN_SIDE_RANGE), assignment[quadrant])
        for quadrant in QUADRANTS
    ]

    extra_count = random.randrange(0, GEN_GROUP_COUNT_RANGE[1] - 4)
    for _ in range(extra_count):
        specs.append(
            (
                random.choice(QUADRANTS),
                random.randrange(*GEN_SIDE_RANGE),
                random.choice(QUADRANTS),
            )
        )

    specs.sort(key=lambda item: (-item[1], item[0], item[2]))
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(*GEN_GRID_RANGE)
    w = random.randrange(*GEN_GRID_RANGE)
    bg = random.choice([c for c in range(10) if c != GEN_MARKER_COLOR])
    grid = np.full((h, w), bg, dtype=int)

    palette_row_min = 7
    palette_row_max = h - 10
    palette_col_min = 7
    palette_col_max = w - 10
    if palette_row_min > palette_row_max or palette_col_min > palette_col_max:
        return None

    palette_rc = (
        random.randint(palette_row_min, palette_row_max),
        random.randint(palette_col_min, palette_col_max),
    )
    palette_colors = _sample_palette_colors(bg)
    pr, pc = palette_rc
    grid[pr : pr + 2, pc : pc + 2] = _palette_array(palette_colors)

    blocked = np.zeros((h, w), dtype=bool)
    er0, ec0, er1, ec1 = _expand_bbox((pr, pc, pr + 1, pc + 1), GEN_GROUP_PADDING, grid.shape)
    blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True

    placed_groups: list[dict] = []
    for quadrant, side, missing_kind in _sample_specs():
        placed = _place_group(grid, blocked, palette_rc, quadrant, side, missing_kind)
        if placed is None:
            return None
        placed_groups.append(placed)

    quadrant_set = {group["quadrant"] for group in placed_groups}
    missing_set = {group["missing_kind"] for group in placed_groups}
    mismatch_count = sum(group["quadrant"] != group["missing_kind"] for group in placed_groups)
    if quadrant_set != set(QUADRANTS):
        return None
    if missing_set != set(QUADRANTS):
        return None
    if mismatch_count < 2:
        return None

    expected = _build_expected_output(grid, palette_colors, placed_groups)

    bg2 = _get_bg(grid)
    if _find_palette_blocks(grid, bg2) != [palette_rc]:
        return None

    recovered_groups = _recover_three_corner_groups(np.argwhere(grid == GEN_MARKER_COLOR))
    if _intended_signature(recovered_groups) != _intended_signature(placed_groups):
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-palette ARC input")
