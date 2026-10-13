import numpy as np
from collections import defaultdict


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BELT_COLOR = 5
PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 5000


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _detect_belt_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(colors) > 0, "expected at least one foreground color"
    return int(colors[np.argmax(counts)])


def _split_belt_pixels(
    grid: np.ndarray, belt_color: int, cargo_cells: set[tuple[int, int]] | None = None
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    raw_mask = grid == belt_color
    h, w = grid.shape
    path_mask = raw_mask.copy()
    for r, c in cargo_cells or set():
        path_mask[r, c] = True
    arrow_pixels: set[tuple[int, int]] = set()
    for r, c in np.argwhere(raw_mask):
        r = int(r)
        c = int(c)
        degree = 0
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc]:
                degree += 1
        if degree == 0:
            path_mask[r, c] = False
            arrow_pixels.add((r, c))
    return path_mask, arrow_pixels


def _make_belt(cells: list[tuple[int, int]], orientation: str, direction: int) -> dict:
    return {
        "orientation": orientation,
        "direction": int(direction),
        "cells": cells,
        "index": {coord: idx for idx, coord in enumerate(cells)},
    }


def _extract_belts(path_mask: np.ndarray, arrow_pixels: set[tuple[int, int]]) -> list[dict]:
    h, w = path_mask.shape
    belts: list[dict] = []

    for r in range(h):
        c = 0
        while c < w:
            if not path_mask[r, c]:
                c += 1
                continue
            c0 = c
            while c < w and path_mask[r, c]:
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 < 2:
                continue
            left_arrow = (r - 1, c0 - 1) in arrow_pixels and (r + 1, c0 - 1) in arrow_pixels
            right_arrow = (r - 1, c1 + 1) in arrow_pixels and (r + 1, c1 + 1) in arrow_pixels
            if left_arrow ^ right_arrow:
                direction = 1 if left_arrow else -1
                span = [(r, cc) for cc in range(c0, c1 + 1)]
                if direction == -1:
                    span.reverse()
                belts.append(_make_belt(span, "h", direction))

    for c in range(w):
        r = 0
        while r < h:
            if not path_mask[r, c]:
                r += 1
                continue
            r0 = r
            while r < h and path_mask[r, c]:
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 < 2:
                continue
            top_arrow = (r0 - 1, c - 1) in arrow_pixels and (r0 - 1, c + 1) in arrow_pixels
            bottom_arrow = (r1 + 1, c - 1) in arrow_pixels and (r1 + 1, c + 1) in arrow_pixels
            if top_arrow ^ bottom_arrow:
                direction = 1 if top_arrow else -1
                span = [(rr, c) for rr in range(r0, r1 + 1)]
                if direction == -1:
                    span.reverse()
                belts.append(_make_belt(span, "v", direction))

    assert belts, "expected at least one belt"
    return belts


def _find_legend_and_cargos(
    grid: np.ndarray, bg: int, belt_color: int
) -> tuple[list[int], dict[int, tuple[int, int]]]:
    colored = (grid != bg) & (grid != belt_color)
    comps = _components(colored)
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for comp in comps:
        rows = sorted(r for r, _ in comp)
        cols = {c for _, c in comp}
        if len(comp) < 2 or len(cols) != 1:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        col = next(iter(cols))
        if col not in (0, w - 1):
            continue
        if rows[0] != 0 and rows[-1] != h - 1:
            continue
        candidates.append(comp)

    if not candidates:
        for comp in comps:
            rows = sorted(r for r, _ in comp)
            cols = {c for _, c in comp}
            if len(comp) >= 2 and len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
                candidates.append(comp)

    assert len(candidates) == 1, "expected a unique legend"
    legend_cells = sorted(candidates[0], key=lambda coord: coord[0])
    legend_colors = [int(grid[r, c]) for r, c in legend_cells]
    legend_set = set(legend_cells)

    cargos: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(colored):
        coord = (int(r), int(c))
        if coord in legend_set:
            continue
        color = int(grid[coord])
        assert color not in cargos, "expected unique cargo colors"
        cargos[color] = coord

    assert cargos, "expected at least one cargo"
    assert set(cargos) == set(legend_colors), "legend colors must match cargo colors"
    return legend_colors, cargos


def _cell_to_belts(belts: list[dict]) -> dict[tuple[int, int], list[int]]:
    mapping: dict[tuple[int, int], list[int]] = defaultdict(list)
    for belt_id, belt in enumerate(belts):
        for coord in belt["cells"]:
            mapping[coord].append(belt_id)
    for coord, owners in mapping.items():
        assert len(owners) <= 2, f"unsupported intersection degree at {coord}"
        if len(owners) == 2:
            orientations = {belts[belt_id]["orientation"] for belt_id in owners}
            assert orientations == {"h", "v"}, f"invalid intersection at {coord}"
    return mapping


def _step_from_intersection(belts: list[dict], belt_id: int, intersection: tuple[int, int]) -> tuple[int, int]:
    belt = belts[belt_id]
    idx = belt["index"][intersection]
    return belt["cells"][(idx + 1) % len(belt["cells"])]


def _backtrack_from_intersection(
    belts: list[dict], belt_id: int, intersection: tuple[int, int]
) -> tuple[int, int]:
    belt = belts[belt_id]
    idx = belt["index"][intersection]
    return belt["cells"][idx - 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each cargo one conveyor step, using the corner legend as the tie-breaker at crossings.

    Belts are straight horizontal/vertical gray segments with a two-pixel V marker at
    one end. The marker end is the wrap destination and the flow direction points away
    from that marker. Cargo colors are listed again in a vertical corner legend from
    highest to lowest priority.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    belt_color = _detect_belt_color(g, bg)
    legend_colors, cargos = _find_legend_and_cargos(g, bg, belt_color)
    path_mask, arrow_pixels = _split_belt_pixels(g, belt_color, set(cargos.values()))
    belts = _extract_belts(path_mask, arrow_pixels)
    priority = {color: rank for rank, color in enumerate(legend_colors)}

    cell_to_belts = _cell_to_belts(belts)
    intersections = {coord: owners for coord, owners in cell_to_belts.items() if len(owners) == 2}

    assigned_belts: dict[int, int] = {}
    for color, coord in cargos.items():
        owners = cell_to_belts.get(coord, [])
        assert len(owners) == 1, f"cargo at ambiguous or invalid cell {coord}"
        assigned_belts[color] = owners[0]

    final_positions: dict[int, tuple[int, int]] = {}
    approaches: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for color, coord in cargos.items():
        belt_id = assigned_belts[color]
        belt = belts[belt_id]
        idx = belt["index"][coord]
        step1 = belt["cells"][(idx + 1) % len(belt["cells"])]
        if step1 in intersections:
            approaches[step1].append({"color": color, "belt_id": belt_id})
        else:
            final_positions[color] = step1

    for intersection, entrants in approaches.items():
        owners = intersections[intersection]
        if len(entrants) == 1:
            entrant = entrants[0]
            final_positions[entrant["color"]] = _step_from_intersection(
                belts, entrant["belt_id"], intersection
            )
            continue

        assert len(entrants) == 2, f"unsupported simultaneous entry count at {intersection}"
        entrants.sort(key=lambda item: priority[item["color"]])
        winner, loser = entrants
        winner_pos = _step_from_intersection(
            belts, winner["belt_id"], intersection
        )
        final_positions[winner["color"]] = winner_pos
        loser_other_belt = owners[0] if owners[1] == loser["belt_id"] else owners[1]
        loser_pos = _step_from_intersection(belts, loser_other_belt, intersection)
        if loser_pos == winner_pos:
            loser_pos = _backtrack_from_intersection(belts, loser_other_belt, intersection)
        final_positions[loser["color"]] = loser_pos

    assert len(final_positions) == len(cargos)
    assert len(set(final_positions.values())) == len(final_positions), "cargoes collided after moving"

    out = g.copy()
    for coord in cargos.values():
        out[coord] = belt_color
    for color, coord in final_positions.items():
        out[coord] = color
    return out


def _sample_spaced_values(
    rng: np.random.Generator, lo: int, hi: int, count: int, min_gap: int
) -> list[int] | None:
    values = list(range(lo, hi + 1))
    rng.shuffle(values)
    chosen: list[int] = []
    for value in values:
        if all(abs(value - other) >= min_gap for other in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_generated_belt(
    orientation: str, fixed: int, lo: int, hi: int, direction: int
) -> dict:
    if orientation == "h":
        cells = [(fixed, c) for c in range(lo, hi + 1)]
        if direction == -1:
            cells.reverse()
        arrows = [(fixed - 1, lo - 1), (fixed + 1, lo - 1)] if direction == 1 else [
            (fixed - 1, hi + 1),
            (fixed + 1, hi + 1),
        ]
    else:
        cells = [(r, fixed) for r in range(lo, hi + 1)]
        if direction == -1:
            cells.reverse()
        arrows = [(lo - 1, fixed - 1), (lo - 1, fixed + 1)] if direction == 1 else [
            (hi + 1, fixed - 1),
            (hi + 1, fixed + 1),
        ]
    belt = _make_belt(cells, orientation, direction)
    belt["arrows"] = arrows
    return belt


def _legend_coords(h: int, w: int, size: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r, 0) for r in range(size)]
    if corner == "tr":
        return [(r, w - 1) for r in range(size)]
    if corner == "bl":
        return [(h - size + r, 0) for r in range(size)]
    assert corner == "br"
    return [(h - size + r, w - 1) for r in range(size)]


def _predecessor(belt: dict, coord: tuple[int, int]) -> tuple[int, int]:
    idx = belt["index"][coord]
    return belt["cells"][idx - 1]


def _successor(belt: dict, coord: tuple[int, int]) -> tuple[int, int]:
    idx = belt["index"][coord]
    return belt["cells"][(idx + 1) % len(belt["cells"])]


def _is_input_safe(
    coord: tuple[int, int],
    used: set[tuple[int, int]],
    legend_cells: set[tuple[int, int]],
) -> bool:
    if coord in used or coord in legend_cells:
        return False
    for rr, cc in used | legend_cells:
        if abs(coord[0] - rr) + abs(coord[1] - cc) == 1:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 23))
    w = int(rng.integers(16, 23))
    bg_choices = [color for color in PALETTE if color != BELT_COLOR]
    bg = int(rng.choice(bg_choices))

    rows = [int(rng.integers(5, h - 5))]
    cols = [int(rng.integers(5, w - 5))]
    left, right = 4, w - 5
    top, bottom = 4, h - 5
    belts: list[dict] = []
    for row in rows:
        direction = int(rng.choice([-1, 1]))
        belts.append(_build_generated_belt("h", row, left, right, direction))
    for col in cols:
        direction = int(rng.choice([-1, 1]))
        belts.append(_build_generated_belt("v", col, top, bottom, direction))

    path_cells = set()
    arrow_cells = set()
    for belt in belts:
        for cell in belt["cells"]:
            path_cells.add(cell)
        for arrow in belt["arrows"]:
            if arrow in path_cells or arrow in arrow_cells:
                return None
            arrow_cells.add(arrow)

    grid = np.full((h, w), bg, dtype=int)
    for r, c in path_cells | arrow_cells:
        grid[r, c] = BELT_COLOR

    legend_len = int(rng.integers(2, 5))
    color_pool = [color for color in PALETTE if color not in {bg, BELT_COLOR}]
    if len(color_pool) < legend_len:
        return None
    legend_colors = list(rng.choice(color_pool, size=legend_len, replace=False))
    corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    legend = _legend_coords(h, w, legend_len, corner)
    legend_set = set(legend)
    if any(coord in path_cells or coord in arrow_cells for coord in legend):
        return None
    for color, coord in zip(legend_colors, legend):
        grid[coord] = int(color)

    cell_to_belts = _cell_to_belts(belts)
    intersections = [coord for coord, owners in cell_to_belts.items() if len(owners) == 2]
    assert len(intersections) == 1
    intersection = intersections[0]
    owners = cell_to_belts[intersection]

    pred_map: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
    succ_map: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
    for belt_id, belt in enumerate(belts):
        pred_map[(belt_id, intersection)] = _predecessor(belt, intersection)
        succ_map[(belt_id, intersection)] = _successor(belt, intersection)

    regular_pool: list[tuple[int, int]] = []
    end_pool: list[tuple[int, int]] = []
    for belt in belts:
        end_pool.append(belt["cells"][-1])
        for coord in belt["cells"]:
            if coord in pred_map.values() or coord in succ_map.values() or coord == intersection:
                continue
            if coord == belt["cells"][-1]:
                continue
            regular_pool.append(coord)

    regular_pool = list(dict.fromkeys(regular_pool))
    end_pool = list(dict.fromkeys(end_pool))
    if not regular_pool or not end_pool:
        return None

    placements: dict[int, tuple[int, int]] = {}
    used_positions: set[tuple[int, int]] = set()
    forbidden_extra: set[tuple[int, int]] = {intersection}

    shuffled_colors = legend_colors.copy()
    rng.shuffle(shuffled_colors)

    if len(shuffled_colors) < 2:
        return None
    for color, belt_id in zip(shuffled_colors[:2], owners):
        pos = pred_map[(belt_id, intersection)]
        if not _is_input_safe(pos, used_positions, legend_set):
            return None
        placements[int(color)] = pos
        used_positions.add(pos)
    forbidden_extra.update(pred_map.values())
    forbidden_extra.update(succ_map.values())

    for color in legend_colors:
        if color in placements:
            continue
        choices = (regular_pool + end_pool).copy()
        rng.shuffle(choices)
        picked = None
        for coord in choices:
            if coord in forbidden_extra:
                continue
            if not _is_input_safe(coord, used_positions, legend_set):
                continue
            picked = coord
            break
        if picked is None:
            return None
        placements[color] = picked
        used_positions.add(picked)

    if set(placements) != set(legend_colors):
        return None

    for color, coord in placements.items():
        grid[coord] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == BELT_COLOR) != np.count_nonzero(grid == BELT_COLOR):
        return None
    for color in legend_colors:
        if int(np.count_nonzero(grid == color)) != 2:
            return None
        if int(np.count_nonzero(out == color)) != 2:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
