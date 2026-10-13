from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
REGION_COLORS = (BLUE, RED, GREEN)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_RANGE = (18, 22)
REGION_SIDE_RANGE = (3, 4)
MIN_GAP = 3
MIN_MARGIN = 1
GEN_ATTEMPTS = 600

_GEN_COUNTER = 0
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing region color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _region_infos(grid: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for color in REGION_COLORS:
        r0, c0, r1, c1 = _bbox_of_color(grid, color)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each region must be a solid monochrome rectangle"
        infos.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "mask": grid == color,
            }
        )

    for i in range(len(infos)):
        r0a, c0a, r1a, c1a = infos[i]["bbox"]
        for j in range(i + 1, len(infos)):
            r0b, c0b, r1b, c1b = infos[j]["bbox"]
            assert r1a < r0b or r1b < r0a, "regions must not share rows"
            assert c1a < c0b or c1b < c0a, "regions must not share columns"
    return infos


def _trace_ray(
    grid: np.ndarray, start: tuple[int, int], dr: int, dc: int
) -> tuple[list[tuple[int, int]], tuple[int, int] | None, int | None]:
    r, c = start
    rr, cc = r + dr, c + dc
    path: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, rr, cc):
        value = int(grid[rr, cc])
        if value != BG:
            return path, (rr, cc), value
        path.append((rr, cc))
        rr += dr
        cc += dc
    return path, None, None


def _vote_data(grid: np.ndarray) -> tuple[dict[int, int], dict[tuple[int, int], int], dict]:
    votes = {color: 0 for color in REGION_COLORS}
    marker_votes: dict[tuple[int, int], int] = {}
    ray_details: dict[tuple[int, int], dict[tuple[int, int], dict[str, object]]] = {}

    markers = [tuple(map(int, pos)) for pos in np.argwhere(grid == ORANGE)]
    for marker in markers:
        marker_votes[marker] = 0
        ray_details[marker] = {}
        for dr, dc in DIRS:
            path, stop, stop_color = _trace_ray(grid, marker, dr, dc)
            ray_details[marker][(dr, dc)] = {
                "path": path,
                "stop": stop,
                "stop_color": stop_color,
            }
            if stop_color in REGION_COLORS:
                votes[int(stop_color)] += 1
                marker_votes[marker] += 1
    return votes, marker_votes, ray_details


def _winner_region(
    votes: dict[int, int], infos: list[dict[str, object]]
) -> dict[str, object]:
    return min(
        infos,
        key=lambda info: (
            -votes[int(info["color"])],
            int(info["bbox"][0]),
            int(info["bbox"][1]),
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the rectangular region that wins the obstacle-filtered four-direction ray vote.

    Input structure:
    - The grid has black background, exactly three solid rectangular color regions
      in blue, red, and green, plus 2-4 isolated orange marker pixels.
    - Grey pixels form fixed obstacles as isolated cells or short orthogonal segments.
    - Any two colored regions occupy disjoint row ranges and disjoint column ranges,
      so a marker can only align horizontally with one region and vertically with one region.

    Rule:
    - From each orange marker, cast rays up, down, left, and right through background cells.
    - A ray stops when it first reaches a grey obstacle, a colored region, another marker,
      or the grid edge; only rays whose first hit is a blue/red/green region cast a vote.
    - Sum all votes by region, break ties by the smaller top-left bounding-box coordinate,
      and recolor the winning region yellow while leaving everything else unchanged.

    Key insight:
    - The winner is determined by which aligned rays survive the obstacle layout, not by
      simple proximity: barriers remove specific row/column contacts before the votes are tallied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = _region_infos(g)
    votes, _marker_votes, _details = _vote_data(g)
    winner = _winner_region(votes, infos)

    out = g.copy()
    out[winner["mask"]] = YELLOW
    return out


def _distribute_extra(extra: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(extra):
        parts[random.randrange(slots)] += 1
    return parts


def _sample_band_starts(total: int, sizes: list[int]) -> list[int] | None:
    base = sum(sizes) + 2 * MIN_GAP + 2 * MIN_MARGIN
    if base > total:
        return None
    top_extra, gap1_extra, gap2_extra, bottom_extra = _distribute_extra(total - base, 4)
    top = MIN_MARGIN + top_extra
    gap1 = MIN_GAP + gap1_extra
    gap2 = MIN_GAP + gap2_extra
    _bottom = MIN_MARGIN + bottom_extra
    starts = [top]
    starts.append(starts[-1] + sizes[0] + gap1)
    starts.append(starts[-1] + sizes[1] + gap2)
    return starts


def _ordered_boxes() -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    row_sizes = [random.randint(*REGION_SIDE_RANGE) for _ in range(3)]
    col_sizes = [random.randint(*REGION_SIDE_RANGE) for _ in range(3)]
    row_starts = _sample_band_starts(height, row_sizes)
    col_starts = _sample_band_starts(width, col_sizes)
    if row_starts is None or col_starts is None:
        return None

    col_perm = random.sample(range(3), 3)
    boxes: list[tuple[int, int, int, int]] = []
    for slot in range(3):
        r0 = row_starts[slot]
        c_band = col_perm[slot]
        c0 = col_starts[c_band]
        boxes.append(
            (
                r0,
                c0,
                r0 + row_sizes[slot] - 1,
                c0 + col_sizes[c_band] - 1,
            )
        )
    random.shuffle(boxes)
    return height, width, boxes


def _slot_winner(counts: list[int], boxes: list[tuple[int, int, int, int]]) -> int:
    return min(range(3), key=lambda idx: (-counts[idx], boxes[idx][0], boxes[idx][1]))


def _search_plan(
    boxes: list[tuple[int, int, int, int]], desired_slot: int, preferred_markers: int
) -> list[dict[str, object]] | None:
    base_pairs = [(r_slot, c_slot) for r_slot in range(3) for c_slot in range(3) if r_slot != c_slot]
    marker_orders = [preferred_markers] + [n for n in (2, 3, 4) if n != preferred_markers]
    modes = ("both", "row_only", "col_only")

    for marker_count in marker_orders:
        for _ in range(5000):
            chosen_pairs = random.sample(base_pairs, marker_count)
            chosen_modes = [random.choice(modes) for _ in range(marker_count)]
            if not any(mode == "both" for mode in chosen_modes):
                continue
            if not any(mode != "both" for mode in chosen_modes):
                continue

            counts = [0, 0, 0]
            free_counts = [0, 0, 0]
            specs: list[dict[str, object]] = []
            for (row_slot, col_slot), mode in zip(chosen_pairs, chosen_modes):
                free_counts[row_slot] += 1
                free_counts[col_slot] += 1
                if mode in ("both", "row_only"):
                    counts[row_slot] += 1
                if mode in ("both", "col_only"):
                    counts[col_slot] += 1
                specs.append(
                    {
                        "row_slot": row_slot,
                        "col_slot": col_slot,
                        "mode": mode,
                    }
                )

            if len(set(counts)) != 3:
                continue
            if _slot_winner(counts, boxes) != desired_slot:
                continue
            if _slot_winner(free_counts, boxes) == desired_slot:
                continue
            return specs
    return None


def _assign_positions(
    specs: list[dict[str, object]], boxes: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]] | None:
    rows_by_slot: dict[int, list[int]] = {}
    cols_by_slot: dict[int, list[int]] = {}

    for slot in range(3):
        row_need = sum(1 for spec in specs if spec["row_slot"] == slot)
        col_need = sum(1 for spec in specs if spec["col_slot"] == slot)
        r0, c0, r1, c1 = boxes[slot]
        row_choices = list(range(r0, r1 + 1))
        col_choices = list(range(c0, c1 + 1))
        if row_need > len(row_choices) or col_need > len(col_choices):
            return None
        rows_by_slot[slot] = random.sample(row_choices, row_need)
        cols_by_slot[slot] = random.sample(col_choices, col_need)

    positions: list[tuple[int, int]] = []
    for spec in specs:
        row = rows_by_slot[int(spec["row_slot"])].pop()
        col = cols_by_slot[int(spec["col_slot"])].pop()
        positions.append((row, col))
    return positions


def _region_path(
    box: tuple[int, int, int, int], marker: tuple[int, int], axis: str
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r, c = marker
    r0, c0, r1, c1 = box
    if axis == "row":
        if c < c0:
            return [(r, cc) for cc in range(c + 1, c0)], (r, c0)
        return [(r, cc) for cc in range(c - 1, c1, -1)], (r, c1)
    if r < r0:
        return [(rr, c) for rr in range(r + 1, r0)], (r0, c)
    return [(rr, c) for rr in range(r - 1, r1, -1)], (r1, c)


def _safe_to_place_barrier(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = cell
    if grid[r, c] != BG:
        return False
    for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
        for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
            if grid[rr, cc] == ORANGE:
                return False
    return True


def _place_core_barriers(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    positions: list[tuple[int, int]],
    boxes: list[tuple[int, int, int, int]],
) -> list[dict[str, object]] | None:
    path_info: list[dict[str, object]] = []
    for spec, marker in zip(specs, positions):
        row_box = boxes[int(spec["row_slot"])]
        col_box = boxes[int(spec["col_slot"])]
        row_path, _row_stop = _region_path(row_box, marker, "row")
        col_path, _col_stop = _region_path(col_box, marker, "col")
        if len(row_path) < 2 or len(col_path) < 2:
            return None

        blocked_axis = None
        if spec["mode"] == "row_only":
            blocked_axis = "col"
        elif spec["mode"] == "col_only":
            blocked_axis = "row"

        blocked_length = None
        kept_length = None
        if blocked_axis == "row":
            candidates = row_path[1:]
            if not candidates:
                return None
            cell = random.choice(candidates)
            if not _safe_to_place_barrier(grid, cell):
                return None
            grid[cell] = GREY
            blocked_length = len(row_path)
            kept_length = len(col_path)
        elif blocked_axis == "col":
            candidates = col_path[1:]
            if not candidates:
                return None
            cell = random.choice(candidates)
            if not _safe_to_place_barrier(grid, cell):
                return None
            grid[cell] = GREY
            blocked_length = len(col_path)
            kept_length = len(row_path)

        path_info.append(
            {
                "marker": marker,
                "row_length": len(row_path),
                "col_length": len(col_path),
                "blocked_axis": blocked_axis,
                "blocked_length": blocked_length,
                "kept_length": kept_length,
            }
        )
    return path_info


def _try_place_decoy_segment(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    region_rows = set()
    region_cols = set()
    for r0, c0, r1, c1 in boxes:
        region_rows.update(range(r0, r1 + 1))
        region_cols.update(range(c0, c1 + 1))
    marker_rows = {int(r) for r, _ in np.argwhere(grid == ORANGE)}
    marker_cols = {int(c) for _, c in np.argwhere(grid == ORANGE)}

    neutral_rows = [r for r in range(grid.shape[0]) if r not in region_rows and r not in marker_rows]
    neutral_cols = [c for c in range(grid.shape[1]) if c not in region_cols and c not in marker_cols]
    attempts = []
    if neutral_rows:
        attempts.append("h")
    if neutral_cols:
        attempts.append("v")
    if not attempts:
        return False

    for _ in range(40):
        orient = random.choice(attempts)
        seg_len = random.randint(1, 3)
        if orient == "h":
            r = random.choice(neutral_rows)
            if seg_len > grid.shape[1]:
                continue
            c0 = random.randint(0, grid.shape[1] - seg_len)
            cells = [(r, c0 + offset) for offset in range(seg_len)]
        else:
            c = random.choice(neutral_cols)
            if seg_len > grid.shape[0]:
                continue
            r0 = random.randint(0, grid.shape[0] - seg_len)
            cells = [(r0 + offset, c) for offset in range(seg_len)]

        if any(not _safe_to_place_barrier(grid, cell) for cell in cells):
            continue
        if any(grid[cell] != BG for cell in cells):
            continue
        touches_existing = False
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == GREY and (rr, cc) not in cells:
                    touches_existing = True
                    break
            if touches_existing:
                break
        if touches_existing:
            continue
        for cell in cells:
            grid[cell] = GREY
        return True
    return False


def _barrier_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == GREY):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in DIRS:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and grid[nr, nc] == GREY:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append(comp)
    return components


def _valid_barriers(grid: np.ndarray) -> bool:
    for comp in _barrier_components(grid):
        if not 1 <= len(comp) <= 3:
            return False
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) != 1 and len(cols) != 1:
            return False
        ordered = sorted(comp)
        if len(rows) == 1:
            cs = sorted(c for _, c in comp)
            if cs != list(range(cs[0], cs[-1] + 1)):
                return False
        else:
            rs = sorted(r for r, _ in comp)
            if rs != list(range(rs[0], rs[-1] + 1)):
                return False
    return True


def _markers_are_isolated(grid: np.ndarray) -> bool:
    markers = [tuple(map(int, pos)) for pos in np.argwhere(grid == ORANGE)]
    if not (2 <= len(markers) <= 4):
        return False
    for r, c in markers:
        for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
                if (rr, cc) == (r, c):
                    continue
                if grid[rr, cc] != BG:
                    return False
    return True


def _semantic_checks(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    path_info: list[dict[str, object]],
) -> bool:
    if not _markers_are_isolated(grid):
        return False
    if not _valid_barriers(grid):
        return False

    infos = _region_infos(grid)
    votes, marker_votes, details = _vote_data(grid)
    if any(count == 0 for count in marker_votes.values()):
        return False
    counts = [votes[color] for color in REGION_COLORS]
    if len(set(counts)) != 3:
        return False

    barrier_free = grid.copy()
    barrier_free[barrier_free == GREY] = BG
    free_votes, _free_marker_votes, free_details = _vote_data(barrier_free)
    if _winner_region(votes, infos)["color"] == _winner_region(free_votes, infos)["color"]:
        return False

    blocked_some_region = False
    blocked_shorter_than_open = False
    for info in path_info:
        marker = tuple(info["marker"])
        if info["blocked_axis"] is None:
            continue
        blocked_some_region = True
        if info["blocked_length"] is not None and info["kept_length"] is not None:
            if int(info["blocked_length"]) < int(info["kept_length"]):
                blocked_shorter_than_open = True
        actual_stops = details[marker]
        free_stops = free_details[marker]
        if info["blocked_axis"] == "row":
            row_dirs = [(0, -1), (0, 1)]
            if not any(
                free_stops[d]["stop_color"] in REGION_COLORS and actual_stops[d]["stop_color"] == GREY
                for d in row_dirs
            ):
                return False
        else:
            col_dirs = [(-1, 0), (1, 0)]
            if not any(
                free_stops[d]["stop_color"] in REGION_COLORS and actual_stops[d]["stop_color"] == GREY
                for d in col_dirs
            ):
                return False
    if not blocked_some_region or not blocked_shorter_than_open:
        return False

    out = main(grid)
    recolored = [int(np.sum(out == YELLOW))]
    if recolored[0] == 0:
        return False

    out_sig = (tuple(out.shape), out.tobytes())
    if out_sig in _SEEN_OUTPUTS:
        return False
    _SEEN_OUTPUTS.add(out_sig)
    return True


def _attempt_generate(desired_slot: int, preferred_markers: int) -> np.ndarray | None:
    layout = _ordered_boxes()
    if layout is None:
        return None
    height, width, boxes = layout

    specs = _search_plan(boxes, desired_slot, preferred_markers)
    if specs is None:
        return None
    positions = _assign_positions(specs, boxes)
    if positions is None:
        return None

    colors_by_slot = random.sample(list(REGION_COLORS), 3)
    grid = np.zeros((height, width), dtype=int)
    for slot, (r0, c0, r1, c1) in enumerate(boxes):
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors_by_slot[slot]
    for r, c in positions:
        grid[r, c] = ORANGE

    path_info = _place_core_barriers(grid, specs, positions, boxes)
    if path_info is None:
        return None

    for _ in range(random.randint(1, 3)):
        _try_place_decoy_segment(grid, boxes)

    if not _semantic_checks(grid, boxes, path_info):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    preferred_markers = [2, 3, 4][_GEN_COUNTER % 3]
    desired_slot = _GEN_COUNTER % 3
    _GEN_COUNTER += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(desired_slot, preferred_markers)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-ray voting grid")
