from __future__ import annotations

from collections import Counter
import random

import numpy as np


WALL = 0
OBSTACLE = 5
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(c for c in range(10) if c not in (WALL, OBSTACLE))


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    """
    The legend is a single vertical strip near the bottom-right corner.
    We accept the rightmost 1-cell-wide non-(bg,wall,obstacle) run that touches
    the bottom edge and has length 3 or 4.
    """
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    special = {int(bg), WALL, OBSTACLE}
    for c in range(w - 1, max(-1, w - 4), -1):
        rows = [r for r in range(h) if int(grid[r, c]) not in special]
        if not rows:
            continue
        r0 = rows[0]
        r1 = rows[-1]
        if rows != list(range(r0, r1 + 1)):
            continue
        if r1 != h - 1:
            continue
        if r1 - r0 + 1 not in (3, 4):
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for dc in (-1, 1):
                cc = c + dc
                if 0 <= cc < w and int(grid[r, cc]) not in special:
                    ok = False
        if ok:
            candidates.append((c, r0, r1))

    assert candidates, "expected a bottom-right vertical legend strip"
    c, r0, r1 = max(candidates)
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c] = True
    colors = [int(v) for v in grid[r0 : r1 + 1, c].tolist()]
    return colors, mask


def _lamp_positions(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[tuple[int, int]]:
    lamp_mask = (grid != int(bg)) & (grid != WALL) & (grid != OBSTACLE) & (~legend_mask)
    lamps = [tuple(map(int, rc)) for rc in np.argwhere(lamp_mask)]
    assert lamps, "expected at least one lamp"
    return lamps


def _illumination_counts(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    lamps: list[tuple[int, int]],
) -> np.ndarray:
    h, w = grid.shape
    blocked = (grid == WALL) | (grid == OBSTACLE) | legend_mask
    counts = np.zeros((h, w), dtype=int)
    for r, c in lamps:
        for dr, dc in DIRS:
            rr = r + dr
            cc = c + dc
            while 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc]:
                if int(grid[rr, cc]) == int(bg):
                    counts[rr, cc] += 1
                rr += dr
                cc += dc
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rooms are partitioned by black walls; each room has one colored lamp and
    optional grey obstacles. Lamps shine orthogonally through door gaps, but
    stop at walls or obstacles. Background cells are recolored from the vertical
    legend at bottom-right according to how many distinct lamps illuminate them.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend_colors, legend_mask = _find_legend(g, bg)
    lamps = _lamp_positions(g, bg, legend_mask)
    counts = _illumination_counts(g, bg, legend_mask, lamps)

    bg_mask = (g == int(bg)) & (~legend_mask)
    if np.any(bg_mask):
        max_count = int(counts[bg_mask].max())
    else:
        max_count = 0
    assert max_count < len(legend_colors), "legend must cover every illumination count"

    out = g.copy()
    for count, color in enumerate(legend_colors):
        out[bg_mask & (counts == count)] = int(color)
    out[legend_mask] = int(bg)
    return out


# --------------- generator ---------------

ROOM_SIZE_MIN = 4
ROOM_SIZE_MAX = 6
TOP_PAD_MAX = 2
LEFT_PAD_MAX = 2
BOTTOM_PAD_MAX = 2
RIGHT_GAP_MIN = 1
RIGHT_GAP_MAX = 2
EXTRA_DOOR_PROB = 0.25
EXTRA_OBSTACLE_PROB = 0.45
SECOND_OBSTACLE_PROB = 0.25
GEN_ATTEMPTS = 400


def _build_scene(row_sizes: list[int], col_sizes: list[int], bg: int) -> tuple[np.ndarray, list[dict]]:
    row_bounds: list[tuple[int, int]] = []
    col_bounds: list[tuple[int, int]] = []

    cursor = 1
    for size in row_sizes:
        r0 = cursor
        r1 = r0 + int(size) - 1
        row_bounds.append((r0, r1))
        cursor = r1 + 2
    scene_h = row_bounds[-1][1] + 2

    cursor = 1
    for size in col_sizes:
        c0 = cursor
        c1 = c0 + int(size) - 1
        col_bounds.append((c0, c1))
        cursor = c1 + 2
    scene_w = col_bounds[-1][1] + 2

    scene = np.full((scene_h, scene_w), int(bg), dtype=int)
    scene[0, :] = WALL
    scene[-1, :] = WALL
    scene[:, 0] = WALL
    scene[:, -1] = WALL
    for r0, _r1 in row_bounds[:-1]:
        wall_r = _r1 + 1
        scene[wall_r, :] = WALL
    for c0, _c1 in col_bounds[:-1]:
        wall_c = _c1 + 1
        scene[:, wall_c] = WALL

    rooms: list[dict] = []
    for i, (r0, r1) in enumerate(row_bounds):
        for j, (c0, c1) in enumerate(col_bounds):
            rooms.append(
                {
                    "row": int(i),
                    "col": int(j),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                }
            )
    return scene, rooms


def _room_map(rooms: list[dict]) -> dict[tuple[int, int], dict]:
    return {(room["row"], room["col"]): room for room in rooms}


def _global_from_local(room: dict, local_r: int, local_c: int) -> tuple[int, int]:
    return int(room["r0"] + local_r), int(room["c0"] + local_c)


def _adjacent_choices(index: int, count: int) -> list[int]:
    out: list[int] = []
    if index > 0:
        out.append(index - 1)
    if index + 1 < count:
        out.append(index + 1)
    return out


def _sample_layout(target_max: int, orientation: str) -> tuple[int, int]:
    if target_max == 2:
        if orientation == "horizontal":
            return random.choice([1, 2, 3]), random.choice([2, 3])
        return random.choice([2, 3]), random.choice([1, 2, 3])

    if orientation == "horizontal":
        return random.choice([2, 3]), 3
    return 3, random.choice([2, 3])


def _choose_special_lamps(
    rooms_by_key: dict[tuple[int, int], dict],
    nr: int,
    nc: int,
    target_max: int,
    orientation: str,
) -> tuple[dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]], dict]:
    lamps: dict[tuple[int, int], tuple[int, int]] = {}
    doors: set[tuple[int, int]] = set()
    meta: dict[str, object] = {"orientation": orientation, "target_max": int(target_max)}

    if orientation == "horizontal":
        beam_len = int(target_max)
        beam_row = random.randrange(nr)
        beam_start = random.randint(0, nc - beam_len)
        beam_cols = list(range(beam_start, beam_start + beam_len))
        beam_room = rooms_by_key[(beam_row, beam_cols[0])]
        beam_local_row = random.randint(1, beam_room["height"] - 2)
        beam_global_row = beam_room["r0"] + beam_local_row
        helper_col = random.choice(beam_cols)
        helper_local_col = random.randint(1, rooms_by_key[(beam_row, helper_col)]["width"] - 2)

        for j in beam_cols:
            room = rooms_by_key[(beam_row, j)]
            if j == helper_col:
                local_c = helper_local_col
            else:
                local_c = random.randint(1, room["width"] - 2)
            lamps[(beam_row, j)] = _global_from_local(room, beam_local_row, local_c)
        for left, right in zip(beam_cols, beam_cols[1:], strict=False):
            wall_c = rooms_by_key[(beam_row, left)]["c1"] + 1
            doors.add((beam_global_row, wall_c))

        meta.update(
            {
                "beam_row": int(beam_row),
                "beam_cols": tuple(int(v) for v in beam_cols),
                "beam_local_row": int(beam_local_row),
                "helper_col": int(helper_col),
                "helper_local_col": int(helper_local_col),
            }
        )

        if target_max == 3:
            helper_row = random.choice(_adjacent_choices(beam_row, nr))
            helper_room = rooms_by_key[(helper_row, helper_col)]
            helper_local_row = random.randint(1, helper_room["height"] - 2)
            lamps[(helper_row, helper_col)] = _global_from_local(helper_room, helper_local_row, helper_local_col)
            wall_r = min(beam_row, helper_row)
            wall_row = rooms_by_key[(wall_r, helper_col)]["r1"] + 1
            helper_global_col = rooms_by_key[(beam_row, helper_col)]["c0"] + helper_local_col
            doors.add((wall_row, helper_global_col))
            meta.update({"helper_row": int(helper_row), "helper_local_row": int(helper_local_row)})
    else:
        beam_len = int(target_max)
        beam_col = random.randrange(nc)
        beam_start = random.randint(0, nr - beam_len)
        beam_rows = list(range(beam_start, beam_start + beam_len))
        beam_room = rooms_by_key[(beam_rows[0], beam_col)]
        beam_local_col = random.randint(1, beam_room["width"] - 2)
        beam_global_col = beam_room["c0"] + beam_local_col
        helper_row = random.choice(beam_rows)
        helper_local_row = random.randint(1, rooms_by_key[(helper_row, beam_col)]["height"] - 2)

        for i in beam_rows:
            room = rooms_by_key[(i, beam_col)]
            if i == helper_row:
                local_r = helper_local_row
            else:
                local_r = random.randint(1, room["height"] - 2)
            lamps[(i, beam_col)] = _global_from_local(room, local_r, beam_local_col)
        for top, bottom in zip(beam_rows, beam_rows[1:], strict=False):
            wall_r = rooms_by_key[(top, beam_col)]["r1"] + 1
            doors.add((wall_r, beam_global_col))

        meta.update(
            {
                "beam_col": int(beam_col),
                "beam_rows": tuple(int(v) for v in beam_rows),
                "beam_local_col": int(beam_local_col),
                "helper_row": int(helper_row),
                "helper_local_row": int(helper_local_row),
            }
        )

        if target_max == 3:
            helper_col = random.choice(_adjacent_choices(beam_col, nc))
            helper_room = rooms_by_key[(helper_row, helper_col)]
            helper_local_col = random.randint(1, helper_room["width"] - 2)
            lamps[(helper_row, helper_col)] = _global_from_local(helper_room, helper_local_row, helper_local_col)
            wall_c = min(beam_col, helper_col)
            wall_col = rooms_by_key[(helper_row, wall_c)]["c1"] + 1
            helper_global_row = rooms_by_key[(helper_row, beam_col)]["r0"] + helper_local_row
            doors.add((helper_global_row, wall_col))
            meta.update({"helper_col": int(helper_col), "helper_local_col": int(helper_local_col)})

    return lamps, doors, meta


def _candidate_cells(room: dict) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(room["r0"], room["r1"] + 1)
        for c in range(room["c0"], room["c1"] + 1)
    ]


def _choose_other_lamps(
    rooms: list[dict],
    prechosen: dict[tuple[int, int], tuple[int, int]],
    meta: dict,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    lamps = dict(prechosen)
    orientation = str(meta["orientation"])
    beam_row = meta.get("beam_row")
    beam_cols = set(meta.get("beam_cols", ()))
    helper_col = meta.get("helper_col")
    helper_row = meta.get("helper_row")
    beam_col = meta.get("beam_col")
    beam_rows = set(meta.get("beam_rows", ()))

    for room in rooms:
        key = (room["row"], room["col"])
        if key in lamps:
            continue
        candidates = _candidate_cells(room)
        filtered: list[tuple[int, int]] = []
        for r, c in candidates:
            ok = True
            if orientation == "horizontal":
                if room["row"] == beam_row and room["col"] not in beam_cols:
                    if r == lamps[(int(beam_row), int(next(iter(beam_cols))))][0]:
                        ok = False
                if helper_col is not None and room["col"] == helper_col and room["row"] not in {beam_row, helper_row}:
                    helper_global_col = prechosen[(int(beam_row), int(helper_col))][1]
                    if c == helper_global_col:
                        ok = False
            else:
                if room["col"] == beam_col and room["row"] not in beam_rows:
                    if c == lamps[(int(next(iter(beam_rows))), int(beam_col))][1]:
                        ok = False
                if helper_row is not None and room["row"] == helper_row and room["col"] not in {beam_col, meta.get("helper_col")}:
                    helper_global_row = prechosen[(int(helper_row), int(beam_col))][0]
                    if r == helper_global_row:
                        ok = False
            if ok:
                filtered.append((r, c))
        if not filtered:
            return None
        lamps[key] = random.choice(filtered)
    return lamps


def _add_extra_doors(
    scene: np.ndarray,
    rooms_by_key: dict[tuple[int, int], dict],
    nr: int,
    nc: int,
    required_doors: set[tuple[int, int]],
    bg: int,
) -> None:
    for i in range(nr):
        for j in range(nc - 1):
            wall_c = rooms_by_key[(i, j)]["c1"] + 1
            candidates = [(r, wall_c) for r in range(rooms_by_key[(i, j)]["r0"], rooms_by_key[(i, j)]["r1"] + 1)]
            if any(cell in required_doors for cell in candidates):
                continue
            if random.random() < EXTRA_DOOR_PROB:
                scene[random.choice(candidates)] = int(bg)

    for i in range(nr - 1):
        for j in range(nc):
            wall_r = rooms_by_key[(i, j)]["r1"] + 1
            candidates = [(wall_r, c) for c in range(rooms_by_key[(i, j)]["c0"], rooms_by_key[(i, j)]["c1"] + 1)]
            if any(cell in required_doors for cell in candidates):
                continue
            if random.random() < EXTRA_DOOR_PROB:
                scene[random.choice(candidates)] = int(bg)


def _place_obstacles(
    scene: np.ndarray,
    rooms: list[dict],
    lamps: dict[tuple[int, int], tuple[int, int]],
    meta: dict,
    bg: int,
) -> bool:
    orientation = str(meta["orientation"])
    beam_row = meta.get("beam_row")
    beam_cols = set(meta.get("beam_cols", ()))
    beam_col = meta.get("beam_col")
    beam_rows = set(meta.get("beam_rows", ()))
    helper_row = meta.get("helper_row")
    helper_col = meta.get("helper_col")
    placed_any = False

    def allowed(room: dict, lamp: tuple[int, int], r: int, c: int) -> bool:
        if (r, c) == lamp:
            return False
        if int(scene[r, c]) != int(bg):
            return False
        if orientation == "horizontal":
            if room["row"] == beam_row and room["col"] in beam_cols and r == lamp[0]:
                return False
            if helper_row is not None and room["row"] == helper_row and room["col"] == helper_col and c == lamp[1]:
                return False
        else:
            if room["col"] == beam_col and room["row"] in beam_rows and c == lamp[1]:
                return False
            if helper_row is not None and room["row"] == helper_row and room["col"] == helper_col and r == lamp[0]:
                return False
        return True

    special_room_key: tuple[int, int]
    if orientation == "horizontal":
        special_room_key = (int(meta["beam_row"]), int(meta["beam_cols"][0]))
    else:
        special_room_key = (int(meta["beam_rows"][0]), int(meta["beam_col"]))

    for room in rooms:
        key = (room["row"], room["col"])
        lamp = lamps[key]
        room_cells = _candidate_cells(room)
        obstacles: list[tuple[int, int]] = []

        if key == special_room_key:
            if orientation == "horizontal":
                if lamp[0] + 2 <= room["r1"]:
                    candidate = (lamp[0] + 1, lamp[1])
                elif lamp[0] - 2 >= room["r0"]:
                    candidate = (lamp[0] - 1, lamp[1])
                else:
                    candidate = None
            else:
                if lamp[1] + 2 <= room["c1"]:
                    candidate = (lamp[0], lamp[1] + 1)
                elif lamp[1] - 2 >= room["c0"]:
                    candidate = (lamp[0], lamp[1] - 1)
                else:
                    candidate = None
            if candidate is not None and allowed(room, lamp, *candidate):
                obstacles.append(candidate)

        if random.random() < EXTRA_OBSTACLE_PROB:
            random.shuffle(room_cells)
            for cell in room_cells:
                if allowed(room, lamp, *cell) and cell not in obstacles:
                    obstacles.append(cell)
                    break
        if random.random() < SECOND_OBSTACLE_PROB:
            random.shuffle(room_cells)
            for cell in room_cells:
                if allowed(room, lamp, *cell) and cell not in obstacles:
                    obstacles.append(cell)
                    break

        for r, c in obstacles:
            scene[r, c] = OBSTACLE
            placed_any = True

    return placed_any


def _render_input(
    bg: int,
    row_sizes: list[int],
    col_sizes: list[int],
    lamps: dict[tuple[int, int], tuple[int, int]],
    required_doors: set[tuple[int, int]],
    legend_colors: list[int],
    lamp_palette: list[int],
    meta: dict,
) -> np.ndarray | None:
    scene, rooms = _build_scene(row_sizes, col_sizes, bg)
    rooms_by_key = _room_map(rooms)
    nr = len(row_sizes)
    nc = len(col_sizes)

    for r, c in required_doors:
        scene[r, c] = bg
    _add_extra_doors(scene, rooms_by_key, nr, nc, required_doors, bg)

    if not _place_obstacles(scene, rooms, lamps, meta, bg):
        return None

    for room in rooms:
        key = (room["row"], room["col"])
        lamp_r, lamp_c = lamps[key]
        scene[lamp_r, lamp_c] = random.choice(lamp_palette)

    top_pad = random.randint(0, TOP_PAD_MAX)
    left_pad = random.randint(0, LEFT_PAD_MAX)
    bottom_pad = random.randint(0, BOTTOM_PAD_MAX)
    right_gap = random.randint(RIGHT_GAP_MIN, RIGHT_GAP_MAX)
    h = top_pad + scene.shape[0] + bottom_pad
    w = left_pad + scene.shape[1] + right_gap + 1
    grid = np.full((h, w), int(bg), dtype=int)
    grid[top_pad : top_pad + scene.shape[0], left_pad : left_pad + scene.shape[1]] = scene

    legend_len = len(legend_colors)
    legend_col = w - 1
    start_row = h - legend_len
    grid[start_row:, legend_col] = np.array(legend_colors, dtype=int)
    return grid


def _choose_colors(target_max: int) -> tuple[int, list[int], list[int]]:
    bg = random.choice(PALETTE)
    rest = [c for c in PALETTE if c != bg]
    random.shuffle(rest)
    legend = [int(v) for v in rest[: target_max + 1]]
    lamp_palette = [int(v) for v in rest[target_max + 1 :]]
    assert lamp_palette, "need at least one lamp color"
    return int(bg), legend, lamp_palette


def _valid_counts(inp: np.ndarray, target_max: int) -> bool:
    bg = _most_common_color(inp)
    legend_colors, legend_mask = _find_legend(inp, bg)
    if len(legend_colors) != target_max + 1:
        return False
    lamps = _lamp_positions(inp, bg, legend_mask)
    counts = _illumination_counts(inp, bg, legend_mask, lamps)
    bg_mask = (inp == bg) & (~legend_mask)
    present = set(int(v) for v in counts[bg_mask].tolist())
    return present == set(range(target_max + 1)) and int(counts[bg_mask].max()) == target_max


def _gen_attempt() -> np.ndarray | None:
    target_max = random.choice([2, 3])
    orientation = random.choice(["horizontal", "vertical"])
    nr, nc = _sample_layout(target_max, orientation)
    row_sizes = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(nr)]
    col_sizes = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(nc)]
    bg, legend_colors, lamp_palette = _choose_colors(target_max)

    scene, rooms = _build_scene(row_sizes, col_sizes, bg)
    rooms_by_key = _room_map(rooms)
    special_lamps, required_doors, meta = _choose_special_lamps(rooms_by_key, nr, nc, target_max, orientation)
    all_lamps = _choose_other_lamps(rooms, special_lamps, meta)
    if all_lamps is None:
        return None

    inp = _render_input(
        bg=bg,
        row_sizes=row_sizes,
        col_sizes=col_sizes,
        lamps=all_lamps,
        required_doors=required_doors,
        legend_colors=legend_colors,
        lamp_palette=lamp_palette,
        meta=meta,
    )
    if inp is None:
        return None

    counts = Counter(int(v) for v in inp.ravel())
    if counts[bg] != max(counts.values()):
        return None
    if list(counts.values()).count(counts[bg]) != 1:
        return None
    if not _valid_counts(inp, target_max):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    bg2 = _most_common_color(inp)
    legend_colors2, legend_mask = _find_legend(inp, bg2)
    if not np.all(out[legend_mask] == bg2):
        return None
    if len(legend_colors2) != target_max + 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lighting-room instance")
