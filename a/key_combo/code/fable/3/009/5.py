from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]

BG = 0
GRAY = 5

TOP = (-1, 0)
BOTTOM = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [TOP, BOTTOM, LEFT, RIGHT]
SIDE_NAMES = ("top", "bottom", "left", "right")

PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 900


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _solid_side_color(values: np.ndarray, bg: int) -> int | None:
    if values.size == 0:
        return None
    color = int(values.flat[0])
    if color in (int(bg), GRAY):
        return None
    if not bool(np.all(values == color)):
        return None
    return color


def _bead_columns(row: np.ndarray, side_colors: set[int], bg: int) -> list[int] | None:
    allowed = np.isin(row, list(side_colors))
    if bool(np.any((row != int(bg)) & ~allowed)):
        return None
    cols = [int(c) for c in np.flatnonzero(allowed)]
    if not cols:
        return None
    if cols[-1] - cols[0] + 1 != len(cols):
        return None
    return cols


def _find_frame(g: np.ndarray, bg: int) -> dict:
    height, width = map(int, g.shape)
    candidates = []

    for r0 in range(2, height - 3):
        if not bool(np.all(g[r0 - 1, :] == int(bg))):
            continue
        bead_r = r0 - 2
        for r1 in range(r0 + 3, height):
            for c0 in range(0, width - 3):
                for c1 in range(c0 + 3, width):
                    if (
                        int(g[r0, c0]) != int(bg)
                        or int(g[r0, c1]) != int(bg)
                        or int(g[r1, c0]) != int(bg)
                        or int(g[r1, c1]) != int(bg)
                    ):
                        continue

                    top = _solid_side_color(g[r0, c0 + 1 : c1], bg)
                    bottom = _solid_side_color(g[r1, c0 + 1 : c1], bg)
                    left = _solid_side_color(g[r0 + 1 : r1, c0], bg)
                    right = _solid_side_color(g[r0 + 1 : r1, c1], bg)
                    if None in (top, bottom, left, right):
                        continue

                    side_colors = {
                        int(top),
                        int(bottom),
                        int(left),
                        int(right),
                    }
                    if len(side_colors) != 4:
                        continue
                    beads = _bead_columns(g[bead_r, :], side_colors, bg)
                    if beads is None:
                        continue

                    candidates.append(
                        {
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "bead_r": int(bead_r),
                            "bead_cols": beads,
                            "colors": {
                                TOP: int(top),
                                BOTTOM: int(bottom),
                                LEFT: int(left),
                                RIGHT: int(right),
                            },
                            "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                        }
                    )

    assert candidates, "expected one colored rectangular frame with a bead row above it"
    candidates.sort(key=lambda item: int(item["area"]), reverse=True)
    best = candidates[0]
    assert len(candidates) == 1 or int(best["area"]) > int(candidates[1]["area"])
    return best


def _slide(
    start: Coord,
    direction: Coord,
    field_h: int,
    field_w: int,
    obstacles: set[Coord],
) -> tuple[Coord, int, str]:
    r, c = map(int, start)
    dr, dc = map(int, direction)
    distance = 0

    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < int(field_h) and 0 <= nc < int(field_w)):
            return (int(r), int(c)), int(distance), "border"
        if (int(nr), int(nc)) in obstacles:
            return (int(r), int(c)), int(distance), "obstacle"
        r, c = nr, nc
        distance += 1


def _run_program(
    start: Coord,
    program: list[Coord],
    field_h: int,
    field_w: int,
    obstacles: set[Coord],
) -> tuple[Coord, list[dict]]:
    pos = (int(start[0]), int(start[1]))
    steps = []
    for direction in program:
        nxt, distance, blocker = _slide(pos, direction, field_h, field_w, obstacles)
        steps.append(
            {
                "start": pos,
                "direction": direction,
                "end": nxt,
                "distance": int(distance),
                "blocker": blocker,
            }
        )
        pos = nxt
    return pos, steps


def _run_one_step(
    start: Coord,
    program: list[Coord],
    field_h: int,
    field_w: int,
    obstacles: set[Coord],
) -> Coord:
    r, c = map(int, start)
    for dr, dc in program:
        nr = r + int(dr)
        nc = c + int(dc)
        if 0 <= nr < int(field_h) and 0 <= nc < int(field_w) and (nr, nc) not in obstacles:
            r, c = nr, nc
    return int(r), int(c)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    frame = _find_frame(g, bg)
    r0, c0, r1, c1 = frame["bbox"]
    side_by_direction = dict(frame["colors"])
    side_colors = set(side_by_direction.values())
    color_to_direction = {color: direction for direction, color in side_by_direction.items()}

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert not bool(np.any(np.isin(interior, list(side_colors)))), "side colors are reserved for the frame and beads"

    sprite_mask = (interior != int(bg)) & (interior != GRAY)
    sprite_cells = np.argwhere(sprite_mask)
    assert sprite_cells.shape[0] == 1, "expected exactly one sprite cell inside the frame"
    sr, sc = map(int, sprite_cells[0])
    sprite_color = int(interior[sr, sc])

    obstacles = {
        (int(r), int(c))
        for r, c in np.argwhere(interior == GRAY)
    }
    bead_values = [int(g[frame["bead_r"], c]) for c in frame["bead_cols"]]
    program = [color_to_direction[color] for color in bead_values]

    return {
        "grid": g,
        "bg": int(bg),
        "frame": frame,
        "field_shape": (int(r1 - r0 - 1), int(c1 - c0 - 1)),
        "start": (int(sr), int(sc)),
        "sprite_color": int(sprite_color),
        "obstacles": obstacles,
        "program": program,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the bead program by sliding the sprite according to the frame-color legend.

    Input structure:
    - A background grid contains a lower single-cell rectangular frame whose four corners are background.
    - The four frame sides have four distinct colors; inside the frame are one non-frame-color sprite cell and gray obstacle cells.
    - One background row separates the frame from a horizontal bead string above it, and every bead color is one of the side colors.

    Rule:
    - Read the bead string from left to right; each bead uses the direction of the frame side with the same color.
    - For every bead, slide the sprite in that direction until the next cell would be outside the frame interior or gray.
    - Erase the bead string, erase the sprite's old location, and draw the sprite at its final stop; keep frame and obstacles unchanged.

    Key insight:
    - The frame is a direction legend, not decoration: bead color selects a side direction, and each instruction is a full slide until blocked.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    frame = scene["frame"]
    field_h, field_w = scene["field_shape"]

    final, _steps = _run_program(
        scene["start"],
        scene["program"],
        field_h,
        field_w,
        scene["obstacles"],
    )

    r0, c0, _r1, _c1 = frame["bbox"]
    out = g.copy()
    for c in frame["bead_cols"]:
        out[frame["bead_r"], int(c)] = bg

    sr, sc = scene["start"]
    fr, fc = final
    out[int(r0 + 1 + sr), int(c0 + 1 + sc)] = bg
    out[int(r0 + 1 + fr), int(c0 + 1 + fc)] = int(scene["sprite_color"])
    return out


def _random_program() -> list[Coord]:
    length = random.randint(4, 7)
    program = DIRECTIONS[:]
    while len(program) < length:
        program.append(random.choice(DIRECTIONS))
    random.shuffle(program)
    return [(int(dr), int(dc)) for dr, dc in program]


def _motion_case() -> tuple[int, int, set[Coord], Coord, list[Coord], Coord] | None:
    field_h = random.randint(6, 13)
    field_w = random.randint(6, 14)
    cells = [(r, c) for r in range(field_h) for c in range(field_w)]

    max_obstacles = min(12, max(3, (field_h * field_w) // 5))
    obstacle_count = random.randint(3, max_obstacles)
    obstacles = set(random.sample(cells, obstacle_count))
    free = [cell for cell in cells if cell not in obstacles]
    if not free:
        return None

    start = random.choice(free)
    program = _random_program()
    final, steps = _run_program(start, program, field_h, field_w, obstacles)

    if final == start:
        return None
    if any(int(step["distance"]) == 0 for step in steps):
        return None
    if sum(int(step["distance"]) > 1 for step in steps) < 2:
        return None
    if {str(step["blocker"]) for step in steps} != {"border", "obstacle"}:
        return None
    if _run_program(start, program, field_h, field_w, set())[0] == final:
        return None
    if _run_one_step(start, program, field_h, field_w, obstacles) == final:
        return None
    if _run_program(start, list(reversed(program)), field_h, field_w, obstacles)[0] == final:
        return None
    for i in range(len(program)):
        shorter = program[:i] + program[i + 1 :]
        if _run_program(start, shorter, field_h, field_w, obstacles)[0] == final:
            return None

    return int(field_h), int(field_w), obstacles, start, program, final


def _render_case(
    field_h: int,
    field_w: int,
    obstacles: set[Coord],
    start: Coord,
    program: list[Coord],
    final: Coord,
) -> np.ndarray | None:
    colors = random.sample(PALETTE, 5)
    side_colors = dict(zip(DIRECTIONS, colors[:4]))
    sprite_color = int(colors[4])

    top_margin = random.randint(0, 2)
    frame_h = int(field_h + 2)
    frame_w = int(field_w + 2)
    height = int(top_margin + 2 + frame_h)
    min_width = max(frame_w, len(program) + 2)
    width = random.randint(min_width, min(30, min_width + 8))
    if height > 30 or width > 30:
        return None

    frame_r0 = int(top_margin + 2)
    frame_c0 = random.randint(0, width - frame_w)
    frame_r1 = int(frame_r0 + frame_h - 1)
    frame_c1 = int(frame_c0 + frame_w - 1)
    bead_r = int(top_margin)
    if len(program) <= frame_w:
        bead_c0 = random.randint(frame_c0, frame_c0 + frame_w - len(program))
    else:
        bead_c0 = random.randint(0, width - len(program))

    g = np.full((height, width), BG, dtype=int)
    g[frame_r0, frame_c0 + 1 : frame_c1] = side_colors[TOP]
    g[frame_r1, frame_c0 + 1 : frame_c1] = side_colors[BOTTOM]
    g[frame_r0 + 1 : frame_r1, frame_c0] = side_colors[LEFT]
    g[frame_r0 + 1 : frame_r1, frame_c1] = side_colors[RIGHT]

    for r, c in obstacles:
        g[frame_r0 + 1 + int(r), frame_c0 + 1 + int(c)] = GRAY
    sr, sc = start
    g[frame_r0 + 1 + int(sr), frame_c0 + 1 + int(sc)] = sprite_color

    for i, direction in enumerate(program):
        g[bead_r, bead_c0 + i] = int(side_colors[direction])

    expected = g.copy()
    expected[bead_r, bead_c0 : bead_c0 + len(program)] = BG
    expected[frame_r0 + 1 + int(sr), frame_c0 + 1 + int(sc)] = BG
    fr, fc = final
    expected[frame_r0 + 1 + int(fr), frame_c0 + 1 + int(fc)] = sprite_color

    try:
        solved = main(g)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _motion_case()
        if case is None:
            continue
        grid = _render_case(*case)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-frame program input")
