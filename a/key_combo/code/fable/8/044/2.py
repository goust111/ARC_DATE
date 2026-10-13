import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
GRID_SIDE_RANGE = (12, 25)
CENTER_MARGIN = 2
MIN_FIRST_CENTER_DIST = 3
MIN_CENTER_DIST = 4
MAX_HIT_COUNT = 4
MAX_DECOY_COUNT = 3

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
CORNER_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
MIRROR_CORNERS = {
    "/": ("NE", "SW"),
    "\\": ("NW", "SE"),
}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _reflect(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    if mirror_kind == "/":
        mapping = {
            DIRS["U"]: DIRS["R"],
            DIRS["R"]: DIRS["U"],
            DIRS["D"]: DIRS["L"],
            DIRS["L"]: DIRS["D"],
        }
    else:
        mapping = {
            DIRS["U"]: DIRS["L"],
            DIRS["L"]: DIRS["U"],
            DIRS["D"]: DIRS["R"],
            DIRS["R"]: DIRS["D"],
        }
    assert direction in mapping
    return mapping[direction]


def _mirror_kind_from_turn(
    incoming: tuple[int, int], outgoing: tuple[int, int]
) -> str:
    if _reflect(incoming, "/") == outgoing:
        return "/"
    assert _reflect(incoming, "\\") == outgoing
    return "\\"


def _bbox_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _shell_cells(
    center: tuple[int, int], missing_corner: str
) -> set[tuple[int, int]]:
    r, c = center
    cells = _bbox_cells(center)
    cells.remove((r, c))
    dr, dc = CORNER_OFFSETS[missing_corner]
    cells.remove((r + dr, c + dc))
    return cells


def _missing_coord(center: tuple[int, int], missing_corner: str) -> tuple[int, int]:
    r, c = center
    dr, dc = CORNER_OFFSETS[missing_corner]
    return r + dr, c + dc


def _segment_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec > sc else -1
        return [(sr, c) for c in range(sc + step, ec + step, step)]
    assert sc == ec
    step = 1 if er > sr else -1
    return [(r, sc) for r in range(sr + step, er + step, step)]


def _ray_to_boundary(
    start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cells.append((r, c))


def _center_candidates(
    current: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    min_dist: int,
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = current
    dr, dc = direction
    out = []
    rr, cc = r + dr * min_dist, c + dc * min_dist
    while CENTER_MARGIN <= rr < h - CENTER_MARGIN and CENTER_MARGIN <= cc < w - CENTER_MARGIN:
        out.append((rr, cc))
        rr += dr
        cc += dc
    random.shuffle(out)
    return out


def _segment_is_clear(
    segment: list[tuple[int, int]],
    used_path: set[tuple[int, int]],
    reserved_boxes: set[tuple[int, int]],
    current_bbox: set[tuple[int, int]],
    candidate_bbox: set[tuple[int, int]] | None,
) -> bool:
    allowed_reserved = set(current_bbox)
    if candidate_bbox is not None:
        allowed_reserved |= candidate_bbox
    for cell in segment:
        if cell in used_path:
            return False
        if cell in reserved_boxes and cell not in allowed_reserved:
            return False
    if candidate_bbox is None:
        return True
    for cell in candidate_bbox:
        if cell in reserved_boxes and cell not in current_bbox:
            return False
        if cell in used_path and cell not in segment and cell not in current_bbox:
            return False
    return True


def _render_input_from_plan(plan: dict) -> np.ndarray:
    grid = np.full((plan["height"], plan["width"]), BG, dtype=int)
    sr, sc = plan["source"]
    grid[sr, sc] = int(plan["source_color"])
    for square in plan["squares"]:
        for r, c in _shell_cells(square["center"], square["missing_corner"]):
            grid[r, c] = int(square["color"])
    return grid


def _render_expected_from_plan(plan: dict, inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    square_shell = set()
    for square in plan["squares"]:
        square_shell |= _shell_cells(square["center"], square["missing_corner"])

    for segment in plan["segments"]:
        for r, c in segment:
            if (r, c) not in square_shell and out[r, c] == BG:
                out[r, c] = int(plan["source_color"])

    for square in plan["squares"]:
        if square["hit"]:
            mr, mc = _missing_coord(square["center"], square["missing_corner"])
            out[mr, mc] = int(square["color"])
    return out


def _choose_hit_count() -> int:
    return random.choices([0, 1, 2, 3, 4], weights=[1, 4, 5, 4, 2], k=1)[0]


def _search_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    current_bbox: set[tuple[int, int]],
    turns_left: int,
    used_path: set[tuple[int, int]],
    reserved_boxes: set[tuple[int, int]],
    hits_so_far: list[dict],
    segments_so_far: list[list[tuple[int, int]]],
) -> tuple[list[dict], list[list[tuple[int, int]]]] | None:
    if turns_left == 0:
        exit_segment = _ray_to_boundary(current, direction, shape)
        if len(exit_segment) < 2:
            return None
        if not _segment_is_clear(
            exit_segment, used_path, reserved_boxes, current_bbox, None
        ):
            return None
        return hits_so_far, segments_so_far + [exit_segment]

    min_dist = MIN_FIRST_CENTER_DIST if not hits_so_far else MIN_CENTER_DIST
    candidates = []
    for center in _center_candidates(current, direction, shape, min_dist):
        for turn_fn in random.sample([_turn_left, _turn_right], k=2):
            out_dir = turn_fn(direction)
            mirror_kind = _mirror_kind_from_turn(direction, out_dir)
            candidates.append((center, out_dir, mirror_kind))
    random.shuffle(candidates)

    for center, out_dir, mirror_kind in candidates:
        candidate_bbox = _bbox_cells(center)
        segment = _segment_cells(current, center)
        if not _segment_is_clear(
            segment, used_path, reserved_boxes, current_bbox, candidate_bbox
        ):
            continue
        missing_corner = random.choice(MIRROR_CORNERS[mirror_kind])
        square = {
            "center": center,
            "color": None,
            "missing_corner": missing_corner,
            "mirror_kind": mirror_kind,
            "hit": True,
        }
        result = _search_path(
            shape,
            center,
            out_dir,
            candidate_bbox,
            turns_left - 1,
            used_path | set(segment),
            reserved_boxes | candidate_bbox,
            hits_so_far + [square],
            segments_so_far + [segment],
        )
        if result is not None:
            return result
    return None


def _sample_source(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return (0, random.randint(1, w - 2)), DIRS["D"]
    if side == "bottom":
        return (h - 1, random.randint(1, w - 2)), DIRS["U"]
    if side == "left":
        return (random.randint(1, h - 2), 0), DIRS["R"]
    return (random.randint(1, h - 2), w - 1), DIRS["L"]


def _free_decoy_centers(
    shape: tuple[int, int],
    reserved_boxes: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(CENTER_MARGIN, h - CENTER_MARGIN):
        for c in range(CENTER_MARGIN, w - CENTER_MARGIN):
            bbox = _bbox_cells((r, c))
            if bbox & reserved_boxes:
                continue
            if bbox & used_path:
                continue
            out.append((r, c))
    random.shuffle(out)
    return out


def _assign_colors(plan: dict) -> bool:
    square_count = len(plan["squares"])
    if square_count + 1 > len(PALETTE):
        return False
    colors = random.sample(PALETTE, square_count + 1)
    plan["source_color"] = colors[0]
    for square, color in zip(plan["squares"], colors[1:]):
        square["color"] = color
    return True


def _build_plan() -> dict | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    shape = (h, w)
    source, direction = _sample_source(shape)
    hit_count = _choose_hit_count()

    search = _search_path(
        shape,
        source,
        direction,
        set(),
        hit_count,
        {source},
        set(),
        [],
        [],
    )
    if search is None:
        return None
    hit_squares, segments = search

    used_path = {source}
    for segment in segments:
        used_path |= set(segment)
    reserved_boxes = set()
    for square in hit_squares:
        reserved_boxes |= _bbox_cells(square["center"])

    decoy_centers = _free_decoy_centers(shape, reserved_boxes, used_path)
    if not decoy_centers:
        return None
    decoy_count = random.randint(1, min(MAX_DECOY_COUNT, len(decoy_centers)))
    decoys = []
    decoy_reserved = set(reserved_boxes)
    for center in decoy_centers:
        bbox = _bbox_cells(center)
        if bbox & decoy_reserved:
            continue
        mirror_kind = random.choice(["/", "\\"])
        decoys.append(
            {
                "center": center,
                "color": None,
                "missing_corner": random.choice(MIRROR_CORNERS[mirror_kind]),
                "mirror_kind": mirror_kind,
                "hit": False,
            }
        )
        decoy_reserved |= bbox
        if len(decoys) == decoy_count:
            break
    if len(decoys) != decoy_count:
        return None

    plan = {
        "height": h,
        "width": w,
        "source": source,
        "source_dir": direction,
        "source_color": None,
        "squares": hit_squares + decoys,
        "segments": segments,
        "hit_count": hit_count,
    }
    if not _assign_colors(plan):
        return None
    return plan


def _detect_squares(grid: np.ndarray, bg: int) -> dict[tuple[int, int], dict]:
    h, w = grid.shape
    squares = {}
    occupied = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            edge_vals = [int(patch[0, 1]), int(patch[1, 0]), int(patch[1, 2]), int(patch[2, 1])]
            if any(v == bg for v in edge_vals):
                continue
            color = edge_vals[0]
            if any(v != color for v in edge_vals[1:]):
                continue
            corners = {
                "NW": int(patch[0, 0]),
                "NE": int(patch[0, 2]),
                "SW": int(patch[2, 0]),
                "SE": int(patch[2, 2]),
            }
            missing = [name for name, val in corners.items() if val == bg]
            if len(missing) != 1:
                continue
            if any(val not in {bg, color} for val in corners.values()):
                continue
            if sum(val == color for val in corners.values()) != 3:
                continue
            missing_corner = missing[0]
            shell = _shell_cells((r, c), missing_corner)
            if any(cell in occupied for cell in shell):
                raise AssertionError("overlapping hollow squares are not allowed")
            occupied |= shell
            squares[(r, c)] = {
                "center": (r, c),
                "color": color,
                "missing_corner": missing_corner,
                "mirror_kind": "/" if missing_corner in {"NE", "SW"} else "\\",
                "shell": shell,
            }
    return squares


def _find_source(
    grid: np.ndarray, bg: int, square_shell: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], int]:
    h, w = grid.shape
    edge_cells = set()
    for c in range(w):
        edge_cells.add((0, c))
        edge_cells.add((h - 1, c))
    for r in range(h):
        edge_cells.add((r, 0))
        edge_cells.add((r, w - 1))
    candidates = [cell for cell in edge_cells if int(grid[cell]) != bg and cell not in square_shell]
    assert len(candidates) == 1, "expected exactly one non-square edge source pixel"
    source = candidates[0]
    r, c = source
    sides = []
    if r == 0:
        sides.append(DIRS["D"])
    if r == h - 1:
        sides.append(DIRS["U"])
    if c == 0:
        sides.append(DIRS["R"])
    if c == w - 1:
        sides.append(DIRS["L"])
    assert len(sides) == 1, "source must not sit at a corner"
    return source, sides[0], int(grid[source])


def _trace_from_input(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[tuple[int, int], dict], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5
    bg = get_bg(g)
    assert bg == BG

    squares = _detect_squares(g, bg)
    assert squares, "expected at least one hollow square"
    shell_cells = set()
    for square in squares.values():
        shell_cells |= square["shell"]

    source, direction, source_color = _find_source(g, bg, shell_cells)
    allowed_non_bg = set(shell_cells) | {source}
    for r, c in np.argwhere(g != bg):
        assert (int(r), int(c)) in allowed_non_bg, "unexpected foreground cell"

    out = g.copy()
    hits = []
    seen_states = set()
    position = source
    current_dir = direction

    while True:
        state = position + current_dir
        assert state not in seen_states, "ray should not loop forever"
        seen_states.add(state)
        nr = position[0] + current_dir[0]
        nc = position[1] + current_dir[1]
        if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
            break
        if (nr, nc) not in shell_cells and int(g[nr, nc]) == bg:
            out[nr, nc] = source_color
        position = (nr, nc)
        if position in squares:
            hits.append(position)
            current_dir = _reflect(current_dir, squares[position]["mirror_kind"])

    for center in hits:
        square = squares[center]
        mr, mc = _missing_coord(center, square["missing_corner"])
        out[mr, mc] = int(square["color"])

    return out, hits, squares, source, source_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the edge-fired ray through missing-corner hollow squares, then complete only the mirrors it actually hits.

    Input structure:
    - The background is black, with exactly one non-square source pixel on a grid edge.
    - The remaining foreground consists of several monochrome 3x3 hollow squares whose centers are black and whose ring is missing exactly one corner.
    - A missing NE or SW corner encodes a "/" mirror, while a missing NW or SE corner encodes a "\\" mirror.

    Rule:
    - The source emits a ray straight inward, perpendicular to its edge.
    - Every background cell traversed by the ray is painted with the source color, including any mirror center cell it reaches.
    - When the ray reaches a hollow square's center, it turns 90 degrees according to that square's encoded mirror and continues until it leaves the grid.
    - The ray never repaints the square shell itself, and only squares whose centers are reached have their missing corner filled with the square's own color.

    Key insight:
    - The missing corner carries two visible roles at once: it tells you which way the center reflection turns, and it also identifies the exact corner that must be restored after a hit.
    """
    out, _hits, _squares, _source, _source_color = _trace_from_input(grid)
    return out


def _validate_plan(inp: np.ndarray, expected: np.ndarray, plan: dict) -> bool:
    if inp.shape != (plan["height"], plan["width"]):
        return False
    if int(inp[plan["source"]]) != int(plan["source_color"]):
        return False

    hit_centers = {square["center"] for square in plan["squares"] if square["hit"]}
    untouched_centers = {square["center"] for square in plan["squares"] if not square["hit"]}
    visible_painted = set()
    shell = set()
    for square in plan["squares"]:
        shell |= _shell_cells(square["center"], square["missing_corner"])

    for segment in plan["segments"]:
        for cell in segment:
            if cell not in shell:
                visible_painted.add(cell)
    if not visible_painted:
        return False

    for square in plan["squares"]:
        miss = _missing_coord(square["center"], square["missing_corner"])
        if int(inp[miss]) != BG:
            return False
        want = int(square["color"]) if square["hit"] else BG
        if int(expected[miss]) != want:
            return False
    for center in untouched_centers:
        if center in visible_painted:
            return False
    for center in hit_centers:
        if center not in visible_painted:
            return False
    if len(hit_centers) != plan["hit_count"]:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    traced, hits, _squares, source, source_color = _trace_from_input(inp)
    if not np.array_equal(traced, expected):
        return False
    if set(hits) != hit_centers:
        return False
    if source != plan["source"] or source_color != int(plan["source_color"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    plan = _build_plan()
    if plan is None:
        return None
    inp = _render_input_from_plan(plan)
    expected = _render_expected_from_plan(plan, inp)
    if not _validate_plan(inp, expected, plan):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
