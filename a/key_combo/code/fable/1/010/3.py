import random

import numpy as np


BLACK = 0
GREY = 5
WALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
EMITTER_MARGIN = 2
MAX_REFLECTIVE_HITS = 18
MAX_GENERATE_ATTEMPTS = 600


def _sign(value: int) -> int:
    return 1 if value > 0 else -1


def _hit_side(r: int, c: int, height: int, width: int) -> str | None:
    on_top_or_bottom = r in (0, height - 1)
    on_left_or_right = c in (0, width - 1)
    if not on_top_or_bottom and not on_left_or_right:
        return None
    if on_top_or_bottom and on_left_or_right:
        return "corner"
    if r == 0:
        return "top"
    if r == height - 1:
        return "bottom"
    if c == 0:
        return "left"
    return "right"


def _find_markers(grid: np.ndarray) -> tuple[tuple[int, int], int, tuple[int, int], tuple[int, int]]:
    height, width = grid.shape
    if height < 5 or width < 5:
        raise AssertionError("grid too small for an interior emitter and border walls")

    interior_coords = np.argwhere(grid[1:-1, 1:-1] != BLACK)
    interior_coords = interior_coords + np.array([1, 1])
    if interior_coords.shape[0] != 2:
        raise AssertionError("expected exactly two non-black interior markers")

    colors = [int(grid[r, c]) for r, c in interior_coords]
    grey_indices = [idx for idx, color in enumerate(colors) if color == GREY]
    if len(grey_indices) != 1:
        raise AssertionError("expected exactly one grey exhaust marker")
    tail_idx = grey_indices[0]
    emit_idx = 1 - tail_idx

    tail = tuple(int(v) for v in interior_coords[tail_idx])
    emitter = tuple(int(v) for v in interior_coords[emit_idx])
    emitter_color = int(grid[emitter])
    if emitter_color in (BLACK, GREY):
        raise AssertionError("emitter must be a non-black, non-grey color")

    if abs(emitter[0] - tail[0]) != 1 or abs(emitter[1] - tail[1]) != 1:
        raise AssertionError("tail must be diagonally adjacent to the emitter")

    direction = (_sign(emitter[0] - tail[0]), _sign(emitter[1] - tail[1]))
    return emitter, emitter_color, tail, direction


def _trace(
    grid: np.ndarray,
    *,
    recolor_on_bounce: bool = True,
    black_absorbs: bool = True,
    stop_on_any_wall: bool = False,
) -> tuple[np.ndarray, dict]:
    height, width = grid.shape
    emitter, emitter_color, tail, (dr, dc) = _find_markers(grid)

    out = np.asarray(grid, dtype=int).copy()
    current_color = emitter_color
    r, c = emitter
    preserved = {emitter, tail}
    painted_counts: dict[tuple[int, int], int] = {}
    hits: list[dict] = []
    seen_states: set[tuple[int, int, int, int, int]] = set()
    absorbed = False

    for _ in range(height * width * 8):
        state = (r, c, dr, dc, current_color)
        if state in seen_states:
            raise AssertionError("trajectory cycled before termination")
        seen_states.add(state)

        segment: list[tuple[int, int]] = []
        while True:
            nr, nc = r + dr, c + dc
            side = _hit_side(nr, nc, height, width)
            if side is not None:
                if side == "corner":
                    raise AssertionError("trajectory hit a corner wall cell")
                wall_color = int(grid[nr, nc])
                hit = {
                    "wall": (nr, nc),
                    "side": side,
                    "wall_color": wall_color,
                    "incoming_color": current_color,
                    "segment": tuple(segment),
                    "state_before_bounce": (r, c, dr, dc, current_color),
                }
                hits.append(hit)

                if stop_on_any_wall:
                    absorbed = wall_color == BLACK
                    return out, {
                        "hits": hits,
                        "unique_painted": len(painted_counts),
                        "revisits": sum(count - 1 for count in painted_counts.values()),
                        "absorbed": absorbed,
                    }

                if wall_color == BLACK and black_absorbs:
                    absorbed = True
                    return out, {
                        "hits": hits,
                        "unique_painted": len(painted_counts),
                        "revisits": sum(count - 1 for count in painted_counts.values()),
                        "absorbed": absorbed,
                    }

                if wall_color != BLACK and recolor_on_bounce:
                    current_color = wall_color
                if side in ("top", "bottom"):
                    dr = -dr
                else:
                    dc = -dc
                break

            if (nr, nc) not in preserved:
                painted_counts[(nr, nc)] = painted_counts.get((nr, nc), 0) + 1
                out[nr, nc] = current_color
            r, c = nr, nc
            segment.append((r, c))

    raise AssertionError("trajectory exceeded safety bound")


def _reflective_hits(
    height: int,
    width: int,
    emitter: tuple[int, int],
    direction: tuple[int, int],
) -> list[dict] | None:
    r, c = emitter
    dr, dc = direction
    seen_states = {(r, c, dr, dc)}
    hits: list[dict] = []

    for _ in range(MAX_REFLECTIVE_HITS):
        segment: list[tuple[int, int]] = []
        while True:
            nr, nc = r + dr, c + dc
            side = _hit_side(nr, nc, height, width)
            if side is not None:
                if side == "corner":
                    return None
                hits.append(
                    {
                        "wall": (nr, nc),
                        "side": side,
                        "segment": tuple(segment),
                    }
                )
                if side in ("top", "bottom"):
                    dr = -dr
                else:
                    dc = -dc
                break
            r, c = nr, nc
            segment.append((r, c))

        state = (r, c, dr, dc)
        if state in seen_states:
            break
        seen_states.add(state)

    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a diagonal billiard ray that reflects on colored border walls and stops on black border walls.

    Input structure:
    - The outermost ring is a wall border made of colored segments and black absorbing segments.
    - The interior is otherwise black except for one colored emitter pixel and one grey exhaust pixel diagonally behind it.
    - The exhaust-to-emitter diagonal uniquely determines the launch direction, and valid inputs never force a corner collision.

    Rule:
    - Launch from the emitter in the diagonal direction from the grey tail to the emitter.
    - Paint each traversed interior background cell with the current segment color, starting with the emitter's own color.
    - When the next step would hit a colored wall cell, reflect across that wall face and change the paint color to that wall cell's color; when the next step would hit a black wall cell, stop immediately.

    Key insight:
    - The border is both geometry and palette: each colored hit both turns the path and announces the next segment's color, while a black wall is the unique absorber.
    """
    out, _ = _trace(np.asarray(grid, dtype=int))
    return out


def _sample_branch() -> tuple[bool, bool]:
    zero_bounce = random.random() < 0.2
    need_crossing = (not zero_bounce) and (random.random() < 0.4)
    return zero_bounce, need_crossing


def _collect_prefix_infos(
    hits: list[dict],
    markers: set[tuple[int, int]],
    interior_area: int,
) -> list[dict]:
    infos: list[dict] = []
    painted: dict[tuple[int, int], int] = {}
    seen_walls: set[tuple[int, int]] = set()
    invalid = False

    for idx, hit in enumerate(hits):
        wall = hit["wall"]
        if wall in seen_walls:
            invalid = True
        seen_walls.add(wall)

        for cell in hit["segment"]:
            if cell in markers:
                invalid = True
            painted[cell] = painted.get(cell, 0) + 1

        unique_painted = len(painted)
        revisits = sum(count - 1 for count in painted.values())
        next_segment_len = len(hits[idx + 1]["segment"]) if idx + 1 < len(hits) else 0
        infos.append(
            {
                "index": idx,
                "unique_painted": unique_painted,
                "revisits": revisits,
                "distinct_walls": len(seen_walls) == idx + 1,
                "next_segment_len": next_segment_len,
                "paint_fraction": unique_painted / interior_area,
                "invalid": invalid,
            }
        )
    return infos


def _choose_final_index(
    infos: list[dict],
    *,
    zero_bounce: bool,
    need_crossing: bool,
) -> int | None:
    candidates: list[dict] = []
    for info in infos:
        idx = info["index"]
        if info["invalid"] or not info["distinct_walls"]:
            continue
        if info["next_segment_len"] <= 0:
            continue
        if info["unique_painted"] < 4 or info["paint_fraction"] > 0.45:
            continue
        if zero_bounce and idx != 0:
            continue
        if not zero_bounce and idx == 0:
            continue
        if need_crossing and info["revisits"] == 0:
            continue
        if idx > 6:
            continue
        candidates.append(info)

    if not candidates:
        return None

    weights = []
    for info in candidates:
        weight = 1 + min(info["index"], 4)
        if info["revisits"] > 0:
            weight += 3
        weights.append(weight)
    return random.choices(candidates, weights=weights, k=1)[0]["index"]


def _pick_bounce_colors(emitter_color: int, bounce_count: int) -> list[int]:
    colors: list[int] = []
    current = emitter_color
    for _ in range(bounce_count):
        choices = [color for color in WALL_COLORS if color != current]
        color = random.choice(choices)
        colors.append(color)
        current = color
    return colors


def _side_coords(height: int, width: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(width)]
    if side == "bottom":
        return [(height - 1, c) for c in range(width)]
    if side == "left":
        return [(r, 0) for r in range(height)]
    if side == "right":
        return [(r, width - 1) for r in range(height)]
    raise ValueError("unknown side")


def _paint_random_border(grid: np.ndarray) -> None:
    height, width = grid.shape
    for side in ("top", "bottom", "left", "right"):
        coords = _side_coords(height, width, side)
        idx = 0
        previous = None
        while idx < len(coords):
            remaining = len(coords) - idx
            run_len = random.randint(1, min(5, remaining))
            if previous is None or random.random() < 0.18:
                choices = [BLACK] + WALL_COLORS
            else:
                choices = [BLACK] + [color for color in WALL_COLORS if color != previous]
            color = random.choice(choices)
            for offset in range(run_len):
                r, c = coords[idx + offset]
                grid[r, c] = color
            previous = color
            idx += run_len


def _expand_required_wall_colors(
    grid: np.ndarray,
    required: dict[tuple[int, int], int],
) -> None:
    height, width = grid.shape
    for (r, c), color in required.items():
        side = _hit_side(r, c, height, width)
        assert side in ("top", "bottom", "left", "right")
        span = 1 if color == BLACK else random.randint(0, 2)
        for delta in range(-span, span + 1):
            if side in ("top", "bottom"):
                rr, cc = r, c + delta
            else:
                rr, cc = r + delta, c
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if _hit_side(rr, cc, height, width) != side:
                continue
            if (rr, cc) in required and required[(rr, cc)] != color:
                continue
            grid[rr, cc] = color


def _build_grid(
    height: int,
    width: int,
    emitter: tuple[int, int],
    tail: tuple[int, int],
    emitter_color: int,
    hits: list[dict],
    final_index: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    _paint_random_border(grid)

    bounce_colors = _pick_bounce_colors(emitter_color, final_index)
    required: dict[tuple[int, int], int] = {}
    for idx, color in enumerate(bounce_colors):
        required[hits[idx]["wall"]] = color
    required[hits[final_index]["wall"]] = BLACK

    _expand_required_wall_colors(grid, required)
    for cell, color in required.items():
        grid[cell] = color

    grid[tail] = GREY
    grid[emitter] = emitter_color

    border_colors = set(int(grid[r, c]) for r in range(height) for c in range(width) if r in (0, height - 1) or c in (0, width - 1))
    non_black_border = border_colors - {BLACK}
    if len(non_black_border) < 2:
        r, c = random.choice([(0, 1), (height - 1, width - 2), (1, 0), (height - 2, width - 1)])
        grid[r, c] = random.choice(WALL_COLORS)
    return grid


def _validate_instance(
    grid: np.ndarray,
    *,
    expect_zero_bounce: bool,
    expect_crossing: bool,
) -> bool:
    try:
        out, meta = _trace(grid)
        stop_out, _ = _trace(grid, stop_on_any_wall=True)
        if meta["hits"][:-1]:
            no_recolor_out, _ = _trace(grid, recolor_on_bounce=False)
        else:
            no_recolor_out = None
    except AssertionError:
        return False

    if not meta["absorbed"]:
        return False
    colored_bounces = len(meta["hits"]) - 1
    if colored_bounces < 0:
        return False
    if expect_zero_bounce != (colored_bounces == 0):
        return False
    if expect_crossing and meta["revisits"] == 0:
        return False
    if meta["unique_painted"] < 4:
        return False
    if np.array_equal(out, grid):
        return False

    emitter, _, _, direction = _find_markers(grid)
    reflective = _reflective_hits(grid.shape[0], grid.shape[1], emitter, direction)
    if reflective is None:
        return False
    final_index = len(meta["hits"]) - 1
    if final_index + 1 >= len(reflective):
        return False
    if len(reflective[final_index + 1]["segment"]) == 0:
        return False

    if colored_bounces > 0:
        incoming = int(grid[_find_markers(grid)[0]])
        wall_colors = []
        for hit in meta["hits"][:-1]:
            wall_colors.append(hit["wall_color"])
        if all(color == incoming for color in wall_colors):
            return False
        if no_recolor_out is None or np.array_equal(out, no_recolor_out):
            return False
        if np.array_equal(out, stop_out):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    zero_bounce, need_crossing = _sample_branch()

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    direction = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    emitter = (
        random.randint(EMITTER_MARGIN, height - EMITTER_MARGIN - 1),
        random.randint(EMITTER_MARGIN, width - EMITTER_MARGIN - 1),
    )
    tail = (emitter[0] - direction[0], emitter[1] - direction[1])
    if not (1 <= tail[0] < height - 1 and 1 <= tail[1] < width - 1):
        return None

    hits = _reflective_hits(height, width, emitter, direction)
    if hits is None or len(hits) < 2:
        return None

    infos = _collect_prefix_infos(hits, {emitter, tail}, (height - 2) * (width - 2))
    final_index = _choose_final_index(infos, zero_bounce=zero_bounce, need_crossing=need_crossing)
    if final_index is None:
        return None

    emitter_color = random.choice(WALL_COLORS)
    grid = _build_grid(height, width, emitter, tail, emitter_color, hits, final_index)

    if not _validate_instance(
        grid,
        expect_zero_bounce=zero_bounce,
        expect_crossing=need_crossing,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid colored-billiards input")
