from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
SIDE_NAMES = ("top", "bottom", "left", "right")
FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
OBSTACLE_SIZES = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
)
GRID_SIZE_MIN = 10
GRID_SIZE_MAX = 18
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 10
GENERATION_RETRIES = 400


@dataclass(frozen=True)
class Candidate:
    source: tuple[int, int]
    dest: tuple[int, int]
    color: int
    side: str
    blocked: bool
    far: bool


def _uniform_color(values: np.ndarray, side_name: str) -> int:
    colors = {int(v) for v in np.asarray(values).ravel()}
    assert len(colors) == 1, f"{side_name} frame side must be monochrome"
    return colors.pop()


def _extract_side_colors(grid: np.ndarray) -> dict[str, int]:
    assert grid.ndim == 2, "expected a 2D grid"
    height, width = map(int, grid.shape)
    assert height >= 5 and width >= 5, "grid must have room for a one-cell frame and interior"
    assert (
        int(grid[0, 0]) == BLACK
        and int(grid[0, width - 1]) == BLACK
        and int(grid[height - 1, 0]) == BLACK
        and int(grid[height - 1, width - 1]) == BLACK
    ), "frame corners must be black"

    side_colors = {
        "top": _uniform_color(grid[0, 1:-1], "top"),
        "bottom": _uniform_color(grid[height - 1, 1:-1], "bottom"),
        "left": _uniform_color(grid[1:-1, 0], "left"),
        "right": _uniform_color(grid[1:-1, width - 1], "right"),
    }
    assert len(set(side_colors.values())) == 4, "the four frame sides must use four distinct colors"
    assert BLACK not in side_colors.values() and GREY not in side_colors.values(), (
        "frame side colors must be non-black and non-grey"
    )
    return side_colors


def _path_flags(
    obstacles: np.ndarray,
    side: str,
    r: int,
    c: int,
    height: int,
    width: int,
) -> tuple[bool, bool]:
    if side == "top":
        return bool(obstacles[1:r, c].any()), bool(obstacles[r + 1 : height - 1, c].any())
    if side == "bottom":
        return bool(obstacles[r + 1 : height - 1, c].any()), bool(obstacles[1:r, c].any())
    if side == "left":
        return bool(obstacles[r, 1:c].any()), bool(obstacles[r, c + 1 : width - 1].any())
    assert side == "right"
    return bool(obstacles[r, c + 1 : width - 1].any()), bool(obstacles[r, 1:c].any())


def _target_cell(side: str, blocked: bool, r: int, c: int, height: int, width: int) -> tuple[int, int]:
    if side == "top":
        return (height - 2, c) if blocked else (1, c)
    if side == "bottom":
        return (1, c) if blocked else (height - 2, c)
    if side == "left":
        return (r, width - 2) if blocked else (r, 1)
    assert side == "right"
    return (r, 1) if blocked else (r, width - 2)


def _matched_is_farther(side: str, r: int, c: int, height: int, width: int) -> bool:
    if side == "top":
        return (r - 1) > (height - 2 - r)
    if side == "bottom":
        return (height - 2 - r) > (r - 1)
    if side == "left":
        return (c - 1) > (width - 2 - c)
    assert side == "right"
    return (width - 2 - c) > (c - 1)


def _move_destination(
    obstacles: np.ndarray,
    side: str,
    r: int,
    c: int,
    height: int,
    width: int,
) -> tuple[tuple[int, int], bool]:
    blocked, opposite_blocked = _path_flags(obstacles, side, r, c, height, width)
    assert not (blocked and opposite_blocked), "valid inputs must keep the opposite path clear"
    return _target_cell(side, blocked, r, c, height, width), blocked


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    side_colors = _extract_side_colors(g)
    color_to_side = {color: side for side, color in side_colors.items()}
    height, width = map(int, g.shape)

    interior = g[1:-1, 1:-1]
    mover_mask = (interior != BLACK) & (interior != GREY)
    movers = np.argwhere(mover_mask)
    if movers.size == 0:
        return g.copy()

    movers = movers + np.array([1, 1], dtype=int)
    obstacles = g == GREY
    out = g.copy()
    out[movers[:, 0], movers[:, 1]] = BLACK

    used_destinations: set[tuple[int, int]] = set()
    for r_raw, c_raw in movers:
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        assert color in color_to_side, "every movable interior pixel must match one frame side color"
        side = color_to_side[color]
        dest, _blocked = _move_destination(obstacles, side, r, c, height, width)
        assert dest not in used_destinations, "distinct pixels must land on distinct cells"
        dr, dc = dest
        assert int(g[dr, dc]) != GREY, "destination cells must not be blocked"
        used_destinations.add(dest)
        out[dr, dc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior pixel straight to the inner cell beside its same-colored frame side, flipping to the opposite side when a grey obstacle blocks that route.

    Input structure:
    - The outermost ring is a one-cell-thick frame whose four corners are black and whose four sides are four distinct non-grey colors.
    - The interior contains black background, side-matching colored pixels, and one or more solid grey rectangular obstacles.

    Rule:
    - For each interior colored pixel, find the unique frame side with the same color.
    - Move the pixel in the perpendicular straight line toward that side and stop in the cell just inside the frame.
    - If any grey obstacle cell lies anywhere on that route, ignore the matched side and send the pixel instead to the corresponding cell just inside the opposite side.
    - Apply all moves simultaneously and keep the frame, background, and grey obstacles unchanged.

    Key insight:
    - Frame-side colors define each pixel's intended direction, but grey obstacles do not stop pixels locally: they flip the entire destination to the opposite inner border.
    """
    return _transform(grid)


def _random_choice(seq: list, rng: np.random.Generator):
    return seq[int(rng.integers(0, len(seq)))]


def _place_obstacles(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    target_count = int(rng.integers(1, 4))
    placed = 0

    for _ in range(80):
        if placed == target_count:
            break
        rect_h, rect_w = _random_choice(list(OBSTACLE_SIZES), rng)
        if height - 4 - rect_h <= 0 or width - 4 - rect_w <= 0:
            continue
        r0 = int(rng.integers(2, height - 2 - rect_h + 1))
        c0 = int(rng.integers(2, width - 2 - rect_w + 1))
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1

        rr0 = max(1, r0 - 1)
        cc0 = max(1, c0 - 1)
        rr1 = min(height - 2, r1 + 1)
        cc1 = min(width - 2, c1 + 1)
        if mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        placed += 1

    if placed == 0:
        return None
    return mask


def _build_candidates(
    obstacle_mask: np.ndarray,
    side_colors: dict[str, int],
) -> list[Candidate]:
    height, width = map(int, obstacle_mask.shape)
    candidates: list[Candidate] = []

    for r in range(2, height - 2):
        for c in range(2, width - 2):
            if obstacle_mask[r, c]:
                continue
            for side in SIDE_NAMES:
                blocked, opposite_blocked = _path_flags(obstacle_mask, side, r, c, height, width)
                if blocked and opposite_blocked:
                    continue
                candidates.append(
                    Candidate(
                        source=(r, c),
                        dest=_target_cell(side, blocked, r, c, height, width),
                        color=side_colors[side],
                        side=side,
                        blocked=blocked,
                        far=_matched_is_farther(side, r, c, height, width),
                    )
                )
    return candidates


def _compatible(
    candidate: Candidate,
    used_sources: set[tuple[int, int]],
    used_dests: set[tuple[int, int]],
) -> bool:
    return candidate.source not in used_sources and candidate.dest not in used_dests


def _pick_candidate(
    pool: list[Candidate],
    rng: np.random.Generator,
    used_sources: set[tuple[int, int]],
    used_dests: set[tuple[int, int]],
    *,
    blocked: bool | None = None,
    far: bool | None = None,
) -> Candidate | None:
    options = [
        cand
        for cand in pool
        if _compatible(cand, used_sources, used_dests)
        and (blocked is None or cand.blocked == blocked)
        and (far is None or cand.far == far)
    ]
    if not options:
        return None
    return _random_choice(options, rng)


def _add_candidate(
    candidate: Candidate,
    selected: list[Candidate],
    used_sources: set[tuple[int, int]],
    used_dests: set[tuple[int, int]],
) -> None:
    selected.append(candidate)
    used_sources.add(candidate.source)
    used_dests.add(candidate.dest)


def _select_pixels(candidates: list[Candidate], rng: np.random.Generator) -> list[Candidate] | None:
    by_side = {side: [cand for cand in candidates if cand.side == side] for side in SIDE_NAMES}
    if any(not by_side[side] for side in SIDE_NAMES):
        return None

    blocked_sides = [side for side in SIDE_NAMES if any(cand.blocked for cand in by_side[side])]
    if not blocked_sides or not any(not cand.blocked for cand in candidates):
        return None

    target_count = int(rng.integers(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1))

    for _ in range(120):
        selected: list[Candidate] = []
        used_sources: set[tuple[int, int]] = set()
        used_dests: set[tuple[int, int]] = set()

        blocked_side_count = 2 if len(blocked_sides) >= 2 and rng.random() < 0.75 else 1
        blocked_order = list(blocked_sides)
        rng.shuffle(blocked_order)
        forced_blocked = set(blocked_order[:blocked_side_count])

        side_order = list(SIDE_NAMES)
        rng.shuffle(side_order)

        success = True
        for side in side_order:
            if side in forced_blocked:
                cand = _pick_candidate(
                    by_side[side],
                    rng,
                    used_sources,
                    used_dests,
                    blocked=True,
                )
            else:
                cand = _pick_candidate(
                    by_side[side],
                    rng,
                    used_sources,
                    used_dests,
                    blocked=False,
                )
                if cand is None:
                    cand = _pick_candidate(by_side[side], rng, used_sources, used_dests)
            if cand is None:
                success = False
                break
            _add_candidate(cand, selected, used_sources, used_dests)

        if not success:
            continue

        if not any(cand.far for cand in selected):
            far_cand = _pick_candidate(candidates, rng, used_sources, used_dests, far=True)
            if far_cand is None:
                continue
            _add_candidate(far_cand, selected, used_sources, used_dests)

        while sum(cand.blocked for cand in selected) < 2:
            blocked_cand = _pick_candidate(candidates, rng, used_sources, used_dests, blocked=True)
            if blocked_cand is None:
                break
            _add_candidate(blocked_cand, selected, used_sources, used_dests)

        remaining = list(candidates)
        rng.shuffle(remaining)
        for cand in remaining:
            if len(selected) >= target_count:
                break
            if _compatible(cand, used_sources, used_dests):
                _add_candidate(cand, selected, used_sources, used_dests)

        blocked_count = sum(cand.blocked for cand in selected)
        direct_count = len(selected) - blocked_count
        if (
            len(selected) >= PIXEL_COUNT_MIN
            and blocked_count >= 1
            and direct_count >= 1
            and any(cand.far for cand in selected)
            and len({cand.side for cand in selected}) == 4
        ):
            return selected

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    width = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    side_palette = rng.choice(np.array(FRAME_COLOR_POOL, dtype=int), size=4, replace=False)
    side_colors = {side: int(color) for side, color in zip(SIDE_NAMES, side_palette)}

    obstacle_mask = _place_obstacles(height, width, rng)
    if obstacle_mask is None:
        return None

    candidates = _build_candidates(obstacle_mask, side_colors)
    if not candidates:
        return None

    selected = _select_pixels(candidates, rng)
    if selected is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, 1:-1] = side_colors["top"]
    grid[height - 1, 1:-1] = side_colors["bottom"]
    grid[1:-1, 0] = side_colors["left"]
    grid[1:-1, width - 1] = side_colors["right"]
    grid[obstacle_mask] = GREY

    for cand in selected:
        r, c = cand.source
        grid[r, c] = cand.color

    out = _transform(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        if np.array_equal(main(grid), _transform(grid)):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
