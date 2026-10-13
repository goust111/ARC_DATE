import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAROON = 9

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (10, 16)
OBJECT_W_RANGE = (5, 7)
GENERATE_RETRIES = 500


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    return runs


def _find_object_span(grid: np.ndarray) -> tuple[int, int]:
    mask = (grid[0] == GREEN) & (grid[1] == GREEN) & np.isin(grid[2], [BLUE, MAROON])
    runs = _contiguous_runs(mask)
    assert len(runs) == 1, "expected exactly one top control object"
    return runs[0]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _fragment_info(coords: np.ndarray) -> dict[str, object]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    left_offsets = {int(r - r0) for r, c in coords if int(c) == c0}
    right_offsets = {int(r - r0) for r, c in coords if int(c) == c1}
    return {
        "coords": coords,
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "left_offsets": left_offsets,
        "right_offsets": right_offsets,
    }


def _matching_fragments(infos: list[dict[str, object]]) -> set[int]:
    matched: set[int] = set()
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            left_idx, right_idx = (i, j)
            if int(infos[j]["c0"]) < int(infos[i]["c0"]):
                left_idx, right_idx = j, i
            left = infos[left_idx]
            right = infos[right_idx]
            if set(left["right_offsets"]) & set(right["left_offsets"]):
                matched.add(left_idx)
                matched.add(right_idx)
    return matched


def _transform(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    c0, c1 = _find_object_span(g)

    out = g.copy()
    out[0:3, c0 : c1 + 1] = BG

    for c in range(c0, c1 + 1):
        control = int(g[2, c])
        if control == BLUE:
            below = np.flatnonzero(g[3:, c] == GREY)
            if len(below):
                first_obstacle = int(below[0] + 3)
                top = first_obstacle - 3
            else:
                top = H - 3
        else:
            assert control == MAROON, "control row must contain only blue or maroon"
            top = H - 3
        assert 0 <= top <= H - 2
        assert np.all(out[top : top + 2, c] != GREY), "green data pixels must not overlap obstacles"
        out[top : top + 2, c] = GREEN

    components = _components_of_color(out, GREEN)
    infos = [_fragment_info(coords) for coords in components]
    matched = _matching_fragments(infos)

    for idx in matched:
        coords = infos[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = RED

    if return_meta:
        return out, infos, matched
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top controlled object by column, then recolor fragments whose facing edges share a row offset.

    Input structure:
    - The background is black, with one wide 3-row object touching the top edge.
    - The object's first two rows are green data pixels, and its third row is a blue/maroon control strip.
    - Grey obstacle pixels or short grey horizontal bars are scattered in the middle and lower part of the grid.

    Rule:
    - Erase the original top object and treat each object column as an independent 3-pixel stack.
    - Blue-controlled columns stop with their control pixel directly above the first grey obstacle below them, while maroon-controlled columns ignore obstacles and bottom-align.
    - Remove every landed control pixel, keep only the two green data pixels per column, split the result into 4-connected green fragments, and compare each left fragment's right edge with each right fragment's left edge using row offsets relative to each fragment's own top.
    - Recolor every fragment that has at least one such edge-offset overlap to red, and leave all other fragments green while keeping grey obstacles fixed.

    Key insight:
    - The control strip makes different columns fall to different heights, and the remaining puzzle is to detect which separated fragments still have compatible facing edge profiles.
    """
    return _transform(grid, return_meta=False)


def _right_edge_low_pattern(length: int, base: int) -> list[int]:
    assert length >= 2
    return [base + 1] + [base] * (length - 1)


def _left_edge_low_pattern(length: int, base: int) -> list[int]:
    assert length >= 2
    return [base] * (length - 1) + [base + 1]


def _ascending_pattern(length: int, base: int) -> list[int]:
    assert length >= 2
    return [base + i for i in range(length)]


def _descending_pattern(length: int, base: int) -> list[int]:
    assert length >= 2
    return [base + length - 1 - i for i in range(length)]


def _sample_base(H: int, max_raise: int) -> int:
    low = 3
    high = H - 5 - max_raise
    assert low <= high
    return random.randint(low, high)


def _scenario_two_match(H: int, width: int) -> tuple[list[int], list[int], str]:
    blue_len = random.randint(2, width - 2)
    maroon_len = width - blue_len
    blue_on_left = random.choice([True, False])
    if blue_on_left:
        blue_tops = _right_edge_low_pattern(blue_len, _sample_base(H, 1))
        tops = blue_tops + [H - 3] * maroon_len
        controls = [BLUE] * blue_len + [MAROON] * maroon_len
    else:
        blue_tops = _left_edge_low_pattern(blue_len, _sample_base(H, 1))
        tops = [H - 3] * maroon_len + blue_tops
        controls = [MAROON] * maroon_len + [BLUE] * blue_len
    return controls, tops, "two_match"


def _scenario_two_no_match(H: int, width: int) -> tuple[list[int], list[int], str]:
    blue_len = random.randint(3, width - 2)
    maroon_len = width - blue_len
    blue_on_left = random.choice([True, False])
    if blue_on_left:
        blue_tops = _ascending_pattern(blue_len, _sample_base(H, blue_len - 1))
        tops = blue_tops + [H - 3] * maroon_len
        controls = [BLUE] * blue_len + [MAROON] * maroon_len
    else:
        blue_tops = _descending_pattern(blue_len, _sample_base(H, blue_len - 1))
        tops = [H - 3] * maroon_len + blue_tops
        controls = [MAROON] * maroon_len + [BLUE] * blue_len
    return controls, tops, "two_no_match"


def _scenario_three_mix(H: int) -> tuple[list[int], list[int], str]:
    match_base = _sample_base(H, 1)
    no_match_base = _sample_base(H, 2)
    if random.choice([True, False]):
        controls = [BLUE, BLUE, MAROON, MAROON, BLUE, BLUE, BLUE]
        tops = _right_edge_low_pattern(2, match_base) + [H - 3, H - 3] + _descending_pattern(3, no_match_base)
    else:
        controls = [BLUE, BLUE, BLUE, MAROON, MAROON, BLUE, BLUE]
        tops = _ascending_pattern(3, no_match_base) + [H - 3, H - 3] + _left_edge_low_pattern(2, match_base)
    return controls, tops, "three_mix"


def _build_column_spec(H: int) -> tuple[list[int], list[int], str]:
    scenario = random.choices(
        ["two_match", "two_no_match", "three_mix"],
        weights=[35, 30, 35],
        k=1,
    )[0]
    if scenario == "three_mix":
        return _scenario_three_mix(H)
    width = random.randint(*OBJECT_W_RANGE)
    if scenario == "two_match":
        return _scenario_two_match(H, width)
    return _scenario_two_no_match(H, width)


def _stamp_required_obstacles(
    grid: np.ndarray,
    c_start: int,
    controls: list[int],
    tops: list[int],
) -> set[tuple[int, int]]:
    H, _ = grid.shape
    reserved: set[tuple[int, int]] = set()
    width = len(controls)
    for r in range(3):
        for c in range(c_start, c_start + width):
            reserved.add((r, c))
    for idx, control in enumerate(controls):
        c = c_start + idx
        top = tops[idx]
        reserved.add((top, c))
        reserved.add((top + 1, c))
        if control == BLUE:
            obstacle_r = top + 3
            grid[obstacle_r, c] = GREY
            for r in range(3, obstacle_r + 1):
                reserved.add((r, c))
        else:
            for r in range(H - 3, H):
                reserved.add((r, c))
    return reserved


def _maybe_add_maroon_path_obstacle(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    c_start: int,
    controls: list[int],
) -> None:
    H, W = grid.shape
    maroon_cols = [c_start + i for i, control in enumerate(controls) if control == MAROON]
    if not maroon_cols or random.random() >= 0.6:
        return
    random.shuffle(maroon_cols)
    for c in maroon_cols:
        candidates = [r for r in range(4, H - 3) if (r, c) not in reserved and int(grid[r, c]) == BG]
        if not candidates:
            continue
        r = random.choice(candidates)
        grid[r, c] = GREY
        return


def _add_noise_obstacles(grid: np.ndarray, reserved: set[tuple[int, int]]) -> None:
    H, W = grid.shape
    target = random.randint(4, 9)
    placed = 0
    for _ in range(200):
        if placed >= target:
            break
        length = random.choice([1, 1, 2, 2, 3])
        r = random.randint(4, H - 2)
        c0 = random.randint(0, W - length)
        cells = [(r, c0 + dc) for dc in range(length)]
        if any(cell in reserved or int(grid[cell]) != BG for cell in cells):
            continue
        for cell in cells:
            grid[cell] = GREY
            reserved.add(cell)
        placed += 1


def _validate_generated(grid: np.ndarray, scenario: str) -> bool:
    try:
        out, infos, matched = _transform(grid, return_meta=True)
    except AssertionError:
        return False
    if len(infos) < 2:
        return False
    if scenario == "two_match" and not matched:
        return False
    if scenario == "two_no_match" and matched:
        return False
    if scenario == "three_mix":
        if len(infos) < 3:
            return False
        if not matched or len(matched) == len(infos):
            return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    controls, tops, scenario = _build_column_spec(H)
    width = len(controls)
    if width > W:
        return None

    c_start = random.randint(0, W - width)
    grid = np.full((H, W), BG, dtype=int)
    grid[0:2, c_start : c_start + width] = GREEN
    grid[2, c_start : c_start + width] = np.array(controls, dtype=int)

    reserved = _stamp_required_obstacles(grid, c_start, controls, tops)
    _maybe_add_maroon_path_obstacle(grid, reserved, c_start, controls)
    _add_noise_obstacles(grid, reserved)

    blue_tops = [tops[i] for i, control in enumerate(controls) if control == BLUE]
    if len(set(blue_tops)) < 2:
        return None
    if not _validate_generated(grid, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
