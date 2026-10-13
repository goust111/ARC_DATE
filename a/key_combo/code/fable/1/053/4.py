from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
VELOCITY_VALUES = (-2, -1, 0, 1, 2)
SPECIAL_BOUNCE_MODES = ("h", "v", "corner")
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []
    for start_r, start_c in np.argwhere(mask):
        r0 = int(start_r)
        c0 = int(start_c)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        coord_array = np.array(coords, dtype=int)
        out.append(
            {
                "color": int(color),
                "coords": coord_array,
                "bbox": _bbox_from_coords(coord_array),
                "area": int(len(coords)),
            }
        )
    return out


def _is_hollow_frame(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(patch == color, expected)


def _detect_panels(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for color in map(int, np.unique(grid)):
        for comp in _components_of_color(grid, color):
            if not _is_hollow_frame(grid, comp):
                continue
            r0, c0, r1, c1 = comp["bbox"]
            groups[(int(color), r1 - r0 + 1, c1 - c0 + 1)].append(comp["bbox"])

    candidates = [(key, boxes) for key, boxes in groups.items() if len(boxes) >= 2]
    assert candidates, "expected at least two same-sized framed panels"

    max_count = max(len(boxes) for _key, boxes in candidates)
    strongest = [(key, boxes) for key, boxes in candidates if len(boxes) == max_count]
    assert len(strongest) == 1, "panel frames must be uniquely identifiable"
    (wall_color, panel_h, panel_w), boxes = strongest[0]

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = {r0 for r0, _c0, _r1, _c1 in boxes}
    assert len(rows) == 1, "panels must lie in one row"
    for r0, c0, r1, c1 in boxes:
        assert r1 - r0 + 1 == panel_h and c1 - c0 + 1 == panel_w
    return int(wall_color), int(panel_h), int(panel_w), boxes


def _extract_panel_points(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    wall_color: int,
) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch[0, :] == wall_color)
    assert np.all(patch[-1, :] == wall_color)
    assert np.all(patch[:, 0] == wall_color)
    assert np.all(patch[:, -1] == wall_color)

    inner = patch[1:-1, 1:-1]
    colors = sorted(set(map(int, inner.ravel())) - {int(bg)})
    points: dict[int, tuple[int, int]] = {}
    for color in colors:
        coords = np.argwhere(inner == int(color))
        assert coords.shape == (1, 2), "each moving object must be a unique single pixel"
        points[int(color)] = (int(coords[0, 0]), int(coords[0, 1]))
    return points


def _reflect_1d(pos: int, delta: int, limit: int) -> int:
    if limit <= 1:
        return 0
    period = 2 * (limit - 1)
    mirrored = (int(pos) + int(delta)) % period
    if mirrored > limit - 1:
        mirrored = period - mirrored
    return int(mirrored)


def _base_panel(panel_h: int, panel_w: int, bg: int, wall_color: int) -> np.ndarray:
    panel = np.full((panel_h, panel_w), int(bg), dtype=int)
    panel[0, :] = int(wall_color)
    panel[-1, :] = int(wall_color)
    panel[:, 0] = int(wall_color)
    panel[:, -1] = int(wall_color)
    return panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next framed panel by advancing each colored dot one more step with wall bounces.

    Input structure:
    - The grid contains one horizontal sequence of equal-sized framed panels on a shared background.
    - Every panel has the same one-cell wall border and contains the same set of differently colored single-pixel dots.
    - The dots move inside the panel interior while colors stay unique, so each color identifies one trajectory across time.

    Rule:
    - Detect the framed panels and sort them from left to right.
    - For each dot color, read its positions in the last two panels to obtain the current displacement vector.
    - Advance that dot once more inside the panel interior; whenever a row or column step would cross a wall, mirror-reflect that axis and finish the remaining distance.
    - Output one panel with the unchanged wall/background and all dots drawn at their predicted next positions.

    Key insight:
    - The final transition is not plain linear extrapolation: the last observed displacement is the motion vector, but the next step must be reflected against the panel interior whenever it would leave the box.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall_color, panel_h, panel_w, boxes = _detect_panels(g)
    states = [_extract_panel_points(g, box, bg, wall_color) for box in boxes]
    assert len(states) >= 2, "need at least two panels to infer motion"

    colors = set(states[0])
    assert colors, "expected at least one moving dot"
    for state in states[1:]:
        assert set(state) == colors, "every panel must contain the same colored dots"

    out = _base_panel(panel_h, panel_w, bg, wall_color)
    inner_h = panel_h - 2
    inner_w = panel_w - 2
    occupied: set[tuple[int, int]] = set()

    for color in sorted(colors):
        prev_r, prev_c = states[-2][color]
        cur_r, cur_c = states[-1][color]
        dr = cur_r - prev_r
        dc = cur_c - prev_c
        next_r = _reflect_1d(cur_r, dr, inner_h)
        next_c = _reflect_1d(cur_c, dc, inner_w)
        assert (next_r, next_c) not in occupied, "predicted dots must not overlap"
        occupied.add((next_r, next_c))
        out[next_r + 1, next_c + 1] = int(color)

    return out


def _axis_positions(last: int, delta: int, n_panels: int) -> list[int]:
    return [int(last - (n_panels - 1 - t) * delta) for t in range(n_panels)]


def _axis_last_options(limit: int, delta: int, n_panels: int, should_bounce: bool) -> list[int]:
    options = []
    for last in range(limit):
        positions = _axis_positions(last, delta, n_panels)
        if not all(0 <= pos < limit for pos in positions):
            continue
        bounced = not (0 <= last + delta < limit)
        if bounced == bool(should_bounce):
            options.append(int(last))
    return options


def _build_candidates(inner_h: int, inner_w: int, n_panels: int, mode: str) -> list[dict]:
    bounce_r = mode in {"v", "corner"}
    bounce_c = mode in {"h", "corner"}
    candidates: list[dict] = []
    for dr in VELOCITY_VALUES:
        if bounce_r and dr == 0:
            continue
        row_lasts = _axis_last_options(inner_h, dr, n_panels, bounce_r)
        if not row_lasts:
            continue
        for dc in VELOCITY_VALUES:
            if bounce_c and dc == 0:
                continue
            if dr == 0 and dc == 0:
                continue
            col_lasts = _axis_last_options(inner_w, dc, n_panels, bounce_c)
            if not col_lasts:
                continue
            for last_r in row_lasts:
                row_positions = _axis_positions(last_r, dr, n_panels)
                next_r = _reflect_1d(last_r, dr, inner_h)
                for last_c in col_lasts:
                    col_positions = _axis_positions(last_c, dc, n_panels)
                    frames = [(r, c) for r, c in zip(row_positions, col_positions)]
                    next_pos = (next_r, _reflect_1d(last_c, dc, inner_w))
                    candidates.append(
                        {
                            "delta": (int(dr), int(dc)),
                            "frames": frames,
                            "next": next_pos,
                            "mode": mode,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2


def _compatible(candidate: dict, chosen: list[dict], n_panels: int) -> bool:
    for other in chosen:
        for t in range(n_panels):
            if not _far_enough(candidate["frames"][t], other["frames"][t]):
                return False
        if not _far_enough(candidate["next"], other["next"]):
            return False
    return True


def _render_panel(
    panel_h: int,
    panel_w: int,
    bg: int,
    wall_color: int,
    dots: dict[int, tuple[int, int]],
) -> np.ndarray:
    panel = _base_panel(panel_h, panel_w, bg, wall_color)
    for color, (r, c) in dots.items():
        panel[r + 1, c + 1] = int(color)
    return panel


def _clamp_1d(pos: int, delta: int, limit: int) -> int:
    return max(0, min(limit - 1, int(pos) + int(delta)))


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    specs: list[tuple[int, dict]],
    n_panels: int,
    panel_h: int,
    panel_w: int,
    bg: int,
    wall_color: int,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    parsed_wall, parsed_h, parsed_w, boxes = _detect_panels(grid)
    if (parsed_wall, parsed_h, parsed_w, len(boxes)) != (wall_color, panel_h, panel_w, n_panels):
        return False

    states = [_extract_panel_points(grid, box, bg, wall_color) for box in boxes]
    expected_colors = {color for color, _spec in specs}
    if any(set(state) != expected_colors for state in states):
        return False

    any_bounce = False
    any_nonbounce = False
    differs_from_clamp = False
    for color, spec in specs:
        if any(states[t][color] != spec["frames"][t] for t in range(n_panels)):
            return False
        last_r, last_c = spec["frames"][-1]
        dr, dc = spec["delta"]
        row_bounce = not (0 <= last_r + dr < panel_h - 2)
        col_bounce = not (0 <= last_c + dc < panel_w - 2)
        any_bounce |= row_bounce or col_bounce
        any_nonbounce |= not (row_bounce or col_bounce)
        predicted = spec["next"]
        clamped = (
            _clamp_1d(last_r, dr, panel_h - 2),
            _clamp_1d(last_c, dc, panel_w - 2),
        )
        if predicted != clamped:
            differs_from_clamp = True

    if not any_bounce or not any_nonbounce or not differs_from_clamp:
        return False

    input_panels = [_render_panel(panel_h, panel_w, bg, wall_color, state) for state in states]
    if any(np.array_equal(expected, panel) for panel in input_panels):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    bg = 0
    wall_color = random.randint(1, 9)
    n_panels = random.choice([3, 4])
    panel_h = random.randint(7, 10)
    panel_w = random.randint(6, 8)
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(n_panels - 1)]
    grid_h = top + panel_h + bottom
    grid_w = left + right + n_panels * panel_w + sum(gaps)
    if grid_h > 30 or grid_w > 30:
        return None

    inner_h = panel_h - 2
    inner_w = panel_w - 2
    if inner_h < 4 or inner_w < 4:
        return None

    mode_candidates = {mode: _build_candidates(inner_h, inner_w, n_panels, mode) for mode in ("none",) + SPECIAL_BOUNCE_MODES}
    special_modes = [mode for mode in SPECIAL_BOUNCE_MODES if mode_candidates[mode]]
    if not special_modes or not mode_candidates["none"]:
        return None

    dot_count = random.choice([2, 3])
    special_mode = random.choice(special_modes)
    modes = [special_mode, "none"]
    while len(modes) < dot_count:
        modes.append(random.choice(["none", "h", "v", "corner"]))
    random.shuffle(modes)

    color_pool = [color for color in range(1, 10) if color != wall_color]
    if len(color_pool) < dot_count:
        return None
    dot_colors = random.sample(color_pool, dot_count)

    chosen: list[dict] = []
    for mode in modes:
        picked = None
        for candidate in mode_candidates[mode]:
            if _compatible(candidate, chosen, n_panels):
                picked = candidate
                break
        if picked is None:
            return None
        chosen.append(picked)

    specs = list(zip(dot_colors, chosen))
    panel_states: list[dict[int, tuple[int, int]]] = []
    for t in range(n_panels):
        state = {color: spec["frames"][t] for color, spec in specs}
        if len(set(state.values())) != len(state):
            return None
        panel_states.append(state)

    next_state = {color: spec["next"] for color, spec in specs}
    if len(set(next_state.values())) != len(next_state):
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    col = left
    for t in range(n_panels):
        panel = _render_panel(panel_h, panel_w, bg, wall_color, panel_states[t])
        grid[top : top + panel_h, col : col + panel_w] = panel
        col += panel_w
        if t < len(gaps):
            col += gaps[t]

    expected = _render_panel(panel_h, panel_w, bg, wall_color, next_state)
    if not _validate_instance(grid, expected, specs, n_panels, panel_h, panel_w, bg, wall_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-panel input")
