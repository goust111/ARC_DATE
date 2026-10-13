from __future__ import annotations

import numpy as np


BG = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4
CENTER = 9

ORDER = (
    ("left", 0, -1, LEFT),
    ("down", 1, 0, DOWN),
    ("right", 0, 1, RIGHT),
    ("up", -1, 0, UP),
)
DIRECTION_COLORS = {name: color for name, _dr, _dc, color in ORDER}

_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_GEN_HEIGHT_CHOICES = np.array([15, 17, 19, 21], dtype=int)
_GEN_WIDTH_CHOICES = np.array([15, 17, 19, 21], dtype=int)
_GEN_MAX_ATTEMPTS = 500
_GEN_PLANS = (
    "paired_return",
    "all_distinct_reenter",
    "mixed_reenter",
)
_CENTER_OFFSETS_SMALL = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
    (-1, 1),
    (1, -1),
)
_CENTER_OFFSETS_LARGE = (
    (-3, -1),
    (-3, 1),
    (-2, -2),
    (-2, 2),
    (-1, -3),
    (-1, 3),
    (1, -3),
    (1, 3),
    (2, -2),
    (2, 2),
    (3, -1),
    (3, 1),
)


def _run_length(values: np.ndarray, color: int) -> int:
    count = 0
    for value in values:
        if int(value) != color:
            break
        count += 1
    return count


def _parse_input(grid: np.ndarray) -> tuple[tuple[int, int], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "grid must have an interior"

    allowed = {BG, DOWN, UP, LEFT, RIGHT, CENTER}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"

    center_coords = np.argwhere(g == CENTER)
    assert center_coords.shape == (1, 2), "expected exactly one center marker"
    center = (int(center_coords[0, 0]), int(center_coords[0, 1]))

    steps = {
        "up": _run_length(g[0, :], UP),
        "down": _run_length(g[h - 1, :], DOWN),
        "left": _run_length(g[1:, 0], LEFT),
        "right": _run_length(g[1:, w - 1], RIGHT),
    }
    assert all(1 <= value <= 4 for value in steps.values()), "ruler lengths must be 1-4"
    assert len(set(steps.values())) >= 2, "at least two ruler lengths must differ"

    expected = np.zeros_like(g)
    expected[0, : steps["up"]] = UP
    expected[h - 1, : steps["down"]] = DOWN
    expected[1 : 1 + steps["left"], 0] = LEFT
    expected[1 : 1 + steps["right"], w - 1] = RIGHT
    expected[center] = CENTER
    assert np.array_equal(g, expected), "input must contain only the four rulers and one center marker"
    return center, steps


def _simulate_spiral(
    shape: tuple[int, int],
    center: tuple[int, int],
    steps: dict[str, int],
    *,
    protect_center: bool,
) -> tuple[np.ndarray, dict]:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    out[center] = CENTER

    r, c = center
    counts = {name: 0 for name, _dr, _dc, _color in ORDER}
    segments = []
    max_cycles = 12 * (h + w)
    stable_cycles = 0

    for _cycle in range(max_cycles):
        cycle_visible = 0
        cycle_changes = 0
        for name, dr, dc, color in ORDER:
            counts[name] += 1
            length = counts[name] * int(steps[name])
            start_r, start_c = r, c
            first_visible_step = None
            visible_count = 0
            center_hits = 0
            changed_count = 0

            for step_index in range(1, length + 1):
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    continue

                if first_visible_step is None:
                    first_visible_step = step_index
                visible_count += 1
                old_value = int(out[r, c])
                if (r, c) == center:
                    center_hits += 1
                    if not protect_center:
                        out[r, c] = color
                else:
                    out[r, c] = color
                if int(out[r, c]) != old_value:
                    changed_count += 1

            cycle_visible += visible_count
            cycle_changes += changed_count
            segments.append(
                {
                    "direction": name,
                    "length": length,
                    "start": (start_r, start_c),
                    "end": (r, c),
                    "start_in_bounds": 0 <= start_r < h and 0 <= start_c < w,
                    "end_in_bounds": 0 <= r < h and 0 <= c < w,
                    "visible_count": visible_count,
                    "first_visible_step": first_visible_step,
                    "center_hits": center_hits,
                    "changed_count": changed_count,
                }
            )

        if cycle_visible == 0:
            break
        if cycle_changes == 0:
            stable_cycles += 1
            if stable_cycles >= 3:
                break
        else:
            stable_cycles = 0
    else:
        raise RuntimeError("spiral simulation did not terminate")

    return out, {"segments": segments}


def _stop_at_first_clip_output(
    shape: tuple[int, int],
    center: tuple[int, int],
    steps: dict[str, int],
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    out[center] = CENTER
    r, c = center
    counts = {name: 0 for name, _dr, _dc, _color in ORDER}

    while True:
        for name, dr, dc, color in ORDER:
            counts[name] += 1
            length = counts[name] * int(steps[name])
            for _ in range(length):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    return out
                r, c = nr, nc
                if (r, c) != center:
                    out[r, c] = color


def _shared_axis_output(
    shape: tuple[int, int],
    center: tuple[int, int],
    steps: dict[str, int],
) -> np.ndarray:
    shared = {
        "left": int(steps["left"]),
        "right": int(steps["left"]),
        "up": int(steps["up"]),
        "down": int(steps["up"]),
    }
    return _simulate_spiral(shape, center, shared, protect_center=True)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert four edge rulers into a four-color clipped spiral with one independent growth rate per direction.

    Input structure:
    - A black grid contains exactly one maroon center marker near the middle.
    - A red ruler lies on the top edge from the left corner, a blue ruler lies on the bottom edge from the left corner,
      a green ruler lies down the left edge starting one row below the top, and a yellow ruler does the same on the right edge.
    - Each ruler length is 1-4 and encodes the multiplier for one spiral direction.

    Rule:
    - Erase the four rulers back to black and keep the maroon center marker.
    - Starting at the marker, trace directions left, down, right, up, repeating forever in that order.
    - On the k-th segment of a direction, use length k*S_direction and paint that whole segment in that direction's ruler color.
    - The walk is traced in the plane and clipped to the canvas: only in-bounds cells are painted, later segments overwrite earlier ones,
      and the center marker never changes color.
    - Stop once an entire left-down-right-up cycle adds no visible contribution any more:
      either the whole cycle is off-grid, or repeated clipped passes no longer change the canvas.

    Key insight:
    - The border bars are consumed as four separate numeric controls, so the visible spiral is not a uniform spiral:
      each direction grows at its own rate and clipped off-grid excursions can later re-enter the canvas.
    """
    center, steps = _parse_input(np.asarray(grid, dtype=int))
    return _simulate_spiral(np.asarray(grid).shape, center, steps, protect_center=True)[0]


def _build_input(
    shape: tuple[int, int],
    center: tuple[int, int],
    steps: dict[str, int],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG, dtype=int)
    grid[0, : steps["up"]] = UP
    grid[h - 1, : steps["down"]] = DOWN
    grid[1 : 1 + steps["left"], 0] = LEFT
    grid[1 : 1 + steps["right"], w - 1] = RIGHT
    grid[center] = CENTER
    return grid


def _choose_shape() -> tuple[int, int]:
    h = int(_GEN_RNG.choice(_GEN_HEIGHT_CHOICES))
    w = int(_GEN_RNG.choice(_GEN_WIDTH_CHOICES))
    return h, w


def _choose_center(shape: tuple[int, int], plan: str) -> tuple[int, int]:
    h, w = shape
    offsets = _CENTER_OFFSETS_LARGE if plan != "paired_return" else _CENTER_OFFSETS_SMALL
    for _ in range(50):
        off_r, off_c = offsets[int(_GEN_RNG.integers(len(offsets)))]
        r = h // 2 + int(off_r)
        c = w // 2 + int(off_c)
        if 2 <= r < h - 2 and 2 <= c < w - 2:
            return r, c
    raise RuntimeError("failed to pick a center offset")


def _sample_steps(plan: str) -> dict[str, int]:
    if plan == "paired_return":
        picks = _GEN_RNG.choice(np.array([1, 2, 3, 4], dtype=int), size=2, replace=False)
        a, b = int(picks[0]), int(picks[1])
        return {"left": a, "down": b, "right": a, "up": b}

    if plan == "all_distinct_reenter":
        perm = [int(v) for v in _GEN_RNG.permutation(np.array([1, 2, 3, 4], dtype=int))]
        return {"left": perm[0], "down": perm[1], "right": perm[2], "up": perm[3]}

    while True:
        vals = [int(v) for v in _GEN_RNG.integers(1, 5, size=4)]
        steps = {"left": vals[0], "down": vals[1], "right": vals[2], "up": vals[3]}
        if len(set(vals)) < 3:
            continue
        if steps["left"] == steps["right"] or steps["up"] == steps["down"]:
            continue
        if 4 not in vals or 1 not in vals:
            continue
        return steps


def _all_spiral_colors_visible(out: np.ndarray) -> bool:
    return all(int(np.count_nonzero(out == color)) >= 2 for color in (LEFT, DOWN, RIGHT, UP))


def _has_partial_clip(segments: list[dict]) -> bool:
    return any(0 < seg["visible_count"] < seg["length"] for seg in segments)


def _has_invisible_then_visible(segments: list[dict]) -> bool:
    seen_invisible = False
    for seg in segments:
        if seg["visible_count"] == 0:
            seen_invisible = True
        elif seen_invisible:
            return True
    return False


def _has_reentry_within_segment(segments: list[dict]) -> bool:
    return any(
        seg["first_visible_step"] is not None and int(seg["first_visible_step"]) > 1
        for seg in segments
    )


def _center_is_offset(center: tuple[int, int], shape: tuple[int, int]) -> bool:
    return center != (shape[0] // 2, shape[1] // 2)


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    center: tuple[int, int],
    steps: dict[str, int],
    trace: dict,
    plan: str,
) -> bool:
    h, w = map(int, inp.shape)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not _center_is_offset(center, inp.shape):
        return False
    if not np.array_equal(inp, _build_input(inp.shape, center, steps)):
        return False
    if not _all_spiral_colors_visible(out):
        return False
    if int(out[center]) != CENTER:
        return False
    if not _has_partial_clip(trace["segments"]):
        return False

    fill_ratio = float(np.count_nonzero(out != BG)) / float(h * w)
    if not (0.08 <= fill_ratio <= 0.60):
        return False

    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(_stop_at_first_clip_output(inp.shape, center, steps), out):
        return False

    if plan == "paired_return":
        if steps["left"] != steps["right"] or steps["up"] != steps["down"]:
            return False
        if steps["left"] == steps["up"]:
            return False
        if not any(seg["center_hits"] > 0 for seg in trace["segments"]):
            return False
        overwritten = _simulate_spiral(inp.shape, center, steps, protect_center=False)[0]
        if np.array_equal(overwritten, out):
            return False
    elif plan == "all_distinct_reenter":
        if sorted(steps.values()) != [1, 2, 3, 4]:
            return False
        if not _has_reentry_within_segment(trace["segments"]):
            return False
        if np.array_equal(_shared_axis_output(inp.shape, center, steps), out):
            return False
    else:
        if len(set(steps.values())) < 3:
            return False
        if not _has_invisible_then_visible(trace["segments"]):
            return False
        if np.array_equal(_shared_axis_output(inp.shape, center, steps), out):
            return False

    return True


def _gen_attempt(plan: str) -> np.ndarray | None:
    shape = _choose_shape()
    center = _choose_center(shape, plan)
    steps = _sample_steps(plan)
    inp = _build_input(shape, center, steps)
    out, trace = _simulate_spiral(shape, center, steps, protect_center=True)
    return inp if _semantic_checks(inp, out, center, steps, trace, plan) else None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    plan = _GEN_PLANS[_GEN_CALL_COUNT % len(_GEN_PLANS)]
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(plan)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle-8 input for plan {plan}")
