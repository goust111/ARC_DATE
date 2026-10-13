import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


PANEL_H = 6
SPRITE = np.full((2, 2), Color.RED, dtype=int)
SEPARATOR = Color.GREY
GEN_ATTEMPTS = 400


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    assert len(sep_rows) == 3, "expected exactly three separator rows"
    cuts = [-1, *sep_rows, g.shape[0]]
    panels = [g[cuts[i] + 1 : cuts[i + 1], :].copy() for i in range(4)]
    heights = {panel.shape[0] for panel in panels}
    widths = {panel.shape[1] for panel in panels}
    assert len(heights) == 1 and len(widths) == 1, "panels must share a single size"
    return panels


def _closed_sides(panel: np.ndarray) -> dict[str, bool]:
    sides = {
        "top": bool(np.all(panel[0, :] == Color.BLUE)),
        "bottom": bool(np.all(panel[-1, :] == Color.BLUE)),
        "left": bool(np.all(panel[:, 0] == Color.BLUE)),
        "right": bool(np.all(panel[:, -1] == Color.BLUE)),
    }
    assert sum(sides.values()) == 3, "panel must have exactly one missing frame side"
    return sides


def _gap_side(closed: dict[str, bool]) -> str:
    for side, is_closed in closed.items():
        if not is_closed:
            return side
    raise AssertionError("missing gap side")


def _extract_sprite(panel: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    red_components = find_components(
        panel,
        bg=Color.BLACK,
        colors=[Color.RED],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(red_components) == 1, "expected exactly one red sprite"
    red_box = bbox(panel == Color.RED)
    assert red_box is not None and tuple(red_box) == tuple(red_components[0]["bbox"])
    patch = crop_bbox(panel, red_box)
    return patch, (int(red_box[0]), int(red_box[1]))


def _occupied_extents(sprite: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(sprite != Color.BLACK)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _safe_bounds(panel_shape: tuple[int, int], sprite: np.ndarray, closed: dict[str, bool]) -> tuple[int, int, int, int]:
    height, width = map(int, panel_shape)
    occ_r0, occ_c0, occ_r1, occ_c1 = _occupied_extents(sprite)
    row_low = 1 - occ_r0 if closed["top"] else -occ_r0
    row_high = height - 2 - occ_r1 if closed["bottom"] else height - 1 - occ_r1
    col_low = 1 - occ_c0 if closed["left"] else -occ_c0
    col_high = width - 2 - occ_c1 if closed["right"] else width - 1 - occ_c1
    assert row_low <= row_high and col_low <= col_high
    return int(row_low), int(row_high), int(col_low), int(col_high)


def _reflect_between(pos: int, low: int, high: int) -> int:
    if low == high:
        return int(low)
    span = int(high - low)
    period = 2 * span
    phase = (int(pos) - int(low)) % period
    if phase > span:
        phase = period - phase
    return int(low + phase)


def _resolve_against_bounds(pos: int, low: int, high: int, closed_low: bool, closed_high: bool) -> int:
    pos = int(pos)
    if closed_low and closed_high:
        return _reflect_between(pos, low, high)
    if closed_low and pos < low:
        return int(2 * low - pos)
    if closed_high and pos > high:
        return int(2 * high - pos)
    return pos


def _predict_next_position(panel: np.ndarray, sprite: np.ndarray, pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    closed = _closed_sides(panel)
    row_low, row_high, col_low, col_high = _safe_bounds(panel.shape, sprite, closed)
    candidate_r = int(pos[0] + delta[0])
    candidate_c = int(pos[1] + delta[1])
    final_r = _resolve_against_bounds(candidate_r, row_low, row_high, closed["top"], closed["bottom"])
    final_c = _resolve_against_bounds(candidate_c, col_low, col_high, closed["left"], closed["right"])
    return int(final_r), int(final_c)


def _base_panel(width: int, gap_side: str) -> np.ndarray:
    panel = np.zeros((PANEL_H, int(width)), dtype=int)
    if gap_side != "top":
        panel[0, :] = Color.BLUE
    if gap_side != "bottom":
        panel[-1, :] = Color.BLUE
    if gap_side != "left":
        panel[:, 0] = Color.BLUE
    if gap_side != "right":
        panel[:, -1] = Color.BLUE
    return panel


def _render_panel(width: int, gap_side: str, pos: tuple[int, int]) -> np.ndarray:
    panel = _base_panel(width, gap_side)
    paste(panel, SPRITE, int(pos[0]), int(pos[1]), transparent=Color.BLACK, clip=False)
    return panel


def _stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    rows = []
    for idx, panel in enumerate(panels):
        rows.append(panel)
        if idx < len(panels) - 1:
            rows.append(np.full((1, panel.shape[1]), SEPARATOR, dtype=int))
    return np.vstack(rows)


def _axis_sequences(
    low: int,
    high: int,
    deltas: list[int],
    outcome: str,
    *,
    max_overshoot: int = 4,
    touch: int | None = None,
) -> list[dict[str, object]]:
    sequences = []
    for delta in deltas:
        for start in range(low, high + 1):
            positions = [int(start + step * delta) for step in range(5)]
            if not all(low <= pos <= high for pos in positions[:4]):
                continue
            candidate = positions[4]
            if outcome == "stay_inside":
                ok = low <= candidate <= high
            elif outcome == "hit_low":
                ok = low - max_overshoot <= candidate < low
            elif outcome == "hit_high":
                ok = high < candidate <= high + max_overshoot
            else:
                raise ValueError(f"unsupported axis outcome: {outcome}")
            if not ok:
                continue
            if touch is not None and touch not in positions[:4]:
                continue
            sequences.append({"delta": int(delta), "positions": tuple(positions)})
    return sequences


def _candidate_hits(meta: dict[str, object]) -> tuple[dict[str, bool], dict[str, bool]]:
    base = _base_panel(int(meta["width"]), str(meta["gap_side"]))
    closed = _closed_sides(base)
    row_low, row_high, col_low, col_high = _safe_bounds(base.shape, SPRITE, closed)
    candidate_r, candidate_c = meta["candidate"]
    closed_hits = {
        "top": closed["top"] and candidate_r < row_low,
        "bottom": closed["bottom"] and candidate_r > row_high,
        "left": closed["left"] and candidate_c < col_low,
        "right": closed["right"] and candidate_c > col_high,
    }
    gap_hits = {
        "top": (not closed["top"]) and candidate_r < row_low,
        "bottom": (not closed["bottom"]) and candidate_r > row_high,
        "left": (not closed["left"]) and candidate_c < col_low,
        "right": (not closed["right"]) and candidate_c > col_high,
    }
    return closed_hits, gap_hits


def _case_options(width: int, gap_side: str, family: str) -> list[dict[str, object]]:
    base = _base_panel(width, gap_side)
    closed = _closed_sides(base)
    row_low, row_high, col_low, col_high = _safe_bounds(base.shape, SPRITE, closed)

    if family == "horizontal_bounce":
        assert gap_side in {"left", "right"}
        row_sequences = _axis_sequences(row_low, row_high, [0], "stay_inside")
        col_touch = col_low if gap_side == "left" else col_high
        col_outcome = "hit_high" if gap_side == "left" else "hit_low"
        col_sequences = _axis_sequences(
            col_low,
            col_high,
            [-4, -3, -2, -1, 1, 2, 3, 4],
            col_outcome,
            max_overshoot=4,
            touch=col_touch,
        )
    elif family == "vertical_bounce":
        assert gap_side in {"top", "bottom"}
        row_touch = row_low if gap_side == "top" else row_high
        row_outcome = "hit_high" if gap_side == "top" else "hit_low"
        row_sequences = _axis_sequences(row_low, row_high, [-1, 1], row_outcome, max_overshoot=1, touch=row_touch)
        col_sequences = _axis_sequences(col_low, col_high, [-4, -3, -2, -1, 1, 2, 3, 4], "stay_inside")
    elif family == "gap_and_bounce":
        assert gap_side in {"top", "bottom"}
        row_touch = row_low if gap_side == "top" else row_high
        row_outcome = "hit_low" if gap_side == "top" else "hit_high"
        row_sequences = _axis_sequences(row_low, row_high, [-1, 1], row_outcome, max_overshoot=1, touch=row_touch)
        col_outcome = random.choice(["hit_low", "hit_high"])
        col_sequences = _axis_sequences(
            col_low,
            col_high,
            [-4, -3, -2, -1, 1, 2, 3, 4],
            col_outcome,
            max_overshoot=4,
        )
    else:
        raise ValueError(f"unknown family: {family}")

    options = []
    for row_seq in row_sequences:
        for col_seq in col_sequences:
            delta = (int(row_seq["delta"]), int(col_seq["delta"]))
            if delta == (0, 0):
                continue
            positions = [
                (int(row_seq["positions"][step]), int(col_seq["positions"][step]))
                for step in range(4)
            ]
            if len(set(positions)) != 4:
                continue
            candidate = (int(row_seq["positions"][4]), int(col_seq["positions"][4]))
            final = (
                _resolve_against_bounds(candidate[0], row_low, row_high, closed["top"], closed["bottom"]),
                _resolve_against_bounds(candidate[1], col_low, col_high, closed["left"], closed["right"]),
            )
            if final in positions:
                continue
            meta = {
                "family": family,
                "width": int(width),
                "gap_side": gap_side,
                "positions": tuple(positions),
                "candidate": candidate,
                "delta": delta,
                "final": (int(final[0]), int(final[1])),
            }
            closed_hits, gap_hits = _candidate_hits(meta)
            closed_count = sum(closed_hits.values())
            gap_count = sum(gap_hits.values())
            if family in {"horizontal_bounce", "vertical_bounce"} and not (closed_count == 1 and gap_count == 0):
                continue
            if family == "gap_and_bounce" and not (closed_count == 1 and gap_count == 1):
                continue
            if family == "vertical_bounce" and delta[1] == 0:
                continue
            if family == "horizontal_bounce" and final[1] in {pos[1] for pos in positions}:
                continue
            options.append(meta)
    return options


def _build_input(meta: dict[str, object]) -> np.ndarray:
    panels = [_render_panel(int(meta["width"]), str(meta["gap_side"]), pos) for pos in meta["positions"]]
    return _stack_panels(panels)


def _expected_output(meta: dict[str, object]) -> np.ndarray:
    output = _base_panel(int(meta["width"]), str(meta["gap_side"]))
    paste(output, SPRITE, int(meta["final"][0]), int(meta["final"][1]), transparent=Color.BLACK, clip=True)
    return output


def _validate_case(inp: np.ndarray, meta: dict[str, object]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    panels = _split_panels(inp)
    if len(panels) != 4 or panels[0].shape[0] != PANEL_H:
        return False
    closed = [_closed_sides(panel) for panel in panels]
    if len({_gap_side(side_flags) for side_flags in closed}) != 1:
        return False
    if _gap_side(closed[0]) != meta["gap_side"]:
        return False

    positions = []
    sprites = []
    for panel in panels:
        sprite, pos = _extract_sprite(panel)
        sprites.append(sprite)
        positions.append(pos)
        if np.any((panel == Color.RED) & (panel == Color.BLUE)):
            return False
    if any(not np.array_equal(sprite, SPRITE) for sprite in sprites):
        return False
    if tuple(positions) != tuple(meta["positions"]):
        return False

    deltas = [
        (positions[idx + 1][0] - positions[idx][0], positions[idx + 1][1] - positions[idx][1])
        for idx in range(3)
    ]
    if not (deltas[0] == deltas[1] == deltas[2] == tuple(meta["delta"])):
        return False

    candidate = (positions[-1][0] + deltas[0][0], positions[-1][1] + deltas[0][1])
    if candidate != tuple(meta["candidate"]):
        return False

    closed_hits, gap_hits = _candidate_hits(meta)
    closed_count = sum(closed_hits.values())
    gap_count = sum(gap_hits.values())
    if meta["family"] in {"horizontal_bounce", "vertical_bounce"} and not (closed_count == 1 and gap_count == 0):
        return False
    if meta["family"] == "gap_and_bounce" and not (closed_count == 1 and gap_count == 1):
        return False

    expected = _expected_output(meta)
    if np.count_nonzero(expected == Color.RED) == 0:
        return False
    if tuple(meta["final"]) in tuple(meta["positions"]):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the fifth framed panel, reflecting only the blocked motion components.

    Input structure:
    - The input is a vertical stack of four equal-width animation panels separated by single grey rows.
    - Each panel has a black background, a blue one-cell frame on exactly three sides, and one red 2x2 sprite inside.
    - The missing blue side is the frame gap and stays fixed across all four observed panels.

    Rule:
    - Split the strip into four panels and read the red sprite position in each one.
    - Infer the constant translation vector from the first four positions and apply it once more to get a fifth candidate position.
    - If a candidate coordinate crosses a closed frame side, reflect only that motion component across the last legal in-panel coordinate; if it crosses the gap side, do not reflect and simply clip the sprite to the panel bounds.
    - Return one panel: the shared blue frame together with the reflected or clipped fifth sprite.

    Key insight:
    - The sequence extrapolation is not enough by itself: the frame geometry changes only the velocity component that hits a closed side, while the gap side is the only direction that stays unguarded.
    """
    g = np.asarray(grid, dtype=int)
    panels = _split_panels(g)
    frame = panels[-1].copy()
    sprites_and_pos = [_extract_sprite(panel) for panel in panels]
    sprites = [item[0] for item in sprites_and_pos]
    positions = [item[1] for item in sprites_and_pos]
    assert all(np.array_equal(sprite, sprites[0]) for sprite in sprites[1:]), "sprite shape must stay fixed"

    deltas = [
        (positions[idx + 1][0] - positions[idx][0], positions[idx + 1][1] - positions[idx][1])
        for idx in range(3)
    ]
    assert deltas[0] == deltas[1] == deltas[2], "expected a constant translation vector"
    next_pos = _predict_next_position(frame, sprites[-1], positions[-1], deltas[0])

    output = frame.copy()
    output[output == Color.RED] = Color.BLACK
    paste(output, sprites[-1], next_pos[0], next_pos[1], transparent=Color.BLACK, clip=True)
    return output


def generate_input() -> np.ndarray:
    family_specs = [
        ("horizontal_bounce", ["left", "right"], [16, 18]),
        ("vertical_bounce", ["top", "bottom"], list(range(12, 25))),
        ("gap_and_bounce", ["top", "bottom"], list(range(12, 25))),
    ]

    for _ in range(GEN_ATTEMPTS):
        family, gap_choices, width_choices = random.choice(family_specs)
        gap_side = random.choice(gap_choices)
        width = random.choice(width_choices)
        options = _case_options(width, gap_side, family)
        if not options:
            continue
        meta = random.choice(options)
        inp = _build_input(meta)
        if _validate_case(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid rebound-panel input")
