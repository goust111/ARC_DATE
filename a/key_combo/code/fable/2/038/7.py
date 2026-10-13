from __future__ import annotations

from collections import Counter
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _line_axes(coords: np.ndarray) -> tuple[str, int, list[int]]:
    """Return (orientation, fixed_axis, sorted_positions) for one monochrome dot line."""
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(rows) == 1 and len(cols) == len(coords):
        return "h", rows[0], cols
    if len(cols) == 1 and len(rows) == len(coords):
        return "v", cols[0], rows
    raise AssertionError("each color must form one non-overlapping horizontal or vertical dot line")


def _gaps(positions: list[int]) -> list[int]:
    return [int(b - a) for a, b in zip(positions, positions[1:])]


def _lawful_direction(positions: list[int]) -> int:
    """
    Return +1 for left-to-right / top-to-bottom lawful lines, -1 for the reverse
    lawful direction, and 0 for unlawful lines.
    """
    gaps = _gaps(positions)
    forward = gaps == list(range(1, len(gaps) + 1))
    backward = gaps == list(range(len(gaps), 0, -1))
    if forward:
        return 1
    if backward:
        return -1
    return 0


def _extension_positions(positions: list[int], axis_len: int) -> list[int]:
    """Return the newly added axis positions for a lawful line."""
    direction = _lawful_direction(positions)
    if direction == 0:
        return []

    next_gap = len(positions)
    cursor = positions[-1] if direction > 0 else positions[0]
    extra: list[int] = []
    while True:
        nxt = cursor + direction * next_gap
        if not (0 <= nxt < axis_len):
            return extra
        extra.append(int(nxt))
        cursor = nxt
        next_gap += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only dot lines whose gaps are 1,2,3,... and continue them to the edge.

    Input structure:
    - A dominant background contains several single-color dot lines.
    - Each non-background color appears exactly once, all on one row or one column.
    - Every line has at least four dots, and different lines use different colors.

    Rule:
    - For each colored line, sort its dots along the row or column it occupies.
    - A line is lawful only when its adjacent gaps are exactly 1,2,3,... in one
      visible direction; the reversed direction is also allowed when the sorted
      gaps read ...,3,2,1.
    - Lawful lines keep their existing dots and place new dots by continuing the
      next gap as +1 each time until the next placement would leave the grid.
    - Unlawful lines are erased completely; all background cells stay unchanged.

    Key insight:
    - The deciding signal is the accelerating gap pattern itself. Even neat lines
      such as equal-spacing or shifted consecutive gaps must vanish if they do not
      start with 1 and then grow by exactly one each step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        assert len(coords) >= 4, "every line must contain at least four dots"
        orientation, fixed, positions = _line_axes(coords)
        direction = _lawful_direction(positions)

        if direction == 0:
            out[coords[:, 0], coords[:, 1]] = bg
            continue

        axis_len = g.shape[1] if orientation == "h" else g.shape[0]
        for pos in _extension_positions(positions, axis_len):
            if orientation == "h":
                out[fixed, pos] = color
            else:
                out[pos, fixed] = color

    return out


GRID_MIN = 20
GRID_MAX = 30
POINT_COUNT_CHOICES = (4, 5)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300
NEIGHBOR_RADIUS = 1


def _positions_from_gaps(base: int, gaps: list[int], direction: int) -> list[int]:
    positions = [int(base)]
    cursor = int(base)
    for gap in gaps:
        cursor += direction * int(gap)
        positions.append(cursor)
    return positions


def _legal_axis_positions(axis_len: int, count: int, direction: int, rng: random.Random) -> list[int] | None:
    required_span = sum(range(1, count + 1))
    if axis_len - 1 < required_span:
        return None
    if direction > 0:
        base = rng.randint(0, axis_len - 1 - required_span)
    else:
        base = rng.randint(required_span, axis_len - 1)
    positions = _positions_from_gaps(base, list(range(1, count)), direction)
    if len(_extension_positions(sorted(positions), axis_len)) < 1:
        return None
    return positions


def _equal_gap_list(axis_len: int, count: int, rng: random.Random) -> list[int] | None:
    max_gap = (axis_len - 1) // (count - 1)
    candidates = [gap for gap in range(2, min(5, max_gap + 1))]
    if not candidates:
        return None
    gap = rng.choice(candidates)
    return [gap] * (count - 1)


def _shifted_gap_list(axis_len: int, count: int, rng: random.Random) -> list[int] | None:
    candidates = []
    for start in range(2, 6):
        gaps = [start + i for i in range(count - 1)]
        if sum(gaps) <= axis_len - 1:
            candidates.append(gaps)
    if not candidates:
        return None
    return list(rng.choice(candidates))


def _random_invalid_gap_list(axis_len: int, count: int, rng: random.Random) -> list[int] | None:
    for _ in range(80):
        gaps = [rng.randint(1, 5) for _ in range(count - 1)]
        if sum(gaps) > axis_len - 1:
            continue
        if len(set(gaps)) == 1:
            continue
        if all(gaps[i + 1] - gaps[i] == 1 for i in range(len(gaps) - 1)):
            continue
        if all(gaps[i] - gaps[i + 1] == 1 for i in range(len(gaps) - 1)):
            continue
        return gaps
    return None


def _coords_for_positions(orientation: str, fixed: int, positions: list[int]) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(fixed, int(pos)) for pos in positions]
    return [(int(pos), fixed) for pos in positions]


def _touches_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, r - NEIGHBOR_RADIUS)
        r1 = min(h, r + NEIGHBOR_RADIUS + 1)
        c0 = max(0, c - NEIGHBOR_RADIUS)
        c1 = min(w, c + NEIGHBOR_RADIUS + 1)
        if blocked[r0:r1, c0:c1].any():
            return True
    return False


def _mark(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        blocked[r, c] = True


def _place_sequence(
    shape: tuple[int, int],
    blocked: np.ndarray,
    orientation: str,
    lawful: bool,
    kind: str,
    rng: random.Random,
) -> dict | None:
    axis_len = shape[1] if orientation == "h" else shape[0]
    fixed_len = shape[0] if orientation == "h" else shape[1]

    for _ in range(PLACEMENT_ATTEMPTS):
        count = rng.choice(POINT_COUNT_CHOICES)
        direction = rng.choice([-1, 1])

        if lawful:
            axis_positions = _legal_axis_positions(axis_len, count, direction, rng)
        else:
            if kind == "equal":
                gaps = _equal_gap_list(axis_len, count, rng)
            elif kind == "shifted":
                gaps = _shifted_gap_list(axis_len, count, rng)
            else:
                gaps = _random_invalid_gap_list(axis_len, count, rng)
            if gaps is None:
                continue
            total = sum(gaps)
            if direction > 0:
                if total > axis_len - 1:
                    continue
                base = rng.randint(0, axis_len - 1 - total)
            else:
                if total > axis_len - 1:
                    continue
                base = rng.randint(total, axis_len - 1)
            axis_positions = _positions_from_gaps(base, gaps, direction)
            if _lawful_direction(sorted(axis_positions)) != 0:
                continue

        assert axis_positions is not None
        protect_positions = list(axis_positions)
        if lawful:
            protect_positions += _extension_positions(sorted(axis_positions), axis_len)

        fixed_values = list(range(fixed_len))
        rng.shuffle(fixed_values)
        for fixed in fixed_values:
            input_coords = _coords_for_positions(orientation, fixed, axis_positions)
            protect_coords = _coords_for_positions(orientation, fixed, protect_positions)
            if _touches_blocked(blocked, protect_coords):
                continue
            if len(set(input_coords)) != len(input_coords):
                continue
            _mark(blocked, protect_coords)
            return {
                "orientation": orientation,
                "fixed": fixed,
                "input_positions": list(axis_positions),
                "protect_positions": list(protect_positions),
                "lawful": lawful,
                "kind": kind,
            }
    return None


def _extract_line_infos(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    infos: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        coords = np.argwhere(grid == color)
        orientation, fixed, positions = _line_axes(coords)
        axis_len = grid.shape[1] if orientation == "h" else grid.shape[0]
        gaps = _gaps(positions)
        infos.append(
            {
                "color": color,
                "orientation": orientation,
                "fixed": fixed,
                "positions": positions,
                "gaps": gaps,
                "axis_len": axis_len,
                "lawful": _lawful_direction(positions) != 0,
            }
        )
    return infos


def _is_equal_gap(gaps: list[int]) -> bool:
    return len(gaps) >= 1 and len(set(gaps)) == 1 and gaps[0] > 1


def _is_shifted_consecutive(gaps: list[int]) -> bool:
    if not gaps:
        return False
    inc = all(gaps[i + 1] - gaps[i] == 1 for i in range(len(gaps) - 1)) and gaps[0] > 1
    dec = all(gaps[i] - gaps[i + 1] == 1 for i in range(len(gaps) - 1)) and gaps[-1] > 1
    return inc or dec


def _expected_output(shape: tuple[int, int], bg: int, sequences: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for seq in sequences:
        if not seq["lawful"]:
            continue
        for r, c in seq["protect_coords"]:
            out[r, c] = int(seq["color"])
    return out


def _validate_pair(inp: np.ndarray, expected: np.ndarray) -> bool:
    infos = _extract_line_infos(inp)
    lawful_infos = [info for info in infos if info["lawful"]]
    unlawful_infos = [info for info in infos if not info["lawful"]]

    if len(lawful_infos) < 2 or len(unlawful_infos) < 2:
        return False
    if {info["orientation"] for info in infos} != {"h", "v"}:
        return False
    if not any(_is_equal_gap(info["gaps"]) for info in unlawful_infos):
        return False
    if not any(_is_shifted_consecutive(info["gaps"]) for info in unlawful_infos):
        return False
    if not any(len(_extension_positions(info["positions"], info["axis_len"])) >= 1 for info in lawful_infos):
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = get_bg(inp)
    for info in unlawful_infos:
        if np.any(actual == info["color"]):
            return False
    for info in lawful_infos:
        added = _extension_positions(info["positions"], info["axis_len"])
        if not added:
            continue
        for pos in added:
            r, c = (info["fixed"], pos) if info["orientation"] == "h" else (pos, info["fixed"])
            if int(inp[r, c]) != bg or int(actual[r, c]) != info["color"]:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))
    specs = [
        {"orientation": "h", "lawful": True, "kind": "legal"},
        {"orientation": "v", "lawful": True, "kind": "legal"},
        {"orientation": rng.choice(["h", "v"]), "lawful": False, "kind": "equal"},
        {"orientation": rng.choice(["h", "v"]), "lawful": False, "kind": "shifted"},
    ]
    if rng.random() < 0.7:
        extra_lawful = rng.random() < 0.5
        specs.append(
            {
                "orientation": rng.choice(["h", "v"]),
                "lawful": extra_lawful,
                "kind": "legal" if extra_lawful else rng.choice(["equal", "shifted", "random"]),
            }
        )

    blocked = np.zeros(shape, dtype=bool)
    sequences: list[dict] = []
    for spec in specs:
        seq = _place_sequence(
            shape,
            blocked,
            spec["orientation"],
            bool(spec["lawful"]),
            str(spec["kind"]),
            rng,
        )
        if seq is None:
            return None
        sequences.append(seq)

    bg = rng.randint(0, 9)
    colors = rng.sample([c for c in range(10) if c != bg], len(sequences))
    for seq, color in zip(sequences, colors):
        seq["color"] = int(color)
        seq["input_coords"] = _coords_for_positions(seq["orientation"], seq["fixed"], seq["input_positions"])
        seq["protect_coords"] = _coords_for_positions(seq["orientation"], seq["fixed"], seq["protect_positions"])

    inp = np.full(shape, bg, dtype=int)
    for seq in sequences:
        for r, c in seq["input_coords"]:
            inp[r, c] = int(seq["color"])

    expected = _expected_output(shape, bg, sequences)
    if not _validate_pair(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lawful-extend input")
