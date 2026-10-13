from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
CONTROL_COLOR = 4
BODY_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_RETRIES = 500
COMPONENT_RETRIES = 60


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    assert h == w + 2, "input must be an n x n body plus one blank row and one control row"
    n = w

    assert np.all(g[n, :] == BG), "expected a blank separator row under the square body"
    control = g[n + 1, :]
    assert set(map(int, np.unique(control))).issubset({BG, CONTROL_COLOR}), "control row must use only black and yellow"
    assert 1 <= int(np.sum(control == CONTROL_COLOR)) <= n, "expected at least one controlled column"

    body = g[:n, :].copy()
    colors = sorted(int(v) for v in np.unique(body) if int(v) != BG)
    assert 2 <= len(colors) <= 3, "body must contain 2-3 non-background colors"
    return body, control.copy()


def _stitched_quadrants(body: np.ndarray) -> np.ndarray:
    q0 = body
    q1 = np.rot90(body, 1)
    q2 = np.rot90(body, 2)
    q3 = np.rot90(body, 3)
    return np.block([[q0, q1], [q2, q3]])


def _apply_control_columns(body: np.ndarray, control: np.ndarray) -> np.ndarray:
    n = body.shape[0]
    out = _stitched_quadrants(body)
    for c in np.flatnonzero(control == CONTROL_COLOR):
        col = body[:, int(c)].copy()
        out[:n, int(c)] = col
        out[:n, n + int(c)] = col
        out[n:, int(c)] = col
        out[n:, n + int(c)] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a square motif into four quadrants, then force selected local columns to agree with the original quadrant.

    Input structure:
    - The input has an n x n square motif on a black background.
    - One fully black separator row sits directly below that square.
    - The last row is a width-n control row containing only black pixels and
      yellow control pixels; each yellow pixel marks one local column.

    Rule:
    - Build a 2n x 2n grid with four quadrant copies of the motif:
      top-left = original, top-right = 90 degree counterclockwise rotation,
      bottom-left = 180 degree rotation, bottom-right = 270 degree rotation.
    - For every local coordinate (r, c), compare the four quadrant colors at
      that same local position.
    - If the control row has black at column c, keep the ordinary rotated
      quadrant colors.
    - If the control row has yellow at column c, overwrite the other three
      quadrants at every row r in that local column so they all equal the
      top-left quadrant's value body[r, c].

    Key insight:
    - The output is mostly a standard four-rotation tiling, but only the
      columns named by the detached control row collapse across quadrants.
    """
    body, control = _parse_input(grid)
    return _apply_control_columns(body, control)


def _grow_component(n: int, occupied: set[tuple[int, int]], size: int, rng: random.Random) -> list[tuple[int, int]] | None:
    free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied]
    if not free:
        return None

    for _ in range(COMPONENT_RETRIES):
        start = rng.choice(free)
        cells = {start}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in occupied and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) == size:
            return list(cells)
    return None


def _build_body(n: int, rng: random.Random) -> np.ndarray | None:
    body = np.full((n, n), BG, dtype=int)
    num_colors = rng.randint(2, 3)
    palette = rng.sample(BODY_COLORS, num_colors)
    occupied: set[tuple[int, int]] = set()

    for color in palette:
        size = rng.randint(2, min(5, n))
        cells = _grow_component(n, occupied, size, rng)
        if cells is None:
            return None
        for r, c in cells:
            body[r, c] = color
            occupied.add((r, c))

    extra_pixels = rng.randint(1, max(2, n - 2))
    free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied]
    rng.shuffle(free)
    for r, c in free[:extra_pixels]:
        body[r, c] = rng.choice(palette)
        occupied.add((r, c))

    return body


def _rotation_stack(body: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(body, k) for k in range(4)]


def _eligible_columns(body: np.ndarray) -> list[int]:
    rots = _rotation_stack(body)
    n = body.shape[0]
    eligible = []
    for c in range(n):
        changed_rows = 0
        for r in range(n):
            vals = {int(rots[k][r, c]) for k in range(4)}
            if len(vals) >= 2:
                changed_rows += 1
        if changed_rows >= 2:
            eligible.append(c)
    return eligible


def _build_input(body: np.ndarray, control_cols: list[int]) -> np.ndarray:
    n = body.shape[0]
    control = np.full((1, n), BG, dtype=int)
    control[0, control_cols] = CONTROL_COLOR
    return np.vstack([body, np.zeros((1, n), dtype=int), control])


def _toggle_column(control: np.ndarray, c: int, on: bool) -> np.ndarray:
    updated = control.copy()
    updated[int(c)] = CONTROL_COLOR if on else BG
    return updated


def _is_valid_body(body: np.ndarray) -> bool:
    colors = sorted(int(v) for v in np.unique(body) if int(v) != BG)
    if len(colors) not in (2, 3):
        return False
    if len({np.rot90(body, k).tobytes() for k in range(4)}) != 4:
        return False
    if np.count_nonzero(body) < body.shape[0] + 1:
        return False
    return True


def _candidate_input(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(5, 7)
    body = _build_body(n, rng)
    if body is None or not _is_valid_body(body):
        return None

    eligible = _eligible_columns(body)
    if len(eligible) < 3:
        return None

    max_controls = min(4, len(eligible) - 1)
    if max_controls < 2:
        return None
    control_count = rng.randint(2, max_controls)
    control_cols = sorted(rng.sample(eligible, control_count))
    inp = _build_input(body, control_cols)

    control = inp[-1].copy()
    plain = _stitched_quadrants(body)
    expected = _apply_control_columns(body, control)

    if np.array_equal(expected, plain):
        return None

    uncontrolled_eligible = [c for c in eligible if c not in control_cols]
    if not uncontrolled_eligible:
        return None

    for c in control_cols:
        muted = _apply_control_columns(body, _toggle_column(control, c, False))
        if np.array_equal(muted, expected):
            return None

    for c in uncontrolled_eligible[:2]:
        forced = _apply_control_columns(body, _toggle_column(control, c, True))
        if np.array_equal(forced, expected):
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        candidate = _candidate_input(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 8")
