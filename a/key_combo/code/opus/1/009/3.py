from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
GREY = 5
CONNECTOR_COLORS = (2, 3, 4, 6, 7, 8, 9)
CARDINAL_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ROTATION_BY_OFFSET = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
BASE_TEMPLATE = np.array(
    [
        [GREY, BLUE, GREY],
        [BLUE, BACKGROUND, BLUE],
        [GREY, BLUE, GREY],
    ],
    dtype=int,
)
GENERATION_RETRIES = 300


def _parse_instances(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    height, width = map(int, g.shape)
    grey_positions = np.argwhere(g == GREY)
    if len(grey_positions) == 0:
        raise AssertionError("expected at least one grey seed pixel")

    colored_positions = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BACKGROUND) & (g != GREY))
    }
    claimed_connectors: set[tuple[int, int]] = set()
    records: list[dict[str, int]] = []

    for raw_r, raw_c in grey_positions:
        r, c = int(raw_r), int(raw_c)
        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            raise AssertionError("grey seed too close to the border")

        neighborhood = g[r - 1 : r + 2, c - 1 : c + 2]
        if int(np.count_nonzero(neighborhood == GREY)) != 1:
            raise AssertionError("grey seeds must stay isolated inside their 3x3 neighborhoods")
        if int(np.count_nonzero((neighborhood != BACKGROUND) & (neighborhood != GREY))) != 1:
            raise AssertionError("each 3x3 neighborhood must contain exactly one colored connector")

        connectors = []
        for dr, dc in CARDINAL_DIRS:
            rr, cc = int(r + dr), int(c + dc)
            color = int(g[rr, cc])
            if color not in (BACKGROUND, GREY):
                connectors.append((dr, dc, rr, cc, color))
        if len(connectors) != 1:
            raise AssertionError("each grey seed must have exactly one 4-neighbor connector")

        dr, dc, rr, cc, color = connectors[0]
        if (rr, cc) in claimed_connectors:
            raise AssertionError("a connector cannot serve multiple grey seeds")
        claimed_connectors.add((rr, cc))
        records.append(
            {
                "r": r,
                "c": c,
                "dr": int(dr),
                "dc": int(dc),
                "connector_r": rr,
                "connector_c": cc,
                "color": color,
            }
        )

    if claimed_connectors != colored_positions:
        raise AssertionError("every colored pixel must be the unique connector of one grey seed")

    for i, first in enumerate(records):
        for second in records[i + 1 :]:
            if abs(int(first["r"]) - int(second["r"])) <= 2 and abs(int(first["c"]) - int(second["c"])) <= 2:
                raise AssertionError("3x3 stamp footprints must not overlap")

    records.sort(key=lambda rec: (int(rec["r"]), int(rec["c"])))
    return records


def _solve_with_rotations(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for rec in _parse_instances(g):
        turns_clockwise = ROTATION_BY_OFFSET[(int(rec["dr"]), int(rec["dc"]))]
        stamp = np.rot90(BASE_TEMPLATE, k=-turns_clockwise)
        r0 = int(rec["r"]) - 1
        c0 = int(rec["c"]) - 1

        out[r0 : r0 + 3, c0 : c0 + 3] = stamp
        out[int(rec["r"]), int(rec["c"])] = int(rec["color"])
        # The connector is inside the stamp footprint; erasing it makes the direction visible.
        out[int(rec["connector_r"]), int(rec["connector_c"])] = BACKGROUND

    return out


def _solve_coordinatewise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for rec in _parse_instances(g):
        r, c = int(rec["r"]), int(rec["c"])
        hidden_arm = (int(rec["dr"]), int(rec["dc"]))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r + dr), int(c + dc)
                if dr == 0 and dc == 0:
                    out[rr, cc] = int(rec["color"])
                elif abs(dr) == 1 and abs(dc) == 1:
                    out[rr, cc] = GREY
                elif (dr, dc) == hidden_arm:
                    out[rr, cc] = BACKGROUND
                else:
                    out[rr, cc] = BLUE

    return out


def _choose_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []

    for r, c in candidates:
        if all(abs(r - rr) > 2 or abs(c - cc) > 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _validate_generated_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        raise AssertionError("generated grid must stay within ARC size bounds")

    records = _parse_instances(g)
    if len(records) not in (3, 4):
        raise AssertionError("generator must place 3 or 4 grey seeds")

    colors = [int(rec["color"]) for rec in records]
    if len(set(colors)) != len(colors):
        raise AssertionError("generated connectors must use distinct colors")

    directions = {(int(rec["dr"]), int(rec["dc"])) for rec in records}
    if len(directions) != len(records):
        raise AssertionError("generated examples should cover distinct direction branches")

    via_rotation = _solve_with_rotations(g)
    via_coordinates = _solve_coordinatewise(g)
    if not np.array_equal(via_rotation, via_coordinates):
        raise AssertionError("independent semantic checks must agree on the output")

    for rec in records:
        r, c = int(rec["r"]), int(rec["c"])
        if int(via_rotation[r, c]) != int(rec["color"]):
            raise AssertionError("stamp center must inherit the connector color")
        if int(via_rotation[int(rec["connector_r"]), int(rec["connector_c"])]) != BACKGROUND:
            raise AssertionError("connector must be erased in the output")


def _gen_attempt() -> np.ndarray | None:
    count = 4 if random.random() < 0.65 else 3
    height = random.randint(11, 16)
    width = random.randint(11, 16)
    centers = _choose_centers(height, width, count)
    if centers is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    directions = random.sample(CARDINAL_DIRS, count)
    colors = random.sample(CONNECTOR_COLORS, count)
    random.shuffle(centers)

    for (r, c), (dr, dc), color in zip(centers, directions, colors):
        grid[r, c] = GREY
        grid[r + dr, c + dc] = int(color)

    if random.random() < 0.5:
        grid = np.fliplr(grid).copy()
    grid = np.rot90(grid, k=random.randrange(4)).copy()
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each isolated grey seed into a directional 3x3 stamp whose missing arm is encoded by its connector.

    Input structure:
    - The grid is a black background with several isolated grey single pixels.
    - Each grey pixel has exactly one colored 4-neighbor connector, and no other colored pixel appears in its 3x3 neighborhood.
    - Different grey seeds are far enough apart that their 3x3 stamp footprints do not overlap.

    Rule:
    - Use the connector position relative to the grey seed to choose a rotation: up -> 0 degrees, right -> 90 degrees clockwise, down -> 180 degrees, left -> 270 degrees clockwise.
    - Rotate the fixed 3x3 stamp whose diagonals are grey, orthogonal arms are blue, and center is black.
    - Cover the rotated stamp onto the grid centered on the grey seed, replace the stamp center with the connector color, and then erase the connector pixel to black.

    Key insight:
    - The connector carries two independent channels at once: its direction chooses which arm disappears after the erase step, and its color becomes the center of the stamp.
    """
    return _solve_with_rotations(np.asarray(grid, dtype=int))


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        _validate_generated_case(grid)
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
