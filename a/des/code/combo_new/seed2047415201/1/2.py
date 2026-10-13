import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
FOREGROUND_COLORS = (1, 2, 3, 4, 6, 8, 9)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    has_horizontal = h % 2 == 1 and bool(np.all(grid[h // 2, :] == GREY))
    has_vertical = w % 2 == 1 and bool(np.all(grid[:, w // 2] == GREY))
    assert has_horizontal != has_vertical, "expected exactly one centered grey separator line"

    if has_horizontal:
        sep = h // 2
        a = grid[:sep, :]
        b = grid[sep + 1 :, :]
    else:
        sep = w // 2
        a = grid[:, :sep]
        b = grid[:, sep + 1 :]

    assert a.shape == b.shape, "separator must split the grid into equal panels"
    assert not bool(np.any(a == GREY)) and not bool(np.any(b == GREY)), "grey appears only in the separator"
    return a, b


def _merge_panels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    a_fg = a != BLACK
    b_fg = b != BLACK

    out = np.full(a.shape, BLACK, dtype=int)
    only_a = a_fg & ~b_fg
    only_b = ~a_fg & b_fg
    both = a_fg & b_fg
    same = both & (a == b)
    different = both & (a != b)

    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    out[same] = a[same]
    out[different] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two incomplete patterns on either side of the separator and mark color conflicts.

    Input structure:
    - The grid has black background and one complete grey row or column separator.
    - The separator divides the grid into two equal-size panels.
    - Each panel contains a sparse multicolor foreground pattern in the same local coordinate range.

    Rule:
    - Remove the grey separator and align the two panels by their local coordinates.
    - A coordinate that is black in both panels stays black.
    - A coordinate with foreground in only one panel keeps that foreground color.
    - A coordinate with two equal foreground colors keeps that color, while two different foreground colors become orange.

    Key insight:
    - The separator is a fold cue: the answer is a coordinate-wise overlay, with orange reserved for visible disagreement at overlapping colored pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    a, b = _split_panels(g)
    return _merge_panels(a, b)


# --------------- generator ---------------

ORIENTATIONS = ("horizontal", "vertical")
MIN_PANEL_SIZE = 8
MAX_SPLIT_PANEL = 14
MAX_UNSPLIT_PANEL = 22
MIN_BBOX_SIZE = 5
MAX_BBOX_SIZE = 11
GENERATION_ATTEMPTS = 300


def _panel_shape(orientation: str) -> tuple[int, int]:
    if orientation == "horizontal":
        return (
            random.randint(MIN_PANEL_SIZE, MAX_SPLIT_PANEL),
            random.randint(MIN_PANEL_SIZE, MAX_UNSPLIT_PANEL),
        )
    return (
        random.randint(MIN_PANEL_SIZE, MAX_UNSPLIT_PANEL),
        random.randint(MIN_PANEL_SIZE, MAX_SPLIT_PANEL),
    )


def _choose_bbox(panel_h: int, panel_w: int) -> tuple[int, int, int, int]:
    bbox_h = random.randint(MIN_BBOX_SIZE, min(MAX_BBOX_SIZE, panel_h - 2))
    bbox_w = random.randint(MIN_BBOX_SIZE, min(MAX_BBOX_SIZE, panel_w - 2))
    r0 = random.randint(1, panel_h - bbox_h - 1)
    c0 = random.randint(1, panel_w - bbox_w - 1)
    return r0, c0, r0 + bbox_h - 1, c0 + bbox_w - 1


def _same_pair() -> tuple[int, int]:
    color = random.choice(FOREGROUND_COLORS)
    return color, color


def _different_pair() -> tuple[int, int]:
    a, b = random.sample(FOREGROUND_COLORS, 2)
    return a, b


def _assign_cell(specs: dict[tuple[int, int], tuple[int, int]], coord: tuple[int, int], role: str) -> None:
    if role == "same":
        specs[coord] = _same_pair()
    elif role == "different":
        specs[coord] = _different_pair()
    elif role == "only_a":
        specs[coord] = (random.choice(FOREGROUND_COLORS), BLACK)
    elif role == "only_b":
        specs[coord] = (BLACK, random.choice(FOREGROUND_COLORS))
    else:
        raise ValueError(role)


def _take_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    count = min(count, len(cells))
    chosen = random.sample(cells, count)
    chosen_set = set(chosen)
    cells[:] = [cell for cell in cells if cell not in chosen_set]
    return chosen


def _make_panels(panel_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    r0, c0, r1, c1 = _choose_bbox(panel_h, panel_w)
    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    remaining = [cell for cell in all_cells if cell not in set(corners)]
    random.shuffle(remaining)

    specs: dict[tuple[int, int], tuple[int, int]] = {}
    shuffled_corners = corners[:]
    random.shuffle(shuffled_corners)
    for coord in shuffled_corners[:2]:
        _assign_cell(specs, coord, "same")
    for coord in shuffled_corners[2:]:
        _assign_cell(specs, coord, "different")

    required_counts = {
        "only_a": random.randint(3, 5),
        "only_b": random.randint(3, 5),
        "same": random.randint(1, 3),
        "different": random.randint(1, 3),
    }
    for role, count in required_counts.items():
        for coord in _take_cells(remaining, count):
            _assign_cell(specs, coord, role)

    for coord in _take_cells(remaining, random.randint(0, min(8, len(remaining)))):
        _assign_cell(specs, coord, random.choice(("only_a", "only_b", "same", "different")))

    a = np.full((panel_h, panel_w), BLACK, dtype=int)
    b = np.full((panel_h, panel_w), BLACK, dtype=int)
    for (r, c), (ca, cb) in specs.items():
        a[r, c] = ca
        b[r, c] = cb

    if len(set(map(int, a[a != BLACK]))) < 2 or len(set(map(int, b[b != BLACK]))) < 2:
        return None
    if not bool(np.any((a != BLACK) & (b == BLACK))):
        return None
    if not bool(np.any((a == BLACK) & (b != BLACK))):
        return None
    if not bool(np.any((a != BLACK) & (b != BLACK) & (a == b))):
        return None
    if not bool(np.any((a != BLACK) & (b != BLACK) & (a != b))):
        return None
    return a, b


def _assemble_input(a: np.ndarray, b: np.ndarray, orientation: str) -> np.ndarray:
    if random.choice([True, False]):
        a, b = b, a

    panel_h, panel_w = a.shape
    if orientation == "horizontal":
        sep = np.full((1, panel_w), GREY, dtype=int)
        return np.vstack([a, sep, b])

    sep = np.full((panel_h, 1), GREY, dtype=int)
    return np.hstack([a, sep, b])


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(ORIENTATIONS)
    panel_h, panel_w = _panel_shape(orientation)
    panels = _make_panels(panel_h, panel_w)
    if panels is None:
        return None
    a, b = panels
    expected = _merge_panels(a, b)
    grid = _assemble_input(a, b, orientation)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
