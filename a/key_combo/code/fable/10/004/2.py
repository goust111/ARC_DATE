from __future__ import annotations

from collections import Counter
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _orbit_values(panel: np.ndarray, r: int, c: int) -> list[int]:
    h, w = panel.shape
    return [
        int(panel[r, c]),
        int(panel[r, w - 1 - c]),
        int(panel[h - 1 - r, c]),
        int(panel[h - 1 - r, w - 1 - c]),
    ]


def _quadrant_position(panel_shape: tuple[int, int], r: int, c: int, quadrant: int) -> tuple[int, int]:
    h, w = panel_shape
    if quadrant == 0:
        return r, c
    if quadrant == 1:
        return r, w - 1 - c
    if quadrant == 2:
        return h - 1 - r, c
    if quadrant == 3:
        return h - 1 - r, w - 1 - c
    raise ValueError("quadrant must be in 0..3")


def _canonical_panel(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    box = bbox(g != bg)
    assert box is not None, "expected one non-background panel"
    panel = crop_bbox(g, box)
    h, w = panel.shape
    assert h % 2 == 0 and w % 2 == 0, "panel must have even height and width"
    assert h >= 4 and w >= 4
    return bg, panel


def _ghost_output_from_panel(bg: int, panel: np.ndarray) -> np.ndarray:
    h, w = panel.shape
    qh, qw = h // 2, w // 2
    out = np.full((qh, qw), bg, dtype=int)

    for r in range(qh):
        for c in range(qw):
            values = _orbit_values(panel, r, c)
            counts = Counter(values).most_common()
            truth, freq = counts[0]
            assert freq >= 3, "each symmetry orbit must have at most one tampered cell"
            mismatches = sum(v != truth for v in values)
            assert mismatches in (0, 1)
            if mismatches == 1:
                out[r, c] = int(truth)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold tampered reflective quadrants into one coordinate system and assemble the hidden ghost fragment map.

    Input structure:
    - A background canvas contains one even-by-even foreground panel.
    - The panel is a 2x2 arrangement of reflected quadrants: upper-right is a left-right reflection,
      lower-left is an up-down reflection, and lower-right is a 180-degree reflection.
    - Some panel pixels were tampered using colors already present elsewhere in the panel, with at most
      one tampered cell inside any four-way symmetry orbit.

    Rule:
    - Crop the foreground panel.
    - For each upper-left quadrant coordinate, reflect-align the corresponding four quadrant cells and take
      their majority value; the three agreeing cells define the true color for that orbit.
    - Whenever one aligned cell disagrees with the majority, fold that mismatch back into the upper-left
      quadrant coordinate system and draw the majority color there on a background-filled output grid.

    Key insight:
    - No full "correct quadrant" exists to crop directly; the hidden image is reconstructed only by
      majority-voting each four-way reflective orbit and then canonically folding the scattered defects.
    """
    bg, panel = _canonical_panel(grid)
    return _ghost_output_from_panel(bg, panel)


GEN_ATTEMPTS = 400
GRID_MIN = 16
GRID_MAX = 30
QUADRANT_H_RANGE = (4, 8)
QUADRANT_W_RANGE = (4, 8)
PANEL_COLOR_COUNT_RANGE = (3, 5)
TAMPER_COUNT_CAP = 8


def _build_panel(source: np.ndarray) -> np.ndarray:
    qh, qw = source.shape
    panel = np.empty((2 * qh, 2 * qw), dtype=int)
    panel[:qh, :qw] = source
    panel[:qh, qw:] = source[:, ::-1]
    panel[qh:, :qw] = source[::-1, :]
    panel[qh:, qw:] = source[::-1, ::-1]
    return panel


def _random_source(qh: int, qw: int, palette: list[int]) -> np.ndarray | None:
    target_colors = random.randint(3, min(len(palette), 5))
    want = palette[:]
    random.shuffle(want)
    want = want[:target_colors]

    for _ in range(120):
        source = np.random.choice(want, size=(qh, qw))
        if len(set(map(int, source.ravel()))) < target_colors:
            continue
        if np.array_equal(source, source[:, ::-1]):
            continue
        if np.array_equal(source, source[::-1, :]):
            continue
        if np.array_equal(source, source[::-1, ::-1]):
            continue
        return source.astype(int)
    return None


def _sample_tamper_coords(source: np.ndarray, tamper_count: int) -> list[tuple[int, int]] | None:
    qh, qw = source.shape
    coords = [(r, c) for r in range(qh) for c in range(qw)]
    picked = random.sample(coords, tamper_count)
    truth_colors = {int(source[r, c]) for r, c in picked}
    if tamper_count >= 2 and len(truth_colors) < 2:
        return None
    rows = {r for r, _ in picked}
    cols = {c for _, c in picked}
    if tamper_count >= 3 and (len(rows) < 2 or len(cols) < 2):
        return None
    return picked


def _quadrant_assignments(tamper_count: int) -> tuple[str, list[int]]:
    if random.random() < 0.3:
        quad = random.randrange(4)
        return "concentrated", [quad] * tamper_count

    use_count = random.randint(2, min(4, tamper_count))
    selected = random.sample(range(4), use_count)
    assigned = selected + random.choices(selected, k=tamper_count - use_count)
    random.shuffle(assigned)
    return "distributed", assigned


def _tamper_panel(panel: np.ndarray, source: np.ndarray, palette: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    tampered = panel.copy()
    qh, qw = source.shape
    area = qh * qw
    min_tampers = 2
    max_tampers = min(TAMPER_COUNT_CAP, max(min_tampers, area // 2))
    tamper_count = random.randint(min_tampers, max_tampers)

    coords = _sample_tamper_coords(source, tamper_count)
    if coords is None:
        raise ValueError("degenerate tamper coordinate set")

    mode, quadrants = _quadrant_assignments(tamper_count)
    if len(quadrants) != tamper_count:
        raise ValueError("assignment length mismatch")

    out = np.full((qh, qw), -1, dtype=int)
    used_quadrants: list[int] = []

    for (r, c), quadrant in zip(coords, quadrants):
        truth = int(source[r, c])
        candidates = [color for color in palette if color != truth]
        replacement = random.choice(candidates)
        rr, cc = _quadrant_position(tampered.shape, r, c, quadrant)
        tampered[rr, cc] = replacement
        out[r, c] = truth
        used_quadrants.append(int(quadrant))

    meta = {
        "mode": mode,
        "tamper_coords": tuple(coords),
        "tamper_quadrants": tuple(used_quadrants),
    }
    return tampered, out, meta


def _expected_output(bg: int, truth_map: np.ndarray) -> np.ndarray:
    out = np.full(truth_map.shape, bg, dtype=int)
    mask = truth_map >= 0
    out[mask] = truth_map[mask]
    return out


def _validate_instance(
    grid: np.ndarray,
    bg: int,
    source: np.ndarray,
    expected: np.ndarray,
    meta: dict,
) -> bool:
    box = bbox(grid != bg)
    if box is None:
        return False
    panel = crop_bbox(grid, box)
    if panel.shape != (2 * source.shape[0], 2 * source.shape[1]):
        return False

    used_quadrants = set(meta["tamper_quadrants"])
    if meta["mode"] == "concentrated" and len(used_quadrants) != 1:
        return False
    if meta["mode"] == "distributed" and len(used_quadrants) < 2:
        return False

    for r in range(source.shape[0]):
        for c in range(source.shape[1]):
            values = _orbit_values(panel, r, c)
            counts = Counter(values).most_common()
            if counts[0][1] < 3:
                return False
            if sum(v != counts[0][0] for v in values) not in (0, 1):
                return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    painted = np.argwhere(expected != bg)
    if painted.shape[0] < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    qh = random.randint(*QUADRANT_H_RANGE)
    qw = random.randint(*QUADRANT_W_RANGE)

    panel_h = 2 * qh
    panel_w = 2 * qw
    if panel_h + 2 > GRID_MAX or panel_w + 2 > GRID_MAX:
        return None

    palette_size = random.randint(*PANEL_COLOR_COUNT_RANGE)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)
    source = _random_source(qh, qw, palette)
    if source is None:
        return None

    panel = _build_panel(source)
    try:
        tampered_panel, truth_map, meta = _tamper_panel(panel, source, palette)
    except ValueError:
        return None
    expected = _expected_output(bg, truth_map)

    min_rows = max(GRID_MIN, panel_h + 2)
    min_cols = max(GRID_MIN, panel_w + 2)
    rows = random.randint(min_rows, GRID_MAX)
    cols = random.randint(min_cols, GRID_MAX)

    grid = np.full((rows, cols), bg, dtype=int)
    r0 = random.randint(1, rows - panel_h - 1)
    c0 = random.randint(1, cols - panel_w - 1)
    grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = tampered_panel

    if not _validate_instance(grid, bg, source, expected, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ghost-fragment panel")
