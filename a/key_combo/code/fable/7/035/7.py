from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _separator(grid: np.ndarray) -> tuple[int, int, int]:
    """Return the centered separator row, column, and color."""
    h, w = grid.shape
    assert h % 2 == 1 and w % 2 == 1, "the cross must split four equal panels"
    sr, sc = h // 2, w // 2
    color = int(grid[sr, sc])
    assert np.all(grid[sr, :] == color) and np.all(grid[:, sc] == color)
    assert not np.any(np.delete(grid == color, sr, axis=0)[:, :sc])
    assert not np.any(np.delete(grid == color, sr, axis=0)[:, sc + 1 :])
    return sr, sc, color


def _panels(grid: np.ndarray, sr: int, sc: int) -> list[np.ndarray]:
    return [
        grid[:sr, :sc].copy(),
        grid[:sr, sc + 1 :].copy(),
        grid[sr + 1 :, :sc].copy(),
        grid[sr + 1 :, sc + 1 :].copy(),
    ]


def _unique_least_color(panel: np.ndarray) -> int:
    counts = Counter(map(int, panel.ravel()))
    least = min(counts.values())
    colors = [color for color, count in counts.items() if count == least]
    assert len(colors) == 1, "each pattern panel needs one least-frequent color"
    return int(colors[0])


def _is_isolated(points: np.ndarray) -> bool:
    cells = [tuple(map(int, point)) for point in points]
    return all(
        max(abs(r1 - r2), abs(c1 - c2)) > 1
        for i, (r1, c1) in enumerate(cells)
        for r2, c2 in cells[i + 1 :]
    )


def _decode(panels: list[np.ndarray]) -> tuple[int, int, dict[int, np.ndarray]]:
    """Identify the sparse control panel and its signature-to-panel mapping."""
    candidates: list[tuple[int, int, dict[int, np.ndarray]]] = []
    for control_index, control in enumerate(panels):
        counts = Counter(map(int, control.ravel()))
        largest = max(counts.values())
        backgrounds = [color for color, count in counts.items() if count == largest]
        if len(backgrounds) != 1:
            continue
        bg = int(backgrounds[0])
        points = np.argwhere(control != bg)
        if len(points) == 0 or not _is_isolated(points):
            continue

        sources = [panel for i, panel in enumerate(panels) if i != control_index]
        try:
            signatures = [_unique_least_color(panel) for panel in sources]
        except AssertionError:
            continue
        if len(set(signatures)) != len(signatures):
            continue
        mapping = {signature: panel for signature, panel in zip(signatures, sources)}
        marker_colors = set(map(int, control[control != bg]))
        if marker_colors and marker_colors <= set(mapping):
            candidates.append((control_index, bg, mapping))

    assert len(candidates) == 1, "the sparse control panel must be uniquely identifiable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a whole pattern panel into each output block selected by a rare-color control marker.

    Input structure:
    - A centered one-color separator cross divides the grid into four equal panels.
    - One panel is a background field with isolated one-pixel control markers; the other
      three are dense patterns, each with a distinct unique least-frequent signature color.

    Rule:
    - Match each non-background control cell to the pattern whose unique least-frequent
      color equals that cell's color, and copy that entire pattern into the corresponding
      block of a control-height by control-width block array. Leave unmarked blocks as
      background and omit the separator cross.

    Key insight:
    - A marker names a whole source panel by its rare-color signature, not by the panel's
      dominant color and not as a color to enlarge into a solid block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sr, sc, _separator_color = _separator(g)
    panels = _panels(g, sr, sc)
    control_index, bg, mapping = _decode(panels)
    control = panels[control_index]
    ph, pw = control.shape
    out = np.full((ph * ph, pw * pw), bg, dtype=int)

    for r, c in np.argwhere(control != bg):
        panel = mapping[int(control[r, c])]
        r0, c0 = int(r) * ph, int(c) * pw
        out[r0 : r0 + ph, c0 : c0 + pw] = panel
    return out


_PALETTE = tuple(range(10))
_QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
_ATTEMPTS = 300


def _make_pattern(
    h: int,
    w: int,
    bg: int,
    main_color: int,
    signatures: list[int],
    own_index: int,
) -> np.ndarray:
    """Make a dense panel where all keys occur but only its own key is rarest."""
    area = h * w
    plans: list[tuple[int, list[int], int, int]] = []
    for own_count in (1, 2):
        other_counts = [own_count + 1, own_count + 2]
        for bg_count in range(max(other_counts) + 1, max(other_counts) + 4):
            main_count = area - own_count - sum(other_counts) - bg_count
            if main_count > bg_count:
                plans.append((own_count, other_counts, bg_count, main_count))
    if not plans:
        raise ValueError("panel is too small for a shared dominant main color")
    own_count, other_counts, bg_count, main_count = random.choice(plans)
    other_counts = other_counts.copy()
    random.shuffle(other_counts)

    counts = [0, 0, 0]
    counts[own_index] = own_count
    others = [i for i in range(3) if i != own_index]
    for i, count in zip(others, other_counts):
        counts[i] = count

    cells = [bg] * bg_count + [main_color] * main_count
    for signature, count in zip(signatures, counts):
        cells.extend([signature] * count)
    random.shuffle(cells)
    return np.asarray(cells, dtype=int).reshape(h, w)


def _control_positions(h: int, w: int, count: int) -> list[tuple[int, int]]:
    # One parity lattice leaves at least one row/column between every pair of markers.
    choices: list[list[tuple[int, int]]] = []
    for ro in (0, 1):
        for co in (0, 1):
            lattice = [(r, c) for r in range(ro, h, 2) for c in range(co, w, 2)]
            if len(lattice) >= count:
                choices.append(lattice)
    lattice = random.choice(choices)
    return random.sample(lattice, count)


def _place_panel(grid: np.ndarray, panel: np.ndarray, quadrant: tuple[int, int]) -> None:
    h, w = panel.shape
    qr, qc = quadrant
    r0 = 0 if qr == 0 else h + 1
    c0 = 0 if qc == 0 else w + 1
    grid[r0 : r0 + h, c0 : c0 + w] = panel


def _semantic_validation(
    grid: np.ndarray,
    control_index: int,
    bg: int,
    signatures: list[int],
    source_by_signature: dict[int, np.ndarray],
) -> bool:
    """Check the stated semantics and deliberate anti-shortcut counterexamples."""
    try:
        sr, sc, _ = _separator(grid)
        panels = _panels(grid, sr, sc)
        decoded_index, decoded_bg, decoded = _decode(panels)
        if decoded_index != control_index or decoded_bg != bg:
            return False
        control = panels[control_index]
        if set(map(int, control[control != bg])) != set(signatures):
            return False  # every source mapping is demonstrated

        sources = list(source_by_signature.values())
        if len({panel.tobytes() for panel in sources}) != 3:
            return False
        dominant_non_bg = []
        for signature, panel in source_by_signature.items():
            counts = Counter(map(int, panel.ravel()))
            if _unique_least_color(panel) != signature:
                return False
            if not set(signatures) <= set(counts):
                return False  # mere color presence cannot select a source
            non_bg_counts = {c: n for c, n in counts.items() if c != bg}
            dominant_non_bg.append(max(non_bg_counts, key=non_bg_counts.get))
        if len(set(dominant_non_bg)) != 1:
            return False  # dominant color cannot select a source

        ph, pw = control.shape
        expected = np.full((ph * ph, pw * pw), bg, dtype=int)
        for r, c in np.argwhere(control != bg):
            patch = source_by_signature[int(control[r, c])]
            expected[int(r) * ph : (int(r) + 1) * ph,
                     int(c) * pw : (int(c) + 1) * pw] = patch
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        active = sum(
            not np.all(actual[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] == bg)
            for r in range(ph) for c in range(pw)
        )
        return active == int(np.count_nonzero(control != bg)) and max(actual.shape) <= 30
    except (AssertionError, ValueError, KeyError):
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(4, 5)
    w = random.randint(4, 5)
    bg, separator_color, main_color, *signatures = random.sample(_PALETTE, 6)

    try:
        patterns = [
            _make_pattern(h, w, bg, main_color, signatures, i) for i in range(3)
        ]
    except ValueError:
        return None
    source_by_signature = dict(zip(signatures, patterns))

    marker_count = random.randint(3, 4 if min(h, w) == 4 else 6)
    positions = _control_positions(h, w, marker_count)
    marker_colors = signatures.copy()
    marker_colors.extend(random.choice(signatures) for _ in range(marker_count - 3))
    random.shuffle(marker_colors)
    control = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(positions, marker_colors):
        control[r, c] = color

    control_index = random.randrange(4)
    grid = np.full((2 * h + 1, 2 * w + 1), separator_color, dtype=int)
    source_iter = iter(patterns)
    for i, quadrant in enumerate(_QUADRANTS):
        _place_panel(grid, control if i == control_index else next(source_iter), quadrant)

    if not _semantic_validation(grid, control_index, bg, signatures, source_by_signature):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rare-signature stamp puzzle")
