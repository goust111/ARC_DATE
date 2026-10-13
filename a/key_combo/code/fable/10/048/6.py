from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER = 4
SEPARATOR = 5
PATTERN_COLORS = (1, 2, 3, 6, 7, 8, 9)
PANEL_SIZE_RANGE = (6, 8)
GENERATION_RETRIES = 400
MIN_ROTATION_DISTANCE = 5

# Orientation = clockwise quarter-turns from the canonical panel.
CORNER_BY_ORIENTATION = {
    0: lambda n: (0, 0),
    1: lambda n: (0, n - 1),
    2: lambda n: (n - 1, n - 1),
    3: lambda n: (n - 1, 0),
}
ORIENTATION_BY_CORNER = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}

DECLARATION_PATTERNS = (
    (0, 0, 1, 1, 2, 2),
    (0, 0, 0, 1, 1, 2),
    (0, 0, 1, 1, 1, 2),
    (0, 0, 1, 2, 2, 3),
    (0, 1, 1, 2, 2, 3),
    (0, 1, 1, 2, 3, 3),
    (0, 0, 1, 2, 3, 3),
    (0, 1, 2, 2, 3, 3),
)


def rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    """Rotate a square grid clockwise by quarter-turns."""
    return np.rot90(grid, -int(turns) % 4)


def _panel_signature(panel: np.ndarray) -> bytes:
    return panel.tobytes()


def _marker_corner(panel: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(panel == MARKER)
    assert len(coords) == 1, "each panel must contain exactly one yellow marker"
    r, c = map(int, coords[0])
    n = int(panel.shape[0])
    if (r, c) == (0, 0):
        return r, c
    if (r, c) == (0, n - 1):
        return r, c
    if (r, c) == (n - 1, n - 1):
        return r, c
    if (r, c) == (n - 1, 0):
        return r, c
    raise AssertionError("yellow marker must lie on a panel corner")


def _declared_orientation(panel: np.ndarray) -> int:
    n = int(panel.shape[0])
    r, c = _marker_corner(panel)
    if (r, c) == (0, 0):
        return ORIENTATION_BY_CORNER["tl"]
    if (r, c) == (0, n - 1):
        return ORIENTATION_BY_CORNER["tr"]
    if (r, c) == (n - 1, n - 1):
        return ORIENTATION_BY_CORNER["br"]
    return ORIENTATION_BY_CORNER["bl"]


def _panel_without_marker(panel: np.ndarray) -> np.ndarray:
    out = panel.copy()
    r, c = _marker_corner(panel)
    out[r, c] = BG
    return out


def _extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[slice, slice]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    row_seps = [r for r in range(h) if np.all(g[r, :] == SEPARATOR)]
    col_seps = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(row_seps) == 1, "expected exactly one horizontal separator row"
    assert len(col_seps) == 2, "expected exactly two vertical separator columns"

    row_ranges = [(0, row_seps[0]), (row_seps[0] + 1, h)]
    col_ranges = [(0, col_seps[0]), (col_seps[0] + 1, col_seps[1]), (col_seps[1] + 1, w)]
    heights = [r1 - r0 for r0, r1 in row_ranges]
    widths = [c1 - c0 for c0, c1 in col_ranges]
    assert len(set(heights)) == 1 and len(set(widths)) == 1, "all panels must have equal size"
    assert heights[0] == widths[0], "rotated copies require square panels"

    panels: list[np.ndarray] = []
    locs: list[tuple[slice, slice]] = []
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            panel = g[r0:r1, c0:c1]
            assert panel.shape == (heights[0], widths[0])
            assert not np.any(panel == SEPARATOR), "separator color must not appear inside panels"
            panels.append(panel.copy())
            locs.append((slice(r0, r1), slice(c0, c1)))
    assert len(panels) == 6
    return panels, locs


def _analyze_panels(grid: np.ndarray) -> dict:
    panels, locs = _extract_panels(grid)
    declarations = [_declared_orientation(panel) for panel in panels]
    normalized = [np.rot90(panel, k) for panel, k in zip(panels, declarations)]
    sigs = [_panel_signature(panel) for panel in normalized]
    counts = Counter(sigs)
    majority_sig, majority_count = counts.most_common(1)[0]
    majority_index = sigs.index(majority_sig)
    mismatch_indices = [i for i, sig in enumerate(sigs) if sig != majority_sig]
    return {
        "panels": panels,
        "panel_locs": locs,
        "declarations": declarations,
        "normalized": normalized,
        "signatures": sigs,
        "counts": counts,
        "majority_sig": majority_sig,
        "majority_count": majority_count,
        "majority_panel": normalized[majority_index].copy(),
        "mismatch_indices": mismatch_indices,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Correct the unique panel whose rotated pattern contradicts its corner orientation marker.

    Input structure:
    - A grey separator row and two grey separator columns split the grid into 2x3 equal square panels.
    - Each panel contains the same non-symmetric mother pattern under some rotation plus exactly one yellow corner marker.
    - In the canonical orientation the yellow marker is at the top-left corner; rotating a truthful panel moves the marker to the matching corner.

    Rule:
    - Read each panel's declared orientation from its yellow corner and rotate the whole panel back to the canonical orientation.
    - The five consistent panels then coincide exactly; the one normalized outlier is the lying panel.
    - Replace only that panel by rotating the majority canonical panel back to the outlier panel's declared orientation, keeping all other panels and separators unchanged.

    Key insight:
    - The yellow corner is an orientation claim, not decoration: only after normalizing by that claim does the five-panel majority reveal the true mother pattern and expose the unique contradiction.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_panels(g)
    counts = info["counts"]
    assert sorted(counts.values()) == [1, 5], "expected exactly one mismatching normalized panel"

    mismatch_index = info["mismatch_indices"][0]
    declared = info["declarations"][mismatch_index]
    canonical = info["majority_panel"]
    corrected_panel = rotate_cw(canonical, declared)

    out = g.copy()
    rs, cs = info["panel_locs"][mismatch_index]
    out[rs, cs] = corrected_panel
    return out


def _transform_local(coords: list[tuple[int, int]], h: int, w: int, turns: int) -> list[tuple[int, int]]:
    turns %= 4
    if turns == 0:
        return coords
    if turns == 1:
        return [(c, h - 1 - r) for r, c in coords]
    if turns == 2:
        return [(h - 1 - r, w - 1 - c) for r, c in coords]
    return [(w - 1 - c, r) for r, c in coords]


def _random_shape_cells(n: int, kind: str) -> list[tuple[int, int]]:
    if kind == "rect":
        h = random.randint(1, min(3, n - 1))
        w = random.randint(1, min(3, n - 1))
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        return [(r0 + dr, c0 + dc) for dr in range(h) for dc in range(w)]
    if kind == "hline":
        w = random.randint(2, min(4, n - 1))
        r = random.randint(0, n - 1)
        c0 = random.randint(0, n - w)
        return [(r, c0 + dc) for dc in range(w)]
    if kind == "vline":
        h = random.randint(2, min(4, n - 1))
        r0 = random.randint(0, n - h)
        c = random.randint(0, n - 1)
        return [(r0 + dr, c) for dr in range(h)]
    if kind == "diag":
        length = random.randint(2, min(4, n - 1))
        if random.random() < 0.5:
            r0 = random.randint(0, n - length)
            c0 = random.randint(0, n - length)
            return [(r0 + i, c0 + i) for i in range(length)]
        r0 = random.randint(0, n - length)
        c0 = random.randint(length - 1, n - 1)
        return [(r0 + i, c0 - i) for i in range(length)]

    h = random.randint(2, min(4, n - 1))
    w = random.randint(2, min(4, n - 1))
    local = [(0, c) for c in range(w)] + [(r, 0) for r in range(1, h)]
    local = _transform_local(local, h, w, random.randrange(4))
    max_r = max(r for r, _ in local)
    max_c = max(c for _, c in local)
    r0 = random.randint(0, n - max_r - 1)
    c0 = random.randint(0, n - max_c - 1)
    return [(r0 + r, c0 + c) for r, c in local]


def _pattern_is_valid(pattern: np.ndarray) -> bool:
    occupied = np.argwhere(pattern != BG)
    if not (int(pattern.shape[0]) + 1 <= len(occupied) <= int(pattern.size * 0.55)):
        return False
    if len(set(map(int, pattern[pattern != BG].flatten()))) < 2:
        return False

    rows = occupied[:, 0]
    cols = occupied[:, 1]
    if rows.max() - rows.min() < 3 or cols.max() - cols.min() < 3:
        return False

    for corner in ((0, 0), (0, pattern.shape[1] - 1), (pattern.shape[0] - 1, 0), (pattern.shape[0] - 1, pattern.shape[1] - 1)):
        if int(pattern[corner]) != BG:
            return False

    for turns in (1, 2, 3):
        rotated = rotate_cw(pattern, turns)
        if np.array_equal(pattern, rotated):
            return False
        if int(np.sum(pattern != rotated)) < MIN_ROTATION_DISTANCE:
            return False
    return True


def _generate_pattern(n: int) -> np.ndarray:
    shape_kinds = ["rect", "rect", "hline", "vline", "diag", "L"]
    for _ in range(GENERATION_RETRIES):
        pattern = np.zeros((n, n), dtype=int)
        used = np.zeros((n, n), dtype=bool)
        reserved = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}

        palette = random.sample(PATTERN_COLORS, random.randint(2, 4))
        n_objects = random.randint(4, 6)
        object_colors = palette[:]
        while len(object_colors) < n_objects:
            object_colors.append(random.choice(palette))
        random.shuffle(object_colors)

        ok = True
        for color in object_colors:
            placed = False
            for _ in range(80):
                cells = _random_shape_cells(n, random.choice(shape_kinds))
                unique_cells = list(dict.fromkeys(cells))
                if any(cell in reserved for cell in unique_cells):
                    continue
                if any(used[r, c] for r, c in unique_cells):
                    continue
                if len(unique_cells) > max(4, n // 2) and random.random() < 0.35:
                    continue
                for r, c in unique_cells:
                    pattern[r, c] = color
                    used[r, c] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and _pattern_is_valid(pattern):
            return pattern
    raise RuntimeError("failed to generate a non-symmetric mother pattern")


def _make_panel(pattern: np.ndarray, declared: int, actual: int) -> np.ndarray:
    panel = rotate_cw(pattern, actual).copy()
    r, c = CORNER_BY_ORIENTATION[int(declared)](int(pattern.shape[0]))
    panel[r, c] = MARKER
    return panel


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    n = int(panels[0].shape[0])
    grid = np.full((2 * n + 1, 3 * n + 2), SEPARATOR, dtype=int)
    anchors = [
        (0, 0),
        (0, n + 1),
        (0, 2 * n + 2),
        (n + 1, 0),
        (n + 1, n + 1),
        (n + 1, 2 * n + 2),
    ]
    for panel, (r0, c0) in zip(panels, anchors):
        grid[r0 : r0 + n, c0 : c0 + n] = panel
    return grid


def _choose_orientations() -> tuple[list[int], int, int]:
    declared = list(random.choice(DECLARATION_PATTERNS))
    random.shuffle(declared)
    counts = Counter(declared)
    mismatch_candidates = [i for i, ori in enumerate(declared) if counts[ori] >= 2]
    mismatch_index = random.choice(mismatch_candidates)
    wrong_actual_choices = [ori for ori in sorted(set(declared)) if ori != declared[mismatch_index]]
    wrong_actual = random.choice(wrong_actual_choices)
    return declared, mismatch_index, wrong_actual


def _normalized_agreement_count(grid: np.ndarray) -> int:
    info = _analyze_panels(grid)
    return max(info["counts"].values())


def _semantic_check(
    grid: np.ndarray,
    pattern: np.ndarray,
    declared: list[int],
    actual: list[int],
    mismatch_index: int,
) -> bool:
    info = _analyze_panels(grid)
    if sorted(info["counts"].values()) != [1, 5]:
        return False
    if info["mismatch_indices"] != [mismatch_index]:
        return False
    if info["declarations"] != declared:
        return False

    canonical = info["majority_panel"]
    if int(canonical[0, 0]) != MARKER:
        return False
    if not np.array_equal(_panel_without_marker(canonical), pattern):
        return False

    if Counter(actual)[actual[mismatch_index]] < 2:
        return False
    if Counter(declared)[declared[mismatch_index]] < 2:
        return False

    out = main(grid)
    out_info = _analyze_panels(out)
    if list(out_info["counts"].values()) != [6]:
        return False

    expected_panel = _make_panel(pattern, declared[mismatch_index], declared[mismatch_index])
    rs, cs = info["panel_locs"][mismatch_index]
    if not np.array_equal(out[rs, cs], expected_panel):
        return False

    panels_in, _ = _extract_panels(grid)
    panels_out, _ = _extract_panels(out)
    for i in range(6):
        if i == mismatch_index:
            if np.array_equal(panels_in[i], panels_out[i]):
                return False
        elif not np.array_equal(panels_in[i], panels_out[i]):
            return False

    return _normalized_agreement_count(out) == 6


def _generate_case() -> np.ndarray | None:
    n = random.randint(*PANEL_SIZE_RANGE)
    pattern = _generate_pattern(n)
    declared, mismatch_index, wrong_actual = _choose_orientations()
    actual = declared[:]
    actual[mismatch_index] = wrong_actual

    panels = [
        _make_panel(pattern, dec, act)
        for dec, act in zip(declared, actual)
    ]
    grid = _assemble_grid(panels)
    if _semantic_check(grid, pattern, declared, actual, mismatch_index):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orientation-correction puzzle")
