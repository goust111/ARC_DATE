from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components


def _foreground_colors(panel: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in panel.flat if int(v) != Color.BLACK)
    assert len(counts) >= 2, "expected two foreground colors inside the panel"
    colors = sorted(counts, key=lambda color: (-counts[color], color))
    return int(colors[0]), int(colors[1])


def _majority_foreground(sub: np.ndarray) -> int | None:
    counts = Counter(int(v) for v in sub.flat if int(v) != Color.BLACK)
    if not counts:
        return None
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _denoise_panel(panel: np.ndarray, fg_colors: tuple[int, int] | None = None) -> np.ndarray:
    colors = _foreground_colors(panel) if fg_colors is None else tuple(map(int, fg_colors))
    clean = panel.copy()
    comps = find_components(panel, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=colors)
    for comp in comps:
        if int(comp["area"]) != 1:
            continue
        r, c = map(int, comp["coords"][0])
        majority = _majority_foreground(panel[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2])
        if majority is not None and majority != int(panel[r, c]):
            clean[r, c] = majority
    return clean


def _defect_heatmap(panel: np.ndarray) -> np.ndarray:
    h, w = panel.shape
    row_mid = h // 2
    col_mid = w // 2
    out = np.zeros((2, 2), dtype=int)
    if np.any(panel[:row_mid, :col_mid] == Color.BLACK):
        out[0, 0] = Color.TEAL
    if np.any(panel[:row_mid, col_mid:] == Color.BLACK):
        out[0, 1] = Color.TEAL
    if np.any(panel[row_mid:, :col_mid] == Color.BLACK):
        out[1, 0] = Color.TEAL
    if np.any(panel[row_mid:, col_mid:] == Color.BLACK):
        out[1, 1] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise a two-color panel, keep true black dents, then compress defect presence into a 2x2 quadrant heatmap.

    Input structure:
    - The grid has a black background and one large rectangular panel inset from the border.
    - The panel is filled by two foreground colors forming two large regions separated by a roughly central horizontal or vertical boundary.
    - Inside the panel there are a few black dents and a few isolated one-pixel wrong-color noise specks.

    Rule:
    - Find the panel bbox as the non-black bounding rectangle.
    - Within the panel, recolor each foreground 4-connected singleton whose 3x3 neighborhood has a different unique majority foreground color.
    - Treat the black pixels remaining after this denoising step as true defects.
    - Split the panel bbox at its horizontal and vertical midlines and output a 2x2 grid whose cell is teal iff the matching quadrant contains at least one true defect, otherwise black.

    Key insight:
    - The isolated wrong-color specks are local contamination, while the persistent black holes are the real signal; the final answer only keeps quadrant-level defect presence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_box = bbox(g != Color.BLACK)
    assert panel_box is not None, "expected a non-black panel"
    panel = crop_bbox(g, panel_box)
    clean = _denoise_panel(panel)
    return _defect_heatmap(clean)


GEN_BG = Color.BLACK
GEN_OUT = Color.TEAL
GEN_PANEL_DIMENSIONS = [10, 12, 14]
GEN_MARGIN_CHOICES = [1, 2]
GEN_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.MAROON]
GEN_MAX_ATTEMPTS = 500
GEN_QUADRANTS = (0, 1, 2, 3)
GEN_DEFECT_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)


def _quadrant_ranges(h: int, w: int) -> dict[int, tuple[range, range]]:
    row_mid = h // 2
    col_mid = w // 2
    return {
        0: (range(0, row_mid), range(0, col_mid)),
        1: (range(0, row_mid), range(col_mid, w)),
        2: (range(row_mid, h), range(0, col_mid)),
        3: (range(row_mid, h), range(col_mid, w)),
    }


def _quadrant_of_cell(h: int, w: int, r: int, c: int) -> int:
    return (0 if r < h // 2 else 2) + (0 if c < w // 2 else 1)


def _build_clean_panel(h: int, w: int, colors: tuple[int, int], orientation: str, rng: random.Random) -> np.ndarray:
    a, b = map(int, colors)
    panel = np.zeros((h, w), dtype=int)

    if orientation == "vertical":
        center = w // 2
        lo = max(3, center - 2)
        hi = min(w - 3, center + 2)
        cut = rng.randint(lo, hi)
        cuts: list[int] = []
        for _ in range(h):
            cut += rng.choice([-1, 0, 1])
            cut = min(max(cut, lo), hi)
            cuts.append(cut)
        if len(set(cuts)) == 1:
            idx = rng.randrange(h)
            delta = -1 if cuts[idx] == hi else 1
            cuts[idx] += delta
        left_color, right_color = (a, b) if rng.random() < 0.5 else (b, a)
        for r, cut in enumerate(cuts):
            panel[r, :cut] = left_color
            panel[r, cut:] = right_color
    else:
        center = h // 2
        lo = max(3, center - 2)
        hi = min(h - 3, center + 2)
        cut = rng.randint(lo, hi)
        cuts = []
        for _ in range(w):
            cut += rng.choice([-1, 0, 1])
            cut = min(max(cut, lo), hi)
            cuts.append(cut)
        if len(set(cuts)) == 1:
            idx = rng.randrange(w)
            delta = -1 if cuts[idx] == hi else 1
            cuts[idx] += delta
        top_color, bottom_color = (a, b) if rng.random() < 0.5 else (b, a)
        for c, cut in enumerate(cuts):
            panel[:cut, c] = top_color
            panel[cut:, c] = bottom_color

    return panel


def _cell_majority_matches(panel: np.ndarray, cells: list[tuple[int, int]], host: int) -> bool:
    for r, c in cells:
        majority = _majority_foreground(panel[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2])
        if majority != int(host):
            return False
    return True


def _quadrant_candidate_shapes(
    clean_panel: np.ndarray,
    quadrant: int,
) -> list[list[tuple[int, int]]]:
    h, w = clean_panel.shape
    row_range, col_range = _quadrant_ranges(h, w)[quadrant]
    r0 = row_range.start
    r1 = row_range.stop - 1
    c0 = col_range.start
    c1 = col_range.stop - 1

    candidates: list[list[tuple[int, int]]] = []
    for shape in GEN_DEFECT_SHAPES:
        max_dr = max(dr for dr, _ in shape)
        max_dc = max(dc for _, dc in shape)
        for r in range(r0 + 1, r1 - max_dr):
            for c in range(c0 + 1, c1 - max_dc):
                cells = [(r + dr, c + dc) for dr, dc in shape]
                host = int(clean_panel[cells[0]])
                if host == GEN_BG or any(int(clean_panel[rr, cc]) != host for rr, cc in cells):
                    continue
                if not _cell_majority_matches(clean_panel, cells, host):
                    continue
                candidates.append(cells)
    return candidates


def _far_enough(existing: set[tuple[int, int]], cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in existing:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _choose_active_quadrants(rng: random.Random) -> list[int]:
    size = rng.choice([2, 2, 3])
    return sorted(rng.sample(list(GEN_QUADRANTS), size))


def _defect_cluster_counts(active_quadrants: list[int], rng: random.Random) -> dict[int, int]:
    patterns = {
        2: [(1, 1), (2, 1), (1, 2), (2, 2)],
        3: [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
    }
    chosen = rng.choice(patterns[len(active_quadrants)])
    return {quadrant: int(count) for quadrant, count in zip(active_quadrants, chosen, strict=False)}


def _place_defects(clean_panel: np.ndarray, rng: random.Random) -> tuple[np.ndarray, set[tuple[int, int]], np.ndarray] | None:
    active_quadrants = _choose_active_quadrants(rng)
    counts = _defect_cluster_counts(active_quadrants, rng)

    defect_cells: set[tuple[int, int]] = set()
    panel = clean_panel.copy()
    expected = np.zeros((2, 2), dtype=int)

    for quadrant in active_quadrants:
        candidates = _quadrant_candidate_shapes(clean_panel, quadrant)
        rng.shuffle(candidates)
        chosen = 0
        for cells in candidates:
            if not _far_enough(defect_cells, cells):
                continue
            for r, c in cells:
                panel[r, c] = GEN_BG
                defect_cells.add((r, c))
            chosen += 1
            if chosen == counts[quadrant]:
                break
        if chosen != counts[quadrant]:
            return None
        expected[quadrant // 2, quadrant % 2] = GEN_OUT

    if np.all(expected == GEN_OUT) or np.all(expected == GEN_BG):
        return None
    return panel, defect_cells, expected


def _noise_candidate_cells(panel_with_dents: np.ndarray, clean_panel: np.ndarray, quadrant: int) -> list[tuple[int, int]]:
    h, w = clean_panel.shape
    row_range, col_range = _quadrant_ranges(h, w)[quadrant]
    out: list[tuple[int, int]] = []
    for r in range(row_range.start + 1, row_range.stop - 1):
        for c in range(col_range.start + 1, col_range.stop - 1):
            host = int(clean_panel[r, c])
            if host == GEN_BG or int(panel_with_dents[r, c]) != host:
                continue
            nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if any(int(panel_with_dents[rr, cc]) != host for rr, cc in nbrs):
                continue
            majority = _majority_foreground(panel_with_dents[r - 1 : r + 2, c - 1 : c + 2])
            if majority != host:
                continue
            out.append((r, c))
    return out


def _place_noise(
    panel_with_dents: np.ndarray,
    clean_panel: np.ndarray,
    expected: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    color_a, color_b = _foreground_colors(clean_panel)
    active = {2 * r + c for r in range(2) for c in range(2) if int(expected[r, c]) == GEN_OUT}
    inactive = [quadrant for quadrant in GEN_QUADRANTS if quadrant not in active]
    noise_count = rng.randint(1, 3)

    desired_quadrants: list[int] = []
    if inactive:
        desired_quadrants.append(rng.choice(inactive))
    while len(desired_quadrants) < noise_count:
        desired_quadrants.append(rng.choice(list(GEN_QUADRANTS)))

    noisy = panel_with_dents.copy()
    noise_map: dict[tuple[int, int], int] = {}
    used: set[tuple[int, int]] = set()
    for quadrant in desired_quadrants:
        candidates = _noise_candidate_cells(noisy, clean_panel, quadrant)
        rng.shuffle(candidates)
        placed = None
        for r, c in candidates:
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
                continue
            placed = (r, c)
            break
        if placed is None:
            return None
        r, c = placed
        host = int(clean_panel[r, c])
        other = color_b if host == color_a else color_a
        noisy[r, c] = int(other)
        noise_map[(r, c)] = host
        used.add((r, c))

    return noisy, noise_map


def _semantic_checks(
    grid: np.ndarray,
    clean_panel: np.ndarray,
    panel_with_dents: np.ndarray,
    defect_cells: set[tuple[int, int]],
    noise_map: dict[tuple[int, int], int],
    expected: np.ndarray,
) -> bool:
    panel_box = bbox(grid != GEN_BG)
    if panel_box is None:
        return False
    observed_panel = crop_bbox(grid, panel_box)
    if observed_panel.shape != clean_panel.shape:
        return False

    fg_colors = _foreground_colors(observed_panel)
    if not np.array_equal(_denoise_panel(clean_panel, _foreground_colors(clean_panel)), clean_panel):
        return False

    denoised = _denoise_panel(observed_panel, fg_colors)
    if not np.array_equal(denoised, panel_with_dents):
        return False

    changed = {tuple(map(int, rc)) for rc in np.argwhere(denoised != observed_panel)}
    if changed != set(noise_map):
        return False

    for pos, host in noise_map.items():
        if int(denoised[pos]) != int(host):
            return False

    observed_defects = {tuple(map(int, rc)) for rc in np.argwhere(denoised == GEN_BG)}
    if observed_defects != defect_cells:
        return False

    inactive_noise = [pos for pos in noise_map if int(expected[_quadrant_of_cell(*clean_panel.shape, *pos) // 2, _quadrant_of_cell(*clean_panel.shape, *pos) % 2]) == GEN_BG]
    if not inactive_noise:
        return False

    if not np.array_equal(_defect_heatmap(denoised), expected):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    panel_h = rng.choice(GEN_PANEL_DIMENSIONS)
    panel_w = rng.choice(GEN_PANEL_DIMENSIONS)
    top = rng.choice(GEN_MARGIN_CHOICES)
    bottom = rng.choice(GEN_MARGIN_CHOICES)
    left = rng.choice(GEN_MARGIN_CHOICES)
    right = rng.choice(GEN_MARGIN_CHOICES)
    colors = tuple(rng.sample(GEN_COLORS, 2))
    orientation = rng.choice(["vertical", "horizontal"])

    clean_panel = _build_clean_panel(panel_h, panel_w, colors, orientation, rng)
    if not np.array_equal(_denoise_panel(clean_panel, colors), clean_panel):
        return None

    defect_result = _place_defects(clean_panel, rng)
    if defect_result is None:
        return None
    panel_with_dents, defect_cells, expected = defect_result

    noise_result = _place_noise(panel_with_dents, clean_panel, expected, rng)
    if noise_result is None:
        return None
    noisy_panel, noise_map = noise_result

    grid = np.zeros((top + panel_h + bottom, left + panel_w + right), dtype=int)
    grid[top : top + panel_h, left : left + panel_w] = noisy_panel

    if not _semantic_checks(grid, clean_panel, panel_with_dents, defect_cells, noise_map, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
