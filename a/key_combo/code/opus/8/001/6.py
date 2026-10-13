from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _overlay(panel_a: np.ndarray, panel_b: np.ndarray) -> np.ndarray:
    out = panel_a.copy()
    mask = panel_b != BG
    out[mask] = panel_b[mask]
    return out


def _component_count_by_color(grid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in (int(v) for v in np.unique(grid) if int(v) != BG):
        counts[color] = len(
            find_components(
                grid,
                bg=BG,
                connectivity=4,
                monochromatic=True,
                colors=[color],
            )
        )
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the lower panel onto the upper panel, then summarize per-color component counts as a right-aligned histogram.

    Input structure:
    - The grid has two equal-height black-background panels stacked vertically.
    - The upper half is panel A and the lower half is panel B, aligned cell-for-cell.
    - Each panel contains several small 4-connected monochrome objects, and some projected positions overlap between panels.

    Rule:
    - Shift panel B upward so its rows align with panel A, then overlay it on top of panel A.
    - Every non-black pixel from panel B overwrites whatever color panel A had at the same position.
    - In the composite panel, count the 4-connected components of each non-black color, sort colors by component count descending, and build a histogram whose rows are right-aligned runs of those colors.

    Key insight:
    The lower panel is not just another set of visible pixels: by overwriting or bridging cells, it changes the final same-color connectivity, and those changed component counts are the quantities encoded in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0

    panel_h = g.shape[0] // 2
    panel_a = g[:panel_h]
    panel_b = g[panel_h:]
    composite = _overlay(panel_a, panel_b)
    counts = _component_count_by_color(composite)
    assert counts

    ordered = sorted(counts.items(), key=lambda item: -item[1])
    heights = [count for _color, count in ordered]
    assert len(set(heights)) == len(heights)

    out = np.zeros((len(ordered), max(heights)), dtype=int)
    for r, (color, count) in enumerate(ordered):
        out[r, -count:] = color
    return out


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _transform(cells: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out = list(cells)
    if random.random() < 0.5:
        out = [(r, w - 1 - c) for r, c in out]
    if random.random() < 0.5:
        out = [(h - 1 - r, c) for r, c in out]
    return _normalize(out)


def _random_shape(size: int) -> list[tuple[int, int]]:
    size = int(size)
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            nr, nc = random.choice(list(frontier))
            cells.add((nr, nc))

        shape = _normalize(list(cells))
        h, w = _dims(shape)
        if h <= min(5, size) and w <= min(6, size):
            return shape
    raise RuntimeError("failed to sample connected shape")


def _mark(mask: np.ndarray, cells: list[tuple[int, int]], *, diag: bool) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not diag and abs(dr) + abs(dc) > 1:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True


def _paint(panel: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        panel[r, c] = color


def _shift(cells: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in cells]


def _place_filler(
    panel: np.ndarray,
    panel_blocked: np.ndarray,
    proj_occupied: np.ndarray,
    proj_event_blocked: np.ndarray,
    color_blocked: np.ndarray,
    color: int,
    size_range: tuple[int, int],
) -> bool:
    h, w = panel.shape
    for _ in range(120):
        shape = _random_shape(random.randint(*size_range))
        sh, sw = _dims(shape)
        options: list[list[tuple[int, int]]] = []
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                cells = _shift(shape, r0, c0)
                if any(panel_blocked[r, c] for r, c in cells):
                    continue
                if any(proj_occupied[r, c] for r, c in cells):
                    continue
                if any(proj_event_blocked[r, c] for r, c in cells):
                    continue
                if any(color_blocked[r, c] for r, c in cells):
                    continue
                options.append(cells)
        if not options:
            continue

        cells = random.choice(options)
        _paint(panel, cells, color)
        _mark(panel_blocked, cells, diag=True)
        for r, c in cells:
            proj_occupied[r, c] = True
        _mark(color_blocked, cells, diag=False)
        return True
    return False


def _panel_ok(panel: np.ndarray) -> bool:
    colors = [int(v) for v in np.unique(panel) if int(v) != BG]
    if not 2 <= len(colors) <= 3:
        return False
    for color in colors:
        comps = find_components(
            panel,
            bg=BG,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        if not 1 <= len(comps) <= 3:
            return False
        if any(not 3 <= int(comp["area"]) <= 8 for comp in comps):
            return False
    return True


def _valid(grid: np.ndarray, event_color: int, expected_counts: dict[int, int], source_count: int) -> bool:
    panel_h = grid.shape[0] // 2
    panel_a = grid[:panel_h]
    panel_b = grid[panel_h:]
    if not _panel_ok(panel_a) or not _panel_ok(panel_b):
        return False

    composite = _overlay(panel_a, panel_b)
    counts = _component_count_by_color(composite)
    if counts != expected_counts:
        return False
    if len(set(counts.values())) != len(counts):
        return False
    if counts.get(event_color) == source_count:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == (3, 3)


def _generate_split_case(panel_h: int, width: int) -> np.ndarray | None:
    colors = random.sample(list(range(1, 10)), 3)
    color3, color2, color1 = colors
    panel_a = np.zeros((panel_h, width), dtype=int)
    panel_b = np.zeros((panel_h, width), dtype=int)
    blocked_a = np.zeros_like(panel_a, dtype=bool)
    blocked_b = np.zeros_like(panel_b, dtype=bool)
    proj_occupied = np.zeros_like(panel_a, dtype=bool)
    proj_event_blocked = np.zeros_like(panel_a, dtype=bool)
    color2_blocked = np.zeros_like(panel_a, dtype=bool)

    split_a = _transform(
        [(0, 0), (1, 0), (1, 1), (0, 2), (1, 2), (1, 3), (0, 4), (1, 4)],
        2,
        5,
    )
    cut_b = _transform([(1, 1), (1, 2), (1, 3)], 2, 5)
    top = random.randint(0, panel_h - 2)
    left = random.randint(0, width - 5)
    split_cells = _shift(split_a, top, left)
    cut_cells = _shift(cut_b, top, left)

    _paint(panel_a, split_cells, color3)
    _paint(panel_b, cut_cells, color1)
    _mark(blocked_a, split_cells, diag=True)
    _mark(blocked_b, cut_cells, diag=True)

    event_union = sorted(set(split_cells) | set(cut_cells))
    for r, c in event_union:
        proj_occupied[r, c] = True
    _mark(proj_event_blocked, event_union, diag=True)

    if not _place_filler(
        panel_a,
        blocked_a,
        proj_occupied,
        proj_event_blocked,
        color2_blocked,
        color2,
        (3, 5),
    ):
        return None
    if not _place_filler(
        panel_b,
        blocked_b,
        proj_occupied,
        proj_event_blocked,
        color2_blocked,
        color2,
        (3, 5),
    ):
        return None

    grid = np.vstack([panel_a, panel_b])
    expected_counts = {color3: 3, color2: 2, color1: 1}
    if not _valid(grid, color3, expected_counts, source_count=1):
        return None
    return grid


def _generate_merge_case(panel_h: int, width: int) -> np.ndarray | None:
    colors = random.sample(list(range(1, 10)), 3)
    color1, color2, color3 = colors
    panel_a = np.zeros((panel_h, width), dtype=int)
    panel_b = np.zeros((panel_h, width), dtype=int)
    blocked_a = np.zeros_like(panel_a, dtype=bool)
    blocked_b = np.zeros_like(panel_b, dtype=bool)
    proj_occupied = np.zeros_like(panel_a, dtype=bool)
    proj_event_blocked = np.zeros_like(panel_a, dtype=bool)
    color2_blocked = np.zeros_like(panel_a, dtype=bool)
    color3_blocked = np.zeros_like(panel_a, dtype=bool)

    left_obj = _transform([(0, 0), (1, 0), (1, 1)], 2, 2)
    right_obj = _transform([(0, 1), (1, 0), (1, 1)], 2, 2)
    bridge = _transform([(1, 1), (1, 2), (1, 3)], 2, 5)
    top = random.randint(0, panel_h - 2)
    left = random.randint(0, width - 5)
    left_cells = _shift(left_obj, top, left)
    right_cells = _shift([(r, c + 3) for r, c in right_obj], top, left)
    bridge_cells = _shift(bridge, top, left)

    _paint(panel_a, left_cells, color1)
    _paint(panel_a, right_cells, color1)
    _paint(panel_b, bridge_cells, color1)
    _mark(blocked_a, left_cells + right_cells, diag=True)
    _mark(blocked_b, bridge_cells, diag=True)

    event_union = sorted(set(left_cells) | set(right_cells) | set(bridge_cells))
    for r, c in event_union:
        proj_occupied[r, c] = True
    _mark(proj_event_blocked, event_union, diag=True)

    if not _place_filler(
        panel_a,
        blocked_a,
        proj_occupied,
        proj_event_blocked,
        color2_blocked,
        color2,
        (3, 5),
    ):
        return None
    if not _place_filler(
        panel_b,
        blocked_b,
        proj_occupied,
        proj_event_blocked,
        color2_blocked,
        color2,
        (3, 5),
    ):
        return None

    a_count = 1 if random.random() < 0.5 else 2
    b_count = 3 - a_count
    for _ in range(a_count):
        if not _place_filler(
            panel_a,
            blocked_a,
            proj_occupied,
            proj_event_blocked,
            color3_blocked,
            color3,
            (3, 5),
        ):
            return None
    for _ in range(b_count):
        if not _place_filler(
            panel_b,
            blocked_b,
            proj_occupied,
            proj_event_blocked,
            color3_blocked,
            color3,
            (3, 5),
        ):
            return None

    grid = np.vstack([panel_a, panel_b])
    expected_counts = {color3: 3, color2: 2, color1: 1}
    if not _valid(grid, color1, expected_counts, source_count=2):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        panel_h = random.randint(7, 10)
        width = random.randint(10, 14)
        if random.random() < 0.5:
            grid = _generate_split_case(panel_h, width)
        else:
            grid = _generate_merge_case(panel_h, width)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
