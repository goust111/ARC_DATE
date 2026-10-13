from __future__ import annotations

from collections import Counter
import random

import numpy as np


MATCH_NUMERATOR = 4
MATCH_DENOMINATOR = 5

GEN_BG = 0
GEN_SEPARATOR = 5
GEN_FOREGROUND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_PANEL_COUNT_RANGE = (2, 6)
GEN_PANEL_HEIGHT_RANGE = (6, 10)
GEN_PANEL_WIDTH_RANGE = (6, 9)
GEN_RETRY_LIMIT = 400
GEN_LOCAL_RETRY_LIMIT = 80


def _mode_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _separator_runs(columns: list[int]) -> list[tuple[int, int]]:
    assert columns, "expected at least one separator column"
    runs: list[tuple[int, int]] = []
    start = columns[0]
    end = columns[0]
    for col in columns[1:]:
        if col == end + 1:
            end = col
        else:
            runs.append((start, end))
            start = col
            end = col
    runs.append((start, end))
    return runs


def _panel_spans(separator_columns: list[int], width: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    previous_end = -1
    for start, end in _separator_runs(separator_columns):
        if start > previous_end + 1:
            spans.append((previous_end + 1, start))
        previous_end = end
    if previous_end + 1 < width:
        spans.append((previous_end + 1, width))
    return spans


def _infer_separator_color(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    _, width = g.shape
    columns_by_color: dict[int, list[int]] = {}
    for col in range(width):
        value = int(g[0, col])
        if value == bg:
            continue
        if np.all(g[:, col] == value):
            columns_by_color.setdefault(value, []).append(col)

    candidates: list[tuple[int, int, int, int]] = []
    for color, cols in columns_by_color.items():
        spans = _panel_spans(cols, width)
        if len(spans) < 2:
            continue
        widths = [c1 - c0 for c0, c1 in spans]
        same_width_penalty = 0 if len(set(widths)) == 1 else 1
        candidates.append((same_width_penalty, -len(spans), -len(cols), int(color)))

    assert candidates, "expected full-height separator columns"
    candidates.sort()
    separator = candidates[0][3]
    columns = columns_by_color[separator]
    return int(separator), sorted(columns)


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _mode_color(g)
    separator, separator_columns = _infer_separator_color(g, bg)
    spans = _panel_spans(separator_columns, g.shape[1])
    panels = [g[:, c0:c1].copy() for c0, c1 in spans]
    assert len(panels) >= 2, "expected multiple panels"
    return panels, int(bg), int(separator)


def _match_rate(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    return float(np.count_nonzero(left == right)) / float(left.size)


def _is_merge_pair(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    same = int(np.count_nonzero(left == right))
    total = int(left.size)
    return same * MATCH_DENOMINATOR > total * MATCH_NUMERATOR


def _merge_group(group: list[np.ndarray]) -> np.ndarray:
    if len(group) == 1:
        return group[0].copy()

    stack = np.stack(group, axis=0)
    out = group[0].copy()
    height, width = out.shape
    for r in range(height):
        for c in range(width):
            values = [int(v) for v in stack[:, r, c]]
            counts = Counter(values)
            best = max(counts.values())
            for value in values:
                if counts[value] == best:
                    out[r, c] = value
                    break
    return out


def _assemble_panels(panels: list[np.ndarray], bg: int, separator: int) -> np.ndarray:
    assert panels, "expected at least one output panel"
    height = int(panels[0].shape[0])
    total_width = sum(int(panel.shape[1]) for panel in panels) + max(0, len(panels) - 1)
    out = np.full((height, total_width), int(bg), dtype=int)

    col = 0
    for index, panel in enumerate(panels):
        width = int(panel.shape[1])
        out[:, col : col + width] = panel
        col += width
        if index != len(panels) - 1:
            out[:, col] = int(separator)
            col += 1
    return out


def _group_panels(panels: list[np.ndarray]) -> list[list[np.ndarray]]:
    groups = [[panels[0]]]
    for panel in panels[1:]:
        if _is_merge_pair(groups[-1][-1], panel):
            groups[-1].append(panel)
        else:
            groups.append([panel])
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge adjacent panels when their patterns match in more than 80% of cells.

    Each input panel is separated by a full-height grey column. Panels whose
    corresponding cells agree on more than 80% of positions are fused into a
    single output panel; the fused panel keeps the per-cell majority color, with
    ties resolved in favor of the leftmost panel.
    """
    panels, bg, separator = _split_panels(np.asarray(grid, dtype=int))
    groups = _group_panels(panels)
    merged = [_merge_group(group) for group in groups]
    return _assemble_panels(merged, bg, separator)


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nb in _neighbors4(r, c, height, width):
                if nb not in cells:
                    frontier.add(nb)
        assert frontier, "connected growth frontier vanished unexpectedly"
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _paint_mask(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    patch = np.full(mask.shape, GEN_BG, dtype=int)
    phase = random.randrange(len(colors))
    style = random.randrange(4)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for index, (r, c) in enumerate(coords):
        if style == 0:
            color = colors[(r + c + phase) % len(colors)]
        elif style == 1:
            color = colors[(2 * r + c + phase) % len(colors)]
        elif style == 2:
            color = colors[(index + phase) % len(colors)]
        else:
            color = colors[((r % 2) + 2 * (c % 2) + phase) % len(colors)]
        patch[r, c] = int(color)
    return patch


def _random_panel(height: int, width: int) -> np.ndarray:
    panel = np.full((height, width), GEN_BG, dtype=int)

    box_height = random.randint(3, height - 1)
    box_width = random.randint(3, width - 1)
    top = random.randint(0, height - box_height)
    left = random.randint(0, width - box_width)

    box_area = box_height * box_width
    min_area = max(5, (2 * box_area) // 5)
    max_area = max(min_area, (3 * box_area) // 4)
    area = random.randint(min_area, max_area)

    mask = _connected_mask(box_height, box_width, area)
    color_count = random.choice([1, 2, 2, 3])
    colors = random.sample(GEN_FOREGROUND_COLORS, color_count)
    patch = _paint_mask(mask, colors)
    panel[top : top + box_height, left : left + box_width] = patch
    return panel


def _similar_panel(panel: np.ndarray) -> np.ndarray:
    fg_cells = [tuple(map(int, rc)) for rc in np.argwhere(panel != GEN_BG)]
    assert fg_cells, "expected a non-empty foreground pattern"
    base_colors = sorted({int(v) for v in np.unique(panel) if int(v) != GEN_BG})
    extra_colors = [c for c in GEN_FOREGROUND_COLORS if c not in base_colors]

    for _ in range(GEN_LOCAL_RETRY_LIMIT):
        candidate = panel.copy()
        max_changes = max(1, min(4, candidate.size // 10))
        change_count = random.randint(1, min(max_changes, len(fg_cells)))
        for r, c in random.sample(fg_cells, change_count):
            options = [GEN_BG, *base_colors]
            if extra_colors and random.random() < 0.35:
                options.append(random.choice(extra_colors))
            options = [value for value in options if value != int(candidate[r, c])]
            candidate[r, c] = random.choice(options)
        if np.array_equal(candidate, panel):
            continue
        if _is_merge_pair(panel, candidate):
            return candidate
    raise RuntimeError("failed to generate a similar panel")


def _different_panel(height: int, width: int, previous: np.ndarray) -> np.ndarray:
    for _ in range(GEN_LOCAL_RETRY_LIMIT):
        candidate = _random_panel(height, width)
        if _match_rate(previous, candidate) <= 0.7:
            return candidate
    raise RuntimeError("failed to generate a dissimilar panel")


def _sample_boundaries() -> list[bool]:
    scenario = random.choice(["mixed", "all_merge", "all_split"])
    if scenario == "mixed":
        panel_count = random.randint(max(3, GEN_PANEL_COUNT_RANGE[0]), GEN_PANEL_COUNT_RANGE[1])
        for _ in range(GEN_LOCAL_RETRY_LIMIT):
            boundaries = [bool(random.getrandbits(1)) for _ in range(panel_count - 1)]
            if any(boundaries) and not all(boundaries):
                return boundaries
        raise RuntimeError("failed to sample mixed merge boundaries")

    panel_count = random.randint(*GEN_PANEL_COUNT_RANGE)
    boundary_value = scenario == "all_merge"
    return [boundary_value] * (panel_count - 1)


def _expected_from_panels(panels: list[np.ndarray], boundaries: list[bool]) -> np.ndarray:
    groups: list[list[np.ndarray]] = [[panels[0]]]
    for merge_flag, panel in zip(boundaries, panels[1:]):
        if merge_flag:
            groups[-1].append(panel)
        else:
            groups.append([panel])
    merged = [_merge_group(group) for group in groups]
    return _assemble_panels(merged, GEN_BG, GEN_SEPARATOR)


def _gen_attempt() -> np.ndarray | None:
    boundaries = _sample_boundaries()
    panel_count = len(boundaries) + 1
    height = random.randint(*GEN_PANEL_HEIGHT_RANGE)
    width = random.randint(*GEN_PANEL_WIDTH_RANGE)

    panels = [_random_panel(height, width)]
    for merge_flag in boundaries:
        if merge_flag:
            panels.append(_similar_panel(panels[-1]))
        else:
            panels.append(_different_panel(height, width, panels[-1]))

    for merge_flag, left, right in zip(boundaries, panels, panels[1:]):
        if merge_flag and not _is_merge_pair(left, right):
            return None
        if (not merge_flag) and _is_merge_pair(left, right):
            return None

    inp = _assemble_panels(panels, GEN_BG, GEN_SEPARATOR)
    expected = _expected_from_panels(panels, boundaries)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
