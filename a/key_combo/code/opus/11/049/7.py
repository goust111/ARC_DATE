from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEP = 5
FG_COLORS = (1, 2, 3, 4)
PANEL_H = 5
PANEL_W = 5
SEP_W = 1
PANEL_COUNT = 3
INPUT_W = PANEL_COUNT * PANEL_W + (PANEL_COUNT - 1) * SEP_W
MAX_GENERATION_ATTEMPTS = 1000


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _partition_spans(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert sep_cols, "expected separator columns"
    groups = _groups(sep_cols)
    spans: list[tuple[int, int]] = []
    start = 0
    for c0, c1 in groups:
        if start < c0:
            spans.append((start, c0 - 1))
        start = c1 + 1
    if start < grid.shape[1]:
        spans.append((start, grid.shape[1] - 1))
    assert len(spans) == PANEL_COUNT, "expected three partitions"
    widths = {c1 - c0 + 1 for c0, c1 in spans}
    assert len(widths) == 1, "partitions must be equal width"
    return spans


def _settle_panel(panel: np.ndarray) -> np.ndarray:
    out = np.zeros_like(panel)
    height, width = panel.shape
    for c in range(width):
        values = panel[:, c][panel[:, c] != BG]
        out[height - len(values) :, c] = values
    return out


def _compose_panels(panels: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(panels[0])
    for panel in panels:
        mask = panel != BG
        out[mask] = panel[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle each separator-defined partition under gravity, then overlay the settled partitions with rightmost priority.

    Input structure:
    - The grid has a black background and two full-height gray separator columns.
    - The separator columns divide the canvas into three equal-width vertical partitions.
    - Each partition contains a sparse set of colored foreground pixels from the same ARC palette.

    Rule:
    - Ignore the gray separators and extract the three equal partitions.
    - In each partition independently, settle every column downward: keep the column's non-background pixels in top-to-bottom order and pack them against the bottom edge.
    - Create an output canvas with the same size as one partition and overlay the settled partitions from left to right, with every non-background pixel overwriting whatever was there before.

    Key insight:
    - The separators define three separate gravity worlds; only after each one settles can the aligned partitions be composited, and conflicts are resolved by spatial order rather than color value.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a non-empty grid"

    spans = _partition_spans(g)
    panels = [g[:, c0 : c1 + 1] for c0, c1 in spans]
    widths = {panel.shape[1] for panel in panels}
    heights = {panel.shape[0] for panel in panels}
    assert len(widths) == 1 and len(heights) == 1, "partitions must share one shape"

    settled = [_settle_panel(panel) for panel in panels]
    return _compose_panels(settled)


def _random_column_counts(total: int) -> list[int]:
    options: list[list[int]] = []
    for counts in itertools.product(range(4), repeat=PANEL_W):
        if sum(counts) != total:
            continue
        if max(counts) == 0:
            continue
        options.append(list(counts))
    return random.choice(options)


def _random_settled_panel() -> np.ndarray:
    total = random.randint(2, 4)
    counts = _random_column_counts(total)
    panel = np.zeros((PANEL_H, PANEL_W), dtype=int)
    for c, count in enumerate(counts):
        if count == 0:
            continue
        colors = [random.choice(FG_COLORS) for _ in range(count)]
        panel[PANEL_H - count :, c] = colors
    return panel


def _column_values(panel: np.ndarray, c: int) -> list[int]:
    return [int(v) for v in panel[:, c] if v != BG]


def _row_choices(count: int, *, require_unsettled: bool, require_gap: bool) -> list[tuple[int, ...]]:
    settled = tuple(range(PANEL_H - count, PANEL_H))
    choices: list[tuple[int, ...]] = []
    for rows in itertools.combinations(range(PANEL_H), count):
        if require_unsettled and rows == settled:
            continue
        if require_gap:
            gaps = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
            if not any(gap > 0 for gap in gaps):
                continue
        choices.append(rows)
    return choices


def _scatter_panel(
    settled_panel: np.ndarray,
    *,
    must_move_cols: set[int],
    gap_col: int | None,
) -> np.ndarray | None:
    panel = np.zeros_like(settled_panel)
    for c in range(PANEL_W):
        values = _column_values(settled_panel, c)
        count = len(values)
        if count == 0:
            continue
        rows = _row_choices(
            count,
            require_unsettled=c in must_move_cols,
            require_gap=gap_col == c,
        )
        if not rows:
            return None
        chosen_rows = random.choice(rows)
        for r, color in zip(chosen_rows, values):
            panel[r, c] = color
    return panel


def _owner_map(panels: list[np.ndarray]) -> np.ndarray:
    owners = np.full((PANEL_H, PANEL_W), -1, dtype=int)
    for idx, panel in enumerate(panels):
        owners[panel != BG] = idx
    return owners


def _has_visible_from_all(panels: list[np.ndarray]) -> bool:
    owners = _owner_map(panels)
    return all(np.any(owners == idx) for idx in range(PANEL_COUNT))


def _has_conflict_with_priority(panels: list[np.ndarray]) -> bool:
    p0, p1, p2 = panels
    has_mid_over_left = False
    has_right_over_earlier = False
    for r in range(PANEL_H):
        for c in range(PANEL_W):
            vals = [int(p0[r, c]), int(p1[r, c]), int(p2[r, c])]
            if vals[0] and vals[1] and vals[1] != vals[0]:
                has_mid_over_left = True
            if vals[2] and any(v != BG and v != vals[2] for v in vals[:2]):
                has_right_over_earlier = True
    return has_mid_over_left and has_right_over_earlier


def _avoids_color_shortcut(panels: list[np.ndarray]) -> bool:
    composed = _compose_panels(panels)
    numeric_max = np.maximum.reduce(panels)
    return not np.array_equal(composed, numeric_max)


def _avoids_reverse_priority_shortcut(panels: list[np.ndarray]) -> bool:
    composed = _compose_panels(panels)
    reverse = _compose_panels(list(reversed(panels)))
    return not np.array_equal(composed, reverse)


def _has_order_sensitive_column(panels: list[np.ndarray]) -> bool:
    for panel in panels:
        for c in range(PANEL_W):
            values = _column_values(panel, c)
            if len(values) >= 2 and len(set(values)) >= 2:
                return True
    return False


def _settled_panels_are_valid(panels: list[np.ndarray]) -> bool:
    counts = [int(np.count_nonzero(panel)) for panel in panels]
    return (
        all(2 <= count <= 4 for count in counts)
        and _has_visible_from_all(panels)
        and _has_conflict_with_priority(panels)
        and _avoids_color_shortcut(panels)
        and _avoids_reverse_priority_shortcut(panels)
        and _has_order_sensitive_column(panels)
    )


def _pick_scatter_requirements(settled_panels: list[np.ndarray]) -> tuple[list[set[int]], tuple[int, int] | None]:
    nonempty_columns: list[tuple[int, int]] = []
    multi_columns: list[tuple[int, int]] = []
    for p_idx, panel in enumerate(settled_panels):
        for c in range(PANEL_W):
            count = len(_column_values(panel, c))
            if count:
                nonempty_columns.append((p_idx, c))
            if count >= 2:
                multi_columns.append((p_idx, c))
    if len(nonempty_columns) < 2 or not multi_columns:
        return [set() for _ in range(PANEL_COUNT)], None

    forced_move = [set() for _ in range(PANEL_COUNT)]
    gap_target = random.choice(multi_columns)
    forced_move[gap_target[0]].add(gap_target[1])

    other_choices = [item for item in nonempty_columns if item != gap_target]
    if other_choices:
        other = random.choice(other_choices)
        forced_move[other[0]].add(other[1])

    for p_idx, c in nonempty_columns:
        if random.random() < 0.35:
            forced_move[p_idx].add(c)
    return forced_move, gap_target


def _build_input_from_settled(settled_panels: list[np.ndarray]) -> np.ndarray | None:
    forced_move, gap_target = _pick_scatter_requirements(settled_panels)
    scattered: list[np.ndarray] = []
    for p_idx, panel in enumerate(settled_panels):
        gap_col = gap_target[1] if gap_target is not None and gap_target[0] == p_idx else None
        scattered_panel = _scatter_panel(panel, must_move_cols=forced_move[p_idx], gap_col=gap_col)
        if scattered_panel is None:
            return None
        scattered.append(scattered_panel)

    expected = _compose_panels(settled_panels)
    raw_overlay = _compose_panels(scattered)
    if np.array_equal(raw_overlay, expected):
        return None

    grid = np.full((PANEL_H, INPUT_W), BG, dtype=int)
    sep_cols = [PANEL_W, PANEL_W + SEP_W + PANEL_W]
    for c in sep_cols:
        grid[:, c] = SEP
    for p_idx, panel in enumerate(scattered):
        start = p_idx * (PANEL_W + SEP_W)
        grid[:, start : start + PANEL_W] = panel
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _generate_case() -> np.ndarray | None:
    settled_panels = [_random_settled_panel() for _ in range(PANEL_COUNT)]
    if not _settled_panels_are_valid(settled_panels):
        return None
    return _build_input_from_settled(settled_panels)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partitioned-gravity overlay input")
