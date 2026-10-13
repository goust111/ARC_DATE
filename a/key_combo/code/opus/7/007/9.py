from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

PANEL_COUNT = 4
SEPARATOR_COUNT = PANEL_COUNT - 1
PANEL_HEIGHT_CHOICES = (3, 4, 5, 6)
PANEL_WIDTH_CHOICES = (5, 6)
GENERATION_ATTEMPTS = 400

_SPEC_CACHE: dict[int, dict[tuple[bool, bool], list[tuple[int, int, int, int]]]] = {}


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_cols = [int(c) for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == SEPARATOR_COUNT

    widths = [
        int(sep_cols[0]),
        int(sep_cols[1] - sep_cols[0] - 1),
        int(sep_cols[2] - sep_cols[1] - 1),
        int(w - sep_cols[2] - 1),
    ]
    assert min(widths) > 0 and len(set(widths)) == 1

    panel_w = widths[0]
    starts = [0, sep_cols[0] + 1, sep_cols[1] + 1, sep_cols[2] + 1]
    panels = [g[:, start : start + panel_w].copy() for start in starts]
    assert all(panel.shape == (h, panel_w) for panel in panels)
    assert all(not np.any(panel == GREY) for panel in panels)
    return panels


def _observed_columns(panel: np.ndarray, color: int) -> tuple[int, ...]:
    if color == BLUE:
        mask = (panel == BLUE) | (panel == RED)
    else:
        assert color == GREEN
        mask = (panel == GREEN) | (panel == RED)
    return tuple(int(c) for c in np.flatnonzero(np.any(mask, axis=0)))


def _plain_columns(panel: np.ndarray, color: int) -> tuple[int, ...]:
    return tuple(int(c) for c in np.flatnonzero(np.any(panel == color, axis=0)))


def _model_columns(panel_w: int, period: int, step: int, frame: int) -> tuple[int, ...]:
    return tuple(
        int(c)
        for c in range(panel_w)
        if (int(c) - int(frame) * int(step)) % int(period) == 0
    )


def _fit_candidates(observed: tuple[tuple[int, ...], ...], panel_w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for period in range(2, panel_w + 1):
        for step in range(period):
            expected = tuple(
                _model_columns(panel_w, period, step, frame)
                for frame in range(len(observed))
            )
            if expected == observed:
                candidates.append((int(period), int(step)))
    return candidates


def _fit_model(observed: tuple[tuple[int, ...], ...], panel_w: int) -> tuple[int, int]:
    candidates = _fit_candidates(observed, panel_w)
    assert candidates
    return min(candidates)


def _render_panel(panel_h: int, panel_w: int, blue_cols: tuple[int, ...], green_cols: tuple[int, ...]) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    if blue_cols:
        panel[:, list(blue_cols)] = BLUE
    for c in green_cols:
        if c in blue_cols:
            panel[:, c] = RED
        else:
            panel[:, c] = GREEN
    return panel


def _complete_template(template: np.ndarray, green_cols: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(template, dtype=int).copy()
    for c in green_cols:
        black_mask = out[:, c] == BG
        blue_mask = out[:, c] == BLUE
        out[black_mask, c] = GREEN
        out[blue_mask, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the template frame by extrapolating the missing periodic green columns.

    Input structure:
    - The grid is a horizontal strip of four equal panels separated by full-height grey columns.
    - The first three panels overlay a blue column-period pattern and a green column-period pattern on black.
    - Red marks cells where the blue and green patterns overlap, and the fourth panel keeps only the blue next-frame template.

    Rule:
    - Split the strip on the grey separator columns to recover frames 0, 1, 2, and 3.
    - In frames 0, 1, and 2, treat each red cell as belonging to both the blue mask and the green mask.
    - For each color independently, find the smallest column period and per-frame shift whose predicted column sets match the observed frames.
    - Extrapolate the green columns to frame 3 and overlay them onto the template: black becomes green, while blue becomes red at overlaps.

    Key insight:
    - Red is not a third pattern; it is the evidence that one cell belongs to both periodic schedules at once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels = _split_panels(g)
    panel_h, panel_w = panels[0].shape

    blue_observed = tuple(_observed_columns(panel, BLUE) for panel in panels[:3])
    green_observed = tuple(_observed_columns(panel, GREEN) for panel in panels[:3])

    blue_period, blue_step = _fit_model(blue_observed, panel_w)
    green_period, green_step = _fit_model(green_observed, panel_w)

    expected_blue_template = _model_columns(panel_w, blue_period, blue_step, 3)
    assert _observed_columns(panels[3], BLUE) == expected_blue_template
    assert not np.any((panels[3] == GREEN) | (panels[3] == RED))

    predicted_green = _model_columns(panel_w, green_period, green_step, 3)
    return _complete_template(panels[3], predicted_green)


def _enumerate_specs(panel_w: int) -> dict[tuple[bool, bool], list[tuple[int, int, int, int]]]:
    if panel_w in _SPEC_CACHE:
        return _SPEC_CACHE[panel_w]

    grouped: dict[tuple[bool, bool], list[tuple[int, int, int, int]]] = {}
    for blue_period in range(2, panel_w + 1):
        for blue_step in range(1, blue_period):
            blue_frames = tuple(
                _model_columns(panel_w, blue_period, blue_step, frame)
                for frame in range(PANEL_COUNT)
            )
            for green_period in range(2, panel_w + 1):
                if green_period == blue_period:
                    continue
                for green_step in range(1, green_period):
                    green_frames = tuple(
                        _model_columns(panel_w, green_period, green_step, frame)
                        for frame in range(PANEL_COUNT)
                    )

                    if not any(set(blue_frames[k]) & set(green_frames[k]) for k in range(3)):
                        continue

                    blue_observed = tuple(blue_frames[:3])
                    green_observed = tuple(green_frames[:3])
                    if len(_fit_candidates(blue_observed, panel_w)) != 1:
                        continue
                    if len(_fit_candidates(green_observed, panel_w)) != 1:
                        continue

                    blue_plain = tuple(
                        tuple(sorted(set(blue_frames[k]) - set(green_frames[k])))
                        for k in range(3)
                    )
                    green_plain = tuple(
                        tuple(sorted(set(green_frames[k]) - set(blue_frames[k])))
                        for k in range(3)
                    )
                    if _fit_candidates(blue_plain, panel_w) == [(blue_period, blue_step)]:
                        continue
                    if _fit_candidates(green_plain, panel_w) == [(green_period, green_step)]:
                        continue

                    completed = _render_panel(
                        1,
                        panel_w,
                        blue_frames[3],
                        green_frames[3],
                    )[0]
                    if any(
                        np.array_equal(
                            completed,
                            _render_panel(1, panel_w, blue_frames[k], green_frames[k])[0],
                        )
                        for k in range(3)
                    ):
                        continue

                    key = (
                        bool(set(blue_frames[3]) & set(green_frames[3])),
                        green_frames[3] not in green_frames[:3],
                    )
                    grouped.setdefault(key, []).append(
                        (blue_period, blue_step, green_period, green_step)
                    )

    assert grouped
    _SPEC_CACHE[panel_w] = grouped
    return grouped


def _build_input(
    panel_h: int,
    panel_w: int,
    spec: tuple[int, int, int, int],
) -> np.ndarray:
    blue_period, blue_step, green_period, green_step = spec
    blue_frames = [
        _model_columns(panel_w, blue_period, blue_step, frame)
        for frame in range(PANEL_COUNT)
    ]
    green_frames = [
        _model_columns(panel_w, green_period, green_step, frame)
        for frame in range(PANEL_COUNT)
    ]

    panels = [
        _render_panel(panel_h, panel_w, blue_frames[frame], green_frames[frame])
        for frame in range(3)
    ]
    panels.append(_render_panel(panel_h, panel_w, blue_frames[3], ()))

    full_w = PANEL_COUNT * panel_w + SEPARATOR_COUNT
    grid = np.full((panel_h, full_w), BG, dtype=int)
    cursor = 0
    for idx, panel in enumerate(panels):
        grid[:, cursor : cursor + panel_w] = panel
        cursor += panel_w
        if idx != PANEL_COUNT - 1:
            grid[:, cursor] = GREY
            cursor += 1
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        panels = _split_panels(grid)
    except AssertionError:
        return False

    panel_h, panel_w = panels[0].shape
    if panel_h > 30 or panel_w > 30:
        return False

    blue_observed = tuple(_observed_columns(panel, BLUE) for panel in panels[:3])
    green_observed = tuple(_observed_columns(panel, GREEN) for panel in panels[:3])

    blue_candidates = _fit_candidates(blue_observed, panel_w)
    green_candidates = _fit_candidates(green_observed, panel_w)
    if len(blue_candidates) != 1 or len(green_candidates) != 1:
        return False

    blue_period, blue_step = blue_candidates[0]
    green_period, green_step = green_candidates[0]

    if blue_period == green_period:
        return False
    if not any(
        set(_model_columns(panel_w, blue_period, blue_step, k))
        & set(_model_columns(panel_w, green_period, green_step, k))
        for k in range(3)
    ):
        return False

    if np.any((panels[3] == GREEN) | (panels[3] == RED)):
        return False
    if _observed_columns(panels[3], BLUE) != _model_columns(panel_w, blue_period, blue_step, 3):
        return False

    predicted_green = _model_columns(panel_w, green_period, green_step, 3)
    expected = _complete_template(panels[3], predicted_green)

    blue_plain = tuple(_plain_columns(panel, BLUE) for panel in panels[:3])
    green_plain = tuple(_plain_columns(panel, GREEN) for panel in panels[:3])
    if _fit_candidates(blue_plain, panel_w) == [(blue_period, blue_step)]:
        return False
    if _fit_candidates(green_plain, panel_w) == [(green_period, green_step)]:
        return False

    if any(np.array_equal(expected, panel) for panel in panels[:3]):
        return False
    if np.array_equal(expected, panels[3]):
        return False

    actual = main(grid)
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        panel_w = random.choice(PANEL_WIDTH_CHOICES)
        grouped_specs = _enumerate_specs(panel_w)
        categories = [key for key, specs in grouped_specs.items() if specs]
        category = random.choice(categories)
        spec = random.choice(grouped_specs[category])
        panel_h = random.choice(PANEL_HEIGHT_CHOICES)

        grid = _build_input(panel_h, panel_w, spec)
        if _validate_semantics(grid):
            return grid

    raise RuntimeError("failed to generate a valid periodic overlay puzzle")
