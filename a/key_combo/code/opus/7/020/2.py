from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


LABEL_POOL = (Color.BLUE, Color.RED, Color.GREEN)
SPRITE_SIZE = 3
MACRO_SIZE = 3
GREY = Color.GREY
TEAL = Color.TEAL
BG = Color.BLACK

BASE_MASKS = (
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=bool),
)

GEN_RETRIES = 400


def _infer_min_period(seq: list[int]) -> int:
    assert seq, "expected a non-empty marker sequence"
    for period in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return period
    raise AssertionError("unreachable: every sequence has a period")


def _render_stamp(sprite: np.ndarray) -> np.ndarray:
    return np.where(sprite == TEAL, TEAL, GREY).astype(int)


def _extract_parts(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g[-1].min()) >= 0

    components = find_components(g, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    controls: dict[int, np.ndarray] = {}
    pattern_bbox = None
    pattern = None

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        region = crop_bbox(g, comp["bbox"])
        colors = set(map(int, np.unique(region)))
        h, w = region.shape

        if (
            (h, w) == (SPRITE_SIZE, SPRITE_SIZE)
            and colors != {TEAL}
            and TEAL in colors
            and len(colors) == 2
            and int(comp["area"]) == SPRITE_SIZE * SPRITE_SIZE
        ):
            label = next(color for color in colors if color != TEAL)
            assert label in LABEL_POOL, f"unexpected control label color {label}"
            assert label not in controls, "control labels must be unique"
            controls[int(label)] = region
            continue

        if (
            GREY in colors
            and colors <= {GREY, TEAL}
            and h % MACRO_SIZE == 0
            and w % MACRO_SIZE == 0
            and h >= 2 * MACRO_SIZE
            and w >= 2 * MACRO_SIZE
            and int(comp["area"]) == h * w
        ):
            assert pattern_bbox is None, "expected exactly one macro pattern block"
            pattern_bbox = (r0, c0, r1, c1)
            pattern = region

    assert 2 <= len(controls) <= 3, "expected 2 or 3 control sprites"
    assert pattern_bbox is not None and pattern is not None, "expected one macro pattern region"

    marker_cols = [int(c) for c in np.flatnonzero(g[-1] != BG)]
    marker_seq = [int(g[-1, c]) for c in marker_cols]
    assert len(marker_seq) >= 3, "expected a bottom marker sequence"
    assert all(color in controls for color in marker_seq), "every marker must match a control label"

    period = _infer_min_period(marker_seq)
    assert period in (2, 3), f"unexpected marker period {period}"
    phase_labels = marker_seq[:period]
    assert len(set(phase_labels)) == period, "phase labels must be unique to fix the minimal period"

    macro_rows = pattern.shape[0] // MACRO_SIZE
    macro_cols = pattern.shape[1] // MACRO_SIZE
    selector = np.zeros((macro_rows, macro_cols), dtype=bool)
    for r in range(macro_rows):
        for c in range(macro_cols):
            cell = pattern[
                r * MACRO_SIZE : (r + 1) * MACRO_SIZE,
                c * MACRO_SIZE : (c + 1) * MACRO_SIZE,
            ]
            cell_colors = set(map(int, np.unique(cell)))
            assert len(cell_colors) == 1, f"macro-cell {(r, c)} is not uniform"
            color = next(iter(cell_colors))
            assert color in {GREY, TEAL}, f"unexpected macro-cell color {color}"
            selector[r, c] = color == TEAL

    return {
        "grid": g,
        "controls": controls,
        "pattern_bbox": pattern_bbox,
        "selector": selector,
        "period": period,
        "phase_labels": phase_labels,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each teal macro-cell by the teal mask of the phase-selected control sprite.

    Input structure:
    - Black is the background.
    - The left side contains 2 or 3 full 3x3 control sprites made of teal plus one unique label color.
    - The upper-right region is a full grey/teal rectangle partitioned into uniform 3x3 macro-cells.
    - The bottom row contains a left-to-right sequence of label-color marker pixels whose minimal repeat period is 2 or 3.

    Rule:
    - Read the non-black bottom-row markers in left-to-right order and infer their minimal repeat period P.
    - Number the macro-pattern columns from 0 and map column j to the marker color at phase j mod P.
    - Use the control sprite with that label color as the stamp for every teal macro-cell in column j.
    - Inside a stamped 3x3 block, keep the control sprite's teal pixels as teal and turn its label-color pixels into grey; grey macro-cells remain solid grey.
    - Leave the control sprites and the bottom marker row unchanged.

    Key insight:
    - The bottom strip is not a direct per-column legend; it gives a periodic phase-to-sprite mapping that selects different 3x3 stamp masks for different macro-pattern columns.
    """
    parts = _extract_parts(np.asarray(grid, dtype=int))
    g = parts["grid"]
    out = g.copy()
    r0, c0, _r1, _c1 = parts["pattern_bbox"]
    selector = parts["selector"]
    period = parts["period"]
    phase_labels = parts["phase_labels"]
    rendered = {label: _render_stamp(sprite) for label, sprite in parts["controls"].items()}

    for r in range(selector.shape[0]):
        for c in range(selector.shape[1]):
            rs = r0 + r * MACRO_SIZE
            cs = c0 + c * MACRO_SIZE
            if selector[r, c]:
                label = phase_labels[c % period]
                paste(out, rendered[label], rs, cs)
            else:
                out[rs : rs + MACRO_SIZE, cs : cs + MACRO_SIZE] = GREY

    return out


def _random_mask() -> np.ndarray:
    while True:
        mask = BASE_MASKS[random.randrange(len(BASE_MASKS))].copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, k=random.randrange(4))
        if 2 <= int(mask.sum()) <= 7:
            return mask


def _sample_unique_masks(count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen = set()
    while len(masks) < count:
        mask = _random_mask()
        key = tuple(map(int, mask.ravel()))
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    return masks


def _make_selector(rows: int, cols: int, period: int) -> np.ndarray:
    selector = np.zeros((rows, cols), dtype=bool)
    required = []
    for phase in range(period):
        col_choices = [c for c in range(cols) if c % period == phase]
        c = random.choice(col_choices)
        r = random.randrange(rows)
        selector[r, c] = True
        required.append((r, c))

    density = random.uniform(0.35, 0.65)
    for r in range(rows):
        for c in range(cols):
            if selector[r, c]:
                continue
            selector[r, c] = random.random() < density

    if not np.any(~selector):
        free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in required]
        if free:
            rr, cc = random.choice(free)
            selector[rr, cc] = False
        else:
            selector[required[0]] = False
            selector[random.choice(required)] = True

    assert selector.any(), "selector needs at least one teal macro-cell"
    assert np.any(~selector), "selector needs at least one grey macro-cell"
    for phase in range(period):
        phase_cols = [c for c in range(cols) if c % period == phase]
        assert np.any(selector[:, phase_cols]), f"phase {phase} must affect some teal macro-cell"
    return selector


def _assemble_input(
    controls: dict[int, np.ndarray],
    control_order: list[int],
    pattern: np.ndarray,
    markers: list[int],
) -> np.ndarray:
    control_h = len(control_order) * SPRITE_SIZE + (len(control_order) - 1)
    control_w = SPRITE_SIZE
    pattern_h, pattern_w = pattern.shape

    top_control = random.randint(1, 3)
    top_pattern = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    gap_x = random.randint(3, 6)
    gap_y = random.randint(2, 4)
    right_pad = random.randint(1, 3)

    control_left = left_pad
    pattern_left = control_left + control_w + gap_x
    body_bottom = max(top_control + control_h, top_pattern + pattern_h)
    height = body_bottom + gap_y + 1
    width = pattern_left + pattern_w + right_pad

    grid = np.full((height, width), BG, dtype=int)
    for idx, label in enumerate(control_order):
        rr = top_control + idx * (SPRITE_SIZE + 1)
        paste(grid, controls[label], rr, control_left)

    paste(grid, pattern, top_pattern, pattern_left)
    for j, color in enumerate(markers):
        grid[-1, pattern_left + j * MACRO_SIZE + 1] = int(color)
    return grid


def _build_pattern(selector: np.ndarray) -> np.ndarray:
    rows, cols = selector.shape
    pattern = np.full((rows * MACRO_SIZE, cols * MACRO_SIZE), GREY, dtype=int)
    teal_block = np.full((MACRO_SIZE, MACRO_SIZE), TEAL, dtype=int)
    for r in range(rows):
        for c in range(cols):
            if selector[r, c]:
                paste(pattern, teal_block, r * MACRO_SIZE, c * MACRO_SIZE)
    return pattern


def _expected_from_spec(
    grid: np.ndarray,
    pattern_top_left: tuple[int, int],
    selector: np.ndarray,
    phase_labels: list[int],
    rendered: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0 = pattern_top_left
    period = len(phase_labels)
    for r in range(selector.shape[0]):
        for c in range(selector.shape[1]):
            rs = r0 + r * MACRO_SIZE
            cs = c0 + c * MACRO_SIZE
            if selector[r, c]:
                paste(out, rendered[phase_labels[c % period]], rs, cs)
            else:
                out[rs : rs + MACRO_SIZE, cs : cs + MACRO_SIZE] = GREY
    return out


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    *,
    phase_labels: list[int],
    selector: np.ndarray,
    controls: dict[int, np.ndarray],
    observed_marker_count: int,
) -> bool:
    try:
        parts = _extract_parts(grid)
    except Exception:
        return False

    if parts["period"] != len(phase_labels):
        return False
    if list(parts["phase_labels"]) != list(phase_labels):
        return False
    if not np.array_equal(parts["selector"], selector):
        return False

    rendered = {label: _render_stamp(sprite) for label, sprite in controls.items()}
    used_labels = list(phase_labels)
    for i in range(len(used_labels)):
        for j in range(i + 1, len(used_labels)):
            if np.array_equal(rendered[used_labels[i]], rendered[used_labels[j]]):
                return False

    if not np.any(expected != grid):
        return False

    pattern_r0, pattern_c0, _, _ = parts["pattern_bbox"]
    pattern_before = grid[
        pattern_r0 : pattern_r0 + selector.shape[0] * MACRO_SIZE,
        pattern_c0 : pattern_c0 + selector.shape[1] * MACRO_SIZE,
    ]
    pattern_after = expected[
        pattern_r0 : pattern_r0 + selector.shape[0] * MACRO_SIZE,
        pattern_c0 : pattern_c0 + selector.shape[1] * MACRO_SIZE,
    ]
    if not np.any(pattern_before != pattern_after):
        return False

    for phase in range(len(phase_labels)):
        phase_cols = [c for c in range(selector.shape[1]) if c % len(phase_labels) == phase]
        if not np.any(selector[:, phase_cols]):
            return False

    if observed_marker_count < len(phase_labels) + 1:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if len(phase_labels) >= 2:
        swapped = list(phase_labels)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        alt = _expected_from_spec(grid, (pattern_r0, pattern_c0), selector, swapped, rendered)
        if np.array_equal(alt, expected):
            return False

    return True


def _attempt() -> np.ndarray | None:
    control_count = random.choice((2, 3))
    period = 2 if control_count == 2 else random.choice((2, 3))
    macro_rows = random.randint(2, 4)
    macro_cols = random.randint(period + 1, 4)
    observed_markers = random.randint(period + 1, macro_cols)

    labels = random.sample(list(LABEL_POOL), control_count)
    masks = _sample_unique_masks(control_count)
    controls = {
        label: np.where(mask, TEAL, label).astype(int)
        for label, mask in zip(labels, masks)
    }

    phase_labels = random.sample(labels, period)
    selector = _make_selector(macro_rows, macro_cols, period)
    pattern = _build_pattern(selector)
    marker_seq = [phase_labels[i % period] for i in range(observed_markers)]

    control_order = labels[:]
    random.shuffle(control_order)
    grid = _assemble_input(controls, control_order, pattern, marker_seq)

    pattern_candidates = [
        comp["bbox"]
        for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
        if GREY in set(map(int, np.unique(crop_bbox(grid, comp["bbox"]))))
    ]
    if len(pattern_candidates) != 1:
        return None
    pattern_bbox = pattern_candidates[0]
    expected = _expected_from_spec(
        grid,
        (pattern_bbox[0], pattern_bbox[1]),
        selector,
        phase_labels,
        {label: _render_stamp(sprite) for label, sprite in controls.items()},
    )

    if not _validate_semantics(
        grid,
        expected,
        phase_labels=phase_labels,
        selector=selector,
        controls=controls,
        observed_marker_count=observed_markers,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic multi-stamp ARC input")
