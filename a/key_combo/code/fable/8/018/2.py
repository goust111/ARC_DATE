from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BG = BLACK
SEP = GREY
OCCLUDER = PINK
OUT_FG = GREEN
PANEL_FG_CHOICES = (BLUE, RED, YELLOW, ORANGE, TEAL, MAROON)

PANEL_H_RANGE = (8, 15)
PANEL_W_RANGE = (7, 14)
TILE_H_MAX = 5
TILE_W_MAX = 6
RECT_COUNT_RANGE = (1, 3)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 6
TILE_ATTEMPTS = 200
RECT_ATTEMPTS = 120
GEN_ATTEMPTS = 500


def _render_template(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rr = np.arange(shape[0]) % template.shape[0]
    cc = np.arange(shape[1]) % template.shape[1]
    return template[np.ix_(rr, cc)]


def _infer_min_template(panel: np.ndarray, unknown: int) -> np.ndarray:
    h, w = panel.shape
    observed = np.argwhere(panel != unknown)
    assert observed.size > 0, "expected at least one observed cell"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            template = np.full((ph, pw), -1, dtype=int)
            seen = np.zeros((ph, pw), dtype=bool)
            ok = True
            for r, c in observed:
                tr = int(r % ph)
                tc = int(c % pw)
                value = int(panel[r, c])
                if template[tr, tc] == -1:
                    template[tr, tc] = value
                    seen[tr, tc] = True
                elif template[tr, tc] != value:
                    ok = False
                    break
            if ok and np.all(seen):
                candidates.append((ph * pw, ph, pw, template.copy()))

    assert candidates, "no consistent fully observed period rectangle found"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _panel_foreground_color(panel: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(panel))) - {BG, OCCLUDER})
    assert len(colors) == 1, "each panel must expose exactly one visible foreground color"
    return colors[0]


def _restore_panel(panel: np.ndarray) -> tuple[np.ndarray, int]:
    fg = _panel_foreground_color(panel)
    template = _infer_min_template(panel, OCCLUDER)
    assert set(map(int, np.unique(template))).issubset({BG, fg}), "panel template must be binary"
    restored = _render_template(template, panel.shape)
    visible = panel != OCCLUDER
    assert np.array_equal(restored[visible], panel[visible]), "restoration must preserve visible cells"
    return restored, fg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each occluded periodic side panel, then paint the green overlap of their foreground cells.

    Input structure:
    - A solid grey separator column splits the grid into equal-width left and right panels.
    - Each side panel is a black-background binary periodic pattern with its own single foreground color.
    - Each panel also contains one or more pink rectangular occluders that hide part of the pattern.

    Rule:
    - Treat pink cells as unknown and infer, for each panel independently, the smallest rectangular period consistent with all visible cells.
    - Tile that period across the full panel to restore the hidden cells.
    - Output one panel-sized grid: paint green where both restored panels have foreground, otherwise paint black.

    Key insight:
    - The two panels must be restored before comparison; the correct overlap depends on hidden periodic cells, and the two periods are inferred separately.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert w % 2 == 1 and w >= 3, "expected equal panels with one separator column"

    sep_col = w // 2
    assert np.all(g[:, sep_col] == SEP), "expected a solid grey separator column"

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape == (h, sep_col)

    left_restored, left_fg = _restore_panel(left)
    right_restored, right_fg = _restore_panel(right)
    assert left_fg != right_fg, "the two panels must use different foreground colors"

    out = np.full(left.shape, BG, dtype=int)
    overlap = (left_restored == left_fg) & (right_restored == right_fg)
    out[overlap] = OUT_FG
    return out


def _sample_template(panel_h: int, panel_w: int, fg: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = min(TILE_H_MAX, panel_h)
    max_w = min(TILE_W_MAX, panel_w)
    for _ in range(TILE_ATTEMPTS):
        ph = random.randint(1, max_h)
        pw = random.randint(1, max_w)
        density = random.uniform(0.25, 0.6)
        mask = np.random.random((ph, pw)) < density
        if not (0 < int(mask.sum()) < ph * pw):
            continue

        template = np.where(mask, fg, BG).astype(int)
        full = _render_template(template, (panel_h, panel_w))
        inferred = _infer_min_template(full, unknown=-1)
        if inferred.shape != template.shape:
            continue
        if not np.array_equal(inferred, template):
            continue
        return template, full
    return None


def _all_residues_visible(covered: np.ndarray, tile_shape: tuple[int, int]) -> bool:
    ph, pw = tile_shape
    for r_mod in range(ph):
        for c_mod in range(pw):
            if not np.any(~covered[r_mod::ph, c_mod::pw]):
                return False
    return True


def _sample_occlusion_mask(
    panel: np.ndarray,
    tile_shape: tuple[int, int],
    fg: int,
) -> np.ndarray | None:
    h, w = panel.shape
    max_rect_h = min(RECT_SIDE_MAX, h)
    max_rect_w = min(RECT_SIDE_MAX, w)
    min_cover = max(4, (h * w) // 8)
    max_cover = max(min_cover, (h * w) // 2)

    for _ in range(RECT_ATTEMPTS):
        covered = np.zeros((h, w), dtype=bool)
        rect_count = random.randint(*RECT_COUNT_RANGE)
        ok = True
        for _ in range(rect_count):
            placed = False
            for _ in range(40):
                rh = random.randint(RECT_SIDE_MIN, max_rect_h)
                rw = random.randint(RECT_SIDE_MIN, max_rect_w)
                r0 = random.randint(0, h - rh)
                c0 = random.randint(0, w - rw)
                if np.any(covered[r0 : r0 + rh, c0 : c0 + rw]):
                    continue
                covered[r0 : r0 + rh, c0 : c0 + rw] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        covered_count = int(covered.sum())
        if not (min_cover <= covered_count <= max_cover):
            continue
        if not _all_residues_visible(covered, tile_shape):
            continue
        if not np.any(covered & (panel == fg)):
            continue
        return covered
    return None


def _expected_output(left_full: np.ndarray, right_full: np.ndarray, left_fg: int, right_fg: int) -> np.ndarray:
    out = np.full(left_full.shape, BG, dtype=int)
    out[(left_full == left_fg) & (right_full == right_fg)] = OUT_FG
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    if panel_h > 30 or panel_w * 2 + 1 > 30:
        return None

    left_fg, right_fg = random.sample(PANEL_FG_CHOICES, 2)
    left_sample = _sample_template(panel_h, panel_w, left_fg)
    right_sample = _sample_template(panel_h, panel_w, right_fg)
    if left_sample is None or right_sample is None:
        return None

    left_template, left_full = left_sample
    right_template, right_full = right_sample
    if left_template.shape == right_template.shape:
        return None

    left_cover = _sample_occlusion_mask(left_full, left_template.shape, left_fg)
    right_cover = _sample_occlusion_mask(right_full, right_template.shape, right_fg)
    if left_cover is None or right_cover is None:
        return None

    left_inp = left_full.copy()
    right_inp = right_full.copy()
    left_inp[left_cover] = OCCLUDER
    right_inp[right_cover] = OCCLUDER

    expected = _expected_output(left_full, right_full, left_fg, right_fg)
    naive = np.full(expected.shape, BG, dtype=int)
    naive[(left_inp == left_fg) & (right_inp == right_fg)] = OUT_FG

    green_count = int(np.count_nonzero(expected == OUT_FG))
    area = panel_h * panel_w
    if not (max(4, area // 12) <= green_count <= area - max(4, area // 12)):
        return None
    if int(np.count_nonzero(naive == OUT_FG)) == 0:
        return None
    if green_count <= int(np.count_nonzero(naive == OUT_FG)):
        return None
    if int(np.count_nonzero(expected != naive)) < 2:
        return None

    inp = np.full((panel_h, panel_w * 2 + 1), BG, dtype=int)
    inp[:, :panel_w] = left_inp
    inp[:, panel_w] = SEP
    inp[:, panel_w + 1 :] = right_inp

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-panel periodic-restore input")
