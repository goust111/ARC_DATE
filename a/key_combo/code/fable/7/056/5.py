from __future__ import annotations

import numpy as np


BG = 0
OUT_FG = 8
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_GEN_ATTEMPTS = 500


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(int(v) for v in np.unique(grid))
    if exclude:
        colors -= set(int(v) for v in exclude)
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    return bool(mask.sum() == (r1 - r0 + 1) * (c1 - c0 + 1) and mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _separator_row(grid: np.ndarray) -> int:
    h, _ = grid.shape
    assert h % 2 == 1, "input height must be odd so one separator row splits equal panels"
    sep = h // 2
    assert np.all(grid[sep] == grid[sep, 0]), "separator row must be monochrome"
    return sep


def _classify_panel(panel: np.ndarray) -> tuple[str, int, int | None]:
    colors = _colors_present(panel, exclude={BG})
    if len(colors) == 1:
        return "plain", int(colors[0]), None
    if len(colors) != 2:
        raise AssertionError("each panel must contain one foreground color, plus optional occluder")

    c0, c1 = map(int, colors)
    rect0 = _is_solid_rectangle(panel == c0)
    rect1 = _is_solid_rectangle(panel == c1)
    if rect0 == rect1:
        raise AssertionError("occluded panel must contain exactly one solid rectangle color")

    occ = c0 if rect0 else c1
    fg = c1 if rect0 else c0
    return "occluded", fg, occ


def _restore_symmetric_panel(panel: np.ndarray, fg: int, occ: int) -> np.ndarray:
    mirrored = np.fliplr(panel)
    occ_mask = panel == int(occ)
    assert not np.any(occ_mask & (mirrored == int(occ))), "occluder may not hide both cells of a mirror pair"

    restored = panel == int(fg)
    restored[occ_mask] = mirrored[occ_mask] == int(fg)
    assert np.array_equal(restored, np.fliplr(restored)), "restored panel must be left-right symmetric"
    return restored


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]

    top_kind, top_fg, top_occ = _classify_panel(top)
    bot_kind, bot_fg, bot_occ = _classify_panel(bottom)
    assert {top_kind, bot_kind} == {"plain", "occluded"}, "need one plain panel and one occluded panel"

    if top_kind == "occluded":
        restored = _restore_symmetric_panel(top, top_fg, int(top_occ))
        other = bottom == int(bot_fg)
    else:
        restored = _restore_symmetric_panel(bottom, bot_fg, int(bot_occ))
        other = top == int(top_fg)

    return restored, other


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the occluded mirror-symmetric panel, then keep only the foreground overlap shared by both panels.

    Input structure:
    - A single monochrome separator row splits the grid into two equal-sized top and bottom panels.
    - Each panel uses black as background and contains one foreground color; one panel also contains a different solid rectangular occluder.
    - The occluded panel's true foreground pattern is left-right mirror-symmetric before the rectangle hides part of it.

    Rule:
    - Detect which panel contains the solid rectangular occluder and treat that color as unknown rather than background.
    - Restore the hidden cells in that panel from their left-right mirror counterparts.
    - Align the restored panel with the other panel and output a black grid of panel size with cyan only where both panels have foreground.

    Key insight:
    - The rectangle is transparent for symmetry recovery, so the overlap must be computed after restoring the hidden foreground rather than on the raw visible pixels.
    """
    restored, other = _parse_input(grid)
    out = np.full(restored.shape, BG, dtype=int)
    out[restored & other] = OUT_FG
    return out


def _mirror_left_half(left: np.ndarray, width: int) -> np.ndarray:
    if width % 2 == 1:
        return np.concatenate([left, np.fliplr(left[:, :-1])], axis=1)
    return np.concatenate([left, np.fliplr(left)], axis=1)


def _random_symmetric_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    half_w = w // 2 + (w % 2)
    for _ in range(100):
        left = np.zeros((h, half_w), dtype=bool)
        for _ in range(int(rng.integers(2, 6))):
            rh = int(rng.integers(1, max(2, min(h, 5)) + 1))
            rw = int(rng.integers(1, max(2, min(half_w, 4)) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, half_w - rw + 1))
            left[r0 : r0 + rh, c0 : c0 + rw] = True
        if rng.random() < 0.6:
            rr = int(rng.integers(0, h))
            cc = int(rng.integers(0, half_w))
            left[rr, cc] = True

        full = _mirror_left_half(left, w)
        area = int(full.sum())
        if area < max(8, (h * w) // 7):
            continue
        if area > (h * w * 3) // 5:
            continue
        if _is_solid_rectangle(full):
            continue
        if np.count_nonzero(full.any(axis=0)) < 4:
            continue
        if np.count_nonzero(full.any(axis=1)) < 3:
            continue
        return full
    return None


def _random_panel_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(100):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(int(rng.integers(2, 6))):
            rh = int(rng.integers(1, max(2, min(h, 5)) + 1))
            rw = int(rng.integers(1, max(2, min(w, 5)) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            mask[r0 : r0 + rh, c0 : c0 + rw] = True
        if rng.random() < 0.5:
            rr = int(rng.integers(0, h))
            c0 = int(rng.integers(0, max(1, w - 2)))
            length = int(rng.integers(2, min(6, w - c0 + 1)))
            mask[rr, c0 : c0 + length] = True
        if rng.random() < 0.5:
            cc = int(rng.integers(0, w))
            r0 = int(rng.integers(0, max(1, h - 2)))
            length = int(rng.integers(2, min(6, h - r0 + 1)))
            mask[r0 : r0 + length, cc] = True

        area = int(mask.sum())
        if area < max(6, (h * w) // 8):
            continue
        if area > (h * w * 3) // 5:
            continue
        if np.count_nonzero(mask.any(axis=0)) < 3:
            continue
        if np.count_nonzero(mask.any(axis=1)) < 3:
            continue
        return mask
    return None


def _pick_occluder(sym_mask: np.ndarray, overlap: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    h, w = sym_mask.shape
    left_limit = w // 2
    right_start = (w + 1) // 2
    for side in rng.permutation(2):
        if side == 0:
            col_start_min, col_start_max = 0, left_limit
        else:
            col_start_min, col_start_max = right_start, w
        if col_start_max - col_start_min < 2:
            continue

        for _ in range(120):
            rh = int(rng.integers(2, min(h, 5) + 1))
            rw = int(rng.integers(2, min(col_start_max - col_start_min, 4) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(col_start_min, col_start_max - rw + 1))
            rect = np.zeros((h, w), dtype=bool)
            rect[r0 : r0 + rh, c0 : c0 + rw] = True
            hidden_fg = int((sym_mask & rect).sum())
            hidden_overlap = int((overlap & rect).sum())
            hidden_bg = int((~sym_mask & rect).sum())
            if hidden_fg < 2 or hidden_overlap < 1 or hidden_bg < 1:
                continue
            return rect
    return None


def _assemble_grid(
    sym_mask: np.ndarray,
    other_mask: np.ndarray,
    rect: np.ndarray,
    top_is_occluded: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = sym_mask.shape
    fg_sym, fg_other, occ, sep = rng.choice(np.array(INPUT_COLORS, dtype=int), size=4, replace=False)

    occluded_panel = np.full((h, w), BG, dtype=int)
    occluded_panel[sym_mask] = int(fg_sym)
    occluded_panel[rect] = int(occ)

    plain_panel = np.full((h, w), BG, dtype=int)
    plain_panel[other_mask] = int(fg_other)

    sep_row = np.full((1, w), int(sep), dtype=int)
    if top_is_occluded:
        return np.vstack([occluded_panel, sep_row, plain_panel])
    return np.vstack([plain_panel, sep_row, occluded_panel])


def _validate_case(grid: np.ndarray, sym_mask: np.ndarray, other_mask: np.ndarray, rect: np.ndarray) -> None:
    expected = np.full(sym_mask.shape, BG, dtype=int)
    overlap = sym_mask & other_mask
    expected[overlap] = OUT_FG

    produced = main(grid)
    assert np.array_equal(produced, expected), "main() must match the intended restored intersection"

    sep = _separator_row(grid)
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    top_kind, top_fg, _top_occ = _classify_panel(top)
    bot_kind, bot_fg, _bot_occ = _classify_panel(bottom)
    if top_kind == "occluded":
        raw_visible = top == int(top_fg)
    else:
        raw_visible = bottom == int(bot_fg)
    raw_overlap = raw_visible & other_mask

    assert overlap.sum() > 0, "output must be non-empty"
    assert not np.array_equal(raw_overlap, overlap), "restore-before-intersection must be necessary"
    assert np.any(overlap & rect), "the occluder must hide at least one true overlap pixel"
    assert np.any(overlap & ~rect), "some overlap should remain visibly outside the occluder"
    assert np.any(sym_mask & ~other_mask), "other panel must not already equal the restored symmetric panel"
    assert np.any(other_mask & ~sym_mask), "intersection must not collapse to the symmetric panel"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GEN_ATTEMPTS):
        h = int(rng.integers(7, 15))
        w = int(rng.integers(8, 17))

        sym_mask = _random_symmetric_mask(h, w, rng)
        if sym_mask is None:
            continue

        other_mask = _random_panel_mask(h, w, rng)
        if other_mask is None:
            continue

        overlap = sym_mask & other_mask
        if int(overlap.sum()) < max(4, (h * w) // 14):
            continue
        if np.array_equal(overlap, sym_mask):
            continue
        if np.array_equal(overlap, other_mask):
            continue

        rect = _pick_occluder(sym_mask, overlap, rng)
        if rect is None:
            continue

        top_is_occluded = bool(rng.integers(0, 2))
        grid = _assemble_grid(sym_mask, other_mask, rect, top_is_occluded, rng)

        try:
            _validate_case(grid, sym_mask, other_mask, rect)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
