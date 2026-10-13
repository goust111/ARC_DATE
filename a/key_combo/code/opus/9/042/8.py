from __future__ import annotations

import itertools
from typing import Iterable

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

PALETTE = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL)
GEN_ATTEMPTS = 400


def _group_runs(cols: Iterable[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for col in sorted(int(c) for c in cols):
        if not runs or col != runs[-1][1] + 1:
            runs.append((col, col))
        else:
            runs[-1] = (runs[-1][0], col)
    return runs


def _separator_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    runs = _group_runs(sep_cols)
    assert runs, "expected at least one separator"
    for c0, c1 in runs:
        assert 2 <= c1 - c0 + 1 <= 3, "separator width must be 2 or 3"
    return runs


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    H, W = map(int, grid.shape)
    runs = _separator_runs(grid)
    panels: list[np.ndarray] = []
    left = 0
    for c0, c1 in runs:
        assert left < c0, "empty panel"
        panels.append(grid[:, left:c0].copy())
        left = c1 + 1
    assert left < W, "empty trailing panel"
    panels.append(grid[:, left:W].copy())
    assert 3 <= len(panels) <= 4, "expected 3 or 4 panels"
    heights = {int(panel.shape[0]) for panel in panels}
    assert len(heights) == 1 and H in heights
    return panels


def _decode_offset_from_corner(corner: np.ndarray) -> int:
    coords = np.argwhere(corner == GREY)
    assert coords.shape == (2, 2), "glyph must contain exactly two grey pixels"
    rows = {int(r) for r, _c in coords}
    cols = {int(c) for _r, c in coords}
    if len(rows) == 1:
        return 0
    if len(cols) == 1:
        return 1
    return -1


def _parse_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    panels = _split_panels(np.asarray(grid, dtype=int))
    offsets: list[int] = []
    cleaned: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        assert panel.shape[0] >= 2 and panel.shape[1] >= 2
        if i == 0:
            assert not np.any(panel == GREY), "first panel must not contain a glyph"
            cleaned.append(panel.copy())
            continue
        corner = panel[:2, -2:]
        offsets.append(_decode_offset_from_corner(corner))
        clear = panel.copy()
        clear[clear == GREY] = BLACK
        cleaned.append(clear)
    return cleaned, offsets


def _row_starts_cumulative(offsets: list[int]) -> list[int]:
    rows = [0]
    for off in offsets:
        rows.append(rows[-1] + int(off))
    return rows


def _row_starts_direct(offsets: list[int]) -> list[int]:
    rows = [0] + [int(off) for off in offsets]
    shift = -min(rows)
    return [r + shift for r in rows]


def _stitch_panels(
    panels: list[np.ndarray],
    rows: list[int],
    *,
    fill: int = MAROON,
) -> np.ndarray:
    assert len(panels) == len(rows)
    H = int(panels[0].shape[0])
    widths = [int(panel.shape[1]) for panel in panels]
    top_shift = -min(0, min(rows))
    placed_rows = [int(r + top_shift) for r in rows]
    out_h = max(int(r) + H for r in placed_rows)
    out_w = sum(widths)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    c = 0
    for panel, r in zip(panels, placed_rows):
        w = int(panel.shape[1])
        out[r : r + H, c : c + w] = panel
        c += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode corner glyphs into vertical offsets, erase them, and stitch the panels into one staggered strip.

    Input structure:
    - The grid has black background with 3 or 4 equal-height rectangular panels separated by full-height maroon columns.
    - The first panel has no grey marker; every later panel has a 2-pixel grey glyph inside its top-right 2x2 corner.
    - Panel contents use non-black colors, while maroon appears only in the separator columns and grey appears only in the glyphs.

    Rule:
    - Split the grid at the maroon separator runs to recover the left-to-right panel sequence.
    - Decode each later panel's grey corner glyph by shape: horizontal pair -> row offset 0, vertical pair -> row offset +1, diagonal pair -> row offset -1.
    - Erase all grey glyph pixels to black, then place the first panel at the top-left and each later panel immediately to the right with its start row equal to the previous panel's start row plus the decoded offset.
    - The output canvas is filled with maroon first; each panel rectangle, including black cells, is copied onto it.

    Key insight:
    - The grey glyph does not encode a color replacement; its local 2-pixel shape encodes the next panel's vertical step in the stitched layout.
    """
    cleaned, offsets = _parse_panels(np.asarray(grid, dtype=int))
    rows = _row_starts_cumulative(offsets)
    return _stitch_panels(cleaned, rows, fill=MAROON)


def _random_connected_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    area: int,
    forbidden: set[tuple[int, int]],
) -> np.ndarray | None:
    allowed = [(r, c) for r in range(height) for c in range(width) if (r, c) not in forbidden]
    if len(allowed) < area:
        return None
    mask = np.zeros((height, width), dtype=bool)
    start = allowed[int(rng.integers(len(allowed)))]
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc] and (nr, nc) not in forbidden:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True
    return mask


def _build_panel(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    reserve_corner: bool,
    force_first_corner_color: bool,
) -> np.ndarray | None:
    panel = np.zeros((height, width), dtype=int)
    forbidden = {(r, c) for r in range(2) for c in range(width - 2, width)} if reserve_corner else set()
    max_area = min(height * width - len(forbidden), max(8, (height * width) // 2 + 2))
    min_area = max(6, (height * width) // 4)
    if max_area < min_area:
        return None
    area = int(rng.integers(min_area, max_area + 1))
    mask = _random_connected_mask(rng, height, width, area, forbidden)
    if mask is None:
        return None

    main_color, accent_color = rng.choice(np.array(PALETTE, dtype=int), size=2, replace=False)
    panel[mask] = int(main_color)
    coords = np.argwhere(mask)
    accent_count = int(rng.integers(1, min(4, len(coords)) + 1))
    accent_idx = rng.choice(len(coords), size=accent_count, replace=False)
    for idx in np.atleast_1d(accent_idx):
        r, c = coords[int(idx)]
        panel[int(r), int(c)] = int(accent_color)

    if force_first_corner_color:
        corner_cells = [(r, c) for r in range(2) for c in range(width - 2, width)]
        rng.shuffle(corner_cells)
        use = int(rng.integers(1, 3))
        fill_color = int(rng.choice(np.array(PALETTE, dtype=int)))
        for r, c in corner_cells[:use]:
            panel[r, c] = fill_color

    if reserve_corner:
        assert np.all(panel[:2, -2:] == BLACK)

    if not np.any(panel != BLACK):
        return None
    if len(np.unique(panel[panel != BLACK])) < 2:
        return None
    return panel


def _glyph_patch(offset: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.zeros((2, 2), dtype=int)
    if offset == 0:
        row = int(rng.integers(0, 2))
        patch[row, :] = GREY
        return patch
    if offset == 1:
        col = int(rng.integers(0, 2))
        patch[:, col] = GREY
        return patch
    diag = int(rng.integers(0, 2))
    if diag == 0:
        patch[0, 0] = GREY
        patch[1, 1] = GREY
    else:
        patch[0, 1] = GREY
        patch[1, 0] = GREY
    return patch


def _offset_sequence(rng: np.random.Generator, count: int) -> list[int]:
    if count == 3:
        valid = [
            [0, 1, -1],
            [1, 0, -1],
            [1, -1, 0],
        ]
        return list(valid[int(rng.integers(len(valid)))])
    valid_pairs = [
        [1, -1],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    return list(valid_pairs[int(rng.integers(len(valid_pairs)))])


def _build_input_from_parts(
    panels_clean: list[np.ndarray],
    offsets: list[int],
    separators: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    H = int(panels_clean[0].shape[0])
    total_w = sum(int(p.shape[1]) for p in panels_clean) + sum(int(w) for w in separators)
    grid = np.zeros((H, total_w), dtype=int)
    c = 0
    for i, panel in enumerate(panels_clean):
        w = int(panel.shape[1])
        grid[:, c : c + w] = panel
        if i > 0:
            glyph = _glyph_patch(offsets[i - 1], rng)
            grid[:2, c + w - 2 : c + w] = np.where(glyph == GREY, GREY, grid[:2, c + w - 2 : c + w])
        c += w
        if i < len(separators):
            sw = int(separators[i])
            grid[:, c : c + sw] = MAROON
            c += sw
    return grid


def _shortcut_outputs(inp: np.ndarray) -> list[np.ndarray]:
    panels, offsets = _parse_panels(inp)
    wrongs = [
        _stitch_panels(panels, [0] * len(panels), fill=MAROON),
        _stitch_panels(panels, _row_starts_direct(offsets), fill=MAROON),
        _stitch_panels(panels, _row_starts_cumulative(offsets), fill=BLACK),
        _stitch_panels(_split_panels(inp), _row_starts_cumulative(offsets), fill=MAROON),
    ]
    return wrongs


def _input_constraints_ok(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if np.any(inp == MAROON):
        runs = _separator_runs(inp)
        if len(runs) not in (2, 3):
            return False
    panels = _split_panels(inp)
    if not (3 <= len(panels) <= 4):
        return False
    if np.any(panels[0] == GREY):
        return False
    for panel in panels[1:]:
        if int(np.count_nonzero(panel == GREY)) != 2:
            return False
        if int(np.count_nonzero(panel[:2, -2:] == GREY)) != 2:
            return False
        if int(np.count_nonzero(panel[:, :-2] == GREY)) != 0 or int(np.count_nonzero(panel[2:, -2:] == GREY)) != 0:
            return False
    return True


def _nondegenerate(inp: np.ndarray, out: np.ndarray) -> bool:
    if np.array_equal(inp, out):
        return False
    if not np.any(out == MAROON):
        return False
    if not np.any(out == BLACK):
        return False
    cleaned, offsets = _parse_panels(inp)
    if len({tuple(panel.flatten()) for panel in cleaned}) != len(cleaned):
        return False
    if len(offsets) == 3 and set(offsets) != {-1, 0, 1}:
        return False
    if len(offsets) == 2 and len(set(offsets)) < 2:
        return False
    rows = _row_starts_cumulative(offsets)
    if len(set(rows)) < 2:
        return False
    if rows == _row_starts_direct(offsets):
        return False
    for wrong in _shortcut_outputs(inp):
        if wrong.shape == out.shape and np.array_equal(wrong, out):
            return False
    return True


def _attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_count = 4 if float(rng.random()) < 0.65 else 3
    height = int(rng.integers(6, 9))
    if panel_count == 4:
        widths = [int(rng.integers(5, 7)) for _ in range(panel_count)]
    else:
        widths = [int(rng.integers(5, 8)) for _ in range(panel_count)]
    separators = [int(rng.integers(2, 4)) for _ in range(panel_count - 1)]
    if sum(widths) + sum(separators) > 30:
        return None

    offsets = _offset_sequence(rng, panel_count - 1)
    rows = _row_starts_cumulative(offsets)
    if min(rows) < 0:
        return None
    if height + max(rows) > 30:
        return None

    panels_clean: list[np.ndarray] = []
    for idx, width in enumerate(widths):
        panel = _build_panel(
            rng,
            height,
            width,
            reserve_corner=idx > 0,
            force_first_corner_color=idx == 0 and float(rng.random()) < 0.75,
        )
        if panel is None:
            return None
        panels_clean.append(panel)
    if len({tuple(p.flatten()) for p in panels_clean}) != len(panels_clean):
        return None

    inp = _build_input_from_parts(panels_clean, offsets, separators, rng)
    if not _input_constraints_ok(inp):
        return None

    expected = _stitch_panels(panels_clean, rows, fill=MAROON)
    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    if max(out.shape) > 30:
        return None
    if not _nondegenerate(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        inp = _attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid offset-stitch input")
