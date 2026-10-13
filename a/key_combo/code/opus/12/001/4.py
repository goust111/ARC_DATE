from __future__ import annotations

import random
from typing import Iterable

import numpy as np


SPRITE = 7
SEPARATOR = 5
BACKGROUND_PALETTE = (1, 2, 3, 4, 6, 8, 9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    if not coords:
        return False
    stack = [next(iter(coords))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == SEPARATOR)]
    assert len(sep_rows) == 3, "expected exactly three separator rows"
    panel_h = sep_rows[0]
    assert panel_h > 0, "panels must be non-empty"
    expected = [panel_h, 2 * panel_h + 1, 3 * panel_h + 2]
    assert sep_rows == expected, "separator rows must evenly split four panels"
    assert g.shape[0] == 4 * panel_h + 3
    panels = []
    start = 0
    for sep in sep_rows:
        panels.append(g[start:sep].copy())
        start = sep + 1
    panels.append(g[start:].copy())
    assert len({panel.shape for panel in panels}) == 1, "all panels must share a shape"
    return panels


def _residue_map(
    panels: Iterable[np.ndarray], period_r: int, period_c: int
) -> dict[tuple[int, int], int] | None:
    residue_to_color: dict[tuple[int, int], int] = {}
    for panel in panels:
        h, w = panel.shape
        for r in range(h):
            for c in range(w):
                color = int(panel[r, c])
                if color == SPRITE:
                    continue
                key = (r % period_r, c % period_c)
                old = residue_to_color.get(key)
                if old is None:
                    residue_to_color[key] = color
                elif old != color:
                    return None
    if len(residue_to_color) != period_r * period_c:
        return None
    if len(set(residue_to_color.values())) < 2:
        return None
    return residue_to_color


def _detect_period(panels: list[np.ndarray]) -> tuple[int, int, dict[tuple[int, int], int]]:
    h, w = panels[0].shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for period_r in range(1, h + 1):
        for period_c in range(1, w + 1):
            residue_to_color = _residue_map(panels, period_r, period_c)
            if residue_to_color is not None:
                candidates.append((period_r, period_c, residue_to_color))
    assert candidates, "no valid translational period found"
    candidates.sort(key=lambda item: (item[0] * item[1], item[0] + item[1], item[0], item[1]))
    return candidates[0]


def _clean_background(
    shape: tuple[int, int], period_r: int, period_c: int, residue_to_color: dict[tuple[int, int], int]
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = residue_to_color[(r % period_r, c % period_c)]
    return out


def _extract_sprite(panel: np.ndarray, clean_bg: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    mask = panel != clean_bg
    assert mask.any(), "each frame must contain a sprite"
    assert np.all(panel[mask] == SPRITE), "only the sprite may differ from the clean background"
    assert _is_connected(mask), "sprite must be 4-connected"
    r0, c0, _r1, _c1 = _bbox(mask)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in np.argwhere(mask)))
    return (r0, c0), rel


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the periodic panel background, isolate the moving orange sprite, and extrapolate its fifth frame.

    Input structure:
    - The input is a vertical strip of four equal panels separated by solid grey full-width rows.
    - Every panel shares one axis-aligned periodic background tile that uses non-grey, non-orange colors.
    - A small 4-connected orange sprite overwrites some background cells inside each panel.

    Rule:
    - Detect the smallest horizontal/vertical background periods from all visible non-orange cells.
    - Reconstruct the clean repeating background by assigning each hidden orange cell the unique color of its period orbit.
    - Compare each observed panel with that clean background to recover the sprite shape and its top-left position, verify the constant per-frame translation, then place the sprite one more step ahead on the clean background.

    Key insight:
    - The animation cannot be completed from the last frame alone because the sprite hides needed background pixels; translational periodicity reveals the clean panel, and only then does the sprite trajectory become easy to extrapolate.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    period_r, period_c, residue_to_color = _detect_period(panels)
    clean_bg = _clean_background(panels[0].shape, period_r, period_c, residue_to_color)

    positions: list[tuple[int, int]] = []
    shapes: list[tuple[tuple[int, int], ...]] = []
    for panel in panels:
        pos, rel = _extract_sprite(panel, clean_bg)
        positions.append(pos)
        shapes.append(rel)

    assert all(shape == shapes[0] for shape in shapes[1:]), "sprite shape must stay constant"
    dr = positions[1][0] - positions[0][0]
    dc = positions[1][1] - positions[0][1]
    assert (dr, dc) != (0, 0), "sprite must move"
    for i in range(1, 3):
        assert positions[i + 1][0] - positions[i][0] == dr
        assert positions[i + 1][1] - positions[i][1] == dc

    target_r = positions[3][0] + dr
    target_c = positions[3][1] + dc
    out = clean_bg.copy()
    h, w = out.shape
    for rr, cc in shapes[0]:
        r = target_r + rr
        c = target_c + cc
        assert 0 <= r < h and 0 <= c < w, "fifth frame must stay inside the panel"
        out[r, c] = SPRITE
    return out


_BASE_HORIZONTAL_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 2)),
)


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _tile_from_shape(rel: tuple[tuple[int, int], ...]) -> np.ndarray:
    r1 = max(r for r, _ in rel) + 1
    c1 = max(c for _, c in rel) + 1
    mask = np.zeros((r1, c1), dtype=bool)
    for r, c in rel:
        mask[r, c] = True
    return mask


def _shape_span(rel: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in rel]
    cols = [c for _, c in rel]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _fundamental_periods(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape

    def smallest_row_period() -> int:
        for p in range(1, h + 1):
            good = True
            for r in range(h):
                if not np.array_equal(tile[r], tile[r % p]):
                    good = False
                    break
            if good:
                return p
        raise AssertionError("unreachable")

    def smallest_col_period() -> int:
        for p in range(1, w + 1):
            good = True
            for c in range(w):
                if not np.array_equal(tile[:, c], tile[:, c % p]):
                    good = False
                    break
            if good:
                return p
        raise AssertionError("unreachable")

    return smallest_row_period(), smallest_col_period()


def _make_tile(period_r: int, period_c: int) -> np.ndarray:
    palette_size = random.randint(2, min(3, len(BACKGROUND_PALETTE)))
    colors = random.sample(BACKGROUND_PALETTE, palette_size)
    for _ in range(200):
        tile = np.array(
            [[random.choice(colors) for _ in range(period_c)] for _ in range(period_r)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 2:
            continue
        if _fundamental_periods(tile) != (period_r, period_c):
            continue
        return tile
    raise RuntimeError("failed to construct a non-degenerate background tile")


def _fill_with_tile(height: int, width: int, tile: np.ndarray) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    th, tw = tile.shape
    for r in range(height):
        for c in range(width):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _place_sprite(bg: np.ndarray, rel: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    out = bg.copy()
    for r, c in rel:
        out[top + r, left + c] = SPRITE
    return out


def _orbit_visibility_ok(
    panel: np.ndarray, period_r: int, period_c: int
) -> bool:
    for rr in range(period_r):
        for cc in range(period_c):
            visible = False
            for r in range(rr, panel.shape[0], period_r):
                for c in range(cc, panel.shape[1], period_c):
                    if int(panel[r, c]) != SPRITE:
                        visible = True
                        break
                if visible:
                    break
            if not visible:
                return False
    return True


def _overlap_all_frames(
    rel: tuple[tuple[int, int], ...], start_left: int, step: int
) -> set[tuple[int, int]]:
    placed = []
    for t in range(4):
        placed.append({(r, c + start_left + t * step) for r, c in rel})
    common = placed[0]
    for item in placed[1:]:
        common = common & item
    return common


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    panel_h = random.choice((5, 6))
    panel_w = random.randint(10, 14)
    period_r, period_c = random.choice(((2, 2), (2, 3), (3, 2)))
    tile = _make_tile(period_r, period_c)
    clean_bg = _fill_with_tile(panel_h, panel_w, tile)

    rel = random.choice(_BASE_HORIZONTAL_SHAPES)
    sprite_h, sprite_w = _shape_span(rel)
    direction = random.choice((-1, 1))
    row = random.randint(0, panel_h - sprite_h)
    if direction == 1:
        left = random.randint(0, panel_w - sprite_w - 4)
    else:
        left = random.randint(4, panel_w - sprite_w)

    panels = []
    for t in range(4):
        panels.append(_place_sprite(clean_bg, rel, row, left + t * direction))

    if not all(_orbit_visibility_ok(panel, period_r, period_c) for panel in panels):
        raise ValueError("a residue class was fully occluded in one panel")
    if not _overlap_all_frames(rel, left, direction):
        raise ValueError("expected at least one absolute cell to stay occluded across all frames")

    expected = _place_sprite(clean_bg, rel, row, left + 4 * direction)
    rows = []
    for i, panel in enumerate(panels):
        rows.append(panel)
        if i != 3:
            rows.append(np.full((1, panel_w), SEPARATOR, dtype=int))
    stacked = np.vstack(rows)
    assert max(stacked.shape) <= 30
    assert max(expected.shape) <= 30
    return stacked, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, expected = _build_case()
        except ValueError:
            continue
        out = main(inp)
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
