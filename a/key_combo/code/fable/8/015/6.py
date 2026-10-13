from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return [np.array(cells, dtype=int) for cells in parts]


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == int(mask.sum())


def _body_and_markers(grid: np.ndarray, coords: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    colors = [int(grid[r, c]) for r, c in coords]
    counts = Counter(colors)
    body_color, body_count = counts.most_common(1)[0]
    assert body_count >= 3, "sprite body must dominate its two marker pixels"
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords if int(grid[r, c]) != body_color]
    assert len(markers) == 2, "expected exactly two marker pixels per sprite"
    assert len({color for _r, _c, color in markers}) == 2, "marker colors must differ"
    for _r, _c, color in markers:
        assert counts[color] == 1, "marker colors must be singleton pixels"
    return int(body_color), markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each interior sprite so its two internal marker colors align to the matching hidden row and column ticks.

    Input structure:
    - The background is black.
    - The leftmost column and the bottom row contain isolated colored tick pixels that act as hidden row and column coordinates.
    - The interior contains several 4-connected monochrome sprites, each with exactly two differently colored marker pixels embedded in the body.

    Rule:
    - For each sprite, find the marker color whose matching tick appears on the left edge and shift the sprite so that marker onto that tick's row.
    - Find the other marker color on the bottom edge and shift the sprite so that marker onto that tick's column.
    - Apply the combined rigid translation, erase the sprite from its old location, and keep all edge ticks unchanged.

    Key insight:
    - Each marker supplies only half of the destination: one fixes the row from the left edge, the other fixes the column from the bottom edge, and only their combination determines the final placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = _get_bg(g)
    row_ticks: dict[int, int] = {}
    col_ticks: dict[int, int] = {}
    for r in range(h - 1):
        color = int(g[r, 0])
        if color != bg:
            assert color not in row_ticks, "left-edge tick colors must be unique"
            row_ticks[color] = r
    for c in range(1, w):
        color = int(g[h - 1, c])
        if color != bg:
            assert color not in col_ticks, "bottom-edge tick colors must be unique"
            col_ticks[color] = c

    interior_mask = g != bg
    interior_mask[:, 0] = False
    interior_mask[h - 1, :] = False
    parts = _components(interior_mask)
    assert parts, "expected interior sprites"

    out = g.copy()
    out[interior_mask] = bg

    for coords in parts:
        body_color, markers = _body_and_markers(g, coords)
        _ = body_color

        row_marker = None
        col_marker = None
        for marker in markers:
            _r, _c, color = marker
            on_left = color in row_ticks
            on_bottom = color in col_ticks
            assert on_left != on_bottom, "each marker color must belong to exactly one edge"
            if on_left:
                row_marker = marker
            else:
                col_marker = marker

        assert row_marker is not None and col_marker is not None
        dr = int(row_ticks[row_marker[2]] - row_marker[0])
        dc = int(col_ticks[col_marker[2]] - col_marker[1])
        moved = coords + np.array([dr, dc], dtype=int)
        assert np.all(moved[:, 0] >= 0) and np.all(moved[:, 0] < h)
        assert np.all(moved[:, 1] >= 0) and np.all(moved[:, 1] < w)

        values = np.array([int(g[r, c]) for r, c in coords], dtype=int)
        for (nr, nc), color in zip(moved, values):
            nr = int(nr)
            nc = int(nc)
            assert not (nc == 0 or nr == h - 1), "translated sprites must stay off tick edges"
            assert int(out[nr, nc]) == bg, "translated sprites must not conflict"
            out[nr, nc] = color

    return out


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c]:
                cells.append((r, c))
    return cells


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        mask = np.ones((h, w), dtype=bool)
        border = [(r, c) for r in range(h) for c in range(w) if r in {0, h - 1} or c in {0, w - 1}]
        rng.shuffle(border)
        max_remove = min(len(border) - 4, int(rng.integers(0, 4)))
        removed = 0
        for r, c in border:
            if removed >= max_remove:
                break
            trial = mask.copy()
            trial[r, c] = False
            if not trial.any():
                continue
            rows = np.where(trial.any(axis=1))[0]
            cols = np.where(trial.any(axis=0))[0]
            if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
                continue
            if not _mask_connected(trial):
                continue
            if len(_interior_cells(trial)) < 2:
                continue
            mask = trial
            removed += 1
        if len(_interior_cells(mask)) >= 2:
            return mask
    raise RuntimeError("failed to build sprite mask")


def _make_sprite(
    rng: np.random.Generator,
    body_color: int,
    marker_colors: tuple[int, int],
) -> dict:
    mask = _random_shape(rng)
    interior = _interior_cells(mask)
    marker_cells = rng.choice(len(interior), size=2, replace=False)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(body_color)
    markers: list[tuple[int, int, int]] = []
    for idx, color in zip(marker_cells, marker_colors):
        r, c = interior[int(idx)]
        patch[r, c] = int(color)
        markers.append((int(r), int(c), int(color)))
    rng.shuffle(markers)
    return {"patch": patch, "markers": markers}


def _occupied_cells(patch: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = top
    cells = set()
    for r, c in np.argwhere(patch != BG):
        cells.add((tr + int(r), tc + int(c)))
    return cells


def _candidate_tops(h: int, w: int, patch: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    return [(r, c) for r in range(0, h - 1 - ph + 1) for c in range(1, w - pw + 1)]


def _place_targets(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    sprites: list[dict],
) -> list[dict] | None:
    h, w = grid_shape
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []

    order = list(range(len(sprites)))
    rng.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        markers = list(sprite["markers"])
        rng.shuffle(markers)
        row_marker, col_marker = markers
        candidates = _candidate_tops(h, w, sprite["patch"])
        rng.shuffle(candidates)
        chosen = None
        for top in candidates:
            tr, tc = top
            row_value = tr + row_marker[0]
            col_value = tc + col_marker[1]
            if row_value in used_rows or col_value in used_cols:
                continue
            cells = _occupied_cells(sprite["patch"], top)
            if cells & occupied:
                continue
            chosen = {
                "patch": sprite["patch"],
                "top": top,
                "row_marker": row_marker,
                "col_marker": col_marker,
                "row_tick": (row_value, row_marker[2]),
                "col_tick": (col_value, col_marker[2]),
            }
            break
        if chosen is None:
            return None
        used_rows.add(int(chosen["row_tick"][0]))
        used_cols.add(int(chosen["col_tick"][0]))
        occupied |= _occupied_cells(sprite["patch"], chosen["top"])
        placed.append(chosen)

    placed.sort(key=lambda item: (item["top"][0], item["top"][1]))
    return placed


def _place_inputs(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    targets: list[dict],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied: set[tuple[int, int]] = set()
    tops: list[tuple[int, int]] = []
    for target in targets:
        patch = target["patch"]
        candidates = _candidate_tops(h, w, patch)
        rng.shuffle(candidates)
        chosen = None
        for top in candidates:
            if top == target["top"]:
                continue
            cells = _occupied_cells(patch, top)
            if cells & occupied:
                continue
            chosen = top
            break
        if chosen is None:
            return None
        tops.append(chosen)
        occupied |= _occupied_cells(patch, chosen)
    return tops


def _stamp(grid: np.ndarray, patch: np.ndarray, top: tuple[int, int]) -> None:
    tr, tc = top
    for r, c in np.argwhere(patch != BG):
        grid[tr + int(r), tc + int(c)] = int(patch[r, c])


def _expected_output(grid_shape: tuple[int, int], targets: list[dict], extras: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    h, w = grid_shape
    out = np.full((h, w), BG, dtype=int)
    extra_row, extra_col = extras
    for row, color in (target["row_tick"] for target in targets):
        out[int(row), 0] = int(color)
    for col, color in (target["col_tick"] for target in targets):
        out[h - 1, int(col)] = int(color)
    out[int(extra_row[0]), 0] = int(extra_row[1])
    out[h - 1, int(extra_col[0])] = int(extra_col[1])
    for target in targets:
        _stamp(out, target["patch"], target["top"])
    return out


def _attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_sprites = int(rng.integers(2, 4))
    colors = PALETTE[:]
    rng.shuffle(colors)

    marker_color_count = 2 * n_sprites + 2
    marker_colors = colors[:marker_color_count]
    body_palette = colors[marker_color_count:]
    if not body_palette:
        return None

    body_choices = list(rng.choice(body_palette, size=n_sprites, replace=True))
    sprites = []
    marker_cursor = 0
    for body_color in body_choices:
        markers = (int(marker_colors[marker_cursor]), int(marker_colors[marker_cursor + 1]))
        marker_cursor += 2
        sprites.append(_make_sprite(rng, int(body_color), markers))

    max_h = max(int(sprite["patch"].shape[0]) for sprite in sprites)
    max_w = max(int(sprite["patch"].shape[1]) for sprite in sprites)
    h = int(rng.integers(max(12, max_h + 5), 25))
    w = int(rng.integers(max(12, max_w + 6), 25))

    targets = _place_targets(rng, (h, w), sprites)
    if targets is None:
        return None

    used_rows = {int(target["row_tick"][0]) for target in targets}
    used_cols = {int(target["col_tick"][0]) for target in targets}
    free_rows = [r for r in range(h - 1) if r not in used_rows]
    free_cols = [c for c in range(1, w) if c not in used_cols]
    if not free_rows or not free_cols:
        return None
    extra_row = (int(rng.choice(free_rows)), int(marker_colors[-2]))
    extra_col = (int(rng.choice(free_cols)), int(marker_colors[-1]))

    inputs = _place_inputs(rng, (h, w), targets)
    if inputs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for target, top in zip(targets, inputs):
        _stamp(grid, target["patch"], top)
    for row, color in (target["row_tick"] for target in targets):
        grid[int(row), 0] = int(color)
    for col, color in (target["col_tick"] for target in targets):
        grid[h - 1, int(col)] = int(color)
    grid[int(extra_row[0]), 0] = int(extra_row[1])
    grid[h - 1, int(extra_col[0])] = int(extra_col[1])

    expected = _expected_output((h, w), targets, (extra_row, extra_col))
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid dual-marker alignment input")
