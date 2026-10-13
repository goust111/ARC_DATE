from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
TILE_COLORS = [1, 2, 3, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Group:
    tile_r0: int
    tile_c0: int
    tile: np.ndarray
    bar_row: int
    ruler_len: int

    @property
    def tile_h(self) -> int:
        return int(self.tile.shape[0])

    @property
    def tile_w(self) -> int:
        return int(self.tile.shape[1])

    @property
    def bar_c0(self) -> int:
        return int(self.tile_c0 + self.tile_w)

    @property
    def output_c1(self) -> int:
        return int(self.bar_c0 + self.tile_w * self.ruler_len - 1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    pts: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return pts


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                pts = _component_from(mask, (r, c))
                for pr, pc in pts:
                    seen[pr, pc] = True
                comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_groups(grid: np.ndarray) -> list[Group]:
    g = np.asarray(grid, dtype=int)
    grey_comps = _components(g == GREY)
    assert grey_comps, "expected at least one grey ruler"
    groups: list[Group] = []

    color_mask = (g != BLACK) & (g != GREY)
    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        assert r0 == r1, "each grey component must be a horizontal ruler"
        assert len(comp) == c1 - c0 + 1, "ruler must be contiguous"
        assert c0 > 0, "ruler must have a tile on its left"
        assert color_mask[r0, c0 - 1], "ruler left end must touch the tile"

        tile_pts = _component_from(color_mask, (r0, c0 - 1))
        tr0, tc0, tr1, tc1 = _bbox(tile_pts)
        assert tc1 == c0 - 1, "tile must end exactly one column left of the ruler"
        assert tr0 <= r0 <= tr1, "ruler row must pass through the tile span"
        tile_h = tr1 - tr0 + 1
        tile_w = tc1 - tc0 + 1
        assert len(tile_pts) == tile_h * tile_w, "tile must be a solid rectangle"

        tile = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
        groups.append(Group(tr0, tc0, tile, r0, c1 - c0 + 1))

    groups.sort(key=lambda grp: (grp.tile_r0, grp.tile_c0))
    return groups


def _apply_transform(grid: np.ndarray, groups: list[Group], *, period_mode: str = "ruler", fill_mode: str = "full") -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for grp in groups:
        start_c = grp.bar_c0
        row_slice = slice(grp.tile_r0, grp.tile_r0 + grp.tile_h)
        if fill_mode == "full":
            fill_w = grp.tile_w * grp.ruler_len
            tiled = np.tile(grp.tile, (1, grp.ruler_len))
        elif fill_mode == "bar_only":
            fill_w = grp.ruler_len
            reps = (fill_w + grp.tile_w - 1) // grp.tile_w
            tiled = np.tile(grp.tile, (1, reps))[:, :fill_w]
        else:
            raise ValueError(fill_mode)

        out[row_slice, start_c : start_c + fill_w] = tiled

        period = grp.ruler_len if period_mode == "ruler" else grp.tile_w
        for offset in range(0, fill_w, period):
            out[row_slice, start_c + offset] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat each attached tile by the ruler length, then recolor every ruler-period column in the tiled area.

    Input structure:
    - A black background contains 2-3 independent structures.
    - Each structure has a horizontal grey ruler and a colored rectangular tile touching the ruler's left end.
    - The tile is 2-3 rows by 2-3 columns and uses only non-black, non-grey colors.

    Rule:
    - For each structure, let L be the grey ruler length.
    - Copy the whole tile to the right exactly L times, starting where the ruler begins; this covers the ruler and may extend past it.
    - Within only that newly tiled rectangle, recolor every column whose local index is 0 mod L to yellow.
    - Leave the original tile in place and process all structures independently.

    Key insight:
    - The same ruler length encodes two parameters at once: how many tile copies to make and the column period for recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    groups = _extract_groups(g)
    return _apply_transform(g, groups, period_mode="ruler", fill_mode="full")


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _make_tile(height: int, width: int, palette: list[int]) -> np.ndarray:
    area = height * width
    values = [random.choice(palette) for _ in range(area)]
    for idx, color in zip(random.sample(range(area), len(palette)), palette):
        values[idx] = color
    tile = np.array(values, dtype=int).reshape(height, width)
    if width > 1 and len({tuple(tile[:, c]) for c in range(width)}) == 1:
        tile[:, -1] = tile[::-1, -1]
        if np.array_equal(tile[:, -1], tile[:, 0]):
            tile[0, -1] = palette[-1]
    if len(set(map(int, tile.ravel()))) < 2:
        tile[0, 0] = palette[0]
        tile[-1, -1] = palette[-1]
    return tile


def _build_groups(shape: tuple[int, int]) -> list[Group] | None:
    h, w = shape
    n_groups = random.choice([2, 3])
    ruler_lengths = random.sample([3, 4, 5, 6], n_groups)
    tile_sizes = [(random.choice([2, 3]), random.choice([2, 3])) for _ in range(n_groups)]

    if not any(length % tile_w != 0 for length, (_, tile_w) in zip(ruler_lengths, tile_sizes)):
        return None

    needed_rows = sum(tile_h for tile_h, _ in tile_sizes) + 2 * (n_groups - 1)
    if needed_rows + 2 > h:
        return None

    available = h - needed_rows - 2
    gaps = _random_partition(available, n_groups + 1)
    row_cursor = 1 + gaps[0]

    colors = random.sample(TILE_COLORS, 2 * n_groups)
    groups: list[Group] = []
    used_ruler_lengths: set[int] = set()
    for idx, ((tile_h, tile_w), ruler_len) in enumerate(zip(tile_sizes, ruler_lengths)):
        total_w = tile_w * (ruler_len + 1)
        if total_w + 2 > w:
            return None
        tile_c0 = random.randint(1, w - total_w - 1)
        bar_row = row_cursor + random.randrange(tile_h)
        palette = colors[2 * idx : 2 * idx + 2]
        tile = _make_tile(tile_h, tile_w, palette)
        groups.append(Group(row_cursor, tile_c0, tile, bar_row, ruler_len))
        used_ruler_lengths.add(ruler_len)
        row_cursor += tile_h + 2 + gaps[idx + 1]

    if len(used_ruler_lengths) < 2:
        return None
    return groups


def _render_input(shape: tuple[int, int], groups: list[Group]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for grp in groups:
        r0, c0 = grp.tile_r0, grp.tile_c0
        tile_h, tile_w = grp.tile_h, grp.tile_w
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = grp.tile
        grid[grp.bar_row, grp.bar_c0 : grp.bar_c0 + grp.ruler_len] = GREY
    return grid


def _semantic_checks(inp: np.ndarray, groups: list[Group], expected: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False

    parsed = _extract_groups(inp)
    if len(parsed) != len(groups):
        return False

    wrong_period = _apply_transform(inp, parsed, period_mode="tile", fill_mode="full")
    if np.array_equal(wrong_period, expected):
        return False

    wrong_fill = _apply_transform(inp, parsed, period_mode="ruler", fill_mode="bar_only")
    if np.array_equal(wrong_fill, expected):
        return False

    for grp in groups:
        region = expected[grp.tile_r0 : grp.tile_r0 + grp.tile_h, grp.bar_c0 : grp.bar_c0 + grp.tile_w * grp.ruler_len]
        yellow_cols = np.where(np.all(region == YELLOW, axis=0))[0]
        if yellow_cols.tolist() != list(range(0, grp.tile_w * grp.ruler_len, grp.ruler_len)):
            return False
        if np.any(expected[grp.tile_r0 : grp.tile_r0 + grp.tile_h, grp.tile_c0 : grp.tile_c0 + grp.tile_w] == YELLOW):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(15, 24), random.randint(22, 30))
    groups = _build_groups(shape)
    if groups is None:
        return None

    inp = _render_input(shape, groups)
    expected = _apply_transform(inp, groups, period_mode="ruler", fill_mode="full")
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    if not _semantic_checks(inp, groups, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
