from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


RING5: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 4),
    (2, 0),
    (2, 4),
    (3, 0),
    (3, 4),
    (4, 1),
    (4, 2),
    (4, 3),
)
BG = 0
PALETTE = (1, 2, 3, 4, 6)
TEMPLATE_SIZE = 5
MIN_VISIBLE = 4

GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
SPRITE_COUNTS = (4, 5)
MAX_GENERATION_ATTEMPTS = 600
MAX_NOISE_ATTEMPTS = 120


def _cells_for_offset(shape: tuple[int, int], offset: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0 = offset
    cells = set()
    for dr, dc in RING5:
        r = r0 + dr
        c = c0 + dc
        if 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
    return cells


def _fit_color(
    grid: np.ndarray,
    color: int,
) -> dict[str, object]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
    assert coords, f"missing color {color}"

    hits_by_offset: dict[tuple[int, int], int] = defaultdict(int)
    for r, c in coords:
        for dr, dc in RING5:
            hits_by_offset[(r - dr, c - dc)] += 1

    best_hits = max(hits_by_offset.values())
    best_offsets = [offset for offset, hits in hits_by_offset.items() if hits == best_hits]
    assert best_hits >= MIN_VISIBLE, f"color {color} has too few on-template hits: {best_hits}"
    assert len(best_offsets) == 1, f"color {color} does not have a unique template fit"

    offset = best_offsets[0]
    expected = _cells_for_offset(grid.shape, offset)
    visible = coords & expected
    assert len(visible) == best_hits
    return {
        "color": int(color),
        "offset": offset,
        "expected": expected,
        "visible": visible,
        "visible_count": int(len(visible)),
        "noise_count": int(len(coords) - len(visible)),
    }


def _recover_scene(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert 4 <= len(colors) <= 5, f"expected 4-5 sprite colors, got {colors}"

    sprites = {color: _fit_color(g, color) for color in colors}
    below_to_above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for color, sprite in sprites.items():
        visible = sprite["visible"]
        expected = sprite["expected"]
        for r, c in expected - visible:
            cover = int(g[r, c])
            assert cover in sprites and cover != color, "every hidden in-bounds ring pixel must be covered by another ring"
            assert (r, c) in sprites[cover]["visible"], "occluding pixels must be visible ring pixels, not noise"
            if cover not in below_to_above[color]:
                below_to_above[color].add(cover)
                indegree[cover] += 1

    layer_order: list[int] = []
    remaining = set(colors)
    while remaining:
        bottoms = [color for color in remaining if indegree[color] == 0]
        assert len(bottoms) == 1, "occlusion relations must define a unique bottom-to-top order"
        color = bottoms[0]
        layer_order.append(color)
        remaining.remove(color)
        for above in below_to_above[color]:
            indegree[above] -= 1

    layer_rank = {color: i for i, color in enumerate(layer_order)}
    visible_order = sorted(
        colors,
        key=lambda color: (
            int(sprites[color]["visible_count"]),
            int(layer_rank[color]),
        ),
    )
    return sprites, layer_order, visible_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the occluded 5x5 ring sprites, sort them by visible size, and cyclically rotate their colors before restacking.

    Input structure:
    - The background is black.
    - Four or five non-black colors each belong to one translated copy of the same
      12-pixel 5x5 hollow ring template; some same-color singleton noise pixels may
      appear away from every valid template location.
    - The rings overlap, so lower rings lose some in-bounds template pixels to
      higher visible ring pixels, and a ring may be clipped by the grid boundary.

    Rule:
    - For each color, ignore off-template noise and find the unique translated 5x5
      ring whose in-bounds template cells explain the visible fragment.
    - Use missing in-bounds template cells to infer occlusion: if ring A expects a
      cell but ring B is visible there, then B is above A; topologically sort these
      relations from bottom to top.
    - Sort the recovered rings by visible template-pixel count ascending, breaking
      ties by the lower ring first, then rotate the original colors one step right
      along that size order.
    - Redraw every complete recovered ring on a blank canvas in the original
      bottom-to-top layer order, using the rotated colors.

    Key insight:
    - Two independent orders must be tracked at once: occlusion order tells where
      each complete ring is drawn, while visible-size order tells which color each
      reconstructed ring receives.
    """
    g = np.asarray(grid, dtype=int)
    sprites, layer_order, visible_order = _recover_scene(g)

    rotated_color = {
        visible_order[i]: int(visible_order[i - 1])
        for i in range(len(visible_order))
    }

    out = np.full_like(g, BG)
    for color in layer_order:
        paint = rotated_color[color]
        for r, c in sprites[color]["expected"]:
            out[r, c] = paint
    return out


def _render_input(shape: tuple[int, int], bottom_to_top: list[int], offsets: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color in bottom_to_top:
        for r, c in _cells_for_offset(shape, offsets[color]):
            grid[r, c] = color
    return grid


def _all_offsets(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    offsets = []
    for r0 in range(-1, h - TEMPLATE_SIZE + 2):
        for c0 in range(-1, w - TEMPLATE_SIZE + 2):
            if len(_cells_for_offset(shape, (r0, c0))) >= 8:
                offsets.append((r0, c0))
    return offsets


def _pick_top_offset(shape: tuple[int, int], offsets: list[tuple[int, int]]) -> tuple[int, int]:
    clipped = [offset for offset in offsets if len(_cells_for_offset(shape, offset)) < len(RING5)]
    full = [offset for offset in offsets if len(_cells_for_offset(shape, offset)) == len(RING5)]
    if clipped and random.random() < 0.65:
        return random.choice(clipped)
    return random.choice(full if full else offsets)


def _sample_offsets(shape: tuple[int, int], n: int) -> list[tuple[int, int]] | None:
    offsets = _all_offsets(shape)
    chosen = [_pick_top_offset(shape, offsets)]
    higher_cells = [_cells_for_offset(shape, chosen[0])]

    for _ in range(1, n):
        covered_by_higher = set().union(*higher_cells)
        visible_prev = higher_cells[-1] - (set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set())
        candidates: list[tuple[tuple[int, int], int, int]] = []
        for offset in offsets:
            if offset in chosen:
                continue
            cells = _cells_for_offset(shape, offset)
            visible = cells - covered_by_higher
            overlap_now = len(cells & visible_prev)
            if len(visible) < MIN_VISIBLE or overlap_now == 0:
                continue
            overlap_total = len(cells & covered_by_higher)
            if overlap_total > 7:
                continue
            candidates.append((offset, len(visible), overlap_total))
        if not candidates:
            return None

        # Bias toward lighter overlaps so visible counts are not forced to follow layer order.
        candidates.sort(key=lambda item: (item[2], -item[1]))
        top_band = candidates[: max(6, len(candidates) // 3)]
        offset = random.choice(top_band)[0]
        chosen.append(offset)
        higher_cells.append(_cells_for_offset(shape, offset))
    return chosen


def _visible_profile(
    shape: tuple[int, int],
    bottom_to_top: list[int],
    offsets: dict[int, tuple[int, int]],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    above_union: set[tuple[int, int]] = set()
    for color in reversed(bottom_to_top):
        cells = _cells_for_offset(shape, offsets[color])
        counts[color] = len(cells - above_union)
        above_union |= cells
    return counts


def _valid_visible_order(layer_order: list[int], visible_order: list[int], counts: dict[int, int], want_tie: bool) -> bool:
    if visible_order == layer_order:
        return False
    values = list(counts.values())
    has_tie = len(set(values)) != len(values)
    if want_tie and not has_tie:
        return False
    if not want_tie and has_tie:
        return False
    if has_tie:
        freq = defaultdict(int)
        for v in values:
            freq[v] += 1
        if max(freq.values()) > 2:
            return False
    return True


def _noise_candidates(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or (r, c) in forbidden:
                continue
            ok = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) in forbidden:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _add_noise(
    grid: np.ndarray,
    sprites: dict[int, dict[str, object]],
    layer_order: list[int],
    visible_order: list[int],
) -> np.ndarray | None:
    forbidden = set().union(*(sprite["expected"] for sprite in sprites.values()))
    candidates = _noise_candidates(grid, forbidden)
    if not candidates:
        return None

    for _ in range(MAX_NOISE_ATTEMPTS):
        noisy = grid.copy()
        random.shuffle(candidates)
        k = random.randint(1, 2)
        picked = candidates[:k]
        if len(picked) < k:
            continue
        for r, c in picked:
            noisy[r, c] = random.choice(layer_order)
        try:
            test_sprites, test_layer, test_visible = _recover_scene(noisy)
        except AssertionError:
            continue
        same_offsets = all(test_sprites[color]["offset"] == sprites[color]["offset"] for color in layer_order)
        same_counts = all(test_sprites[color]["visible_count"] == sprites[color]["visible_count"] for color in layer_order)
        if same_offsets and same_counts and test_layer == layer_order and test_visible == visible_order:
            return noisy
    return None


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    count = random.choice(SPRITE_COUNTS)
    colors = random.sample(PALETTE, count)
    bottom_to_top = random.sample(colors, len(colors))
    top_to_bottom = list(reversed(bottom_to_top))
    want_tie = random.random() < 0.35

    top_offsets = _sample_offsets(shape, count)
    if top_offsets is None:
        return None
    offsets = {color: offset for color, offset in zip(top_to_bottom, top_offsets)}
    raw = _render_input(shape, bottom_to_top, offsets)

    try:
        sprites, layer_order, visible_order = _recover_scene(raw)
    except AssertionError:
        return None

    if layer_order != bottom_to_top:
        return None
    if any(sprites[color]["offset"] != offsets[color] for color in bottom_to_top):
        return None

    counts = _visible_profile(shape, bottom_to_top, offsets)
    if any(int(sprites[color]["visible_count"]) != counts[color] for color in bottom_to_top):
        return None
    if not _valid_visible_order(layer_order, visible_order, counts, want_tie):
        return None

    noisy = _add_noise(raw, sprites, layer_order, visible_order)
    if noisy is None:
        return None

    out = main(noisy)
    if np.array_equal(noisy, out):
        return None
    if len(np.unique(out)) != len(bottom_to_top) + 1:
        return None
    return noisy


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
