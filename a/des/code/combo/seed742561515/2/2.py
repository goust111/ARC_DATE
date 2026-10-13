import random

import numpy as np


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    if values.size == 0:
        raise AssertionError("expected a non-empty grid")
    return int(np.bincount(values, minlength=10).argmax())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0 = int(pts[:, 0].min())
            c0 = int(pts[:, 1].min())
            r1 = int(pts[:, 0].max())
            c1 = int(pts[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 3x3 receiver according to the height rank of its signal tower.

    Input structure
    - The background is black.
    - The top row contains a left-to-right palette of single pixels.
    - Each tower is a bottom-aligned vertical 1-pixel-wide bar.
    - Each tower has one same-colored 3x3 receiver directly above it, separated by blank rows.

    Rule
    - Sort towers by height from shortest to tallest.
    - The palette row gives the replacement colors for ranks 1, 2, 3, ...
    - Recolor only the receivers; keep the towers and palette unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    bg = _get_bg(g)

    palette_cols = [int(c) for c in np.where(g[0] != bg)[0]]
    assert len(palette_cols) >= 2, "expected at least two palette entries"
    palette_colors = [int(g[0, c]) for c in palette_cols]
    assert len(set(palette_colors)) == len(palette_colors), "palette colors must be distinct"

    components = _find_components(g, bg)
    towers = []
    receivers = []

    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])
        hh = _bbox_h(comp["bbox"])
        ww = _bbox_w(comp["bbox"])

        if area == 1 and r0 == 0:
            assert c0 == c1
            continue
        if ww == 1 and area == hh and r1 == h - 1:
            towers.append(comp)
            continue
        if hh == 3 and ww == 3 and area == 9:
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)
            receivers.append(comp)
            continue
        raise AssertionError(f"unexpected component shape at {comp['bbox']}")

    assert len(towers) == len(receivers) == len(palette_colors), "palette, tower, receiver counts must match"

    tower_by_color = {}
    for tower in towers:
        color = int(tower["color"])
        assert color not in tower_by_color, "tower colors must be unique"
        tower_by_color[color] = tower

    receiver_by_color = {}
    for receiver in receivers:
        color = int(receiver["color"])
        assert color not in receiver_by_color, "receiver colors must be unique"
        receiver_by_color[color] = receiver

    assert set(tower_by_color) == set(receiver_by_color), "each tower must have one same-colored receiver"

    heights = []
    for color, tower in tower_by_color.items():
        receiver = receiver_by_color[color]
        tr0, tc0, _tr1, _tc1 = tower["bbox"]
        rr0, rc0, rr1, rc1 = receiver["bbox"]
        assert rc0 + 1 == tc0 == rc1 - 1, "receiver must be centered above the tower"
        assert rr1 < tr0 - 1, "receiver must be separated from the tower by at least one blank row"
        heights.append(_bbox_h(tower["bbox"]))

    assert len(set(heights)) == len(heights), "tower heights must be distinct"

    ordered_towers = sorted(towers, key=lambda comp: _bbox_h(comp["bbox"]))
    out = g.copy()
    for rank, tower in enumerate(ordered_towers):
        color = int(tower["color"])
        target_color = palette_colors[rank]
        r0, c0, r1, c1 = receiver_by_color[color]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = target_color

    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = list(range(1, 10))

PAIR_COUNT_RANGE = (2, 4)
GRID_HEIGHT_RANGE = (12, 20)
WIDTH_PADDING_RANGE = (3, 10)

TOWER_MIN_HEIGHT = 2
TOWER_MIN_GAP = 1
RECEIVER_SIZE = 3
RECEIVER_TOP_MIN = 2
TOWER_COL_MIN = 1
TOWER_MIN_COL_GAP = 4

PALETTE_MIN_COL_GAP = 1
PALETTE_ALLOW_OVERLAP_PROB = 0.55

MAX_GENERATION_ATTEMPTS = 300


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int]:
    span = hi - lo + 1
    required = count + (count - 1) * (min_gap - 1)
    if span < required:
        raise ValueError("not enough room for spaced positions")

    compressed_hi = hi - (count - 1) * (min_gap - 1)
    bases = sorted(random.sample(range(lo, compressed_hi + 1), count))
    return [base + i * (min_gap - 1) for i, base in enumerate(bases)]


def _render_scene(
    shape: tuple[int, int],
    palette_cols: list[int],
    palette_colors: list[int],
    pairs: list[dict],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for col, color in zip(palette_cols, palette_colors):
        grid[0, int(col)] = int(color)

    for pair in pairs:
        col = int(pair["col"])
        tower_color = int(pair["tower_color"])
        tower_height = int(pair["height"])
        receiver_top = int(pair["receiver_top"])
        grid[h - tower_height : h, col] = tower_color
        grid[receiver_top : receiver_top + RECEIVER_SIZE, col - 1 : col + 2] = tower_color

    return grid


def _expected_output(grid: np.ndarray, palette_colors: list[int], pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    ordered_pairs = sorted(pairs, key=lambda pair: int(pair["height"]))
    for rank, pair in enumerate(ordered_pairs):
        receiver_top = int(pair["receiver_top"])
        col = int(pair["col"])
        out[receiver_top : receiver_top + RECEIVER_SIZE, col - 1 : col + 2] = int(palette_colors[rank])
    return out


def _choose_palette_colors(pair_count: int, tower_colors: list[int]) -> list[int]:
    if random.random() < PALETTE_ALLOW_OVERLAP_PROB:
        return random.sample(COLOR_POOL, pair_count)

    remaining = [color for color in COLOR_POOL if color not in tower_colors]
    return random.sample(remaining, pair_count)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    min_width = max(10, 4 * pair_count - 1)
    width = random.randint(min_width + WIDTH_PADDING_RANGE[0], min_width + WIDTH_PADDING_RANGE[1])
    height = random.randint(max(GRID_HEIGHT_RANGE[0], pair_count + 8), GRID_HEIGHT_RANGE[1])

    height_choices = list(range(TOWER_MIN_HEIGHT, height - 5))
    if len(height_choices) < pair_count:
        return None

    try:
        tower_cols = _sample_spaced_positions(
            TOWER_COL_MIN,
            width - 2,
            pair_count,
            TOWER_MIN_COL_GAP,
        )
        palette_cols = _sample_spaced_positions(0, width - 1, pair_count, PALETTE_MIN_COL_GAP)
    except ValueError:
        return None

    tower_colors = random.sample(COLOR_POOL, pair_count)
    palette_colors = _choose_palette_colors(pair_count, tower_colors)
    tower_heights = random.sample(height_choices, pair_count)

    pairs = []
    for col, tower_color, tower_height in zip(tower_cols, tower_colors, tower_heights):
        tower_top = height - tower_height
        max_gap = tower_top - RECEIVER_TOP_MIN - RECEIVER_SIZE
        if max_gap < TOWER_MIN_GAP:
            return None
        gap = random.randint(TOWER_MIN_GAP, max_gap)
        receiver_top = tower_top - gap - RECEIVER_SIZE
        pairs.append(
            {
                "col": int(col),
                "tower_color": int(tower_color),
                "height": int(tower_height),
                "receiver_top": int(receiver_top),
            }
        )

    grid = _render_scene((height, width), palette_cols, palette_colors, pairs)
    expected = _expected_output(grid, palette_colors, pairs)

    original_by_height = [int(pair["tower_color"]) for pair in sorted(pairs, key=lambda pair: int(pair["height"]))]
    if original_by_height == palette_colors:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
