from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_library(
    grid: np.ndarray, bg: int
) -> tuple[int, int, int, list[np.ndarray]]:
    """Find the bottom template strip using its three full-height separator columns."""
    height, width = grid.shape
    best: tuple[tuple[int, int], int, int, int, list[np.ndarray]] | None = None

    for sep_color in (int(v) for v in np.unique(grid) if int(v) != bg):
        groups: dict[int, list[int]] = {}
        for col in range(width):
            if int(grid[height - 1, col]) != sep_color:
                continue
            top = height - 1
            while top > 0 and int(grid[top - 1, col]) == sep_color:
                top -= 1
            groups.setdefault(int(top), []).append(int(col))

        for top, cols in groups.items():
            if len(cols) < 3:
                continue
            cols = sorted(cols)
            colset = set(cols)
            block_h = height - int(top)
            if block_h < 2:
                continue

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    left_sep = int(cols[i])
                    mid_sep = int(cols[j])
                    delta = mid_sep - left_sep
                    if delta < 2:
                        continue
                    right_sep = mid_sep + delta
                    if right_sep not in colset:
                        continue

                    tile_w = delta - 1
                    library_left = left_sep - tile_w
                    library_right = right_sep + tile_w + 1
                    if library_left < 0 or library_right > width:
                        continue

                    tiles: list[np.ndarray] = []
                    ok = True
                    for idx in range(4):
                        c0 = library_left + idx * (tile_w + 1)
                        block = grid[top:height, c0 : c0 + tile_w]
                        if block.shape != (block_h, tile_w):
                            ok = False
                            break
                        if np.any(block == sep_color):
                            ok = False
                            break
                        if not np.any(block != bg):
                            ok = False
                            break
                        tiles.append(block.copy())

                    if not ok:
                        continue

                    if not all(
                        np.all(
                            grid[top:height, library_left + tile_w + k * (tile_w + 1)]
                            == sep_color
                        )
                        for k in range(3)
                    ):
                        continue

                    score = (block_h, tile_w)
                    candidate = (score, int(top), int(sep_color), int(library_left), tiles)
                    if best is None or score > best[0]:
                        best = candidate

    if best is None:
        raise AssertionError("failed to detect the template library")

    _score, top, sep_color, library_left, tiles = best
    return top, sep_color, library_left, tiles


def _detect_towers(
    grid: np.ndarray, bg: int, search_bottom: int
) -> list[tuple[int, int, int, int]]:
    """Return (cap_row, col, body_height, cap_color) for each tower."""
    height, width = grid.shape
    assert 0 < search_bottom <= height

    towers: list[tuple[int, int, int, int]] = []
    for col in range(width):
        row = 0
        while row < search_bottom - 1:
            cap_color = int(grid[row, col])
            if cap_color == bg:
                row += 1
                continue
            if row > 0 and int(grid[row - 1, col]) != bg:
                row += 1
                continue

            body_color = int(grid[row + 1, col])
            if body_color == bg or body_color == cap_color:
                row += 1
                continue

            end = row + 1
            while end < search_bottom and int(grid[end, col]) == body_color:
                end += 1
            body_height = end - (row + 1)
            if not 1 <= body_height <= 4:
                row = end
                continue
            if end < search_bottom and int(grid[end, col]) != bg:
                row = end
                continue

            isolated = True
            for rr in range(row, end):
                if col > 0 and int(grid[rr, col - 1]) != bg:
                    isolated = False
                    break
                if col + 1 < width and int(grid[rr, col + 1]) != bg:
                    isolated = False
                    break
            if isolated:
                towers.append((int(row), int(col), int(body_height), int(cap_color)))
                break
            row = end

    if not towers:
        raise AssertionError("failed to detect any towers")
    return towers


def _stamp_recolored_template(
    out: np.ndarray,
    template: np.ndarray,
    bg: int,
    color: int,
    cap_row: int,
    center_col: int,
) -> None:
    tile_h, tile_w = template.shape
    top = int(cap_row) - tile_h
    left = int(center_col) - tile_w // 2
    mask = template != int(bg)

    for rr, cc in np.argwhere(mask):
        dst_r = top + int(rr)
        dst_c = left + int(cc)
        if 0 <= dst_r < out.shape[0] and 0 <= dst_c < out.shape[1]:
            out[dst_r, dst_c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a bottom-library template by tower height and recolor it with the tower cap color.

    Input structure:
    - Background fills most of the grid.
    - Towers are width-1 vertical bodies of height 1..4 with a differently colored cap.
    - The bottom strip contains four same-size pattern tiles separated by grey columns.

    Rule:
    - A tower with body height H selects the H-th tile from the left.
    - Recolor all non-background pixels in that tile to the tower cap color.
    - Place the recolored tile directly above the cap, aligned by its horizontal center.
    - Keep the towers and the bottom library unchanged; crop anything above the top edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    library_top, _sep_color, _library_left, tiles = _detect_library(g, bg)
    towers = _detect_towers(g, bg, library_top)

    out = g.copy()
    for cap_row, col, body_height, cap_color in towers:
        template = tiles[body_height - 1]
        _stamp_recolored_template(out, template, bg, cap_color, cap_row, col)
    return out


def _mask_from_rows(*rows: str) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


PATTERN_POOLS: dict[int, tuple[np.ndarray, ...]] = {
    3: (
        _mask_from_rows(
            "#..",
            "#..",
            "###",
        ),
        _mask_from_rows(
            "###",
            ".#.",
            ".#.",
        ),
        _mask_from_rows(
            ".#.",
            "###",
            ".#.",
        ),
        _mask_from_rows(
            ".##",
            "##.",
            "#..",
        ),
        _mask_from_rows(
            "#.#",
            "#.#",
            "###",
        ),
        _mask_from_rows(
            "##.",
            "#..",
            "###",
        ),
    ),
    5: (
        _mask_from_rows(
            "#....",
            "#....",
            "#....",
            "#....",
            "#####",
        ),
        _mask_from_rows(
            "#####",
            "..#..",
            "..#..",
            "..#..",
            "..#..",
        ),
        _mask_from_rows(
            "..#..",
            "..#..",
            "#####",
            "..#..",
            "..#..",
        ),
        _mask_from_rows(
            "#...#",
            "#...#",
            "#...#",
            "#...#",
            "#####",
        ),
        _mask_from_rows(
            "....#",
            "...##",
            "..##.",
            ".##..",
            "##...",
        ),
        _mask_from_rows(
            "..#..",
            ".###.",
            "#####",
            "..#..",
            "..#..",
        ),
    ),
}

SEPARATOR_COLOR = 5
ALL_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 400


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for rot in range(4):
        turned = np.rot90(mask, rot)
        for variant in (turned, np.fliplr(turned)):
            key = variant.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _sample_templates(size: int) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in PATTERN_POOLS[size]:
        for variant in _unique_transforms(base):
            key = variant.tobytes()
            if key not in seen:
                seen.add(key)
                pool.append(variant)
    random.shuffle(pool)
    return [pool[i].copy() for i in range(4)]


def _sample_columns(width: int, tile_w: int, count: int) -> list[int] | None:
    half = tile_w // 2
    min_gap = tile_w + 2
    candidates = list(range(half, width - half))
    random.shuffle(candidates)

    chosen: list[int] = []
    for col in candidates:
        if all(abs(col - prev) >= min_gap for prev in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_expected(
    inp: np.ndarray,
    bg: int,
    templates: list[np.ndarray],
    towers: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for cap_row, col, body_height, cap_color in towers:
        _stamp_recolored_template(
            out,
            templates[body_height - 1],
            bg,
            cap_color,
            cap_row,
            col,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.choice((3, 5))
    templates = _sample_templates(tile_size)
    clip_branch = random.choice((True, False))

    max_towers = 5 if tile_size == 3 else 3
    tower_count = random.randint(2, max_towers)
    heights = [random.randint(1, 4) for _ in range(tower_count)]

    grid_h = random.randint(2 * tile_size + 7, 2 * tile_size + 13)
    library_top = grid_h - tile_size
    library_w = 4 * tile_size + 3
    tower_w_need = tile_size + (tower_count - 1) * (tile_size + 2)
    grid_w = random.randint(max(library_w + 2, tower_w_need), max(library_w + 2, tower_w_need) + 8)

    bg_choices = [c for c in ALL_COLORS if c != SEPARATOR_COLOR]
    bg = random.choice(bg_choices)
    non_bg = [c for c in ALL_COLORS if c not in (bg, SEPARATOR_COLOR)]
    if len(non_bg) < 6:
        return None

    template_colors = random.sample(non_bg, 4)
    tower_columns = _sample_columns(grid_w, tile_size, tower_count)
    if tower_columns is None:
        return None

    forced_clip = random.randrange(tower_count) if clip_branch else None
    towers: list[tuple[int, int, int, int]] = []
    for idx, (col, body_height) in enumerate(zip(tower_columns, heights)):
        cap_max = library_top - body_height - 2
        if cap_max < 1:
            return None
        if forced_clip is not None and idx == forced_clip:
            clip_hi = min(tile_size - 1, cap_max)
            if clip_hi < 1:
                return None
            cap_row = random.randint(1, clip_hi)
        else:
            if cap_max < tile_size:
                return None
            cap_row = random.randint(tile_size, cap_max)

        body_color, cap_color = random.sample(non_bg, 2)
        towers.append((int(cap_row), int(col), int(body_height), int(cap_color)))

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    library_left = random.randint(0, grid_w - library_w)
    for idx, (mask, color) in enumerate(zip(templates, template_colors)):
        c0 = library_left + idx * (tile_size + 1)
        patch = inp[library_top:grid_h, c0 : c0 + tile_size]
        patch[mask] = int(color)
        if idx < 3:
            inp[library_top:grid_h, c0 + tile_size] = SEPARATOR_COLOR

    for cap_row, col, body_height, cap_color in towers:
        body_color_choices = [c for c in non_bg if c != cap_color]
        body_color = random.choice(body_color_choices)
        inp[cap_row, col] = cap_color
        inp[cap_row + 1 : cap_row + 1 + body_height, col] = int(body_color)

    expected = _build_expected(inp, bg, templates, towers)
    if clip_branch and not any(cap_row < tile_size for cap_row, _, _, _ in towers):
        return None
    if not clip_branch and any(cap_row < tile_size for cap_row, _, _, _ in towers):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
