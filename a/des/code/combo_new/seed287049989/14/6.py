import random
from collections import deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400
GEN_COUNTER = 0


def _read_palette(row: np.ndarray) -> list[int]:
    palette: list[int] = []
    width = int(row.shape[0])
    for c in range(width):
        color = int(row[c])
        if color == BG:
            continue
        if c > 0:
            assert int(row[c - 1]) == BG, "palette pixels must be separated by black"
        if c + 1 < width:
            assert int(row[c + 1]) == BG, "palette pixels must be separated by black"
        palette.append(color)
    assert len(palette) >= 2, "palette row must contain at least two colors"
    assert len(palette) == len(set(palette)), "palette colors must be distinct"
    return palette


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each multi-color object into palette-ordered color layers, shift them right, and re-stack them.

    Input structure:
    - Black is the background.
    - The bottom row is a palette row made of distinct single-pixel colors separated by Black; their left-to-right order defines shift ranks.
    - Above the palette row are several non-overlapping 4-connected multi-color objects whose colors all appear in the palette.

    Rule:
    - Read the palette row from left to right and assign each palette color its rank k starting at 0.
    - For every non-background pixel above the palette row, move that pixel horizontally to the right by k cells, where k is the rank of its color.
    - If multiple shifted pixels land on the same output cell, keep the color with the higher palette rank; keep the palette row unchanged and clip pixels that move past the right edge.

    Key insight:
    - The bottom-row color order is the offset code: each visible color names a layer and also tells how far that layer should translate before the layers are overlaid again.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    palette = _read_palette(g[-1])
    rank_by_color = {color: rank for rank, color in enumerate(palette)}

    field_colors = set(map(int, np.unique(g[:-1]))) - {BG}
    assert field_colors <= set(palette), "every foreground color must appear in the palette"

    out = np.full((h, w), BG, dtype=int)
    out[-1] = g[-1]

    # Paint from low to high rank so later, higher-rank layers overwrite overlaps.
    for rank, color in enumerate(palette):
        rows, cols = np.where(g[:-1] == color)
        shifted_cols = cols + rank
        keep = shifted_cols < w
        out[rows[keep], shifted_cols[keep]] = color

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    assert total >= 0 and parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _positive_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    extras = _random_composition(total - parts, parts)
    return [value + 1 for value in extras]


def _make_row_intervals(n_rows: int, min_len: int, max_len: int) -> list[tuple[int, int]]:
    for _ in range(80):
        rows: list[tuple[int, int]] = []
        left = 0
        length = random.randint(min_len, max_len)
        rows.append((left, length))
        for _row in range(1, n_rows):
            new_len = random.randint(min_len, max_len)
            overlap_min = left - new_len + 1
            overlap_max = left + length - 1
            jitter_min = left - 2
            jitter_max = left + 2
            low = max(overlap_min, jitter_min)
            high = min(overlap_max, jitter_max)
            if low > high:
                low, high = overlap_min, overlap_max
            left = random.randint(low, high)
            length = new_len
            rows.append((left, length))

        min_left = min(left for left, _ in rows)
        rows = [(left - min_left, length) for left, length in rows]
        width = max(left + length for left, length in rows)
        if width <= 12:
            return rows
    raise RuntimeError("failed to build connected row intervals")


def _paint_ascending_row(length: int, ranks: list[int]) -> list[int]:
    widths = _positive_partition(length, len(ranks))
    out: list[int] = []
    for width, rank in zip(widths, ranks):
        out.extend([rank] * width)
    return out


def _paint_overlap_row(length: int, ranks: list[int]) -> list[int]:
    assert ranks == list(range(ranks[0], ranks[0] + len(ranks))), "overlap row needs consecutive ranks"
    base_rank = ranks[0]
    row = [base_rank] * length
    start = random.randint(0, length - len(ranks))
    for offset, rank in enumerate(reversed(ranks)):
        row[start + offset] = rank
    return row


def _make_object_patch(palette: list[int], require_overlap: bool) -> np.ndarray:
    palette_len = len(palette)
    color_count = random.randint(2, min(4, palette_len))
    if require_overlap:
        start = random.randint(0, palette_len - color_count)
        selected_ranks = list(range(start, start + color_count))
    else:
        selected_ranks = sorted(random.sample(range(palette_len), color_count))

    n_rows = random.randint(2, 5)
    min_len = max(3, color_count)
    max_len = min(8, color_count + 4)
    rows = _make_row_intervals(n_rows, min_len, max_len)
    width = max(left + length for left, length in rows)
    patch = np.zeros((n_rows, width), dtype=int)
    special_row = random.randrange(n_rows) if require_overlap else -1

    for r, (left, length) in enumerate(rows):
        if require_overlap and r == special_row:
            row_ranks = _paint_overlap_row(length, selected_ranks)
        elif not require_overlap and r == 0:
            row_ranks = _paint_ascending_row(length, selected_ranks)
        else:
            active_count = random.randint(2, min(color_count, length))
            if require_overlap and random.random() < 0.6:
                start = random.randint(0, color_count - active_count)
                active_ranks = selected_ranks[start : start + active_count]
            else:
                active_ranks = sorted(random.sample(selected_ranks, active_count))
            row_ranks = _paint_ascending_row(length, active_ranks)
        patch[r, left : left + length] = [palette[rank] for rank in row_ranks]

    return patch


def _transformed_positions(patch: np.ndarray, rank_by_color: dict[int, int]) -> list[tuple[int, int]]:
    rows, cols = np.where(patch != BG)
    out: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        color = int(patch[r, c])
        out.append((int(r), int(c + rank_by_color[color])))
    return out


def _has_internal_overlap(patch: np.ndarray, rank_by_color: dict[int, int]) -> bool:
    positions = _transformed_positions(patch, rank_by_color)
    return len(set(positions)) < len(positions)


def _shift_extents(patch: np.ndarray, rank_by_color: dict[int, int]) -> tuple[int, int]:
    rows, cols = np.where(patch != BG)
    max_input = max(cols.tolist())
    max_shifted = max(int(c + rank_by_color[int(patch[r, c])]) for r, c in zip(rows.tolist(), cols.tolist()))
    return int(max_input), int(max_shifted)


def _palette_positions(width: int, count: int) -> list[int]:
    min_span = 2 * count - 1
    assert width >= min_span
    extras = _random_composition(width - min_span, count + 1)
    positions: list[int] = []
    col = extras[0]
    for idx in range(count):
        positions.append(col)
        if idx < count - 1:
            col += 2 + extras[idx + 1]
    return positions


def _validate_components(grid: np.ndarray, palette: list[int], expected_count: int) -> None:
    field = grid[:-1]
    mask = field != BG
    comps = _components(mask)
    assert len(comps) == expected_count, "unexpected number of foreground objects"
    palette_set = set(palette)
    for comp in comps:
        colors = {int(field[r, c]) for r, c in comp}
        assert 2 <= len(colors) <= 4, "each object must use 2-4 colors"
        assert colors <= palette_set, "object colors must come from the palette"


def _validate_branches(
    grid: np.ndarray,
    palette: list[int],
    need_overlap: bool,
    need_clip: bool,
) -> None:
    rank_by_color = {color: rank for rank, color in enumerate(palette)}
    actual_overlap = False
    actual_clip = False
    h, w = grid.shape
    for r in range(h - 1):
        seen: set[int] = set()
        row = grid[r]
        for c in range(w):
            color = int(row[c])
            if color == BG:
                continue
            target = c + rank_by_color[color]
            if target >= w:
                actual_clip = True
                continue
            if target in seen:
                actual_overlap = True
            seen.add(target)
    assert actual_overlap == need_overlap, "overlap branch mismatch"
    assert actual_clip == need_clip, "clip branch mismatch"


def _gen_attempt() -> np.ndarray | None:
    global GEN_COUNTER
    GEN_COUNTER += 1

    mode = (GEN_COUNTER - 1) % 4
    need_overlap = mode in (1, 3)
    need_clip = mode in (2, 3)

    palette_len = 3 + ((GEN_COUNTER - 1) % 3)
    palette = random.sample(COLOR_POOL, palette_len)
    rank_by_color = {color: rank for rank, color in enumerate(palette)}

    object_count = random.randint(max(2, int(need_overlap) + int(need_clip)), 4)
    specs = [{"overlap": False, "clip": False} for _ in range(object_count)]
    if need_overlap:
        specs[0]["overlap"] = True
    if need_clip:
        clip_idx = 1 if need_overlap else 0
        specs[clip_idx]["clip"] = True
    random.shuffle(specs)

    for spec in specs:
        for _ in range(80):
            patch = _make_object_patch(palette, require_overlap=spec["overlap"])
            has_overlap = _has_internal_overlap(patch, rank_by_color)
            if has_overlap != spec["overlap"]:
                continue
            max_input, max_shifted = _shift_extents(patch, rank_by_color)
            if spec["clip"] and max_shifted <= max_input:
                continue
            spec["patch"] = patch
            spec["max_input"] = max_input
            spec["max_shifted"] = max_shifted
            break
        else:
            return None

    random.shuffle(specs)
    heights = [int(spec["patch"].shape[0]) for spec in specs]
    base_gaps = [random.randint(1, 2) for _ in range(max(0, object_count - 1))]
    top_margin = random.randint(1, 3)
    bottom_gap = random.randint(1, 3)

    min_height = top_margin + sum(heights) + sum(base_gaps) + bottom_gap + 1
    if min_height > 30:
        return None
    extra_height = random.randint(0, min(6, 30 - min_height))
    slack = _random_composition(extra_height, object_count + 2)
    top_margin += slack[0]
    bottom_gap += slack[-1]
    gaps = [gap + slack[i + 1] for i, gap in enumerate(base_gaps)]

    min_width = max(
        2 * palette_len - 1,
        max(int(spec["patch"].shape[1]) for spec in specs),
        max(int(spec["max_shifted"]) + 1 for spec in specs),
    )
    width_low = max(14, min_width)
    if width_low > 30:
        return None
    width = random.randint(width_low, 30)

    grid = np.zeros((top_margin + sum(heights) + sum(gaps) + bottom_gap + 1, width), dtype=int)
    row = top_margin
    for idx, spec in enumerate(specs):
        patch = spec["patch"]
        max_input = int(spec["max_input"])
        max_shifted = int(spec["max_shifted"])
        if spec["clip"]:
            low = max(0, width - max_shifted)
            high = width - 1 - max_input
            if low > high:
                return None
            col = random.randint(low, high)
        else:
            high = width - 1 - max_shifted
            if high < 0:
                return None
            col = random.randint(0, high)

        ph, pw = patch.shape
        block = grid[row : row + ph, col : col + pw]
        if np.any((block != BG) & (patch != BG)):
            return None
        block[patch != BG] = patch[patch != BG]
        row += ph
        if idx < len(gaps):
            row += gaps[idx]

    for pos, color in zip(_palette_positions(width, palette_len), palette):
        grid[-1, pos] = color

    _validate_components(grid, palette, object_count)
    _validate_branches(grid, palette, need_overlap=need_overlap, need_clip=need_clip)
    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
