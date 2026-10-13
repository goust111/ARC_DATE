import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != bg and np.all(row == row[0]):
            candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected one full-width non-background separator line"
    return candidates[0]


def _parse_small_flag(sub: np.ndarray, gray: int, bg: int) -> int | None:
    h, w = sub.shape
    if (h, w) not in {(3, 4), (4, 4)}:
        return None
    if not np.all(sub[:, 0] == gray):
        return None
    body_h = h - 1
    color_block = sub[:body_h, 1:]
    if color_block.shape != (body_h, 3):
        return None
    color = int(color_block[0, 0])
    if color in {bg, gray} or not np.all(color_block == color):
        return None
    if not np.all(sub[body_h, 1:] == bg):
        return None
    if np.any(sub[:body_h, 1:] == gray):
        return None
    return color


def _extract_mapping(grid: np.ndarray, sep_row: int, gray: int, bg: int) -> dict[int, int]:
    top = grid[:sep_row]
    comps = _components(top != bg)
    flags: list[tuple[int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        color = _parse_small_flag(top[r0 : r1 + 1, c0 : c1 + 1], gray, bg)
        assert color is not None, "unexpected component in reference zone"
        flags.append((r0, c0, color))

    assert flags, "expected at least one reference flag"

    grouped: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, color in flags:
        grouped.setdefault(r0, []).append((c0, color))

    mapping: dict[int, int] = {}
    for r0 in sorted(grouped):
        row_flags = sorted(grouped[r0])
        assert len(row_flags) % 2 == 0, "reference flags must form left-right pairs"
        for i in range(0, len(row_flags), 2):
            src = row_flags[i][1]
            dst = row_flags[i + 1][1]
            assert src not in mapping, "reference source colors must be unique"
            mapping[src] = dst

    assert mapping, "reference mapping must be non-empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace template flag stripe colors using the color-map encoded by paired sample flags above a separator.

    Input structure:
    - The background is black, and a single full-width gray row splits the grid into a reference zone and a template zone.
    - Above the separator are small flags: each one is a gray vertical pole with a solid 2x3 or 3x3 color block immediately to its right, and they appear in left-right pairs.
    - Below the separator are one or more larger gray-outlined flags whose interiors are horizontal color stripes using colors from the left flags in the reference pairs.

    Rule:
    - Find the unique gray separator row.
    - Read every paired small flag above it from left to right within each row: the left flag's block color is the source and the right flag's block color is the destination, forming a color replacement table.
    - Apply that replacement table only below the separator, changing stripe colors inside the large template flags while leaving the gray outlines, poles, separator, and reference zone unchanged.

    Key insight:
    - The top flags are not targets to transform; they are a visual lookup table that tells how to recolor matching stripe colors in the lower template flags.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_row, gray = _find_separator(g, bg)
    mapping = _extract_mapping(g, sep_row, gray, bg)

    out = g.copy()
    lower_in = g[sep_row + 1 :]
    lower_out = out[sep_row + 1 :]
    for src, dst in mapping.items():
        lower_out[lower_in == src] = dst
    return out


def _composition(rng: random.Random, total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _draw_small_flag(grid: np.ndarray, top: int, left: int, pole_h: int, body_h: int, color: int) -> None:
    grid[top : top + pole_h, left] = GRAY
    grid[top : top + body_h, left + 1 : left + 4] = color


def _draw_large_flag(
    grid: np.ndarray,
    top: int,
    left: int,
    interior_h: int,
    interior_w: int,
    stripe_heights: list[int],
    stripe_colors: list[int],
) -> None:
    h = interior_h + 2
    w = interior_w + 2
    grid[top, left : left + w] = GRAY
    grid[top + h - 1, left : left + w] = GRAY
    grid[top : top + h, left] = GRAY
    grid[top : top + h, left + w - 1] = GRAY

    rr = top + 1
    for stripe_h, color in zip(stripe_heights, stripe_colors):
        grid[rr : rr + stripe_h, left + 1 : left + w - 1] = color
        rr += stripe_h


def _top_pair_rows(rng: random.Random, pair_count: int) -> list[int]:
    if pair_count == 2:
        return rng.choice([[2], [1, 1]])
    return rng.choice([[2, 1], [1, 2]])


def _top_layout(
    rng: random.Random, pair_count: int, color_pairs: list[tuple[int, int]]
) -> tuple[list[tuple[int, int, int]], int, int, int, int]:
    pole_h = rng.choice([3, 4])
    body_h = pole_h - 1
    flag_w = 4
    inner_gap = 1
    outer_gap = rng.randint(2, 3)
    row_gap = rng.randint(1, 2)
    row_pairs = _top_pair_rows(rng, pair_count)

    placements: list[tuple[int, int, int]] = []
    y = 0
    pair_idx = 0
    content_w = 0
    for pairs_in_row in row_pairs:
        x = 0
        for pi in range(pairs_in_row):
            src, dst = color_pairs[pair_idx]
            pair_idx += 1
            placements.append((y, x, src))
            x += flag_w + inner_gap
            placements.append((y, x, dst))
            x += flag_w
            if pi != pairs_in_row - 1:
                x += outer_gap
        content_w = max(content_w, x)
        y += pole_h + row_gap
    content_h = y - row_gap
    return placements, content_h, content_w, pole_h, body_h


def _template_row_counts(rng: random.Random, flag_count: int) -> list[int]:
    if flag_count == 1:
        return [1]
    if flag_count == 2:
        return rng.choice([[2], [1, 1]])
    return rng.choice([[2, 1], [1, 2]])


def _make_template_flags(
    rng: random.Random, source_colors: list[int], flag_count: int
) -> tuple[list[dict[str, object]], list[list[int]]]:
    flags: list[dict[str, object]] = []
    stripe_counts: list[int] = []
    for _ in range(flag_count):
        stripe_count = rng.randint(2, min(4, max(2, len(source_colors) + 1)))
        interior_h = stripe_count + rng.randint(0, 2)
        interior_w = rng.randint(4, 7)
        stripe_heights = _composition(rng, interior_h, stripe_count)
        flags.append(
            {
                "interior_h": interior_h,
                "interior_w": interior_w,
                "stripe_heights": stripe_heights,
                "h": interior_h + 2,
                "w": interior_w + 2,
            }
        )
        stripe_counts.append(stripe_count)

    total_stripes = sum(stripe_counts)
    color_bag = list(source_colors) + [rng.choice(source_colors) for _ in range(total_stripes - len(source_colors))]
    for _ in range(200):
        rng.shuffle(color_bag)
        groups: list[list[int]] = []
        idx = 0
        ok = True
        for stripe_count in stripe_counts:
            group = color_bag[idx : idx + stripe_count]
            idx += stripe_count
            if any(group[i] == group[i - 1] for i in range(1, len(group))):
                ok = False
                break
            if len(set(group)) < min(2, stripe_count):
                ok = False
                break
            groups.append(group)
        if ok:
            return flags, groups

    groups = []
    offset = 0
    for stripe_count in stripe_counts:
        group = [source_colors[(offset + i) % len(source_colors)] for i in range(stripe_count)]
        if stripe_count >= 2 and len(set(group)) == 1:
            group[1] = source_colors[(offset + 1) % len(source_colors)]
        groups.append(group)
        offset += 1
    return flags, groups


def _template_layout(
    rng: random.Random, flags: list[dict[str, object]]
) -> tuple[list[tuple[int, int]], int, int]:
    row_counts = _template_row_counts(rng, len(flags))
    row_gap = rng.randint(2, 3)
    col_gap = rng.randint(2, 3)

    placements: list[tuple[int, int]] = [(-1, -1)] * len(flags)
    idx = 0
    y = 0
    content_w = 0
    for count in row_counts:
        row_flags = flags[idx : idx + count]
        row_h = max(int(flag["h"]) for flag in row_flags)
        x = 0
        for local_i, flag in enumerate(row_flags):
            placements[idx + local_i] = (y, x)
            x += int(flag["w"])
            if local_i != count - 1:
                x += col_gap
        content_w = max(content_w, x)
        y += row_h
        idx += count
        if idx != len(flags):
            y += row_gap
    content_h = y
    return placements, content_h, content_w


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([2, 3])
    palette = rng.sample(COLOR_POOL, 2 * pair_count)
    source_colors = palette[:pair_count]
    target_colors = palette[pair_count:]
    color_pairs = list(zip(source_colors, target_colors))

    top_flags, top_h, top_w, pole_h, body_h = _top_layout(rng, pair_count, color_pairs)

    flag_count = rng.randint(1, 3)
    template_flags, template_colors = _make_template_flags(rng, source_colors, flag_count)
    template_positions, template_h, template_w = _template_layout(rng, template_flags)

    top_pad_top = rng.randint(1, 2)
    top_pad_bottom = rng.randint(1, 2)
    bottom_pad_top = rng.randint(1, 2)
    bottom_pad_bottom = rng.randint(1, 2)
    min_h = top_pad_top + top_h + top_pad_bottom + 1 + bottom_pad_top + template_h + bottom_pad_bottom
    min_w = max(top_w, template_w) + 2
    if min_h > 30 or min_w > 30:
        return None

    grid_h = min_h + rng.randint(0, min(30 - min_h, 4))
    grid_w = min_w + rng.randint(0, min(30 - min_w, 4))
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    top_y = top_pad_top
    bottom_y = top_pad_top + top_h + top_pad_bottom + 1 + bottom_pad_top
    sep_row = bottom_y - bottom_pad_top - 1
    top_x = rng.randint(1, grid_w - top_w - 1)
    bottom_x = rng.randint(1, grid_w - template_w - 1)

    grid[sep_row, :] = GRAY

    for rel_r, rel_c, color in top_flags:
        _draw_small_flag(grid, top_y + rel_r, top_x + rel_c, pole_h, body_h, color)

    for flag, colors, (rel_r, rel_c) in zip(template_flags, template_colors, template_positions):
        _draw_large_flag(
            grid,
            bottom_y + rel_r,
            bottom_x + rel_c,
            int(flag["interior_h"]),
            int(flag["interior_w"]),
            list(flag["stripe_heights"]),
            colors,
        )

    expected = grid.copy()
    lower_in = grid[sep_row + 1 :]
    lower_out = expected[sep_row + 1 :]
    mapping = dict(color_pairs)
    for src, dst in mapping.items():
        lower_out[lower_in == src] = dst

    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None

    bg_count = int(np.sum(grid == BG))
    if bg_count <= max(int(np.sum(grid == color)) for color in set(np.unique(grid)) if color != BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flag-mapping ARC input")
