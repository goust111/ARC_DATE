from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
BRIDGE = 9
RECT_COLORS = (1, 2, 3, 4, 6, 7)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_monochrome(grid: np.ndarray, *, ignore: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            bbox = _bbox_coords(coords)
            out.append({"color": color, "coords": coords, "bbox": bbox, "area": int(len(cells))})
    return out


def _canonical_pattern(tile: np.ndarray) -> tuple[int, ...]:
    remap: dict[int, int] = {}
    nxt = 0
    sig: list[int] = []
    for value in tile.reshape(-1):
        color = int(value)
        if color not in remap:
            remap[color] = nxt
            nxt += 1
        sig.append(remap[color])
    return tuple(sig)


def _pattern_mapping(src: np.ndarray, dst: np.ndarray) -> dict[int, int] | None:
    if src.shape != dst.shape:
        return None

    forward: dict[int, int] = {}
    backward: dict[int, int] = {}
    for s, t in zip(src.reshape(-1), dst.reshape(-1)):
        sc = int(s)
        tc = int(t)
        if sc == BG or tc == BG:
            if sc != tc:
                return None
            continue
        if sc in forward and forward[sc] != tc:
            return None
        if tc in backward and backward[tc] != sc:
            return None
        forward[sc] = tc
        backward[tc] = sc
    return forward


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(rows) == 1, "expected exactly one grey separator row"
    return int(rows[0])


def _extract_tiles(grid: np.ndarray, sep_row: int) -> list[np.ndarray]:
    top = grid[:sep_row, :]
    active_rows = np.where(np.any(top != BG, axis=1))[0]
    assert active_rows.size > 0, "expected tiles above the separator"
    band = top[int(active_rows[0]) : int(active_rows[-1]) + 1, :]

    active_cols = np.where(np.any(band != BG, axis=0))[0]
    assert active_cols.size > 0
    spans: list[tuple[int, int]] = []
    start = int(active_cols[0])
    prev = int(active_cols[0])
    for col in active_cols[1:]:
        col = int(col)
        if col != prev + 1:
            spans.append((start, prev))
            start = col
        prev = col
    spans.append((start, prev))

    assert len(spans) == 3, "expected exactly three horizontal tiles"
    widths = {c1 - c0 + 1 for c0, c1 in spans}
    assert len(widths) == 1, "expected equal tile widths"
    return [band[:, c0 : c1 + 1].copy() for c0, c1 in spans]


def _tile_mapping(grid: np.ndarray, sep_row: int) -> dict[int, int]:
    tiles = _extract_tiles(grid, sep_row)
    matches: list[tuple[int, int, dict[int, int]]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            mapping = _pattern_mapping(tiles[i], tiles[j])
            if mapping is not None and _canonical_pattern(tiles[i]) == _canonical_pattern(tiles[j]):
                matches.append((i, j, mapping))
    assert len(matches) == 1, "expected exactly one isomorphic tile pair"
    return matches[0][2]


def _extract_rectangles(grid: np.ndarray, sep_row: int) -> list[dict]:
    body = grid[sep_row + 1 :, :]
    rects = _components_monochrome(body, ignore={BG, SEP, BRIDGE})
    assert 3 <= len(rects) <= 5, "expected 3-5 rectangles"

    out: list[dict] = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        patch = body[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each foreground object must be a solid rectangle"
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(rect["area"]) == int(area), "each component must equal its bbox"
        out.append(
            {
                "color": color,
                "bbox": (int(r0 + sep_row + 1), int(c0), int(r1 + sep_row + 1), int(c1)),
            }
        )

    out.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for left, right in zip(out, out[1:]):
        assert int(left["bbox"][3]) < int(right["bbox"][1]), "rectangles must be horizontally separated"
    return out


def _mapped_color(color: int, mapping: dict[int, int]) -> int:
    return int(mapping.get(int(color), int(color)))


def _pair_geometry(rects: list[dict], i: int, j: int) -> tuple[int, int, int, int, int, int] | None:
    a = rects[i]
    b = rects[j]
    if int(a["bbox"][1]) <= int(b["bbox"][1]):
        left_idx, right_idx = i, j
    else:
        left_idx, right_idx = j, i

    left = rects[left_idx]["bbox"]
    right = rects[right_idx]["bbox"]
    gap_c0 = int(left[3]) + 1
    gap_c1 = int(right[1]) - 1
    if gap_c0 > gap_c1:
        return None

    overlap_r0 = max(int(left[0]), int(right[0]))
    overlap_r1 = min(int(left[2]), int(right[2]))
    if overlap_r0 > overlap_r1:
        return None

    return left_idx, right_idx, overlap_r0, overlap_r1, gap_c0, gap_c1


def _has_blocker(
    rects: list[dict],
    left_idx: int,
    right_idx: int,
    overlap_r0: int,
    overlap_r1: int,
    gap_c0: int,
    gap_c1: int,
) -> bool:
    for k, rect in enumerate(rects):
        if k in (left_idx, right_idx):
            continue
        r0, c0, r1, c1 = rect["bbox"]
        if c0 <= gap_c1 and c1 >= gap_c0 and r0 <= overlap_r1 and r1 >= overlap_r0:
            return True
    return False


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_row = _separator_row(g)
    mapping = _tile_mapping(g, sep_row)
    rects = _extract_rectangles(g, sep_row)

    mapped_pairs: list[dict] = []
    eligible_pairs: list[dict] = []
    original_equal_pairs: list[dict] = []

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            geom = _pair_geometry(rects, i, j)
            if geom is None:
                continue
            left_idx, right_idx, r0, r1, c0, c1 = geom
            left = rects[left_idx]
            right = rects[right_idx]
            blocker = _has_blocker(rects, left_idx, right_idx, r0, r1, c0, c1)
            same_original = int(left["color"]) == int(right["color"])
            same_mapped = _mapped_color(left["color"], mapping) == _mapped_color(right["color"], mapping)
            pair = {
                "pair": (left_idx, right_idx),
                "rows": (r0, r1),
                "cols": (c0, c1),
                "blocked": blocker,
            }
            if same_original:
                original_equal_pairs.append(pair)
            if same_mapped:
                mapped_pairs.append(pair)
                if not blocker:
                    eligible_pairs.append(pair)

    return {
        "sep_row": sep_row,
        "mapping": mapping,
        "rectangles": rects,
        "mapped_pairs": mapped_pairs,
        "eligible_pairs": eligible_pairs,
        "original_equal_pairs": original_equal_pairs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a color mapping from the matching tiles and use the mapped rectangle colors to decide which horizontal gaps become bridges.

    Input structure:
    - The background is black, and one full grey row separates an upper tile band from a lower rectangle band.
    - The tile band contains exactly three equal-size horizontal tiles; exactly two tiles share the same pixel pattern up to color renaming.
    - The rectangle band contains 3-5 horizontally separated solid monochrome rectangles whose row spans may overlap.

    Rule:
    - Find the unique isomorphic tile pair and read the ordered color mapping from the left matching tile to the right matching tile.
    - Apply that mapping only as a lookup on rectangle colors: mapped source colors change, other rectangle colors stay as they are.
    - For every rectangle pair whose mapped colors match, whose row spans overlap, whose column spans are separated, and whose gap has no third rectangle bbox overlapping the shared rows, fill the black gap cells in that shared row range with maroon.

    Key insight:
    - Tile colors do not repaint the rectangles; they redefine rectangle color equivalence, and that mapped equivalence alone decides which gaps qualify for bridging.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)

    out = g.copy()
    for pair in info["eligible_pairs"]:
        r0, r1 = pair["rows"]
        c0, c1 = pair["cols"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = BRIDGE
    return out


def _grow_region(h: int, w: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    region = {start}
    while len(region) < area:
        frontier = []
        for r, c in region:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        region.add(random.choice(frontier))
    return region


def _random_label_pattern(h: int, w: int) -> np.ndarray:
    total = h * w
    for _ in range(100):
        area = random.randint(2, total - 2)
        region = _grow_region(h, w, area)
        if region is None:
            continue
        labels = np.ones((h, w), dtype=int)
        for r, c in region:
            labels[r, c] = 0
        if len({tuple(row.tolist()) for row in labels}) == 1:
            continue
        if len({tuple(col.tolist()) for col in labels.T}) == 1:
            continue
        return labels
    raise RuntimeError("failed to generate a non-trivial tile pattern")


def _render_tile(labels: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    tile = np.empty_like(labels, dtype=int)
    tile[labels == 0] = int(colors[0])
    tile[labels == 1] = int(colors[1])
    return tile


def _oriented_pair(src: int, dst: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return int(src), int(dst)
    return int(dst), int(src)


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> tuple[int, int, int, int]:
    grid[top : top + height, left : left + width] = int(color)
    return int(top), int(left), int(top + height - 1), int(left + width - 1)


def _expected_bridge(expected: np.ndarray, rects: list[dict], open_pair: tuple[int, int]) -> None:
    left_idx, right_idx, r0, r1, c0, c1 = _pair_geometry(rects, open_pair[0], open_pair[1])
    assert left_idx is not None
    expected[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE


def _generate_case() -> np.ndarray | None:
    tile_h = random.choice((3, 4))
    tile_w = random.choice((3, 4))
    labels = _random_label_pattern(tile_h, tile_w)
    decoy_labels = _random_label_pattern(tile_h, tile_w)
    if _canonical_pattern(decoy_labels) == _canonical_pattern(labels):
        return None

    palette = list(RECT_COLORS)
    random.shuffle(palette)
    src_colors = (int(palette[0]), int(palette[1]))
    dst_colors = (int(palette[2]), int(palette[3]))
    noise_color = int(palette[4])
    decoy_colors = (int(palette[5]), int(random.choice(palette[:5])))
    mapping = {src_colors[0]: dst_colors[0], src_colors[1]: dst_colors[1]}

    blocked_slot = random.randrange(2)
    open_slot = 1 - blocked_slot
    blocked_pair = (src_colors[blocked_slot], dst_colors[blocked_slot])
    open_pair_colors = (src_colors[open_slot], dst_colors[open_slot])
    blocked_left_color, blocked_right_color = _oriented_pair(*blocked_pair)
    open_left_color, open_right_color = _oriented_pair(*open_pair_colors)

    use_noise = random.random() < 0.65
    rect_colors = [blocked_left_color, open_left_color]
    noise_index = None
    if use_noise:
        noise_index = len(rect_colors)
        rect_colors.append(noise_color)
    open_right_index = len(rect_colors)
    rect_colors.append(open_right_color)
    blocked_right_index = len(rect_colors)
    rect_colors.append(blocked_right_color)

    rect_count = len(rect_colors)
    widths = [random.randint(2, 4) for _ in range(rect_count)]
    heights = [random.randint(2, 4) for _ in range(rect_count)]
    gaps = [random.randint(1, 2) for _ in range(rect_count - 1)]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    body_w = left_margin + sum(widths) + sum(gaps) + right_margin

    tile_gap = random.randint(1, 2)
    tile_span = 3 * tile_w + 2 * tile_gap
    if max(body_w, tile_span) > 30:
        return None

    common_top = random.randint(1, 4)
    tops = [common_top] * rect_count
    if use_noise:
        noise_height = heights[noise_index]
        max_core_height = max(height for idx, height in enumerate(heights) if idx != noise_index)
        place_above = random.random() < 0.5 and common_top >= noise_height + 1
        if place_above:
            tops[noise_index] = common_top - noise_height - 1
        else:
            tops[noise_index] = common_top + max_core_height + 1

    body_h = max(top + height for top, height in zip(tops, heights)) + random.randint(1, 3)

    tile_top = random.randint(0, 1)
    sep_row = tile_top + tile_h + random.randint(0, 1)
    grid_h = sep_row + 1 + body_h
    grid_w = max(body_w, tile_span + random.randint(0, 2))
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[sep_row, :] = SEP

    pair_slots = random.choice(((0, 1), (0, 2), (1, 2)))
    source_slot, target_slot = pair_slots
    decoy_slot = ({0, 1, 2} - {source_slot, target_slot}).pop()
    tile_left = random.randint(0, grid_w - tile_span)
    tile_arrays = [None, None, None]
    tile_arrays[source_slot] = _render_tile(labels, src_colors)
    tile_arrays[target_slot] = _render_tile(labels, dst_colors)
    tile_arrays[decoy_slot] = _render_tile(decoy_labels, decoy_colors)

    for idx, tile in enumerate(tile_arrays):
        left = tile_left + idx * (tile_w + tile_gap)
        grid[tile_top : tile_top + tile_h, left : left + tile_w] = tile

    rects_for_expected: list[dict] = []
    cursor = left_margin
    for idx, (color, top, height, width) in enumerate(zip(rect_colors, tops, heights, widths)):
        abs_top = sep_row + 1 + top
        bbox = _paint_rect(grid, abs_top, cursor, height, width, color)
        rects_for_expected.append({"color": int(color), "bbox": bbox})
        cursor += width
        if idx + 1 < rect_count:
            cursor += gaps[idx]

    expected = grid.copy()
    open_pair = (1, open_right_index)
    _expected_bridge(expected, rects_for_expected, open_pair)

    info = _analyze(grid)
    if info["mapping"] != mapping:
        return None
    if len(info["mapped_pairs"]) != 2:
        return None
    if len(info["eligible_pairs"]) != 1:
        return None
    if info["original_equal_pairs"]:
        return None

    pair_sets = {frozenset(pair["pair"]) for pair in info["mapped_pairs"]}
    eligible_set = frozenset(info["eligible_pairs"][0]["pair"])
    expected_open = frozenset(open_pair)
    expected_blocked = frozenset((0, blocked_right_index))
    if pair_sets != {expected_open, expected_blocked}:
        return None
    if eligible_set != expected_open:
        return None

    if use_noise:
        open_geom = _pair_geometry(rects_for_expected, *open_pair)
        assert open_geom is not None
        _left_idx, _right_idx, open_r0, open_r1, open_c0, open_c1 = open_geom
        nr0, nc0, nr1, nc1 = rects_for_expected[noise_index]["bbox"]
        if not (nc0 <= open_c1 and nc1 >= open_c0):
            return None
        if nr0 <= open_r1 and nr1 >= open_r0:
            return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
