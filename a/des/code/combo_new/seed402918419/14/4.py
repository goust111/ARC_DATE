from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GEN_SEPARATOR = 8
GEN_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]

GEN_ROW_CHOICES = (2, 3, 4)
GEN_COL_CHOICES = (2, 3, 4)
GEN_TILE_MIN = 5
GEN_TILE_MAX = 8
GEN_TEMPLATE_MAX = 4
GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 80
GEN_PARTIAL_SAME_COLOR_PROB = 0.5


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _separator_layout(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    row_hits: dict[int, int] = {}
    col_hits: dict[int, int] = {}
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            row_hits[int(grid[r, 0])] = row_hits.get(int(grid[r, 0]), 0) + 1
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            col_hits[int(grid[0, c])] = col_hits.get(int(grid[0, c]), 0) + 1

    candidates = [
        color
        for color in set(row_hits) | set(col_hits)
        if row_hits.get(color, 0) > 0 and col_hits.get(color, 0) > 0
    ]
    assert candidates, "expected a separator color with full rows and columns"
    sep = max(candidates, key=lambda color: (row_hits.get(color, 0) + col_hits.get(color, 0), color))
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    assert sep_rows and sep_cols, "expected separator rows and columns"
    return int(sep), sep_rows, sep_cols


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted(separators) + [length]:
        if idx - prev > 1:
            spans.append((prev + 1, idx - 1))
        prev = idx
    assert spans, "expected at least one tile span"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """Restore each incomplete ring tile by copying the template shape indexed by its hint color.

    Input structure:
    - Black is the background.
    - One foreground color forms complete separator rows and separator columns,
      partitioning the canvas into equal rectangular tiles.
    - Every tile contains one thin ring-like outline; some tiles also contain a
      single isolated hint pixel whose color matches a complete ring in another tile.

    Rule:
    - Split the grid into tiles using the separator lines.
    - Treat tiles without an isolated hint pixel as complete templates and record
      each template mask by its ring color.
    - For each tile with a hint pixel of color C, remove that hint pixel, look up
      the complete template whose color is C, and add the missing template cells in
      color C while keeping the tile's existing outline pixels unchanged.
    - Preserve the separator lines and all untouched pixels.

    Key insight:
    - The hint pixel is a cross-tile lookup key: its color identifies which complete
      ring supplies the missing geometry, while the incomplete tile already reveals
      the correct placement of that geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep, sep_rows, sep_cols = _separator_layout(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "tiles must be equal-sized"

    templates: dict[int, np.ndarray] = {}
    missing_tiles = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            fg_mask = (tile != BACKGROUND) & (tile != sep)
            comps = _components(fg_mask)
            assert comps, "each tile must contain foreground pixels"

            singletons = [comp for comp in comps if len(comp) == 1]
            if singletons:
                assert len(singletons) == 1, "incomplete tile should have exactly one hint pixel"
                hint = tuple(map(int, singletons[0][0]))
                ring_mask = fg_mask.copy()
                ring_mask[hint] = False
                assert ring_mask.any(), "incomplete tile must still contain an outline"
                missing_tiles.append((r0, r1, c0, c1, hint, int(tile[hint]), ring_mask))
                continue

            ring_colors = [int(color) for color in np.unique(tile[fg_mask])]
            assert len(ring_colors) == 1, "template tile must contain exactly one ring color"
            color = ring_colors[0]
            assert color not in templates, "template ring colors must be unique"
            templates[color] = fg_mask.copy()

    assert templates, "expected at least one complete template tile"
    assert missing_tiles, "expected at least one incomplete tile"

    out = g.copy()
    for r0, r1, c0, c1, hint, hint_color, ring_mask in missing_tiles:
        assert hint_color in templates, "hint color must match a complete template"
        template = templates[hint_color]
        assert template.shape == ring_mask.shape
        assert np.all(~ring_mask | template), "incomplete outline must be a subset of its template"

        tile = out[r0 : r1 + 1, c0 : c1 + 1]
        tile[hint] = BACKGROUND
        fill_mask = template & (tile == BACKGROUND)
        tile[fill_mask] = hint_color

    return out


def _rot_coord(coord: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    if k % 4 == 0:
        return r, c
    if k % 4 == 1:
        return w - 1 - c, r
    if k % 4 == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _rot_groups(groups: list[list[tuple[int, int]]], shape: tuple[int, int], k: int) -> list[list[tuple[int, int]]]:
    return [[_rot_coord(coord, shape, k) for coord in group] for group in groups]


def _rot_points(points: list[tuple[int, int]], shape: tuple[int, int], k: int) -> list[tuple[int, int]]:
    return [_rot_coord(point, shape, k) for point in points]


def _rect_loop(local_h: int, local_w: int) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[tuple[int, int]]]:
    mask = np.zeros((local_h, local_w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    segments = [
        [(0, c) for c in range(local_w)],
        [(local_h - 1, c) for c in range(local_w)],
        [(r, 0) for r in range(local_h)],
        [(r, local_w - 1) for r in range(local_h)],
    ]
    corners = [(0, 0), (0, local_w - 1), (local_h - 1, 0), (local_h - 1, local_w - 1)]
    return mask, segments, corners


def _notched_loop(
    local_h: int, local_w: int, cut_row: int, depth: int
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[tuple[int, int]]]:
    inner_c = local_w - 1 - depth
    mask = np.zeros((local_h, local_w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    mask[: cut_row + 1, local_w - 1] = True
    mask[cut_row, inner_c:] = True
    mask[cut_row:, inner_c] = True
    mask[local_h - 1, : inner_c + 1] = True
    segments = [
        [(0, c) for c in range(local_w)],
        [(r, local_w - 1) for r in range(cut_row + 1)],
        [(cut_row, c) for c in range(inner_c, local_w)],
        [(r, inner_c) for r in range(cut_row, local_h)],
        [(local_h - 1, c) for c in range(inner_c + 1)],
        [(r, 0) for r in range(local_h)],
    ]
    corners = [
        (0, 0),
        (0, local_w - 1),
        (cut_row, local_w - 1),
        (cut_row, inner_c),
        (local_h - 1, inner_c),
        (local_h - 1, 0),
    ]
    return mask, segments, corners


def _place_shape(
    tile_h: int, tile_w: int
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        families = ["rect"]
        if tile_h >= 5 and tile_w >= 5:
            families.append("notch")
        family = random.choice(families)

        if family == "rect":
            local_h = random.randint(4, tile_h)
            local_w = random.randint(4, tile_w)
            mask, segments, corners = _rect_loop(local_h, local_w)
        else:
            local_h = random.randint(5, tile_h)
            local_w = random.randint(5, tile_w)
            cut_row = random.randint(1, local_h - 2)
            depth = random.randint(2, local_w - 3)
            mask, segments, corners = _notched_loop(local_h, local_w, cut_row, depth)

        k = random.randrange(4)
        rot_mask = np.rot90(mask, k)
        rot_shape = mask.shape
        rot_segments = _rot_groups(segments, rot_shape, k)
        rot_corners = _rot_points(corners, rot_shape, k)

        h2, w2 = rot_mask.shape
        if h2 > tile_h or w2 > tile_w:
            continue
        r_off = random.randint(0, tile_h - h2)
        c_off = random.randint(0, tile_w - w2)

        full_mask = np.zeros((tile_h, tile_w), dtype=bool)
        full_mask[r_off : r_off + h2, c_off : c_off + w2] = rot_mask
        full_segments = [[(r_off + r, c_off + c) for r, c in group] for group in rot_segments]
        full_corners = [(r_off + r, c_off + c) for r, c in rot_corners]

        if len(_components(full_mask)) != 1:
            continue
        return full_mask, full_segments, full_corners
    return None


def _hint_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if all(not mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
                candidates.append((r, c))
    return candidates


def _tile_canvas(tile_h: int, tile_w: int) -> np.ndarray:
    return np.full((tile_h, tile_w), BACKGROUND, dtype=int)


def _make_complete_tile(mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    inp = _tile_canvas(*mask.shape)
    inp[mask] = color
    return inp, inp.copy()


def _make_incomplete_tile(
    template_mask: np.ndarray,
    segments: list[list[tuple[int, int]]],
    corners: list[tuple[int, int]],
    template_color: int,
    partial_color: int,
    missing_kind: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    partial_mask = template_mask.copy()
    if missing_kind == "side":
        side_options = [segment for segment in segments if len(segment) >= 2]
        segment = random.choice(side_options)
        for r, c in segment:
            partial_mask[r, c] = False
    else:
        corner = random.choice(corners)
        partial_mask[corner] = False

    if not partial_mask.any() or len(_components(partial_mask)) != 1:
        return None

    hint_options = _hint_candidates(partial_mask)
    if not hint_options:
        return None
    hint = random.choice(hint_options)

    inp = _tile_canvas(*template_mask.shape)
    inp[partial_mask] = partial_color
    inp[hint] = template_color

    out = _tile_canvas(*template_mask.shape)
    out[partial_mask] = partial_color
    fill_mask = template_mask & ~partial_mask
    out[fill_mask] = template_color
    return inp, out


def _grid_size_choices(tile_count: int) -> list[int]:
    return [
        size
        for size in range(GEN_TILE_MIN, GEN_TILE_MAX + 1)
        if tile_count * size + (tile_count - 1) <= 30
    ]


def _balanced_missing_kinds(count: int) -> list[str]:
    kinds = []
    extra_side = random.choice([True, False])
    for idx in range(count):
        if idx < count // 2:
            kinds.append("side")
        elif idx < 2 * (count // 2):
            kinds.append("corner")
        else:
            kinds.append("side" if extra_side else "corner")
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.choice(GEN_ROW_CHOICES)
    tile_cols = random.choice(GEN_COL_CHOICES)
    tile_h = random.choice(_grid_size_choices(tile_rows))
    tile_w = random.choice(_grid_size_choices(tile_cols))
    tile_count = tile_rows * tile_cols

    max_templates = min(GEN_TEMPLATE_MAX, tile_count - 1, len(GEN_COLOR_POOL) - 1)
    if max_templates < 2:
        return None
    template_count = random.randint(2, max_templates)
    missing_count = tile_count - template_count
    if missing_count < 1:
        return None

    colors = random.sample(GEN_COLOR_POOL, len(GEN_COLOR_POOL))
    template_colors = colors[:template_count]
    partial_palette = [color for color in GEN_COLOR_POOL if color not in template_colors]
    if not partial_palette:
        return None

    template_specs = []
    for color in template_colors:
        placed = _place_shape(tile_h, tile_w)
        if placed is None:
            return None
        mask, segments, corners = placed
        template_specs.append(
            {"mask": mask, "segments": segments, "corners": corners, "color": color}
        )

    refs = template_colors[: min(template_count, missing_count)]
    while len(refs) < missing_count:
        refs.append(random.choice(template_colors))
    random.shuffle(refs)
    missing_kinds = _balanced_missing_kinds(missing_count)

    tile_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for spec in template_specs:
        tile_pairs.append(_make_complete_tile(spec["mask"], int(spec["color"])))

    for idx, ref_color in enumerate(refs):
        spec = next(item for item in template_specs if int(item["color"]) == int(ref_color))
        same_color = random.random() < GEN_PARTIAL_SAME_COLOR_PROB
        partial_color = int(ref_color) if same_color else int(random.choice(partial_palette))
        pair = _make_incomplete_tile(
            spec["mask"],
            spec["segments"],
            spec["corners"],
            int(ref_color),
            partial_color,
            missing_kinds[idx],
        )
        if pair is None:
            return None
        tile_pairs.append(pair)

    random.shuffle(tile_pairs)

    height = tile_rows * tile_h + (tile_rows - 1)
    width = tile_cols * tile_w + (tile_cols - 1)
    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)
    inp[tile_h :: tile_h + 1, :] = GEN_SEPARATOR
    inp[:, tile_w :: tile_w + 1] = GEN_SEPARATOR
    out[tile_h :: tile_h + 1, :] = GEN_SEPARATOR
    out[:, tile_w :: tile_w + 1] = GEN_SEPARATOR

    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    for (tr, tc), (tile_in, tile_out) in zip(positions, tile_pairs):
        r0 = tr * (tile_h + 1)
        c0 = tc * (tile_w + 1)
        inp[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile_in
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile_out

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
