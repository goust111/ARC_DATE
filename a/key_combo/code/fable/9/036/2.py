from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("NE", "SE", "SW", "NW")
ORIENTATION_TO_INDEX = {name: idx for idx, name in enumerate(ORIENTATIONS)}
ORIENTATION_ARMS = {
    "NE": ((-1, 0), (0, 1)),
    "SE": ((1, 0), (0, 1)),
    "SW": ((1, 0), (0, -1)),
    "NW": ((-1, 0), (0, -1)),
}

RETRY_LIMIT = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    q = deque([next(iter(coords))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == coords


def _split_segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for sep in separators + [length]:
        if sep <= start:
            return []
        segments.append((start, sep))
        start = sep + 1
    if start != length + 1:
        return []
    return segments


def _rotate_offset_cw(dr: int, dc: int, turns: int) -> tuple[int, int]:
    turns %= 4
    for _ in range(turns):
        dr, dc = dc, -dr
    return dr, dc


def _rotation_turns(src: str, dst: str) -> int:
    return (ORIENTATION_TO_INDEX[dst] - ORIENTATION_TO_INDEX[src]) % 4


def _orientation_from_anchor(anchor_r: int, anchor_c: int, h: int, w: int) -> str:
    if (anchor_r, anchor_c) == (0, 0):
        return "SE"
    if (anchor_r, anchor_c) == (0, w - 1):
        return "SW"
    if (anchor_r, anchor_c) == (h - 1, 0):
        return "NE"
    if (anchor_r, anchor_c) == (h - 1, w - 1):
        return "NW"
    raise AssertionError("anchor must be on a bbox corner")


def _parse_sprite_tile(tile: np.ndarray, bg: int) -> dict:
    mask = tile != bg
    assert np.any(mask), "empty tile"
    sprite = _crop_bbox(tile, mask)
    h, w = sprite.shape
    grey = np.argwhere(sprite == GREY)
    assert grey.shape == (1, 2), "expected exactly one grey anchor"
    ar, ac = map(int, grey[0])
    orientation = _orientation_from_anchor(ar, ac, h, w)

    colored = np.argwhere((sprite != bg) & (sprite != GREY))
    assert colored.shape[0] >= 3, "sprite must have at least three colored pixels"
    assert np.any(sprite == GREY) and np.any(sprite != bg), "missing sprite content"

    rel_cells = []
    rel_set = {(0, 0)}
    rows_hit = {ar}
    cols_hit = {ac}
    has_row_motion = False
    has_col_motion = False
    for rr, cc in colored:
        rr = int(rr)
        cc = int(cc)
        dr = rr - ar
        dc = cc - ac
        if orientation == "SE":
            assert dr >= 0 and dc >= 0
        elif orientation == "SW":
            assert dr >= 0 and dc <= 0
        elif orientation == "NE":
            assert dr <= 0 and dc >= 0
        else:
            assert dr <= 0 and dc <= 0
        assert dr != 0 or dc != 0
        rel_cells.append((dr, dc, int(sprite[rr, cc])))
        rel_set.add((dr, dc))
        rows_hit.add(rr)
        cols_hit.add(cc)
        has_row_motion |= dr != 0
        has_col_motion |= dc != 0

    assert has_row_motion and has_col_motion, "orientation must be visibly two-dimensional"
    assert len(rows_hit) == h and len(cols_hit) == w, "sprite bbox must use every row and col"
    assert _is_connected(rel_set), "sprite must be 4-connected including the anchor"
    return {
        "count": len(rel_cells),
        "orientation": orientation,
        "cells": tuple(rel_cells),
        "shape": (h, w),
    }


def _parse_band_horizontal(region: np.ndarray, bg: int) -> list[dict]:
    h, w = region.shape
    assert not np.all(region == bg), "empty band"
    separators = [c for c in range(w) if np.all(region[:, c] == bg)]
    segments = _split_segments(w, separators)
    assert 2 <= len(segments) <= 4, "band must contain 2-4 tiles"
    tile_widths = {c1 - c0 for c0, c1 in segments}
    assert len(tile_widths) == 1, "tiles must have equal width"
    assert next(iter(tile_widths)) >= 3 and h >= 3
    return [_parse_sprite_tile(region[:, c0:c1], bg) for c0, c1 in segments]


def _parse_band_vertical(region: np.ndarray, bg: int) -> list[dict]:
    h, w = region.shape
    assert not np.all(region == bg), "empty band"
    separators = [r for r in range(h) if np.all(region[r, :] == bg)]
    segments = _split_segments(h, separators)
    assert 2 <= len(segments) <= 4, "band must contain 2-4 tiles"
    tile_heights = {r1 - r0 for r0, r1 in segments}
    assert len(tile_heights) == 1, "tiles must have equal height"
    assert next(iter(tile_heights)) >= 3 and w >= 3
    return [_parse_sprite_tile(region[r0:r1, :], bg) for r0, r1 in segments]


def _parse_l_component(comp: list[tuple[int, int]]) -> dict:
    assert len(comp) == 3, "L marker must have exactly three pixels"
    pts = set(comp)
    degree_map = {
        cell: sum((nr, nc) in pts for nr, nc in _neighbors4(*cell))
        for cell in pts
    }
    elbows = [cell for cell, deg in degree_map.items() if deg == 2]
    assert len(elbows) == 1, "L marker must have one elbow"
    elbow = elbows[0]
    arms = []
    for rr, cc in pts:
        if (rr, cc) == elbow:
            continue
        dr = rr - elbow[0]
        dc = cc - elbow[1]
        assert abs(dr) + abs(dc) == 1, "arms must be 4-neighbors of the elbow"
        arms.append((dr, dc))
    arm_set = frozenset(arms)
    mapping = {
        frozenset({(-1, 0), (0, 1)}): "NE",
        frozenset({(1, 0), (0, 1)}): "SE",
        frozenset({(1, 0), (0, -1)}): "SW",
        frozenset({(-1, 0), (0, -1)}): "NW",
    }
    assert arm_set in mapping, "invalid L orientation"
    return {"pos": elbow, "orientation": mapping[arm_set]}


def _parse_canvas(canvas: np.ndarray, bg: int) -> list[dict]:
    allowed = {bg, GREY}
    assert set(map(int, np.unique(canvas))) <= allowed, "canvas must contain only bg and grey anchors"
    grey_mask = canvas == GREY
    comps = _components(grey_mask)
    anchors = [_parse_l_component(comp) for comp in comps]
    assert 2 <= len(anchors) <= 4, "expected 2-4 L anchors"
    orientations = [anchor["orientation"] for anchor in anchors]
    assert len(set(orientations)) == len(orientations), "L orientations must be distinct"
    return sorted(anchors, key=lambda item: item["pos"])


def _render_canvas(shape: tuple[int, int], sprites: list[dict], anchors: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    occupied: set[tuple[int, int]] = set()
    for sprite, anchor in zip(sprites, anchors):
        turns = _rotation_turns(sprite["orientation"], anchor["orientation"])
        ar, ac = anchor["pos"]
        for dr, dc, color in sprite["cells"]:
            rr, cc = _rotate_offset_cw(dr, dc, turns)
            cell = (ar + rr, ac + cc)
            assert 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1], "placed sprite out of bounds"
            assert cell not in occupied, "placed sprites must not overlap"
            occupied.add(cell)
            out[cell] = color
    return out


def _solve_with_regions(band: np.ndarray, canvas: np.ndarray, strip_kind: str, bg: int) -> np.ndarray:
    sprites = _parse_band_horizontal(band, bg) if strip_kind == "horizontal" else _parse_band_vertical(band, bg)
    anchors = _parse_canvas(canvas, bg)
    assert len(sprites) == len(anchors), "sprite and anchor counts must match"
    counts = [sprite["count"] for sprite in sprites]
    assert len(set(counts)) == len(counts), "sprite counts must be distinct"
    ordered_sprites = sorted(sprites, key=lambda item: item["count"])
    return _render_canvas(canvas.shape, ordered_sprites, anchors, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Rank the strip sprites by colored area, pair them with reading-order L anchors, rotate them to the anchor orientation, and place them onto the canvas.

    Input structure:
    - One full background separator line splits the grid into a sprite-tile strip and a larger canvas.
    - The strip is cut into equal tiles by separator bars; each tile contains one multicolor sprite with exactly one grey anchor pixel.
    - The grey anchor sits on a corner of the sprite bbox, and the sprite extends from that anchor into exactly one quadrant.
    - The canvas contains the same number of grey 3-pixel L markers, each encoding both an anchor position and an orientation.

    Rule:
    - Count the non-grey foreground pixels in each sprite tile and sort the sprites from smallest to largest.
    - Sort the canvas L markers by reading order of their elbow cells: top-to-bottom, then left-to-right.
    - Pair the k-th sprite with the k-th L marker, rotate the sprite so its anchor lands on the L elbow and its occupied quadrant matches the L opening, then paint only the non-grey sprite pixels.
    - Remove all grey pixels; the output is only the transformed canvas region.

    Key insight:
    - The hidden correspondence is rank-to-rank: counts choose which sprite goes to which anchor, while the L shape simultaneously supplies the landing point and the required rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)

    candidates = []
    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == bg)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == bg)]

    for r in full_rows:
        top = g[:r, :]
        bottom = g[r + 1 :, :]
        if min(top.shape[0], bottom.shape[0]) < 3:
            continue
        for band, canvas in ((top, bottom), (bottom, top)):
            try:
                out = _solve_with_regions(band, canvas, "horizontal", bg)
            except AssertionError:
                continue
            candidates.append(out)

    for c in full_cols:
        left = g[:, :c]
        right = g[:, c + 1 :]
        if min(left.shape[1], right.shape[1]) < 3:
            continue
        for band, canvas in ((left, right), (right, left)):
            try:
                out = _solve_with_regions(band, canvas, "vertical", bg)
            except AssertionError:
                continue
            candidates.append(out)

    unique = []
    for cand in candidates:
        if not any(np.array_equal(cand, prev) for prev in unique):
            unique.append(cand)
    assert len(unique) == 1, "input must admit a unique interpretation"
    return unique[0]


def _rotate_cells(cells: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    return [_rotate_offset_cw(dr, dc, turns) for dr, dc in cells]


def _render_sprite_tile(size: int, base_cells: list[tuple[int, int]], colors: list[int], orientation: str) -> np.ndarray:
    tile = np.full((size, size), BG, dtype=int)
    anchor_pos = {
        "SE": (0, 0),
        "SW": (0, size - 1),
        "NE": (size - 1, 0),
        "NW": (size - 1, size - 1),
    }[orientation]
    tile[anchor_pos] = GREY
    turns = _rotation_turns("SE", orientation)
    for (dr, dc), color in zip(_rotate_cells(base_cells, turns), colors):
        tile[anchor_pos[0] + dr, anchor_pos[1] + dc] = color
    return tile


def _gen_base_sprite(size: int, count: int) -> list[tuple[int, int]] | None:
    required = size
    if count < required:
        return None
    for _ in range(200):
        coords = {(0, 0), (0, 1), (1, 1), (size - 1, size - 1)}
        while len(coords) - 1 < count:
            frontier = []
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) - 1 != count:
            continue
        rows_hit = {r for r, _ in coords}
        cols_hit = {c for _, c in coords}
        if len(rows_hit) != size or len(cols_hit) != size:
            continue
        if not _is_connected(coords):
            continue
        if not any(r > 0 and c > 0 for r, c in coords if (r, c) != (0, 0)):
            continue
        colored = sorted(cell for cell in coords if cell != (0, 0))
        return colored
    return None


def _gen_colors(count: int) -> list[int]:
    color_count = random.randint(2, min(4, count))
    chosen = random.sample(PALETTE, color_count)
    colors = [chosen[i % color_count] for i in range(count)]
    random.shuffle(colors)
    if len(set(colors)) < 2:
        colors[0] = chosen[0]
        colors[-1] = chosen[1]
    return colors


def _l_cells(elbow: tuple[int, int], orientation: str) -> set[tuple[int, int]]:
    r, c = elbow
    return {(r, c), (r + ORIENTATION_ARMS[orientation][0][0], c + ORIENTATION_ARMS[orientation][0][1]), (r + ORIENTATION_ARMS[orientation][1][0], c + ORIENTATION_ARMS[orientation][1][1])}


def _placement_signature(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    sig = set()
    for r, c in cells:
        sig.add((r, c))
        for nr, nc in _neighbors4(r, c):
            sig.add((nr, nc))
    return sig


def _gen_anchor_positions(canvas_shape: tuple[int, int], sprites: list[dict], target_orientations: list[str]) -> list[dict] | None:
    h, w = canvas_shape
    anchors: list[dict] = []
    occupied_out: set[tuple[int, int]] = set()
    occupied_l_with_padding: set[tuple[int, int]] = set()

    for sprite, orientation in zip(sprites, target_orientations):
        turns = _rotation_turns("SE", orientation)
        rotated = _rotate_cells(sprite["base_cells"], turns)
        arm_a, arm_b = ORIENTATION_ARMS[orientation]
        drs = [0, arm_a[0], arm_b[0]] + [dr for dr, _ in rotated]
        dcs = [0, arm_a[1], arm_b[1]] + [dc for _, dc in rotated]
        min_r, max_r = min(drs), max(drs)
        min_c, max_c = min(dcs), max(dcs)
        candidates = [
            (r, c)
            for r in range(-min_r, h - max_r)
            for c in range(-min_c, w - max_c)
        ]
        random.shuffle(candidates)

        placed = None
        for r, c in candidates:
            out_cells = {(r + dr, c + dc) for dr, dc in rotated}
            l_cells = _l_cells((r, c), orientation)
            if out_cells & occupied_out:
                continue
            if _placement_signature(l_cells) & occupied_l_with_padding:
                continue
            placed = {
                "pos": (r, c),
                "orientation": orientation,
                "out_cells": out_cells,
                "l_cells": l_cells,
            }
            break
        if placed is None:
            return None

        anchors.append(placed)
        occupied_out |= placed["out_cells"]
        occupied_l_with_padding |= _placement_signature(placed["l_cells"])

    anchors.sort(key=lambda item: item["pos"])

    out_occupied: set[tuple[int, int]] = set()
    l_occupied_with_padding: set[tuple[int, int]] = set()
    checked: list[dict] = []
    for sprite, anchor in zip(sprites, anchors):
        turns = _rotation_turns("SE", anchor["orientation"])
        rotated = _rotate_cells(sprite["base_cells"], turns)
        r, c = anchor["pos"]
        out_cells = {(r + dr, c + dc) for dr, dc in rotated}
        l_cells = _l_cells((r, c), anchor["orientation"])
        if out_cells & out_occupied:
            return None
        if _placement_signature(l_cells) & l_occupied_with_padding:
            return None
        out_occupied |= out_cells
        l_occupied_with_padding |= _placement_signature(l_cells)
        checked.append({"pos": anchor["pos"], "orientation": anchor["orientation"]})
    return checked


def _render_canvas_input(shape: tuple[int, int], anchors: list[dict]) -> np.ndarray:
    canvas = np.full(shape, BG, dtype=int)
    for anchor in anchors:
        for r, c in _l_cells(anchor["pos"], anchor["orientation"]):
            canvas[r, c] = GREY
    return canvas


def _render_expected_canvas(shape: tuple[int, int], sprites: list[dict], anchors: list[dict]) -> np.ndarray:
    ordered = []
    for sprite in sprites:
        ordered.append(
            {
                "count": sprite["count"],
                "orientation": sprite["source_orientation"],
                "cells": tuple((dr, dc, color) for (dr, dc), color in zip(_rotate_cells(sprite["base_cells"], _rotation_turns("SE", sprite["source_orientation"])), sprite["colors"])),
            }
        )
    return _render_canvas(shape, ordered, anchors, BG)


def _render_naive_input_order(shape: tuple[int, int], strip_sprites: list[dict], anchors: list[dict]) -> np.ndarray:
    naive = []
    for sprite in strip_sprites:
        naive.append(
            {
                "count": sprite["count"],
                "orientation": sprite["source_orientation"],
                "cells": tuple((dr, dc, color) for (dr, dc), color in zip(_rotate_cells(sprite["base_cells"], _rotation_turns("SE", sprite["source_orientation"])), sprite["colors"])),
            }
        )
    return _render_canvas(shape, naive, anchors, BG)


def _render_naive_no_rotation(shape: tuple[int, int], sprites: list[dict], anchors: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for sprite, anchor in zip(sprites, anchors):
        ar, ac = anchor["pos"]
        for (dr, dc), color in zip(_rotate_cells(sprite["base_cells"], _rotation_turns("SE", sprite["source_orientation"])), sprite["colors"]):
            rr = ar + dr
            cc = ac + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                out[rr, cc] = color
    return out


def _build_input_grid(
    tile_size: int,
    band_orientation: str,
    band_side: str,
    strip_sprites: list[dict],
    anchors: list[dict],
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(strip_sprites)
    if band_orientation == "horizontal":
        band_w = n * tile_size + (n - 1)
        band = np.full((tile_size, band_w), BG, dtype=int)
        for idx, sprite in enumerate(strip_sprites):
            c0 = idx * (tile_size + 1)
            tile = _render_sprite_tile(tile_size, sprite["base_cells"], sprite["colors"], sprite["source_orientation"])
            band[:, c0 : c0 + tile_size] = tile
        canvas = _render_canvas_input(canvas_shape, anchors)
        divider = np.full((1, band_w), BG, dtype=int)
        if band_side == "top":
            inp = np.vstack([band, divider, canvas])
        else:
            inp = np.vstack([canvas, divider, band])
    else:
        band_h = n * tile_size + (n - 1)
        band = np.full((band_h, tile_size), BG, dtype=int)
        for idx, sprite in enumerate(strip_sprites):
            r0 = idx * (tile_size + 1)
            tile = _render_sprite_tile(tile_size, sprite["base_cells"], sprite["colors"], sprite["source_orientation"])
            band[r0 : r0 + tile_size, :] = tile
        canvas = _render_canvas_input(canvas_shape, anchors)
        divider = np.full((band_h, 1), BG, dtype=int)
        if band_side == "left":
            inp = np.hstack([band, divider, canvas])
        else:
            inp = np.hstack([canvas, divider, band])
    expected = _render_expected_canvas(canvas_shape, sorted(strip_sprites, key=lambda item: item["count"]), anchors)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(2, 4)
    tile_size = random.randint(4, 6)
    band_orientation = random.choice(("horizontal", "vertical"))
    band_side = random.choice(("top", "bottom")) if band_orientation == "horizontal" else random.choice(("left", "right"))

    count_choices = list(range(tile_size, min(tile_size * tile_size - 1, 11) + 1))
    if len(count_choices) < n:
        return None
    counts = sorted(random.sample(count_choices, n))

    sprites = []
    for count in counts:
        base_cells = _gen_base_sprite(tile_size, count)
        if base_cells is None:
            return None
        sprites.append(
            {
                "count": count,
                "base_cells": base_cells,
                "colors": _gen_colors(count),
                "source_orientation": random.choice(ORIENTATIONS),
            }
        )

    target_orientations = random.sample(list(ORIENTATIONS), n)
    if all(sprite["source_orientation"] == target for sprite, target in zip(sprites, target_orientations)):
        return None

    band_len = n * tile_size + (n - 1)
    if band_orientation == "horizontal":
        max_canvas_h = 29 - tile_size
        if max_canvas_h < 7:
            return None
        canvas_shape = (random.randint(max(tile_size + 2, 7), min(14, max_canvas_h)), band_len)
    else:
        max_canvas_w = 29 - tile_size
        if max_canvas_w < 7:
            return None
        canvas_shape = (band_len, random.randint(max(tile_size + 2, 7), min(14, max_canvas_w)))

    anchors = _gen_anchor_positions(canvas_shape, sprites, target_orientations)
    if anchors is None:
        return None

    strip_order = list(range(n))
    random.shuffle(strip_order)
    if strip_order == list(range(n)):
        return None
    strip_sprites = [sprites[idx] for idx in strip_order]
    inp, expected = _build_input_grid(tile_size, band_orientation, band_side, strip_sprites, anchors, canvas_shape)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    strip_counts = [sprite["count"] for sprite in strip_sprites]
    if strip_counts == sorted(strip_counts):
        return None

    try:
        naive_input_order = _render_naive_input_order(canvas_shape, strip_sprites, anchors)
    except AssertionError:
        naive_input_order = None
    if naive_input_order is not None and np.array_equal(naive_input_order, expected):
        return None

    naive_no_rotation = _render_naive_no_rotation(canvas_shape, sorted(strip_sprites, key=lambda item: item["count"]), anchors)
    if np.array_equal(naive_no_rotation, expected):
        return None

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for rank-paired L anchors")
