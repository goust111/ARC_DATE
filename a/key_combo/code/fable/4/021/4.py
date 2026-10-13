from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# Generator bounds.
SPRITE_COUNT_OPTIONS = (3, 4, 5)
WIDTH_RANGE = (3, 5)
HEIGHT_RANGE = (2, 5)
MAX_OUTPUT_HEIGHT = 12
MAX_OUTPUT_WIDTH = 28
MAX_LAYOUT_ATTEMPTS = 200
MAX_MASK_ATTEMPTS = 120
MAX_PLACE_ATTEMPTS = 250
MAX_GENERATE_ATTEMPTS = 300


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox_from_coords(arr)})
    return components


def _edge_side(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = coord
    _r0, c0, _r1, c1 = bbox
    if c0 == c1:
        return None
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    return None


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = _find_components(g)
    assert len(components) >= 2, "expected a chain of at least two sprites"

    cell_to_component: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(components):
        for r, c in comp["coords"]:
            cell_to_component[(int(r), int(c))] = idx

    color_coords: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != BG):
        color_coords.setdefault(int(g[r, c]), []).append((int(r), int(c)))

    connector_entries: dict[int, list[tuple[int, str, tuple[int, int]]]] = {}
    for color, coords in color_coords.items():
        if len(coords) != 2:
            continue
        comp_ids = [cell_to_component[coord] for coord in coords]
        if comp_ids[0] == comp_ids[1]:
            continue
        entries = []
        sides = []
        ok = True
        for coord, comp_id in zip(coords, comp_ids):
            side = _edge_side(coord, components[comp_id]["bbox"])
            if side is None:
                ok = False
                break
            entries.append((comp_id, side, coord))
            sides.append(side)
        if ok and sorted(sides) == ["left", "right"]:
            connector_entries[int(color)] = entries

    connector_colors = set(connector_entries)
    assert connector_colors, "expected connector colors that appear once on each side of a seam"
    assert len(connector_colors) == len(components) - 1, "a chain of n sprites must have n-1 connector colors"

    sprites = []
    for idx, comp in enumerate(components):
        bbox = comp["bbox"]
        raw = _crop(g, bbox)
        present = set(map(int, np.unique(raw))) - {BG}
        body_colors = present - connector_colors
        assert len(body_colors) == 1, "each sprite must have exactly one body color"
        body_color = body_colors.pop()

        own_connectors = []
        for color, entries in connector_entries.items():
            for comp_id, side, coord in entries:
                if comp_id == idx:
                    own_connectors.append((int(color), side, coord))
        left_entries = [item for item in own_connectors if item[1] == "left"]
        right_entries = [item for item in own_connectors if item[1] == "right"]
        assert len(left_entries) <= 1 and len(right_entries) <= 1, "each sprite should have at most one connector per side"
        assert left_entries or right_entries, "every sprite must participate in the chain"

        filled = raw.copy()
        filled[filled != BG] = body_color
        r0, _c0, _r1, _c1 = bbox
        left_color = left_entries[0][0] if left_entries else None
        right_color = right_entries[0][0] if right_entries else None
        left_row = int(left_entries[0][2][0] - r0) if left_entries else None
        right_row = int(right_entries[0][2][0] - r0) if right_entries else None
        sprites.append(
            {
                "body_color": int(body_color),
                "raw": raw,
                "filled": filled,
                "bbox": bbox,
                "left_color": left_color,
                "right_color": right_color,
                "left_row": left_row,
                "right_row": right_row,
            }
        )

    body_colors = [sprite["body_color"] for sprite in sprites]
    assert len(body_colors) == len(set(body_colors)), "sprite body colors must be distinct"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into a horizontal chain by matching colored edge connectors.

    Input structure
    - The grid has a black background with several separated multi-color sprites.
    - Each sprite has one unique body color plus connector pixels on the left and/or right edge of its bbox.
    - Every connector color appears exactly twice overall: once on a sprite's right edge and once on another sprite's left edge.

    Rule
    - Use equal connector colors as directed links from one sprite's right edge to the next sprite's left edge, starting at the only sprite with no left connector.
    - Crop every sprite to its bbox, follow that chain order, and concatenate the crops left to right while vertically offsetting neighbors so matched connector rows lie on the same output row.
    - Recolor each connector pixel to its sprite's body color and return the minimal black rectangle containing the assembled chain.

    Key insight
    - Each connector color is simultaneously an ordering pointer, a seam-alignment pin, and a pixel that must be absorbed back into the owning sprite's body.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _extract_sprites(g)
    left_lookup = {}
    heads = []
    tails = []
    for idx, sprite in enumerate(sprites):
        if sprite["left_color"] is None:
            heads.append(idx)
        else:
            left_lookup[int(sprite["left_color"])] = idx
        if sprite["right_color"] is None:
            tails.append(idx)
    assert len(heads) == 1, "expected exactly one chain head"
    assert len(tails) == 1, "expected exactly one chain tail"

    order = []
    seen = set()
    cur = heads[0]
    while True:
        assert cur not in seen, "chain should not contain a cycle"
        seen.add(cur)
        order.append(cur)
        right_color = sprites[cur]["right_color"]
        if right_color is None:
            break
        assert int(right_color) in left_lookup, "every right connector must point to a later sprite"
        cur = left_lookup[int(right_color)]
    assert len(order) == len(sprites), "the connector chain must visit every sprite"

    tops = [0]
    for left_idx, right_idx in zip(order, order[1:]):
        left_sprite = sprites[left_idx]
        right_sprite = sprites[right_idx]
        assert left_sprite["right_row"] is not None and right_sprite["left_row"] is not None
        tops.append(tops[-1] + int(left_sprite["right_row"]) - int(right_sprite["left_row"]))

    shift = -min(tops)
    norm_tops = [top + shift for top in tops]
    out_h = max(
        top + int(sprites[idx]["filled"].shape[0])
        for top, idx in zip(norm_tops, order)
    )
    out_w = sum(int(sprites[idx]["filled"].shape[1]) for idx in order)
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for top, idx in zip(norm_tops, order):
        sprite = sprites[idx]["filled"]
        h, w = sprite.shape
        out[top : top + h, col : col + w] = sprite
        col += w
    return out


def _sample_layout() -> tuple[list[int], list[int], list[int | None], list[int | None], list[int]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        n = random.choice(SPRITE_COUNT_OPTIONS)
        widths = [random.randint(*WIDTH_RANGE) for _ in range(n)]
        heights = [random.randint(*HEIGHT_RANGE) for _ in range(n)]

        tops = [0]
        for i in range(1, n):
            low = tops[-1] - (heights[i] - 1)
            high = tops[-1] + (heights[i - 1] - 1)
            tops.append(random.randint(low, high))

        left_rows: list[int | None] = [None] * n
        right_rows: list[int | None] = [None] * n
        for i in range(n - 1):
            lo = max(tops[i], tops[i + 1])
            hi = min(tops[i] + heights[i] - 1, tops[i + 1] + heights[i + 1] - 1)
            if lo > hi:
                break
            seam_row = random.randint(lo, hi)
            right_rows[i] = seam_row - tops[i]
            left_rows[i + 1] = seam_row - tops[i + 1]
        else:
            shift = -min(tops)
            norm_tops = [top + shift for top in tops]
            out_h = max(top + h for top, h in zip(norm_tops, heights))
            out_w = sum(widths)
            if out_h <= MAX_OUTPUT_HEIGHT and out_w <= MAX_OUTPUT_WIDTH:
                return widths, heights, left_rows, right_rows, norm_tops
    return None


def _sample_body_mask(h: int, w: int, left_row: int | None, right_row: int | None) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        left_touch_rows = {random.randrange(h)}
        right_touch_rows = {random.randrange(h)}
        if left_row is not None:
            left_touch_rows.add(int(left_row))
        if right_row is not None:
            right_touch_rows.add(int(right_row))

        intervals: list[tuple[int, int]] = []
        ok = True
        for r in range(h):
            candidates = []
            need_left = r in left_touch_rows
            need_right = r in right_touch_rows
            for lo in range(w):
                for hi in range(lo, w):
                    if need_left and lo != 0:
                        continue
                    if need_right and hi != w - 1:
                        continue
                    if intervals:
                        prev_lo, prev_hi = intervals[-1]
                        if hi < prev_lo or lo > prev_hi:
                            continue
                    candidates.append((lo, hi))
            if not candidates:
                ok = False
                break
            weights = [(hi - lo + 1) ** 2 for lo, hi in candidates]
            intervals.append(random.choices(candidates, weights=weights, k=1)[0])
        if not ok:
            continue
        if min(lo for lo, _hi in intervals) != 0 or max(hi for _lo, hi in intervals) != w - 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, (lo, hi) in enumerate(intervals):
            mask[r, lo : hi + 1] = True

        if all(lo == 0 and hi == w - 1 for lo, hi in intervals) and random.random() < 0.75:
            continue
        if left_row is not None and not mask[left_row, 0]:
            continue
        if right_row is not None and not mask[right_row, w - 1]:
            continue
        return mask
    return None


def _build_raw_sprite(
    h: int,
    w: int,
    body_color: int,
    left_color: int | None,
    right_color: int | None,
    left_row: int | None,
    right_row: int | None,
) -> np.ndarray | None:
    mask = _sample_body_mask(h, w, left_row, right_row)
    if mask is None:
        return None

    raw = np.zeros((h, w), dtype=int)
    raw[mask] = int(body_color)
    if left_color is not None:
        raw[int(left_row), 0] = int(left_color)
    if right_color is not None:
        raw[int(right_row), w - 1] = int(right_color)
    if not _is_single_component(raw != BG):
        return None
    return raw


def _render_expected(raw_sprites: list[np.ndarray], body_colors: list[int], tops: list[int]) -> np.ndarray:
    out_h = max(top + raw.shape[0] for top, raw in zip(tops, raw_sprites))
    out_w = sum(raw.shape[1] for raw in raw_sprites)
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for raw, body_color, top in zip(raw_sprites, body_colors, tops):
        filled = np.where(raw != BG, int(body_color), BG)
        h, w = raw.shape
        out[top : top + h, col : col + w] = filled
        col += w
    return out


def _bboxes_are_separate(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _placement_is_too_ordered(placements: dict[int, tuple[int, int]]) -> bool:
    chain = list(range(len(placements)))
    rev = chain[::-1]
    by_col = [idx for idx, _pos in sorted(placements.items(), key=lambda item: (item[1][1], item[1][0]))]
    by_row = [idx for idx, _pos in sorted(placements.items(), key=lambda item: (item[1][0], item[1][1]))]
    return by_col in (chain, rev) or by_row in (chain, rev)


def _scatter_sprites(raw_sprites: list[np.ndarray]) -> np.ndarray | None:
    n = len(raw_sprites)
    heights = [sprite.shape[0] for sprite in raw_sprites]
    widths = [sprite.shape[1] for sprite in raw_sprites]
    base_h = max(12, max(heights) + 6, sum(heights) // 2 + 4)
    base_w = max(16, sum(widths) + 3, max(widths) + 6)

    for _ in range(MAX_PLACE_ATTEMPTS):
        h = random.randint(base_h, min(30, base_h + 6))
        w = random.randint(base_w, min(30, base_w + 6))
        grid = np.zeros((h, w), dtype=int)
        placements: dict[int, tuple[int, int]] = {}
        bboxes: list[tuple[int, int, int, int]] = []
        order = list(range(n))
        random.shuffle(order)

        ok = True
        for idx in order:
            sprite = raw_sprites[idx]
            sh, sw = sprite.shape
            candidates = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
            random.shuffle(candidates)
            placed = False
            for r, c in candidates:
                bbox = (r, c, r + sh - 1, c + sw - 1)
                if all(_bboxes_are_separate(bbox, other, gap=1) for other in bboxes):
                    patch = grid[r : r + sh, c : c + sw]
                    patch[sprite != BG] = sprite[sprite != BG]
                    placements[idx] = (r, c)
                    bboxes.append(bbox)
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok and not _placement_is_too_ordered(placements):
            return grid
    return None


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    widths, heights, left_rows, right_rows, tops = layout
    n = len(widths)

    palette = random.sample(PALETTE, 2 * n - 1)
    body_colors = palette[:n]
    connector_colors = palette[n:]

    raw_sprites = []
    for i in range(n):
        left_color = connector_colors[i - 1] if i > 0 else None
        right_color = connector_colors[i] if i < n - 1 else None
        raw = _build_raw_sprite(
            heights[i],
            widths[i],
            body_colors[i],
            left_color,
            right_color,
            left_rows[i],
            right_rows[i],
        )
        if raw is None:
            return None
        raw_sprites.append(raw)

    expected = _render_expected(raw_sprites, body_colors, tops)
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None

    inp = _scatter_sprites(raw_sprites)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-assembly ARC input")
