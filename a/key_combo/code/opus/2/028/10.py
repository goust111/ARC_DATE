from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BAR = 2
SPRITE_COLORS = (1, 3, 4)

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = view != int(transparent)
    dst_view = dst[rr0:rr1, cc0:cc1]
    dst_view[mask] = view[mask]


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def find_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wanted = {int(color) for color in colors}
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color not in wanted:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            coords = np.array(sorted(cells), dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )

    out.sort(key=lambda comp: comp["bbox"])
    return out


def _split_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, tuple(sorted(set(SPRITE_COLORS + (BAR,)) - {bg})))
    bars: list[dict] = []
    sprites: list[dict] = []

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        color = int(comp["color"])
        if color == BAR and ((h == 1) ^ (w == 1)) and comp["area"] == h * w and 3 <= max(h, w) <= 6:
            bars.append(comp)
        else:
            assert color in SPRITE_COLORS, "expected only sprite colors besides the red bars"
            assert h > 1 and w > 1, "sprites must be non-line shapes"
            assert 3 <= int(comp["area"]) <= 6, "sprites must have 3-6 pixels"
            sprites.append(comp)

    assert 2 <= len(bars) <= 3 and len(bars) == len(sprites), "expected 2-3 sprite/bar pairs"
    return bg, bars, sprites


def _copy_origins(box: BBox, direction: str, grid_shape: tuple[int, int]) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    H, W = map(int, grid_shape)
    origins: list[Coord] = []

    if direction == "up":
        r = r0
        while r >= 0:
            origins.append((int(r), int(c0)))
            r -= h
    elif direction == "down":
        r = r0
        while r + h <= H:
            origins.append((int(r), int(c0)))
            r += h
    elif direction == "left":
        c = c0
        while c >= 0:
            origins.append((int(r0), int(c)))
            c -= w
    elif direction == "right":
        c = c0
        while c + w <= W:
            origins.append((int(r0), int(c)))
            c += w
    else:
        raise ValueError(direction)

    return origins


def _bar_stop_origins(box: BBox, bar_box: BBox, direction: str) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    br0, bc0, _br1, _bc1 = map(int, bar_box)
    origins: list[Coord] = []

    if direction == "up":
        r = r0
        while r + h - 1 >= br0:
            origins.append((int(r), int(c0)))
            r -= h
    elif direction == "down":
        r = r0
        while r <= br0:
            origins.append((int(r), int(c0)))
            r += h
    elif direction == "left":
        c = c0
        while c + w - 1 >= bc0:
            origins.append((int(r0), int(c)))
            c -= w
    elif direction == "right":
        c = c0
        while c <= bc0:
            origins.append((int(r0), int(c)))
            c += w
    else:
        raise ValueError(direction)

    return origins


def _direction_from_bar(sprite_box: BBox, bar_box: BBox) -> tuple[str, str]:
    sr0, sc0, sr1, sc1 = map(int, sprite_box)
    br0, bc0, br1, bc1 = map(int, bar_box)
    bar_h = int(br1 - br0 + 1)
    bar_w = int(bc1 - bc0 + 1)

    if bar_h == 1:
        assert sc0 == bc0 and sc1 == bc1, "horizontal bar must match sprite column span"
        assert br0 < sr0 or br0 > sr1, "horizontal bar must be above or below its sprite"
        return ("up" if br0 < sr0 else "down"), "horizontal"

    assert bar_w == 1, "bars must be straight one-cell-thick lines"
    assert sr0 == br0 and sr1 == br1, "vertical bar must match sprite row span"
    assert bc0 < sc0 or bc0 > sc1, "vertical bar must be left or right of its sprite"
    return ("left" if bc0 < sc0 else "right"), "vertical"


def _pairings(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg, bars, sprites = _split_objects(grid)
    used_sprite_boxes: set[BBox] = set()
    pairs: list[dict] = []

    for bar in bars:
        br0, bc0, br1, bc1 = bar["bbox"]
        if br0 == br1:
            aligned = [sprite for sprite in sprites if sprite["bbox"][1] == bc0 and sprite["bbox"][3] == bc1]
        else:
            aligned = [sprite for sprite in sprites if sprite["bbox"][0] == br0 and sprite["bbox"][2] == br1]
        assert len(aligned) == 1, "each control bar must match exactly one sprite by span"
        sprite = aligned[0]
        sprite_box = tuple(map(int, sprite["bbox"]))
        assert sprite_box not in used_sprite_boxes, "each sprite must be matched by exactly one bar"
        used_sprite_boxes.add(sprite_box)
        direction, bar_orientation = _direction_from_bar(sprite_box, tuple(map(int, bar["bbox"])))
        pairs.append(
            {
                "sprite": sprite,
                "bar": bar,
                "direction": direction,
                "bar_orientation": bar_orientation,
            }
        )

    assert len(used_sprite_boxes) == len(sprites), "every sprite must have one matching bar"
    pairs.sort(key=lambda item: item["sprite"]["bbox"])
    return bg, pairs


def _nearest_boundary_direction(sprite_box: BBox, bar_orientation: str, grid_shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = map(int, sprite_box)
    H, W = map(int, grid_shape)
    if bar_orientation == "horizontal":
        top = int(r0)
        bottom = int(H - 1 - r1)
        assert top != bottom, "generator should avoid nearest-boundary ties"
        return "up" if top < bottom else "down"
    left = int(c0)
    right = int(W - 1 - c1)
    assert left != right, "generator should avoid nearest-boundary ties"
    return "left" if left < right else "right"


def _render_pairs(
    grid: np.ndarray,
    *,
    use_bar_side: bool,
    stop_at_bar: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, pairs = _pairings(g)
    out = np.full(g.shape, int(bg), dtype=int)

    for pair in pairs:
        sprite = pair["sprite"]
        box = tuple(map(int, sprite["bbox"]))
        patch = crop_bbox(g, box)
        if use_bar_side:
            direction = str(pair["direction"])
        else:
            direction = _nearest_boundary_direction(box, str(pair["bar_orientation"]), g.shape)

        if stop_at_bar:
            origins = _bar_stop_origins(box, tuple(map(int, pair["bar"]["bbox"])), direction)
        else:
            origins = _copy_origins(box, direction, g.shape)

        for rr, cc in origins:
            paste(out, patch, rr, cc, transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the red control bars, then tile each matched sprite toward the bar-side boundary.

    Input structure:
    - The grid has a black background, 2-3 small monochrome irregular sprites, and 2-3 red one-cell-thick bars.
    - Every horizontal bar shares exactly one sprite's column span, and every vertical bar shares exactly one sprite's row span.
    - Each sprite is separated from its matched bar and no other input object blocks that sprite's future tiling path.

    Rule:
    - Remove all red bars.
    - Match each bar to the unique sprite with the same span on the perpendicular axis.
    - A horizontal bar makes its sprite tile vertically; a vertical bar makes its sprite tile horizontally.
    - Tile toward the side where the bar sits, starting from the original sprite and repeating every full sprite height or width until the next copy would leave the grid.
    - Process sprites in top-left scan order, so later sprites overwrite earlier copied pixels on overlaps.

    Key insight:
    - The bar is only a control signal: its span picks the sprite, its orientation picks the tiling axis, and its side relative to the sprite picks the tiling direction.
    """
    return _render_pairs(np.asarray(grid, dtype=int), use_bar_side=True, stop_at_bar=False)


def _legacy_bar_stop_main(grid: np.ndarray) -> np.ndarray:
    return _render_pairs(np.asarray(grid, dtype=int), use_bar_side=True, stop_at_bar=True)


def _nearest_boundary_main(grid: np.ndarray) -> np.ndarray:
    return _render_pairs(np.asarray(grid, dtype=int), use_bar_side=False, stop_at_bar=False)


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return crop_bbox(mask.astype(np.uint8), bbox_coords(coords)).astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _normalize_mask(mask)
    return norm.shape, norm.astype(np.uint8).tobytes()


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for k in range(4):
        rotated = np.rot90(mask, k=k)
        for variant in (rotated, np.fliplr(rotated)):
            norm = _normalize_mask(variant)
            variants[_mask_key(norm)] = norm
    return list(variants.values())


_BASE_MASKS = (
    _mask_from_rows(("10", "11")),
    _mask_from_rows(("01", "11")),
    _mask_from_rows(("111", "010")),
    _mask_from_rows(("011", "110")),
    _mask_from_rows(("11", "11", "10")),
    _mask_from_rows(("11", "01", "01")),
    _mask_from_rows(("101", "111")),
    _mask_from_rows(("010", "111", "001")),
)


_SPRITE_MASKS: list[np.ndarray] = []
for _base in _BASE_MASKS:
    for _variant in _all_variants(_base):
        if not any(np.array_equal(_variant, old) for old in _SPRITE_MASKS):
            area = int(_variant.sum())
            if 3 <= area <= 6 and _variant.shape[0] > 1 and _variant.shape[1] > 1:
                _SPRITE_MASKS.append(_variant)


GRID_MIN = 16
GRID_MAX = 24
GEN_RETRIES = 400
CANDIDATE_RETRIES = 160


def _sprite_cells(mask: np.ndarray, r0: int, c0: int) -> set[Coord]:
    coords = np.argwhere(mask)
    return {(int(r0 + r), int(c0 + c)) for r, c in coords}


def _box_cells(r0: int, c0: int, h: int, w: int) -> set[Coord]:
    return {(int(r), int(c)) for r in range(int(r0), int(r0 + h)) for c in range(int(c0), int(c0 + w))}


def _bar_cells(direction: str, r0: int, c0: int, h: int, w: int, gap: int) -> tuple[set[Coord], BBox]:
    if direction == "up":
        br = int(r0 - gap - 1)
        cells = {(br, int(c)) for c in range(c0, c0 + w)}
        return cells, (br, c0, br, c0 + w - 1)
    if direction == "down":
        br = int(r0 + h + gap)
        cells = {(br, int(c)) for c in range(c0, c0 + w)}
        return cells, (br, c0, br, c0 + w - 1)
    if direction == "left":
        bc = int(c0 - gap - 1)
        cells = {(int(r), bc) for r in range(r0, r0 + h)}
        return cells, (r0, bc, r0 + h - 1, bc)
    if direction == "right":
        bc = int(c0 + w + gap)
        cells = {(int(r), bc) for r in range(r0, r0 + h)}
        return cells, (r0, bc, r0 + h - 1, bc)
    raise ValueError(direction)


def _origins_from_position(r0: int, c0: int, h: int, w: int, direction: str, shape: tuple[int, int]) -> list[Coord]:
    return _copy_origins((r0, c0, r0 + h - 1, c0 + w - 1), direction, shape)


def _footprint_from_origins(origins: list[Coord], h: int, w: int) -> set[Coord]:
    cells: set[Coord] = set()
    for rr, cc in origins:
        cells |= _box_cells(rr, cc, h, w)
    return cells


def _painted_from_origins(mask: np.ndarray, origins: list[Coord]) -> set[Coord]:
    cells: set[Coord] = set()
    for rr, cc in origins:
        cells |= _sprite_cells(mask, rr, cc)
    return cells


def _expanded(cells: set[Coord]) -> set[Coord]:
    halo: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((int(r + dr), int(c + dc)))
    return halo


def _opposite_nearest(direction: str, r0: int, c0: int, h: int, w: int, H: int, W: int) -> bool:
    top = int(r0)
    bottom = int(H - 1 - (r0 + h - 1))
    left = int(c0)
    right = int(W - 1 - (c0 + w - 1))
    if direction == "up":
        return top > bottom
    if direction == "down":
        return bottom > top
    if direction == "left":
        return left > right
    if direction == "right":
        return right > left
    raise ValueError(direction)


def _anti_legacy_gap_ok(direction: str, gap: int, copies: int, span: int) -> bool:
    if copies < 3:
        return False
    return int(gap) < int((copies - 2) * span)


def _sample_candidate(
    H: int,
    W: int,
    direction: str,
    color: int,
    *,
    force_opposite_nearest: bool,
    force_anti_legacy: bool,
) -> dict | None:
    for _ in range(CANDIDATE_RETRIES):
        mask = random.choice(_SPRITE_MASKS)
        h, w = map(int, mask.shape)

        if direction in ("up", "down"):
            max_copies = min(5, H // h)
            min_copies = 3 if force_anti_legacy else 2
            if max_copies < min_copies:
                continue
            copies = random.randint(min_copies, max_copies)
            slack_max = min(h - 1, H - copies * h)
            if slack_max < 0:
                continue
            slack = random.randint(0, slack_max)
            r0 = int(slack + (copies - 1) * h) if direction == "up" else int(H - copies * h - slack)
            c0 = random.randint(0, W - w)
            max_gap = int(r0 - 1) if direction == "up" else int(H - (r0 + h) - 1)
            if max_gap < 1:
                continue
            gap_high = min(3, max_gap)
            if force_anti_legacy:
                gap_high = min(gap_high, (copies - 2) * h - 1)
            if gap_high < 1:
                continue
            gap = random.randint(1, gap_high)
        else:
            max_copies = min(5, W // w)
            min_copies = 3 if force_anti_legacy else 2
            if max_copies < min_copies:
                continue
            copies = random.randint(min_copies, max_copies)
            slack_max = min(w - 1, W - copies * w)
            if slack_max < 0:
                continue
            slack = random.randint(0, slack_max)
            c0 = int(slack + (copies - 1) * w) if direction == "left" else int(W - copies * w - slack)
            r0 = random.randint(0, H - h)
            max_gap = int(c0 - 1) if direction == "left" else int(W - (c0 + w) - 1)
            if max_gap < 1:
                continue
            gap_high = min(3, max_gap)
            if force_anti_legacy:
                gap_high = min(gap_high, (copies - 2) * w - 1)
            if gap_high < 1:
                continue
            gap = random.randint(1, gap_high)

        if force_opposite_nearest and not _opposite_nearest(direction, r0, c0, h, w, H, W):
            continue

        if not force_opposite_nearest:
            if direction in ("up", "down"):
                top = int(r0)
                bottom = int(H - (r0 + h))
                if top == bottom:
                    continue
            else:
                left = int(c0)
                right = int(W - (c0 + w))
                if left == right:
                    continue

        if force_anti_legacy and not _anti_legacy_gap_ok(direction, gap, copies, h if direction in ("up", "down") else w):
            continue

        sprite = _sprite_cells(mask, r0, c0)
        bar, bar_box = _bar_cells(direction, r0, c0, h, w, gap)
        origins = _origins_from_position(r0, c0, h, w, direction, (H, W))
        if len(origins) != copies:
            continue

        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        input_cells = sprite | bar
        painted = _painted_from_origins(mask, origins)
        footprint = _footprint_from_origins(origins, h, w)

        return {
            "mask": mask,
            "color": int(color),
            "bbox": bbox,
            "row_span": (bbox[0], bbox[2]),
            "col_span": (bbox[1], bbox[3]),
            "direction": direction,
            "bar_box": bar_box,
            "sprite_cells": sprite,
            "bar_cells": bar,
            "input_cells": input_cells,
            "input_halo": _expanded(input_cells),
            "copy_origins": origins,
            "painted_cells": painted,
            "bbox_footprint": footprint,
            "opposite_nearest": _opposite_nearest(direction, r0, c0, h, w, H, W),
        }

    return None


def _compatible(existing: list[dict], candidate: dict) -> bool:
    for other in existing:
        if tuple(candidate["row_span"]) == tuple(other["row_span"]):
            return False
        if tuple(candidate["col_span"]) == tuple(other["col_span"]):
            return False
        if candidate["input_cells"] & other["input_halo"]:
            return False
        if other["input_cells"] & candidate["input_halo"]:
            return False
        if candidate["input_cells"] & other["bbox_footprint"]:
            return False
        if other["input_cells"] & candidate["bbox_footprint"]:
            return False
    return True


def _render_input(shape: tuple[int, int], candidates: list[dict]) -> np.ndarray:
    H, W = map(int, shape)
    grid = np.full((H, W), BG, dtype=int)
    for cand in candidates:
        for r, c in cand["sprite_cells"]:
            grid[int(r), int(c)] = int(cand["color"])
        for r, c in cand["bar_cells"]:
            grid[int(r), int(c)] = BAR
    return grid


def _render_expected(shape: tuple[int, int], candidates: list[dict]) -> np.ndarray:
    H, W = map(int, shape)
    out = np.full((H, W), BG, dtype=int)
    ordered = sorted(candidates, key=lambda cand: cand["bbox"])
    for cand in ordered:
        patch = np.where(cand["mask"], int(cand["color"]), BG).astype(int)
        for rr, cc in cand["copy_origins"]:
            paste(out, patch, rr, cc, transparent=BG)
    return out


def _ordered_overlap_cells(candidates: list[dict]) -> set[Coord]:
    ordered = sorted(candidates, key=lambda cand: cand["bbox"])
    seen: set[Coord] = set()
    overlap: set[Coord] = set()
    for cand in ordered:
        current = set(cand["painted_cells"])
        overlap |= seen & current
        seen |= current
    return overlap


def _valid_instance(inp: np.ndarray, candidates: list[dict]) -> bool:
    try:
        solved = main(inp)
        legacy = _legacy_bar_stop_main(inp)
        nearest = _nearest_boundary_main(inp)
    except AssertionError:
        return False

    expected = _render_expected(inp.shape, candidates)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, legacy):
        return False
    if np.array_equal(solved, nearest):
        return False
    if not any(bool(cand["opposite_nearest"]) for cand in candidates):
        return False
    if not all(len(cand["copy_origins"]) >= 2 for cand in candidates):
        return False
    if np.count_nonzero(solved) <= np.count_nonzero(inp):
        return False
    if np.count_nonzero(solved) * 100 >= 55 * solved.size:
        return False

    bg, pairs = _pairings(inp)
    if bg != BG or len(pairs) != len(candidates):
        return False
    if any(int(inp[r, c]) != BAR for cand in candidates for r, c in cand["bar_cells"]):
        return False
    if any(int(inp[r, c]) != int(cand["color"]) for cand in candidates for r, c in cand["sprite_cells"]):
        return False

    overlap = _ordered_overlap_cells(candidates)
    if random.random() < 0.25 and not overlap:
        return False
    return True


def _attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    count = random.choice((2, 3))
    directions = random.sample(["up", "down", "left", "right"], k=count)
    colors = random.sample(list(SPRITE_COLORS), k=count)
    special_idx = random.randrange(count)

    order = list(range(count))
    order.sort(key=lambda idx: idx != special_idx)
    candidates: list[dict] = []

    for idx in order:
        direction = directions[idx]
        color = colors[idx]
        force_opposite = idx == special_idx
        force_anti_legacy = force_opposite or random.random() < 0.7
        placed = None
        for _ in range(CANDIDATE_RETRIES):
            candidate = _sample_candidate(
                H,
                W,
                direction,
                color,
                force_opposite_nearest=force_opposite,
                force_anti_legacy=force_anti_legacy,
            )
            if candidate is None:
                continue
            if _compatible(candidates, candidate):
                placed = candidate
                break
        if placed is None:
            return None
        candidates.append(placed)

    inp = _render_input((H, W), candidates)
    if not _valid_instance(inp, candidates):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 10 input")
