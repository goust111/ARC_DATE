import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
KEY_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

KEY_COUNT_RANGE = (2, 4)
DOT_COUNT_RANGE = (1, 3)
SPRITE_WIDTH_RANGE = (2, 4)
SPRITE_HEIGHT_RANGE = (1, 3)
MAX_OFFSET_RANGE = (3, 5)
LEGEND_SLACK_RANGE = (0, 2)
GRID_MAX_SIDE = 30
DISTRACTOR_PROB = 0.55
DISTRACTOR_COUNT_RANGE = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    steps = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    target = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        target[:] = patch
        return
    mask = patch != int(transparent)
    target[mask] = patch[mask]


def _is_frame(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 4:
        return False
    sub = _crop(grid, box)
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(sub[border] == int(color)) and np.all(sub[~border] != int(color)))


def _find_legend_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _components(grid == FRAME, connectivity=8):
        box = _bbox(comp)
        if _is_frame(grid, FRAME, box):
            area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            candidates.append((area, box))
    assert candidates, "expected a grey legend frame"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _parse_legend(
    grid: np.ndarray, legend_box: tuple[int, int, int, int], bg: int
) -> list[tuple[int, int, list[int]]]:
    r0, c0, r1, c1 = legend_box
    inner = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    colored = (inner != int(bg)) & (inner != FRAME)
    assert colored.any(), "legend interior must contain key rows"

    key_col = None
    for cc in range(inner.shape[1]):
        if colored[:, cc].any():
            key_col = cc
            break
    assert key_col is not None

    rows: list[tuple[int, int, list[int]]] = []
    seen_colors = set()
    for rr in range(inner.shape[0]):
        cols = np.flatnonzero(colored[rr])
        if cols.size == 0:
            continue
        assert int(cols[0]) == int(key_col), "key pixel must be the leftmost colored cell of its row"
        color = int(inner[rr, key_col])
        assert color not in seen_colors, "key colors must be unique"
        seen_colors.add(color)
        assert np.all(inner[rr, cols] == color), "legend row must use one color only"
        offsets = [int(cc - key_col) for cc in cols[1:]]
        rows.append((int(rr + r0 + 1), color, offsets))

    rows.sort(key=lambda item: item[0])
    return rows


def _find_single_sprite(
    grid: np.ndarray, color: int, legend_box: tuple[int, int, int, int]
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    mask = grid == int(color)
    r0, c0, r1, c1 = legend_box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(mask, connectivity=8)
    assert len(comps) == 1, "each key color must map to exactly one outside sprite"
    box = _bbox(comps[0])
    return box, _crop(grid, box), comps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each legend-row dot pattern as scaled placement offsets and stamp sprite copies on the sprite's own row.

    Input structure:
    - A grey rectangular legend frame sits in one corner; inside it, the leftmost interior column holds one key pixel per row.
    - On each key row, extra same-colored dots appear to the right at irregular horizontal offsets.
    - Outside the legend, each key color appears in exactly one small sprite, with empty space to its right.

    Rule:
    - For each key row, measure every legend dot's column offset d from that row's key pixel.
    - Find the outside sprite of the same color and let w be that sprite bbox width.
    - Copy the sprite on its original rows so that each copy's left edge is shifted right by d*w columns from the original sprite's left edge.
    - Keep the original legend and original sprites unchanged.

    Key insight:
    - The legend dots are not a count alone; their irregular spacing becomes a scaled horizontal blueprint, and the scale unit is the matched sprite's own width.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_box = _find_legend_box(g)
    instructions = _parse_legend(g, legend_box, bg)

    out = g.copy()
    for _legend_row, color, offsets in instructions:
        if not offsets:
            continue
        box, sprite, _coords = _find_single_sprite(g, color, legend_box)
        sr0, sc0, _sr1, _sc1 = box
        width = int(sprite.shape[1])
        for offset in offsets:
            dst_c = sc0 + int(offset) * width
            assert 0 <= dst_c <= g.shape[1] - width
            _paste(out, sprite, sr0, dst_c, transparent=bg)
    return out


def _sample_offsets() -> list[int]:
    n = random.randint(*DOT_COUNT_RANGE)
    cap = random.randint(*MAX_OFFSET_RANGE)
    options = list(range(1, cap + 1))
    chosen = []
    random.shuffle(options)
    for value in options:
        if all(abs(value - prev) >= 2 for prev in chosen):
            chosen.append(value)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        return _sample_offsets()
    return sorted(chosen)


def _sample_sprite_shape(h: int, w: int) -> np.ndarray:
    min_area = max(2, (h * w + 1) // 2)
    max_area = h * w
    for _ in range(200):
        area = random.randint(min_area, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = [start]
        while len(cells) < area and frontier:
            rr, cc = random.choice(frontier)
            neigh = []
            for dr, dc in _neighbors(4):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    neigh.append((nr, nc))
            if not neigh:
                frontier.remove((rr, cc))
                continue
            nxt = random.choice(neigh)
            cells.add(nxt)
            frontier.append(nxt)
        coords = np.array(sorted(cells), dtype=int)
        box = _bbox(coords)
        if box == (0, 0, h - 1, w - 1):
            mask = np.zeros((h, w), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            return mask
    return np.ones((h, w), dtype=bool)


def _make_sprite(color: int, width: int, bg: int) -> np.ndarray:
    height = random.randint(*SPRITE_HEIGHT_RANGE)
    mask = _sample_sprite_shape(height, width)
    sprite = np.full((height, width), int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _mark_reserved(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_free(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _corner_origin(h: int, w: int, box_h: int, box_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - box_w
    if corner == "bl":
        return h - box_h, 0
    return h - box_h, w - box_w


def _legend_patch(key_rows: list[tuple[int, list[int]]], width: int, bg: int) -> np.ndarray:
    patch = np.full((2 * len(key_rows) + 1, width), int(bg), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    for i, (color, offsets) in enumerate(key_rows):
        rr = 1 + 2 * i
        patch[rr, 1] = int(color)
        for d in offsets:
            patch[rr, 1 + int(d)] = int(color)
    return patch


def _place_group(
    inp: np.ndarray,
    expected: np.ndarray,
    reserved: np.ndarray,
    sprite: np.ndarray,
    offsets: list[int],
    bg: int,
) -> tuple[int, int] | None:
    h, w = sprite.shape
    H, W = inp.shape
    max_offset = max(offsets) if offsets else 0
    span_w = (max_offset + 1) * w
    candidates = [
        (r, c)
        for r in range(H - h + 1)
        for c in range(W - span_w + 1)
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        group_box = (r0, c0, r0 + h - 1, c0 + span_w - 1)
        if not _box_free(reserved, group_box, pad=1):
            continue
        _paste(inp, sprite, r0, c0, transparent=bg)
        _paste(expected, sprite, r0, c0, transparent=bg)
        for d in offsets:
            _paste(expected, sprite, r0, c0 + int(d) * w, transparent=bg)
        _mark_reserved(reserved, group_box, pad=1)
        return r0, c0
    return None


def _place_single_sprite(
    grid: np.ndarray,
    reserved: np.ndarray,
    sprite: np.ndarray,
    bg: int,
) -> tuple[int, int] | None:
    h, w = sprite.shape
    H, W = grid.shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _box_free(reserved, box, pad=1):
            continue
        _paste(grid, sprite, r0, c0, transparent=bg)
        _mark_reserved(reserved, box, pad=1)
        return r0, c0
    return None


def _inspect_key_sprites(
    inp: np.ndarray, legend_box: tuple[int, int, int, int], key_colors: list[int]
) -> dict[int, tuple[int, int, int, int]]:
    boxes = {}
    for color in key_colors:
        box, _sprite, _coords = _find_single_sprite(inp, color, legend_box)
        boxes[color] = box
    return boxes


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    key_info: list[tuple[int, list[int]]],
    legend_box: tuple[int, int, int, int],
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    widths = []
    sprite_boxes = _inspect_key_sprites(inp, legend_box, [color for color, _ in key_info])
    for color, offsets in key_info:
        box = sprite_boxes[color]
        widths.append(box[3] - box[1] + 1)
        if not offsets:
            return False
        if len(offsets) != len(set(offsets)) or min(offsets) <= 0:
            return False
        for a, b in zip(offsets, offsets[1:]):
            if b - a < 2:
                return False
    if len(set(widths)) < 2:
        return False
    if not any(len(offsets) >= 2 and any(b - a > 1 for a, b in zip(offsets, offsets[1:])) for _, offsets in key_info):
        return False

    bg = _get_bg(inp)
    r0, c0, r1, c1 = legend_box
    for color, offsets in key_info:
        box = sprite_boxes[color]
        sr0, sc0, sr1, sc1 = box
        sprite = _crop(inp, box)
        width = sc1 - sc0 + 1
        mask = inp == int(color)
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        if len(_components(mask, connectivity=8)) != 1:
            return False
        for d in offsets:
            left = sc0 + d * width
            patch_in = inp[sr0 : sr1 + 1, left : left + width]
            if np.any((patch_in != int(bg)) & (sprite != int(bg))):
                return False
            patch_out = expected[sr0 : sr1 + 1, left : left + width]
            if not np.array_equal((patch_out != bg), (sprite != bg)):
                return False
            if not np.array_equal(patch_out[sprite != bg], sprite[sprite != bg]):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    key_count = random.randint(*KEY_COUNT_RANGE)
    key_colors = random.sample(KEY_PALETTE, key_count)
    offset_lists = [_sample_offsets() for _ in range(key_count)]

    widths = [random.randint(*SPRITE_WIDTH_RANGE) for _ in range(key_count)]
    if len(set(widths)) < 2:
        return None

    max_offset = max(max(offsets) for offsets in offset_lists)
    legend_w = max_offset + 3 + random.randint(*LEGEND_SLACK_RANGE)
    legend_h = 2 * key_count + 1
    max_sprite_w = max(widths)
    min_h = max(18, legend_h + 3 * key_count + 4)
    min_w = max(22, legend_w + 4, (max_offset + 1) * max_sprite_w + 2)
    if min_h > GRID_MAX_SIDE or min_w > GRID_MAX_SIDE:
        return None

    H = random.randint(min_h, GRID_MAX_SIDE)
    W = random.randint(min_w, GRID_MAX_SIDE)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_r, legend_c = _corner_origin(H, W, legend_h, legend_w, corner)
    legend_box = (legend_r, legend_c, legend_r + legend_h - 1, legend_c + legend_w - 1)

    inp = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    legend = _legend_patch(list(zip(key_colors, offset_lists)), legend_w, BG)
    _paste(inp, legend, legend_r, legend_c)
    _paste(expected, legend, legend_r, legend_c)
    _mark_reserved(reserved, legend_box, pad=1)

    key_info = list(zip(key_colors, offset_lists))
    random.shuffle(key_info)
    for color, offsets in key_info:
        width = widths[key_colors.index(color)]
        sprite = _make_sprite(color, width, BG)
        loc = _place_group(inp, expected, reserved, sprite, offsets, BG)
        if loc is None:
            return None

    remaining = [c for c in KEY_PALETTE if c not in key_colors and c != FRAME]
    if remaining and random.random() < DISTRACTOR_PROB:
        distractor_n = random.randint(*DISTRACTOR_COUNT_RANGE)
        random.shuffle(remaining)
        for color in remaining[:distractor_n]:
            sprite = _make_sprite(color, random.randint(*SPRITE_WIDTH_RANGE), BG)
            loc = _place_single_sprite(inp, reserved, sprite, BG)
            if loc is None:
                return None
            _paste(expected, sprite, loc[0], loc[1], transparent=BG)

    if not _validate_case(inp, expected, list(zip(key_colors, offset_lists)), legend_box):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-blueprint input")
