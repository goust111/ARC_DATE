import random
from collections import deque

import numpy as np


BG = 0
BODY = 5
FILL = 9
PAIR_COLORS = (1, 2, 3, 4)
DIRECTIONS = ("up", "right", "down", "left")
DIR_TO_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rotate_offsets(offsets: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    turns %= 4
    out = list(offsets)
    for _ in range(turns):
        out = [(c, -r) for r, c in out]
    return tuple((int(r), int(c)) for r, c in out)


UP_BODY_VARIANTS = (
    ((1, 0), (2, 0), (1, -1), (1, 1), (0, -1), (0, 1)),
    ((1, 0), (2, 0), (2, -1), (2, 1), (1, -1), (1, 1), (-1, 0)),
    ((1, 0), (2, 0), (1, -1), (1, 1), (2, -1), (0, -1), (0, 1)),
    ((1, 0), (2, 0), (3, 0), (1, -1), (1, 1), (0, -1)),
)


def _body_offsets(direction: str, variant_index: int) -> tuple[tuple[int, int], ...]:
    return _rotate_offsets(UP_BODY_VARIANTS[variant_index % len(UP_BODY_VARIANTS)], DIR_TO_TURNS[direction])


def _infer_direction(gray_coords: np.ndarray, marker: tuple[int, int]) -> str:
    mr, mc = marker
    dr = float(np.mean(gray_coords[:, 0]) - mr)
    dc = float(np.mean(gray_coords[:, 1]) - mc)
    assert abs(dr) > 1e-6 or abs(dc) > 1e-6
    if abs(dr) > abs(dc):
        return "up" if dr > 0 else "down"
    return "left" if dc > 0 else "right"


def _is_square_block(block: np.ndarray, color: int, bg: int) -> bool:
    h, w = block.shape
    if h != w or h not in (4, 5):
        return False
    if not np.all(block[0, :] == color) or not np.all(block[-1, :] == color):
        return False
    if not np.all(block[:, 0] == color) or not np.all(block[:, -1] == color):
        return False
    inner = block[1:-1, 1:-1]
    return np.all((inner == bg) | (inner == FILL))


def _find_blocks(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    blocks: dict[int, np.ndarray] = {}
    for color in PAIR_COLORS:
        comps = components_mask(grid == color, connectivity=8)
        for comp in comps:
            if comp.shape[0] == 1:
                continue
            patch = crop_bbox(grid, bbox_coords(comp))
            if _is_square_block(patch, color, bg):
                assert color not in blocks
                blocks[color] = patch
    return blocks


def _sprite_component(grid: np.ndarray, marker: tuple[int, int], color: int) -> dict:
    h, w = grid.shape
    mr, mc = marker
    allowed = {BODY, color}
    q = deque([(mr, mc)])
    seen = {(mr, mc)}
    cells: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors(r, c, 8):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) in allowed:
                seen.add((nr, nc))
                q.append((nr, nc))
    coords = np.array(cells, dtype=int)
    gray_coords = coords[grid[coords[:, 0], coords[:, 1]] == BODY]
    color_coords = coords[grid[coords[:, 0], coords[:, 1]] == color]
    assert gray_coords.shape[0] in (5, 6, 7)
    assert color_coords.shape[0] == 1
    direction = _infer_direction(gray_coords, marker)
    return {"color": color, "marker": marker, "gray": gray_coords, "direction": direction}


def _find_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for color in PAIR_COLORS:
        comps = components_mask(grid == color, connectivity=8)
        markers = [comp for comp in comps if comp.shape[0] == 1]
        if not markers:
            continue
        assert len(markers) == 1
        marker = tuple(int(x) for x in markers[0][0])
        comp = _sprite_component(grid, marker, color)
        sprites.append(comp)
    return sprites


def _assemble_output(blocks: dict[int, np.ndarray], sprites: list[dict]) -> np.ndarray:
    pieces: list[tuple[int, int, np.ndarray]] = []
    for sprite in sprites:
        turns = DIR_TO_TURNS[sprite["direction"]]
        block = np.rot90(blocks[sprite["color"]], -turns)
        mr, mc = sprite["marker"]
        pieces.append((mc, mr, block))
    pieces.sort(key=lambda item: (item[0], item[1]))
    return np.concatenate([patch for _, _, patch in pieces], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each paired square block by the angle encoded in its directional sprite, then concatenate the rotated blocks.

    Input structure:
    - The background is black and the foreground contains 2 to 4 color-paired objects.
    - Each pair has one directional sprite made of an 8-connected gray body plus one embedded colored marker pixel.
    - Each pair also has one same-colored 4x4 or 5x5 square block whose 1-pixel border uses the marker color and whose interior is a black/maroon asymmetric pattern.

    Rule:
    - Identify each marker color's singleton sprite marker and same-colored square border block.
    - For the sprite, compute the centroid of only its gray body pixels and compare that centroid to the marker position; the dominant signed offset axis gives the facing direction opposite the body bias.
    - Map facing up/right/down/left to clockwise rotations of 0/90/180/270 degrees, rotate the paired square block by that amount, then sort pairs by marker position using column-first order and horizontally concatenate the rotated blocks.

    Key insight:
    - The marker color only links objects; the actual rotation angle is hidden in the gray body's centroid bias relative to the marker, so solving requires geometric direction inference before applying rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    blocks = _find_blocks(g, bg)
    sprites = _find_sprites(g, bg)
    assert 2 <= len(sprites) <= 4
    assert set(blocks) == {sprite["color"] for sprite in sprites}
    side_lengths = {patch.shape[0] for patch in blocks.values()}
    assert len(side_lengths) == 1
    return _assemble_output(blocks, sprites)


def _all_interiors(side: int) -> list[np.ndarray]:
    inner = side - 2
    out: list[np.ndarray] = []
    for bits in range(1, 1 << (inner * inner)):
        arr = np.zeros((inner, inner), dtype=int)
        for idx in range(inner * inner):
            if bits & (1 << idx):
                arr[idx // inner, idx % inner] = FILL
        if arr.sum() == 0 or arr.sum() == FILL * inner * inner:
            continue
        if any(np.array_equal(arr, np.rot90(arr, k)) for k in (1, 2, 3)):
            continue
        out.append(arr)
    return out


INTERIOR_PATTERNS = {4: _all_interiors(4), 5: _all_interiors(5)}


def _make_block(color: int, side: int, inner: np.ndarray) -> np.ndarray:
    block = np.full((side, side), color, dtype=int)
    block[1:-1, 1:-1] = inner
    return block


def _pair_metadata(color: int, direction: str, side: int) -> dict:
    variant_index = random.randrange(len(UP_BODY_VARIANTS))
    offsets = _body_offsets(direction, variant_index)
    rows = [0] + [r for r, _ in offsets]
    cols = [0] + [c for _, c in offsets]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    gap = random.randint(2, 4)
    return {
        "color": color,
        "direction": direction,
        "offsets": offsets,
        "min_r": min_r,
        "max_r": max_r,
        "min_c": min_c,
        "max_c": max_c,
        "sprite_h": max_r - min_r + 1,
        "sprite_w": max_c - min_c + 1,
        "side": side,
        "gap": gap,
    }


def _place_horizontal(grid: np.ndarray, pairs: list[dict]) -> None:
    h, w = grid.shape
    current_top = 1
    for pair in pairs:
        side = pair["side"]
        block_top_rel = -side // 2
        block_bottom_rel = block_top_rel + side - 1
        rel_top = min(pair["min_r"], block_top_rel)
        rel_bottom = max(pair["max_r"], block_bottom_rel)
        anchor_r = current_top - rel_top
        total_w = pair["sprite_w"] + pair["gap"] + side
        start = random.randint(1, w - total_w - 1)
        sprite_left_first = random.choice((True, False))
        if sprite_left_first:
            sprite_left = start
            block_left = sprite_left + pair["sprite_w"] + pair["gap"]
        else:
            block_left = start
            sprite_left = block_left + side + pair["gap"]
        marker_r = anchor_r
        marker_c = sprite_left - pair["min_c"]
        block_top = anchor_r + block_top_rel
        block = pair["block"]
        grid[block_top : block_top + side, block_left : block_left + side] = block
        grid[marker_r, marker_c] = pair["color"]
        for dr, dc in pair["offsets"]:
            grid[marker_r + dr, marker_c + dc] = BODY
        pair["marker"] = (marker_r, marker_c)
        current_top = anchor_r + rel_bottom + random.randint(2, 3)
    assert current_top < h


def _place_vertical(grid: np.ndarray, pairs: list[dict]) -> None:
    h, w = grid.shape
    current_left = 1
    for pair in pairs:
        side = pair["side"]
        block_left_rel = -side // 2
        block_right_rel = block_left_rel + side - 1
        rel_left = min(pair["min_c"], block_left_rel)
        rel_right = max(pair["max_c"], block_right_rel)
        anchor_c = current_left - rel_left
        total_h = pair["sprite_h"] + pair["gap"] + side
        start = random.randint(1, h - total_h - 1)
        sprite_top_first = random.choice((True, False))
        if sprite_top_first:
            sprite_top = start
            block_top = sprite_top + pair["sprite_h"] + pair["gap"]
        else:
            block_top = start
            sprite_top = block_top + side + pair["gap"]
        marker_r = sprite_top - pair["min_r"]
        marker_c = anchor_c
        block_left = anchor_c + block_left_rel
        block = pair["block"]
        grid[block_top : block_top + side, block_left : block_left + side] = block
        grid[marker_r, marker_c] = pair["color"]
        for dr, dc in pair["offsets"]:
            grid[marker_r + dr, marker_c + dc] = BODY
        pair["marker"] = (marker_r, marker_c)
        current_left = anchor_c + rel_right + random.randint(2, 3)
    assert current_left < w


def _scene_shape(pairs: list[dict], axis: str) -> tuple[int, int] | None:
    side = pairs[0]["side"]
    if axis == "horizontal":
        heights = []
        widths = []
        for pair in pairs:
            block_top_rel = -side // 2
            block_bottom_rel = block_top_rel + side - 1
            rel_top = min(pair["min_r"], block_top_rel)
            rel_bottom = max(pair["max_r"], block_bottom_rel)
            heights.append(rel_bottom - rel_top + 1)
            widths.append(pair["sprite_w"] + pair["gap"] + side)
        h = 2 + sum(heights) + 2 * (len(pairs) - 1) + 2
        w = max(widths) + random.randint(4, 8)
    else:
        widths = []
        heights = []
        for pair in pairs:
            block_left_rel = -side // 2
            block_right_rel = block_left_rel + side - 1
            rel_left = min(pair["min_c"], block_left_rel)
            rel_right = max(pair["max_c"], block_right_rel)
            widths.append(rel_right - rel_left + 1)
            heights.append(pair["sprite_h"] + pair["gap"] + side)
        h = max(heights) + random.randint(4, 8)
        w = 2 + sum(widths) + 2 * (len(pairs) - 1) + 2
    if h > 30 or w > 30:
        return None
    return h, w


def _expected_from_pairs(pairs: list[dict]) -> np.ndarray:
    blocks = {pair["color"]: pair["block"] for pair in pairs}
    sprites = [{"color": pair["color"], "marker": pair["marker"], "direction": pair["direction"]} for pair in pairs]
    return _assemble_output(blocks, sprites)


def _color_sorted_output(pairs: list[dict]) -> np.ndarray:
    ordered = sorted(pairs, key=lambda pair: pair["color"])
    pieces = [np.rot90(pair["block"], -DIR_TO_TURNS[pair["direction"]]) for pair in ordered]
    return np.concatenate(pieces, axis=1)


def _validate_pair_semantics(grid: np.ndarray, pairs: list[dict]) -> bool:
    expected = _expected_from_pairs(pairs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    rotated = [np.rot90(pair["block"], -DIR_TO_TURNS[pair["direction"]]) for pair in pairs]
    if len({patch.tobytes() for patch in rotated}) != len(rotated):
        return False
    if np.array_equal(expected, np.concatenate([pair["block"] for pair in sorted(pairs, key=lambda p: (p["marker"][1], p["marker"][0]))], axis=1)):
        return False
    if np.array_equal(expected, _color_sorted_output(pairs)):
        return False
    reversed_pieces = [
        np.rot90(pair["block"], -((DIR_TO_TURNS[pair["direction"]] + 2) % 4))
        for pair in sorted(pairs, key=lambda p: (p["marker"][1], p["marker"][0]))
    ]
    if np.array_equal(expected, np.concatenate(reversed_pieces, axis=1)):
        return False
    if not any(pair["direction"] != "up" for pair in pairs):
        return False
    for pair in pairs:
        if _infer_direction(
            np.array([(pair["marker"][0] + dr, pair["marker"][1] + dc) for dr, dc in pair["offsets"]], dtype=int),
            pair["marker"],
        ) != pair["direction"]:
            return False
    return True


def _choose_pair_count() -> int:
    return random.choices([2, 3, 4], weights=[2, 3, 5], k=1)[0]


def _sample_inners(side: int, count: int) -> list[np.ndarray] | None:
    for _ in range(100):
        inners = random.sample(INTERIOR_PATTERNS[side], count)
        blocks = [_make_block(7, side, inner) for inner in inners]
        if len({np.rot90(block, -turns).tobytes() for block in blocks for turns in range(4)}) < count * 4:
            continue
        return inners
    return None


def _generate_scene() -> np.ndarray | None:
    count = _choose_pair_count()
    side = random.choice((4, 5))
    axis = random.choice(("horizontal", "vertical"))
    colors = random.sample(list(PAIR_COLORS), count)
    directions = random.sample(list(DIRECTIONS), count)
    inners = _sample_inners(side, count)
    if inners is None:
        return None
    pairs = []
    for color, direction, inner in zip(colors, directions, inners):
        pair = _pair_metadata(color, direction, side)
        pair["block"] = _make_block(color, side, inner)
        pairs.append(pair)
    shape = _scene_shape(pairs, axis)
    if shape is None:
        return None
    grid = np.zeros(shape, dtype=int)
    if axis == "horizontal":
        _place_horizontal(grid, pairs)
    else:
        _place_vertical(grid, pairs)
    if not _validate_pair_semantics(grid, pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _generate_scene()
        if grid is not None and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid directional-sprite rotation task")
