from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
TILE_COLORS = (ORANGE, MAROON)
SOURCE_COLORS = (1, 2, 3, 4, 6, 8)

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
TARGET_DIRECTIONS = ("up", "down", "left", "right")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        arr = coords
    else:
        arr = np.array(sorted(coords), dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    r1 = int(arr[:, 0].max())
    c1 = int(arr[:, 1].max())
    return r0, c0, r1, c1


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    deltas = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not mask[r, c]:
                continue
            visited[r, c] = True
            comp = []
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _tile_bbox_from_anchor(anchor: tuple[int, int], direction: str) -> tuple[int, int, int, int]:
    r, c = anchor
    if direction == "up":
        return r - 3, c - 1, r - 1, c + 1
    if direction == "down":
        return r + 1, c - 1, r + 3, c + 1
    if direction == "left":
        return r - 1, c - 3, r + 1, c - 1
    if direction == "right":
        return r - 1, c + 1, r + 1, c + 3
    raise AssertionError(direction)


def _group_cells(anchor: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _tile_bbox_from_anchor(anchor, direction)
    cells = {(anchor[0], anchor[1])}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cells.add((r, c))
    return cells


def _group_box(anchor: tuple[int, int], direction: str) -> tuple[int, int, int, int]:
    return _bbox(_group_cells(anchor, direction))


def _copy_cells(
    anchor: tuple[int, int],
    pixels: list[tuple[int, int, int]],
    source_anchor: tuple[int, int],
    flip: bool,
) -> list[tuple[int, int, int]]:
    sr, sc = source_anchor
    tr, tc = anchor
    out = []
    for r, c, color in pixels:
        dr = r - sr
        dc = c - sc
        if flip:
            dc = -dc
        out.append((tr + dr, tc + dc, color))
    return out


def _parse_target_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    grey_cells = [(r, c) for r, c in coords if int(grid[r, c]) == GREY]
    assert len(grey_cells) == 1, "each control object must contain one gray anchor"
    anchor = grey_cells[0]

    tile_cells = [(r, c) for r, c in coords if int(grid[r, c]) in TILE_COLORS]
    assert len(tile_cells) == 9, "expected a full 3x3 control tile"
    r0, c0, r1, c1 = _bbox(tile_cells)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "control tile must be 3x3"
    assert set(tile_cells) == {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    tile = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    adjacent = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in NEIGHBORS4 if (anchor[0] + dr, anchor[1] + dc) in set(tile_cells)]
    assert adjacent, "target anchor must touch its control tile"

    return {
        "anchor": anchor,
        "tile": tile,
        "coords": coords,
        "box": (r0, c0, r1, c1),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _components(g != BG, connectivity=8)
    source = None
    targets = []
    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        if GREY not in colors:
            continue
        non_grey = colors - {GREY}
        if any(color not in TILE_COLORS for color in non_grey):
            assert source is None, "expected exactly one multicolor source sprite"
            grey_cells = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
            assert len(grey_cells) == 1, "source sprite must contain one gray anchor"
            source = {
                "coords": comp,
                "anchor": grey_cells[0],
                "pixels": [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) != GREY],
                "box": _bbox(comp),
            }
        else:
            targets.append(_parse_target_component(g, comp))

    assert source is not None, "missing source sprite"
    assert len(targets) >= 1, "missing target control groups"
    return {"source": source, "targets": targets}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the anchored multicolor sprite to each gray target, flipping only when the adjacent 3x3 control tile is diagonally asymmetric.

    Input structure
    - The background is black.
    - One 8-connected multicolor source sprite contains exactly one gray anchor pixel.
    - Elsewhere, each target is a gray pixel attached to a separate 3x3 orange/maroon tile.

    Rule
    - Identify the source sprite and extract every non-gray source pixel relative to its gray anchor.
    - For each target, inspect the adjacent 3x3 tile: if the tile equals its transpose, keep the source offsets; otherwise negate the horizontal offsets to place a left-right mirrored copy.
    - Erase every target anchor and every control tile to black, then paint the corresponding translated copy while leaving the original source sprite unchanged.

    Key insight
    - The nearby tile is a control signal, not output content: its main-diagonal symmetry decides whether the anchored sprite copy keeps its orientation or flips horizontally.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    source = scene["source"]

    out = g.copy()
    for target in scene["targets"]:
        for r, c in target["coords"]:
            out[r, c] = BG

    for target in scene["targets"]:
        flip = not np.array_equal(target["tile"], target["tile"].T)
        for r, c, color in _copy_cells(target["anchor"], source["pixels"], source["anchor"], flip):
            assert _in_bounds(out.shape, r, c), "placed copy must stay on the grid"
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (16, 23)
SOURCE_HEIGHT_RANGE = (3, 6)
SOURCE_WIDTH_RANGE = (3, 6)
TARGET_COUNT_OPTIONS = (2, 3)
INPUT_GAP = 1
OUTPUT_GAP = 1
MAX_ATTEMPTS = 400


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    filled = [start]

    while len(filled) < area:
        frontier = set()
        for r, c in filled:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        mask[cell] = True
        filled.append(cell)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not (rows[0] and rows[-1] and cols[0] and cols[-1]):
        return None
    if np.array_equal(mask, np.fliplr(mask)):
        return None
    return mask


def _generate_source_sprite() -> tuple[np.ndarray, tuple[int, int]] | None:
    h = random.randrange(*SOURCE_HEIGHT_RANGE)
    w = random.randrange(*SOURCE_WIDTH_RANGE)
    area_lo = max(6, (h * w + 1) // 2)
    area_hi = min(h * w - 1, h * w - max(2, h + w - 5))
    if area_lo > area_hi:
        return None

    for _ in range(80):
        area = random.randint(area_lo, area_hi)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        anchor_choices = []
        center_col = w // 2 if w % 2 == 1 else None
        for r, c in coords:
            if center_col is not None and c == center_col:
                continue
            anchor_choices.append((r, c))
        if not anchor_choices:
            continue

        anchor = random.choice(anchor_choices)
        non_anchor = [rc for rc in coords if rc != anchor]
        color_count = random.randint(2, min(4, len(non_anchor)))
        palette = random.sample(SOURCE_COLORS, color_count)
        colors = [random.choice(palette) for _ in non_anchor]

        sprite = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(non_anchor, colors):
            sprite[r, c] = color
        sprite[anchor] = GREY

        artwork = sprite.copy()
        artwork[artwork == GREY] = BG
        if np.array_equal(artwork, np.fliplr(artwork)):
            continue
        if len({int(v) for v in artwork.flat if int(v) != BG}) < 2:
            continue
        return sprite, anchor
    return None


def _make_symmetric_tile() -> np.ndarray:
    while True:
        tile = np.full((3, 3), ORANGE, dtype=int)
        for r in range(3):
            for c in range(r, 3):
                color = random.choice(TILE_COLORS)
                tile[r, c] = color
                tile[c, r] = color
        if len({int(v) for v in tile.flat}) == 2:
            return tile


def _make_asymmetric_tile() -> np.ndarray:
    while True:
        tile = np.array([[random.choice(TILE_COLORS) for _ in range(3)] for _ in range(3)], dtype=int)
        if len({int(v) for v in tile.flat}) < 2:
            continue
        if not np.array_equal(tile, tile.T):
            return tile


def _sprite_global_cells(sprite: np.ndarray, origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in np.argwhere(sprite != BG)}


def _render_expected_output(
    shape: tuple[int, int],
    sprite: np.ndarray,
    source_origin: tuple[int, int],
    source_anchor: tuple[int, int],
    placements: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    sr0, sc0 = source_origin
    for r, c in np.argwhere(sprite != BG):
        out[sr0 + int(r), sc0 + int(c)] = int(sprite[int(r), int(c)])

    pixels = [(int(r), int(c), int(sprite[int(r), int(c)])) for r, c in np.argwhere((sprite != BG) & (sprite != GREY))]
    for placement in placements:
        for r, c, color in _copy_cells(placement["anchor"], pixels, source_anchor, placement["flip"]):
            out[r, c] = color
    return out


def _build_input(
    shape: tuple[int, int],
    sprite: np.ndarray,
    source_origin: tuple[int, int],
    placements: list[dict],
) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    sr0, sc0 = source_origin
    for r, c in np.argwhere(sprite != BG):
        g[sr0 + int(r), sc0 + int(c)] = int(sprite[int(r), int(c)])

    for placement in placements:
        ar, ac = placement["anchor"]
        g[ar, ac] = GREY
        r0, c0, r1, c1 = _tile_bbox_from_anchor(placement["anchor"], placement["direction"])
        g[r0 : r1 + 1, c0 : c1 + 1] = placement["tile"]
    return g


def _pick_source_origin(shape: tuple[int, int], sprite: np.ndarray) -> tuple[int, int]:
    h, w = sprite.shape
    return random.randrange(shape[0] - h + 1), random.randrange(shape[1] - w + 1)


def _place_targets(
    shape: tuple[int, int],
    sprite: np.ndarray,
    source_origin: tuple[int, int],
    source_anchor: tuple[int, int],
    branch_flags: list[bool],
) -> list[dict] | None:
    source_cells = _sprite_global_cells(sprite, source_origin)
    source_box = _bbox(source_cells)
    source_pixels = [(int(r), int(c), int(sprite[int(r), int(c)])) for r, c in np.argwhere((sprite != BG) & (sprite != GREY))]

    input_boxes = [source_box]
    input_cells = set(source_cells)
    output_boxes = [source_box]
    output_cells = set(source_cells)
    placements: list[dict] = []

    anchors = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(anchors)

    for symmetric in branch_flags:
        placed = False
        for anchor in anchors:
            dirs = list(TARGET_DIRECTIONS)
            random.shuffle(dirs)
            copy = _copy_cells(anchor, source_pixels, source_anchor, flip=not symmetric)
            copy_coords = {(r, c) for r, c, _ in copy}
            if any(not _in_bounds(shape, r, c) for r, c in copy_coords):
                continue
            copy_box = _bbox(copy_coords)
            if copy_coords & output_cells:
                continue
            if any(not _boxes_separated(copy_box, box, OUTPUT_GAP) for box in output_boxes):
                continue

            for direction in dirs:
                group_box = _group_box(anchor, direction)
                if not _in_bounds(shape, group_box[0], group_box[1]) or not _in_bounds(shape, group_box[2], group_box[3]):
                    continue
                group_cells = _group_cells(anchor, direction)
                if group_cells & input_cells:
                    continue
                if any(not _boxes_separated(group_box, box, INPUT_GAP) for box in input_boxes):
                    continue

                tile = _make_symmetric_tile() if symmetric else _make_asymmetric_tile()
                placements.append(
                    {
                        "anchor": anchor,
                        "direction": direction,
                        "tile": tile,
                        "flip": not symmetric,
                    }
                )
                input_boxes.append(group_box)
                input_cells |= group_cells
                output_boxes.append(copy_box)
                output_cells |= copy_coords
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return placements


def _validate_generation(inp: np.ndarray, expected: np.ndarray, placements: list[dict]) -> bool:
    try:
        scene = _parse_scene(inp)
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(scene["targets"]) != len(placements):
        return False
    flips = [placement["flip"] for placement in placements]
    if not any(flips) or all(flips):
        return False
    return not np.array_equal(inp, expected)


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    shape = (
        random.randrange(*GRID_SIDE_RANGE),
        random.randrange(*GRID_SIDE_RANGE),
    )
    made = _generate_source_sprite()
    if made is None:
        return None
    sprite, source_anchor = made

    target_count = random.choice(TARGET_COUNT_OPTIONS)
    branch_flags = [True, False]
    while len(branch_flags) < target_count:
        branch_flags.append(bool(random.getrandbits(1)))
    random.shuffle(branch_flags)

    for _ in range(80):
        source_origin = _pick_source_origin(shape, sprite)
        placements = _place_targets(shape, sprite, source_origin, source_anchor, branch_flags)
        if placements is None:
            continue
        inp = _build_input(shape, sprite, source_origin, placements)
        expected = _render_expected_output(shape, sprite, source_origin, source_anchor, placements)
        if _validate_generation(inp, expected, placements):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        made = _attempt_generate()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid input for puzzle 4")
