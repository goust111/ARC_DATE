from __future__ import annotations

from collections import defaultdict, deque
import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

BLOCK_SIZE = 3
MARKER_OFFSETS = [
    (-6, 0),
    (-5, 0),
    (5, 0),
    (6, 0),
    (0, -6),
    (0, -5),
    (0, 5),
    (0, 6),
    (-6, -6),
    (-6, 6),
    (6, -6),
    (6, 6),
    (-5, -5),
    (-5, 5),
    (5, -5),
    (5, 5),
]
MAX_GENERATION_ATTEMPTS = 400


GLYPH_ITEMS = [
    (
        "plus",
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        RED,
    ),
    (
        "hook",
        np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1],
            ],
            dtype=int,
        ),
        BLUE,
    ),
    (
        "diag",
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=int,
        ),
        GREEN,
    ),
]

GLYPH_TO_COLOR = {
    tuple(tuple(int(v) for v in row) for row in glyph.tolist()): color
    for _name, glyph, color in GLYPH_ITEMS
}


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _dist2(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ar, ac = _center(box_a)
    br, bc = _center(box_b)
    return (ar - br) ** 2 + (ac - bc) ** 2


def _mask_signature(region: np.ndarray, color: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v == color) for v in row) for row in region.tolist())


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    blocks = []
    for coords in _find_components(grid == YELLOW):
        box = _bbox(coords)
        assert _bbox_h(box) == BLOCK_SIZE and _bbox_w(box) == BLOCK_SIZE
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((region == BLACK) | (region == YELLOW))
        missing = int(np.count_nonzero(region == BLACK))
        assert 1 <= missing <= 3
        blocks.append(
            {
                "bbox": box,
                "coords": coords,
                "pattern": _mask_signature(region, YELLOW),
            }
        )
    assert 3 <= len(blocks) <= 4
    return blocks


def _extract_markers(grid: np.ndarray) -> list[dict]:
    markers = []
    for coords in _find_components(grid == GREY):
        box = _bbox(coords)
        assert _bbox_h(box) == BLOCK_SIZE and _bbox_w(box) == BLOCK_SIZE
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        sig = _mask_signature(region, GREY)
        assert sig in GLYPH_TO_COLOR
        markers.append(
            {
                "bbox": box,
                "coords": coords,
                "glyph": sig,
                "fill_color": GLYPH_TO_COLOR[sig],
            }
        )
    assert 3 <= len(markers) <= 4
    return markers


def _pair_markers_to_blocks(markers: list[dict], blocks: list[dict]) -> dict[int, int]:
    assert len(markers) == len(blocks)
    marker_to_block: dict[int, int] = {}
    used_blocks: set[int] = set()
    for marker_index, marker in enumerate(markers):
        dists = sorted((_dist2(marker["bbox"], block["bbox"]), block_index) for block_index, block in enumerate(blocks))
        assert dists[0][0] < dists[1][0]
        block_index = int(dists[0][1])
        assert block_index not in used_blocks
        used_blocks.add(block_index)
        marker_to_block[marker_index] = block_index
    assert len(used_blocks) == len(blocks)
    return marker_to_block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each incomplete yellow 3x3 block with the color encoded by its nearest grey glyph, then erase the glyphs.

    Input structure:
    - The grid has a black background, 3-4 incomplete yellow 8-connected objects, and 3-4 separate grey marker glyphs.
    - Each yellow object lies in a 3x3 bounding box and contains 1-3 black gaps inside that box.
    - Each grey marker is a 3x3 glyph whose grey-pixel arrangement is one of a fixed set of shape codes.

    Rule:
    - Pair every grey glyph with the yellow block whose 3x3-box center is nearest to the glyph's 3x3-box center.
    - Decode the glyph shape into a fill color.
    - Inside the paired yellow block's 3x3 bounding box, recolor every black gap pixel to that decoded color.
    - Leave the original yellow pixels unchanged and erase all grey glyph pixels back to black.

    Key insight:
    - The grey glyph does not mark cells to repaint directly; it supplies the color needed to complete the nearby yellow block's missing pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g).tolist()) <= {BLACK, YELLOW, GREY}

    blocks = _extract_blocks(g)
    markers = _extract_markers(g)
    pairing = _pair_markers_to_blocks(markers, blocks)

    out = g.copy()
    out[g == GREY] = BLACK
    for marker_index, block_index in pairing.items():
        fill_color = int(markers[marker_index]["fill_color"])
        r0, c0, r1, c1 = blocks[block_index]["bbox"]
        src = g[r0 : r1 + 1, c0 : c1 + 1]
        dst = out[r0 : r1 + 1, c0 : c1 + 1]
        dst[src == BLACK] = fill_color
    return out


def _pattern_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _build_block_patterns() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for yellow_cells in range(6, 9):
        for cells in itertools.combinations(range(9), yellow_cells):
            mask = np.zeros((3, 3), dtype=bool)
            mask.flat[list(cells)] = True
            if not _pattern_has_full_bbox(mask):
                continue
            if len(_find_components(mask)) != 1:
                continue
            patch = np.full((3, 3), BLACK, dtype=int)
            patch[mask] = YELLOW
            patterns.append(patch)
    assert patterns
    return patterns


BLOCK_PATTERNS = _build_block_patterns()


def _pattern_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return _mask_signature(patch, YELLOW)


def _zone_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]


def _zone_candidates(zone: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    zr0, zr1, zc0, zc1 = zone
    candidates: list[tuple[int, int, int, int]] = []
    for dr, dc in MARKER_OFFSETS:
        for r in range(zr0, zr1 - 1):
            mr = r + dr
            if mr < zr0 or mr + 2 > zr1:
                continue
            for c in range(zc0, zc1 - 1):
                mc = c + dc
                if mc < zc0 or mc + 2 > zc1:
                    continue
                candidates.append((r, c, dr, dc))
    return candidates


def _choose_layouts(h: int, w: int, count: int):
    zones = _zone_bounds(h, w)
    chosen_zones = random.sample(range(4), count)
    layouts = []
    for zone_index in chosen_zones:
        candidates = _zone_candidates(zones[zone_index])
        if not candidates:
            return None
        layouts.append((zone_index, random.choice(candidates)))
    random.shuffle(layouts)
    return layouts


def _select_type_ids(count: int) -> list[int]:
    type_ids = [0, 1, 2]
    if count == 4:
        type_ids.append(random.randrange(3))
    random.shuffle(type_ids)
    return type_ids


def _select_patterns(type_ids: list[int]) -> list[np.ndarray]:
    count = len(type_ids)
    shared = random.choice(BLOCK_PATTERNS)
    shared_sig = _pattern_signature(shared)
    patterns: list[np.ndarray | None] = [None] * count

    distinct_pairs = [(i, j) for i in range(count) for j in range(i + 1, count) if type_ids[i] != type_ids[j]]
    pair_i, pair_j = random.choice(distinct_pairs)
    patterns[pair_i] = shared.copy()
    patterns[pair_j] = shared.copy()

    non_shared = [patch for patch in BLOCK_PATTERNS if _pattern_signature(patch) != shared_sig]
    for i in range(count):
        if patterns[i] is None:
            patterns[i] = random.choice(non_shared).copy()
    out = [patch for patch in patterns if patch is not None]
    assert len(out) == count
    return out


def _expected_output(grid: np.ndarray, blocks: list[dict], markers: list[dict], intended_pairs: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    out[out == GREY] = BLACK
    for marker_index, block_index in intended_pairs.items():
        fill_color = int(markers[marker_index]["fill_color"])
        r0, c0, r1, c1 = blocks[block_index]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK] = fill_color
    return out


def _identity_pairing_is_unique(markers: list[dict], blocks: list[dict]) -> bool:
    used_blocks: set[int] = set()
    for marker_index, marker in enumerate(markers):
        dists = sorted((_dist2(marker["bbox"], block["bbox"]), block_index) for block_index, block in enumerate(blocks))
        if len(dists) > 1 and not (dists[0][0] < dists[1][0]):
            return False
        if dists[0][1] != marker_index or dists[0][1] in used_blocks:
            return False
        used_blocks.add(int(dists[0][1]))
    return len(used_blocks) == len(blocks)


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice([3, 4])
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    layouts = _choose_layouts(h, w, count)
    if layouts is None:
        return None

    type_ids = _select_type_ids(count)
    patterns = _select_patterns(type_ids)

    grid = np.full((h, w), BLACK, dtype=int)
    block_meta: list[dict] = []
    marker_meta: list[dict] = []
    intended_pairs: dict[int, int] = {}

    for i, ((_zone_index, (br, bc, dr, dc)), type_id, pattern) in enumerate(zip(layouts, type_ids, patterns)):
        mr = br + dr
        mc = bc + dc
        glyph = GLYPH_ITEMS[type_id][1]
        fill_color = int(GLYPH_ITEMS[type_id][2])

        grid[br : br + 3, bc : bc + 3][pattern == YELLOW] = YELLOW
        grid[mr : mr + 3, mc : mc + 3][glyph == 1] = GREY

        block_meta.append(
            {
                "bbox": (br, bc, br + 2, bc + 2),
                "pattern": _pattern_signature(pattern),
            }
        )
        marker_meta.append(
            {
                "bbox": (mr, mc, mr + 2, mc + 2),
                "glyph": _mask_signature(glyph, 1),
                "fill_color": fill_color,
            }
        )
        intended_pairs[i] = i

    if not _identity_pairing_is_unique(marker_meta, block_meta):
        return None

    expected = _expected_output(grid, block_meta, marker_meta, intended_pairs)
    return grid, expected


def _validate_semantics(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        blocks = _extract_blocks(grid)
        markers = _extract_markers(grid)
        pairing = _pair_markers_to_blocks(markers, blocks)
    except AssertionError:
        return False

    try:
        produced = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(produced, expected):
        return False
    if not np.array_equal(produced, _expected_output(grid, blocks, markers, pairing)):
        return False

    fill_colors = {int(marker["fill_color"]) for marker in markers}
    if fill_colors != {BLUE, RED, GREEN}:
        return False

    pattern_to_colors: dict[tuple[tuple[int, ...], ...], set[int]] = defaultdict(set)
    for marker_index, block_index in pairing.items():
        pattern_to_colors[blocks[block_index]["pattern"]].add(int(markers[marker_index]["fill_color"]))
        dists = sorted((_dist2(markers[marker_index]["bbox"], block["bbox"]), j) for j, block in enumerate(blocks))
        if len(dists) > 1 and not (dists[0][0] < dists[1][0]):
            return False

    if len(pattern_to_colors) < 2:
        return False
    if not any(len(colors) >= 2 for colors in pattern_to_colors.values()):
        return False

    hole_counts = set()
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        src = grid[r0 : r1 + 1, c0 : c1 + 1]
        dst = produced[r0 : r1 + 1, c0 : c1 + 1]
        missing = int(np.count_nonzero(src == BLACK))
        hole_counts.add(missing)
        if not (1 <= missing <= 3):
            return False
        filled_colors = {int(v) for v in dst[src == BLACK].tolist()}
        if len(filled_colors) != 1 or next(iter(filled_colors)) not in fill_colors:
            return False
        if np.any(dst[src == YELLOW] != YELLOW):
            return False

    if len(hole_counts) < 2:
        return False
    if np.any(produced == GREY):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, expected = built
        if _validate_semantics(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
