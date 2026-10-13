from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

ARROW_COLORS = (BLUE, RED, GREEN, YELLOW)
PATTERN_COLORS = (5, 6, 7, 8, 9)
BLOCK_SIZES = (4, 5)
SIDES = ("top", "right", "bottom", "left")
SIDE_ORDER = {"top": 0, "right": 1, "bottom": 2, "left": 3}
SIDE_TO_VEC = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
GEN_ATTEMPTS = 400


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _apply_transform(arr: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.fliplr(arr)
    if color == GREEN:
        return np.flipud(arr)
    if color == BLUE:
        return np.rot90(arr, k=3)
    if color == YELLOW:
        return np.rot90(arr, k=1)
    raise AssertionError(f"unexpected arrow color {color}")


def _decode_arrow(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    shape = np.zeros((h, w), dtype=np.uint8)
    for r, c in coords:
        shape[int(r - r0), int(c - c0)] = 1

    if h == 2 and w == 3:
        if np.array_equal(shape, np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)):
            return {"side": "top", "tip": (r0 + 1, c0 + 1), "vec": SIDE_TO_VEC["top"]}
        if np.array_equal(shape, np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)):
            return {"side": "bottom", "tip": (r0, c0 + 1), "vec": SIDE_TO_VEC["bottom"]}

    if h == 3 and w == 2:
        if np.array_equal(shape, np.array([[1, 0], [0, 1], [1, 0]], dtype=np.uint8)):
            return {"side": "left", "tip": (r0 + 1, c0 + 1), "vec": SIDE_TO_VEC["left"]}
        if np.array_equal(shape, np.array([[0, 1], [1, 0], [0, 1]], dtype=np.uint8)):
            return {"side": "right", "tip": (r0 + 1, c0), "vec": SIDE_TO_VEC["right"]}

    return None


def _is_full_square(g: np.ndarray, coords: np.ndarray) -> tuple[bool, int]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w:
        return False, -1
    if int(coords.shape[0]) != h * w:
        return False, -1
    if np.any(g[r0 : r1 + 1, c0 : c1 + 1] == BG):
        return False, -1
    return True, h


def _ray_target(block_id_grid: np.ndarray, tip: tuple[int, int], vec: tuple[int, int]) -> int | None:
    h, w = block_id_grid.shape
    r, c = tip
    dr, dc = vec
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        bid = int(block_id_grid[r, c])
        if bid >= 0:
            return bid


def _arrow_distance(bbox: tuple[int, int, int, int], side: str, tip: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    tr, tc = tip
    if side == "top":
        return r0 - tr - 1
    if side == "right":
        return tc - c1 - 1
    if side == "bottom":
        return tr - r1 - 1
    return c0 - tc - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply arrow-colored geometric transforms to square pattern blocks, erasing the arrows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components8(g != BG)
    square_candidates: list[dict] = []
    arrows: list[dict] = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if len(coords) == 3 and len(colors) == 1:
            color = next(iter(colors))
            decoded = _decode_arrow(coords)
            if decoded is not None and color in ARROW_COLORS:
                arrows.append(
                    {
                        "coords": coords.copy(),
                        "color": color,
                        "side": decoded["side"],
                        "tip": decoded["tip"],
                        "vec": decoded["vec"],
                    }
                )
                continue

        is_square, size = _is_full_square(g, coords)
        if is_square:
            square_candidates.append({"coords": coords.copy(), "bbox": _bbox(coords), "size": size})

    assert square_candidates, "expected at least one square block"
    counts: dict[int, int] = {}
    for block in square_candidates:
        counts[int(block["size"])] = counts.get(int(block["size"]), 0) + 1
    block_size = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    blocks = [block for block in square_candidates if int(block["size"]) == int(block_size)]
    assert blocks, "failed to identify square blocks"

    block_id_grid = np.full(g.shape, -1, dtype=int)
    for bid, block in enumerate(blocks):
        r0, c0, r1, c1 = block["bbox"]
        assert np.all(block_id_grid[r0 : r1 + 1, c0 : c1 + 1] == -1)
        block_id_grid[r0 : r1 + 1, c0 : c1 + 1] = int(bid)

    arrows_by_block: dict[int, list[dict]] = {bid: [] for bid in range(len(blocks))}
    for arrow in arrows:
        bid = _ray_target(block_id_grid, arrow["tip"], arrow["vec"])
        assert bid is not None, "arrow must point to a block"
        dist = _arrow_distance(blocks[bid]["bbox"], arrow["side"], arrow["tip"])
        assert dist >= 1, "arrows must be separated from the block by at least one black cell"
        arrows_by_block[bid].append(
            {
                "color": int(arrow["color"]),
                "side": str(arrow["side"]),
                "distance": int(dist),
            }
        )

    out = g.copy()
    for arrow in arrows:
        rr = arrow["coords"][:, 0]
        cc = arrow["coords"][:, 1]
        out[rr, cc] = BG

    for bid, block in enumerate(blocks):
        r0, c0, r1, c1 = block["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        ordered = sorted(
            arrows_by_block[bid],
            key=lambda item: (int(item["distance"]), SIDE_ORDER[str(item["side"])]),
        )
        for i in range(1, len(ordered)):
            prev = ordered[i - 1]
            cur = ordered[i]
            assert not (
                int(prev["distance"]) == int(cur["distance"]) and str(prev["side"]) == str(cur["side"])
            ), "ambiguous same-side equal-distance arrows"
        for arrow in ordered:
            patch = _apply_transform(patch, int(arrow["color"]))
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _sequence_is_identity(colors: list[int]) -> bool:
    probe = np.arange(16, dtype=int).reshape(4, 4)
    cur = probe.copy()
    for color in colors:
        cur = _apply_transform(cur, int(color))
    return np.array_equal(cur, probe)


def _random_pattern(size: int) -> np.ndarray:
    for _ in range(200):
        palette = random.sample(PATTERN_COLORS, random.randint(3, min(5, len(PATTERN_COLORS))))
        values = [random.choice(palette) for _ in range(size * size)]
        for i, color in enumerate(palette):
            values[i] = color
        random.shuffle(values)
        arr = np.array(values, dtype=int).reshape(size, size)
        variants = [
            np.fliplr(arr),
            np.flipud(arr),
            np.rot90(arr, 1),
            np.rot90(arr, 2),
            np.rot90(arr, 3),
            arr.T,
            np.fliplr(np.rot90(arr, 1)),
        ]
        if any(np.array_equal(arr, other) for other in variants):
            continue
        return arr
    raise RuntimeError("failed to build a non-symmetric pattern")


def _arrow_cells(
    bbox: tuple[int, int, int, int], side: str, distance: int, offset: int
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        tip = (r0 - 1 - distance, c0 + offset)
        cells = [(tip[0] - 1, tip[1] - 1), tip, (tip[0] - 1, tip[1] + 1)]
    elif side == "right":
        tip = (r0 + offset, c1 + 1 + distance)
        cells = [(tip[0] - 1, tip[1] + 1), tip, (tip[0] + 1, tip[1] + 1)]
    elif side == "bottom":
        tip = (r1 + 1 + distance, c0 + offset)
        cells = [(tip[0] + 1, tip[1] - 1), tip, (tip[0] + 1, tip[1] + 1)]
    else:
        tip = (r0 + offset, c0 - 1 - distance)
        cells = [(tip[0] - 1, tip[1] - 1), tip, (tip[0] + 1, tip[1] - 1)]
    return cells, tip, SIDE_TO_VEC[side]


def _can_place_arrow(
    scene: np.ndarray,
    block_id_grid: np.ndarray,
    block_id: int,
    cells: list[tuple[int, int]],
    tip: tuple[int, int],
    vec: tuple[int, int],
) -> bool:
    h, w = scene.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(scene[r, c]) != BG:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if int(scene[rr, cc]) != BG:
                    return False
    return _ray_target(block_id_grid, tip, vec) == int(block_id)


def _try_place_specs(scene: np.ndarray, block_id_grid: np.ndarray, block: dict, specs: list[dict]) -> list[dict] | None:
    size = int(block["size"])
    placements: list[dict] = []

    def backtrack(i: int) -> bool:
        if i == len(specs):
            return True
        spec = specs[i]
        offsets = list(range(1, size - 1))
        random.shuffle(offsets)
        for offset in offsets:
            cells, tip, vec = _arrow_cells(block["bbox"], str(spec["side"]), int(spec["distance"]), int(offset))
            if not _can_place_arrow(scene, block_id_grid, int(block["id"]), cells, tip, vec):
                continue
            for r, c in cells:
                scene[r, c] = int(spec["color"])
            placements.append(
                {
                    "color": int(spec["color"]),
                    "side": str(spec["side"]),
                    "distance": int(spec["distance"]),
                    "cells": [tuple(cell) for cell in cells],
                }
            )
            if backtrack(i + 1):
                return True
            placements.pop()
            for r, c in cells:
                scene[r, c] = BG
        return False

    if backtrack(0):
        return placements
    return None


def _render_expected(inp: np.ndarray, blocks: list[dict]) -> np.ndarray:
    out = inp.copy()
    for block in blocks:
        for arrow in block["arrows"]:
            for r, c in arrow["cells"]:
                out[r, c] = BG
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        patch = block["pattern"].copy()
        ordered = sorted(
            block["arrows"],
            key=lambda item: (int(item["distance"]), SIDE_ORDER[str(item["side"])]),
        )
        for arrow in ordered:
            patch = _apply_transform(patch, int(arrow["color"]))
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _crop_with_border(inp: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(inp != BG)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    top = random.randint(1, min(4, r0))
    left = random.randint(1, min(4, c0))
    bottom = random.randint(1, min(4, inp.shape[0] - 1 - r1))
    right = random.randint(1, min(4, inp.shape[1] - 1 - c1))
    rs = slice(r0 - top, r1 + 1 + bottom)
    cs = slice(c0 - left, c1 + 1 + right)
    return inp[rs, cs].copy(), expected[rs, cs].copy()


def _build_blocks() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    size = random.choice(BLOCK_SIZES)
    block_count = random.randint(5, 7)
    row_slots = 2 if block_count <= 6 and random.random() < 0.45 else 3
    col_slots = random.choice((3, 4))
    while row_slots * col_slots < block_count:
        row_slots += 1

    pitch = size + 10
    margin = 6
    h = margin * 2 + row_slots * pitch
    w = margin * 2 + col_slots * pitch

    slots = [(sr, sc) for sr in range(row_slots) for sc in range(col_slots)]
    chosen = random.sample(slots, block_count)
    random.shuffle(chosen)

    scene = np.full((h, w), BG, dtype=int)
    block_id_grid = np.full((h, w), -1, dtype=int)
    blocks: list[dict] = []
    for bid, (sr, sc) in enumerate(chosen):
        r0 = margin + sr * pitch + random.randint(0, 2)
        c0 = margin + sc * pitch + random.randint(0, 2)
        pattern = _random_pattern(size)
        r1 = r0 + size - 1
        c1 = c0 + size - 1
        scene[r0 : r1 + 1, c0 : c1 + 1] = pattern
        block_id_grid[r0 : r1 + 1, c0 : c1 + 1] = int(bid)
        blocks.append(
            {
                "id": int(bid),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "size": int(size),
                "pattern": pattern.copy(),
                "arrows": [],
            }
        )
    return scene, block_id_grid, blocks


def _place_arrows(scene: np.ndarray, block_id_grid: np.ndarray, blocks: list[dict]) -> bool:
    ids = list(range(len(blocks)))
    random.shuffle(ids)
    control_id = ids[0]
    equal_id = ids[1]
    dist_id = ids[2]
    single_ids = ids[3:]

    while True:
        perm = random.sample(list(ARROW_COLORS), 4)
        equal_pair = perm[:2]
        dist_pair = perm[2:]
        if not _sequence_is_identity(equal_pair) and not _sequence_is_identity(dist_pair):
            break

    equal_distance = random.choice((1, 2))
    equal_pairs = [(a, b) for i, a in enumerate(SIDES) for b in SIDES[i + 1 :]]
    random.shuffle(equal_pairs)
    placed = False
    for a, b in equal_pairs:
        ordered_sides = sorted((a, b), key=lambda side: SIDE_ORDER[side])
        specs = [
            {"side": ordered_sides[0], "distance": equal_distance, "color": int(equal_pair[0])},
            {"side": ordered_sides[1], "distance": equal_distance, "color": int(equal_pair[1])},
        ]
        placements = _try_place_specs(scene, block_id_grid, blocks[equal_id], specs)
        if placements is not None:
            blocks[equal_id]["arrows"].extend(placements)
            placed = True
            break
    if not placed:
        return False

    dist_pairs = [(a, b) for a in SIDES for b in SIDES if a != b]
    random.shuffle(dist_pairs)
    placed = False
    for near_side, far_side in dist_pairs:
        specs = [
            {"side": near_side, "distance": 1, "color": int(dist_pair[0])},
            {"side": far_side, "distance": 2, "color": int(dist_pair[1])},
        ]
        placements = _try_place_specs(scene, block_id_grid, blocks[dist_id], specs)
        if placements is not None:
            blocks[dist_id]["arrows"].extend(placements)
            placed = True
            break
    if not placed:
        return False

    for block_id in single_ids:
        options = [(side, dist) for side in SIDES for dist in (1, 2)]
        random.shuffle(options)
        color = int(random.choice(ARROW_COLORS))
        placed = False
        for side, dist in options:
            specs = [{"side": side, "distance": dist, "color": color}]
            placements = _try_place_specs(scene, block_id_grid, blocks[block_id], specs)
            if placements is not None:
                blocks[block_id]["arrows"].extend(placements)
                placed = True
                break
        if not placed:
            return False

    assert not blocks[control_id]["arrows"]
    used_colors = {int(arrow["color"]) for block in blocks for arrow in block["arrows"]}
    return used_colors == set(ARROW_COLORS)


def _gen_attempt() -> np.ndarray | None:
    scene, block_id_grid, blocks = _build_blocks()
    if not _place_arrows(scene, block_id_grid, blocks):
        return None
    expected = _render_expected(scene, blocks)
    cropped_inp, cropped_expected = _crop_with_border(scene, expected)
    try:
        actual = main(cropped_inp)
    except Exception:
        return None
    if not np.array_equal(actual, cropped_expected):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-transform scene")
