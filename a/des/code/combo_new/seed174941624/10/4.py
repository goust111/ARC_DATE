from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
CORNERS = ("tl", "tr", "br", "bl")
CORNER_ROT = {"tl": 0, "tr": 3, "br": 2, "bl": 1}
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _match_rotation(piece_mask: np.ndarray, hole_mask: np.ndarray) -> int | None:
    for k in range(4):
        if np.array_equal(np.rot90(piece_mask, int(k)), hole_mask):
            return int(k)
    return None


def _component_info(coords: np.ndarray) -> dict[str, Any]:
    mask, bbox = _tight_mask(coords)
    return {
        "coords": coords,
        "mask": mask,
        "bbox": bbox,
        "area": int(coords.shape[0]),
        "bbox_area": int(_bbox_area(bbox)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each loose L piece to match its same-colored corner notch and restore the full rectangles.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two disconnected objects: a larger
      corner-notched rectangle and a smaller L-shaped piece of the same color.
    - The rectangle's missing part occupies one corner of its bounding box, while
      the loose piece may be rotated by 0, 90, 180, or 270 degrees elsewhere.

    Rule:
    - Pair objects by color.
    - For each color, read the larger object's missing corner shape as the target notch.
    - Rotate the smaller L piece until its shape matches that notch, then conceptually
      place it into the notch.
    - Output the completed solid rectangle at the larger object's original position and
      erase the loose piece back to black.

    Key insight:
    - Color identifies the correct pair, but the actual move depends on matching the
      loose L piece to the larger object's corner gap under rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_most_common_color(g))
    out = g.copy()

    for color in (int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = [_component_info(coords) for coords in _components(g == int(color))]
        assert len(comps) == 2, "each color should form exactly one block and one loose piece"
        comps.sort(key=lambda info: (int(info["bbox_area"]), int(info["area"])), reverse=True)
        block, piece = comps
        assert int(block["bbox_area"]) > int(piece["bbox_area"]), "block bbox should be larger than piece bbox"

        br0, bc0, br1, bc1 = map(int, block["bbox"])
        block_patch = g[br0 : br1 + 1, bc0 : bc1 + 1]
        hole_mask = block_patch != int(color)
        assert bool(hole_mask.any()), "block must contain a corner notch"
        assert bool(np.all(block_patch[hole_mask] == int(bg))), "notch cells must be background"

        hole_components = _components(hole_mask)
        assert len(hole_components) == 1, "notch should be a single connected region"
        hole_shape, _ = _tight_mask(hole_components[0])
        assert _match_rotation(piece["mask"], hole_shape) is not None, "piece must match the notch under rotation"

        pr0, pc0, pr1, pc1 = map(int, piece["bbox"])
        piece_view = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        piece_view[piece["mask"]] = int(bg)

        block_view = out[br0 : br1 + 1, bc0 : bc1 + 1]
        block_view[hole_mask] = int(color)

    return out


def _l_mask(h: int, w: int, arm_h: int, arm_w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:arm_h, :] = True
    mask[:, :arm_w] = True
    return mask


def _corner_origin(corner: str, rect_h: int, rect_w: int, hole_h: int, hole_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, int(rect_w - hole_w)
    if corner == "br":
        return int(rect_h - hole_h), int(rect_w - hole_w)
    return int(rect_h - hole_h), 0


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(BG)
    region[mask] = patch[mask]


def _place_patch(grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        rr0 = max(0, int(r0 - 1))
        cc0 = max(0, int(c0 - 1))
        rr1 = min(h, int(r0 + ph + 1))
        cc1 = min(w, int(c0 + pw + 1))
        if bool(reserve[rr0:rr1, cc0:cc1].any()):
            continue
        _paint_patch(grid, patch, int(r0), int(c0))
        reserve[rr0:rr1, cc0:cc1] = True
        return int(r0), int(c0)
    return None


def _balanced_sample(options: tuple[Any, ...] | list[Any], n: int) -> list[Any]:
    pool = list(options) * ((int(n) + len(options) - 1) // len(options))
    random.shuffle(pool)
    return pool[: int(n)]


def _build_pair(color: int, corner: str, loose_rot: int) -> dict[str, Any] | None:
    base_h = random.randint(3, 5)
    base_w = random.randint(3, 5)
    arm_h = random.randint(1, min(2, base_h - 1))
    arm_w = random.randint(1, min(2, base_w - 1))

    base_mask = _l_mask(base_h, base_w, arm_h, arm_w)
    hole_mask = np.rot90(base_mask, int(CORNER_ROT[corner]))
    hole_h, hole_w = map(int, hole_mask.shape)

    rect_h_max = min(8, hole_h + 3)
    rect_w_max = min(8, hole_w + 3)
    if hole_h + 1 > rect_h_max or hole_w + 1 > rect_w_max:
        return None

    rect_h = random.randint(hole_h + 1, rect_h_max)
    rect_w = random.randint(hole_w + 1, rect_w_max)
    block_patch = np.full((rect_h, rect_w), int(color), dtype=int)
    top, left = _corner_origin(corner, rect_h, rect_w, hole_h, hole_w)
    block_patch[top : top + hole_h, left : left + hole_w][hole_mask] = int(BG)

    if len(_components(block_patch == int(color))) != 1:
        return None

    piece_mask = np.rot90(base_mask, int(loose_rot))
    piece_patch = np.full(piece_mask.shape, int(BG), dtype=int)
    piece_patch[piece_mask] = int(color)

    full_patch = np.full((rect_h, rect_w), int(color), dtype=int)
    return {
        "color": int(color),
        "corner": corner,
        "loose_rot": int(loose_rot),
        "block_patch": block_patch,
        "piece_patch": piece_patch,
        "full_patch": full_patch,
    }


def _candidate_canvas_shapes(pairs: list[dict[str, Any]]) -> list[tuple[int, int]]:
    shapes = [tuple(map(int, pair["block_patch"].shape)) for pair in pairs]
    shapes += [tuple(map(int, pair["piece_patch"].shape)) for pair in pairs]
    max_h = max(h for h, _w in shapes)
    max_w = max(w for _h, w in shapes)
    reserve_area = sum((h + 2) * (w + 2) for h, w in shapes)

    choices: list[tuple[int, int]] = []
    for h in range(max_h, 31):
        for w in range(max_w, 31):
            if h * w < reserve_area + 12:
                continue
            if reserve_area * 100 > 72 * h * w:
                continue
            choices.append((int(h), int(w)))
    return choices


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 1], k=1)[0]
    colors = random.sample(list(COLORS), int(pair_count))
    corners = _balanced_sample(CORNERS, int(pair_count))
    loose_rots = _balanced_sample((0, 1, 2, 3), int(pair_count))

    pairs: list[dict[str, Any]] = []
    for color, corner, loose_rot in zip(colors, corners, loose_rots):
        pair = _build_pair(int(color), str(corner), int(loose_rot))
        if pair is None:
            return None
        pairs.append(pair)

    canvas_choices = _candidate_canvas_shapes(pairs)
    if not canvas_choices:
        return None
    height, width = random.choice(canvas_choices)

    inp = np.full((height, width), int(BG), dtype=int)
    expected = np.full((height, width), int(BG), dtype=int)
    reserve = np.zeros((height, width), dtype=bool)

    block_order = list(range(len(pairs)))
    random.shuffle(block_order)
    block_order.sort(
        key=lambda idx: int(np.prod(pairs[idx]["block_patch"].shape)),
        reverse=True,
    )
    for idx in block_order:
        pair = pairs[idx]
        pos = _place_patch(inp, reserve, pair["block_patch"])
        if pos is None:
            return None
        r0, c0 = pos
        _paint_patch(expected, pair["full_patch"], int(r0), int(c0))

    piece_order = list(range(len(pairs)))
    random.shuffle(piece_order)
    piece_order.sort(
        key=lambda idx: int(np.prod(pairs[idx]["piece_patch"].shape)),
        reverse=True,
    )
    for idx in piece_order:
        pos = _place_patch(inp, reserve, pairs[idx]["piece_patch"])
        if pos is None:
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
