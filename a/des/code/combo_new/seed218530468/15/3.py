from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIR_BY_COLOR = {
    2: (0, 1),   # red -> right
    1: (1, 0),   # blue -> down
    3: (0, -1),  # green -> left
    4: (-1, 0),  # yellow -> up
}
FRAME_COLORS = tuple(DIR_BY_COLOR.keys())
SEED_COLORS = (5, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
_RNG = random.Random()


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray | Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(list(coords) if not isinstance(coords, np.ndarray) else coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    if not np.array_equal(sub, frame):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _contains_concentric(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    top = int(ir0 - or0)
    left = int(ic0 - oc0)
    bottom = int(or1 - ir1)
    right = int(oc1 - ic1)
    return min(top, left, bottom, right) > 0 and top == bottom and left == right


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in FRAME_COLORS:
        for comp in _components(grid == int(color)):
            mask = np.zeros(grid.shape, dtype=bool)
            mask[comp[:, 0], comp[:, 1]] = True
            box = _frame_component_bbox(mask)
            if box is None:
                continue
            frames.append({"color": int(color), "bbox": tuple(map(int, box))})
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3], item["color"]))
    return frames


def _extract_structures(grid: np.ndarray) -> list[dict]:
    frames = _find_frames(grid)
    if not frames:
        return []

    parents: list[int | None] = [None] * len(frames)
    children: list[list[int]] = [[] for _ in frames]

    def area(box: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = box
        return int((r1 - r0 + 1) * (c1 - c0 + 1))

    for i, inner in enumerate(frames):
        contenders = [
            (area(outer["bbox"]), j)
            for j, outer in enumerate(frames)
            if i != j and _contains_concentric(outer["bbox"], inner["bbox"])
        ]
        if not contenders:
            continue
        _, parent_idx = min(contenders)
        parents[i] = int(parent_idx)
        children[parent_idx].append(i)

    leaves = [i for i, kids in enumerate(children) if not kids]
    structures: list[dict] = []
    for leaf_idx in leaves:
        chain_ids: list[int] = []
        cur: int | None = int(leaf_idx)
        while cur is not None:
            chain_ids.append(int(cur))
            cur = parents[cur]
        chain = [frames[idx] for idx in chain_ids]
        chain.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1))
        if not (2 <= len(chain) <= 4):
            continue
        if len({item["color"] for item in chain}) != len(chain):
            continue

        r0, c0, r1, c1 = chain[0]["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        frame_mask = np.isin(interior, FRAME_COLORS)
        seed_mask = (interior != BG) & (~frame_mask)
        seed_comps = _components(seed_mask)
        if len(seed_comps) != 1:
            continue
        seed_coords = seed_comps[0]
        sr0, sc0, sr1, sc1 = _bbox(seed_coords)
        tight = seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
        seed_vals = interior[seed_mask]
        if seed_vals.size == 0:
            continue
        seed_colors = {int(v) for v in seed_vals.tolist()}
        if len(seed_colors) != 1:
            continue
        seed_color = int(next(iter(seed_colors)))
        structures.append(
            {
                "frames_inner_to_outer": chain,
                "seed_color": seed_color,
                "seed_mask": tight.copy(),
                "anchor": (int(r0 + 1 + sr0), int(c0 + 1 + sc0)),
            }
        )

    structures.sort(key=lambda item: (item["anchor"][0], item["anchor"][1], item["seed_color"]))
    return structures


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _coords_span(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _grow_once(coords: set[tuple[int, int]], direction: tuple[int, int]) -> set[tuple[int, int]]:
    min_r, min_c, max_r, max_c = _coords_span(coords)
    height = int(max_r - min_r + 1)
    width = int(max_c - min_c + 1)
    dr, dc = direction
    if (dr, dc) == (0, 1):
        shift = (0, width)
    elif (dr, dc) == (0, -1):
        shift = (0, -width)
    elif (dr, dc) == (1, 0):
        shift = (height, 0)
    else:
        shift = (-height, 0)
    sr, sc = shift
    return set(coords) | {(int(r + sr), int(c + sc)) for r, c in coords}


def _grown_coords(seed_mask: np.ndarray, layer_colors: list[int]) -> set[tuple[int, int]]:
    coords = _coords_from_mask(seed_mask)
    for color in layer_colors:
        coords = _grow_once(coords, DIR_BY_COLOR[int(color)])
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the nested direction-coded frames and grow each enclosed seed by repeated directional doubling.

    Input structure:
    - Black (0) is the background.
    - One or more objects are nested single-pixel rectangular frames made from red, blue, green, and yellow.
    - Each frame chain has 2 to 4 concentric layers, and the innermost frame encloses one small single-color seed pattern.

    Rule:
    - Erase all frames and keep only the seed pattern of each nested object.
    - Read the frame colors from the innermost layer outward.
    - For each layer, duplicate the current seed result and attach the copy flush against the whole current bounding box in the color-coded direction:
      red = right, blue = down, green = left, yellow = up.
    - Draw the final grown pattern back onto an output grid of the same size, anchored so the original seed pattern's top-left stays fixed; clip any pixels that leave the grid.

    Key insight:
    - The nested rectangles are an ordered instruction stack: every frame color is one growth command, and the pattern doubles step by step from the center outward.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    for struct in _extract_structures(g):
        layer_colors = [int(frame["color"]) for frame in struct["frames_inner_to_outer"]]
        coords = _grown_coords(struct["seed_mask"], layer_colors)
        anchor_r, anchor_c = struct["anchor"]
        seed_color = int(struct["seed_color"])
        for dr, dc in coords:
            rr = int(anchor_r + dr)
            cc = int(anchor_c + dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = seed_color

    return out


_SEED_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
]


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        int(a[2] + pad) < int(b[0])
        or int(b[2] + pad) < int(a[0])
        or int(a[3] + pad) < int(b[1])
        or int(b[3] + pad) < int(a[1])
    )


def _sample_spec(seed_color: int) -> dict:
    seed_mask = _SEED_SHAPES[_RNG.randrange(len(_SEED_SHAPES))].copy()
    layer_count = _RNG.randint(2, 4)
    layer_colors = _RNG.sample(list(FRAME_COLORS), layer_count)
    pad_top = _RNG.randint(0, 1)
    pad_left = _RNG.randint(0, 1)
    pad_bottom = _RNG.randint(0, 1)
    pad_right = _RNG.randint(0, 1)

    out_coords = _grown_coords(seed_mask, layer_colors)
    min_r, min_c, max_r, max_c = _coords_span(out_coords)
    seed_h, seed_w = seed_mask.shape
    input_h = int(seed_h + pad_top + pad_bottom + 2 * layer_count)
    input_w = int(seed_w + pad_left + pad_right + 2 * layer_count)
    anchor_off_r = int(layer_count + pad_top)
    anchor_off_c = int(layer_count + pad_left)

    return {
        "seed_color": int(seed_color),
        "seed_mask": seed_mask,
        "layer_colors": [int(color) for color in layer_colors],
        "layer_count": int(layer_count),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "pad_bottom": int(pad_bottom),
        "pad_right": int(pad_right),
        "input_h": int(input_h),
        "input_w": int(input_w),
        "anchor_off_r": int(anchor_off_r),
        "anchor_off_c": int(anchor_off_c),
        "output_rel_bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
    }


def _draw_spec(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    layer_count = int(spec["layer_count"])
    outer_h = int(spec["input_h"])
    outer_w = int(spec["input_w"])
    for depth, color in enumerate(reversed(spec["layer_colors"])):
        r0 = int(top + depth)
        c0 = int(left + depth)
        r1 = int(top + outer_h - 1 - depth)
        c1 = int(left + outer_w - 1 - depth)
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)

    seed_top = int(top + spec["anchor_off_r"])
    seed_left = int(left + spec["anchor_off_c"])
    seed_mask = spec["seed_mask"]
    grid[seed_top : seed_top + seed_mask.shape[0], seed_left : seed_left + seed_mask.shape[1]][seed_mask] = int(spec["seed_color"])


def _gen_attempt() -> np.ndarray | None:
    object_count = _RNG.randint(1, 3)
    h = _RNG.randint(18 + 2 * (object_count - 1), 30)
    w = _RNG.randint(18 + 2 * (object_count - 1), 30)
    grid = np.zeros((h, w), dtype=int)

    seed_colors = _RNG.sample(list(SEED_COLORS), object_count)
    specs = [_sample_spec(color) for color in seed_colors]
    specs.sort(key=lambda item: max(int(item["input_h"] * item["input_w"]), int((item["output_rel_bbox"][2] - item["output_rel_bbox"][0] + 1) * (item["output_rel_bbox"][3] - item["output_rel_bbox"][1] + 1))), reverse=True)

    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for spec in specs:
        min_r, min_c, max_r, max_c = spec["output_rel_bbox"]
        lo_r = max(0, -int(spec["anchor_off_r"] + min_r))
        hi_r = min(int(h - spec["input_h"]), int(h - 1 - (spec["anchor_off_r"] + max_r)))
        lo_c = max(0, -int(spec["anchor_off_c"] + min_c))
        hi_c = min(int(w - spec["input_w"]), int(w - 1 - (spec["anchor_off_c"] + max_c)))
        if lo_r > hi_r or lo_c > hi_c:
            return None

        placed = False
        for _ in range(160):
            top = _RNG.randint(lo_r, hi_r)
            left = _RNG.randint(lo_c, hi_c)
            input_box = (int(top), int(left), int(top + spec["input_h"] - 1), int(left + spec["input_w"] - 1))
            output_box = (
                int(top + spec["anchor_off_r"] + min_r),
                int(left + spec["anchor_off_c"] + min_c),
                int(top + spec["anchor_off_r"] + max_r),
                int(left + spec["anchor_off_c"] + max_c),
            )
            if any(_boxes_too_close(input_box, box, pad=1) for box in input_boxes):
                continue
            if any(_boxes_too_close(output_box, box, pad=1) for box in output_boxes):
                continue
            _draw_spec(grid, spec, int(top), int(left))
            input_boxes.append(input_box)
            output_boxes.append(output_box)
            placed = True
            break
        if not placed:
            return None

    structures = _extract_structures(grid)
    if len(structures) != object_count:
        return None
    for struct in structures:
        if not (2 <= len(struct["frames_inner_to_outer"]) <= 4):
            return None
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
