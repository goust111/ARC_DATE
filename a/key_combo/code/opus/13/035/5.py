from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLOCK_SIZE = 6
MACRO_SIDE = 3
GRID_SIDE = BLOCK_SIZE * MACRO_SIDE
BG = 0
INPUT_COLOR = 5
SIZE_TO_COLOR = {2: 3, 3: 2, 4: 1}

GEN_RETRIES = 600
THREE_SIZE_PATTERNS = (
    (2, 2, 3),
    (2, 3, 2),
    (3, 2, 2),
)
TWO_SIZE_PATTERNS = (
    (5, 2, 0),
    (5, 0, 2),
    (2, 5, 0),
    (0, 5, 2),
    (2, 0, 5),
    (0, 2, 5),
    (4, 2, 1),
    (4, 1, 2),
    (2, 4, 1),
    (1, 4, 2),
    (2, 1, 4),
    (1, 2, 4),
    (4, 3, 0),
    (3, 4, 0),
    (4, 0, 3),
    (3, 0, 4),
    (0, 4, 3),
    (0, 3, 4),
)

SHAPES = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, (r0, c0), r1 - r0 + 1, c1 - c0 + 1


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_SIDE, GRID_SIDE):
        raise AssertionError(f"expected {GRID_SIDE}x{GRID_SIDE} grid, got {g.shape}")

    objects: list[dict] = []
    empty_blocks = 0
    for br in range(MACRO_SIDE):
        for bc in range(MACRO_SIDE):
            r0 = br * BLOCK_SIZE
            c0 = bc * BLOCK_SIZE
            block = g[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
            if not np.all((block == BG) | (block == INPUT_COLOR)):
                raise AssertionError("input should contain only black background and grey objects")

            comps = _components(block == INPUT_COLOR)
            if len(comps) > 1:
                raise AssertionError("each macro-block may contain at most one grey object")
            if not comps:
                empty_blocks += 1
                continue

            coords = comps[0]
            size = len(coords)
            if size not in SIZE_TO_COLOR:
                raise AssertionError("grey objects must have 2-4 pixels")

            norm, local_tl, bbox_h, bbox_w = _normalize(coords)
            objects.append(
                {
                    "block": (br, bc),
                    "size": size,
                    "color": SIZE_TO_COLOR[size],
                    "norm": norm,
                    "local_tl": local_tl,
                    "bbox_h": bbox_h,
                    "bbox_w": bbox_w,
                }
            )
    return objects, empty_blocks


def _render(
    objects: list[dict],
    repeat_fn,
    *,
    anchor_mode: str,
) -> np.ndarray:
    out = np.full((GRID_SIDE, GRID_SIDE), BG, dtype=int)
    for obj in objects:
        br, bc = obj["block"]
        block_r0 = br * BLOCK_SIZE
        block_c0 = bc * BLOCK_SIZE
        if anchor_mode == "macro":
            base_r = block_r0
            base_c = block_c0
        elif anchor_mode == "local":
            base_r = block_r0 + obj["local_tl"][0]
            base_c = block_c0 + obj["local_tl"][1]
        else:
            raise ValueError(f"unknown anchor_mode: {anchor_mode}")

        copies = int(repeat_fn(obj))
        stride = obj["bbox_w"] + 1
        for copy_idx in range(copies):
            col_shift = copy_idx * stride
            for dr, dc in obj["norm"]:
                rr = base_r + dr
                cc = base_c + col_shift + dc
                if block_r0 <= rr < block_r0 + BLOCK_SIZE and block_c0 <= cc < block_c0 + BLOCK_SIZE:
                    out[rr, cc] = obj["color"]
    return out


def _render_main(objects: list[dict]) -> np.ndarray:
    color_counts = Counter(obj["color"] for obj in objects)
    return _render(objects, lambda obj: color_counts[obj["color"]], anchor_mode="macro")


def _wrong_repeat_by_size(grid: np.ndarray) -> np.ndarray:
    objects, _ = _parse_objects(grid)
    return _render(objects, lambda obj: obj["size"], anchor_mode="macro")


def _wrong_keep_local_anchor(grid: np.ndarray) -> np.ndarray:
    objects, _ = _parse_objects(grid)
    color_counts = Counter(obj["color"] for obj in objects)
    return _render(objects, lambda obj: color_counts[obj["color"]], anchor_mode="local")


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey macro-block object by area, then tile it horizontally using the global count of its recolored group.

    Input structure:
    - The grid is an 18x18 black canvas implicitly partitioned into a 3x3 array of 6x6 macro-blocks.
    - Each non-empty macro-block contains exactly one 4-connected grey object with 2, 3, or 4 pixels.
    - Empty macro-blocks are entirely black, and there are no visible separator lines between blocks.

    Rule:
    - For each grey object, count its pixels P and recolor it by area: 2->green, 3->red, 4->blue.
    - Count how many objects share each recolored color across the whole grid.
    - In every non-empty macro-block, erase the original placement and redraw the cropped recolored object from the block's top-left corner.
    - Tile that object horizontally with a one-column gap, repeating it N times where N is the global number of objects of the same recolored color; clip any part that extends past the block's right edge.

    Key insight:
    - The repeat count is not local to one object: it is a global property shared by all objects with the same size-derived color, while the tiled shape and spacing still depend on each object's own cropped geometry.
    """
    objects, _ = _parse_objects(np.asarray(grid, dtype=int))
    return _render_main(objects)


def _sample_size_list() -> list[int]:
    if random.random() < 0.75:
        counts = random.choice(THREE_SIZE_PATTERNS)
    else:
        counts = random.choice(TWO_SIZE_PATTERNS)
    sizes = [2] * counts[0] + [3] * counts[1] + [4] * counts[2]
    random.shuffle(sizes)
    return sizes


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _place_shape(block: np.ndarray, shape: tuple[tuple[int, int], ...], *, force_row_shift: bool = False, force_col_shift: bool = False) -> None:
    shape_h, shape_w = _shape_dims(shape)
    candidates = [
        (r, c)
        for r in range(BLOCK_SIZE - shape_h + 1)
        for c in range(BLOCK_SIZE - shape_w + 1)
        if (r, c) != (0, 0)
    ]
    if force_row_shift:
        shifted = [(r, c) for r, c in candidates if r > 0]
        if shifted:
            candidates = shifted
    if force_col_shift:
        shifted = [(r, c) for r, c in candidates if c > 0]
        if shifted:
            candidates = shifted
    top, left = random.choice(candidates)
    for dr, dc in shape:
        block[top + dr, left + dc] = INPUT_COLOR


def _build_candidate() -> np.ndarray:
    grid = np.full((GRID_SIDE, GRID_SIDE), BG, dtype=int)
    filled_blocks = random.sample(range(MACRO_SIDE * MACRO_SIDE), 7)
    size_list = _sample_size_list()

    force_row_idx = random.randrange(len(size_list))
    force_col_idx = random.randrange(len(size_list))
    for idx, (block_id, size) in enumerate(zip(filled_blocks, size_list)):
        br, bc = divmod(block_id, MACRO_SIDE)
        r0 = br * BLOCK_SIZE
        c0 = bc * BLOCK_SIZE
        shape = random.choice(SHAPES[size])
        block = grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
        _place_shape(
            block,
            shape,
            force_row_shift=idx == force_row_idx,
            force_col_shift=idx == force_col_idx,
        )
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    objects, empty_blocks = _parse_objects(grid)
    if len(objects) != 7 or empty_blocks != 2:
        return False

    size_counts = Counter(obj["size"] for obj in objects)
    if len(size_counts) < 2:
        return False
    if max(size_counts.values()) < 2:
        return False
    if not any(count != size for size, count in size_counts.items()):
        return False
    if not any(obj["local_tl"][0] > 0 for obj in objects):
        return False
    if not any(obj["local_tl"][1] > 0 for obj in objects):
        return False

    clipped = False
    unclipped = False
    varied_group = False
    for size, count in size_counts.items():
        group = [obj for obj in objects if obj["size"] == size]
        if len(group) >= 2 and (len({obj["bbox_w"] for obj in group}) >= 2 or len({obj["norm"] for obj in group}) >= 2):
            varied_group = True
        for obj in group:
            total_span = count * obj["bbox_w"] + (count - 1)
            if total_span > BLOCK_SIZE:
                clipped = True
            else:
                unclipped = True
    if not clipped or not unclipped or not varied_group:
        return False

    out = main(grid)
    if np.array_equal(out, _wrong_repeat_by_size(grid)):
        return False
    if np.array_equal(out, _wrong_keep_local_anchor(grid)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
