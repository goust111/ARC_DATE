from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
ORANGE = 7
BODY_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> np.ndarray:
    h, w = patch.shape
    hh, ww = dst.shape
    r0 = int(r0)
    c0 = int(c0)

    if not clip:
        if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
            raise ValueError("patch out of bounds")
        rr0, cc0, rr1, cc1 = r0, c0, r0 + h, c0 + w
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(hh, r0 + h)
        cc1 = min(ww, c0 + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0

    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    dst_view = dst[rr0:rr1, cc0:cc1]
    patch_view = patch[pr0:pr1, pc0:pc1]

    if transparent is None:
        dst_view[:] = patch_view
    else:
        mask = patch_view != int(transparent)
        dst_view[mask] = patch_view[mask]
    return dst


def _component_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    components: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _analyze_object(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    orange = crop == ORANGE
    assert orange.any(), "each object must contain an orange fracture line"

    orange_coords = np.argwhere(orange)
    rows = np.unique(orange_coords[:, 0])
    cols = np.unique(orange_coords[:, 1])
    h, w = crop.shape

    if len(cols) == 1:
        assert len(rows) == h and orange.sum() == h
        fracture = "vertical"
    elif len(rows) == 1:
        assert len(cols) == w and orange.sum() == w
        fracture = "horizontal"
    else:
        raise AssertionError("fracture marker must be a straight full-span line")

    body = crop.copy()
    body[orange] = bg
    assert np.any(body != bg)

    return {
        "bbox": (r0, c0, r1, c1),
        "crop": crop,
        "body": body,
        "fracture": fracture,
    }


def _tile_render(
    shape: tuple[int, int],
    patch: np.ndarray,
    fracture: str,
    r0: int,
    c0: int,
    *,
    bg: int,
) -> tuple[np.ndarray, int]:
    out = np.full(shape, int(bg), dtype=int)
    ph, pw = patch.shape
    copies = 0

    if fracture == "vertical":
        c = int(c0)
        while c + pw - 1 >= 0:
            _paste(out, patch, int(r0), c, transparent=bg, clip=True)
            copies += 1
            c -= pw
        c = int(c0) + pw
        while c <= shape[1] - 1:
            _paste(out, patch, int(r0), c, transparent=bg, clip=True)
            copies += 1
            c += pw
    else:
        r = int(r0)
        while r + ph - 1 >= 0:
            _paste(out, patch, r, int(c0), transparent=bg, clip=True)
            copies += 1
            r -= ph
        r = int(r0) + ph
        while r <= shape[0] - 1:
            _paste(out, patch, r, int(c0), transparent=bg, clip=True)
            copies += 1
            r += ph

    return out, copies


def main(grid: np.ndarray) -> np.ndarray:
    """Erase each orange fracture line, then tile the remaining object perpendicular to that line.

    Input structure:
    - A black background contains 2-3 separated 8-connected foreground objects.
    - Each object mixes one body color with orange marker pixels and has a non-square rectangular bbox.
    - Inside each object, the orange pixels form one complete straight line across the bbox short axis.

    Rule:
    - Find every 8-connected foreground object.
    - Detect whether its orange marker line is vertical or horizontal.
    - Erase the orange pixels to background; the remaining colored pixels inside the same bbox are the motif.
    - If the fracture line is horizontal, repeat that motif vertically with period equal to the motif bbox height.
    - If the fracture line is vertical, repeat that motif horizontally with period equal to the motif bbox width.
    - Keep the orthogonal coordinate fixed, clip copies at the grid border, and treat background cells inside the motif as transparent.

    Key insight:
    - The fracture line is not a piece to preserve or a cut to split on; its orientation is a control signal that selects the perpendicular tiling axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for coords in _component_coords(g != bg, connectivity=8):
        info = _analyze_object(g, coords, bg)
        r0, c0, _r1, _c1 = info["bbox"]
        tiled, _copies = _tile_render(g.shape, info["body"], info["fracture"], r0, c0, bg=bg)
        mask = tiled != bg
        out[mask] = tiled[mask]

    return out


def _random_path(a: tuple[int, int], b: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    r, c = a
    path = [(r, c)]
    while (r, c) != b:
        options = []
        if r < b[0]:
            options.append((r + 1, c))
        if r > b[0]:
            options.append((r - 1, c))
        if c < b[1]:
            options.append((r, c + 1))
        if c > b[1]:
            options.append((r, c - 1))
        r, c = options[int(rng.integers(len(options)))]
        path.append((r, c))
    return path


def _grow_region(
    h: int,
    w: int,
    required: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    required = list(dict.fromkeys(required))
    area = h * w
    min_area = max(len(required), (area + 2) // 3)
    max_area = max(min_area, area - 1 if area > 1 else area)
    target = int(rng.integers(min_area, max_area + 1))

    cells = {required[0]}
    for anchor in required[1:]:
        start = min(cells, key=lambda cell: abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]))
        for cell in _random_path(start, anchor, rng):
            cells.add(cell)

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cell = tuple(frontier.pop()) if len(frontier) == 1 else list(frontier)[int(rng.integers(len(frontier)))]
        cells.add(cell)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_vertical_object(color: int, rng: np.random.Generator) -> dict | None:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(h + 2, min(9, h + 5)))
    line_c = int(rng.integers(1, w - 1))
    left_w = line_c
    right_w = w - line_c - 1
    if left_w < 1 or right_w < 1:
        return None

    left_required = [(int(rng.integers(h)), 0), (int(rng.integers(h)), left_w - 1)]
    right_required = [(int(rng.integers(h)), 0), (int(rng.integers(h)), right_w - 1)]
    left = _grow_region(h, left_w, left_required, rng)
    right = _grow_region(h, right_w, right_required, rng)

    patch = np.full((h, w), BG, dtype=int)
    patch[:, :line_c][left] = int(color)
    patch[:, line_c] = ORANGE
    patch[:, line_c + 1 :][right] = int(color)

    body = patch.copy()
    body[:, line_c] = BG
    if np.array_equal(body[:, :line_c], np.full((h, left_w), int(color))) and np.array_equal(
        body[:, line_c + 1 :], np.full((h, right_w), int(color))
    ):
        return None
    return {"fracture": "vertical", "input_patch": patch, "body_patch": body}


def _make_horizontal_object(color: int, rng: np.random.Generator) -> dict | None:
    w = int(rng.integers(3, 6))
    h = int(rng.integers(w + 2, min(9, w + 5)))
    line_r = int(rng.integers(1, h - 1))
    top_h = line_r
    bottom_h = h - line_r - 1
    if top_h < 1 or bottom_h < 1:
        return None

    top_required = [(0, int(rng.integers(w))), (top_h - 1, int(rng.integers(w)))]
    bottom_required = [(0, int(rng.integers(w))), (bottom_h - 1, int(rng.integers(w)))]
    top = _grow_region(top_h, w, top_required, rng)
    bottom = _grow_region(bottom_h, w, bottom_required, rng)

    patch = np.full((h, w), BG, dtype=int)
    patch[:line_r, :][top] = int(color)
    patch[line_r, :] = ORANGE
    patch[line_r + 1 :, :][bottom] = int(color)

    body = patch.copy()
    body[line_r, :] = BG
    if np.array_equal(body[:line_r], np.full((top_h, w), int(color))) and np.array_equal(
        body[line_r + 1 :], np.full((bottom_h, w), int(color))
    ):
        return None
    return {"fracture": "horizontal", "input_patch": patch, "body_patch": body}


def _make_object(color: int, fracture: str, rng: np.random.Generator) -> dict | None:
    builder = _make_vertical_object if fracture == "vertical" else _make_horizontal_object
    for _ in range(50):
        obj = builder(color, rng)
        if obj is None:
            continue
        body = obj["body_patch"]
        alt = "horizontal" if fracture == "vertical" else "vertical"
        ref_shape = (max(12, body.shape[0] * 3), max(12, body.shape[1] * 3))
        correct, _ = _tile_render(ref_shape, body, fracture, body.shape[0], body.shape[1], bg=BG)
        wrong_axis, _ = _tile_render(ref_shape, body, alt, body.shape[0], body.shape[1], bg=BG)
        with_line, _ = _tile_render(ref_shape, obj["input_patch"], fracture, body.shape[0], body.shape[1], bg=BG)
        if np.array_equal(correct, wrong_axis):
            continue
        if np.array_equal(correct, with_line):
            continue
        return obj
    return None


def _input_mask_ok(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    hh, ww = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    return not np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask)


def _update_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(mask)
    h, w = blocked.shape
    for rr, cc in coords:
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        for dr, dc in NEIGHBORS8 + [(0, 0)]:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_objects(
    shape: tuple[int, int],
    objects: list[dict],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full(shape, BG, dtype=int)
    expected = np.full(shape, BG, dtype=int)
    blocked = np.zeros(shape, dtype=bool)
    out_occ = np.zeros(shape, dtype=bool)

    order = sorted(objects, key=lambda obj: obj["input_patch"].size, reverse=True)
    for obj in order:
        patch = obj["input_patch"]
        body = obj["body_patch"]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(shape[0] - ph + 1) for c in range(shape[1] - pw + 1)]
        rng.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            mask = patch != BG
            if not _input_mask_ok(blocked, mask, r0, c0):
                continue

            correct, copies = _tile_render(shape, body, obj["fracture"], r0, c0, bg=BG)
            if copies < 2:
                continue
            alt_fracture = "horizontal" if obj["fracture"] == "vertical" else "vertical"
            wrong_axis, _ = _tile_render(shape, body, alt_fracture, r0, c0, bg=BG)
            with_line, _ = _tile_render(shape, patch, obj["fracture"], r0, c0, bg=BG)
            if np.array_equal(correct, wrong_axis):
                continue
            if np.array_equal(correct, with_line):
                continue

            correct_mask = correct != BG
            if np.any(out_occ & correct_mask):
                continue

            _paste(grid, patch, r0, c0, transparent=BG)
            expected[correct_mask] = correct[correct_mask]
            out_occ |= correct_mask
            _update_blocked(blocked, mask, r0, c0)
            placed = True
            break

        if not placed:
            return None

    return grid, expected


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        if not np.array_equal(main(inp), expected):
            return False
    except AssertionError:
        return False

    components = _component_coords(inp != BG, connectivity=8)
    if len(components) not in (2, 3):
        return False

    fractures = []
    for coords in components:
        try:
            info = _analyze_object(inp, coords, BG)
        except AssertionError:
            return False
        fractures.append(info["fracture"])
        r0, c0, _r1, _c1 = info["bbox"]
        correct, copies = _tile_render(inp.shape, info["body"], info["fracture"], r0, c0, bg=BG)
        alt = "horizontal" if info["fracture"] == "vertical" else "vertical"
        wrong_axis, _ = _tile_render(inp.shape, info["body"], alt, r0, c0, bg=BG)
        with_line, _ = _tile_render(inp.shape, info["crop"], info["fracture"], r0, c0, bg=BG)
        if copies < 2:
            return False
        if np.array_equal(correct, wrong_axis):
            return False
        if np.array_equal(correct, with_line):
            return False

    return "vertical" in fractures and "horizontal" in fractures


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(17, 25))
    w = int(rng.integers(17, 25))
    n_objects = int(rng.integers(2, 4))
    fractures = ["vertical", "horizontal"] if n_objects == 2 else ["vertical", "horizontal", random.choice(["vertical", "horizontal"])]
    rng.shuffle(fractures)
    colors = [int(c) for c in rng.choice(np.array(BODY_COLORS, dtype=int), size=n_objects, replace=False)]

    objects = []
    for color, fracture in zip(colors, fractures):
        obj = _make_object(color, fracture, rng)
        if obj is None:
            return None
        objects.append(obj)

    placed = _place_objects((h, w), objects, rng)
    if placed is None:
        return None
    inp, expected = placed
    if not _validate_generated(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fracture-tiling ARC input")
