import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
CORE_PALETTE = (5, 6, 7, 8, 9)
INSTRUCTION_COLORS = (RED, BLUE, GREEN, YELLOW)
GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


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
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_values(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if r0 == r1 and c0 == c1:
        return np.array([int(grid[r0, c0])], dtype=int)
    values: list[np.ndarray] = [grid[r0, c0 : c1 + 1]]
    if r1 > r0:
        values.append(grid[r1, c0 : c1 + 1])
    if c1 > c0:
        values.append(grid[r0 + 1 : r1, c0])
        values.append(grid[r0 + 1 : r1, c1])
    return np.concatenate(values).astype(int, copy=False)


def _shrink_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)


def _is_valid_bbox(bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r1 and c0 <= c1


def _ring_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, tuple[int, int, int, int]] | None:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    mask = np.zeros((h, w), dtype=bool)
    colors = set()
    for r, c in coords:
        mask[r - r0, c - c0] = True
        colors.add(int(grid[r, c]))
    if len(colors) != 1 or BG in colors:
        return None
    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(mask, expected):
        return None
    return int(next(iter(colors))), bbox


def _extract_via_components(grid: np.ndarray) -> tuple[list[int], np.ndarray] | None:
    comps = _components(grid != BG)
    rings: list[tuple[int, tuple[int, int, int, int]]] = []
    non_rings: list[list[tuple[int, int]]] = []
    for comp in comps:
        info = _ring_info(grid, comp)
        if info is None:
            non_rings.append(comp)
        else:
            rings.append(info)
    if not rings or len(non_rings) != 1:
        return None

    ring_boxes = [bbox for _color, bbox in rings]
    ring_boxes_sorted = sorted(ring_boxes, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
    for inner, outer in zip(ring_boxes_sorted, ring_boxes_sorted[1:]):
        ir0, ic0, ir1, ic1 = inner
        or0, oc0, or1, oc1 = outer
        if not (or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1):
            return None

    core_bbox = _bbox_from_coords(non_rings[0])
    for ring_bbox in ring_boxes:
        rr0, rc0, rr1, rc1 = ring_bbox
        cr0, cc0, cr1, cc1 = core_bbox
        if not (rr0 < cr0 and rc0 < cc0 and rr1 > cr1 and rc1 > cc1):
            return None

    rings.sort(key=lambda item: (item[1][2] - item[1][0] + 1) * (item[1][3] - item[1][1] + 1))
    colors = [color for color, _bbox in rings]
    return colors, _crop_bbox(grid, core_bbox)


def _extract_via_peeling(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    bbox = _bbox_from_mask(grid != BG)
    colors: list[int] = []

    while True:
        while _is_valid_bbox(bbox):
            border = _border_values(grid, bbox)
            if not np.all(border == BG):
                break
            bbox = _shrink_bbox(bbox)
        assert _is_valid_bbox(bbox), "failed to locate core sprite"

        border = _border_values(grid, bbox)
        unique = np.unique(border)
        if unique.size == 1 and int(unique[0]) != BG:
            colors.append(int(unique[0]))
            bbox = _shrink_bbox(bbox)
            assert _is_valid_bbox(bbox), "frames cannot consume the whole structure"
            continue
        break

    core = _crop_bbox(grid, bbox)
    return colors, _crop_bbox(core, _bbox_from_mask(core != BG))


def _apply_instruction(sprite: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.fliplr(sprite).copy()
    if color == BLUE:
        return np.flipud(sprite).copy()
    if color == GREEN:
        return np.rot90(sprite, k=3).copy()
    if color == YELLOW:
        return np.rot90(sprite, k=1).copy()
    raise AssertionError(f"unexpected instruction color: {color}")


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _split_extra(extra: int, rng: random.Random) -> tuple[int, int]:
    if extra == 0:
        return 0, 0
    midpoint = extra // 2
    lo = max(0, midpoint - 2)
    hi = min(extra, midpoint + 2)
    first = rng.randint(lo, hi)
    return first, extra - first


def _random_instruction_sequence(rng: random.Random, count: int) -> list[int]:
    seq: list[int] = []
    while len(seq) < count:
        deck = list(INSTRUCTION_COLORS)
        rng.shuffle(deck)
        seq.extend(deck)
    return seq[:count]


def _random_core(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        palette = rng.sample(CORE_PALETTE, rng.randint(2, min(len(CORE_PALETTE), h * w)))
        core = np.array([[rng.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, core.flatten()))) < 2:
            continue
        border = _border_values(core, (0, 0, h - 1, w - 1))
        if len(set(map(int, border.tolist()))) == 1:
            continue
        return core
    raise RuntimeError("failed to generate a non-degenerate core sprite")


def main(grid: np.ndarray) -> np.ndarray:
    """Interpret concentric frame colors as an ordered transform program for the central sprite.

    Input structure:
    - The background is black.
    - A small connected multicolor core sprite is surrounded by several concentric 1-cell rectangular frames.
    - Each frame is monochrome and uses red, blue, green, or yellow to encode one geometric instruction.

    Rule:
    - Read the frame colors from the innermost rectangle to the outermost rectangle.
    - Apply the corresponding transforms to the core sprite in that order: red = horizontal flip, blue = vertical flip, green = rotate 90 degrees clockwise, yellow = rotate 90 degrees counterclockwise.
    - Output only the transformed core sprite, tightly cropped to its own bounding box.

    Key insight:
    - The nested rectangles are not decoration; their depth order forms an instruction queue that must be executed from inside to outside.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _extract_via_components(g)
    if parsed is None:
        frame_colors, core = _extract_via_peeling(g)
    else:
        frame_colors, core = parsed

    out = core.copy()
    for color in frame_colors:
        out = _apply_instruction(out, int(color))
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        core = _random_core(rng)
        max_layers = min(6, (30 - core.shape[0]) // 4, (30 - core.shape[1]) // 4)
        layers = rng.randint(1, max_layers)
        instructions = _random_instruction_sequence(rng, layers)

        expected = core.copy()
        for color in instructions:
            expected = _apply_instruction(expected, color)
        if np.array_equal(expected, core):
            continue

        outer_h = int(core.shape[0] + 4 * layers)
        outer_w = int(core.shape[1] + 4 * layers)
        extra_h = rng.randint(0, min(6, 30 - outer_h))
        extra_w = rng.randint(0, min(6, 30 - outer_w))
        top_pad, bottom_pad = _split_extra(extra_h, rng)
        left_pad, right_pad = _split_extra(extra_w, rng)

        grid = np.full((outer_h + top_pad + bottom_pad, outer_w + left_pad + right_pad), BG, dtype=int)
        core_r0 = top_pad + 2 * layers
        core_c0 = left_pad + 2 * layers
        grid[core_r0 : core_r0 + core.shape[0], core_c0 : core_c0 + core.shape[1]] = core

        for idx, color in enumerate(instructions):
            inset = 2 * (layers - idx - 1)
            bbox = (
                top_pad + inset,
                left_pad + inset,
                top_pad + outer_h - inset - 1,
                left_pad + outer_w - inset - 1,
            )
            _draw_ring(grid, bbox, int(color))

        return grid
    raise RuntimeError("failed to generate a valid sample")
