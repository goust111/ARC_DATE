from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
CORE = 5
RING_COLORS = (1, 2, 3, 4, 7)  # blue, red, green, yellow, orange
DIR_TO_INDEX = {"up": 1, "right": 2, "down": 3, "left": 4}
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Ring:
    color: int
    bbox: tuple[int, int, int, int]
    coords: np.ndarray
    gap_index: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_small_filled_square(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h not in (1, 2) or w not in (1, 2) or h != w:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == color))


def _ring_gap_index(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    misses = {
        "up": int(np.count_nonzero(patch[0, :] != color)),
        "right": int(np.count_nonzero(patch[:, -1] != color)),
        "down": int(np.count_nonzero(patch[-1, :] != color)),
        "left": int(np.count_nonzero(patch[:, 0] != color)),
    }
    sides = [name for name, count in misses.items() if count == 1]
    assert len(sides) == 1, "each ring must have exactly one one-cell side gap"
    assert all(count == 0 for name, count in misses.items() if name != sides[0])
    return DIR_TO_INDEX[sides[0]]


def _is_ring(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    on_border = (coords[:, 0] == r0) | (coords[:, 0] == r1) | (coords[:, 1] == c0) | (coords[:, 1] == c1)
    if not np.all(on_border):
        return False
    full_perimeter = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != full_perimeter - 1:
        return False
    try:
        _ring_gap_index(grid, color, (r0, c0, r1, c1))
    except AssertionError:
        return False
    return True


def _scan_color_order(grid: np.ndarray, ring_colors: set[int]) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for value in grid.flat:
        color = int(value)
        if color in ring_colors and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _analyze(grid: np.ndarray) -> tuple[int, list[Ring], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    rings: list[Ring] = []
    core_boxes: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        if _is_ring(g, color, coords):
            rings.append(Ring(color=color, bbox=_bbox(coords), coords=coords, gap_index=_ring_gap_index(g, color, _bbox(coords))))
        else:
            assert _is_small_filled_square(g, color, coords), "non-ring object must be the grey core"
            core_boxes.append(_bbox(coords))

    assert 3 <= len(rings) <= 5, "expected 3 to 5 rings"
    assert len(core_boxes) == 1, "expected exactly one core marker"

    rings.sort(
        key=lambda ring: (
            (ring.bbox[2] - ring.bbox[0] + 1) * (ring.bbox[3] - ring.bbox[1] + 1),
            -ring.bbox[0],
            -ring.bbox[1],
        ),
        reverse=True,
    )
    return bg, rings, core_boxes[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the gap direction of each concentric rectangular ring, then use that
    direction sequence as 1-based indices into the scan-ordered ring colors.
    Keep the ring geometry, the gaps, and the grey core unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, rings, _core_box = _analyze(g)

    ring_colors = {ring.color for ring in rings}
    color_order = _scan_color_order(g, ring_colors)
    assert len(color_order) == len(rings), "ring colors must be unique in the input"

    out = g.copy()
    for ring in rings:
        assert ring.gap_index <= len(color_order), "gap index must select an existing scan-ordered color"
        out[ring.coords[:, 0], ring.coords[:, 1]] = color_order[ring.gap_index - 1]

    assert _get_bg(out) == bg
    return out


def _gap_cell(rng: random.Random, box: tuple[int, int, int, int], gap_index: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if gap_index == 1:
        return r0, rng.randint(c0 + 1, c1 - 1)
    if gap_index == 2:
        return rng.randint(r0 + 1, r1 - 1), c1
    if gap_index == 3:
        return r1, rng.randint(c0 + 1, c1 - 1)
    if gap_index == 4:
        return rng.randint(r0 + 1, r1 - 1), c0
    raise ValueError(gap_index)


def _draw_ring(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    gap_index: int,
    rng: random.Random,
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = _gap_cell(rng, box, gap_index)
    grid[gr, gc] = BG
    return gr, gc


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    ring_count = rng.choices([3, 4, 5], weights=[2, 3, 3], k=1)[0]
    core_size = rng.choice([1, 2])
    inner_pad_h = rng.randint(0, 2)
    inner_pad_w = rng.randint(0, 2)

    inner_h = core_size + 2 + 2 * inner_pad_h
    inner_w = core_size + 2 + 2 * inner_pad_w
    outer_h = inner_h + 4 * (ring_count - 1)
    outer_w = inner_w + 4 * (ring_count - 1)

    pad_top = rng.randint(1, 4)
    pad_bottom = rng.randint(1, 4)
    pad_left = rng.randint(1, 4)
    pad_right = rng.randint(1, 4)

    grid = np.full((outer_h + pad_top + pad_bottom, outer_w + pad_left + pad_right), BG, dtype=int)
    top = pad_top
    left = pad_left

    ring_colors = rng.sample(list(RING_COLORS), ring_count)
    allowed_gap_indices = [1, 2, 3] if ring_count == 3 else [1, 2, 3, 4]
    gap_indices = [rng.choice(allowed_gap_indices) for _ in range(ring_count)]
    target_colors = [ring_colors[index - 1] for index in gap_indices]

    if target_colors == ring_colors:
        return None
    if len(set(target_colors)) < 2:
        return None

    boxes: list[tuple[int, int, int, int]] = []
    gap_cells: list[tuple[int, int]] = []
    for ring_idx in range(ring_count):
        offset = 2 * ring_idx
        box = (top + offset, left + offset, top + outer_h - 1 - offset, left + outer_w - 1 - offset)
        boxes.append(box)
        gap_cells.append(_draw_ring(grid, box, ring_colors[ring_idx], gap_indices[ring_idx], rng))

    inner_box = boxes[-1]
    core_r0 = inner_box[0] + 1 + inner_pad_h
    core_c0 = inner_box[1] + 1 + inner_pad_w
    grid[core_r0 : core_r0 + core_size, core_c0 : core_c0 + core_size] = CORE

    expected = grid.copy()
    for box, gap_cell, color in zip(boxes, gap_cells, target_colors):
        r0, c0, r1, c1 = box
        mask = np.zeros_like(grid, dtype=bool)
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
        expected[mask] = color
        gr, gc = gap_cell
        expected[gr, gc] = BG

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        built = _build_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
