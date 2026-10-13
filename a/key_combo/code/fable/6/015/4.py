from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BLACK = Color.BLACK
GREY = Color.GREY
ORANGE = Color.ORANGE
PALETTE = tuple(color for color in Color.ALL_COLORS if color not in {BLACK, GREY, ORANGE})
MAX_GENERATION_ATTEMPTS = 400


def _center(comp: dict, axis: int) -> float:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)[axis]


def _is_container_patch(patch: np.ndarray) -> bool:
    h, w = map(int, patch.shape)
    if h < 3 or w < 3:
        return False
    if not np.all(patch[0, :] == BLACK):
        return False
    if not np.all(patch[-1, :] == BLACK):
        return False
    if not np.all(patch[:, 0] == BLACK):
        return False
    if not np.all(patch[:, -1] == BLACK):
        return False
    interior = patch[1:-1, 1:-1]
    if interior.size == 0 or np.any(interior == BLACK):
        return False
    return set(map(int, np.unique(interior))) <= {GREY, ORANGE}


def _parse_scene(grid: np.ndarray) -> tuple[int, dict, list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    containers: list[dict] = []
    pusher: dict | None = None

    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        record = {
            "bbox": tuple(map(int, comp["bbox"])),
            "patch": patch,
        }
        if _is_container_patch(patch):
            containers.append(record)
        else:
            if pusher is not None:
                raise AssertionError("expected exactly one pusher")
            if len(np.unique(patch)) != 1:
                raise AssertionError("pusher must be monochrome")
            pusher = record

    assert pusher is not None, "missing pusher"
    assert len(containers) >= 1, "missing containers"

    row_centers = [_center(pusher, 0), *[_center(comp, 0) for comp in containers]]
    col_centers = [_center(pusher, 1), *[_center(comp, 1) for comp in containers]]
    row_spread = max(row_centers) - min(row_centers)
    col_spread = max(col_centers) - min(col_centers)
    if row_spread <= 0.5:
        axis = 1
    elif col_spread <= 0.5:
        axis = 0
    else:
        raise AssertionError("pusher and containers must share one axis")

    pusher_axis = _center(pusher, axis)
    container_axes = [_center(comp, axis) for comp in containers]
    if pusher_axis < min(container_axes):
        direction = 1
    elif pusher_axis > max(container_axes):
        direction = -1
    else:
        raise AssertionError("pusher must sit at one end of the chain")

    containers.sort(key=lambda comp: direction * _center(comp, axis))
    return bg, pusher, containers, axis, direction


def _shift_bbox(bbox: tuple[int, int, int, int], axis: int, delta: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if axis == 0:
        return r0 + delta, c0, r1 + delta, c1
    return r0, c0 + delta, r1, c1 + delta


def _gap_to_obstacle(
    bbox: tuple[int, int, int, int],
    next_bbox: tuple[int, int, int, int] | None,
    axis: int,
    direction: int,
    shape: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    H, W = map(int, shape)
    if axis == 1:
        if direction > 0:
            return (int(next_bbox[1]) - c1 - 1) if next_bbox is not None else (W - 1 - c1)
        return (c0 - int(next_bbox[3]) - 1) if next_bbox is not None else c0
    if direction > 0:
        return (int(next_bbox[0]) - r1 - 1) if next_bbox is not None else (H - 1 - r1)
    return (r0 - int(next_bbox[2]) - 1) if next_bbox is not None else r0


def _charged_patch(patch: np.ndarray, axis: int, direction: int, steps: int) -> np.ndarray:
    out = np.asarray(patch, dtype=int).copy()
    if steps == 0:
        return out

    h, w = map(int, out.shape)
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if axis == 1:
        key = (lambda rc: (rc[1], rc[0])) if direction > 0 else (lambda rc: (-rc[1], rc[0]))
    else:
        key = (lambda rc: (rc[0], rc[1])) if direction > 0 else (lambda rc: (-rc[0], rc[1]))

    ordered = [rc for rc in sorted(interior, key=key) if int(out[rc]) == GREY]
    assert steps <= len(ordered), "movement exceeds available grey energy"
    for r, c in ordered[:steps]:
        out[r, c] = ORANGE
    return out


def _simulate_relay(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, _pusher, containers, axis, direction = _parse_scene(g)
    out = g.copy()

    for index, comp in enumerate(containers):
        bbox = tuple(map(int, comp["bbox"]))
        patch = np.asarray(comp["patch"], dtype=int)
        next_bbox = None if index + 1 == len(containers) else tuple(map(int, containers[index + 1]["bbox"]))

        gap = _gap_to_obstacle(bbox, next_bbox, axis, direction, g.shape)
        if gap < 0:
            raise AssertionError("containers must start disjoint and ordered")
        steps = min(int(np.sum(patch == GREY)), gap)
        moved_patch = _charged_patch(patch, axis, direction, steps)

        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        new_bbox = _shift_bbox(bbox, axis, direction * steps)
        paste(out, moved_patch, new_bbox[0], new_bbox[1])
        containers[index]["bbox"] = new_bbox
        containers[index]["patch"] = moved_patch

        if steps < gap:
            break

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Relay a bounded push through a chain of bordered containers.

    Input structure:
    - One solid pusher sprite and several separated black-bordered rectangular containers lie on a single row or column.
    - Each container interior uses only grey and orange cells.
    - The far end of the chain points toward a grid edge.

    Rule:
    - The pusher stays fixed.
    - Starting from the nearest container, each container slides away from the pusher.
    - Its maximum travel equals its remaining grey-cell count.
    - Every step turns one grey interior cell on the pusher-facing side orange.
    - A container stops when it reaches the next container or the grid edge; only then does the impulse relay onward.
    - If a container runs out of grey before reaching its target, the chain stops and later containers stay put.

    Key insight:
    - Motion is sequential rather than simultaneous, and the orange growth records the exact distance each container actually traveled.
    """
    return _simulate_relay(input_grid)


def _make_container(inner_len: int, orange_prefix: int) -> np.ndarray:
    assert 0 <= orange_prefix < inner_len
    patch = np.full((3, inner_len + 2), BLACK, dtype=int)
    patch[1, 1 : 1 + orange_prefix] = ORANGE
    patch[1, 1 + orange_prefix : -1] = GREY
    return patch


def _apply_orientation(grid: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "right":
        return grid.copy()
    if orientation == "left":
        return np.fliplr(grid).copy()
    if orientation == "down":
        return np.rot90(grid, -1).copy()
    if orientation == "up":
        return np.rot90(grid, 1).copy()
    raise ValueError(f"unknown orientation: {orientation}")


def _generate_case() -> np.ndarray | None:
    bg, pusher_color = random.sample(PALETTE, 2)
    container_count = random.randint(2, 5)
    inner_lengths = [random.randint(2, 6) for _ in range(container_count)]
    orange_prefixes = [random.randint(0, inner_len - 1) for inner_len in inner_lengths]
    grey_counts = [inner_len - orange for inner_len, orange in zip(inner_lengths, orange_prefixes)]

    break_choices: list[int | None] = [None]
    break_choices.extend(range(container_count))
    break_index = random.choice(break_choices)

    gaps: list[int] = []
    for index, grey_count in enumerate(grey_counts):
        if break_index is None or index < break_index:
            gaps.append(random.randint(1, grey_count))
        elif index == break_index:
            gaps.append(random.randint(grey_count + 1, grey_count + 4))
        else:
            gaps.append(random.randint(1, 4))

    pusher_w = random.randint(1, 3)
    lead_gap = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    widths = [inner_len + 2 for inner_len in inner_lengths]
    total_width = left_margin + pusher_w + lead_gap + sum(widths) + sum(gaps)
    if total_width > 30:
        return None

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    total_height = top_margin + 3 + bottom_margin
    if total_height > 30:
        return None

    grid = np.full((total_height, total_width), bg, dtype=int)
    row0 = top_margin
    grid[row0 : row0 + 3, left_margin : left_margin + pusher_w] = pusher_color

    cursor = left_margin + pusher_w + lead_gap
    for inner_len, orange_prefix, gap in zip(inner_lengths, orange_prefixes, gaps):
        patch = _make_container(inner_len, orange_prefix)
        paste(grid, patch, row0, cursor)
        cursor += patch.shape[1] + gap

    oriented = _apply_orientation(grid, random.choice(("right", "left", "down", "up")))
    if max(oriented.shape) > 30:
        return None
    return oriented


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
