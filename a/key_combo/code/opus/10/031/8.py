from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

RANK_TO_COLOR = {1: RED, 2: BLUE, 3: GREEN}
COUNT_OPTIONS = [(2, 3, 4), (2, 3, 5), (2, 4, 5), (3, 4, 5)]
GRID_ROW_RANGE = (16, 22)
GRID_COL_RANGE = (14, 18)
GEN_RETRY_LIMIT = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _crop_component_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _sprite_components(grid: np.ndarray) -> list[dict]:
    components = []
    for coords in _components4(grid == GREY):
        bbox = _bbox(coords)
        if bbox[2] - bbox[0] != 2 or bbox[3] - bbox[1] != 2:
            raise AssertionError(f"every sprite must have a 3x3 bbox, got {bbox}")
        mask = _crop_component_mask(coords, bbox)
        components.append(
            {
                "coords": coords,
                "bbox": bbox,
                "top": (bbox[0], bbox[1]),
                "mask": mask,
                "key": _mask_key(mask),
            }
        )
    return components


def _group_by_shape(components: list[dict]) -> dict[tuple[tuple[int, ...], ...], list[dict]]:
    groups: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for component in components:
        groups[component["key"]].append(component)
    return groups


def _rank_by_shape(groups: dict[tuple[tuple[int, ...], ...], list[dict]]) -> dict[tuple[tuple[int, ...], ...], int]:
    counts = sorted((len(items) for items in groups.values()), reverse=True)
    if len(groups) != 3 or len(set(counts)) != 3:
        raise AssertionError("expected exactly three shape groups with distinct frequencies")
    count_to_rank = {count: rank for rank, count in enumerate(counts, start=1)}
    return {key: count_to_rank[len(items)] for key, items in groups.items()}


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: tuple[int, int], color: int) -> None:
    r0, c0 = top
    rows, cols = np.where(mask)
    out[r0 + rows, c0 + cols] = color


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor 3x3 gray sprites by shape frequency and composite them in frequency-defined depth order.

    Input structure:
    - A black background contains several gray 4-connected sprites.
    - Every sprite has a 3x3 bounding box, and identical sprite types share the same 3x3 binary mask.
    - There are exactly three shape types, with distinct occurrence counts.

    Rule:
    - Extract each gray 4-connected sprite and normalize it to its 3x3 mask.
    - Group sprites by identical masks and rank the three groups by frequency, from most common to least common.
    - Assign colors by that rank: most common -> red, middle -> blue, least common -> green.
    - Rebuild the full-size output on a black canvas by drawing least common first, then middle, then most common, so higher-frequency groups are in front.

    Key insight:
    - The same hidden variable, shape frequency rank, controls both recoloring and layer order, so the solver must count shape types before it can repaint or composite the sprites.
    """
    g = np.asarray(grid, dtype=int)
    groups = _group_by_shape(_sprite_components(g))
    rank_by_shape = _rank_by_shape(groups)

    out = np.full_like(g, BLACK)
    ordered = sorted(groups.items(), key=lambda item: len(item[1]))
    for key, items in ordered:
        color = RANK_TO_COLOR[rank_by_shape[key]]
        mask = items[0]["mask"]
        for item in items:
            _paint_mask(out, mask, item["top"], color)
    return out


def _enumerate_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for cells in range(3, 7):
        for combo in itertools.combinations(range(9), cells):
            mask = np.zeros((3, 3), dtype=bool)
            for index in combo:
                mask[index // 3, index % 3] = True
            if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
                continue
            if len(_components4(mask)) != 1:
                continue
            masks.append(mask)
    return masks


SHAPE_MASKS = _enumerate_masks()


def _relative_overlap_deltas() -> dict[tuple[int, int], list[tuple[int, int]]]:
    legal: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, mask_a in enumerate(SHAPE_MASKS):
        cells_a = {(int(r), int(c)) for r, c in zip(*np.where(mask_a))}
        for j, mask_b in enumerate(SHAPE_MASKS):
            base_b = {(int(r), int(c)) for r, c in zip(*np.where(mask_b))}
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    row_overlap = max(0, dr) <= min(2, dr + 2)
                    col_overlap = max(0, dc) <= min(2, dc + 2)
                    if not (row_overlap and col_overlap):
                        continue
                    shifted_b = {(r + dr, c + dc) for r, c in base_b}
                    if cells_a & shifted_b:
                        continue
                    touches = False
                    for r, c in cells_a:
                        for nr, nc in _neighbors4(r, c):
                            if (nr, nc) in shifted_b:
                                touches = True
                                break
                        if touches:
                            break
                    if not touches:
                        legal[(i, j)].append((dr, dc))
    return legal


LEGAL_OVERLAP_DELTAS = _relative_overlap_deltas()


def _build_triple_options() -> list[tuple[tuple[int, int, int], list[tuple[int, int, int]]]]:
    options = []
    for triple in itertools.combinations(range(len(SHAPE_MASKS)), 3):
        orders = []
        for order in itertools.permutations(range(3)):
            if LEGAL_OVERLAP_DELTAS[(triple[order[0]], triple[order[1]])] and LEGAL_OVERLAP_DELTAS[
                (triple[order[1]], triple[order[2]])
            ]:
                orders.append(order)
        if orders:
            options.append((triple, orders))
    return options


TRIPLE_OPTIONS = _build_triple_options()


def _absolute_cells(mask: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top
    rows, cols = np.where(mask)
    return {(r0 + int(r), c0 + int(c)) for r, c in zip(rows, cols)}


def _can_place(mask: np.ndarray, top: tuple[int, int], placed: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0 = top
    if r0 < 0 or c0 < 0 or r0 + 3 > h or c0 + 3 > w:
        return False
    new_cells = _absolute_cells(mask, top)
    occupied = set().union(*(item["cells"] for item in placed)) if placed else set()
    if new_cells & occupied:
        return False
    for r, c in new_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_chain(
    masks: list[np.ndarray],
    triple: tuple[int, int, int],
    order: tuple[int, int, int],
    shape: tuple[int, int],
) -> list[dict] | None:
    h, w = shape
    t0, t1, t2 = order
    deltas01 = LEGAL_OVERLAP_DELTAS[(triple[t0], triple[t1])][:]
    deltas12 = LEGAL_OVERLAP_DELTAS[(triple[t1], triple[t2])][:]
    random.shuffle(deltas01)
    random.shuffle(deltas12)

    for _ in range(300):
        base = (random.randint(0, h - 3), random.randint(0, w - 3))
        if not _can_place(masks[t0], base, [], shape):
            continue
        first = {"type": t0, "top": base, "cells": _absolute_cells(masks[t0], base), "bbox": (base[0], base[1], base[0] + 2, base[1] + 2)}
        for d01 in deltas01:
            top1 = (base[0] + d01[0], base[1] + d01[1])
            if not _can_place(masks[t1], top1, [first], shape):
                continue
            second = {
                "type": t1,
                "top": top1,
                "cells": _absolute_cells(masks[t1], top1),
                "bbox": (top1[0], top1[1], top1[0] + 2, top1[1] + 2),
            }
            for d12 in deltas12:
                top2 = (top1[0] + d12[0], top1[1] + d12[1])
                if not _can_place(masks[t2], top2, [first, second], shape):
                    continue
                third = {
                    "type": t2,
                    "top": top2,
                    "cells": _absolute_cells(masks[t2], top2),
                    "bbox": (top2[0], top2[1], top2[0] + 2, top2[1] + 2),
                }
                return [first, second, third]
    return None


def _place_remaining(
    masks: list[np.ndarray],
    counts: list[int],
    placed: list[dict],
    shape: tuple[int, int],
) -> list[dict] | None:
    remaining: list[int] = []
    for type_index, count in enumerate(counts):
        remaining.extend([type_index] * (count - 1))
    random.shuffle(remaining)

    for type_index in remaining:
        candidates = []
        for r in range(shape[0] - 2):
            for c in range(shape[1] - 2):
                top = (r, c)
                if _can_place(masks[type_index], top, placed, shape):
                    candidates.append(top)
        if not candidates:
            return None
        top = random.choice(candidates)
        placed.append(
            {
                "type": type_index,
                "top": top,
                "cells": _absolute_cells(masks[type_index], top),
                "bbox": (top[0], top[1], top[0] + 2, top[1] + 2),
            }
        )
    return placed


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for item in placed:
        for r, c in item["cells"]:
            grid[r, c] = GREY
    return grid


def _render_expected(shape: tuple[int, int], masks: list[np.ndarray], placed: list[dict], counts: list[int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    frequency_order = sorted(range(3), key=lambda t: counts[t], reverse=True)
    rank_by_type = {type_index: rank for rank, type_index in enumerate(frequency_order, start=1)}
    for type_index in sorted(range(3), key=lambda t: counts[t]):
        color = RANK_TO_COLOR[rank_by_type[type_index]]
        for item in placed:
            if item["type"] == type_index:
                _paint_mask(out, masks[type_index], item["top"], color)
    return out


def _count_groups_from_grid(grid: np.ndarray) -> Counter:
    return Counter(component["key"] for component in _sprite_components(grid))


def _first_seen_order(placed: list[dict]) -> list[int]:
    first_top = {type_index: min(item["top"] for item in placed if item["type"] == type_index) for type_index in range(3)}
    return sorted(range(3), key=lambda type_index: first_top[type_index])


def _overlap_pair_count(placed: list[dict]) -> int:
    total = 0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            total += int(_bbox_overlap(placed[i]["bbox"], placed[j]["bbox"]))
    return total


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_ROW_RANGE),
        random.randint(*GRID_COL_RANGE),
    )
    triple, orders = random.choice(TRIPLE_OPTIONS)
    masks = [SHAPE_MASKS[index] for index in triple]
    counts = list(random.choice(COUNT_OPTIONS))
    random.shuffle(counts)

    frequency_order = sorted(range(3), key=lambda t: counts[t], reverse=True)
    area_order = sorted(range(3), key=lambda t: int(masks[t].sum()), reverse=True)
    if area_order == frequency_order:
        return None

    order = random.choice(orders)
    placed = _place_chain(masks, triple, order, shape)
    if placed is None:
        return None
    placed = _place_remaining(masks, counts, placed, shape)
    if placed is None:
        return None

    if _overlap_pair_count(placed) < 2:
        return None
    if _first_seen_order(placed) == frequency_order:
        return None

    grid = _render_input(shape, placed)
    detected = _count_groups_from_grid(grid)
    expected_groups = Counter(_mask_key(masks[type_index]) for type_index in range(3) for _ in range(counts[type_index]))
    if detected != expected_groups:
        return None

    expected = _render_expected(shape, masks, placed, counts)
    if not np.array_equal(main(grid), expected):
        return None
    if set(map(int, np.unique(expected))) != {BLACK, BLUE, RED, GREEN}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
