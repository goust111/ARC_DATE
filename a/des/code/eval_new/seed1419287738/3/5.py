from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 8
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
SIDES = ("top", "bottom", "left", "right")
STRETCH_FACTORS = (2, 3)

PAIR_COUNT_RANGE = (2, 4)
GRID_TALL_RANGE = (24, 30)
GRID_WIDE_RANGE = (24, 30)
BORDER_MARGIN = 2
PLACEMENT_GAP = 1

MASK_RETRIES = 100
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


@dataclass(frozen=True)
class Component:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _build_component(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> Component:
    arr = np.array(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    r1 = int(arr[:, 0].max())
    c0 = int(arr[:, 1].min())
    c1 = int(arr[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return Component(color=color, coords=arr, bbox=(r0, c0, r1, c1), mask=mask, area=int(mask.sum()))


def _find_components(grid: np.ndarray, ignored_colors: set[int]) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignored_colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(_build_component(grid, coords, color))

    return components


def _find_marker_side(grid: np.ndarray, bg: int) -> tuple[int, str, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, str, np.ndarray]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue

        coords = np.argwhere(grid == color)
        if coords.shape[0] < 3:
            continue

        on_top = bool(np.all(coords[:, 0] == 0))
        on_bottom = bool(np.all(coords[:, 0] == h - 1))
        on_left = bool(np.all(coords[:, 1] == 0))
        on_right = bool(np.all(coords[:, 1] == w - 1))
        if sum((on_top, on_bottom, on_left, on_right)) != 1:
            continue

        isolated = True
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
                    isolated = False
                    break
            if not isolated:
                break
        if not isolated:
            continue

        if on_top:
            side = "top"
        elif on_bottom:
            side = "bottom"
        elif on_left:
            side = "left"
        else:
            side = "right"
        candidates.append((coords.shape[0], color, side, coords))

    assert candidates, "expected a unique border marker color"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0], "marker color must be distinguishable from other colors"
    count, color, side, coords = candidates[0]
    assert count >= 3, "marker line must contain multiple pixels"
    return color, side, coords


def _stretch_match(small: np.ndarray, big: np.ndarray, axis: str) -> int:
    sh, sw = small.shape
    bh, bw = big.shape

    for factor in STRETCH_FACTORS:
        if axis == "horizontal":
            if bh != sh or bw != sw * factor:
                continue
            blocks = big.reshape(sh, sw, factor)
            if np.array_equal(blocks[:, :, 0], small) and bool(np.all(blocks == blocks[:, :, :1])):
                return factor
        else:
            if bw != sw or bh != sh * factor:
                continue
            blocks = big.reshape(sh, factor, sw)
            if np.array_equal(blocks[:, 0, :], small) and bool(np.all(blocks == blocks[:, :1, :])):
                return factor
    return 0


def _build_pair_graph(masks: list[np.ndarray], axis: str) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    order = sorted(range(len(masks)), key=lambda idx: (int(masks[idx].sum()), masks[idx].shape))
    edges: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(masks))}
    for pos, i in enumerate(order):
        for j in order[pos + 1 :]:
            factor = _stretch_match(masks[i], masks[j], axis)
            if factor:
                edges[i].append((j, factor))
    return order, edges


def _solve_pairing(masks: list[np.ndarray], axis: str) -> tuple[tuple[int, int, int], ...] | None:
    count = len(masks)
    assert count % 2 == 0 and count > 0
    order, edges = _build_pair_graph(masks, axis)
    full_mask = (1 << count) - 1

    @lru_cache(maxsize=None)
    def solve(used_mask: int) -> tuple[tuple[int, int, int], ...] | None:
        if used_mask == full_mask:
            return ()

        entity_idx = next(idx for idx in order if not (used_mask & (1 << idx)))
        for shadow_idx, factor in edges[entity_idx]:
            if used_mask & (1 << shadow_idx):
                continue
            rest = solve(used_mask | (1 << entity_idx) | (1 << shadow_idx))
            if rest is not None:
                return ((entity_idx, shadow_idx, factor),) + rest
        return None

    return solve(0)


def _count_pairings(masks: list[np.ndarray], axis: str, limit: int = 2) -> int:
    order, edges = _build_pair_graph(masks, axis)
    full_mask = (1 << len(masks)) - 1

    @lru_cache(maxsize=None)
    def count(used_mask: int) -> int:
        if used_mask == full_mask:
            return 1

        entity_idx = next(idx for idx in order if not (used_mask & (1 << idx)))
        total = 0
        for shadow_idx, _factor in edges[entity_idx]:
            if used_mask & (1 << shadow_idx):
                continue
            total += count(used_mask | (1 << entity_idx) | (1 << shadow_idx))
            if total >= limit:
                return limit
        return total

    return count(0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each compact shape with its stretched shadow, move the shadow color onto the compact shape, and erase the shadow.

    Input structure:
    - The grid has a dominant background plus several monochrome 4-connected shapes.
    - One border side contains multiple isolated marker pixels of one color; that side indicates the light-source direction.
    - Every non-marker shape belongs to exactly one pair: a smaller compact shape and a larger shadow created by repeating each column 2 or 3 times or each row 2 or 3 times.

    Rule:
    - Read the border markers to determine the stretch axis: left/right markers mean horizontal stretching, top/bottom markers mean vertical stretching.
    - Match every compact shape to the unique larger shape that compresses back to the same mask when grouped into repeated columns or rows of size 2 or 3.
    - Recolor all pixels of the compact shape with its matched shadow color.
    - Erase the matched shadow pixels and erase the border markers to the background color.

    Key insight:
    - The larger partner is not a different silhouette; it is the same shape with one axis duplicated in fixed-size runs, so solving the task requires undoing that one-axis stretch before transferring color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"

    bg = _dominant_color(g)
    marker_color, side, marker_coords = _find_marker_side(g, bg)
    axis = "horizontal" if side in {"left", "right"} else "vertical"

    components = _find_components(g, ignored_colors={bg, marker_color})
    assert components and len(components) % 2 == 0, "expected compact-shadow pairs"

    masks = [component.mask for component in components]
    pairing = _solve_pairing(masks, axis)
    assert pairing is not None and len(pairing) * 2 == len(components), "every shape must have one valid partner"

    out = g.copy()
    out[marker_coords[:, 0], marker_coords[:, 1]] = bg
    for entity_idx, shadow_idx, _factor in pairing:
        entity = components[entity_idx]
        shadow = components[shadow_idx]
        out[shadow.coords[:, 0], shadow.coords[:, 1]] = bg
        out[entity.coords[:, 0], entity.coords[:, 1]] = shadow.color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return mask.shape, tuple(int(x) for x in mask.astype(np.uint8).ravel())


def _sample_entity_mask(axis: str) -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        if axis == "horizontal":
            height = random.randint(2, 5)
            width = random.randint(2, 4)
        else:
            height = random.randint(2, 4)
            width = random.randint(2, 5)

        min_cells = max(4, height + width - 2)
        max_cells = height * width - 1
        if min_cells > max_cells:
            continue
        target = random.randint(min_cells, max_cells)

        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 4 or bool(mask.all()):
            continue
        return mask
    return None


def _is_self_stretched(mask: np.ndarray, axis: str) -> bool:
    if axis == "horizontal":
        for factor in STRETCH_FACTORS:
            if mask.shape[1] % factor != 0:
                continue
            small = mask[:, ::factor]
            if _stretch_match(small, mask, axis):
                return True
    else:
        for factor in STRETCH_FACTORS:
            if mask.shape[0] % factor != 0:
                continue
            small = mask[::factor, :]
            if _stretch_match(small, mask, axis):
                return True
    return False


def _stretch_mask(mask: np.ndarray, axis: str, factor: int) -> np.ndarray:
    stretch_axis = 1 if axis == "horizontal" else 0
    return np.repeat(mask, factor, axis=stretch_axis)


def _choose_canvas(axis: str, pair_count: int) -> tuple[int, int]:
    if axis == "horizontal":
        height = random.randint(GRID_TALL_RANGE[0], GRID_TALL_RANGE[1] - 1)
        width = random.randint(GRID_WIDE_RANGE[0] + pair_count, GRID_WIDE_RANGE[1])
    else:
        height = random.randint(GRID_TALL_RANGE[0] + pair_count, GRID_TALL_RANGE[1])
        width = random.randint(GRID_WIDE_RANGE[0], GRID_WIDE_RANGE[1] - 1)
    return height, width


def _place_items(items: list[dict[str, object]], height: int, width: int) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []

    ordered = sorted(
        items,
        key=lambda item: (
            int(np.prod(np.asarray(item["mask"]).shape)),
            int(np.asarray(item["mask"]).sum()),
        ),
        reverse=True,
    )

    for item in ordered:
        mask = np.asarray(item["mask"], dtype=bool)
        h, w = mask.shape
        rows = list(range(BORDER_MARGIN, height - h - BORDER_MARGIN + 1))
        cols = list(range(BORDER_MARGIN, width - w - BORDER_MARGIN + 1))
        if not rows or not cols:
            return None

        candidates = [(r, c) for r in rows for c in cols]
        random.shuffle(candidates)
        placed_here = False
        for top, left in candidates[:PLACEMENT_RETRIES]:
            r0 = max(0, top - PLACEMENT_GAP)
            c0 = max(0, left - PLACEMENT_GAP)
            r1 = min(height, top + h + PLACEMENT_GAP)
            c1 = min(width, left + w + PLACEMENT_GAP)
            if forbidden[r0:r1, c0:c1].any():
                continue

            coords = np.argwhere(mask)
            rr = coords[:, 0] + top
            cc = coords[:, 1] + left
            if np.any(grid[rr, cc] != BG_COLOR):
                continue

            grid[rr, cc] = int(item["color"])
            forbidden[r0:r1, c0:c1] = True

            placed_item = dict(item)
            placed_item["top"] = top
            placed_item["left"] = left
            placed.append(placed_item)
            placed_here = True
            break

        if not placed_here:
            return None

    return grid, placed


def _marker_positions(length: int) -> list[int] | None:
    max_count = min(5, (length - 2 + 1) // 2)
    if max_count < 3:
        return None
    count = random.randint(3, max_count)
    available = list(range(1, length - 1))
    random.shuffle(available)
    picked: list[int] = []
    for pos in available:
        if all(abs(pos - other) >= 2 for other in picked):
            picked.append(pos)
            if len(picked) == count:
                return sorted(picked)
    return None


def _add_markers(grid: np.ndarray, side: str) -> None:
    if side in {"top", "bottom"}:
        positions = _marker_positions(grid.shape[1])
        assert positions is not None
        row = 0 if side == "top" else grid.shape[0] - 1
        for col in positions:
            grid[row, col] = MARKER_COLOR
    else:
        positions = _marker_positions(grid.shape[0])
        assert positions is not None
        col = 0 if side == "left" else grid.shape[1] - 1
        for row in positions:
            grid[row, col] = MARKER_COLOR


def _item_coords(item: dict[str, object]) -> np.ndarray:
    mask = np.asarray(item["mask"], dtype=bool)
    coords = np.argwhere(mask)
    coords[:, 0] += int(item["top"])
    coords[:, 1] += int(item["left"])
    return coords


def _expected_output(grid: np.ndarray, placed_items: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    marker_coords = np.argwhere(grid == MARKER_COLOR)
    out[marker_coords[:, 0], marker_coords[:, 1]] = BG_COLOR

    entities = {int(item["pair"]): item for item in placed_items if item["kind"] == "entity"}
    shadows = {int(item["pair"]): item for item in placed_items if item["kind"] == "shadow"}
    for pair_id, entity in entities.items():
        shadow = shadows[pair_id]
        shadow_coords = _item_coords(shadow)
        entity_coords = _item_coords(entity)
        out[shadow_coords[:, 0], shadow_coords[:, 1]] = BG_COLOR
        out[entity_coords[:, 0], entity_coords[:, 1]] = int(shadow["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDES)
    axis = "horizontal" if side in {"left", "right"} else "vertical"
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    palette = random.sample(OBJECT_COLORS, 2 * pair_count)

    items: list[dict[str, object]] = []
    signatures: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    masks_for_uniqueness: list[np.ndarray] = []

    for pair_id in range(pair_count):
        entity_color = palette[2 * pair_id]
        shadow_color = palette[2 * pair_id + 1]
        factor = random.choice(STRETCH_FACTORS)

        entity_mask = None
        for _ in range(MASK_RETRIES):
            candidate = _sample_entity_mask(axis)
            if candidate is None:
                continue
            if _is_self_stretched(candidate, axis):
                continue
            if _mask_signature(candidate) in signatures:
                continue

            shadow_mask = _stretch_mask(candidate, axis, factor)
            trial_masks = masks_for_uniqueness + [candidate, shadow_mask]
            if _count_pairings(trial_masks, axis) != 1:
                continue

            entity_mask = candidate
            signatures.add(_mask_signature(candidate))
            masks_for_uniqueness.extend([candidate, shadow_mask])
            items.append(
                {
                    "pair": pair_id,
                    "kind": "entity",
                    "color": entity_color,
                    "mask": entity_mask,
                }
            )
            items.append(
                {
                    "pair": pair_id,
                    "kind": "shadow",
                    "color": shadow_color,
                    "mask": shadow_mask,
                }
            )
            break

        if entity_mask is None:
            return None

    height, width = _choose_canvas(axis, pair_count)
    placed_result = _place_items(items, height, width)
    if placed_result is None:
        return None
    grid, placed_items = placed_result

    _add_markers(grid, side)
    expected = _expected_output(grid, placed_items)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-pairing ARC input")
