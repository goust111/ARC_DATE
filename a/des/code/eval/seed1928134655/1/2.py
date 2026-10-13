import random
from collections import deque

import numpy as np


BG_COLOR = 0
# A full-bbox shape and its exact complement cannot both be 4-connected in a
# rectangle. Using 8-connectivity is the minimal consistent relaxation.
COMPONENT_CONNECTIVITY = 8
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_RETRIES = 300
PAIR_COUNT_RANGE = (2, 4)
BBOX_HEIGHTS = tuple(range(5, 10))
BBOX_WIDTHS = tuple(range(5, 10))
CANVAS_HEIGHT_RANGE = (22, 31)
CANVAS_WIDTH_RANGE = (22, 31)
PALETTE = tuple(range(1, 10))


def _background_color(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(values))


def _neighbor_dirs(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    return (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )


def _connected(mask: np.ndarray, connectivity: int) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    dirs = _neighbor_dirs(connectivity)

    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _components(grid: np.ndarray, bg: int, connectivity: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = _neighbor_dirs(connectivity)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _relative_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _pair_components(components: list[dict]) -> list[tuple[int, int]]:
    if not components or len(components) % 2 != 0:
        raise AssertionError("expected a non-empty even number of components")

    rel_masks = [_relative_mask(comp) for comp in components]
    candidates: dict[int, list[int]] = {}
    for i, mask_i in enumerate(rel_masks):
        js: list[int] = []
        for j, mask_j in enumerate(rel_masks):
            if i == j or mask_i.shape != mask_j.shape:
                continue
            if np.array_equal(mask_j, ~mask_i):
                js.append(j)
        if not js:
            raise AssertionError("every component must have at least one complement partner")
        candidates[i] = js

    def backtrack(remaining: tuple[int, ...]) -> list[tuple[int, int]] | None:
        if not remaining:
            return []
        i = remaining[0]
        rest = set(remaining[1:])
        for j in candidates[i]:
            if j not in rest:
                continue
            next_remaining = tuple(x for x in remaining[1:] if x != j)
            result = backtrack(next_remaining)
            if result is not None:
                return [(i, j)] + result
        return None

    pairs = backtrack(tuple(range(len(components))))
    if pairs is None:
        raise AssertionError("failed to find a perfect complement matching")
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary monochrome fragments and replace each pair by a solid box.

    The intended puzzle text asks for 4-connected components, but that is
    topologically inconsistent with "same bbox + exact complement" in a
    rectangle. This solver therefore uses the minimal consistent variant:
    monochrome 8-connected components.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    components = _components(g, bg=bg, connectivity=COMPONENT_CONNECTIVITY)
    pairs = _pair_components(components)
    rel_masks = [_relative_mask(comp) for comp in components]

    out = np.full_like(g, bg)
    for i, j in pairs:
        comp_i = components[i]
        comp_j = components[j]
        h, w = rel_masks[i].shape

        area_i = int(comp_i["area"])
        area_j = int(comp_j["area"])
        if area_i == area_j:
            raise AssertionError("paired components must have different areas")
        fill_color = int(comp_i["color"] if area_i < area_j else comp_j["color"])

        anchor_r, anchor_c = min(
            (int(comp_i["bbox"][0]), int(comp_i["bbox"][1])),
            (int(comp_j["bbox"][0]), int(comp_j["bbox"][1])),
        )
        out[anchor_r : anchor_r + h, anchor_c : anchor_c + w] = fill_color

    return out


def _bbox_full(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_pair_mask(mask: np.ndarray) -> bool:
    comp = ~mask
    if mask.sum() == comp.sum():
        return False
    return (
        _bbox_full(mask)
        and _bbox_full(comp)
        and _connected(mask, COMPONENT_CONNECTIVITY)
        and _connected(comp, COMPONENT_CONNECTIVITY)
    )


def _diagonal_path_mask(height: int, width: int, style: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    if height >= width:
        for r in range(height):
            if style == 0:
                c = int(round(r * (width - 1) / max(1, height - 1)))
            elif style == 1:
                c = int((r * (width - 1)) // max(1, height - 1))
            else:
                c = int(-(-r * (width - 1) // max(1, height - 1)))
            mask[r, c] = True
    else:
        for c in range(width):
            if style == 0:
                r = int(round(c * (height - 1) / max(1, width - 1)))
            elif style == 1:
                r = int((c * (height - 1)) // max(1, width - 1))
            else:
                r = int(-(-c * (height - 1) // max(1, width - 1)))
            mask[r, c] = True

    return mask


def _make_pair_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _diagonal_path_mask(height, width, style=random.randint(0, 2))
    if random.choice((False, True)):
        mask = mask[::-1, :]
    if random.choice((False, True)):
        mask = mask[:, ::-1]
    if not _valid_pair_mask(mask):
        return None
    return mask, ~mask


def _place_boxes(canvas_h: int, canvas_w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    for idx in order:
        box_h, box_w = sizes[idx]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, canvas_h - box_h)
            c0 = random.randint(0, canvas_w - box_w)

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(canvas_h, r0 + box_h + 1)
            cc1 = min(canvas_w, c0 + box_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            occupied[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break

        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    h, w = mask.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[mask] = int(color)


def _generate_canvas_sizes(pair_count: int) -> tuple[int, int]:
    min_side = 20 + 2 * pair_count
    canvas_h = random.randint(max(min_side, CANVAS_HEIGHT_RANGE[0]), CANVAS_HEIGHT_RANGE[1])
    canvas_w = random.randint(max(min_side, CANVAS_WIDTH_RANGE[0]), CANVAS_WIDTH_RANGE[1])
    return canvas_h, canvas_w


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    dim_candidates = [(h, w) for h in BBOX_HEIGHTS for w in BBOX_WIDTHS]
    dims = random.sample(dim_candidates, pair_count)

    pair_masks: list[tuple[np.ndarray, np.ndarray]] = []
    for box_h, box_w in dims:
        masks = _make_pair_masks(box_h, box_w)
        if masks is None:
            return None
        pair_masks.append(masks)

    sizes: list[tuple[int, int]] = []
    for box_h, box_w in dims:
        sizes.extend([(box_h, box_w), (box_h, box_w)])

    canvas_h, canvas_w = _generate_canvas_sizes(pair_count)
    positions = _place_boxes(canvas_h, canvas_w, sizes)
    if positions is None:
        return None

    colors = random.sample(PALETTE, 2 * pair_count)
    grid = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)

    for pair_idx, (mask_a, mask_b) in enumerate(pair_masks):
        pos_a = positions[2 * pair_idx]
        pos_b = positions[2 * pair_idx + 1]
        color_a = colors[2 * pair_idx]
        color_b = colors[2 * pair_idx + 1]

        if random.choice((False, True)):
            mask_a, mask_b = mask_b, mask_a

        _paint_mask(grid, mask_a, pos_a, color_a)
        _paint_mask(grid, mask_b, pos_b, color_b)

    try:
        components = _components(grid, bg=BG_COLOR, connectivity=COMPONENT_CONNECTIVITY)
        if len(components) != 2 * pair_count:
            return None
        _pair_components(components)
        main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
