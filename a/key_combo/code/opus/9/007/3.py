import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SPRITE_COLOR_POOL = (1, RED, GREEN, YELLOW, PINK, ORANGE, TEAL)
GEN_RETRIES = 400
SHAPE_RETRIES = 300
MUTATION_RETRIES = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = {int(v) for v in np.unique(grid)}
    if exclude is not None:
        colors -= {int(v) for v in exclude}
    return sorted(colors)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _xor_area(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.count_nonzero(mask_a ^ mask_b))


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _is_valid_shape(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    if not (5 <= area <= 15):
        return False
    if area == h * w:
        return False
    if bbox(mask) != (0, 0, h - 1, w - 1):
        return False
    return len(components_mask(mask)) == 1


def _random_manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        steps = []
        if r < tr:
            steps.append((1, 0))
        elif r > tr:
            steps.append((-1, 0))
        if c < tc:
            steps.append((0, 1))
        elif c > tc:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return list(out)


def _generate_reference_mask(h: int, w: int) -> np.ndarray | None:
    max_area = min(12, h * w - 4)
    if max_area < 8:
        return None
    for _ in range(SHAPE_RETRIES):
        mask = np.zeros((h, w), dtype=bool)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        cells = {anchors[0]}
        for point in anchors[1:]:
            target = random.choice(tuple(cells))
            cells |= _random_manhattan_path(point, target)
        for r, c in cells:
            mask[r, c] = True
        min_area = max(8, int(mask.sum()))
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        while int(mask.sum()) < target_area:
            frontier = _frontier(mask)
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
        if _is_valid_shape(mask):
            return mask
    return None


def _same_area_variant(reference: np.ndarray, swaps: int, forbidden: set[bytes]) -> np.ndarray | None:
    fg = [tuple(map(int, rc)) for rc in np.argwhere(reference)]
    bg = [tuple(map(int, rc)) for rc in np.argwhere(~reference)]
    for _ in range(MUTATION_RETRIES):
        remove_cells = random.sample(fg, swaps)
        add_cells = random.sample(bg, swaps)
        candidate = reference.copy()
        for r, c in remove_cells:
            candidate[r, c] = False
        for r, c in add_cells:
            candidate[r, c] = True
        key = _mask_key(candidate)
        if key in forbidden:
            continue
        if not _is_valid_shape(candidate):
            continue
        if _xor_area(candidate, reference) != 2 * swaps:
            continue
        return candidate
    return None


def _find_separator_row(grid: np.ndarray) -> int:
    color_counts = Counter(map(int, grid.ravel()))
    candidates = []
    for r in range(grid.shape[0]):
        row_color = int(grid[r, 0])
        if np.all(grid[r] == row_color) and color_counts[row_color] == grid.shape[1]:
            candidates.append(r)
    assert len(candidates) == 1, f"expected one separator row, got {candidates}"
    return int(candidates[0])


def _extract_panel(panel: np.ndarray) -> dict:
    colors = colors_present(panel, exclude={BLACK})
    assert len(colors) == 1, f"expected exactly one foreground color in panel, got {colors}"
    fg = panel != BLACK
    comps = components_mask(fg)
    assert len(comps) == 1, f"expected one connected foreground object, got {len(comps)}"
    box = bbox(fg)
    assert box is not None
    assert box == (0, 0, panel.shape[0] - 1, panel.shape[1] - 1), f"shape must occupy full panel bbox, got {box}"
    cropped = crop_bbox(panel, box)
    mask = cropped != BLACK
    assert _is_valid_shape(mask), "panel shape violates ARC generation constraints"
    return {"color": int(colors[0]), "mask": mask, "grid": cropped}


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder the three stacked sprites by increasing XOR disagreement area to the bottom reference mask.

    Input structure:
    - The grid has a single full-width separator row whose color appears nowhere else.
    - Above the separator are three vertically stacked sprite panels of equal height; each panel contains one irregular 4-connected monochrome sprite on black.
    - Below the separator is one reference panel containing a single irregular monochrome mask on black with the same bounding-box size as each sprite panel.

    Rule:
    - Identify the separator row, infer the common panel height, and split the upper area into its three sprite panels.
    - Treat every non-black cell as foreground, and for each sprite count the XOR disagreement area against the reference mask: cells where exactly one of the two panels has foreground.
    - Sort the sprites by that disagreement area in ascending order.
    - Output only the sorted sprite panels stacked from top to bottom, preserving each sprite's original color and shape.

    Key insight:
    - The lower mask is not copied into the answer; it is a similarity ruler whose XOR area with each sprite determines the final stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_r = _find_separator_row(g)
    panel_h = g.shape[0] - sep_r - 1
    assert panel_h > 0 and sep_r == 3 * panel_h, "separator must split 3H rows above and H rows below"

    reference = _extract_panel(g[sep_r + 1 :])
    ref_mask = reference["mask"]

    sprites = []
    for idx in range(3):
        panel = g[idx * panel_h : (idx + 1) * panel_h]
        info = _extract_panel(panel)
        xor_area = _xor_area(info["mask"], ref_mask)
        sprites.append((xor_area, idx, info["grid"]))

    xor_values = [score for score, _, _ in sprites]
    assert len(set(xor_values)) == 3, f"sprite XOR areas must be distinct, got {xor_values}"
    sprites.sort(key=lambda item: item[0])
    return np.vstack([panel for _, _, panel in sprites])


def _paint(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _validate_instance(grid: np.ndarray, expected: np.ndarray) -> None:
    sep_r = _find_separator_row(grid)
    panel_h = grid.shape[0] - sep_r - 1
    assert panel_h == sep_r // 3
    assert grid.shape == (4 * panel_h + 1, grid.shape[1])
    assert panel_h >= 4

    reference = _extract_panel(grid[sep_r + 1 :])
    ref_mask = reference["mask"]
    sprite_infos = []
    for idx in range(3):
        panel = grid[idx * panel_h : (idx + 1) * panel_h]
        info = _extract_panel(panel)
        sprite_infos.append(info)

    sprite_colors = [info["color"] for info in sprite_infos]
    assert len(set(sprite_colors)) == 3, "sprite colors must be distinct"

    sprite_areas = [int(info["mask"].sum()) for info in sprite_infos]
    assert len(set(sprite_areas)) == 1, "all sprite areas should match to rule out area-based sorting"

    xor_values = [_xor_area(info["mask"], ref_mask) for info in sprite_infos]
    assert len(set(xor_values)) == 3, f"XOR areas must be unique, got {xor_values}"
    assert sorted(xor_values) != xor_values, "input order should not already match the sorted order"

    result = main(grid)
    assert np.array_equal(result, expected), "main() disagrees with generator expectation"
    assert result.shape == (3 * panel_h, grid.shape[1])


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(4, 6)
    possible_widths = [w for w in range(4, 7) if h * w >= 12 and 4 * h + 1 <= 30]
    if not possible_widths:
        return None
    w = random.choice(possible_widths)

    reference_mask = _generate_reference_mask(h, w)
    if reference_mask is None:
        return None

    used = {_mask_key(reference_mask)}
    sprites: list[tuple[int, np.ndarray]] = []
    for swaps in (1, 2, 3):
        variant = _same_area_variant(reference_mask, swaps, used)
        if variant is None:
            return None
        used.add(_mask_key(variant))
        sprites.append((2 * swaps, variant))

    colors = random.sample(SPRITE_COLOR_POOL, 3)
    ordered_ids = [0, 1, 2]
    perm = ordered_ids[:]
    while perm == ordered_ids:
        random.shuffle(perm)

    sprite_panels = [_paint(sprites[idx][1], colors[idx]) for idx in range(3)]
    top = [sprite_panels[idx] for idx in perm]
    reference_panel = _paint(reference_mask, MAROON)

    grid = np.vstack(
        [
            top[0],
            top[1],
            top[2],
            np.full((1, w), GREY, dtype=int),
            reference_panel,
        ]
    )
    expected = np.vstack(sprite_panels)
    _validate_instance(grid, expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        made = _gen_attempt()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid XOR-sorting ARC input")
