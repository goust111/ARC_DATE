import random
from collections import Counter

import numpy as np


BACKGROUND = 0
OUTER_BAND = 2
INNER_GAP = 2
QUADRANT_ORDER = ("ul", "ur", "ll", "lr")
PANEL_SIZE_CHOICES = (4, 5, 6)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nonzero_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND)


def _layout(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    panel_h = h - 2 * OUTER_BAND - INNER_GAP
    panel_w = w - 2 * OUTER_BAND - INNER_GAP
    assert panel_h > 0 and panel_h % 2 == 0
    assert panel_w > 0 and panel_w % 2 == 0
    return panel_h // 2, panel_w // 2


def _panel_origins(panel_h: int, panel_w: int) -> dict[str, tuple[int, int]]:
    top = OUTER_BAND
    left = OUTER_BAND
    bottom = OUTER_BAND + panel_h + INNER_GAP
    right = OUTER_BAND + panel_w + INNER_GAP
    return {
        "ul": (top, left),
        "ur": (top, right),
        "ll": (bottom, left),
        "lr": (bottom, right),
    }


def _cue_slices(panel_h: int, panel_w: int) -> dict[str, tuple[slice, slice]]:
    del panel_h, panel_w
    return {
        "ul": (slice(0, OUTER_BAND), slice(0, OUTER_BAND)),
        "ur": (slice(0, OUTER_BAND), slice(-OUTER_BAND, None)),
        "ll": (slice(-OUTER_BAND, None), slice(0, OUTER_BAND)),
        "lr": (slice(-OUTER_BAND, None), slice(-OUTER_BAND, None)),
    }


def _extract_single_color(region: np.ndarray) -> tuple[int, int]:
    colors = _nonzero_colors(region)
    assert len(colors) == 1
    color = colors[0]
    area = int(np.count_nonzero(region == color))
    return color, area


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_rotationally_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_to_panel(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.zeros(mask.shape, dtype=int)
    panel[mask] = int(color)
    return panel


def _sample_rotations() -> list[int]:
    if random.random() < 0.5:
        rots = list(range(4))
        random.shuffle(rots)
        return rots

    repeated = random.randrange(4)
    others = random.sample([v for v in range(4) if v != repeated], 2)
    rots = [repeated, repeated, others[0], others[1]]
    random.shuffle(rots)
    return rots


def _shared_anchor(panel_size: int) -> tuple[int, int]:
    mid = panel_size // 2
    rows = [mid] if panel_size % 2 == 1 else [mid - 1, mid]
    cols = [mid] if panel_size % 2 == 1 else [mid - 1, mid]
    return random.choice(rows), random.choice(cols)


def _grow_mask(
    panel_size: int,
    area: int,
    required: set[tuple[int, int]],
    target: tuple[int, int],
    penalized: set[tuple[int, int]],
) -> np.ndarray | None:
    for _ in range(120):
        cells = set(required)
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < panel_size and 0 <= nc < panel_size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break

            candidates = sorted(frontier)
            weights = []
            for nr, nc in candidates:
                dist = abs(nr - target[0]) + abs(nc - target[1])
                weight = max(0.25, 8.0 - 1.3 * dist)
                if (nr, nc) in penalized and (nr, nc) not in required:
                    weight *= 0.12
                if nr in (0, panel_size - 1) or nc in (0, panel_size - 1):
                    weight *= 0.9
                weights.append(weight)
            cells.add(random.choices(candidates, weights=weights, k=1)[0])

        if len(cells) != area:
            continue
        mask = np.zeros((panel_size, panel_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        if not _is_rotationally_asymmetric(mask):
            continue
        r0, c0, r1, c1 = _bbox(mask)
        if r1 == r0 or c1 == c0:
            continue
        return mask
    return None


def _build_final_masks(panel_size: int) -> list[np.ndarray] | None:
    anchor = _shared_anchor(panel_size)
    targets = [
        (1, 1),
        (1, panel_size - 2),
        (panel_size - 2, 1),
        (panel_size - 2, panel_size - 2),
    ]
    masks: list[np.ndarray] = []
    occupied: set[tuple[int, int]] = set()
    for target in targets:
        area = random.randint(4, min(10, panel_size * panel_size - 1))
        mask = _grow_mask(panel_size, area, {anchor}, target, occupied)
        if mask is None:
            return None
        masks.append(mask)
        occupied |= {tuple(map(int, rc)) for rc in np.argwhere(mask)}

    coverage = np.sum(np.stack(masks).astype(int), axis=0)
    overlap_cells = int(np.count_nonzero(coverage > 1))
    if overlap_cells < 1 or overlap_cells > max(4, panel_size):
        return None
    if any(int(np.count_nonzero(mask & (coverage == 1))) < 2 for mask in masks):
        return None
    return masks


def _compose_layers(panels: list[np.ndarray], rotations: list[int]) -> np.ndarray:
    out = np.zeros_like(panels[0])
    for _, idx in sorted((rotations[i], i) for i in range(4)):
        panel = panels[idx]
        mask = panel != BACKGROUND
        out[mask] = panel[mask]
    return out


def _cue_shape(size: int) -> np.ndarray:
    variants = {
        1: [
            np.array([[1, 0], [0, 0]], dtype=bool),
            np.array([[0, 1], [0, 0]], dtype=bool),
            np.array([[0, 0], [1, 0]], dtype=bool),
            np.array([[0, 0], [0, 1]], dtype=bool),
        ],
        2: [
            np.array([[1, 1], [0, 0]], dtype=bool),
            np.array([[0, 0], [1, 1]], dtype=bool),
            np.array([[1, 0], [1, 0]], dtype=bool),
            np.array([[0, 1], [0, 1]], dtype=bool),
        ],
        3: [
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 1], [0, 1]], dtype=bool),
            np.array([[1, 0], [1, 1]], dtype=bool),
            np.array([[0, 1], [1, 1]], dtype=bool),
        ],
        4: [np.ones((2, 2), dtype=bool)],
    }
    return random.choice(variants[int(size)])


def _rotation_to_cue_size(rotation: int) -> int:
    return 4 if rotation == 0 else int(rotation)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant panel by the size of its same-colored corner cue, then overlay the rotated panels by increasing rotation amount.

    Input structure:
    - A black background contains four equal square panels arranged as a 2x2 layout with a 2-cell black separator band between rows and columns.
    - Each panel contains a single non-black connected monochrome shape, and the four panel colors are distinct.
    - Each outer corner 2x2 cue region contains one small monochrome same-colored blob whose pixel count is 1-4.

    Rule:
    - Match each cue blob to the panel of the same color.
    - Rotate the whole local panel clockwise by `(cue_pixel_count mod 4) * 90` degrees.
    - Overlay the four rotated panels into one panel-sized output, drawing smaller rotation amounts first.
    - If two panels share the same rotation amount, use reading order `upper-left < upper-right < lower-left < lower-right`.

    Key insight:
    - A cue blob's area plays a double role: it sets both how far the matching panel rotates and how high that panel sits in the transparent overlay stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panel_h, panel_w = _layout(g.shape)
    origins = _panel_origins(panel_h, panel_w)
    cue_regions = _cue_slices(panel_h, panel_w)

    rotations_by_color: dict[int, int] = {}
    for quadrant in QUADRANT_ORDER:
        rs, cs = cue_regions[quadrant]
        color, area = _extract_single_color(g[rs, cs])
        assert color not in rotations_by_color
        rotations_by_color[color] = int(area % 4)

    layers: list[tuple[int, int, np.ndarray]] = []
    for idx, quadrant in enumerate(QUADRANT_ORDER):
        r0, c0 = origins[quadrant]
        panel = g[r0 : r0 + panel_h, c0 : c0 + panel_w]
        colors = _nonzero_colors(panel)
        assert len(colors) == 1
        color = colors[0]
        rotation = rotations_by_color[color]
        rotated = np.rot90(panel, -rotation)
        layers.append((rotation, idx, rotated))

    out = np.zeros((panel_h, panel_w), dtype=int)
    for _, _, panel in sorted(layers, key=lambda item: (item[0], item[1])):
        mask = panel != BACKGROUND
        out[mask] = panel[mask]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        panel_size = random.choice(PANEL_SIZE_CHOICES)
        grid_size = 2 * panel_size + 2 * OUTER_BAND + INNER_GAP
        grid = np.zeros((grid_size, grid_size), dtype=int)
        colors = random.sample(list(range(1, 10)), 4)
        rotations = _sample_rotations()
        if len(set(rotations)) < 2:
            continue

        final_masks = _build_final_masks(panel_size)
        if final_masks is None:
            continue

        final_panels = [_mask_to_panel(mask, colors[i]) for i, mask in enumerate(final_masks)]
        expected = _compose_layers(final_panels, rotations)
        visible_colors = set(int(v) for v in np.unique(expected) if int(v) != BACKGROUND)
        if len(visible_colors) < 4:
            continue

        coverage = np.sum(np.stack(final_masks).astype(int), axis=0)
        if int(np.count_nonzero(coverage > 1)) < 1:
            continue

        origins = _panel_origins(panel_size, panel_size)
        for idx, quadrant in enumerate(QUADRANT_ORDER):
            r0, c0 = origins[quadrant]
            panel = np.rot90(final_panels[idx], rotations[idx])
            if len(_nonzero_colors(panel)) != 1:
                break
            if not _is_connected(panel != BACKGROUND):
                break
            area = int(np.count_nonzero(panel))
            if area < 4 or area > 10:
                break
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel
        else:
            cue_regions = _cue_slices(panel_size, panel_size)
            cue_sizes = [_rotation_to_cue_size(rot) for rot in rotations]
            if len(set(cue_sizes)) < 2:
                continue
            for idx, quadrant in enumerate(QUADRANT_ORDER):
                rs, cs = cue_regions[quadrant]
                cue = np.zeros((OUTER_BAND, OUTER_BAND), dtype=int)
                cue_mask = _cue_shape(cue_sizes[idx])
                cue[cue_mask] = colors[idx]
                grid[rs, cs] = cue

            if np.array_equal(main(grid), expected):
                return grid

    raise RuntimeError("failed to generate a valid sample for puzzle 5")
