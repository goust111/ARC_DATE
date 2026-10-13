import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4

DIRS = ("up", "right", "left", "down")
TEMPLATE_LIBRARY = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
)

GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        comp: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=int)
    return np.kron(mask.astype(int), block).astype(bool)


def _corner_templates(grid: np.ndarray) -> dict[str, np.ndarray]:
    h, w = grid.shape
    templates: dict[str, np.ndarray] = {}
    red_comps = _components(grid == RED)
    assert len(red_comps) == 4, "expected exactly four corner templates"
    for comp in red_comps:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        mask = _component_mask(comp, box)
        scores = {
            "up": r0 + c0,
            "right": r0 + (w - 1 - c1),
            "left": (h - 1 - r1) + c0,
            "down": (h - 1 - r1) + (w - 1 - c1),
        }
        direction = min(scores, key=scores.get)
        assert direction not in templates, "corner templates must map one-to-one to directions"
        templates[direction] = mask
    assert set(templates) == set(DIRS)
    return templates


def _open_frame_direction(mask: np.ndarray) -> str:
    h, w = mask.shape
    assert h >= 4 and w >= 4, "frame must leave room for a non-empty interior"
    assert not np.any(mask[1:-1, 1:-1]), "green pixels must lie only on the rectangle border"

    edge_full = {
        "up": bool(np.all(mask[0, :])),
        "down": bool(np.all(mask[-1, :])),
        "left": bool(np.all(mask[:, 0])),
        "right": bool(np.all(mask[:, -1])),
    }
    missing = [direction for direction, full in edge_full.items() if not full]
    assert len(missing) == 1, "each frame must miss exactly one side"
    direction = missing[0]

    if direction == "up":
        assert int(mask[0, :].sum()) == 2 and mask[0, 0] and mask[0, -1]
    elif direction == "down":
        assert int(mask[-1, :].sum()) == 2 and mask[-1, 0] and mask[-1, -1]
    elif direction == "left":
        assert int(mask[:, 0].sum()) == 2 and mask[0, 0] and mask[-1, 0]
    else:
        assert int(mask[:, -1].sum()) == 2 and mask[0, -1] and mask[-1, -1]

    return direction


def _regions(grid: np.ndarray) -> list[tuple[str, tuple[int, int, int, int], int]]:
    regions: list[tuple[str, tuple[int, int, int, int], int]] = []
    for comp in _components(grid == GREEN):
        box = _bbox(comp)
        mask = _component_mask(comp, box)
        direction = _open_frame_direction(mask)
        r0, c0, r1, c1 = box
        inner = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        ir0, ic0, ir1, ic1 = inner
        marker_count = int(np.count_nonzero(grid[ir0 : ir1 + 1, ic0 : ic1 + 1] == YELLOW))
        assert 1 <= marker_count <= 3, "region must contain one to three yellow markers"
        regions.append((direction, inner, marker_count))
    assert regions, "expected at least one green open frame"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a corner template from the frame opening direction, scale it by the
    number of interior yellow markers, and stamp it into the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "background must be black"

    templates = _corner_templates(g)
    out = g.copy()
    for direction, inner, factor in _regions(g):
        ir0, ic0, ir1, ic1 = inner
        interior_h = ir1 - ir0 + 1
        interior_w = ic1 - ic0 + 1
        scaled = _scale_mask(templates[direction], factor)
        stamp_h, stamp_w = map(int, scaled.shape)
        assert stamp_h <= interior_h and stamp_w <= interior_w, "scaled template must fit inside the frame"
        top = ir0 + (interior_h - stamp_h) // 2
        left = ic0 + (interior_w - stamp_w) // 2
        patch = out[top : top + stamp_h, left : left + stamp_w]
        patch[scaled] = RED
    return out


def _direction_anchor(direction: str, grid_shape: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    mh, mw = mask_shape
    if direction == "up":
        return 0, 0
    if direction == "right":
        return 0, w - mw
    if direction == "left":
        return h - mh, 0
    return h - mh, w - mw


def _draw_corner_templates(grid: np.ndarray, templates: dict[str, np.ndarray]) -> None:
    for direction, mask in templates.items():
        r0, c0 = _direction_anchor(direction, grid.shape, mask.shape)
        patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        patch[mask] = RED


def _draw_open_frame(grid: np.ndarray, direction: str, r0: int, c0: int, outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    if direction != "up":
        grid[r0, c0 : c1 + 1] = GREEN
    if direction != "down":
        grid[r1, c0 : c1 + 1] = GREEN
    if direction != "left":
        grid[r0 : r1 + 1, c0] = GREEN
    if direction != "right":
        grid[r0 : r1 + 1, c1] = GREEN
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _place_markers(
    grid: np.ndarray,
    template: np.ndarray,
    factor: int,
    inner: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    ir0, ic0, ir1, ic1 = inner
    scaled = _scale_mask(template, factor)
    sh, sw = scaled.shape
    interior_h = ir1 - ir0 + 1
    interior_w = ic1 - ic0 + 1
    top = ir0 + (interior_h - sh) // 2
    left = ic0 + (interior_w - sw) // 2
    red_cells = [(top + r, left + c) for r, c in np.argwhere(scaled)]
    assert len(red_cells) >= factor
    for r, c in rng.sample(red_cells, factor):
        grid[r, c] = YELLOW


def _random_templates(rng: random.Random) -> dict[str, np.ndarray]:
    picks = rng.sample(TEMPLATE_LIBRARY, 4)
    return {direction: picks[idx].copy() for idx, direction in enumerate(DIRS)}


def _usable_pads(templates: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    top_pad = max(int(templates["up"].shape[0]), int(templates["right"].shape[0])) + 2
    bottom_pad = max(int(templates["left"].shape[0]), int(templates["down"].shape[0])) + 2
    left_pad = max(int(templates["up"].shape[1]), int(templates["left"].shape[1])) + 2
    right_pad = max(int(templates["right"].shape[1]), int(templates["down"].shape[1])) + 2
    return top_pad, bottom_pad, left_pad, right_pad


def _sector_ranges(start: int, length: int) -> tuple[tuple[int, int], tuple[int, int]]:
    split = start + length // 2
    return (start, split), (split + 1, start + length)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    templates = _random_templates(rng)
    region_count = rng.randint(1, 4)
    directions = rng.sample(DIRS, region_count)

    specs = []
    max_outer_h = 0
    max_outer_w = 0
    for direction in directions:
        factor = rng.randint(1, 3)
        template = templates[direction]
        scaled_h = int(template.shape[0]) * factor
        scaled_w = int(template.shape[1]) * factor
        inner_h = scaled_h + rng.randint(0, 4)
        inner_w = scaled_w + rng.randint(0, 4)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        max_outer_h = max(max_outer_h, outer_h)
        max_outer_w = max(max_outer_w, outer_w)
        specs.append(
            {
                "direction": direction,
                "factor": factor,
                "template": template,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )

    top_pad, bottom_pad, left_pad, right_pad = _usable_pads(templates)
    usable_h = rng.randint(2 * max_outer_h + 3, 2 * max_outer_h + 9)
    usable_w = rng.randint(2 * max_outer_w + 3, 2 * max_outer_w + 9)
    h = top_pad + bottom_pad + usable_h
    w = left_pad + right_pad + usable_w

    grid = np.full((h, w), BG, dtype=int)
    _draw_corner_templates(grid, templates)

    row_bands = _sector_ranges(top_pad, usable_h)
    col_bands = _sector_ranges(left_pad, usable_w)
    sectors = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(specs)
    for spec, (rb, cb) in zip(specs, rng.sample(sectors, len(specs))):
        row_start, row_end = row_bands[rb]
        col_start, col_end = col_bands[cb]
        max_r0 = row_end - spec["outer_h"]
        max_c0 = col_end - spec["outer_w"]
        if max_r0 < row_start or max_c0 < col_start:
            return None
        r0 = rng.randint(row_start, max_r0)
        c0 = rng.randint(col_start, max_c0)
        inner = _draw_open_frame(grid, spec["direction"], r0, c0, spec["outer_h"], spec["outer_w"])
        _place_markers(grid, spec["template"], spec["factor"], inner, rng)

    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
