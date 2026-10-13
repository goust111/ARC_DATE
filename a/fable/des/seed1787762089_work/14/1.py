import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6)

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
LAYOUT_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _dominant_non_bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid[grid != BG], return_counts=True)
    assert len(values) > 0
    return int(values[np.argmax(counts)])


def _marker_cells(grid: np.ndarray, grey: int):
    coords = np.argwhere((grid != BG) & (grid != grey))
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    markers.sort()
    return markers


def _candidate_layout(grid: np.ndarray, grey: int, markers, height: int, width: int):
    h, w = grid.shape
    grey_mask = grid == grey
    marker_mask = (grid != BG) & (grid != grey)
    box_mask = np.zeros((h, w), dtype=bool)
    local_masks = []

    for mr, mc, _color in markers:
        r0, c0 = mr + 1, mc + 1
        r1, c1 = r0 + height, c0 + width
        if r1 > h or c1 > w:
            return None

        expanded_r0 = max(0, r0 - 1)
        expanded_c0 = max(0, c0 - 1)
        expanded_r1 = min(h, r1 + 1)
        expanded_c1 = min(w, c1 + 1)
        if np.any(box_mask[expanded_r0:expanded_r1, expanded_c0:expanded_c1]):
            return None

        if np.any(marker_mask[r0:r1, c0:c1]):
            return None

        local = grey_mask[r0:r1, c0:c1]
        if not local.any():
            return None
        if not local[0].any() or not local[-1].any():
            return None
        if not local[:, 0].any() or not local[:, -1].any():
            return None

        box_mask[r0:r1, c0:c1] = True
        local_masks.append(local)

    if np.any(grey_mask & ~box_mask):
        return None

    template = np.zeros((height, width), dtype=bool)
    for local in local_masks:
        template |= local
    if _component_count(template) != 1:
        return None

    return template, local_masks


def _resolve_template(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    grey = _dominant_non_bg_color(g)
    markers = _marker_cells(g, grey)
    assert 3 <= len(markers) <= 5

    min_anchor_r = min(r + 1 for r, _c, _color in markers)
    min_anchor_c = min(c + 1 for _r, c, _color in markers)
    max_height = g.shape[0] - min_anchor_r
    max_width = g.shape[1] - min_anchor_c

    candidates = []
    for height in range(2, max_height + 1):
        for width in range(2, max_width + 1):
            result = _candidate_layout(g, grey, markers, height, width)
            if result is not None:
                template, local_masks = result
                candidates.append((height, width, template, local_masks))

    assert len(candidates) == 1
    return grey, markers, candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _grey, markers, (height, width, template, _locals) = _resolve_template(g)
    out = np.full_like(g, BG)
    for mr, mc, color in markers:
        r0, c0 = mr + 1, mc + 1
        patch = out[r0 : r0 + height, c0 : c0 + width]
        patch[template] = color
    return out


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]):
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        choices = []
        if r < er:
            choices.append((r + 1, c))
        elif r > er:
            choices.append((r - 1, c))
        if c < ec:
            choices.append((r, c + 1))
        elif c > ec:
            choices.append((r, c - 1))
        r, c = random.choice(choices)
        mask[r, c] = True


def _grow_template(height: int, width: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        center = (random.randrange(height), random.randrange(width))
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        for anchor in anchors:
            _random_path(mask, center, anchor)

        target = random.randint(max(mask.sum() + 1, (height * width) // 2), height * width - 1)
        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if random.random() < 0.5:
            mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)

        if mask.shape != (height, width):
            continue
        if _component_count(mask) != 1:
            continue
        if mask.all():
            continue
        side_counts = [
            int(mask[0].sum()),
            int(mask[-1].sum()),
            int(mask[:, 0].sum()),
            int(mask[:, -1].sum()),
        ]
        if min(side_counts) < 2:
            continue
        return mask
    return None


def _make_copies(template: np.ndarray, n_copies: int):
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    top = [cell for cell in coords if cell[0] == 0]
    bottom = [cell for cell in coords if cell[0] == template.shape[0] - 1]
    left = [cell for cell in coords if cell[1] == 0]
    right = [cell for cell in coords if cell[1] == template.shape[1] - 1]
    sides = (top, bottom, left, right)

    for _ in range(120):
        forced_missing = random.sample(coords, n_copies)
        copies = [np.zeros_like(template, dtype=bool) for _ in range(n_copies)]
        keep_prob = random.uniform(0.35, 0.72)

        for cell in coords:
            allowed = [idx for idx in range(n_copies) if forced_missing[idx] != cell]
            assert allowed
            keepers = [idx for idx in allowed if random.random() < keep_prob]
            if not keepers:
                keepers = [random.choice(allowed)]
            for idx in keepers:
                copies[idx][cell] = True

        for idx, missing in enumerate(forced_missing):
            copies[idx][missing] = False
            for side in sides:
                if not any(copies[idx][cell] for cell in side):
                    candidates = [cell for cell in side if cell != missing]
                    if not candidates:
                        break
                    copies[idx][random.choice(candidates)] = True
            else:
                continue
            break
        else:
            if any(np.array_equal(copy, template) for copy in copies):
                continue
            if any(not np.any(copy) for copy in copies):
                continue
            if not np.all(np.logical_or.reduce(copies) == template):
                continue
            return copies

    return None


def _place_instances(shape: tuple[int, int], n_copies: int, box_h: int, box_w: int):
    h, w = shape
    placements = []
    occupied = np.zeros((h, w), dtype=bool)
    for _ in range(PLACEMENT_ATTEMPTS):
        placements.clear()
        occupied[:] = False
        ok = True
        for _idx in range(n_copies):
            placed = False
            for _ in range(LAYOUT_ATTEMPTS):
                r0 = random.randint(1, h - box_h)
                c0 = random.randint(1, w - box_w)
                fr0, fc0 = r0 - 1, c0 - 1
                fr1, fc1 = r0 + box_h - 1, c0 + box_w - 1
                cr0 = max(0, fr0 - 1)
                cc0 = max(0, fc0 - 1)
                cr1 = min(h - 1, fr1 + 1)
                cc1 = min(w - 1, fc1 + 1)
                if np.any(occupied[cr0 : cr1 + 1, cc0 : cc1 + 1]):
                    continue
                occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
                placements.append((r0, c0))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return placements
    return None


def _build_input(template: np.ndarray, copies, markers, placements, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for copy, color, (r0, c0) in zip(copies, markers, placements):
        grid[r0 - 1, c0 - 1] = color
        patch = grid[r0 : r0 + template.shape[0], c0 : c0 + template.shape[1]]
        patch[copy] = GREY
    return grid


def _expected_output(template: np.ndarray, markers, placements, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    for color, (r0, c0) in zip(markers, placements):
        patch = out[r0 : r0 + template.shape[0], c0 : c0 + template.shape[1]]
        patch[template] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_copies = random.randint(3, 5)
    box_h = random.randint(4, 6)
    box_w = random.randint(4, 6)

    template = _grow_template(box_h, box_w)
    if template is None:
        return None

    copies = _make_copies(template, n_copies)
    if copies is None:
        return None

    grid_h = random.randint(max(14, box_h * 2 + 4), 26)
    grid_w = random.randint(max(14, box_w * 2 + 4), 26)
    placements = _place_instances((grid_h, grid_w), n_copies, box_h, box_w)
    if placements is None:
        return None

    marker_colors = list(MARKER_COLORS)
    random.shuffle(marker_colors)
    marker_colors = marker_colors[:n_copies]

    inp = _build_input(template, copies, marker_colors, placements, (grid_h, grid_w))
    out = _expected_output(template, marker_colors, placements, (grid_h, grid_w))
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
