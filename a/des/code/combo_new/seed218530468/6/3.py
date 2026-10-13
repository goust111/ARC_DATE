from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
GREY = 5
ORANGE = 7

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _mask(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_TEMPLATES = (
    _mask("100", "100", "111"),
    _mask("111", "010", "010"),
    _mask("010", "111", "010"),
    _mask("101", "101", "111"),
    _mask("011", "110", "100"),
    _mask("111", "101", "111"),
    _mask("11111", "00100", "00100"),
    _mask("00100", "01110", "11111"),
    _mask("10001", "10101", "11111"),
    _mask("10101", "11111", "00100"),
)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    fences: list[tuple[int, int, int, int]] = []
    for comp in _blue_components(grid):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros((hh, ww), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(box == BLUE, expected):
            fences.append((r0, c0, r1, c1))
    fences.sort()
    assert fences, "expected at least one blue rectangular fence"
    return fences


def _extract_templates(grid: np.ndarray) -> list[np.ndarray]:
    mask = (grid == ORANGE) | (grid == GREY)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected orange templates and grey separators"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    band = grid[r0 : r1 + 1, c0 : c1 + 1]

    templates: list[np.ndarray] = []
    start = 0
    width = band.shape[1]
    while start < width:
        if np.all(band[:, start] == GREY):
            start += 1
            continue
        end = start
        while end < width and not np.all(band[:, end] == GREY):
            end += 1
        segment = band[:, start:end]
        orange = np.argwhere(segment == ORANGE)
        assert orange.size > 0, "each template slot must contain orange pixels"
        sr0, sc0 = orange.min(axis=0)
        sr1, sc1 = orange.max(axis=0)
        templates.append(segment[sr0 : sr1 + 1, sc0 : sc1 + 1] == ORANGE)
        start = end + 1

    assert templates, "expected at least one template"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """Count green dots inside each blue fence, select that numbered lower template, and stamp it below the fence.

    Input structure:
    - Black background with one or more blue one-pixel-thick rectangular fence rings in the upper area.
    - Each fence encloses only black cells and scattered green singleton marker dots.
    - The lower area contains orange template shapes arranged left-to-right, with neighboring templates separated by grey vertical lines.

    Rule:
    - For each blue fence, count the green dots strictly inside the ring to obtain K.
    - Select the K-th template from the left in the orange template band.
    - Repaint that template mask in blue directly below the fence, with the stamp's top row touching the row just below the fence and its horizontal center aligned to the fence center.
    - Keep the original fences, green markers, templates, separators, and all untouched cells unchanged.

    Key insight:
    - The green dots are not objects to copy; their count is an index into the ordered template library shown below, and the chosen template must be reproduced under the matching fence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates = _extract_templates(g)
    fences = _find_fences(g)
    out = g.copy()
    h, w = g.shape

    for r0, c0, r1, c1 in fences:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        k = int(np.count_nonzero(interior == GREEN))
        assert 1 <= k <= len(templates), "fence marker count must map to an existing template"

        template = templates[k - 1]
        th, tw = template.shape
        fence_w = c1 - c0 + 1
        start_r = r1 + 1
        start_c = c0 + (fence_w - tw) // 2
        assert 0 <= start_r and start_r + th <= h
        assert 0 <= start_c and start_c + tw <= w

        region = out[start_r : start_r + th, start_c : start_c + tw]
        assert np.all(region[template] == BG), "stamp area must be empty in valid inputs"
        region[template] = BLUE

    return out


def _transform_template(template: np.ndarray) -> np.ndarray:
    result = template.copy()
    for _ in range(random.randrange(4)):
        result = np.rot90(result)
    if random.random() < 0.5:
        result = np.fliplr(result)
    if random.random() < 0.5:
        result = np.flipud(result)
    return np.array(result, dtype=bool)


def _template_signature(template: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(template.shape), template.tobytes()


def _select_templates(grid_w: int) -> list[np.ndarray] | None:
    count = random.randint(3, 5)
    candidates = list(BASE_TEMPLATES)
    random.shuffle(candidates)
    chosen: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for base in candidates:
        template = _transform_template(base)
        sig = _template_signature(template)
        if sig in seen:
            continue
        chosen.append(template)
        seen.add(sig)
        if len(chosen) == count:
            break

    if len(chosen) != count:
        return None

    total_w = sum(t.shape[1] for t in chosen) + (count - 1)
    if total_w > grid_w:
        return None
    random.shuffle(chosen)
    return chosen


def _choose_fence_size(template: np.ndarray, marker_count: int) -> tuple[int, int]:
    tw = template.shape[1]
    width_choices = [w for w in (5, 7, 9) if w >= tw]
    random.shuffle(width_choices)
    for width in width_choices:
        heights = [4, 5, 6]
        random.shuffle(heights)
        for height in heights:
            if (height - 2) * (width - 2) >= marker_count:
                return height, width
    raise AssertionError("failed to choose fence size")


def _sample_distinct_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    return random.sample(cells, count)


def _place_fences(
    upper_h: int,
    grid_w: int,
    templates: list[np.ndarray],
) -> list[dict[str, object]] | None:
    fence_count = random.randint(1, min(3, len(templates)))
    marker_counts = random.sample(range(1, len(templates) + 1), fence_count)
    specs: list[dict[str, object]] = []
    occupied: list[tuple[int, int, int, int]] = []

    for marker_count in marker_counts:
        template = templates[marker_count - 1]
        fence_h, fence_w = _choose_fence_size(template, marker_count)
        stamp_h = int(template.shape[0])
        box_h = fence_h + stamp_h
        if box_h > upper_h or fence_w > grid_w:
            return None

        placed = False
        for _ in range(200):
            r0 = random.randint(0, upper_h - box_h)
            c0 = random.randint(0, grid_w - fence_w)
            nr0 = r0 - 1
            nc0 = c0 - 1
            nr1 = r0 + box_h
            nc1 = c0 + fence_w
            if any(not (nr1 < or0 or or1 < nr0 or nc1 < oc0 or oc1 < nc0) for or0, oc0, or1, oc1 in occupied):
                continue

            interior_h = fence_h - 2
            interior_w = fence_w - 2
            markers = _sample_distinct_cells(interior_h, interior_w, marker_count)
            specs.append(
                {
                    "marker_count": marker_count,
                    "template": template,
                    "fence_r0": r0,
                    "fence_c0": c0,
                    "fence_h": fence_h,
                    "fence_w": fence_w,
                    "markers": markers,
                }
            )
            occupied.append((nr0, nc0, nr1, nc1))
            placed = True
            break
        if not placed:
            return None

    return specs


def _draw_templates(grid: np.ndarray, templates: list[np.ndarray], top: int) -> None:
    band_h = max(t.shape[0] for t in templates)
    total_w = sum(t.shape[1] for t in templates) + (len(templates) - 1)
    left = random.randint(0, grid.shape[1] - total_w)
    cursor = left
    for idx, template in enumerate(templates):
        th, tw = template.shape
        row = top + (band_h - th) // 2
        grid[row : row + th, cursor : cursor + tw][template] = ORANGE
        cursor += tw
        if idx != len(templates) - 1:
            grid[top : top + band_h, cursor] = GREY
            cursor += 1


def _draw_fences(grid: np.ndarray, fence_specs: list[dict[str, object]]) -> None:
    for spec in fence_specs:
        r0 = int(spec["fence_r0"])
        c0 = int(spec["fence_c0"])
        hh = int(spec["fence_h"])
        ww = int(spec["fence_w"])
        r1 = r0 + hh - 1
        c1 = c0 + ww - 1

        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE

        for rr, cc in spec["markers"]:
            grid[r0 + 1 + rr, c0 + 1 + cc] = GREEN


def _expected_output(grid: np.ndarray, fence_specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in fence_specs:
        template = np.asarray(spec["template"], dtype=bool)
        r0 = int(spec["fence_r0"])
        c0 = int(spec["fence_c0"])
        hh = int(spec["fence_h"])
        ww = int(spec["fence_w"])
        th, tw = template.shape
        sr = r0 + hh
        sc = c0 + (ww - tw) // 2
        out[sr : sr + th, sc : sc + tw][template] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 30)
    grid_w = random.randint(20, 30)
    templates = _select_templates(grid_w)
    if templates is None:
        return None

    band_h = max(t.shape[0] for t in templates)
    min_template_top = max(grid_h // 2, 8)
    max_template_top = grid_h - band_h
    if min_template_top > max_template_top:
        return None
    template_top = random.randint(min_template_top, max_template_top)
    upper_h = template_top - 1
    if upper_h < 7:
        return None

    fence_specs = _place_fences(upper_h, grid_w, templates)
    if fence_specs is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_templates(grid, templates, template_top)
    _draw_fences(grid, fence_specs)

    try:
        parsed_templates = _extract_templates(grid)
        if len(parsed_templates) != len(templates):
            return None
        if any(not np.array_equal(a, b) for a, b in zip(parsed_templates, templates)):
            return None
        expected = _expected_output(grid, fence_specs)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
