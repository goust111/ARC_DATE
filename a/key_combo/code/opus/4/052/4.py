from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 7]
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _color_groups(grid: np.ndarray, bg: int) -> list[dict]:
    groups: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = np.argwhere(grid == color)
        comps = _components_mask(grid == color)
        groups.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
                "component_count": len(comps),
                "component_areas": sorted((int(comp.shape[0]) for comp in comps), reverse=True),
            }
        )
    groups.sort(key=lambda group: (-int(group["area"]), group["bbox"][0], group["bbox"][1], int(group["color"])))
    return groups


def _group_patch(grid: np.ndarray, group: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = group["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = group["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(group["color"])
    return patch


def _output_from_patch(patch: np.ndarray, sep_color: int, bg: int) -> np.ndarray:
    h, w = patch.shape
    out = np.full((2 * h, 2 * w), int(bg), dtype=int)
    out[:h, :w] = patch
    out[h:, w:] = patch
    gap_rows = [r for r in range(h) if np.all(patch[r] == int(bg))]
    assert gap_rows, "expected at least one all-background separator row"
    for row in gap_rows:
        out[row, :] = int(sep_color)
        out[row + h, :] = int(sep_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the largest split row-band object, copy it on the diagonal, and color its blank separator rows with the runner-up color.

    Input structure:
    - The background is black.
    - One foreground color occupies two separated horizontal bands inside an odd-sized bounding box, so that the cropped box contains at least one fully black row.
    - Other colors form disconnected distractor objects elsewhere; the second-largest color group matters only by its color.

    Rule:
    - Rank non-background colors by total occupied area, breaking ties by the upper-left corner of each color's full bounding box.
    - Crop the tight bounding box of the largest color group, preserving its black internal rows.
    - Create a 2H x 2W black canvas and paste the crop into the upper-left and lower-right blocks.
    - For every row that is entirely black in the crop, fill that row and its corresponding row in the lower copy with the second-largest color across the full output width.

    Key insight:
    - The source pattern is identified by total area of one repeated color across separated bands, while the runner-up object contributes only the separator-row color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    groups = _color_groups(g, bg)
    assert len(groups) >= 2, "expected at least two foreground color groups"

    src = groups[0]
    sep = groups[1]
    patch = _group_patch(g, src, bg)
    h, w = patch.shape
    assert h % 2 == 1 and w % 2 == 1, "source bbox must be odd-sized"
    return _output_from_patch(patch, int(sep["color"]), bg)


def _random_connected_patch(area: int, color: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)

    for _ in range(random.randint(0, 3)):
        patch = np.rot90(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch.copy()


def _make_source_patch(color: int) -> np.ndarray:
    mode = random.choice(["thin", "thick"])
    if mode == "thin":
        width = random.choice([7, 9, 11])
        candidates = [
            (a, b)
            for a in range(5, min(width, 8) + 1)
            for b in range(5, min(width, 8) + 1)
            if 12 <= a + b <= 20
        ]
        top_w, bot_w = random.choice(candidates)
        patch = np.zeros((3, width), dtype=int)
        patch[0, :top_w] = int(color)
        patch[2, width - bot_w :] = int(color)
        return patch

    width = random.choice([5, 7])
    candidates = [
        (a, b)
        for a in (3, 4)
        for b in (3, 4)
        if 12 <= 2 * (a + b) <= 20
    ]
    top_w, bot_w = random.choice(candidates)
    patch = np.zeros((5, width), dtype=int)
    patch[0:2, :top_w] = int(color)
    patch[3:5, width - bot_w :] = int(color)
    return patch


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rows, cols = np.where(patch != int(bg))
    for rr, cc in zip(rows, cols):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if int(grid[r, c]) != int(bg):
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < gh and 0 <= nc < gw and int(grid[nr, nc]) != int(bg):
                    return False
    return True


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _place_random(grid: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    gh, gw = grid.shape
    h, w = patch.shape
    if h > gh or w > gw:
        return False
    candidates = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if _can_place(grid, patch, r0, c0, bg):
            _paste(grid, patch, r0, c0, bg)
            return True
    return False


def _wrong_largest_component_output(grid: np.ndarray, bg: int, sep_color: int) -> np.ndarray:
    comps = _find_components(grid, bg)
    comps.sort(key=lambda comp: (-int(comp["area"]), comp["bbox"][0], comp["bbox"][1], int(comp["color"])))
    largest = comps[0]
    r0, c0, r1, c1 = largest["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = largest["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(largest["color"])
    h, w = patch.shape
    out = np.full((2 * h, 2 * w), int(bg), dtype=int)
    out[:h, :w] = patch
    out[h:, w:] = patch
    gap_rows = [r for r in range(h) if np.all(patch[r] == int(bg))]
    for row in gap_rows:
        out[row, :] = int(sep_color)
        out[row + h, :] = int(sep_color)
    return out


def _validate_instance(grid: np.ndarray) -> None:
    bg = _get_bg(grid)
    groups = _color_groups(grid, bg)
    assert 3 <= len(groups) <= 4
    assert all(groups[i]["area"] > groups[i + 1]["area"] for i in range(len(groups) - 1))

    src = groups[0]
    sep = groups[1]
    assert int(src["component_count"]) == 2
    assert int(sep["component_count"]) == 1
    assert max(src["component_areas"]) < int(sep["area"]), "largest single component must not be the source"

    patch = _group_patch(grid, src, bg)
    h, w = patch.shape
    assert h % 2 == 1 and w % 2 == 1
    gap_rows = [r for r in range(h) if np.all(patch[r] == int(bg))]
    assert len(gap_rows) == 1
    assert 12 <= int(src["area"]) <= 20
    assert 5 <= int(sep["area"]) <= 10
    assert all(2 <= int(group["area"]) <= 4 for group in groups[2:])

    out = main(grid)
    oracle = _output_from_patch(patch, int(sep["color"]), bg)
    assert np.array_equal(out, oracle)

    plain = np.full_like(out, int(bg))
    plain[:h, :w] = patch
    plain[h:, w:] = patch
    assert not np.array_equal(out, plain), "separator fill must affect the output"

    wrong = _wrong_largest_component_output(grid, bg, int(sep["color"]))
    assert not np.array_equal(out, wrong), "grouping by repeated color must be necessary"


def _attempt() -> np.ndarray | None:
    small_count = random.choice([1, 2])
    colors = random.sample(PALETTE, 2 + small_count)
    src_color = colors[0]
    sep_color = colors[1]
    extra_colors = colors[2:]

    src_patch = _make_source_patch(src_color)
    src_component_areas = [int(comp.shape[0]) for comp in _components_mask(src_patch == src_color)]
    second_candidates = [area for area in range(5, 11) if area not in src_component_areas]
    second_area = random.choice(second_candidates)
    sep_patch = _random_connected_patch(second_area, sep_color)

    small_areas_pool = [area for area in (2, 3, 4) if area != second_area]
    random.shuffle(small_areas_pool)
    small_areas = small_areas_pool[:small_count]
    if len(small_areas) != small_count:
        return None

    extra_patches = [_random_connected_patch(area, color) for area, color in zip(small_areas, extra_colors)]

    sh, sw = src_patch.shape
    gh_min = max(10, sh + 5)
    gw_min = max(12, sw + 5)
    gh = random.randint(gh_min, min(18, gh_min + 5))
    gw = random.randint(gw_min, min(20, gw_min + 6))
    grid = np.zeros((gh, gw), dtype=int)

    if not _place_random(grid, src_patch, BG):
        return None
    if not _place_random(grid, sep_patch, BG):
        return None
    for patch in extra_patches:
        if not _place_random(grid, patch, BG):
            return None

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
