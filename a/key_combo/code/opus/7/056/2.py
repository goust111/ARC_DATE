from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 5
NON_BG_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
TEMPLATE_COLORS = [1, 2, 3, 4]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for idx, coords in enumerate(_components_from_mask(grid != bg)):
        rr, cc = coords[:, 0], coords[:, 1]
        colors = sorted({int(v) for v in grid[rr, cc]})
        comps.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(len(coords)),
                "colors": colors,
            }
        )
    return comps


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    bitmask = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, bitmask.shape)), bitmask.tobytes()


def _is_legend_component(comp: dict) -> bool:
    if comp["area"] < 2:
        return False
    coords = comp["coords"]
    rows = sorted(int(r) for r in coords[:, 0])
    cols = {int(c) for c in coords[:, 1]}
    if len(cols) != 1 or len(rows) != comp["area"]:
        return False
    return rows[-1] - rows[0] + 1 == len(rows)


def _find_legend_component(components: list[dict]) -> dict:
    candidates = [comp for comp in components if _is_legend_component(comp)]
    if not candidates:
        raise ValueError("legend component not found")
    return max(candidates, key=lambda comp: (comp["bbox"][1], comp["area"], -comp["bbox"][0]))


def _translation_subset(fragment_coords: list[tuple[int, int]], template_coords: set[tuple[int, int]]) -> bool:
    if len(fragment_coords) > len(template_coords):
        return False
    template_points = list(template_coords)
    for tr, tc in template_points:
        for fr, fc in fragment_coords:
            dr, dc = tr - fr, tc - fc
            if all((r + dr, c + dc) in template_coords for r, c in fragment_coords):
                return True
    return False


def _matches_template(fragment_mask: np.ndarray, template_mask: np.ndarray) -> bool:
    fragment_coords = [tuple(map(int, rc)) for rc in np.argwhere(fragment_mask)]
    for rot in range(4):
        rotated = np.rot90(template_mask, rot)
        template_coords = {tuple(map(int, rc)) for rc in np.argwhere(rotated)}
        if _translation_subset(fragment_coords, template_coords):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor matched template fragments according to their spatial rank and the right-edge legend.

    Input structure:
    - A gray background contains one multicolor template component near the upper-left, several scattered monochrome fragments, and a compact one-column legend on the right edge.
    - The template is the connected component with the most distinct colors.
    - Each fragment is a 4-connected subset of some rotation of the template's occupied cells, while the legend is a vertical run of one pixel per row.

    Rule:
    - Identify the right-edge vertical legend and the multicolor template.
    - For every other foreground component, verify that its occupied shape can align as a subset of one rotated template mask.
    - Rank the matching fragments by bounding-box top row, then left column; rank legend pixels from top to bottom.
    - Recolor every pixel of fragment k with the color of legend pixel k, leaving the template, legend, and background unchanged.

    Key insight:
    - The template is only used to validate fragment identity; the final color comes solely from spatial ordering matched against the compact legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _extract_components(g, bg)
    if not components:
        return g.copy()

    legend = _find_legend_component(components)
    non_legend = [comp for comp in components if comp["id"] != legend["id"]]
    template = max(non_legend, key=lambda comp: (len(comp["colors"]), comp["area"]))
    template_mask = _mask_from_component(template)

    legend_pixels = sorted(
        [(int(r), int(c), int(g[r, c])) for r, c in legend["coords"]],
        key=lambda item: (item[0], item[1]),
    )

    fragments = []
    for comp in non_legend:
        if comp["id"] == template["id"]:
            continue
        fragment_mask = _mask_from_component(comp)
        if _matches_template(fragment_mask, template_mask):
            fragments.append(comp)

    fragments.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    out = g.copy()
    for comp, (_r, _c, color) in zip(fragments, legend_pixels):
        rr, cc = comp["coords"][:, 0], comp["coords"][:, 1]
        out[rr, cc] = int(color)
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_cells(h: int, w: int, area: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _random_template_mask() -> np.ndarray | None:
    for _ in range(120):
        h = random.choice([3, 4])
        w = random.choice([3, 4])
        max_area = h * w - 1
        min_area = max(6, math.ceil(0.6 * h * w))
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        cells = _grow_connected_cells(h, w, area)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _touches_all_sides(mask):
            continue
        rotations = {_mask_key(np.rot90(mask, k)) for k in range(4)}
        if len(rotations) < 4:
            continue
        return mask
    return None


def _paint_mask(mask: np.ndarray, colors: list[int], bg: int = BG) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    patch = np.full(mask.shape, bg, dtype=int)
    assigned = []
    for color in colors:
        assigned.append(color)
    while len(assigned) < len(coords):
        assigned.append(random.choice(colors))
    random.shuffle(assigned)
    for (r, c), color in zip(coords, assigned):
        patch[r, c] = int(color)
    return patch


def _random_connected_subset(mask: np.ndarray) -> np.ndarray | None:
    template_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    template_set = set(template_cells)
    area = len(template_cells)
    min_keep = max(2, math.ceil(area * 0.6))
    if min_keep >= area:
        return None
    for _ in range(120):
        target = random.randint(min_keep, area - 1)
        chosen = {random.choice(template_cells)}
        while len(chosen) < target:
            frontier = []
            for r, c in chosen:
                for dr, dc in DIRS4:
                    nbr = (r + dr, c + dc)
                    if nbr in template_set and nbr not in chosen:
                        frontier.append(nbr)
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) != target:
            continue
        rows = [r for r, _ in chosen]
        cols = [c for _, c in chosen]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        subset = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in chosen:
            subset[r - r0, c - c0] = True
        return subset
    return None


def _patch_from_mask(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _rect_is_free(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    blocked[r0:r0 + h, c0:c0 + w] = True


def _place_rect(
    blocked: np.ndarray,
    patch_shape: tuple[int, int],
    row_choices: list[int],
    col_choices: list[int],
    *,
    pad: int = 1,
) -> tuple[int, int] | None:
    h, w = map(int, patch_shape)
    positions = [(r, c) for r in row_choices for c in col_choices]
    random.shuffle(positions)
    for r0, c0 in positions:
        if _rect_is_free(blocked, r0, c0, h, w, pad=pad):
            _mark_rect(blocked, r0, c0, h, w)
            return int(r0), int(c0)
    return None


def _make_legend_sequence(count: int, fragment_colors: list[int], template_color_count: int) -> list[int] | None:
    pool = [color for color in NON_BG_COLORS if color not in fragment_colors]
    max_distinct = min(template_color_count - 1, len(pool))
    if max_distinct < 2:
        return None
    distinct = random.randint(2, max_distinct)
    palette = random.sample(pool, distinct)
    colors = palette[:]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _build_expected(grid: np.ndarray, fragments: list[dict], legend_colors: list[int]) -> np.ndarray:
    out = grid.copy()
    ordered = sorted(fragments, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    for frag, color in zip(ordered, legend_colors):
        r0, c0, patch = frag["r0"], frag["c0"], frag["patch"]
        rr, cc = np.where(patch != BG)
        out[r0 + rr, c0 + cc] = int(color)
    return out


def _generate_one() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    num_fragments = random.randint(3, 5)

    template_mask = _random_template_mask()
    if template_mask is None:
        return None

    template_color_count = 4 if num_fragments == 5 else random.choice([3, 4])
    template_colors = random.sample(TEMPLATE_COLORS, template_color_count)
    template_patch = _paint_mask(template_mask, template_colors)

    fragment_colors = random.sample(NON_BG_COLORS, num_fragments)
    legend_colors = _make_legend_sequence(num_fragments, fragment_colors, template_color_count)
    if legend_colors is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    template_rows = list(range(1, min(5, h - template_patch.shape[0])))
    template_cols = list(range(1, min(5, w - template_patch.shape[1] - 3)))
    if not template_rows or not template_cols:
        return None
    template_pos = _place_rect(blocked, template_patch.shape, template_rows, template_cols, pad=1)
    if template_pos is None:
        return None
    tr, tc = template_pos
    grid[tr:tr + template_patch.shape[0], tc:tc + template_patch.shape[1]] = np.where(
        template_patch != BG,
        template_patch,
        grid[tr:tr + template_patch.shape[0], tc:tc + template_patch.shape[1]],
    )

    legend_r0 = random.randint(1, h - num_fragments - 2)
    legend_c = w - 1
    if not _rect_is_free(blocked, legend_r0, legend_c, num_fragments, 1, pad=1):
        return None
    _mark_rect(blocked, legend_r0, legend_c, num_fragments, 1)
    for idx, color in enumerate(legend_colors):
        grid[legend_r0 + idx, legend_c] = int(color)

    fragments = []
    row_choices = list(range(0, h - 1))
    col_limit = w - 2
    for idx, fragment_color in enumerate(fragment_colors):
        rotated_mask = np.rot90(template_mask, random.randrange(4))
        fragment_mask = _random_connected_subset(rotated_mask)
        if fragment_mask is None:
            return None
        fragment_patch = _patch_from_mask(fragment_mask, fragment_color)
        fh, fw = fragment_patch.shape
        valid_rows = [r for r in row_choices if r + fh <= h]
        valid_cols = [c for c in range(0, col_limit) if c + fw <= col_limit]
        pos = _place_rect(blocked, fragment_patch.shape, valid_rows, valid_cols, pad=1)
        if pos is None:
            return None
        fr, fc = pos
        rr, cc = np.where(fragment_patch != BG)
        grid[fr + rr, fc + cc] = int(fragment_color)
        fragments.append(
            {
                "id": idx,
                "r0": int(fr),
                "c0": int(fc),
                "patch": fragment_patch,
                "bbox": (int(fr), int(fc), int(fr + fh - 1), int(fc + fw - 1)),
            }
        )

    expected = _build_expected(grid, fragments, legend_colors)

    components = _extract_components(grid, BG)
    legend = _find_legend_component(components)
    non_legend = [comp for comp in components if comp["id"] != legend["id"]]
    if max(len(comp["colors"]) for comp in non_legend) != template_color_count:
        return None
    if sum(len(comp["colors"]) == template_color_count for comp in non_legend) != 1:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 2")
