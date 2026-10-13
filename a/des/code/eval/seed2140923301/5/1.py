from __future__ import annotations

from collections import deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _touches_edge(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _analyze_fence(
    grid: np.ndarray,
    fence_color: int,
    coords: np.ndarray,
    bg: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _border_cells(r0, c0, r1, c1)
    border_set = set(border)
    cell_set = {tuple(map(int, rc)) for rc in coords}
    if not cell_set <= border_set:
        return None
    if len(cell_set) != len(border) - 1:
        return None

    missing = [cell for cell in border if cell not in cell_set]
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r, gap_c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if gap_r == r0:
        direction = (-1, 0)
    elif gap_r == r1:
        direction = (1, 0)
    elif gap_c == c0:
        direction = (0, -1)
    else:
        direction = (0, 1)

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    fragment_colors = np.unique(interior[(interior != bg) & (interior != fence_color)])
    if fragment_colors.size != 1:
        return None
    fragment_color = int(fragment_colors[0])
    fragment_comps = _components(interior == fragment_color)
    if len(fragment_comps) != 1:
        return None

    return {
        "fence_color": int(fence_color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gap_r), int(gap_c)),
        "direction": direction,
        "fragment_color": int(fragment_color),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fence's interior fragment color to the same-color edge template, and
    extend a fence-colored ray outward from the gap by the template's area.

    The gap itself stays unchanged; only the outside cells are painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _most_common_color(g)

    color_components: dict[int, list[np.ndarray]] = {}
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        color_components[color] = _components(g == color)

    template_areas: dict[int, int] = {}
    for color, comps in color_components.items():
        edge_components = [coords for coords in comps if _touches_edge(coords, height, width)]
        if edge_components:
            assert len(edge_components) == 1, "each template color must have one edge component"
            template_areas[color] = int(edge_components[0].shape[0])

    fences: list[dict[str, object]] = []
    for color, comps in color_components.items():
        for coords in comps:
            info = _analyze_fence(g, color, coords, bg)
            if info is not None:
                fences.append(info)

    assert fences, "expected at least one fenced enclosure"

    out = g.copy()
    for fence in fences:
        fragment_color = int(fence["fragment_color"])
        assert fragment_color in template_areas, "missing matching edge template"
        ray_length = int(template_areas[fragment_color])
        gap_r, gap_c = fence["gap"]
        dr, dc = fence["direction"]
        rr = gap_r + dr
        cc = gap_c + dc
        for _ in range(ray_length):
            assert 0 <= rr < height and 0 <= cc < width, "ray must stay in bounds"
            assert int(g[rr, cc]) == bg, "ray corridor must be empty in the input"
            out[rr, cc] = int(fence["fence_color"])
            rr += dr
            cc += dc

    return out


def _mask_from_lines(*lines: str) -> np.ndarray:
    return np.array([[ch == "#" for ch in line] for line in lines], dtype=bool)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[rows - r0, cols - c0] = True
    return out


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            norm = _normalize_mask(variant)
            key = (tuple(map(int, norm.shape)), norm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
    return out


_FRAGMENT_BASES = {
    1: [_mask_from_lines("#")],
    2: [_mask_from_lines("##")],
    3: [_mask_from_lines("###"), _mask_from_lines("##", "#.")],
}

_TEMPLATE_BASES = {
    3: [_mask_from_lines("###"), _mask_from_lines("##", "#.")],
    4: [_mask_from_lines("####"), _mask_from_lines("##", "##"), _mask_from_lines(".#.", "###")],
    5: [
        _mask_from_lines("#####"),
        _mask_from_lines("###", "#..", "#.."),
        _mask_from_lines(".#.", "###", ".#."),
    ],
    6: [
        _mask_from_lines("######"),
        _mask_from_lines("###", "###"),
        _mask_from_lines("###", "#..", "#..", "#.."),
        _mask_from_lines(".##", "###", "#.."),
    ],
}

_FRAGMENT_VARIANTS = {
    area: [variant for mask in masks for variant in _mask_variants(mask)]
    for area, masks in _FRAGMENT_BASES.items()
}
_TEMPLATE_VARIANTS = {
    area: [variant for mask in masks for variant in _mask_variants(mask)]
    for area, masks in _TEMPLATE_BASES.items()
}


GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
FENCE_COUNT_CHOICES = (1, 2, 3)
FENCE_SIDE_RANGE = (4, 7)
GEN_ATTEMPTS = 400
PLACEMENT_RETRIES = 200
PADDING = 1


def _padded_cells(cells: set[tuple[int, int]], height: int, width: int, pad: int = PADDING) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(height - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(width - 1, c + pad) + 1):
                out.add((rr, cc))
    return out


def _box_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _fits(blocked: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    height, width = map(int, blocked.shape)
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if bool(blocked[r, c]):
            return False
    return True


def _apply_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[rows + int(top), cols + int(left)] = int(color)


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    rows, cols = zip(*cells)
    blocked[list(rows), list(cols)] = True


def _choose_variant(variants: list[np.ndarray], max_h: int, max_w: int) -> np.ndarray | None:
    candidates = [mask for mask in variants if int(mask.shape[0]) <= max_h and int(mask.shape[1]) <= max_w]
    if not candidates:
        return None
    return random.choice(candidates).copy()


def _place_fence(
    grid: np.ndarray,
    blocked: np.ndarray,
    fence_color: int,
    fragment_color: int,
    ray_length: int,
) -> dict[str, object] | None:
    height, width = map(int, grid.shape)
    for _ in range(PLACEMENT_RETRIES):
        fence_h = random.randint(*FENCE_SIDE_RANGE)
        fence_w = random.randint(*FENCE_SIDE_RANGE)
        side = random.choice(("top", "bottom", "left", "right"))

        if side == "top":
            r0_min, r0_max = ray_length, height - fence_h
            c0_min, c0_max = 0, width - fence_w
        elif side == "bottom":
            r0_min, r0_max = 0, height - fence_h - ray_length
            c0_min, c0_max = 0, width - fence_w
        elif side == "left":
            r0_min, r0_max = 0, height - fence_h
            c0_min, c0_max = ray_length, width - fence_w
        else:
            r0_min, r0_max = 0, height - fence_h
            c0_min, c0_max = 0, width - fence_w - ray_length

        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        r1 = r0 + fence_h - 1
        c1 = c0 + fence_w - 1

        if side in ("top", "bottom"):
            gap_c = c0 + random.randint(1, fence_w - 2)
            gap_r = r0 if side == "top" else r1
        else:
            gap_r = r0 + random.randint(1, fence_h - 2)
            gap_c = c0 if side == "left" else c1

        if side == "top":
            direction = (-1, 0)
        elif side == "bottom":
            direction = (1, 0)
        elif side == "left":
            direction = (0, -1)
        else:
            direction = (0, 1)

        ray_cells = {
            (gap_r + direction[0] * step, gap_c + direction[1] * step)
            for step in range(1, ray_length + 1)
        }
        reserve_cells = _box_cells(r0, c0, fence_h, fence_w) | ray_cells
        padded = _padded_cells(reserve_cells, height, width)
        if not _fits(blocked, padded):
            continue

        fragment_area = random.randint(1, 3)
        fragment_mask = _choose_variant(_FRAGMENT_VARIANTS[fragment_area], fence_h - 2, fence_w - 2)
        if fragment_mask is None:
            continue
        frag_h, frag_w = map(int, fragment_mask.shape)
        frag_r = random.randint(r0 + 1, r1 - frag_h)
        frag_c = random.randint(c0 + 1, c1 - frag_w)

        border_cells = set(_border_cells(r0, c0, r1, c1))
        border_cells.remove((gap_r, gap_c))
        for br, bc in border_cells:
            grid[br, bc] = fence_color
        _apply_mask(grid, fragment_mask, frag_r, frag_c, fragment_color)

        _mark_blocked(blocked, padded)
        return {
            "fence_color": int(fence_color),
            "fragment_color": int(fragment_color),
            "gap": (int(gap_r), int(gap_c)),
            "direction": direction,
            "ray_length": int(ray_length),
        }

    return None


def _place_template(
    grid: np.ndarray,
    blocked: np.ndarray,
    template_mask: np.ndarray,
    color: int,
) -> bool:
    height, width = map(int, grid.shape)
    mask_h, mask_w = map(int, template_mask.shape)
    for _ in range(PLACEMENT_RETRIES):
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            r0 = 0
            c0 = random.randint(0, width - mask_w)
        elif side == "bottom":
            r0 = height - mask_h
            c0 = random.randint(0, width - mask_w)
        elif side == "left":
            r0 = random.randint(0, height - mask_h)
            c0 = 0
        else:
            r0 = random.randint(0, height - mask_h)
            c0 = width - mask_w

        box = _box_cells(r0, c0, mask_h, mask_w)
        padded = _padded_cells(box, height, width)
        if not _fits(blocked, padded):
            continue
        _apply_mask(grid, template_mask, r0, c0, color)
        _mark_blocked(blocked, padded)
        return True
    return False


def _expected_output(grid: np.ndarray, fences: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for fence in fences:
        gap_r, gap_c = fence["gap"]
        dr, dc = fence["direction"]
        rr = gap_r + dr
        cc = gap_c + dc
        for _ in range(int(fence["ray_length"])):
            out[rr, cc] = int(fence["fence_color"])
            rr += dr
            cc += dc
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_fences = random.choice(FENCE_COUNT_CHOICES)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    fence_color = palette[1]
    fragment_colors = palette[2 : 2 + n_fences]

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    fence_infos: list[dict[str, object]] = []
    for fragment_color in fragment_colors:
        ray_length = random.randint(3, 6)
        template_mask = random.choice(_TEMPLATE_VARIANTS[ray_length]).copy()
        fence_info = _place_fence(grid, blocked, fence_color, fragment_color, ray_length)
        if fence_info is None:
            return None
        if not _place_template(grid, blocked, template_mask, fragment_color):
            return None
        fence_infos.append(fence_info)

    if _most_common_color(grid) != bg:
        return None

    expected = _expected_output(grid, fence_infos)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
