from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
BORDER = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
    return tuple(map(int, cropped.shape)), cropped.tobytes()


def _color_ranking(patch: np.ndarray, bg: int) -> list[int]:
    colors = [int(c) for c in np.unique(patch) if int(c) != bg]
    ranked: list[tuple[int, int, int, int, tuple[int, int], bytes]] = []
    for color in colors:
        mask = patch == color
        coords = np.argwhere(mask)
        count = int(coords.shape[0])
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
        ranked.append(
            (
                int(color),
                count,
                int(r0),
                int(c0),
                tuple(map(int, cropped.shape)),
                cropped.tobytes(),
            )
        )
    ranked.sort(key=lambda item: (-item[1], item[2], item[3], item[4], item[5]))
    return [item[0] for item in ranked]


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in [int(c) for c in np.unique(grid) if int(c) != bg]:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            perim_area = 2 * h + 2 * w - 4
            if len(coords) != perim_area:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border": int(color),
                    "patch": patch.copy(),
                }
            )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _extract_region_info(frame: dict, bg: int) -> dict:
    patch = frame["patch"]
    inner = patch[1:-1, 1:-1]
    key_mask = inner != bg
    assert key_mask.any()
    return {
        "shape": _shape_signature(key_mask),
        "colors": _color_ranking(inner, bg),
    }


def _extract_template_info(frame: dict, bg: int) -> dict:
    patch = frame["patch"]
    inner = patch[1:-1, 1:-1]
    marker = inner == RED
    assert marker.any()
    return {
        "shape": _shape_signature(marker),
        "colors": _color_ranking(inner, bg),
        "patch": patch.copy(),
    }


def _recolor_template(template_patch: np.ndarray, key_colors: list[int], template_colors: list[int], bg: int) -> np.ndarray:
    assert len(key_colors) == len(template_colors)
    out = template_patch.copy()
    inner = out[1:-1, 1:-1].copy()
    source = inner.copy()
    for src_color, dst_color in zip(template_colors, key_colors):
        inner[source == src_color] = int(dst_color)
    out[1:-1, 1:-1] = inner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enclosed key shape to a bottom template marker, then stamp the matched template
    into the region with colors remapped by descending color area. The template library is cleared.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG

    frames = _find_frames(g, bg)
    assert frames

    template_bottom = max(frame["bbox"][2] for frame in frames)
    templates = [frame for frame in frames if frame["bbox"][2] == template_bottom]
    regions = [frame for frame in frames if frame["bbox"][2] != template_bottom]

    assert templates and regions

    template_by_shape: dict[tuple[tuple[int, int], bytes], dict] = {}
    for frame in templates:
        info = _extract_template_info(frame, bg)
        shape = info["shape"]
        assert shape not in template_by_shape
        template_by_shape[shape] = info

    out = g.copy()
    for frame in regions:
        region = _extract_region_info(frame, bg)
        assert region["shape"] in template_by_shape
        template = template_by_shape[region["shape"]]
        assert frame["patch"].shape == template["patch"].shape
        transformed = _recolor_template(template["patch"], region["colors"], template["colors"], bg)
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    for frame in templates:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    return out


# --------------- generator ---------------

RECT_H_CHOICES = (8, 9)
RECT_W_CHOICES = (8, 9)
TEMPLATE_COUNT_CHOICES = (2, 3)
KEY_AREA_CHOICES = (7, 8)
KEY_BBOX_CHOICES = ((3, 3), (3, 4), (4, 3), (4, 4))
GRID_EXTRA_WIDTH = (1, 5)
GRID_EXTRA_HEIGHT = (3, 7)
TEMPLATE_GAP_CHOICES = (1, 2)
REGION_PAD = 1
MAX_GEN_ATTEMPTS = 400


def _count_partitions(total: int) -> list[tuple[int, int, int]]:
    parts = set()
    for a in range(1, total - 1):
        for b in range(1, total - a):
            c = total - a - b
            if len({a, b, c}) == 3:
                parts.add(tuple(sorted((a, b, c), reverse=True)))
    return sorted(parts)


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(80):
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set()
        for nr, nc in _neighbors4(*seed):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            if pick in cells:
                continue
            cells.add(pick)
            for nr, nc in _neighbors4(*pick):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if int(cropped.sum()) == cropped.shape[0] * cropped.shape[1]:
            continue
        return cropped
    return None


def _make_unique_key_masks(n: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(n):
        mask = None
        for _ in range(100):
            area = random.choice(KEY_AREA_CHOICES)
            dims = [dim for dim in KEY_BBOX_CHOICES if dim[0] * dim[1] >= area]
            h, w = random.choice(dims)
            candidate = _grow_connected_mask(h, w, area)
            if candidate is None:
                continue
            sig = _shape_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            mask = candidate
            break
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _paint_colored_shape(mask: np.ndarray, colors: list[int], counts: tuple[int, int, int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    patch = np.full(mask.shape, BG, dtype=int)
    idx = 0
    for color, count in zip(colors, counts):
        for _ in range(count):
            r, c = coords[idx]
            patch[r, c] = int(color)
            idx += 1
    return patch


def _template_count_options(marker_area: int, free_cells: int, marker_rank: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for a in range(2, free_cells):
        for b in range(2, free_cells):
            if a == b or a == marker_area or b == marker_area or a + b > free_cells:
                continue
            ranks = sorted((marker_area, a, b), reverse=True)
            if ranks.index(marker_area) == marker_rank:
                options.append((a, b))
    return options


def _grow_blob(allowed: np.ndarray, size: int, bias_bottom_right: bool) -> set[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if not coords:
        return None
    if bias_bottom_right:
        coords.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]), reverse=True)
        seeds = coords[: max(1, len(coords) // 3)]
    else:
        seeds = coords
    for _ in range(80):
        seed = random.choice(seeds)
        cells = {seed}
        frontier = set()
        for nr, nc in _neighbors4(*seed):
            if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc]:
                frontier.add((nr, nc))
        while len(cells) < size and frontier:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            if pick in cells or not allowed[pick]:
                continue
            cells.add(pick)
            for nr, nc in _neighbors4(*pick):
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc] and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) == size:
            return cells
    return None


def _build_template_inner(mask: np.ndarray, rect_h: int, rect_w: int, colors: list[int]) -> tuple[np.ndarray, int] | None:
    inner_h = rect_h - 2
    inner_w = rect_w - 2
    marker_area = int(mask.sum())
    free_cells = inner_h * inner_w - marker_area
    marker_rank = random.randrange(3)
    count_options = _template_count_options(marker_area, free_cells, marker_rank)
    if not count_options:
        return None
    count_a, count_b = random.choice(count_options)

    inner = np.full((inner_h, inner_w), BG, dtype=int)
    mh, mw = mask.shape
    inner[:mh, :mw][mask] = RED

    allowed = inner == BG
    blob_a = _grow_blob(allowed, count_a, bias_bottom_right=True)
    if blob_a is None:
        return None
    for r, c in blob_a:
        inner[r, c] = int(colors[0])

    allowed = inner == BG
    blob_b = _grow_blob(allowed, count_b, bias_bottom_right=False)
    if blob_b is None:
        return None
    for r, c in blob_b:
        inner[r, c] = int(colors[1])

    return inner, marker_rank


def _build_frame(inner: np.ndarray, border: int) -> np.ndarray:
    h, w = inner.shape[0] + 2, inner.shape[1] + 2
    patch = np.full((h, w), border, dtype=int)
    patch[1:-1, 1:-1] = inner
    return patch


def _place_rectangles(
    grid: np.ndarray,
    patches: list[np.ndarray],
    max_row: int,
    occupied: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    placements: list[tuple[int, int]] = []
    for patch in patches:
        ph, pw = patch.shape
        placed = False
        for _ in range(200):
            r0 = random.randint(1, max_row - ph)
            c0 = random.randint(1, w - pw - 1)
            rr0 = max(0, r0 - REGION_PAD)
            cc0 = max(0, c0 - REGION_PAD)
            rr1 = min(h, r0 + ph + REGION_PAD)
            cc1 = min(w, c0 + pw + REGION_PAD)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None
    return placements


def _template_row_layout(rect_w: int, n: int) -> tuple[int, list[int]]:
    gaps = [random.choice(TEMPLATE_GAP_CHOICES) for _ in range(n - 1)]
    width = n * rect_w + sum(gaps)
    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + rect_w + gap)
    return width, starts


def _build_expected(
    inp: np.ndarray,
    region_frames: list[tuple[int, int, np.ndarray]],
    template_frames: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    out = inp.copy()
    template_infos = []
    for r0, c0, patch in template_frames:
        frame = {"bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1), "patch": patch}
        template_infos.append(_extract_template_info(frame, BG))

    by_shape = {info["shape"]: info for info in template_infos}
    for r0, c0, patch in region_frames:
        frame = {"bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1), "patch": patch}
        region = _extract_region_info(frame, BG)
        template = by_shape[region["shape"]]
        recolored = _recolor_template(template["patch"], region["colors"], template["colors"], BG)
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = recolored

    for r0, c0, patch in template_frames:
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(TEMPLATE_COUNT_CHOICES)
    rect_h = random.choice(RECT_H_CHOICES)
    rect_w = random.choice(RECT_W_CHOICES)
    masks = _make_unique_key_masks(n)
    if masks is None:
        return None

    region_patches: list[np.ndarray] = []
    template_patches: list[np.ndarray] = []
    for mask in masks:
        all_colors = random.sample(PALETTE, 5)
        key_colors = all_colors[:3]
        motif_colors = all_colors[3:]
        count_options = _count_partitions(int(mask.sum()))
        if not count_options:
            return None
        key_counts = random.choice(count_options)
        key_patch = _paint_colored_shape(mask, key_colors, key_counts)

        inner = np.full((rect_h - 2, rect_w - 2), BG, dtype=int)
        kh, kw = key_patch.shape
        max_r = inner.shape[0] - kh
        max_c = inner.shape[1] - kw
        offset_r = random.randint(0, max_r)
        offset_c = random.randint(0, max_c)
        key_mask = key_patch != BG
        inner[offset_r : offset_r + kh, offset_c : offset_c + kw][key_mask] = key_patch[key_mask]
        region_patches.append(_build_frame(inner, BORDER))

        built = _build_template_inner(mask, rect_h, rect_w, motif_colors)
        if built is None:
            return None
        template_inner, _marker_rank = built
        template_patches.append(_build_frame(template_inner, BORDER))

    row_width, col_starts = _template_row_layout(rect_w, n)
    grid_w = row_width + random.randint(*GRID_EXTRA_WIDTH) + 2
    grid_h = rect_h * 2 + random.randint(*GRID_EXTRA_HEIGHT) + 2
    if grid_w < row_width + 2 or grid_h < rect_h * 2 + 4:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    template_r0 = grid_h - rect_h - 1
    template_margin = random.randint(1, grid_w - row_width - 1)
    template_frames: list[tuple[int, int, np.ndarray]] = []
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for idx, patch in enumerate(template_patches):
        c0 = template_margin + col_starts[idx]
        grid[template_r0 : template_r0 + rect_h, c0 : c0 + rect_w] = patch
        rr0 = max(0, template_r0 - REGION_PAD)
        cc0 = max(0, c0 - REGION_PAD)
        rr1 = min(grid_h, template_r0 + rect_h + REGION_PAD)
        cc1 = min(grid_w, c0 + rect_w + REGION_PAD)
        occupied[rr0:rr1, cc0:cc1] = True
        template_frames.append((template_r0, c0, patch))

    order = list(range(n))
    random.shuffle(order)
    shuffled_regions = [region_patches[i] for i in order]
    max_region_row = template_r0 - 1
    if max_region_row <= rect_h + 1:
        return None
    region_positions = _place_rectangles(grid, shuffled_regions, max_region_row, occupied)
    if region_positions is None:
        return None

    region_frames = [(r0, c0, patch) for (r0, c0), patch in zip(region_positions, shuffled_regions)]
    expected = _build_expected(grid, region_frames, template_frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
