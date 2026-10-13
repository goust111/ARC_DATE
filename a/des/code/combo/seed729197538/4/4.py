import random
from collections import deque

import numpy as np


BG = 0
FENCE_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return comps


def _crop_mask(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1] == color


def _gap_range(edge: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~edge)
    if len(missing) == 0:
        return None
    if missing[0] == 0 or missing[-1] == len(edge) - 1:
        return None
    if not np.all(np.diff(missing) == 1):
        return None
    if not np.all(edge[: missing[0]]) or not np.all(edge[missing[-1] + 1 :]):
        return None
    return int(missing[0]), int(missing[-1])


def _parse_fence(mask: np.ndarray) -> dict | None:
    h, w = mask.shape
    if h < 5 or w < 5:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None

    sides = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    gap_info = {side: _gap_range(edge) for side, edge in sides.items()}
    gapped = [side for side, gap in gap_info.items() if gap is not None]
    if len(gapped) != 1:
        return None
    direction = gapped[0]
    gap_start, gap_end = gap_info[direction]
    return {
        "direction": direction,
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
        "height": int(h),
        "width": int(w),
    }


def _infer_fence_color(grid: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    candidates = []
    for color in colors:
        comps = _components_of_color(grid, color)
        if not comps:
            continue
        parsed = [_parse_fence(_crop_mask(grid, color, comp["bbox"])) for comp in comps]
        if all(info is not None for info in parsed):
            candidates.append((len(comps), color))
    if candidates:
        candidates.sort(reverse=True)
        return int(candidates[0][1])
    if FENCE_COLOR in colors:
        return FENCE_COLOR
    raise AssertionError("failed to infer fence color")


def _find_templates(grid: np.ndarray, fence_color: int) -> dict[int, np.ndarray]:
    band_h = min(5, grid.shape[0])
    band = grid[:band_h, :]
    active = np.any((band != BG) & (band != fence_color), axis=0)
    templates: dict[int, np.ndarray] = {}
    c = 0
    while c < grid.shape[1]:
        if not active[c]:
            c += 1
            continue
        c0 = c
        while c < grid.shape[1] and active[c]:
            c += 1
        c1 = c - 1
        sub = band[:, c0 : c1 + 1]
        colors = sorted({int(v) for v in np.unique(sub) if int(v) not in (BG, fence_color)})
        assert len(colors) == 1, "each top-band template must be monochrome"
        color = colors[0]
        mask = sub == color
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        assert len(rows) > 0 and len(cols) > 0
        assert rows[0] == 0, "templates must touch the top edge"
        tight = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        assert tight.shape[0] <= 5 and tight.shape[1] <= 5
        assert color not in templates, "template colors must be unique"
        templates[color] = tight
    assert templates, "expected at least one top-edge template"
    return templates


def _orient_template(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction == "bottom":
        return mask.copy()
    if direction == "top":
        return np.rot90(mask, 2)
    if direction == "left":
        return np.rot90(mask, 1)
    if direction == "right":
        return np.rot90(mask, 3)
    raise AssertionError(f"unknown direction: {direction}")


def _placement_for_growth(
    bbox: tuple[int, int, int, int],
    direction: str,
    gap_start: int,
    gap_end: int,
    patch_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    ph, pw = patch_shape
    span = gap_end - gap_start + 1
    if direction == "top":
        return r0 - ph, c0 + gap_start + (span - pw) // 2
    if direction == "bottom":
        return r1 + 1, c0 + gap_start + (span - pw) // 2
    if direction == "left":
        return r0 + gap_start + (span - ph) // 2, c0 - pw
    if direction == "right":
        return r0 + gap_start + (span - ph) // 2, c1 + 1
    raise AssertionError(f"unknown direction: {direction}")


def _paste_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    ph, pw = mask.shape
    assert 0 <= top and 0 <= left
    assert top + ph <= grid.shape[0] and left + pw <= grid.shape[1]
    target = grid[top : top + ph, left : left + pw]
    assert np.all(target[mask] == BG), "growth must land on background"
    target[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each fenced seed, read the same-color top-edge template, and grow it
    outward through the fence gap with direction-dependent rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fence_color = _infer_fence_color(g)
    templates = _find_templates(g, fence_color)
    out = g.copy()

    fence_components = _components_of_color(g, fence_color)
    assert fence_components, "expected at least one fence"
    for comp in fence_components:
        bbox = comp["bbox"]
        patch = _crop_mask(g, fence_color, bbox)
        parsed = _parse_fence(patch)
        assert parsed is not None, "every fence-colored component must be a gapped ring"

        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((inner != BG) & (inner != fence_color))
        assert len(seed_coords) == 1, "each fence must contain exactly one seed pixel"
        seed_r, seed_c = map(int, seed_coords[0])
        seed_color = int(inner[seed_r, seed_c])
        assert seed_color in templates, "every seed color must have a top-edge template"

        growth_mask = _orient_template(templates[seed_color], parsed["direction"])
        top, left = _placement_for_growth(
            bbox,
            parsed["direction"],
            parsed["gap_start"],
            parsed["gap_end"],
            growth_mask.shape,
        )
        _paste_mask(out, growth_mask, seed_color, int(top), int(left))

    return out


def _random_template_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        area_lo = max(3, (h * w + 2) // 3)
        area_hi = h * w - 1
        if area_lo > area_hi:
            continue
        target_area = random.randint(area_lo, area_hi)

        start = (0, random.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], h, w))
        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nxt in _neighbors4(cell[0], cell[1], h, w):
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric template mask")


def _draw_template(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    assert np.all(patch[mask] == BG)
    patch[mask] = color


def _draw_fence(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    direction: str,
    gap_start: int,
    gap_end: int,
    seed_color: int,
) -> None:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = FENCE_COLOR
    patch[-1, :] = FENCE_COLOR
    patch[:, 0] = FENCE_COLOR
    patch[:, -1] = FENCE_COLOR
    if direction == "top":
        patch[0, gap_start : gap_end + 1] = BG
    elif direction == "bottom":
        patch[-1, gap_start : gap_end + 1] = BG
    elif direction == "left":
        patch[gap_start : gap_end + 1, 0] = BG
    elif direction == "right":
        patch[gap_start : gap_end + 1, -1] = BG
    else:
        raise AssertionError(f"unknown direction: {direction}")

    target = grid[top : top + height, left : left + width]
    assert np.all(target == BG)
    target[:, :] = patch

    seed_r = top + random.randint(1, height - 2)
    seed_c = left + random.randint(1, width - 2)
    assert grid[seed_r, seed_c] == BG
    grid[seed_r, seed_c] = seed_color


def _sample_gap_length(side_len: int, attach_len: int) -> int | None:
    options = [
        gap_len
        for gap_len in range(1, min(5, side_len - 2) + 1)
        if gap_len <= side_len - 2 and (gap_len - attach_len) % 2 == 0
    ]
    if not options:
        return None
    return int(random.choice(options))


def _place_fence_items(
    height: int,
    width: int,
    top_clearance: int,
    fence_specs: list[dict],
) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    occupied[:top_clearance, :] = True
    placed: list[dict] = []

    for spec in fence_specs:
        color = int(spec["color"])
        direction = spec["direction"]
        template = spec["template"]
        grown = _orient_template(template, direction)
        ph, pw = grown.shape

        success = False
        for _ in range(300):
            fence_h = random.randint(5, 9)
            fence_w = random.randint(5, 9)
            side_len = fence_w if direction in ("top", "bottom") else fence_h
            gap_len = _sample_gap_length(side_len, pw if direction in ("top", "bottom") else ph)
            if gap_len is None:
                continue
            gap_start = (side_len - gap_len) // 2
            gap_end = gap_start + gap_len - 1

            top = random.randint(1, height - fence_h - 1)
            left = random.randint(1, width - fence_w - 1)
            bbox = (top, left, top + fence_h - 1, left + fence_w - 1)
            grow_top, grow_left = _placement_for_growth(
                bbox, direction, gap_start, gap_end, grown.shape
            )
            grow_bottom = grow_top + ph - 1
            grow_right = grow_left + pw - 1
            if grow_top < 0 or grow_left < 0 or grow_bottom >= height or grow_right >= width:
                continue

            rr0 = max(0, min(top, grow_top) - 1)
            cc0 = max(0, min(left, grow_left) - 1)
            rr1 = min(height - 1, max(top + fence_h - 1, grow_bottom) + 1)
            cc1 = min(width - 1, max(left + fence_w - 1, grow_right) + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed.append(
                {
                    "color": color,
                    "direction": direction,
                    "template": template,
                    "top": top,
                    "left": left,
                    "height": fence_h,
                    "width": fence_w,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                }
            )
            success = True
            break

        if not success:
            return None

    return placed


def _build_input() -> np.ndarray | None:
    fence_count = random.randint(1, 4)
    directions = random.sample(list(DIRECTIONS), k=fence_count)

    template_color_count = random.randint(1, min(4, fence_count))
    template_colors = random.sample(PALETTE, k=template_color_count)
    color_assignments = template_colors.copy()
    while len(color_assignments) < fence_count:
        color_assignments.append(random.choice(template_colors))
    random.shuffle(color_assignments)

    templates = {color: _random_template_mask() for color in template_colors}
    ordered_templates = template_colors[:]
    random.shuffle(ordered_templates)

    gaps = [random.randint(1, 2) for _ in range(max(0, len(ordered_templates) - 1))]
    template_span = sum(templates[color].shape[1] for color in ordered_templates) + sum(gaps)
    top_clearance = max(templates[color].shape[0] for color in ordered_templates) + 2

    size_floor = {1: 22, 2: 26, 3: 30, 4: 32}[fence_count]
    min_width = max(template_span + 2, size_floor)
    min_height = max(top_clearance + 10, size_floor)
    width = random.randint(min_width, min_width + 6)
    height = random.randint(min_height, min_height + 6)

    fence_specs = [
        {"color": color_assignments[i], "direction": directions[i], "template": templates[color_assignments[i]]}
        for i in range(fence_count)
    ]
    random.shuffle(fence_specs)
    placed_fences = _place_fence_items(height, width, top_clearance, fence_specs)
    if placed_fences is None:
        return None

    grid = np.zeros((height, width), dtype=int)

    slack = width - template_span
    left = random.randint(0, slack)
    for idx, color in enumerate(ordered_templates):
        mask = templates[color]
        _draw_template(grid, mask, color, 0, left)
        left += mask.shape[1]
        if idx < len(gaps):
            left += gaps[idx]

    for item in placed_fences:
        _draw_fence(
            grid,
            item["top"],
            item["left"],
            item["height"],
            item["width"],
            item["direction"],
            item["gap_start"],
            item["gap_end"],
            item["color"],
        )

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
