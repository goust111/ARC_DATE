import random
from typing import Iterable

import numpy as np


AZURE = 8
GREY = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("top", "bottom", "left", "right")
SEPARATIONS = (1, 2)
MAX_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_mask = grid == GREY
    boxes: list[tuple[int, int, int, int]] = []
    for pts in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub, border):
            boxes.append((r0, c0, r1, c1))
    assert boxes, "expected at least one grey rectangular fence"
    return sorted(boxes)


def _template_masks(
    grid: np.ndarray, fence_boxes: list[tuple[int, int, int, int]]
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    fence_region = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in fence_boxes:
        fence_region[r0 : r1 + 1, c0 : c1 + 1] = True

    templates: dict[int, np.ndarray] = {}
    colors = [int(color) for color in np.unique(grid) if int(color) not in {AZURE, GREY}]
    for color in colors:
        pts = np.argwhere((grid == color) & ~fence_region)
        if len(pts) == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
        templates[color] = mask

    assert templates, "expected at least one outside template"
    return templates, fence_region


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fenced seed color to an outside template and scale that template into the fence interior.

    Input structure:
    - The background is azure.
    - One or more one-pixel-thick grey rectangular fences appear in the grid; each fence encloses exactly one colored seed pixel on an otherwise azure interior.
    - Outside the fences, a template library contains one compact monochrome pattern per seed color, drawn on the azure background.

    Rule:
    - Find every grey rectangular fence and read the single seed color inside it.
    - For that color, locate the outside template of the same color and crop it to its tight bounding box.
    - Compute the integer scale factor whose block expansion makes the template exactly match the fence interior size, then paint the scaled template inside the fence while leaving azure holes untouched.
    - Keep the grey borders unchanged and clear the original outside templates back to azure.

    Key insight:
    - The seed pixel is only a color key; the real content comes from the same-colored outside template, which must be block-scaled to the fenced area.
    """
    g = np.asarray(grid, dtype=int)
    fences = _frame_bboxes(g)
    templates, fence_region = _template_masks(g, fences)

    out = g.copy()
    outside = ~fence_region
    for color in templates:
        out[(g == color) & outside] = AZURE

    for r0, c0, r1, c1 in fences:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_cells = np.argwhere((interior != AZURE) & (interior != GREY))
        assert len(seed_cells) == 1, "each fence must contain exactly one seed pixel"
        seed_r, seed_c = seed_cells[0]
        color = int(interior[seed_r, seed_c])
        assert color in templates, f"missing template for seed color {color}"

        template = templates[color]
        ih, iw = interior.shape
        th, tw = template.shape
        assert ih % th == 0 and iw % tw == 0, "template must scale evenly into the fence interior"
        scale_h = ih // th
        scale_w = iw // tw
        assert scale_h == scale_w, "scaling must be isotropic"

        scaled = np.kron(template, np.ones((scale_h, scale_w), dtype=bool))
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[:, :] = AZURE
        patch[scaled] = color

    return out


def _random_template(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        target = rng.randint(max(2, (h * w) // 2), h * w - 1)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        pts = np.argwhere(mask)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(mask.shape) < 2:
            continue
        if np.all(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate template")


def _choose_scale(rng: random.Random, template: np.ndarray) -> int:
    th, tw = template.shape
    feasible = [k for k in (1, 2, 3, 4) if th * k <= 10 and tw * k <= 10]
    assert feasible, "every template should admit at least one scale"
    larger = [k for k in feasible if k > 1]
    if larger and rng.random() < 0.8:
        return rng.choice(larger)
    return rng.choice(feasible)


def _place_rectangles(
    rng: random.Random, field_h: int, field_w: int, rects: list[tuple[int, int]], gap: int = 1
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)
    occupied = np.zeros((field_h, field_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(rects)

    for idx in order:
        rh, rw = rects[idx]
        candidates: list[tuple[int, int]] = []
        for r in range(field_h - rh + 1):
            for c in range(field_w - rw + 1):
                er0 = max(0, r - gap)
                ec0 = max(0, c - gap)
                er1 = min(field_h, r + rh + gap)
                ec1 = min(field_w, c + rw + gap)
                if not np.any(occupied[er0:er1, ec0:ec1]):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        placements[idx] = (r, c)
        er0 = max(0, r - gap)
        ec0 = max(0, c - gap)
        er1 = min(field_h, r + rh + gap)
        ec1 = min(field_w, c + rw + gap)
        occupied[er0:er1, ec0:ec1] = True

    return [placement for placement in placements if placement is not None]


def _pack_fences(
    rng: random.Random, rects: list[tuple[int, int]], field_h_max: int, field_w_max: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(h for h, _ in rects)
    max_w = max(w for _, w in rects)
    area = sum((h + 1) * (w + 1) for h, w in rects)
    for _ in range(120):
        field_h = rng.randint(max_h, field_h_max)
        field_w = rng.randint(max_w, field_w_max)
        if field_h * field_w < area:
            continue
        placements = _place_rectangles(rng, field_h, field_w, rects, gap=1)
        if placements is not None:
            return field_h, field_w, placements
    return None


def _major_extent(lengths: list[int], gaps: list[int]) -> int:
    return sum(lengths) + sum(gaps)


def _draw_fence(grid: np.ndarray, top: int, left: int, info: dict) -> None:
    scaled = info["scaled"]
    color = info["color"]
    h, w = scaled.shape
    bottom = top + h + 1
    right = left + w + 1
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    seed_r, seed_c = info["seed_pos"]
    grid[top + 1 + seed_r, left + 1 + seed_c] = color


def _layout_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(SIDES)
    separation = rng.choice(SEPARATIONS)
    count = rng.randint(1, 3)
    colors = rng.sample(TEMPLATE_COLORS, count)

    infos: list[dict] = []
    rects: list[tuple[int, int]] = []
    for color in colors:
        template = _random_template(rng)
        scale = _choose_scale(rng, template)
        scaled = np.kron(template, np.ones((scale, scale), dtype=bool))
        filled = list(map(tuple, np.argwhere(scaled)))
        empty = list(map(tuple, np.argwhere(~scaled)))
        if not filled or not empty:
            return None
        use_fill_seed = rng.choice((False, True))
        seed_pool = filled if use_fill_seed else empty
        infos.append(
            {
                "color": color,
                "template": template,
                "scaled": scaled,
                "seed_pos": rng.choice(seed_pool),
            }
        )
        sh, sw = scaled.shape
        rects.append((sh + 2, sw + 2))

    if side in {"top", "bottom"}:
        strip_h = max(info["template"].shape[0] for info in infos) + 2
        field_h_max = 30 - strip_h - separation
        field_w_max = 30
    else:
        strip_w = max(info["template"].shape[1] for info in infos) + 2
        field_h_max = 30
        field_w_max = 30 - strip_w - separation

    if field_h_max < max(h for h, _ in rects) or field_w_max < max(w for _, w in rects):
        return None

    packed = _pack_fences(rng, rects, field_h_max, field_w_max)
    if packed is None:
        return None
    field_h, field_w, fence_positions = packed

    if side in {"top", "bottom"}:
        gaps = [rng.randint(1, 3) for _ in range(count + 1)]
        lengths = [info["template"].shape[1] for info in infos]
        strip_need = _major_extent(lengths, gaps)
        total_h = strip_h + separation + field_h
        total_w = max(field_w, strip_need)
        if total_h > 30 or total_w > 30:
            return None
        grid = np.full((total_h, total_w), AZURE, dtype=int)

        strip_top = 0 if side == "top" else total_h - strip_h
        field_top = strip_h + separation if side == "top" else 0
        strip_left_slack = rng.randint(0, total_w - strip_need)
        field_left = rng.randint(0, total_w - field_w)
        cursor = strip_left_slack + gaps[0]
        for idx, info in enumerate(infos):
            template = info["template"]
            th, tw = template.shape
            row = strip_top + rng.randint(1, strip_h - th - 1)
            grid[row : row + th, cursor : cursor + tw][template] = info["color"]
            cursor += tw + gaps[idx + 1]

        for (fr, fc), info in zip(fence_positions, infos):
            _draw_fence(grid, field_top + fr, field_left + fc, info)
    else:
        gaps = [rng.randint(1, 3) for _ in range(count + 1)]
        lengths = [info["template"].shape[0] for info in infos]
        strip_need = _major_extent(lengths, gaps)
        total_h = max(field_h, strip_need)
        total_w = strip_w + separation + field_w
        if total_h > 30 or total_w > 30:
            return None
        grid = np.full((total_h, total_w), AZURE, dtype=int)

        strip_left = 0 if side == "left" else total_w - strip_w
        field_left = strip_w + separation if side == "left" else 0
        strip_top_slack = rng.randint(0, total_h - strip_need)
        field_top = rng.randint(0, total_h - field_h)
        cursor = strip_top_slack + gaps[0]
        for idx, info in enumerate(infos):
            template = info["template"]
            th, tw = template.shape
            col = strip_left + rng.randint(1, strip_w - tw - 1)
            grid[cursor : cursor + th, col : col + tw][template] = info["color"]
            cursor += th + gaps[idx + 1]

        for (fr, fc), info in zip(fence_positions, infos):
            _draw_fence(grid, field_top + fr, field_left + fc, info)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _layout_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
