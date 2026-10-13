from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500
ARC_MAX = 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    rows = points[:, 0]
    cols = points[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rect_frame(points: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[int(r) - r0, int(c) - c0] = True
    return np.array_equal(local, _rect_border_mask(h, w))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _find_template_box(grid: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    grey_points = np.argwhere(grid == GREY)
    assert len(grey_points) > 0, "expected a grey template separator region"
    box = _bbox(grey_points)
    h, w = grid.shape
    r0, c0, r1, c1 = box
    if r1 == h - 1 and c1 < w - 1:
        return box, "bottom"
    if c1 == w - 1 and r1 < h - 1:
        return box, "right"
    if r1 == h - 1 and c1 == w - 1:
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        return box, "bottom" if height <= width else "right"
    raise AssertionError("template region must touch the bottom or right edge")


def _extract_templates(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    box, orientation = _find_template_box(grid)
    r0, c0, r1, c1 = box
    patch = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert patch.size > 0, "template patch must have a non-empty interior"

    templates: list[np.ndarray] = []
    if orientation == "bottom":
        separators = [int(i) for i in np.where(np.all(patch == GREY, axis=0))[0]]
        start = 0
        for sep in separators + [patch.shape[1]]:
            if start < sep:
                tmpl = patch[:, start:sep].copy()
                assert np.all(tmpl != GREY)
                templates.append(tmpl)
            start = sep + 1
    else:
        separators = [int(i) for i in np.where(np.all(patch == GREY, axis=1))[0]]
        start = 0
        for sep in separators + [patch.shape[0]]:
            if start < sep:
                tmpl = patch[start:sep, :].copy()
                assert np.all(tmpl != GREY)
                templates.append(tmpl)
            start = sep + 1

    assert templates, "expected at least one template"
    return templates, box


def _frame_components(grid: np.ndarray, template_box: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = template_box
    usable = (grid != BLACK) & (grid != GREY)
    usable[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = []
    for pts in _components(usable):
        color = int(grid[tuple(pts[0])])
        if not np.all(grid[pts[:, 0], pts[:, 1]] == color):
            continue
        if not _is_rect_frame(pts):
            continue
        comps.append({"color": color, "bbox": _bbox(pts), "points": pts})
    assert comps, "expected at least one frame component"
    return comps


def _frame_families(components: list[dict]) -> list[list[dict]]:
    n = len(components)
    parents: list[int | None] = [None] * n
    children: dict[int, list[int]] = {i: [] for i in range(n)}

    for i, comp in enumerate(components):
        candidates = [
            j
            for j, other in enumerate(components)
            if i != j
            and comp["color"] == other["color"]
            and _contains(other["bbox"], comp["bbox"])
        ]
        if candidates:
            parent = min(candidates, key=lambda j: _bbox_area(components[j]["bbox"]))
            parents[i] = parent
            children[parent].append(i)

    families: list[list[dict]] = []
    for i in range(n):
        if parents[i] is not None:
            continue
        stack = [i]
        members: list[dict] = []
        while stack:
            cur = stack.pop()
            members.append(components[cur])
            stack.extend(children[cur])
        members.sort(key=lambda comp: _bbox_area(comp["bbox"]), reverse=True)
        families.append(members)
    return families


def _tile_fill(
    out: np.ndarray,
    frame_box: tuple[int, int, int, int],
    frame_color: int,
    template: np.ndarray,
) -> None:
    r0, c0, r1, c1 = frame_box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h <= 0 or inner_w <= 0:
        return

    pattern = template != BLACK
    ph, pw = pattern.shape
    for rr in range(inner_h):
        for cc in range(inner_w):
            out[r0 + 1 + rr, c0 + 1 + cc] = frame_color if pattern[rr % ph, cc % pw] else BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each nested color frame with the edge template selected by its nesting depth.

    Input structure:
    - The background is black.
    - The main field contains one or more single-color, one-pixel-thick rectangular frames; a frame may contain smaller same-color nested rectangular frames with one black cell of spacing between layers.
    - A grey-bordered template strip sits on the bottom edge or right edge, holding several small colored templates separated by grey divider lines and ordered left-to-right or top-to-bottom.

    Rule:
    - Count the nesting depth of each frame family, where the outermost frame is depth 1 and each enclosed same-color frame adds one more layer.
    - Read the template strip in order and select the template whose index matches that depth.
    - Recolor that template to the frame color and tile it from the top-left corner across the interior of the innermost frame in the family.
    - Leave every frame border and the template strip unchanged.

    Key insight:
    - The number of nested same-color frame layers is an index into the ordered edge templates; the chosen template is not copied once but repeatedly tiled inside the deepest hollow area.
    """
    g = np.asarray(grid, dtype=int)
    templates, template_box = _extract_templates(g)
    components = _frame_components(g, template_box)
    families = _frame_families(components)

    out = g.copy()
    for family in families:
        depth = len(family)
        assert 1 <= depth <= len(templates), "frame depth must map to an existing template"
        innermost = family[-1]
        _tile_fill(out, innermost["bbox"], int(innermost["color"]), templates[depth - 1])
    return out


def _random_template_mask(rng: random.Random, h: int, w: int, seen: set[bytes]) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        count = rng.randint(1, h * w - 1)
        idxs = rng.sample(range(h * w), count)
        for idx in idxs:
            mask[idx // w, idx % w] = True
        if not mask.any() or mask.all():
            continue
        key = mask.astype(np.uint8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        return mask
    raise RuntimeError("failed to generate a distinct template mask")


def _make_frame_spec(rng: random.Random, depth: int) -> dict:
    fill_min = 1 if depth >= 3 else 2
    fill_max = max(fill_min, 7 - depth)
    fill_h = rng.randint(fill_min, fill_max)
    fill_w = rng.randint(fill_min, fill_max)
    outer_h = fill_h + 2 + 4 * (depth - 1)
    outer_w = fill_w + 2 + 4 * (depth - 1)
    return {
        "depth": depth,
        "fill_h": fill_h,
        "fill_w": fill_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
    }


def _area_bounds(specs: list[dict]) -> tuple[int, int]:
    padded_area = sum((spec["outer_h"] + 2) * (spec["outer_w"] + 2) for spec in specs)
    side = int(np.ceil(np.sqrt(padded_area)))
    min_h = max(side, max(spec["outer_h"] + 2 for spec in specs))
    min_w = max(int(np.ceil(padded_area / min_h)), max(spec["outer_w"] + 2 for spec in specs))
    return min_h, min_w


def _place_specs(rng: random.Random, area_h: int, area_w: int, specs: list[dict]) -> bool:
    occupied = np.zeros((area_h, area_w), dtype=bool)
    ordered = sorted(specs, key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    for spec in ordered:
        h = spec["outer_h"]
        w = spec["outer_w"]
        if area_h < h + 2 or area_w < w + 2:
            return False
        candidates: list[tuple[int, int]] = []
        for r in range(1, area_h - h):
            for c in range(1, area_w - w):
                if occupied[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        spec["top"] = r0
        spec["left"] = c0
        occupied[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
    return True


def _draw_frame_family(grid: np.ndarray, spec: dict, color: int) -> None:
    r0 = spec["top"]
    c0 = spec["left"]
    h = spec["outer_h"]
    w = spec["outer_w"]
    for layer in range(spec["depth"]):
        rr0 = r0 + 2 * layer
        cc0 = c0 + 2 * layer
        rr1 = r0 + h - 1 - 2 * layer
        cc1 = c0 + w - 1 - 2 * layer
        grid[rr0, cc0 : cc1 + 1] = color
        grid[rr1, cc0 : cc1 + 1] = color
        grid[rr0 : rr1 + 1, cc0] = color
        grid[rr0 : rr1 + 1, cc1] = color


def _draw_bottom_templates(
    grid: np.ndarray,
    templates: list[np.ndarray],
    colors: list[int],
    x0: int,
) -> None:
    ph, pw = templates[0].shape
    band_h = ph + 2
    band_w = 2 + len(templates) * pw + (len(templates) - 1)
    y0 = grid.shape[0] - band_h
    grid[y0 : y0 + band_h, x0 : x0 + band_w] = BLACK
    grid[y0, x0 : x0 + band_w] = GREY
    grid[y0 + band_h - 1, x0 : x0 + band_w] = GREY
    grid[y0 : y0 + band_h, x0] = GREY
    grid[y0 : y0 + band_h, x0 + band_w - 1] = GREY
    cur = x0 + 1
    for i, (mask, color) in enumerate(zip(templates, colors)):
        patch = np.where(mask, color, BLACK)
        grid[y0 + 1 : y0 + 1 + ph, cur : cur + pw] = patch
        cur += pw
        if i != len(templates) - 1:
            grid[y0 + 1 : y0 + 1 + ph, cur] = GREY
            cur += 1


def _draw_right_templates(
    grid: np.ndarray,
    templates: list[np.ndarray],
    colors: list[int],
    y0: int,
) -> None:
    ph, pw = templates[0].shape
    band_h = 2 + len(templates) * ph + (len(templates) - 1)
    band_w = pw + 2
    x0 = grid.shape[1] - band_w
    grid[y0 : y0 + band_h, x0 : x0 + band_w] = BLACK
    grid[y0, x0 : x0 + band_w] = GREY
    grid[y0 + band_h - 1, x0 : x0 + band_w] = GREY
    grid[y0 : y0 + band_h, x0] = GREY
    grid[y0 : y0 + band_h, x0 + band_w - 1] = GREY
    cur = y0 + 1
    for i, (mask, color) in enumerate(zip(templates, colors)):
        patch = np.where(mask, color, BLACK)
        grid[cur : cur + ph, x0 + 1 : x0 + 1 + pw] = patch
        cur += ph
        if i != len(templates) - 1:
            grid[cur, x0 + 1 : x0 + 1 + pw] = GREY
            cur += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["bottom", "right"])
    template_count = rng.randint(2, 4)
    frame_count = rng.randint(2, min(3, template_count))
    chosen_depths = {template_count}
    if frame_count > 1:
        chosen_depths.add(1)
    while len(chosen_depths) < frame_count:
        chosen_depths.add(rng.randint(1, template_count))
    depths = list(chosen_depths)
    rng.shuffle(depths)

    spec_map = [_make_frame_spec(rng, depth) for depth in depths]
    ph = rng.randint(2, 4)
    pw = rng.randint(2, 4)
    band_h = ph + 2 if orientation == "bottom" else 2 + template_count * ph + (template_count - 1)
    band_w = 2 + template_count * pw + (template_count - 1) if orientation == "bottom" else pw + 2

    min_area_h, min_area_w = _area_bounds(spec_map)
    total_h_min = min_area_h + (band_h + 1 if orientation == "bottom" else 0)
    total_w_min = min_area_w + (band_w + 1 if orientation == "right" else 0)
    total_h_min = max(total_h_min, band_h + (1 if orientation == "right" else 0))
    total_w_min = max(total_w_min, band_w + (1 if orientation == "bottom" else 0))
    if total_h_min > ARC_MAX or total_w_min > ARC_MAX:
        return None

    total_h = rng.randint(total_h_min, ARC_MAX)
    total_w = rng.randint(total_w_min, ARC_MAX)
    area_h = total_h - band_h - 1 if orientation == "bottom" else total_h
    area_w = total_w - band_w - 1 if orientation == "right" else total_w

    if not _place_specs(rng, area_h, area_w, spec_map):
        return None

    palette = COLORS.copy()
    rng.shuffle(palette)
    template_colors = palette[:template_count]
    frame_colors = palette[template_count : template_count + frame_count]

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    for spec, color in zip(spec_map, frame_colors):
        _draw_frame_family(grid, spec, color)

    seen_masks: set[bytes] = set()
    template_masks = [_random_template_mask(rng, ph, pw, seen_masks) for _ in range(template_count)]
    if orientation == "bottom":
        max_x0 = total_w - band_w - 1
        if max_x0 < 0:
            return None
        x0 = rng.randint(0, max_x0)
        _draw_bottom_templates(grid, template_masks, template_colors, x0)
    else:
        max_y0 = total_h - band_h - 1
        if max_y0 < 0:
            return None
        y0 = rng.randint(0, max_y0)
        _draw_right_templates(grid, template_masks, template_colors, y0)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
