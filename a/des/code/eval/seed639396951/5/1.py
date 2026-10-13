from __future__ import annotations

from collections import deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, box[0] - margin),
        max(0, box[1] - margin),
        min(h - 1, box[2] + margin),
        min(w - 1, box[3] + margin),
    )


def _component_records(grid: np.ndarray, wall: int, bg: int) -> list[dict]:
    records: list[dict] = []
    for coords in _components_mask(grid != wall):
        if _touches_boundary(coords, grid.shape):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        vals = grid[coords[:, 0], coords[:, 1]]
        records.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "has_bg": bool(np.any(vals == bg)),
                "full_rect": len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )
    return records


def _find_wall_and_parts(grid: np.ndarray, bg: int) -> tuple[int, list[dict], list[dict]]:
    candidates: list[tuple[tuple[int, int, int, int], int, list[dict], list[dict]]] = []
    for wall in map(int, np.unique(grid)):
        if wall == bg:
            continue
        comps = _component_records(grid, wall, bg)
        panels = [comp for comp in comps if comp["full_rect"] and not comp["has_bg"]]
        regions = [comp for comp in comps if comp["has_bg"]]
        if len(panels) < 2 or len(regions) < 1:
            continue
        if len({(p["bbox"][2] - p["bbox"][0] + 1, p["bbox"][3] - p["bbox"][1] + 1) for p in panels}) != 1:
            continue
        panel_top = min(p["bbox"][0] for p in panels)
        if not all(r["bbox"][2] < panel_top for r in regions):
            continue
        score = (len(panels) + len(regions), len(regions), len(panels), int(np.sum(grid == wall)))
        candidates.append((score, wall, panels, regions))

    assert candidates, "failed to identify the grey wall color"
    best_score = max(score for score, *_ in candidates)
    best = [entry for entry in candidates if entry[0] == best_score]
    assert len(best) == 1, "wall color must be structurally unique"
    _, wall, panels, regions = best[0]
    return int(wall), panels, regions


def _extract_templates(grid: np.ndarray, panels: list[dict]) -> list[np.ndarray]:
    ordered = sorted(panels, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    templates = []
    dims = None
    for comp in ordered:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        if dims is None:
            dims = patch.shape
        assert patch.shape == dims, "template panels must share one size"
        templates.append(patch)
    assert len(templates) >= 2
    return templates


def _fragment_signature(grid: np.ndarray, region: dict, bg: int) -> list[tuple[int, int, int]]:
    r0, c0, _, _ = region["bbox"]
    pieces: list[tuple[int, int, int]] = []
    for r, c in region["coords"]:
        color = int(grid[int(r), int(c)])
        if color != bg:
            pieces.append((int(r) - r0, int(c) - c0, color))
    return sorted(pieces)


def _match_template(fragments: list[tuple[int, int, int]], templates: list[np.ndarray]) -> int:
    matches = []
    for idx, template in enumerate(templates):
        th, tw = template.shape
        if all(int(template[r % th, c % tw]) == color for r, c, color in fragments):
            matches.append(idx)
    assert len(matches) == 1, "fragments must identify exactly one template"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer which bottom template matches each enclosed region's colored fragments,
    then tile that template across the region interior from its bbox top-left.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    _wall, panels, regions = _find_wall_and_parts(g, bg)
    templates = _extract_templates(g, panels)

    out = g.copy()
    for region in regions:
        fragments = _fragment_signature(g, region, bg)
        assert 2 <= len(fragments) <= 3, "each region must contain 2-3 fragments"
        template = templates[_match_template(fragments, templates)]
        th, tw = template.shape
        r0, c0, _, _ = region["bbox"]
        for r, c in region["coords"]:
            rr, cc = int(r), int(c)
            color = int(template[(rr - r0) % th, (cc - c0) % tw])
            if int(g[rr, cc]) == bg:
                out[rr, cc] = color
            else:
                assert int(g[rr, cc]) == color, "fragment colors must agree with the matched template"
    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (24, 35)
TEMPLATE_COUNT_CHOICES = (3, 4)
TEMPLATE_SIZE_RANGE = (3, 5)
REGION_COUNT_CHOICES = (2, 4)
FRAGMENT_COUNT_CHOICES = (2, 3)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120
FRAGMENT_RETRIES = 240

SHAPE_TYPES = ("rect", "lshape", "cross")
TEMPLATE_FAMILIES = ("vertical", "horizontal", "checker", "diag", "rings", "quadrants")


def _template_from_family(family: str, h: int, w: int, colors: list[int]) -> np.ndarray:
    arr = np.zeros((h, w), dtype=int)
    if family == "vertical":
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[c % len(colors)]
    elif family == "horizontal":
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[r % len(colors)]
    elif family == "checker":
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r + c) % len(colors)]
    elif family == "diag":
        for r in range(h):
            for c in range(w):
                arr[r, c] = colors[(r + 2 * c) % len(colors)]
    elif family == "rings":
        for r in range(h):
            for c in range(w):
                dist = min(r, h - 1 - r, c, w - 1 - c)
                arr[r, c] = colors[dist % len(colors)]
    elif family == "quadrants":
        mid_r = h // 2
        mid_c = w // 2
        for r in range(h):
            for c in range(w):
                idx = (0 if r < mid_r else 1) + (0 if c < mid_c else 2)
                arr[r, c] = colors[idx % len(colors)]
    else:
        raise AssertionError("unknown template family")
    return arr


def _make_templates(fill_colors: list[int], count: int, h: int, w: int) -> list[np.ndarray] | None:
    families = random.sample(TEMPLATE_FAMILIES, count)
    templates: list[np.ndarray] = []
    for family in families:
        palette_size = 3 if family in {"diag", "rings", "quadrants"} else 2
        colors = random.sample(fill_colors, palette_size)
        random.shuffle(colors)
        template = _template_from_family(family, h, w, colors)
        if len(set(map(int, template.ravel()))) < 2:
            return None
        templates.append(template)
    if len({tuple(t.ravel().tolist()) for t in templates}) != len(templates):
        return None
    return templates


def _shape_rect() -> np.ndarray:
    h = random.randint(4, 8)
    w = random.randint(4, 9)
    return np.ones((h, w), dtype=bool)


def _shape_lshape() -> np.ndarray:
    h = random.randint(5, 9)
    w = random.randint(5, 9)
    vert = random.randint(2, min(3, w - 1))
    horiz = random.randint(2, min(3, h - 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :vert] = True
    mask[-horiz:, :] = True
    return np.rot90(mask, random.randrange(4)).copy()


def _shape_cross() -> np.ndarray:
    h = random.randint(5, 9)
    w = random.randint(5, 9)
    bar_h = random.randint(2, min(3, h - 1))
    bar_w = random.randint(2, min(3, w - 1))
    mask = np.zeros((h, w), dtype=bool)
    r0 = (h - bar_h) // 2
    c0 = (w - bar_w) // 2
    mask[r0 : r0 + bar_h, :] = True
    mask[:, c0 : c0 + bar_w] = True
    return mask


def _random_shape(shape_type: str) -> np.ndarray:
    if shape_type == "rect":
        return _shape_rect()
    if shape_type == "lshape":
        return _shape_lshape()
    if shape_type == "cross":
        return _shape_cross()
    raise AssertionError("unknown region shape")


def _region_patch(mask: np.ndarray, wall: int, bg: int) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1 : h + 1, 1 : w + 1] = mask
    for r in range(h + 2):
        for c in range(w + 2):
            if inner[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h + 2 and 0 <= nc < w + 2 and inner[nr, nc]:
                        patch[r, c] = wall
                        dr = 2
                        break
                else:
                    continue
                break
    return patch


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _choose_fragments(mask: np.ndarray, template_idx: int, templates: list[np.ndarray], count: int) -> list[tuple[int, int, int]] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    target = templates[template_idx]
    th, tw = target.shape
    for _ in range(FRAGMENT_RETRIES):
        picks = random.sample(cells, count)
        if count >= 2 and max(_manhattan(a, b) for i, a in enumerate(picks) for b in picks[i + 1 :]) < 2:
            continue
        fragments = [(r, c, int(target[r % th, c % tw])) for r, c in picks]
        matches = []
        for idx, template in enumerate(templates):
            hh, ww = template.shape
            if all(int(template[r % hh, c % ww]) == color for r, c, color in fragments):
                matches.append(idx)
        if matches == [template_idx]:
            return fragments
    return None


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    max_bottom: int,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    h, w = grid.shape
    if ph > max_bottom or pw > w - 2:
        return None
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(0, max_bottom - ph)
        left = random.randint(1, w - pw - 1)
        box = (top, left, top + ph - 1, left + pw - 1)
        keep_clear = _expand_box(box, 1, grid.shape)
        if any(_boxes_overlap(keep_clear, other) for other in reserved):
            continue
        grid[top : top + ph, left : left + pw] = patch
        reserved.append(keep_clear)
        return top, left
    return None


def _render_expected(
    inp: np.ndarray,
    regions: list[dict],
    templates: list[np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        top = int(region["top"])
        left = int(region["left"])
        mask = region["mask"]
        template = templates[int(region["template_idx"])]
        th, tw = template.shape
        for r, c in np.argwhere(mask):
            rr = top + int(r)
            cc = left + int(c)
            out[rr, cc] = int(template[int(r) % th, int(c) % tw])
    return out


def _valid_sample(inp: np.ndarray, bg: int) -> bool:
    vals, cnts = np.unique(inp, return_counts=True)
    return int(vals[int(np.argmax(cnts))]) == bg


def _gen_attempt() -> np.ndarray | None:
    colors = list(range(10))
    random.shuffle(colors)
    bg, wall = colors[:2]
    fill_colors = colors[2:]

    template_count = random.randint(TEMPLATE_COUNT_CHOICES[0], TEMPLATE_COUNT_CHOICES[1])
    template_h = random.randint(TEMPLATE_SIZE_RANGE[0], TEMPLATE_SIZE_RANGE[1])
    template_w = random.randint(TEMPLATE_SIZE_RANGE[0], TEMPLATE_SIZE_RANGE[1])
    templates = _make_templates(fill_colors, template_count, template_h, template_w)
    if templates is None:
        return None

    panel_h = template_h + 2
    panel_w = template_w + 2
    strip_w = panel_w * template_count
    grid_h = max(random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1]), panel_h + 10)
    grid_w = max(random.randint(max(GRID_W_RANGE[0], strip_w + 3), GRID_W_RANGE[1]), strip_w + 3)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    strip_top = grid_h - panel_h
    strip_left = random.randint(1, grid_w - strip_w - 1)
    for idx, template in enumerate(templates):
        left = strip_left + idx * panel_w
        grid[strip_top : strip_top + panel_h, left : left + panel_w] = wall
        grid[strip_top + 1 : strip_top + 1 + template_h, left + 1 : left + 1 + template_w] = template

    reserved = [_expand_box((strip_top, strip_left, grid_h - 1, strip_left + strip_w - 1), 1, grid.shape)]
    region_count = random.randint(REGION_COUNT_CHOICES[0], REGION_COUNT_CHOICES[1])
    assignments = [random.randrange(template_count) for _ in range(region_count)]
    if region_count >= 2 and len(set(assignments)) == 1:
        choices = [idx for idx in range(template_count) if idx != assignments[0]]
        assignments[random.randrange(region_count)] = random.choice(choices)

    regions: list[dict] = []
    max_bottom = strip_top - 2
    if max_bottom < 8:
        return None

    for template_idx in assignments:
        mask = _random_shape(random.choice(SHAPE_TYPES))
        fragments = _choose_fragments(mask, template_idx, templates, random.choice(FRAGMENT_COUNT_CHOICES))
        if fragments is None:
            return None
        patch = _region_patch(mask, wall, bg)
        placed = _place_patch(grid, patch, reserved, max_bottom)
        if placed is None:
            return None
        patch_top, patch_left = placed
        top = patch_top + 1
        left = patch_left + 1
        for r, c, color in fragments:
            grid[top + r, left + c] = color
        regions.append({"top": top, "left": left, "mask": mask, "template_idx": template_idx})

    inp = grid
    expected = _render_expected(inp, regions, templates)
    if not _valid_sample(inp, bg):
        return None
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
