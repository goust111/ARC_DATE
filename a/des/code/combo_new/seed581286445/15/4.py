import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
WORK_H = 30
WORK_W = 28
UPPER_H = 15
LOWER_H = WORK_H - UPPER_H - 1
TEMPLATE_COUNT_RANGE = (2, 4)
TEMPLATE_DIM_RANGE = (4, 7)
OBJECT_DIM_RANGE = (2, 6)
PLACEMENT_ATTEMPTS = 160
GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    corners = np.array(
        [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]],
        dtype=int,
    )
    values, counts = np.unique(corners, return_counts=True)
    if counts.max() >= 2:
        return int(values[np.argmax(counts)])
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(grid: np.ndarray, bg: int, monochromatic: bool) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) == int(bg):
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append({"coords": arr, "bbox": (r0, c0, r1, c1)})
    return comps


def _uniform_non_bg_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != int(bg):
            rows.append(int(r))
    return rows


def _component_mask(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lower object to the upper template sharing its anchor color, then stamp the object into that template's bbox using the anchor position as the placement offset.

    Input structure:
    - The background is Black, and one full Grey row splits the grid into an upper template half and a lower object half.
    - The upper half contains several disconnected colored template regions; each template has one differently colored anchor pixel inside it.
    - The lower half contains the same number of disconnected monochrome objects, one unique color per object, and each object color matches exactly one template anchor color.

    Rule:
    - Find the Grey separator row, then extract upper-half template components as non-Black connected regions and lower-half objects as monochrome connected regions.
    - For each template, identify its dominant fill color and its unique one-pixel anchor color; use that anchor color to match the template with the lower object of the same color.
    - Crop the output to the full upper half only, keep the original template half unchanged, and overlay each matched object so the object's bbox top-left lands on the anchor's relative position inside the template bbox; clip any object pixels that extend beyond that bbox.

    Key insight:
    - The anchor color names the correct object, while the anchor's location inside the template bbox tells where that object's own bbox origin must be placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    sep_rows = _uniform_non_bg_rows(g, bg)
    assert len(sep_rows) == 1, "expected exactly one full non-background separator row"
    sep_r = int(sep_rows[0])

    upper = g[:sep_r].copy()
    lower = g[sep_r + 1 :]
    assert upper.size > 0 and lower.size > 0, "expected content above and below the separator"

    templates: dict[int, dict] = {}
    for comp in _components(upper, bg=bg, monochromatic=False):
        coords = comp["coords"]
        vals = upper[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(vals, return_counts=True)
        assert len(colors) == 2, "template should have one base color and one anchor color"

        base_color = int(colors[np.argmax(counts)])
        anchor_choices = [int(color) for color, count in zip(colors, counts) if int(count) == 1 and int(color) != base_color]
        assert len(anchor_choices) == 1, "template should contain exactly one one-pixel anchor"
        anchor_color = int(anchor_choices[0])
        anchor_pts = coords[vals == anchor_color]
        assert anchor_pts.shape[0] == 1

        ar, ac = map(int, anchor_pts[0])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert anchor_color not in templates, "anchor colors must be unique across templates"
        templates[anchor_color] = {
            "bbox": (r0, c0, r1, c1),
            "base_color": base_color,
            "anchor_rel": (ar - r0, ac - c0),
        }

    objects: dict[int, np.ndarray] = {}
    for comp in _components(lower, bg=bg, monochromatic=True):
        coords = comp["coords"]
        vals = lower[coords[:, 0], coords[:, 1]]
        colors = np.unique(vals)
        assert len(colors) == 1, "lower objects must be monochrome"
        color = int(colors[0])
        assert color not in objects, "object colors must be unique"
        objects[color] = _component_mask(lower, comp["bbox"], color)

    assert templates, "expected at least one template"
    assert set(objects) == set(templates), "object colors must match template anchor colors exactly"

    out = upper.copy()
    for color, template in templates.items():
        obj_mask = objects[color]
        r0, c0, r1, c1 = template["bbox"]
        anchor_r, anchor_c = template["anchor_rel"]
        for rr, cc in np.argwhere(obj_mask):
            dst_r = int(r0 + anchor_r + rr)
            dst_c = int(c0 + anchor_c + cc)
            if r0 <= dst_r <= r1 and c0 <= dst_c <= c1:
                out[dst_r, dst_c] = int(color)
    return out


def _template_mask(shape_type: str, h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    if shape_type == "rect":
        return np.ones((h, w), dtype=bool)

    if shape_type == "l":
        arm_h = random.randint(1, h - 1)
        arm_w = random.randint(1, w - 1)
        orient = random.choice(("tl", "tr", "bl", "br"))
        if orient == "tl":
            return (rr < arm_h) | (cc < arm_w)
        if orient == "tr":
            return (rr < arm_h) | (cc >= w - arm_w)
        if orient == "bl":
            return (rr >= h - arm_h) | (cc < arm_w)
        return (rr >= h - arm_h) | (cc >= w - arm_w)

    bar = random.randint(1, min(2, h - 1 if shape_type in ("t_up", "t_down") else w - 1))
    stem = random.randint(1, min(2, w if shape_type in ("t_up", "t_down") else h))
    mask = np.zeros((h, w), dtype=bool)

    if shape_type == "t_up":
        stem_c0 = random.randint(0, w - stem)
        mask[:bar, :] = True
        mask[bar:, stem_c0 : stem_c0 + stem] = True
    elif shape_type == "t_down":
        stem_c0 = random.randint(0, w - stem)
        mask[h - bar :, :] = True
        mask[: h - bar, stem_c0 : stem_c0 + stem] = True
    elif shape_type == "t_left":
        stem_r0 = random.randint(0, h - stem)
        mask[:, :bar] = True
        mask[stem_r0 : stem_r0 + stem, bar:] = True
    else:
        stem_r0 = random.randint(0, h - stem)
        mask[:, w - bar :] = True
        mask[stem_r0 : stem_r0 + stem, : w - bar] = True
    return mask


def _random_template_mask(h: int, w: int) -> np.ndarray:
    shape_type = random.choice(("rect", "l", "t_up", "t_down", "t_left", "t_right"))
    return _template_mask(shape_type, h, w)


def _path_growth_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    steps = ["D"] * (h - 1) + ["R"] * (w - 1)
    random.shuffle(steps)
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    while int(mask.sum()) < int(target):
        frontier: list[tuple[int, int]] = []
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                if not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _object_mask(mask_type: str, h: int, w: int) -> np.ndarray:
    if mask_type == "rect":
        return np.ones((h, w), dtype=bool)

    if mask_type == "hook":
        rr, cc = np.indices((h, w))
        arm_h = random.randint(1, h - 1)
        arm_w = random.randint(1, w - 1)
        orient = random.choice(("tl", "tr", "bl"))
        if orient == "tl":
            mask = (rr < arm_h) | (cc < arm_w)
        elif orient == "tr":
            mask = (rr < arm_h) | (cc >= w - arm_w)
        else:
            mask = (rr >= h - arm_h) | (cc < arm_w)
        assert mask[0, 0]
        return mask

    area = h * w
    min_target = max(h + w - 1, (area + 1) // 2)
    target = random.randint(min_target, area)
    return _path_growth_mask(h, w, target)


def _random_object_mask(h: int, w: int) -> np.ndarray:
    for _ in range(20):
        mask = _object_mask(random.choice(("rect", "hook", "poly")), h, w)
        if mask[0, 0] and mask[-1].any() and mask[:, -1].any():
            return mask
    raise RuntimeError("failed to build object mask")


def _choose_anchor_and_object(template_mask: np.ndarray, clip_mode: bool) -> tuple[tuple[int, int], np.ndarray]:
    th, tw = map(int, template_mask.shape)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
    random.shuffle(cells)

    for ar, ac in cells:
        room_h = th - ar
        room_w = tw - ac

        dims: list[tuple[int, int]] = []
        if clip_mode:
            for oh in range(OBJECT_DIM_RANGE[0], OBJECT_DIM_RANGE[1] + 1):
                for ow in range(OBJECT_DIM_RANGE[0], OBJECT_DIM_RANGE[1] + 1):
                    if oh > room_h or ow > room_w:
                        dims.append((oh, ow))
        else:
            max_h = min(OBJECT_DIM_RANGE[1], room_h)
            max_w = min(OBJECT_DIM_RANGE[1], room_w)
            for oh in range(OBJECT_DIM_RANGE[0], max_h + 1):
                for ow in range(OBJECT_DIM_RANGE[0], max_w + 1):
                    dims.append((oh, ow))
        random.shuffle(dims)

        for oh, ow in dims[:60]:
            mask = _random_object_mask(oh, ow)
            if clip_mode == ((oh > room_h) or (ow > room_w)) or clip_mode:
                return (ar, ac), mask
    raise RuntimeError("failed to choose anchor/object pair")


def _make_specs() -> list[dict]:
    count = random.randint(*TEMPLATE_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, 2 * count)
    base_colors = colors[:count]
    anchor_colors = colors[count:]

    clip_modes = [True, False] * (count // 2)
    if count % 2 == 1:
        clip_modes.append(random.choice((True, False)))
    random.shuffle(clip_modes)

    specs: list[dict] = []
    for idx in range(count):
        for _ in range(80):
            th = random.randint(*TEMPLATE_DIM_RANGE)
            tw = random.randint(*TEMPLATE_DIM_RANGE)
            template_mask = _random_template_mask(th, tw)
            if int(template_mask.sum()) < max(6, th + tw - 2):
                continue
            try:
                anchor, object_mask = _choose_anchor_and_object(template_mask, clip_modes[idx])
            except RuntimeError:
                continue
            specs.append(
                {
                    "template_mask": template_mask,
                    "template_h": th,
                    "template_w": tw,
                    "base_color": int(base_colors[idx]),
                    "anchor_color": int(anchor_colors[idx]),
                    "anchor": anchor,
                    "object_mask": object_mask,
                    "clip_mode": bool(clip_modes[idx]),
                }
            )
            break
        else:
            raise RuntimeError("failed to make pair spec")
    return specs


def _placement_starts(limit: int, span: int) -> list[int]:
    lo = 1
    hi = limit - span - 1
    if hi < lo:
        lo = 0
        hi = limit - span
    return list(range(lo, hi + 1))


def _place_in_region(region_h: int, region_w: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order = sorted(range(len(spans)), key=lambda i: spans[i][0] * spans[i][1], reverse=True)
    blocked = np.zeros((region_h, region_w), dtype=bool)
    placed: list[tuple[int, int] | None] = [None] * len(spans)

    for idx in order:
        h, w = spans[idx]
        row_starts = _placement_starts(region_h, h)
        col_starts = _placement_starts(region_w, w)
        candidates = [(r, c) for r in row_starts for c in col_starts]
        random.shuffle(candidates)

        success = False
        for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            placed[idx] = (r0, c0)
            blocked[max(0, r0 - 1) : min(region_h, r1 + 2), max(0, c0 - 1) : min(region_w, c1 + 2)] = True
            success = True
            break
        if not success:
            raise RuntimeError("placement failed")

    return [(int(r), int(c)) for r, c in placed]


def _render_expected_upper(specs: list[dict], template_pos: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((UPPER_H, WORK_W), BG, dtype=int)
    for spec, (r0, c0) in zip(specs, template_pos):
        mask = spec["template_mask"]
        th, tw = mask.shape
        patch = out[r0 : r0 + th, c0 : c0 + tw]
        patch[mask] = int(spec["base_color"])
        ar, ac = spec["anchor"]
        out[r0 + ar, c0 + ac] = int(spec["anchor_color"])

    for spec, (r0, c0) in zip(specs, template_pos):
        ar, ac = spec["anchor"]
        obj_mask = spec["object_mask"]
        th, tw = spec["template_mask"].shape
        for rr, cc in np.argwhere(obj_mask):
            dst_r = int(r0 + ar + rr)
            dst_c = int(c0 + ac + cc)
            if r0 <= dst_r < r0 + th and c0 <= dst_c < c0 + tw:
                out[dst_r, dst_c] = int(spec["anchor_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    try:
        specs = _make_specs()
    except RuntimeError:
        return None
    template_spans = [(spec["template_h"], spec["template_w"]) for spec in specs]
    object_spans = [tuple(map(int, spec["object_mask"].shape)) for spec in specs]

    template_pos = object_pos = None
    for _ in range(20):
        try:
            template_pos = _place_in_region(UPPER_H, WORK_W, template_spans)
            object_pos = _place_in_region(LOWER_H, WORK_W, object_spans)
            break
        except RuntimeError:
            continue
    if template_pos is None or object_pos is None:
        return None

    grid = np.full((WORK_H, WORK_W), BG, dtype=int)
    grid[UPPER_H, :] = SEP

    for spec, (r0, c0) in zip(specs, template_pos):
        mask = spec["template_mask"]
        th, tw = mask.shape
        grid[r0 : r0 + th, c0 : c0 + tw][mask] = int(spec["base_color"])
        ar, ac = spec["anchor"]
        grid[r0 + ar, c0 + ac] = int(spec["anchor_color"])

    for spec, (r0, c0) in zip(specs, object_pos):
        mask = spec["object_mask"]
        oh, ow = mask.shape
        abs_r = UPPER_H + 1 + r0
        patch = grid[abs_r : abs_r + oh, c0 : c0 + ow]
        patch[mask] = int(spec["anchor_color"])

    content = (grid != BG).copy()
    content[UPPER_H, :] = False
    coords = np.argwhere(content)
    if coords.size == 0:
        return None

    min_r = int(coords[:, 0].min())
    max_r = int(coords[:, 0].max())
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())
    if min_r < 1 or min_c < 1 or max_r > WORK_H - 2 or max_c > WORK_W - 2:
        return None

    top_pad = random.randint(1, min(2, min_r))
    left_pad = random.randint(1, min(2, min_c))
    bottom_pad = random.randint(1, min(2, WORK_H - 1 - max_r))
    right_pad = random.randint(1, min(2, WORK_W - 1 - max_c))

    r0 = min_r - top_pad
    c0 = min_c - left_pad
    r1 = max_r + bottom_pad
    c1 = max_c + right_pad

    inp = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected_full = _render_expected_upper(specs, template_pos)
    expected = expected_full[r0:UPPER_H, c0 : c1 + 1]

    if inp.ndim != 2 or max(inp.shape) > 30 or expected.size == 0:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except RuntimeError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
