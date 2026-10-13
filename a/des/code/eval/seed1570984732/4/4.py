import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
QUADRANT_NAMES = ("tl", "tr", "bl", "br")
GREY_COLOR = 5
QUADRANT_SIZE_RANGE = (10, 13)
TEMPLATE_RETRIES = 400
GENERATION_RETRIES = 500
TEMPLATE_SPANS = (3, 5)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_asymmetric(mask: np.ndarray) -> bool:
    checks = [np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    if mask.shape[0] == mask.shape[1]:
        checks.extend([np.rot90(mask, 1), np.rot90(mask, 3), mask.T, np.fliplr(mask.T)])
    return all(not np.array_equal(mask, other) for other in checks)


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    center = (h // 2, w // 2)
    mask[center] = True
    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True
    return mask


def _sample_template_mask() -> np.ndarray:
    for _ in range(TEMPLATE_RETRIES):
        h = random.choice(TEMPLATE_SPANS)
        w = random.choice(TEMPLATE_SPANS)
        target = random.randint(6, min(h * w - 1, 8))
        mask = _grow_connected_mask(h, w, target)
        tight = _crop_mask(mask)
        if tight.shape != (h, w):
            continue
        if not tight[tight.shape[0] // 2, tight.shape[1] // 2]:
            continue
        if not _is_asymmetric(tight):
            continue
        if random.choice((False, True)):
            tight = np.flipud(tight)
        if random.choice((False, True)):
            tight = np.fliplr(tight)
        if tight.shape[0] == tight.shape[1]:
            tight = np.rot90(tight, random.randrange(4))
        return tight
    raise RuntimeError("failed to sample an asymmetric template mask")


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows: dict[int, list[int]] = {}
    full_cols: dict[int, list[int]] = {}
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            full_rows.setdefault(int(grid[r, 0]), []).append(r)
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            full_cols.setdefault(int(grid[0, c]), []).append(c)
    candidates = [
        (color, rows[0], full_cols[color][0])
        for color, rows in full_rows.items()
        if color in full_cols and color != bg and len(rows) == 1 and len(full_cols[color]) == 1
    ]
    assert len(candidates) == 1, "expected exactly one full-width/full-height cross"
    return candidates[0]


def _quadrant_specs(h: int, w: int, cross_r: int, cross_c: int) -> dict[str, dict[str, int]]:
    return {
        "tl": {"r0": 0, "c0": 0, "h": cross_r, "w": cross_c},
        "tr": {"r0": 0, "c0": cross_c + 1, "h": cross_r, "w": w - cross_c - 1},
        "bl": {"r0": cross_r + 1, "c0": 0, "h": h - cross_r - 1, "w": cross_c},
        "br": {"r0": cross_r + 1, "c0": cross_c + 1, "h": h - cross_r - 1, "w": w - cross_c - 1},
    }


def _extract_template_offsets(quad: np.ndarray, bg: int, grey: int) -> tuple[int, list[tuple[int, int]]]:
    mask = (quad != bg) & (quad != grey)
    colors = {int(x) for x in quad[mask]}
    assert len(colors) == 1, "template quadrant must contain one foreground color"
    comps = _components4(mask)
    assert len(comps) == 1 and len(comps[0]) >= 2, "template must be a single connected component"
    coords = np.array(comps[0], dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_h = int(r1 - r0 + 1)
    bbox_w = int(c1 - c0 + 1)
    assert bbox_h <= 5 and bbox_w <= 5 and bbox_h % 2 == 1 and bbox_w % 2 == 1
    center_r = (int(r0) + int(r1)) // 2
    center_c = (int(c0) + int(c1)) // 2
    offsets = [(int(r) - center_r, int(c) - center_c) for r, c in comps[0]]
    return next(iter(colors)), offsets


def _cross_distance(name: str, h: int, w: int, r: int, c: int) -> int:
    if name == "tl":
        return (h - 1 - r) + (w - 1 - c)
    if name == "tr":
        return (h - 1 - r) + c
    if name == "bl":
        return r + (w - 1 - c)
    return r + c


def _quadrant_growth(
    anchor: tuple[int, int],
    offsets: list[tuple[int, int]],
    layers: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], bool]:
    inside: set[tuple[int, int]] = set()
    clipped = False
    for scale in range(1, layers + 1):
        for dr, dc in offsets:
            rr = anchor[0] + scale * dr
            cc = anchor[1] + scale * dc
            if 0 <= rr < h and 0 <= cc < w:
                inside.add((rr, cc))
            else:
                clipped = True
    return inside, clipped


def _identify_template_quadrant(grid: np.ndarray, bg: int, grey: int, specs: dict[str, dict[str, int]]) -> str:
    candidates = []
    for name, spec in specs.items():
        quad = grid[spec["r0"] : spec["r0"] + spec["h"], spec["c0"] : spec["c0"] + spec["w"]]
        mask = (quad != bg) & (quad != grey)
        if not mask.any():
            continue
        colors = {int(x) for x in quad[mask]}
        comps = _components4(mask)
        if len(colors) == 1 and len(comps) == 1 and len(comps[0]) >= 2:
            candidates.append(name)
    assert len(candidates) == 1, "expected a unique template quadrant"
    return candidates[0]


def _quadrant_anchor_and_layers(
    name: str,
    quad: np.ndarray,
    bg: int,
    grey: int,
) -> tuple[int, tuple[int, int], int]:
    mask = (quad != bg) & (quad != grey)
    colors = {int(x) for x in quad[mask]}
    assert len(colors) == 1, "count quadrant must use one color"
    color = next(iter(colors))
    coords = [tuple(map(int, coord)) for coord in np.argwhere(quad == color)]
    assert 2 <= len(coords) <= 4, "count quadrant must contain one anchor plus 1-3 dots"
    h, w = quad.shape
    dists = [_cross_distance(name, h, w, r, c) for r, c in coords]
    min_dist = min(dists)
    anchors = [coord for coord, dist in zip(coords, dists) if dist == min_dist]
    assert len(anchors) == 1, "anchor must be the unique closest same-color dot to the cross"
    return color, anchors[0], len(coords) - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grid split into quadrants by a grey cross, copy the template shape onto the other quadrants
    by scaling its center-relative offsets around each anchor for as many layers as the matching dot count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    grey, cross_r, cross_c = _find_cross(g, bg)
    specs = _quadrant_specs(g.shape[0], g.shape[1], cross_r, cross_c)
    template_name = _identify_template_quadrant(g, bg, grey, specs)
    template_spec = specs[template_name]
    template_quad = g[
        template_spec["r0"] : template_spec["r0"] + template_spec["h"],
        template_spec["c0"] : template_spec["c0"] + template_spec["w"],
    ]
    _template_color, offsets = _extract_template_offsets(template_quad, bg, grey)

    out = g.copy()
    for name, spec in specs.items():
        if name == template_name:
            continue
        quad = g[spec["r0"] : spec["r0"] + spec["h"], spec["c0"] : spec["c0"] + spec["w"]]
        color, anchor, layers = _quadrant_anchor_and_layers(name, quad, bg, grey)
        for scale in range(1, layers + 1):
            for dr, dc in offsets:
                rr = anchor[0] + scale * dr
                cc = anchor[1] + scale * dc
                if 0 <= rr < spec["h"] and 0 <= cc < spec["w"]:
                    out[spec["r0"] + rr, spec["c0"] + cc] = color
    return out


def _paint_mask(
    grid: np.ndarray,
    origin_r: int,
    origin_c: int,
    mask: np.ndarray,
    color: int,
) -> None:
    for r, c in np.argwhere(mask):
        grid[origin_r + int(r), origin_c + int(c)] = color


def _place_template(
    grid: np.ndarray,
    spec: dict[str, int],
    mask: np.ndarray,
    color: int,
) -> list[tuple[int, int]]:
    top = random.randint(0, spec["h"] - mask.shape[0])
    left = random.randint(0, spec["w"] - mask.shape[1])
    _paint_mask(grid, spec["r0"] + top, spec["c0"] + left, mask, color)
    center = (mask.shape[0] // 2, mask.shape[1] // 2)
    return [(int(r) - center[0], int(c) - center[1]) for r, c in np.argwhere(mask)]


def _pick_anchor(
    name: str,
    h: int,
    w: int,
    offsets: list[tuple[int, int]],
    layers: int,
    prefer_clipped: bool,
) -> tuple[tuple[int, int], set[tuple[int, int]], bool] | None:
    candidates = []
    for r in range(h):
        for c in range(w):
            growth, clipped = _quadrant_growth((r, c), offsets, layers, h, w)
            if not (growth - {(r, c)}):
                continue
            candidates.append(
                {
                    "anchor": (r, c),
                    "growth": growth,
                    "clipped": clipped,
                    "dist": _cross_distance(name, h, w, r, c),
                }
            )
    if not candidates:
        return None
    desired = [cand for cand in candidates if cand["clipped"] == prefer_clipped]
    pool = desired if desired else candidates
    min_dist = min(int(cand["dist"]) for cand in pool)
    close_pool = [cand for cand in pool if int(cand["dist"]) <= min_dist + 4]
    choice = random.choice(close_pool)
    return choice["anchor"], set(choice["growth"]), bool(choice["clipped"])


def _mark_blocked(blocked: set[tuple[int, int]], cell: tuple[int, int], h: int, w: int) -> None:
    r, c = cell
    blocked.add(cell)
    for dr, dc in DIR8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            blocked.add((nr, nc))


def _pick_extra_dots(
    name: str,
    h: int,
    w: int,
    anchor: tuple[int, int],
    anchor_dist: int,
    count: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked = set(forbidden)
    _mark_blocked(blocked, anchor, h, w)
    chosen: list[tuple[int, int]] = []
    for _ in range(count):
        pool = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in blocked and _cross_distance(name, h, w, r, c) >= anchor_dist + 2
        ]
        if not pool:
            return None
        cell = random.choice(pool)
        chosen.append(cell)
        _mark_blocked(blocked, cell, h, w)
    return chosen


def _build_expected(
    inp: np.ndarray,
    specs: dict[str, dict[str, int]],
    template_name: str,
    offsets: list[tuple[int, int]],
    placements: dict[str, dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for name, spec in specs.items():
        if name == template_name:
            continue
        anchor = placements[name]["anchor"]
        color = int(placements[name]["color"])
        layers = int(placements[name]["layers"])
        assert isinstance(anchor, tuple)
        for scale in range(1, layers + 1):
            for dr, dc in offsets:
                rr = anchor[0] + scale * dr
                cc = anchor[1] + scale * dc
                if 0 <= rr < spec["h"] and 0 <= cc < spec["w"]:
                    out[spec["r0"] + rr, spec["c0"] + cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(*QUADRANT_SIZE_RANGE)
    bottom_h = random.randint(*QUADRANT_SIZE_RANGE)
    left_w = random.randint(*QUADRANT_SIZE_RANGE)
    right_w = random.randint(*QUADRANT_SIZE_RANGE)
    h = top_h + bottom_h + 1
    w = left_w + right_w + 1

    bg = random.choice([color for color in range(10) if color != GREY_COLOR])
    palette = [color for color in range(10) if color not in {bg, GREY_COLOR}]
    random.shuffle(palette)
    template_color = palette[0]
    anchor_colors = palette[1:4]
    template_name = random.choice(QUADRANT_NAMES)
    template_mask = _sample_template_mask()

    grid = np.full((h, w), bg, dtype=int)
    grid[top_h, :] = GREY_COLOR
    grid[:, left_w] = GREY_COLOR
    specs = _quadrant_specs(h, w, top_h, left_w)

    template_offsets = _place_template(grid, specs[template_name], template_mask, template_color)
    placements: dict[str, dict[str, object]] = {}
    color_iter = iter(anchor_colors)
    clipped_flags = []
    for name in QUADRANT_NAMES:
        if name == template_name:
            continue
        spec = specs[name]
        color = next(color_iter)
        layers = random.randint(1, 3)
        prefer_clipped = random.choice((False, True))
        picked = _pick_anchor(name, spec["h"], spec["w"], template_offsets, layers, prefer_clipped)
        if picked is None:
            return None
        anchor, growth, clipped = picked
        anchor_dist = _cross_distance(name, spec["h"], spec["w"], anchor[0], anchor[1])
        extras = _pick_extra_dots(name, spec["h"], spec["w"], anchor, anchor_dist, layers, set(growth))
        if extras is None:
            return None
        for r, c in [anchor, *extras]:
            grid[spec["r0"] + r, spec["c0"] + c] = color
        placements[name] = {
            "anchor": anchor,
            "color": color,
            "layers": layers,
            "extras": extras,
        }
        clipped_flags.append(clipped)

    expected = _build_expected(grid, specs, template_name, template_offsets, placements)
    if np.array_equal(expected, grid):
        return None

    bg_counts = Counter(int(x) for x in grid.ravel())
    if bg_counts.most_common(1)[0][0] != bg:
        return None
    if random.choice((False, True)) and not any(clipped_flags):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
