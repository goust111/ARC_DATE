import random
from collections import Counter

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_ORDER = ("top", "right", "bottom", "left")
ROT_K_BY_SIDE = {"top": 0, "right": 3, "bottom": 2, "left": 1}

BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (26, 34)
GRID_W_RANGE = (26, 34)
NUM_FRAME_CHOICES = (4, 5)
THICKNESS_CHOICES = (1, 2)
TOP_BOTTOM_DIMS = (2, 3, 4, 5)
SIDE_SQUARE_DIMS = (2, 3, 4, 5)
CORNERS = ("tl", "tr", "bl", "br")
MAX_RETRIES = 300


def _boundary_background(grid: np.ndarray) -> int:
    edge = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    return int(Counter(edge.tolist()).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg_mask = ~mask
    if not bg_mask.any():
        return False
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        cr, cc = stack.pop()
        for dr, dc in DIR4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(bg_mask & ~seen))


def _frame_info(sub: np.ndarray, bg: int, origin_r: int, origin_c: int) -> dict | None:
    comp_mask = sub != bg
    if not (np.all(comp_mask[0]) and np.all(comp_mask[-1]) and np.all(comp_mask[:, 0]) and np.all(comp_mask[:, -1])):
        return None

    vals = sub[comp_mask]
    colors = [int(v) for v in np.unique(vals)]
    if len(colors) != 2:
        return None

    counts = Counter(vals.tolist())
    singleton = [color for color, cnt in counts.items() if int(cnt) == 1]
    if len(singleton) != 1:
        return None

    anchor_color = int(singleton[0])
    frame_color = next(color for color in colors if color != anchor_color)

    anchor_pts = np.argwhere(sub == anchor_color)
    if anchor_pts.shape[0] != 1:
        return None
    ar, ac = map(int, anchor_pts[0])
    h, w = sub.shape
    if ar == 0 and 0 < ac < w - 1:
        side = "top"
    elif ac == w - 1 and 0 < ar < h - 1:
        side = "right"
    elif ar == h - 1 and 0 < ac < w - 1:
        side = "bottom"
    elif ac == 0 and 0 < ar < h - 1:
        side = "left"
    else:
        return None

    hole = ~comp_mask
    if not hole.any():
        return None
    if hole[0].any() or hole[-1].any() or hole[:, 0].any() or hole[:, -1].any():
        return None

    hole_pts = np.argwhere(hole)
    hr0, hc0, hr1, hc1 = _bbox(hole_pts)
    rect = np.zeros_like(hole, dtype=bool)
    rect[hr0 : hr1 + 1, hc0 : hc1 + 1] = True
    if not np.array_equal(hole, rect):
        return None

    return {
        "anchor_color": anchor_color,
        "frame_color": frame_color,
        "side": side,
        "cavity_bbox": (origin_r + hr0, origin_c + hc0, origin_r + hr1, origin_c + hc1),
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    bg = _boundary_background(g)
    frames: list[dict] = []
    templates: dict[int, np.ndarray] = {}

    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        vals = sub[sub != bg]
        colors = [int(v) for v in np.unique(vals)]

        frame = _frame_info(sub, bg, r0, c0)
        if frame is not None:
            frames.append(frame)
            continue

        if len(colors) == 1:
            mask = sub != bg
            if int(mask.sum()) >= 3 and not _has_hole(mask):
                color = colors[0]
                assert color not in templates
                templates[color] = mask.copy()

    assert frames, "expected at least one frame"
    frame_colors = [frame["anchor_color"] for frame in frames]
    assert len(frame_colors) == len(set(frame_colors))
    assert set(frame_colors) == set(templates)
    return frames, templates, bg


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames, templates, bg = _parse_scene(g)
    out = g.copy()

    for frame in frames:
        color = int(frame["anchor_color"])
        cavity = tuple(int(x) for x in frame["cavity_bbox"])
        r0, c0, r1, c1 = cavity
        target_h = int(r1 - r0 + 1)
        target_w = int(c1 - c0 + 1)
        mask = np.rot90(templates[color], ROT_K_BY_SIDE[frame["side"]])
        assert mask.shape == (target_h, target_w)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch[mask] == bg)
        patch[mask] = color

    return out


def _make_connected_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    cells = [(r, c) for r in range(height) for c in range(width)]
    min_area = max(3, (height * width + 2) // 3)
    max_area = max(min_area, height * width - 1)

    for _ in range(200):
        area = rng.randint(min_area, max_area)
        start = (rng.randrange(height), rng.randrange(width))
        chosen = {start}
        frontier = {start}
        while len(chosen) < area:
            options: set[tuple[int, int]] = set()
            for r, c in frontier:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                        options.add((nr, nc))
            if not options:
                break
            pick = rng.choice(tuple(options))
            chosen.add(pick)
            frontier.add(pick)

        if len(chosen) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _has_hole(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if height == width and (
            np.array_equal(mask, np.rot90(mask, 1)) or np.array_equal(mask, np.rot90(mask, 3))
        ):
            continue
        return mask

    raise RuntimeError("failed to create asymmetric template")


def _pack_templates(
    sizes: list[tuple[int, int]], rng: random.Random
) -> tuple[list[tuple[int, int]], int, int]:
    order = list(range(len(sizes)))
    rng.shuffle(order)
    order.sort(key=lambda i: (-sizes[i][0], -sizes[i][1], i))
    max_row_width = 15
    gap = 2
    positions = [None] * len(sizes)
    cur_r = 0
    cur_c = 0
    row_h = 0
    used_w = 0
    for idx in order:
        h, w = sizes[idx]
        if cur_c > 0 and cur_c + w > max_row_width:
            cur_r += row_h + gap
            cur_c = 0
            row_h = 0
        positions[idx] = (cur_r, cur_c)
        cur_c += w + gap
        row_h = max(row_h, h)
        used_w = max(used_w, cur_c - gap)
    total_h = cur_r + row_h
    total_w = used_w
    return positions, total_h, total_w


def _anchor_position(
    side: str, top: int, left: int, outer_h: int, outer_w: int, rng: random.Random
) -> tuple[int, int]:
    if side == "top":
        return top, left + rng.randint(1, outer_w - 2)
    if side == "right":
        return top + rng.randint(1, outer_h - 2), left + outer_w - 1
    if side == "bottom":
        return top + outer_h - 1, left + rng.randint(1, outer_w - 2)
    return top + rng.randint(1, outer_h - 2), left


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, gap: int) -> bool:
    h, w = blocked.shape
    if top < 1 or left < 1 or top + height > h - 1 or left + width > w - 1:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    return not blocked[r0:r1, c0:c1].any()


def _mark(blocked: np.ndarray, top: int, left: int, height: int, width: int, gap: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    blocked[r0:r1, c0:c1] = True


def _expected_output(
    grid: np.ndarray, frames: list[dict], template_masks: dict[int, np.ndarray]
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["cavity_bbox"]
        color = frame["anchor_color"]
        mask = np.rot90(template_masks[color], ROT_K_BY_SIDE[frame["side"]])
        assert mask.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.choice(NUM_FRAME_CHOICES)
    colors = rng.sample(COLOR_POOL, num_frames)
    sides = list(SIDE_ORDER)
    for _ in range(num_frames - 4):
        sides.append(rng.choice(SIDE_ORDER))
    rng.shuffle(sides)

    specs = []
    template_masks: dict[int, np.ndarray] = {}
    for color, side in zip(colors, sides):
        thickness = rng.choice(THICKNESS_CHOICES)
        if side in ("left", "right"):
            size = rng.choice(SIDE_SQUARE_DIMS)
            cavity_h = size
            cavity_w = size
        else:
            cavity_h = rng.choice(TOP_BOTTOM_DIMS)
            cavity_w = rng.choice(TOP_BOTTOM_DIMS)
            if cavity_h * cavity_w < 4:
                cavity_h = cavity_w = 2
        mask = _make_connected_mask(cavity_h, cavity_w, rng)
        if cavity_h == cavity_w:
            mask = np.rot90(mask, rng.randrange(4))
        else:
            mask = np.rot90(mask, rng.choice((0, 2)))
        outer_h = cavity_h + 2 * thickness
        outer_w = cavity_w + 2 * thickness
        specs.append(
            {
                "anchor_color": color,
                "side": side,
                "thickness": thickness,
                "cavity_h": cavity_h,
                "cavity_w": cavity_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
        template_masks[color] = mask

    template_sizes = [tuple(template_masks[spec["anchor_color"]].shape) for spec in specs]
    template_positions, cluster_h, cluster_w = _pack_templates(template_sizes, rng)

    min_h = max(GRID_H_RANGE[0], cluster_h + 8)
    min_w = max(GRID_W_RANGE[0], cluster_w + 8)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    grid_h = rng.randint(min_h, GRID_H_RANGE[1])
    grid_w = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    corner = rng.choice(CORNERS)
    if corner in ("tl", "tr"):
        base_r = 1
    else:
        base_r = grid_h - 1 - cluster_h
    if corner in ("tl", "bl"):
        base_c = 1
    else:
        base_c = grid_w - 1 - cluster_w

    for spec, (dr, dc) in zip(specs, template_positions):
        top = base_r + dr
        left = base_c + dc
        mask = template_masks[spec["anchor_color"]]
        h, w = mask.shape
        if not _can_place(blocked, top, left, h, w, 1):
            return None
        grid[top : top + h, left : left + w][mask] = spec["anchor_color"]
        _mark(blocked, top, left, h, w, 1)

    frames = []
    place_order = list(range(num_frames))
    rng.shuffle(place_order)
    place_order.sort(key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)
    for idx in place_order:
        spec = specs[idx]
        placed = False
        for _ in range(250):
            top = rng.randint(1, grid_h - spec["outer_h"] - 1)
            left = rng.randint(1, grid_w - spec["outer_w"] - 1)
            if not _can_place(blocked, top, left, spec["outer_h"], spec["outer_w"], 1):
                continue

            thickness = spec["thickness"]
            cavity_top = top + thickness
            cavity_left = left + thickness
            grid[top : top + spec["outer_h"], left : left + spec["outer_w"]] = FRAME
            grid[
                cavity_top : cavity_top + spec["cavity_h"],
                cavity_left : cavity_left + spec["cavity_w"],
            ] = BG
            ar, ac = _anchor_position(spec["side"], top, left, spec["outer_h"], spec["outer_w"], rng)
            grid[ar, ac] = spec["anchor_color"]
            _mark(blocked, top, left, spec["outer_h"], spec["outer_w"], 1)
            frames.append(
                {
                    "anchor_color": spec["anchor_color"],
                    "side": spec["side"],
                    "cavity_bbox": (
                        cavity_top,
                        cavity_left,
                        cavity_top + spec["cavity_h"] - 1,
                        cavity_left + spec["cavity_w"] - 1,
                    ),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, frames, template_masks)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid sample")
