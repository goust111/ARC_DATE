from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: set[int] | None = None,
    allowed: np.ndarray | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            if allowed is not None and not allowed[r, c]:
                continue
            color = int(g[r, c])
            if color == bg or (colors is not None and color not in colors):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if allowed is not None and not allowed[nr, nc]:
                            continue
                        if int(g[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox(pts)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts),
                }
            )

    return comps


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _background_components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[list[tuple[int, int]], bool]] = []

    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append((pts, touches_border))

    return out


def _hole_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    holes: list[tuple[int, int, int, int]] = []
    for pts, touches_border in _background_components(mask):
        if touches_border:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if len(pts) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        holes.append((r0, c0, r1, c1))
    holes.sort()
    return holes


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in _find_components(grid, bg=bg):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        sub = np.zeros((h, w), dtype=bool)
        for r, c in comp["coords"]:
            sub[r - r0, c - c0] = True
        if np.array_equal(sub, expected):
            candidates.append((int(comp["area"]), color, (r0, c0, r1, c1)))

    assert candidates, "expected a rectangular frame"
    candidates.sort()
    largest_area = candidates[-1][0]
    largest = [cand for cand in candidates if cand[0] == largest_area]
    assert len(largest) == 1, "frame must be structurally unique"
    _area, color, bbox = largest[0]
    return color, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the chosen framed template and fill its enlarged holes with size-matched sprites.

    Input structure:
    - The background is the most common color.
    - Above a lower grey rectangular frame sit several monochrome sprites and several monochrome template masks.
    - Each template is a small binary rectangle of one color with one or more enclosed rectangular background holes.
    - Inside the frame are a few seed pixels of one template color; the frame interior is an integer block scaling of that template's bbox.

    Rule:
    - Crop to the rectangular frame and read the seed color inside it.
    - Find the outside component of that color whose hole-bearing mask can be block-scaled to the frame interior while covering every seed pixel.
    - Paint the full scaled template into the frame interior, then find each enclosed rectangular hole of the scaled mask.
    - Match each hole's height and width to the unique outside sprite with the same bbox size and paste that sprite into the hole aligned to the hole's top-left corner.

    Key insight:
    The seed color identifies which outside template to enlarge, and the enlarged template's holes then define the exact sprite slots that must be filled by bbox-size matching.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    out = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    interior = out[1:-1, 1:-1]
    ih, iw = interior.shape

    seed_colors = set(map(int, np.unique(interior))) - {bg, frame_color}
    assert len(seed_colors) == 1, "frame interior must contain one seed color"
    seed_color = seed_colors.pop()
    seed_mask = interior == seed_color
    assert np.any(seed_mask), "frame must contain at least one seed"

    outside = np.ones_like(g, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    outside_comps = _find_components(g, bg=bg, allowed=outside)

    template_mask = None
    scaled_mask = None
    for comp in outside_comps:
        if int(comp["color"]) != seed_color:
            continue
        mask = _mask_from_component(comp)
        if not _hole_bboxes(mask):
            continue
        th, tw = mask.shape
        if ih % th != 0 or iw % tw != 0:
            continue
        scale_h = ih // th
        scale_w = iw // tw
        if scale_h != scale_w:
            continue
        scaled = np.kron(mask, np.ones((scale_h, scale_w), dtype=bool))
        if np.all(~seed_mask | scaled):
            assert template_mask is None, "seed color must select one template"
            template_mask = mask
            scaled_mask = scaled

    assert template_mask is not None and scaled_mask is not None, "could not identify the template"

    sprites: list[dict] = []
    for comp in outside_comps:
        mask = _mask_from_component(comp)
        if _hole_bboxes(mask):
            continue
        h, w = mask.shape
        sprites.append(
            {
                "color": int(comp["color"]),
                "mask": mask,
                "shape": (h, w),
            }
        )

    interior[:, :] = bg
    interior[scaled_mask] = seed_color

    used_sprite_ids: set[int] = set()
    for hr0, hc0, hr1, hc1 in _hole_bboxes(scaled_mask):
        hole_shape = (hr1 - hr0 + 1, hc1 - hc0 + 1)
        matches = [i for i, sprite in enumerate(sprites) if sprite["shape"] == hole_shape and i not in used_sprite_ids]
        assert len(matches) == 1, "each hole must have exactly one bbox-matching sprite"
        idx = matches[0]
        used_sprite_ids.add(idx)
        sprite = sprites[idx]
        region = interior[hr0 : hr1 + 1, hc0 : hc1 + 1]
        region[sprite["mask"]] = sprite["color"]

    return out


BG = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
TARGET_BRANCHES = ("scale2_one", "scale2_two", "scale3_one", "scale4_one")
SCALE2_HOLE_DIMS = [(1, 1), (1, 2), (2, 1), (2, 2)]


def _row_positions(widths: list[int], total_w: int, rng: random.Random, margin: int = 1, gap: int = 1) -> list[int]:
    if not widths:
        return []
    required = sum(widths) + gap * (len(widths) - 1)
    slack = total_w - 2 * margin - required
    assert slack >= 0
    extras = [0] * (len(widths) + 1)
    for _ in range(slack):
        extras[rng.randrange(len(extras))] += 1
    positions: list[int] = []
    cur = margin + extras[0]
    for i, width in enumerate(widths):
        positions.append(cur)
        cur += width
        if i + 1 < len(widths):
            cur += gap + extras[i + 1]
    return positions


def _random_sprite_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)

    min_area = min(h * w, max(3, h + w - 1))
    for _ in range(300):
        target = rng.randint(min_area, h * w)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            cr, cc = cell
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(start)
        while len(cells) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(cell)

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _hole_bboxes(mask):
            continue
        return mask

    return np.ones((h, w), dtype=bool)


def _make_template_mask(
    th: int,
    tw: int,
    hole_dims: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(400):
        mask = np.ones((th, tw), dtype=bool)
        taken = np.zeros((th, tw), dtype=bool)
        dims = hole_dims[:]
        rng.shuffle(dims)
        ok = True
        for hh, ww in dims:
            candidates = [(r, c) for r in range(1, th - hh) for c in range(1, tw - ww)]
            rng.shuffle(candidates)
            placed = False
            for r, c in candidates:
                if np.any(taken[r : r + hh, c : c + ww]):
                    continue
                mask[r : r + hh, c : c + ww] = False
                taken[r : r + hh, c : c + ww] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        holes = _hole_bboxes(mask)
        shapes = sorted((r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in holes)
        if len(holes) == len(hole_dims) and shapes == sorted(hole_dims):
            return mask, holes
    return None


def _generate_target_template(
    rng: random.Random,
) -> tuple[np.ndarray, int, list[tuple[int, int, int, int]], str] | None:
    branch = rng.choice(TARGET_BRANCHES)
    for _ in range(200):
        if branch == "scale4_one":
            th = 3
            tw = rng.choice([3, 4, 5])
            scale = 4
            hole_dims = [(1, 1)]
        elif branch == "scale3_one":
            th = rng.choice([3, 4])
            tw = rng.choice([3, 4, 5])
            scale = 3
            hole_dims = [(1, 1)]
        elif branch == "scale2_two":
            th = rng.choice([4, 5])
            tw = rng.choice([4, 5])
            scale = 2
            hole_dims = rng.sample(SCALE2_HOLE_DIMS, 2)
        else:
            th = rng.choice([3, 4, 5])
            tw = rng.choice([3, 4, 5])
            scale = 2
            hole_dims = [rng.choice(SCALE2_HOLE_DIMS)]

        made = _make_template_mask(th, tw, hole_dims, rng)
        if made is None:
            continue
        mask, holes = made
        return mask, scale, holes, branch
    return None


def _generate_distractor_template(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        th = rng.choice([3, 4, 5])
        tw = rng.choice([3, 4, 5])
        if min(th, tw) < 3:
            continue
        choices = [(1, 1), (1, 2), (2, 1)]
        if th >= 4 and tw >= 4:
            choices.append((2, 2))
        hole_count = 2 if min(th, tw) >= 4 and rng.random() < 0.4 else 1
        hole_dims = rng.sample(choices, hole_count)
        made = _make_template_mask(th, tw, hole_dims, rng)
        if made is not None:
            return made[0]
    return None


def _color_patch(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int = BG) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    region[patch != bg] = patch[patch != bg]


def _build_frame_input(
    scaled_mask: np.ndarray,
    seed_color: int,
    rng: random.Random,
) -> np.ndarray:
    ih, iw = scaled_mask.shape
    patch = np.full((ih + 2, iw + 2), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    coords = np.argwhere(scaled_mask)
    seed_max = min(8, max(2, len(coords) // 3))
    seed_count = rng.randint(2, seed_max)
    pick = rng.sample(range(len(coords)), seed_count)
    interior = patch[1:-1, 1:-1]
    for idx in pick:
        r, c = coords[idx]
        interior[int(r), int(c)] = seed_color
    return patch


def _build_expected_output(
    scaled_mask: np.ndarray,
    template_color: int,
    hole_boxes: list[tuple[int, int, int, int]],
    sprites: list[dict],
) -> np.ndarray:
    ih, iw = scaled_mask.shape
    out = np.full((ih + 2, iw + 2), BG, dtype=int)
    out[0, :] = FRAME_COLOR
    out[-1, :] = FRAME_COLOR
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR

    interior = out[1:-1, 1:-1]
    interior[scaled_mask] = template_color

    sprite_by_shape = {tuple(sprite["shape"]): sprite for sprite in sprites}
    for hr0, hc0, hr1, hc1 in hole_boxes:
        hole_shape = (hr1 - hr0 + 1, hc1 - hc0 + 1)
        sprite = sprite_by_shape[hole_shape]
        region = interior[hr0 : hr1 + 1, hc0 : hc1 + 1]
        region[sprite["mask"]] = sprite["color"]

    return out


def _scaled_hole_boxes(
    holes: list[tuple[int, int, int, int]],
    scale: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for r0, c0, r1, c1 in holes:
        out.append((r0 * scale, c0 * scale, (r1 + 1) * scale - 1, (c1 + 1) * scale - 1))
    out.sort()
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    target = _generate_target_template(rng)
    if target is None:
        return None
    target_mask, scale, target_holes, _branch = target

    scaled_mask = np.kron(target_mask, np.ones((scale, scale), dtype=bool))
    scaled_holes = _scaled_hole_boxes(target_holes, scale)
    slot_shapes = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in scaled_holes]
    if len(slot_shapes) != len(set(slot_shapes)):
        return None

    match_sprites: list[dict] = []
    for h, w in slot_shapes:
        match_sprites.append({"shape": (h, w), "mask": _random_sprite_mask(h, w, rng)})

    distractor_shapes = [(h, w) for h in range(1, 6) for w in range(1, 6) if (h, w) not in set(slot_shapes)]
    rng.shuffle(distractor_shapes)
    distractor_count = rng.randint(0, min(2, len(distractor_shapes)))
    distractor_sprites = [
        {"shape": shape, "mask": _random_sprite_mask(shape[0], shape[1], rng)}
        for shape in distractor_shapes[:distractor_count]
    ]

    template_total = rng.randint(2, 4)
    distractor_templates: list[np.ndarray] = []
    while len(distractor_templates) < template_total - 1:
        mask = _generate_distractor_template(rng)
        if mask is not None:
            distractor_templates.append(mask)

    object_count = template_total + len(match_sprites) + len(distractor_sprites)
    if object_count > len(COLOR_POOL):
        return None
    colors = COLOR_POOL[:]
    rng.shuffle(colors)

    template_colors = colors[:template_total]
    sprite_colors = colors[template_total : template_total + len(match_sprites) + len(distractor_sprites)]

    target_template_color = template_colors[0]
    templates = [{"mask": target_mask, "color": target_template_color}]
    for mask, color in zip(distractor_templates, template_colors[1:], strict=False):
        templates.append({"mask": mask, "color": color})
    rng.shuffle(templates)

    all_sprites = match_sprites + distractor_sprites
    for sprite, color in zip(all_sprites, sprite_colors, strict=False):
        sprite["color"] = int(color)
    rng.shuffle(all_sprites)

    frame_patch = _build_frame_input(scaled_mask, target_template_color, rng)
    expected = _build_expected_output(
        scaled_mask,
        target_template_color,
        scaled_holes,
        match_sprites,
    )

    max_sprite_h = max(sprite["shape"][0] for sprite in all_sprites)
    max_template_h = max(int(item["mask"].shape[0]) for item in templates)
    frame_h, frame_w = frame_patch.shape
    grid_h = 1 + max_sprite_h + 1 + max_template_h + 2 + frame_h + 1
    if grid_h > 30:
        return None

    sprite_widths = [sprite["shape"][1] for sprite in all_sprites]
    template_widths = [int(item["mask"].shape[1]) for item in templates]
    sprite_row_need = sum(sprite_widths) + max(0, len(sprite_widths) - 1)
    template_row_need = sum(template_widths) + max(0, len(template_widths) - 1)
    grid_w = max(frame_w + 2, sprite_row_need + 2, template_row_need + 2)
    if grid_w > 30:
        return None
    grid_w += rng.randint(0, min(30 - grid_w, 3))

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    sprite_top = 1
    template_top = sprite_top + max_sprite_h + 1
    frame_top = template_top + max_template_h + 2

    sprite_lefts = _row_positions(sprite_widths, grid_w, rng)
    for sprite, left in zip(all_sprites, sprite_lefts, strict=False):
        h, _w = sprite["shape"]
        top = sprite_top + rng.randrange(max_sprite_h - h + 1)
        patch = _color_patch(sprite["mask"], int(sprite["color"]))
        _paste(grid, patch, top, left)

    template_lefts = _row_positions(template_widths, grid_w, rng)
    for item, left in zip(templates, template_lefts, strict=False):
        mask = item["mask"]
        top = template_top + rng.randrange(max_template_h - mask.shape[0] + 1)
        patch = _color_patch(mask, int(item["color"]))
        _paste(grid, patch, top, left)

    frame_left = rng.randint(1, grid_w - frame_w - 1) if grid_w - frame_w >= 2 else 0
    _paste(grid, frame_patch, frame_top, frame_left)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
