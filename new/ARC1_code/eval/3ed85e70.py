from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, paste, random_free_location
import random


NEIGHBORS8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]


def _touches(mask: np.ndarray, coords: np.ndarray) -> bool:
    height, width = map(int, mask.shape)
    for r, c in coords:
        for dr, dc in NEIGHBORS8:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc]:
                return True
    return False


def _component_array(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    arr[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return box, arr


def _catalog_background(grid: np.ndarray) -> tuple[int, np.ndarray]:
    comps = []
    for color in sorted(set(map(int, grid.flat)) - {0}):
        for coords in components_mask(grid == color, connectivity=4):
            comps.append((len(coords), color, coords))
    assert comps, "expected a non-black catalog background"
    comps.sort(key=lambda item: item[0], reverse=True)
    assert len(comps) == 1 or comps[0][0] > comps[1][0], "expected a unique largest non-black component"

    _area, color, coords = comps[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    assert np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
    return int(color), mask


def _templates(grid: np.ndarray, catalog_color: int, catalog_mask: np.ndarray) -> list[dict]:
    templates = []
    visible = (grid != 0) & (grid != catalog_color)
    for coords in components_mask(visible, connectivity=8):
        if not _touches(catalog_mask, coords):
            continue
        box = bbox_coords(coords)
        raw = crop_bbox(grid, box)
        sprite = np.where(raw == catalog_color, 0, raw).astype(int)
        colors = sorted(set(map(int, sprite[sprite != 0].flat)))
        assert len(colors) >= 2, "expected multi-color catalog templates"
        templates.append(
            {
                "grid": sprite,
                "visible": sprite != 0,
                "colors": colors,
                "counts": {color: int(np.sum(sprite == color)) for color in colors},
            }
        )
    assert templates, "expected at least one catalog template"
    return templates


def _color_mask(template: dict, color: int) -> tuple[tuple[int, int], np.ndarray, bool]:
    sprite = template["grid"]
    mask = sprite == color
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected color to be present in template"
    r0, c0, r1, c1 = bbox_coords(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    arr[mask[r0 : r1 + 1, c0 : c1 + 1]] = color

    height, width = map(int, sprite.shape)
    touches_border = bool(
        np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == height - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == width - 1)
        )
    )
    return (r0, c0), arr, touches_border


def _template_offsets(part: np.ndarray, template: dict) -> list[tuple[int, int]]:
    sprite = template["grid"]
    height, width = map(int, sprite.shape)
    part_height, part_width = map(int, part.shape)
    part_colors = sorted(set(map(int, part[part != 0].flat)))
    assert part_colors, "expected a non-empty workspace object"

    offsets = []

    if (
        part_height == height
        and part_width == width
        and len(part_colors) == 1
        and np.all(part == part_colors[0])
        and part_colors[0] in template["colors"]
    ):
        _offset, _mask, color_touches_border = _color_mask(template, part_colors[0])
        if color_touches_border:
            offsets.append((0, 0))

    if len(part_colors) == 1 and part_colors[0] in template["colors"]:
        color = part_colors[0]
        count_freq = Counter(template["counts"].values())
        if count_freq[template["counts"][color]] == 1:
            offset, mask_arr, _touches_border = _color_mask(template, color)
            if part.shape == mask_arr.shape and np.array_equal(part, mask_arr):
                offsets.append(offset)

    if len(part_colors) >= 2:
        for r in range(height - part_height + 1):
            for c in range(width - part_width + 1):
                sub = sprite[r : r + part_height, c : c + part_width]
                if np.array_equal(part, sub) and (
                    r == 0 or c == 0 or r + part_height == height or c + part_width == width
                ):
                    offsets.append((r, c))

    return offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partial workspace sprites by matching them to full templates shown in the non-black catalog region.

    Input structure
    - A large non-black connected region is the catalog background.
    - Multi-color template sprites touch that catalog background; all other
      foreground objects lie in the black workspace.
    - Some workspace objects are partial views of a catalog template, while
      distractor objects do not match any template part.

    Rule
    - For each workspace object, find the unique catalog template that it matches:
      either a full template bbox filled with one template border color, a
      uniquely sized single-color mask of the template, or a multi-color boundary
      window of the template.
    - Stamp the full matched template at the aligned position, treating catalog
      background cells inside the template bbox as transparent.
    - Leave workspace objects with no template match unchanged.

    Key insight
    - The colored objects in the catalog are not to be copied in order; they are
      a legend of target sprites, and each incomplete workspace object determines
      its template and alignment from the visible sub-sprite it already contains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    catalog_color, catalog_mask = _catalog_background(g)
    templates = _templates(g, catalog_color, catalog_mask)

    planned: dict[tuple[int, int], int] = {}
    workspace = (g != 0) & (~catalog_mask)
    for coords in components_mask(workspace, connectivity=8):
        if _touches(catalog_mask, coords):
            continue

        box, part = _component_array(g, coords)
        r0, c0, _r1, _c1 = box
        source_mask = np.zeros_like(g, dtype=bool)
        source_mask[coords[:, 0], coords[:, 1]] = True

        matches = []
        for template in templates:
            sprite = template["grid"]
            for off_r, off_c in _template_offsets(part, template):
                anchor_r, anchor_c = r0 - off_r, c0 - off_c
                height, width = map(int, sprite.shape)
                if not (
                    0 <= anchor_r
                    and 0 <= anchor_c
                    and anchor_r + height <= g.shape[0]
                    and anchor_c + width <= g.shape[1]
                ):
                    continue

                ok = True
                for rr, cc in np.argwhere(template["visible"]):
                    gr, gc = anchor_r + int(rr), anchor_c + int(cc)
                    value = int(sprite[rr, cc])
                    if g[gr, gc] != 0 and not source_mask[gr, gc] and g[gr, gc] != value:
                        ok = False
                        break
                if ok:
                    cells = tuple(
                        (int(rr), int(cc), int(sprite[rr, cc]))
                        for rr, cc in np.argwhere(template["visible"])
                    )
                    matches.append((anchor_r, anchor_c, cells))

        unique_matches = []
        seen = set()
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        assert len(unique_matches) <= 1, "expected each workspace object to match at most one template"

        if not unique_matches:
            continue

        anchor_r, anchor_c, cells = unique_matches[0]
        for rr, cc, value in cells:
            target = (anchor_r + rr, anchor_c + cc)
            assert planned.get(target, value) == value, "expected overlapping template stamps to agree"
            planned[target] = value

    out = g.copy()
    for (r, c), value in planned.items():
        out[r, c] = value
    return out

# --------------- generator (o5x) ---------------

# Canvas and catalog strip.
GRID_SIZE, CATALOG_COLOR, CATALOG_TEMPLATE_GAP = 30, 3, 1
CATALOG_EDGES, HORIZONTAL_CATALOG_EDGES = ("top", "bottom", "left"), ("top", "bottom")
CATALOG_THICKNESS_RANGE = (6, 8)

# Template shape, style, and palette.
TEMPLATE_COUNT_RANGE, TEMPLATE_SIZE_RANGE = (2, 3), (2, 5)
TEMPLATE_FRAME_MIN_SIZE, TEMPLATE_PATTERN_BORDER_PROB = 3, 0.55
TEMPLATE_STYLES, TEMPLATE_PALETTE = ("frame", "pattern"), (1, 2, 4, 6, 7, 8)

# Workspace object counts and spacing.
TOTAL_OBJECT_COUNT_RANGE = (11, 16)
MATCHED_OBJECT_COUNT_RANGE, DISTRACTOR_COUNT_RANGE = (2, 5), (7, 11)
DISTRACTOR_SIZE_RANGE, PLACEMENT_PADDING = (1, 3), 1

# Generation retry budget.
GENERATION_RETRY_LIMIT = 300


def _gen_template_sprite(height: int, width: int, border_color: int, fill_color: int) -> np.ndarray:
    style = random.choice(TEMPLATE_STYLES)
    sprite = np.full((height, width), int(fill_color), dtype=int)

    if style == "frame" and height >= TEMPLATE_FRAME_MIN_SIZE and width >= TEMPLATE_FRAME_MIN_SIZE:
        sprite[[0, -1], :] = border_color
        sprite[:, [0, -1]] = border_color
        return sprite

    mask = np.zeros((height, width), dtype=bool)
    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    for r, c in coords:
        rr, cc = height - 1 - r, width - 1 - c
        value = random.random() < TEMPLATE_PATTERN_BORDER_PROB
        mask[r, c] = value
        mask[rr, cc] = value

    c = random.randrange(width)
    mask[(0, height - 1), (c, width - 1 - c)] = True
    if not np.any(~mask):
        r, c = random.randrange(height), random.randrange(width)
        mask[(r, height - 1 - r), (c, width - 1 - c)] = False
    sprite[mask] = border_color
    return sprite


def _gen_pack_starts(length: int, sizes: list[int]) -> list[int]:
    base_gaps = [0] + [CATALOG_TEMPLATE_GAP] * (len(sizes) - 1) + [0]
    spare = length - sum(sizes) - sum(base_gaps)
    if spare < 0:
        raise ValueError("templates do not fit in catalog strip")
    extras = [0] * (len(sizes) + 1)
    for _ in range(spare):
        extras[random.randrange(len(extras))] += 1
    gaps = [base + extra for base, extra in zip(base_gaps, extras)]
    starts = []
    pos = gaps[0]
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size + gaps[i + 1]
    return starts


def _gen_color_touches_border(sprite: np.ndarray, color: int) -> bool:
    mask = sprite == int(color)
    return bool(np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]))


def _gen_color_mask_is_solid_rect(sprite: np.ndarray, color: int, height: int, width: int) -> bool:
    mask = sprite == int(color)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) == (height, width) and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _gen_solid_matches_template(height: int, width: int, color: int, templates: list[dict]) -> bool:
    for template in templates:
        sprite = template["sprite"]
        colors = template["colors"]
        if color not in colors:
            continue

        if sprite.shape == (height, width) and _gen_color_touches_border(sprite, color):
            return True

        counts = {col: int(np.sum(sprite == col)) for col in colors}
        if (
            list(counts.values()).count(counts[color]) == 1
            and _gen_color_mask_is_solid_rect(sprite, color, height, width)
        ):
            return True
    return False


def _gen_catalog_slice(edge: str, thickness: int) -> tuple[slice, slice]:
    if edge == "top":
        return slice(0, thickness), slice(0, GRID_SIZE)
    if edge == "bottom":
        return slice(GRID_SIZE - thickness, GRID_SIZE), slice(0, GRID_SIZE)
    return slice(0, GRID_SIZE), slice(0, thickness)


def _gen_place_catalog_templates(out: np.ndarray, edge: str, thickness: int, templates: list[dict]) -> None:
    if edge in HORIZONTAL_CATALOG_EDGES:
        starts = _gen_pack_starts(GRID_SIZE, [t["sprite"].shape[1] for t in templates])
        for template, c0 in zip(templates, starts):
            sprite = template["sprite"]
            h, _w = sprite.shape
            local_r = random.randrange(thickness - h + 1)
            r0 = local_r if edge == "top" else GRID_SIZE - thickness + local_r
            paste(out, sprite, r0, c0)
    else:
        starts = _gen_pack_starts(GRID_SIZE, [t["sprite"].shape[0] for t in templates])
        for template, r0 in zip(templates, starts):
            sprite = template["sprite"]
            _h, w = sprite.shape
            c0 = random.randrange(thickness - w + 1)
            paste(out, sprite, r0, c0)


def _gen_workspace_counts() -> tuple[int, int]:
    total = random.randint(*TOTAL_OBJECT_COUNT_RANGE)
    lo = max(MATCHED_OBJECT_COUNT_RANGE[0], total - DISTRACTOR_COUNT_RANGE[1])
    hi = min(MATCHED_OBJECT_COUNT_RANGE[1], total - DISTRACTOR_COUNT_RANGE[0])
    matched = random.randint(lo, hi)
    return matched, total - matched


def _gen_templates(edge: str, thickness: int) -> list[dict] | None:
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    palette = list(TEMPLATE_PALETTE)
    random.shuffle(palette)

    templates = []
    used_shape_border_colors: dict[tuple[int, int], set[int]] = {}
    for i in range(template_count):
        height_cap = thickness if edge in HORIZONTAL_CATALOG_EDGES else TEMPLATE_SIZE_RANGE[1]
        width_cap = thickness if edge == "left" else TEMPLATE_SIZE_RANGE[1]
        height = random.randint(TEMPLATE_SIZE_RANGE[0], min(TEMPLATE_SIZE_RANGE[1], height_cap))
        width = random.randint(TEMPLATE_SIZE_RANGE[0], min(TEMPLATE_SIZE_RANGE[1], width_cap))
        border_color, fill_color = palette[2 * i], palette[2 * i + 1]
        if border_color in used_shape_border_colors.setdefault((height, width), set()):
            return None
        used_shape_border_colors[(height, width)].add(border_color)
        templates.append(
            {
                "sprite": _gen_template_sprite(height, width, border_color, fill_color),
                "border_color": border_color,
                "colors": [border_color, fill_color],
            }
        )
    return templates


def _gen_place_workspace_patch(out: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    occupied_patch = np.ones(patch.shape, dtype=int)
    try:
        r0, c0 = random_free_location(occupied, occupied_patch, transparent=0, padding=PLACEMENT_PADDING)
    except ValueError:
        return None
    paste(out, patch, r0, c0)
    paste(occupied, occupied_patch, r0, c0)
    return r0, c0


def _gen_place_matched_objects(
    out: np.ndarray, occupied: np.ndarray, templates: list[dict], matched_count: int
) -> list[tuple[int, int, tuple[int, int], int]] | None:
    stamp_records = []
    for _ in range(matched_count):
        template = random.choice(templates)
        sprite = template["sprite"]
        loc = _gen_place_workspace_patch(out, occupied, sprite)
        if loc is None:
            return None
        r0, c0 = loc
        stamp_records.append((r0, c0, sprite.shape, template["border_color"]))
    return stamp_records


def _gen_distractor_options(catalog_color: int, templates: list[dict]) -> list[tuple[int, int, int]]:
    distractor_colors = sorted({catalog_color, *(color for t in templates for color in t["colors"])})
    return [
        (h, w, color)
        for h in range(DISTRACTOR_SIZE_RANGE[0], DISTRACTOR_SIZE_RANGE[1] + 1)
        for w in range(DISTRACTOR_SIZE_RANGE[0], DISTRACTOR_SIZE_RANGE[1] + 1)
        for color in distractor_colors
        if not _gen_solid_matches_template(h, w, color, templates)
    ]


def _gen_place_distractors(
    out: np.ndarray, occupied: np.ndarray, distractors: list[tuple[int, int, int]], count: int
) -> bool:
    for _ in range(count):
        h, w, color = random.choice(distractors)
        patch = np.full((h, w), color, dtype=int)
        if _gen_place_workspace_patch(out, occupied, patch) is None:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(CATALOG_EDGES)
    thickness = random.randint(*CATALOG_THICKNESS_RANGE)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    rs, cs = _gen_catalog_slice(edge, thickness)
    out[rs, cs] = CATALOG_COLOR

    templates = _gen_templates(edge, thickness)
    if templates is None:
        return None
    random.shuffle(templates)
    _gen_place_catalog_templates(out, edge, thickness, templates)

    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    occupied[rs, cs] = 1
    matched_count, distractor_count = _gen_workspace_counts()

    stamp_records = _gen_place_matched_objects(out, occupied, templates, matched_count)
    if stamp_records is None:
        return None

    safe_distractors = _gen_distractor_options(CATALOG_COLOR, templates)
    if not safe_distractors:
        return None
    if not _gen_place_distractors(out, occupied, safe_distractors, distractor_count):
        return None

    inp = out.copy()
    for r0, c0, (height, width), color in stamp_records:
        inp[r0 : r0 + height, c0 : c0 + width] = color
    try:
        return inp if np.array_equal(main(inp), out) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 3ed85e70")
