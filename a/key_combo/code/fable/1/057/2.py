from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BG = 0
TEMPLATE_INK = 5
SEPARATOR_COLOR = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)

MAX_GENERATION_ATTEMPTS = 400
MAX_TEMPLATE_PLACEMENT_ATTEMPTS = 120


def get_bg(grid: np.ndarray) -> int:
    """Adapted from arc_utils.get_bg: return the most common grid value."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Adapted from arc_utils.bbox_coords using the shared (r, c) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Small self-contained 4/8-connected component extractor."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbor_dirs = DIRS4
    else:
        neighbor_dirs = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbor_dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    """Minimal arc_utils-style component API for this task."""
    colors = sorted(set(map(int, grid.ravel())) - {int(bg)})
    out: list[dict] = []
    if monochromatic:
        for color in colors:
            for coords in components_mask(grid == color, connectivity=connectivity):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    for coords in components_mask(grid != bg, connectivity=connectivity):
        out.append(
            {
                "color": None,
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(components_mask(mask.astype(bool), connectivity=4)) == 1


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    return np.array(out, copy=True)


def _build_template_mask_library() -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    template_masks: dict[int, list[np.ndarray]] = {}
    connected_templates: dict[int, list[np.ndarray]] = {}
    object_masks: dict[int, list[np.ndarray]] = {}

    for bits in range(1, 1 << 9):
        flat = [(bits >> idx) & 1 for idx in range(9)]
        mask = np.array(flat, dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        connected = _mask_connected(mask)

        if 2 <= area <= 9 and mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any():
            template_masks.setdefault(area, []).append(mask.copy())
            if connected:
                connected_templates.setdefault(area, []).append(mask.copy())

        if 2 <= area <= 6 and connected:
            coords = np.argwhere(mask)
            r0, c0, r1, c1 = bbox_coords(coords)
            cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
            object_masks.setdefault(area, []).append(np.array(cropped, copy=True))

    return template_masks, connected_templates, object_masks


TEMPLATE_MASKS_BY_COUNT, CONNECTED_TEMPLATE_MASKS_BY_COUNT, OBJECT_MASKS_BY_AREA = _build_template_mask_library()


def _separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])]
    assert len(rows) == 1, "expected exactly one full non-background separator row"
    return int(rows[0])


def _extract_templates(top: np.ndarray, bg: int) -> dict[int, dict]:
    colors = sorted(set(map(int, top.ravel())) - {int(bg)})
    assert len(colors) == 1, "top legend must use exactly one template ink color"
    ink = colors[0]

    found: list[tuple[int, int, np.ndarray]] = []
    h, w = top.shape
    for r in range(h - 2):
        for c in range(w - 2):
            mask = top[r : r + 3, c : c + 3] == ink
            if not bool(mask.any()):
                continue
            if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
                continue

            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(h, r + 4)
            cc1 = min(w, c + 4)
            local_total = int((top[rr0:rr1, cc0:cc1] == ink).sum())
            if local_total != int(mask.sum()):
                continue
            found.append((r, c, mask.copy()))

    assert found, "expected at least one legend template"
    by_count: dict[int, dict] = {}
    used_cells = np.zeros(top.shape, dtype=bool)
    for r, c, mask in found:
        count = int(mask.sum())
        assert count not in by_count, "legend template pixel counts must be unique"
        by_count[count] = {"mask": mask, "pos": (int(r), int(c))}
        used_cells[r : r + 3, c : c + 3] |= mask

    assert np.array_equal(used_cells, top == ink), "legend templates must explain all top-panel ink"
    return by_count


def _color_object_counts(bottom: np.ndarray, bg: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for comp in find_components(bottom, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        counts[color] = counts.get(color, 0) + 1
    assert counts, "expected colored objects in the lower panel"
    assert len(set(counts.values())) == len(counts), "per-color object counts must be unique"
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each legend template by the lower-panel color whose object count equals that template's pixel count.

    Input structure:
    - One full separator row splits the canvas into an upper legend panel and a lower object panel.
    - The upper panel contains several scattered grey 3x3 template masks with distinct grey-pixel counts.
    - The lower panel has a black background and several foreground colors, each appearing as multiple 4-connected solid objects.

    Rule:
    - Count the number of 4-connected objects for each non-background color in the lower panel.
    - In the upper legend, find the 3x3 grey template whose number of grey pixels equals that color's object count, then recolor that template with the color.
    - Return the matched 3x3 templates as a single vertical strip, ordered from larger object count to smaller, with one background row between adjacent tiles.

    Key insight:
    - The legend never draws an explicit key: a template's own grey-pixel count is the hidden key that identifies which lower-panel color should claim it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep = _separator_row(g, bg)
    top = g[:sep]
    bottom = g[sep + 1 :]

    templates = _extract_templates(top, bg)
    color_counts = _color_object_counts(bottom, bg)
    assert set(templates) == set(color_counts.values()), "template pixel counts must match the lower object counts exactly"

    ordered = sorted(color_counts.items(), key=lambda item: (-item[1], item[0]))
    out = np.full((4 * len(ordered) - 1, 3), bg, dtype=int)
    for idx, (color, count) in enumerate(ordered):
        mask = templates[count]["mask"]
        rr = idx * 4
        out[rr : rr + 3, :] = np.where(mask, int(color), bg)
    return out


def _choose_counts() -> list[int]:
    while True:
        k = random.choice((3, 4))
        counts = sorted(random.sample([3, 4, 5, 6, 7, 8], k))
        if sum(counts) <= 20:
            return counts


def _choose_template_masks(counts: list[int], decoy_mode: str) -> dict[int, np.ndarray] | None:
    masks: dict[int, np.ndarray] = {}
    connected_counts = [count for count in counts if count in CONNECTED_TEMPLATE_MASKS_BY_COUNT]
    if decoy_mode != "none" and not connected_counts:
        return None

    if decoy_mode == "matched":
        special = random.choice(connected_counts)
    elif decoy_mode == "mismatched":
        special = random.choice(connected_counts)
    else:
        special = None

    for count in counts:
        pool = CONNECTED_TEMPLATE_MASKS_BY_COUNT[count] if count == special else TEMPLATE_MASKS_BY_COUNT[count]
        masks[count] = _transform_mask(random.choice(pool))
    return masks


def _place_templates(width: int, upper_h: int, template_masks: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    entries = list(template_masks.items())
    random.shuffle(entries)

    for _ in range(MAX_TEMPLATE_PLACEMENT_ATTEMPTS):
        grid = np.full((upper_h, width), BG, dtype=int)
        reserved = np.zeros((upper_h, width), dtype=bool)
        positions: dict[int, tuple[int, int]] = {}
        ok = True

        for count, mask in entries:
            options = []
            for r in range(1, upper_h - 2):
                for c in range(1, width - 2):
                    rr0 = max(0, r - 1)
                    cc0 = max(0, c - 1)
                    rr1 = min(upper_h, r + 4)
                    cc1 = min(width, c + 4)
                    if reserved[rr0:rr1, cc0:cc1].any():
                        continue
                    options.append((r, c))
            if not options:
                ok = False
                break
            r, c = random.choice(options)
            grid[r : r + 3, c : c + 3] = np.where(mask, TEMPLATE_INK, grid[r : r + 3, c : c + 3])
            reserved[max(0, r - 1) : min(upper_h, r + 4), max(0, c - 1) : min(width, c + 4)] = True
            positions[count] = (r, c)

        if not ok:
            continue
        if len({r for r, _c in positions.values()}) < 2 or len({c for _r, c in positions.values()}) < 2:
            continue

        spatial_order = [count for count, _pos in sorted(positions.items(), key=lambda item: item[1])]
        if spatial_order == sorted(spatial_order) or spatial_order == sorted(spatial_order, reverse=True):
            continue
        return grid, positions
    return None


def _choose_object_shapes(
    counts: list[int],
    template_masks: dict[int, np.ndarray],
    decoy_mode: str,
) -> tuple[dict[int, np.ndarray], dict[int, int], bool] | None:
    areas = [2, 3, 4, 5, 6]
    color_shapes: dict[int, np.ndarray] = {}
    count_to_area: dict[int, int] = {}
    decoy_applied = False

    matched_count = None
    mismatched_src = None
    mismatched_dst = None
    connected_counts = [count for count in counts if _mask_connected(template_masks[count])]

    if decoy_mode == "matched" and connected_counts:
        matched_count = random.choice(connected_counts)
    elif decoy_mode == "mismatched" and connected_counts:
        mismatched_dst = random.choice(connected_counts)
        sources = [count for count in counts if count != mismatched_dst]
        if not sources:
            return None
        mismatched_src = random.choice(sources)

    for count in counts:
        if matched_count == count:
            mask = np.array(template_masks[count], copy=True)
            color_shapes[count] = mask
            count_to_area[count] = int(mask.sum())
            decoy_applied = True
            continue
        if mismatched_src == count and mismatched_dst is not None:
            mask = np.array(template_masks[mismatched_dst], copy=True)
            color_shapes[count] = mask
            count_to_area[count] = int(mask.sum())
            decoy_applied = True
            continue

        area = random.choice(areas)
        mask = _transform_mask(random.choice(OBJECT_MASKS_BY_AREA[area]))
        color_shapes[count] = mask
        count_to_area[count] = int(mask.sum())

    if decoy_mode != "none" and not decoy_applied:
        return None
    return color_shapes, count_to_area, decoy_applied


def _place_lower_objects(
    width: int,
    lower_h: int,
    counts: list[int],
    colors: list[int],
    color_shapes: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, list[tuple[int, int, np.ndarray]]]] | None:
    grid = np.full((lower_h, width), BG, dtype=int)
    reserved = np.zeros((lower_h, width), dtype=bool)
    placements: dict[int, list[tuple[int, int, np.ndarray]]] = {color: [] for color in colors}

    instances: list[tuple[int, np.ndarray]] = []
    for color, count in zip(colors, counts):
        mask = color_shapes[count]
        for _ in range(count):
            instances.append((color, mask))

    random.shuffle(instances)
    instances.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    for color, mask in instances:
        h, w = mask.shape
        options = []
        for r in range(1, lower_h - h):
            for c in range(1, width - w):
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(lower_h, r + h + 1)
                cc1 = min(width, c + w + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                options.append((r, c))
        if not options:
            return None
        r, c = random.choice(options)
        grid[r : r + h, c : c + w][mask] = color
        reserved[max(0, r - 1) : min(lower_h, r + h + 1), max(0, c - 1) : min(width, c + w + 1)] = True
        placements[color].append((r, c, np.array(mask, copy=True)))

    return grid, placements


def _expected_output(counts: list[int], colors: list[int], template_masks: dict[int, np.ndarray]) -> np.ndarray:
    count_to_color = {count: color for count, color in zip(counts, colors)}
    ordered_counts = sorted(counts, reverse=True)
    out = np.full((4 * len(ordered_counts) - 1, 3), BG, dtype=int)
    for idx, count in enumerate(ordered_counts):
        rr = idx * 4
        mask = template_masks[count]
        out[rr : rr + 3, :] = np.where(mask, count_to_color[count], BG)
    return out


def _validate_sample(
    inp: np.ndarray,
    expected: np.ndarray,
    counts: list[int],
    colors: list[int],
    template_masks: dict[int, np.ndarray],
    template_positions: dict[int, tuple[int, int]],
    lower_placements: dict[int, list[tuple[int, int, np.ndarray]]],
    count_to_area: dict[int, int],
    decoy_mode: str,
    decoy_applied: bool,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    bg = get_bg(inp)
    sep = _separator_row(inp, bg)
    if int(inp[sep, 0]) == bg:
        return False

    top = inp[:sep]
    bottom = inp[sep + 1 :]
    extracted = _extract_templates(top, bg)
    if set(extracted) != set(counts):
        return False
    for count in counts:
        if not np.array_equal(extracted[count]["mask"], template_masks[count]):
            return False

    object_counts = _color_object_counts(bottom, bg)
    if set(object_counts) != set(colors):
        return False
    if sorted(object_counts.values()) != sorted(counts):
        return False

    if len({r for r, _c in template_positions.values()}) < 2 or len({c for _r, c in template_positions.values()}) < 2:
        return False
    spatial_order = [count for count, _pos in sorted(template_positions.items(), key=lambda item: item[1])]
    if spatial_order == sorted(spatial_order) or spatial_order == sorted(spatial_order, reverse=True):
        return False

    total_pixels = {
        color: len(lower_placements[color]) * int(lower_placements[color][0][2].sum())
        for color in colors
    }
    color_counts = {color: len(lower_placements[color]) for color in colors}
    count_rank = [color for color, _count in sorted(color_counts.items(), key=lambda item: (-item[1], item[0]))]
    area_rank = [color for color, _area in sorted(total_pixels.items(), key=lambda item: (-item[1], item[0]))]
    if count_rank == area_rank:
        return False

    if any(count_to_area[count] == 1 for count in counts):
        return False

    if decoy_mode != "none" and not decoy_applied:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    counts = _choose_counts()
    colors = random.sample(list(OBJECT_COLORS), len(counts))
    decoy_mode = random.choices(["none", "matched", "mismatched"], weights=[3, 2, 2], k=1)[0]

    template_masks = _choose_template_masks(counts, decoy_mode)
    if template_masks is None:
        return None

    shape_bundle = _choose_object_shapes(counts, template_masks, decoy_mode)
    if shape_bundle is None:
        return None
    color_shapes, count_to_area, decoy_applied = shape_bundle

    width = random.randint(24, 30)
    upper_h = random.randint(7, 10)
    lower_h = random.randint(12, 30 - upper_h - 1)

    placed_templates = _place_templates(width, upper_h, template_masks)
    if placed_templates is None:
        return None
    upper, template_positions = placed_templates

    lower_bundle = _place_lower_objects(width, lower_h, counts, colors, color_shapes)
    if lower_bundle is None:
        return None
    lower, lower_placements = lower_bundle

    count_rank = sorted(
        ((count, count_to_area[count] * count) for count in counts),
        key=lambda item: (-item[1], item[0]),
    )
    if [count for count, _total in count_rank] == sorted(counts, reverse=True):
        return None

    inp = np.full((upper_h + 1 + lower_h, width), BG, dtype=int)
    inp[:upper_h] = upper
    inp[upper_h, :] = SEPARATOR_COLOR
    inp[upper_h + 1 :] = lower

    expected = _expected_output(counts, colors, template_masks)
    if not _validate_sample(
        inp,
        expected,
        counts,
        colors,
        template_masks,
        template_positions,
        lower_placements,
        count_to_area,
        decoy_mode,
        decoy_applied,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for the hidden-key legend puzzle")
