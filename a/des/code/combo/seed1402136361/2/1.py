from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5

# --------------- solver ---------------

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key_from_coords(
    coords: list[tuple[int, int]] | set[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    key = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return key, max(rows) - r0 + 1, max(cols) - c0 + 1


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    return _shape_key_from_coords(coords)


def _mask_from_key(
    key: tuple[tuple[int, int], ...], height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in key:
        mask[r, c] = True
    return mask


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _extract_legend(
    grid: np.ndarray, bg: int
) -> tuple[int, dict[tuple[tuple[int, int], ...], dict[str, object]]]:
    colored = (grid != bg) & (grid != GREY)
    comps = _components(colored)
    assert comps, "expected at least one colored legend template"

    legend_top = min(r for comp in comps for r, _ in comp)
    entries = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each legend template must be monochrome"
        key, height, width = _shape_key_from_coords(comp)
        entries.append(
            {
                "left": min(c for _, c in comp),
                "shape": key,
                "height": height,
                "width": width,
                "color": colors.pop(),
            }
        )

    entries.sort(key=lambda entry: int(entry["left"]))
    mapping: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    for index, entry in enumerate(entries, start=1):
        key = entry["shape"]
        assert key not in mapping, "legend shapes must be unique"
        mapping[key] = {
            "index": index,
            "color": int(entry["color"]),
            "mask": _mask_from_key(
                key,
                int(entry["height"]),
                int(entry["width"]),
            ),
        }
    return legend_top, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey objects by matching them to bottom legend templates.

    Odd-indexed templates recolor in place. Even-indexed templates move to the
    vertically mirrored bounding-box position inside the non-legend area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    legend_top, legend = _extract_legend(g, bg)
    work = g[:legend_top]
    out = np.where(work == GREY, bg, work)

    for comp in _components(work == GREY):
        key, height, _width = _shape_key_from_coords(comp)
        assert key in legend, "every grey object must match a legend template"
        spec = legend[key]
        color = int(spec["color"])
        index = int(spec["index"])

        if index % 2 == 1:
            for r, c in comp:
                out[r, c] = color
            continue

        r0, _c0, _r1, _c1 = _bbox(comp)
        dst_r0 = work.shape[0] - height - r0
        assert 0 <= dst_r0 <= work.shape[0] - height
        for r, c in comp:
            out[dst_r0 + (r - r0), c] = color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
TEMPLATE_RETRIES = 200
PLACEMENT_RETRIES = 400

TEMPLATE_COUNT_RANGE = (2, 5)
EXTRA_OBJECTS_RANGE = (0, 4)
MAIN_H_RANGE = (10, 18)
GRID_W_RANGE = (18, 28)
TEMPLATE_H_RANGE = (2, 5)
TEMPLATE_W_RANGE = (2, 6)
TEMPLATE_AREA_RANGE = (3, 8)
LEGEND_GAP_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (0, 2)
BOTTOM_PAD_RANGE = (0, 2)
BOX_PAD = 1


def _random_connected_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        raw_h = rng.randint(*TEMPLATE_H_RANGE)
        raw_w = rng.randint(*TEMPLATE_W_RANGE)
        max_area = min(TEMPLATE_AREA_RANGE[1], raw_h * raw_w)
        if max_area < TEMPLATE_AREA_RANGE[0]:
            continue

        target_area = rng.randint(TEMPLATE_AREA_RANGE[0], max_area)
        cells = {(rng.randrange(raw_h), rng.randrange(raw_w))}
        while len(cells) < target_area:
            frontier = []
            seen = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            weights = [
                1 + sum((fr + dr, fc + dc) in cells for dr, dc in DIRS4)
                for fr, fc in frontier
            ]
            cells.add(rng.choices(frontier, weights=weights, k=1)[0])

        key, height, width = _shape_key_from_coords(cells)
        area = len(key)
        if area < TEMPLATE_AREA_RANGE[0] or height < 2 or width < 2:
            continue
        if area == height * width and rng.random() < 0.8:
            continue
        return _mask_from_key(key, height, width)
    return None


def _sample_templates(
    rng: random.Random, count: int
) -> list[dict[str, object]] | None:
    templates: list[dict[str, object]] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    while len(templates) < count:
        mask = _random_connected_mask(rng)
        if mask is None:
            return None
        key, height, width = _shape_key_from_mask(mask)
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        templates.append(
            {
                "mask": mask,
                "shape": key,
                "height": height,
                "width": width,
            }
        )
    rng.shuffle(templates)
    return templates


def _balanced_object_ids(rng: random.Random, template_count: int) -> list[int]:
    object_ids = list(range(template_count))
    target = template_count + rng.randint(*EXTRA_OBJECTS_RANGE)
    odd_ids = [idx for idx in range(template_count) if (idx + 1) % 2 == 1]
    even_ids = [idx for idx in range(template_count) if (idx + 1) % 2 == 0]
    odd_count = len(odd_ids)
    even_count = len(even_ids)

    while len(object_ids) < target:
        if odd_count < even_count:
            chosen = rng.choice(odd_ids)
            odd_count += 1
        elif even_count < odd_count:
            chosen = rng.choice(even_ids)
            even_count += 1
        else:
            pool = odd_ids if rng.random() < 0.5 else even_ids
            chosen = rng.choice(pool)
            if pool is odd_ids:
                odd_count += 1
            else:
                even_count += 1
        object_ids.append(chosen)

    rng.shuffle(object_ids)
    return object_ids


def _box_is_free(
    occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _occupy_box(
    occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int
) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_objects(
    rng: random.Random,
    templates: list[dict[str, object]],
    object_ids: list[int],
    main_h: int,
    width: int,
) -> list[tuple[int, int, int, int]] | None:
    occ_in = np.zeros((main_h, width), dtype=bool)
    occ_out = np.zeros((main_h, width), dtype=bool)
    placements: list[tuple[int, int, int, int] | None] = [None] * len(object_ids)

    order = sorted(
        range(len(object_ids)),
        key=lambda idx: int(np.count_nonzero(templates[object_ids[idx]]["mask"])),
        reverse=True,
    )

    for slot in order:
        template_id = object_ids[slot]
        height = int(templates[template_id]["height"])
        width_i = int(templates[template_id]["width"])
        is_even = (template_id + 1) % 2 == 0

        candidates = [
            (r, c)
            for r in range(main_h - height + 1)
            for c in range(width - width_i + 1)
            if not is_even or (main_h - height - r) != r
        ]
        rng.shuffle(candidates)

        placed = False
        for r, c in candidates:
            out_r = main_h - height - r if is_even else r
            if not _box_is_free(occ_in, r, c, height, width_i, BOX_PAD):
                continue
            if not _box_is_free(occ_out, out_r, c, height, width_i, BOX_PAD):
                continue
            _occupy_box(occ_in, r, c, height, width_i, BOX_PAD)
            _occupy_box(occ_out, out_r, c, height, width_i, BOX_PAD)
            placements[slot] = (template_id, r, c, out_r)
            placed = True
            break

        if not placed:
            return None

    return [placement for placement in placements if placement is not None]


def _render_main_pair(
    bg: int,
    templates: list[dict[str, object]],
    placements: list[tuple[int, int, int, int]],
    main_h: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((main_h, width), bg, dtype=int)
    out = np.full((main_h, width), bg, dtype=int)
    for template_id, r, c, out_r in placements:
        mask = templates[template_id]["mask"]
        color = int(templates[template_id]["color"])
        _paint(inp, mask, r, c, GREY)
        _paint(out, mask, r if (template_id + 1) % 2 == 1 else out_r, c, color)
    return inp, out


def _render_legend(
    bg: int,
    templates: list[dict[str, object]],
    total_width: int,
    rng: random.Random,
) -> np.ndarray:
    max_height = max(int(template["height"]) for template in templates)
    bottom_pad = rng.randint(*BOTTOM_PAD_RANGE)
    legend = np.full((max_height + bottom_pad, total_width), bg, dtype=int)
    left_margin = rng.randint(*SIDE_MARGIN_RANGE)
    c = left_margin
    for idx, template in enumerate(templates):
        _paint(
            legend,
            template["mask"],
            0,
            c,
            int(template["color"]),
        )
        c += int(template["width"])
        if idx + 1 < len(templates):
            c += rng.randint(*LEGEND_GAP_RANGE)
    return legend


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([color for color in range(10) if color != GREY])
    template_count = rng.randint(*TEMPLATE_COUNT_RANGE)
    templates = _sample_templates(rng, template_count)
    if templates is None:
        return None

    colors = rng.sample([color for color in range(10) if color not in {bg, GREY}], template_count)
    for template, color in zip(templates, colors):
        template["color"] = color

    object_ids = _balanced_object_ids(rng, template_count)
    main_h_min = max(MAIN_H_RANGE[0], max(int(template["height"]) for template in templates) + 4)
    if main_h_min > MAIN_H_RANGE[1]:
        return None
    main_h = rng.randint(main_h_min, MAIN_H_RANGE[1])

    legend_width_min = sum(int(template["width"]) for template in templates)
    legend_width_min += (len(templates) - 1) * LEGEND_GAP_RANGE[1]
    legend_width_min += SIDE_MARGIN_RANGE[1]
    width_min = max(GRID_W_RANGE[0], legend_width_min)
    if width_min > GRID_W_RANGE[1]:
        return None
    width = rng.randint(width_min, GRID_W_RANGE[1])

    legend = _render_legend(bg, templates, width, rng)
    placements = _place_objects(rng, templates, object_ids, main_h, width)
    if placements is None:
        return None

    inp_main, expected = _render_main_pair(bg, templates, placements, main_h, width)
    inp = np.vstack([inp_main, legend])

    if _bg_color(inp) != bg:
        return None
    if len(_components(inp_main == GREY)) != len(object_ids):
        return None
    legend_top, legend_map = _extract_legend(inp, bg)
    if legend_top != main_h or len(legend_map) != template_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
