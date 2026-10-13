from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
TEMPLATE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    dirs = DIR4 if connectivity == 4 else DIR8
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _touches_mask(mask: np.ndarray, coords: np.ndarray, connectivity: int = 8) -> bool:
    dirs = DIR4 if connectivity == 4 else DIR8
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
                return True
    return False


def _catalog_mask(grid: np.ndarray) -> np.ndarray:
    grey_comps = _components(grid == GREY, connectivity=4)
    assert grey_comps, "expected a gray catalog region"
    grey_comps.sort(key=lambda comp: (-len(comp), int(comp[:, 0].min()), int(comp[:, 1].min())))
    if len(grey_comps) > 1:
        assert len(grey_comps[0]) > len(grey_comps[1]), "catalog must be the unique largest gray component"
    comp = grey_comps[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[comp[:, 0], comp[:, 1]] = True
    assert np.any(mask[0]), "catalog should touch the top edge"
    return mask


def _extract_templates(grid: np.ndarray, catalog: np.ndarray) -> list[np.ndarray]:
    visible = (grid != BLACK) & (grid != GREY)
    templates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(visible, connectivity=8):
        if not _touches_mask(catalog, coords, connectivity=8):
            continue
        box = _bbox_coords(coords)
        sprite = _crop_bbox(grid, box)
        sprite[sprite == GREY] = BLACK
        templates.append((box, sprite))
    templates.sort(key=lambda item: (item[0][0], item[0][1]))
    assert templates, "expected at least one catalog template"
    return [sprite for _box, sprite in templates]


def _workspace_parts(grid: np.ndarray, catalog: np.ndarray) -> list[np.ndarray]:
    visible = (grid != BLACK) & (grid != GREY)
    parts: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(visible, connectivity=8):
        if _touches_mask(catalog, coords, connectivity=8):
            continue
        box = _bbox_coords(coords)
        arr = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=int)
        arr[coords[:, 0] - box[0], coords[:, 1] - box[1]] = grid[coords[:, 0], coords[:, 1]]
        parts.append((box, arr))
    parts.sort(key=lambda item: (item[0][0], item[0][1]))
    return parts


def _matching_offsets(part: np.ndarray, template: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = part.shape
    th, tw = template.shape
    if ph > th or pw > tw:
        return []
    coords = np.argwhere(part != BLACK)
    if len(coords) == 0:
        return []
    offsets: list[tuple[int, int]] = []
    for r0 in range(th - ph + 1):
        for c0 in range(tw - pw + 1):
            ok = True
            for pr_raw, pc_raw in coords:
                pr = int(pr_raw)
                pc = int(pc_raw)
                if int(template[r0 + pr, c0 + pc]) != int(part[pr, pc]):
                    ok = False
                    break
            if ok:
                offsets.append((r0, c0))
    return offsets


def _part_counters(grid: np.ndarray, box: tuple[int, int, int, int], part: np.ndarray, catalog: np.ndarray) -> set[tuple[int, int]]:
    counters: set[tuple[int, int]] = set()
    r0, c0, _r1, _c1 = box
    for rr_raw, cc_raw in np.argwhere(part != BLACK):
        rr = r0 + int(rr_raw)
        cc = c0 + int(cc_raw)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(grid.shape, nr, nc) and not catalog[nr, nc] and int(grid[nr, nc]) == GREY:
                counters.add((nr, nc))
    return counters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each workspace fragment to a catalog template, then tile that template horizontally by the local gray count.

    Input structure:
    - A large gray catalog strip occupies the top of the grid and contains 2-3 multicolor template sprites.
    - The workspace below uses black background and contains 2-3 multicolor partial copies of those templates.
    - Each partial copy has 1-4 gray counter pixels in its 4-neighborhood; gray catalog cells are background, not counters.

    Rule:
    - Identify the unique largest gray 4-connected region as the catalog background, then extract every non-black sprite touching it as a template, treating gray inside the template bbox as transparent.
    - For each workspace partial copy, find the unique template and unique offset for which every visible partial pixel matches the template.
    - Count the gray workspace pixels that are side-adjacent to that partial; this count N is the number of horizontal repetitions.
    - Stamp the matched full template N times starting from the aligned template origin, shifting each repeat right by one template width; template black cells are transparent.
    - Erase all gray workspace counters to black and keep the catalog unchanged.

    Key insight:
    - The puzzle splits into two independent inferences: partial pattern matching decides which template to stamp and where its origin is, while local gray counting decides how many adjacent copies to place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    catalog = _catalog_mask(g)
    templates = _extract_templates(g, catalog)
    parts = _workspace_parts(g, catalog)
    assert parts, "expected workspace partials"

    out = g.copy()
    out[(out == GREY) & (~catalog)] = BLACK

    for box, part in parts:
        matches: list[tuple[int, int, int]] = []
        for idx, template in enumerate(templates):
            for off_r, off_c in _matching_offsets(part, template):
                matches.append((idx, off_r, off_c))
        assert len(matches) == 1, "each partial copy must match exactly one template alignment"

        template_idx, off_r, off_c = matches[0]
        template = templates[template_idx]
        counters = _part_counters(g, box, part, catalog)
        repeat_count = len(counters)
        assert 1 <= repeat_count <= 4, "expected 1-4 adjacent counters"

        anchor_r = box[0] - off_r
        anchor_c = box[1] - off_c
        th, tw = template.shape
        assert 0 <= anchor_r and 0 <= anchor_c and anchor_r + th <= g.shape[0]
        assert anchor_c + repeat_count * tw <= g.shape[1], "tiled template must fit"

        for k in range(repeat_count):
            _paste(out, template, anchor_r, anchor_c + k * tw, transparent=BLACK)

    return out


BASE_TEMPLATES = [
    np.array([[1, 2, 1], [2, 3, 2], [0, 1, 2]], dtype=int),
    np.array([[1, 1, 2, 0], [0, 2, 3, 3], [1, 0, 2, 0]], dtype=int),
    np.array([[0, 1, 2], [3, 1, 0], [2, 3, 1], [0, 2, 1]], dtype=int),
    np.array([[1, 2, 0, 3], [0, 1, 2, 0], [3, 0, 1, 2], [1, 3, 0, 2]], dtype=int),
    np.array([[1, 0, 2, 1], [2, 3, 1, 0], [0, 2, 3, 1]], dtype=int),
]


def _array_signature(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in arr)


def _random_template() -> np.ndarray:
    sprite = np.array(random.choice(BASE_TEMPLATES), copy=True)
    sprite = np.rot90(sprite, random.randrange(4))
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)

    distinct_colors = random.choice([2, 3])
    palette = random.sample(TEMPLATE_COLORS, distinct_colors)
    if distinct_colors == 2:
        mapping = {1: palette[0], 2: palette[1], 3: random.choice(palette)}
    else:
        mapping = {1: palette[0], 2: palette[1], 3: palette[2]}

    out = np.zeros_like(sprite)
    for symbol, color in mapping.items():
        out[sprite == symbol] = color
    used = sorted(set(int(v) for v in out[out != BLACK]))
    assert len(used) >= 2
    return out


def _connected_subset(template: np.ndarray) -> np.ndarray | None:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(template != BLACK)]
    total = len(coords)
    target = random.randint((total + 1) // 2, total - 1)
    coord_set = set(coords)
    seed = random.choice(coords)
    chosen = {seed}
    frontier = set()
    for dr, dc in DIR8:
        nxt = (seed[0] + dr, seed[1] + dc)
        if nxt in coord_set:
            frontier.add(nxt)

    while len(chosen) < target and frontier:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        if nxt in chosen:
            continue
        chosen.add(nxt)
        for dr, dc in DIR8:
            rr, cc = nxt[0] + dr, nxt[1] + dc
            if (rr, cc) in coord_set and (rr, cc) not in chosen:
                frontier.add((rr, cc))

    if len(chosen) != target:
        return None

    sub_mask = np.zeros_like(template, dtype=bool)
    for r, c in chosen:
        sub_mask[r, c] = True
    if len(_components(sub_mask, connectivity=8)) != 1:
        return None
    return sub_mask


def _external_counter_candidates(retained_coords: np.ndarray, template_shape: tuple[int, int], repeat_count: int) -> list[tuple[int, int]]:
    h, w = template_shape
    total_width = repeat_count * w
    candidates: set[tuple[int, int]] = set()
    for r_raw, c_raw in retained_coords:
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < total_width:
                continue
            candidates.add((nr, nc))
    return sorted(candidates)


def _choose_partial(template_idx: int, templates: list[np.ndarray]) -> dict | None:
    template = templates[template_idx]
    visible = np.argwhere(template != BLACK)
    total = len(visible)
    for _ in range(200):
        subset_mask = _connected_subset(template)
        if subset_mask is None:
            continue
        retained = np.argwhere(subset_mask)
        if len(retained) < (total + 1) // 2 or len(retained) >= total:
            continue
        part_box = _bbox_coords(retained)
        part = np.zeros((part_box[2] - part_box[0] + 1, part_box[3] - part_box[1] + 1), dtype=int)
        for rr_raw, cc_raw in retained:
            rr = int(rr_raw)
            cc = int(cc_raw)
            part[rr - part_box[0], cc - part_box[1]] = int(template[rr, cc])
        if len(set(int(v) for v in part[part != BLACK])) < 2:
            continue

        matches: list[tuple[int, int, int]] = []
        for idx, other in enumerate(templates):
            for off_r, off_c in _matching_offsets(part, other):
                matches.append((idx, off_r, off_c))
        expected_match = (template_idx, part_box[0], part_box[1])
        if matches != [expected_match]:
            continue

        counter_options = {
            n: _external_counter_candidates(retained, template.shape, n) for n in range(1, 5)
        }
        valid_counts = [n for n, cells in counter_options.items() if len(cells) >= n]
        if not valid_counts:
            continue
        return {
            "template_idx": template_idx,
            "part": part,
            "offset": (part_box[0], part_box[1]),
            "retained": retained,
            "valid_counts": valid_counts,
            "counter_options": counter_options,
        }
    return None


def _build_templates(count: int) -> list[np.ndarray] | None:
    for _ in range(200):
        templates = [_random_template() for _ in range(count)]
        sigs = {_array_signature(template) for template in templates}
        if len(sigs) != count:
            continue
        return templates
    return None


def _pick_repeat_counts(partials: list[dict]) -> list[int] | None:
    options = [partial["valid_counts"] for partial in partials]
    all_choices = [list(map(int, opts)) for opts in options]
    for _ in range(300):
        counts = [random.choice(opts) for opts in all_choices]
        if len(set(counts)) == 1:
            continue
        if max(counts) == 1:
            continue
        return counts
    for counts in np.array(np.meshgrid(*all_choices)).T.reshape(-1, len(all_choices)):
        counts = [int(v) for v in counts]
        if len(set(counts)) > 1 and max(counts) > 1:
            return counts
    return None


def _catalog_layout(width: int, catalog_h: int, templates: list[np.ndarray]) -> list[tuple[int, int]] | None:
    min_total = sum(template.shape[1] for template in templates) + 2 * (len(templates) - 1) + 2
    if min_total > width:
        return None
    positions: list[tuple[int, int]] = []
    col = 1
    slack = width - min_total
    extra_gaps = [0] * (len(templates) + 1)
    for _ in range(slack):
        extra_gaps[random.randrange(len(extra_gaps))] += 1
    col += extra_gaps[0]
    for idx, template in enumerate(templates):
        h, w = template.shape
        if catalog_h - h - 1 < 1:
            return None
        row = random.randint(1, catalog_h - h - 1)
        positions.append((row, col))
        col += w
        if idx < len(templates) - 1:
            col += 2 + extra_gaps[idx + 1]
    return positions


def _workspace_layout(
    shape: tuple[int, int], catalog_h: int, templates: list[np.ndarray], partials: list[dict], counts: list[int]
) -> list[dict] | None:
    h, w = shape
    n = len(templates)
    row_step = 5
    first_top = catalog_h + 1
    if first_top + row_step * (n - 1) + 4 >= h:
        return None

    placed: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    for idx, (template, partial, repeat_count) in enumerate(zip(templates, partials, counts)):
        th, tw = template.shape
        tiled_w = tw * repeat_count
        if tiled_w + 2 > w:
            return None

        band_top = first_top + idx * row_step
        row_candidates = [band_top, band_top + 1]
        row_candidates = [r for r in row_candidates if r + th < h - 1]
        random.shuffle(row_candidates)
        placed_record = None
        for anchor_r in row_candidates:
            if anchor_r <= catalog_h:
                continue
            max_c = w - tiled_w - 1
            col_candidates = list(range(1, max_c + 1))
            random.shuffle(col_candidates)
            for anchor_c in col_candidates:
                box = (anchor_r, anchor_c, anchor_r + th - 1, anchor_c + tiled_w - 1)
                overlap = False
                for or0, oc0, or1, oc1 in used_boxes:
                    if not (box[2] + 1 < or0 or or1 + 1 < box[0] or box[3] + 1 < oc0 or oc1 + 1 < box[1]):
                        overlap = True
                        break
                if overlap:
                    continue

                counter_rel = partial["counter_options"][repeat_count]
                counter_abs = [(anchor_r + rr, anchor_c + cc) for rr, cc in counter_rel]
                if any(not _in_bounds(shape, rr, cc) for rr, cc in counter_abs):
                    continue

                chosen_counters = random.sample(counter_abs, repeat_count)
                if any(any(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in chosen_counters) for r0, c0, r1, c1 in used_boxes):
                    continue

                placed_record = {
                    "template_idx": idx,
                    "anchor": (anchor_r, anchor_c),
                    "part": partial["part"],
                    "offset": partial["offset"],
                    "retained": partial["retained"],
                    "count": repeat_count,
                    "counters": chosen_counters,
                }
                used_boxes.append(box)
                break
            if placed_record is not None:
                break
        if placed_record is None:
            return None
        placed.append(placed_record)
    return placed


def _construct_instance() -> tuple[np.ndarray, np.ndarray] | None:
    template_count = random.choice([2, 3])
    templates = _build_templates(template_count)
    if templates is None:
        return None

    partials = []
    for idx in range(template_count):
        partial = _choose_partial(idx, templates)
        if partial is None:
            return None
        partials.append(partial)

    counts = _pick_repeat_counts(partials)
    if counts is None:
        return None

    if not any(partial["offset"] != (0, 0) for partial in partials):
        return None

    catalog_h = random.randint(5, 6)
    min_width = max(
        18,
        sum(template.shape[1] for template in templates) + 2 * (template_count - 1) + 2,
        max(template.shape[1] * count for template, count in zip(templates, counts)) + 3,
    )
    if min_width > 28:
        return None
    width = random.randint(min_width, 28)
    height = catalog_h + 1 + 5 * (template_count - 1) + 5 + random.randint(0, 1)
    if height > 22:
        return None

    catalog_positions = _catalog_layout(width, catalog_h, templates)
    if catalog_positions is None:
        return None

    placements = _workspace_layout((height, width), catalog_h, templates, partials, counts)
    if placements is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[:catalog_h, :] = GREY
    for template, (r0, c0) in zip(templates, catalog_positions):
        _paste(inp, template, r0, c0, transparent=BLACK)

    for record in placements:
        anchor_r, anchor_c = record["anchor"]
        off_r, off_c = record["offset"]
        _paste(inp, record["part"], anchor_r + off_r, anchor_c + off_c, transparent=BLACK)
        for rr, cc in record["counters"]:
            inp[rr, cc] = GREY

    expected = inp.copy()
    catalog = _catalog_mask(inp)
    expected[(expected == GREY) & (~catalog)] = BLACK
    for record in placements:
        template = templates[record["template_idx"]]
        anchor_r, anchor_c = record["anchor"]
        for k in range(record["count"]):
            _paste(expected, template, anchor_r, anchor_c + k * template.shape[1], transparent=BLACK)

    if not _validate_instance(inp, expected, templates, template_count):
        return None
    return inp, expected


def _validate_instance(inp: np.ndarray, expected: np.ndarray, templates: list[np.ndarray], template_count: int) -> bool:
    try:
        catalog = _catalog_mask(inp)
        extracted = _extract_templates(inp, catalog)
        if len(extracted) != template_count:
            return False

        parts = _workspace_parts(inp, catalog)
        if len(parts) != template_count:
            return False

        matched_templates: list[int] = []
        repeat_counts: list[int] = []
        offsets: list[tuple[int, int]] = []
        for box, part in parts:
            matches: list[tuple[int, int, int]] = []
            for idx, template in enumerate(extracted):
                for off_r, off_c in _matching_offsets(part, template):
                    matches.append((idx, off_r, off_c))
            if len(matches) != 1:
                return False
            matched_templates.append(matches[0][0])
            offsets.append((matches[0][1], matches[0][2]))
            repeat_counts.append(len(_part_counters(inp, box, part, catalog)))

        if len(set(matched_templates)) != template_count:
            return False
        if max(repeat_counts) <= 1 or len(set(repeat_counts)) == 1:
            return False
        if not any(offset != (0, 0) for offset in offsets):
            return False
        if not np.array_equal(main(inp), expected):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _construct_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid template-count tiling puzzle")
