import random
from collections import Counter, deque

import numpy as np


CATALOG_BG = 5
WORK_BG = 0
ORIENTATIONS = ("orig", "hflip", "vflip", "hvflip")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    assert comps, "expected a grey catalog background"
    largest = max(comps, key=len)
    out = np.zeros_like(mask, dtype=bool)
    for r, c in largest:
        out[r, c] = True
    return out


def _touches_mask(coords: list[tuple[int, int]], mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _patch_from_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return bbox, patch


def _rank_template_colors(sprite: np.ndarray) -> tuple[int, int]:
    colors = [int(v) for v in np.unique(sprite) if v not in (WORK_BG, CATALOG_BG)]
    assert len(colors) == 2, "each template must use exactly two colors"
    counts = Counter(int(v) for v in sprite.flat if v not in (WORK_BG, CATALOG_BG))
    ranked = sorted(colors, key=lambda color: (-counts[color], color))
    return ranked[0], ranked[1]


def _orient(arr: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "orig":
        return arr.copy()
    if orientation == "hflip":
        return np.fliplr(arr)
    if orientation == "vflip":
        return np.flipud(arr)
    if orientation == "hvflip":
        return np.flipud(np.fliplr(arr))
    raise ValueError(f"unknown orientation: {orientation}")


def _swap_two_colors(arr: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = arr.copy()
    mask_a = arr == int(color_a)
    mask_b = arr == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _stamp_for_orientation(template: dict, orientation: str) -> np.ndarray:
    stamp = _orient(template["sprite"], orientation)
    if orientation in {"hflip", "vflip"}:
        stamp = _swap_two_colors(stamp, template["color_a"], template["color_b"])
    return stamp


def _transparent_paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _extract_templates_and_workspace(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    catalog_mask = _largest_component_mask(grid == CATALOG_BG)
    foreground = (grid != WORK_BG) & (grid != CATALOG_BG)
    templates: list[dict] = []
    workspace: list[dict] = []
    for coords in _components(foreground):
        bbox, patch = _patch_from_component(grid, coords, WORK_BG)
        if _touches_mask(coords, catalog_mask):
            color_a, color_b = _rank_template_colors(patch)
            templates.append(
                {
                    "bbox": bbox,
                    "sprite": patch,
                    "color_a": int(color_a),
                    "color_b": int(color_b),
                }
            )
        else:
            workspace.append({"bbox": bbox, "patch": patch})
    templates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    workspace.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return templates, workspace


def _all_matches(patch: np.ndarray, templates: list[dict]) -> list[tuple[int, str, int, int]]:
    ph, pw = patch.shape
    colored = np.argwhere(patch != WORK_BG)
    matches: list[tuple[int, str, int, int]] = []
    for tidx, template in enumerate(templates):
        for orientation in ORIENTATIONS:
            stamp = _stamp_for_orientation(template, orientation)
            th, tw = stamp.shape
            if ph > th or pw > tw:
                continue
            for off_r in range(th - ph + 1):
                for off_c in range(tw - pw + 1):
                    ok = True
                    for rr, cc in colored:
                        if int(stamp[off_r + int(rr), off_c + int(cc)]) != int(patch[rr, cc]):
                            ok = False
                            break
                    if ok:
                        matches.append((tidx, orientation, off_r, off_c))
    return matches


def _unique_match(patch: np.ndarray, templates: list[dict]) -> tuple[int, str, int, int] | None:
    matches = _all_matches(patch, templates)
    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each uniquely matched partial workspace sprite using the catalog template, flip type, and flip-driven two-color swap rule.
    
    Input structure:
    - A large grey catalog strip on the left contains 2-3 rectangular multicolor templates.
    - Each template uses exactly two non-black, non-grey colors; grey cells belong to the catalog background.
    - The black workspace on the right contains several non-black connected objects: some are partial template views and some are distractors.
    
    Rule:
    - Extract the catalog templates as the non-grey, non-black components that touch the grey catalog background, and rank each template's two colors into `colorA` and `colorB` by pixel count.
    - For each workspace object, try every template in four geometric views: original, horizontal flip, vertical flip, and both flips; single-axis flips also swap `colorA` and `colorB`, while original and both-flip views keep the colors unchanged.
    - If the workspace object matches exactly one aligned subregion of one transformed template, stamp the whole transformed template back into the workspace at that alignment, treating grey catalog cells as transparent; otherwise leave the object unchanged.
    
    Key insight:
    - The flip type is not only a geometric alignment cue: it also encodes whether the template's two foreground colors must be exchanged before completion.
    """
    g = np.asarray(grid, dtype=int)
    templates, workspace = _extract_templates_and_workspace(g)

    out = g.copy()
    for obj in workspace:
        match = _unique_match(obj["patch"], templates)
        if match is None:
            continue
        tidx, orientation, off_r, off_c = match
        anchor_r = obj["bbox"][0] - off_r
        anchor_c = obj["bbox"][1] - off_c
        stamp = _stamp_for_orientation(templates[tidx], orientation)
        _transparent_paste(out, stamp, anchor_r, anchor_c, CATALOG_BG)
    return out


def _random_template(color_a: int, color_b: int, height: int, width: int) -> np.ndarray | None:
    area = height * width
    for _ in range(250):
        count_a = random.randint(area // 2 + 1, area - 3)
        flat = np.array([True] * count_a + [False] * (area - count_a))
        np.random.shuffle(flat)
        mask = flat.reshape(height, width)
        sprite = np.where(mask, int(color_a), int(color_b)).astype(int)
        finals = (
            sprite,
            _swap_two_colors(np.fliplr(sprite), color_a, color_b),
            _swap_two_colors(np.flipud(sprite), color_a, color_b),
            np.flipud(np.fliplr(sprite)),
        )
        if len({arr.tobytes() for arr in finals}) < 4:
            continue
        if np.array_equal(sprite, np.fliplr(sprite)) or np.array_equal(sprite, np.flipud(sprite)):
            continue
        return sprite
    return None


def _random_connected_subset(shape: tuple[int, int], target_area: int) -> np.ndarray:
    h, w = shape
    chosen: set[tuple[int, int]] = set()
    start = (random.randrange(h), random.randrange(w))
    chosen.add(start)
    frontier = set()
    frontier.update(
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if 0 <= nr < h and 0 <= nc < w
    )
    while len(chosen) < target_area:
        if not frontier:
            frontier = set()
            for r, c in chosen:
                frontier.update(
                    (nr, nc)
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen
                )
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                frontier.add((nr, nc))
    mask = np.zeros(shape, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _partial_from_stamp(stamp: np.ndarray, templates: list[dict], tidx: int, orientation: str) -> tuple[np.ndarray, int, int] | None:
    area = int(np.count_nonzero(stamp != CATALOG_BG))
    min_area = max((area + 2) // 3, 3)
    for _ in range(350):
        target_area = random.randint(min_area, area - 1)
        subset = _random_connected_subset(stamp.shape, target_area)
        coords = [(int(r), int(c)) for r, c in np.argwhere(subset)]
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = int(stamp[r, c])
        match = _unique_match(patch, templates)
        if match == (tidx, orientation, r0, c0):
            return patch, r0, c0
    return None


def _boxes_clear(boxes: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], gap: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _random_distractor(color: int) -> np.ndarray:
    shapes = [
        np.array([[color]], dtype=int),
        np.array([[color, color]], dtype=int),
        np.array([[color], [color]], dtype=int),
        np.array([[color, 0], [color, color]], dtype=int),
        np.array([[color, color], [0, color]], dtype=int),
    ]
    return random.choice(shapes).copy()


def _orientation_schedule(count: int) -> list[str]:
    base = list(ORIENTATIONS)
    random.shuffle(base)
    picked = base[: min(count, 4)]
    while len(picked) < count:
        picked.append(random.choice(ORIENTATIONS))
    if not any(o in {"hflip", "vflip"} for o in picked):
        picked[0] = random.choice(("hflip", "vflip"))
    if not any(o in {"orig", "hvflip"} for o in picked):
        picked[-1] = random.choice(("orig", "hvflip"))
    random.shuffle(picked)
    return picked


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    template_count = random.randint(2, 3)
    max_matches = min(5, 2 * template_count)
    match_count = random.randint(3, max_matches)
    distractor_count = random.randint(1, 2)

    height = random.randint(20, 25)
    width = random.randint(22, 28)
    catalog_width = random.randint(7, 9)
    workspace_left = catalog_width + 1
    if width - workspace_left < 11:
        return None

    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    template_colors = [tuple(color_pool[2 * i : 2 * i + 2]) for i in range(template_count)]
    distractor_colors = [c for c in PALETTE if c not in {v for pair in template_colors for v in pair}]
    if len(distractor_colors) < 1:
        return None

    templates_data = []
    for idx in range(template_count):
        th = random.randint(4, 5)
        tw = random.randint(4, min(5, catalog_width - 2))
        color_a, color_b = template_colors[idx]
        sprite = _random_template(color_a, color_b, th, tw)
        if sprite is None:
            return None
        templates_data.append({"sprite": sprite, "size": sprite.shape})

    needed_height = 2 + sum(item["size"][0] for item in templates_data) + (template_count - 1)
    if needed_height > height - 1:
        return None

    output = np.zeros((height, width), dtype=int)
    output[:, :catalog_width] = CATALOG_BG

    y = 1
    template_bboxes = []
    for item in templates_data:
        sprite = item["sprite"]
        th, tw = sprite.shape
        max_x = catalog_width - tw - 1
        if max_x < 1:
            return None
        x = random.randint(1, max_x)
        if y + th > height - 1:
            return None
        _place_patch(output, sprite, y, x)
        template_bboxes.append((y, x, y + th - 1, x + tw - 1))
        item["catalog_bbox"] = template_bboxes[-1]
        y += th + random.randint(1, 2)

    templates, _ = _extract_templates_and_workspace(output)
    if len(templates) != template_count:
        return None

    orientations = _orientation_schedule(match_count)
    counts = [0] * template_count
    used_orients = [set() for _ in range(template_count)]
    assignments: list[tuple[int, str]] = []
    for orientation in orientations:
        options = [
            idx
            for idx in range(template_count)
            if counts[idx] < 2 and orientation not in used_orients[idx]
        ]
        if not options:
            options = [idx for idx in range(template_count) if counts[idx] < 2]
        if not options:
            return None
        tidx = random.choice(options)
        counts[tidx] += 1
        used_orients[tidx].add(orientation)
        assignments.append((tidx, orientation))

    occupied_boxes = template_bboxes[:]
    partial_specs = []
    for tidx, orientation in assignments:
        stamp = _stamp_for_orientation(templates[tidx], orientation)
        sh, sw = stamp.shape
        partial = _partial_from_stamp(stamp, templates, tidx, orientation)
        if partial is None:
            return None
        patch, off_r, off_c = partial
        placed = False
        for _ in range(250):
            anchor_r = random.randint(1, height - sh - 1)
            anchor_c = random.randint(workspace_left, width - sw)
            bbox = (anchor_r, anchor_c, anchor_r + sh - 1, anchor_c + sw - 1)
            if _boxes_clear(occupied_boxes, bbox, gap=1):
                occupied_boxes.append(bbox)
                _transparent_paste(output, stamp, anchor_r, anchor_c, CATALOG_BG)
                partial_specs.append((patch, anchor_r + off_r, anchor_c + off_c))
                placed = True
                break
        if not placed:
            return None

    distractors: list[tuple[np.ndarray, int, int]] = []
    for _ in range(distractor_count):
        color = random.choice(distractor_colors)
        patch = _random_distractor(color)
        dh, dw = patch.shape
        placed = False
        for _ in range(200):
            top = random.randint(1, height - dh - 1)
            left = random.randint(workspace_left, width - dw)
            bbox = (top, left, top + dh - 1, left + dw - 1)
            if _boxes_clear(occupied_boxes, bbox, gap=1):
                occupied_boxes.append(bbox)
                distractors.append((patch, top, left))
                placed = True
                break
        if not placed:
            return None

    inp = np.zeros_like(output)
    inp[:, :catalog_width] = CATALOG_BG
    for item in templates_data:
        r0, c0, _, _ = item["catalog_bbox"]
        _place_patch(inp, item["sprite"], r0, c0)
    for patch, top, left in partial_specs:
        _place_patch(inp, patch, top, left)
    for patch, top, left in distractors:
        _transparent_paste(inp, patch, top, left, WORK_BG)
        _transparent_paste(output, patch, top, left, WORK_BG)

    actual = main(inp)
    if not np.array_equal(actual, output):
        return None
    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid puzzle input")
