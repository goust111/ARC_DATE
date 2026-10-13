from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, paste, random_free_location


BG = 0
CATALOG_BG = 5
YELLOW = 4
TEMPLATE_PALETTE = (1, 2, 3, 6, 7, 8, 9)

GRID_MIN = 20
GRID_MAX = 26
TEMPLATE_COUNT_RANGE = (2, 3)
TEMPLATE_SIZE_RANGE = (3, 5)
CATALOG_GAP_RANGE = (1, 2)
DISTRACTOR_COUNT_RANGE = (1, 2)
GEN_RETRIES = 400


def _rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(grid, dtype=int), -int(turns) % 4).copy()


def _corner_turns(shape: tuple[int, int], coord: tuple[int, int]) -> int | None:
    h, w = map(int, shape)
    mapping = {
        (0, 0): 0,
        (0, w - 1): 1,
        (h - 1, w - 1): 2,
        (h - 1, 0): 3,
    }
    return mapping.get((int(coord[0]), int(coord[1])))


def _catalog_component(grid: np.ndarray) -> dict:
    grey_components = find_components(grid, bg=BG, colors=[CATALOG_BG], connectivity=4)
    assert grey_components, "expected a grey catalog background"
    grey_components.sort(key=lambda comp: int(comp["area"]), reverse=True)
    assert len(grey_components) == 1 or int(grey_components[0]["area"]) > int(grey_components[1]["area"])
    return grey_components[0]


def _extract_templates(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[np.ndarray]]:
    catalog = _catalog_component(grid)
    bbox = tuple(map(int, catalog["bbox"]))
    r0, c0, r1, c1 = bbox
    region = crop_bbox(grid, bbox)
    region[region == CATALOG_BG] = BG

    templates = []
    for comp in sorted(
        find_components(region, bg=BG, connectivity=8, monochromatic=False),
        key=lambda item: (item["bbox"][0], item["bbox"][1]),
    ):
        sprite = crop_bbox(region, comp["bbox"])
        yellow = np.argwhere(sprite == YELLOW)
        assert yellow.shape == (1, 2), "each template must contain exactly one yellow anchor"
        assert tuple(map(int, yellow[0])) == (0, 0), "catalog templates must anchor yellow at top-left"
        templates.append(sprite)

    assert 2 <= len(templates) <= 3, "expected 2-3 catalog templates"
    return (r0, c0, r1, c1), templates


def _workspace_components(grid: np.ndarray, catalog_bbox: tuple[int, int, int, int]) -> list[dict]:
    workspace = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = map(int, catalog_bbox)
    workspace[r0 : r1 + 1, c0 : c1 + 1] = BG
    return sorted(
        find_components(workspace, bg=BG, connectivity=8, monochromatic=False),
        key=lambda comp: (comp["bbox"][0], comp["bbox"][1]),
    )


def _match_component(
    grid: np.ndarray,
    component: dict,
    template: np.ndarray,
) -> dict | None:
    patch = crop_bbox(grid, component["bbox"])
    yellow = np.argwhere(patch == YELLOW)
    if yellow.shape != (1, 2):
        return None

    turns = _corner_turns(patch.shape, tuple(map(int, yellow[0])))
    if turns is None:
        return None

    rotated = _rotate_cw(template, turns)
    rot_yellow = tuple(map(int, np.argwhere(rotated == YELLOW)[0]))

    r0, c0, _r1, _c1 = map(int, component["bbox"])
    abs_yellow = (r0 + int(yellow[0, 0]), c0 + int(yellow[0, 1]))
    top_r = int(abs_yellow[0] - rot_yellow[0])
    top_c = int(abs_yellow[1] - rot_yellow[1])
    h, w = map(int, rotated.shape)
    if top_r < 0 or top_c < 0 or top_r + h > grid.shape[0] or top_c + w > grid.shape[1]:
        return None

    source_mask = np.zeros(grid.shape, dtype=bool)
    coords = component["coords"]
    source_mask[coords[:, 0], coords[:, 1]] = True

    for rr_raw, cc_raw in coords:
        rr = int(rr_raw)
        cc = int(cc_raw)
        tr = rr - top_r
        tc = cc - top_c
        if not (0 <= tr < h and 0 <= tc < w):
            return None
        if int(rotated[tr, tc]) != int(grid[rr, cc]):
            return None

    for tr_raw, tc_raw in np.argwhere(rotated != BG):
        tr = int(tr_raw)
        tc = int(tc_raw)
        rr = top_r + tr
        cc = top_c + tc
        if int(grid[rr, cc]) != BG and not source_mask[rr, cc] and int(grid[rr, cc]) != int(rotated[tr, tc]):
            return None

    return {
        "turns": turns,
        "top_left": (top_r, top_c),
        "rotated": rotated,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Complete rotated workspace fragments by matching them to catalog templates and stamping the full rotated sprite.

    Input structure:
    - A grey rectangular catalog sits in the top-left corner on an otherwise black grid.
    - The catalog contains 2-3 separated multicolor template sprites, each with exactly
      one yellow pixel at its top-left corner.
    - The black workspace contains 2-3 partial multicolor fragments plus optional
      distractor objects; matched fragments keep one yellow pixel at one bbox corner.

    Rule:
    - Read every catalog sprite as a complete template.
    - For each workspace component, use the yellow corner position to decode whether the
      fragment is rotated 0/90/180/270 degrees clockwise, then find the unique template
      whose equally rotated full sprite contains every visible fragment pixel at the
      aligned coordinates.
    - Stamp that full rotated template back into the workspace with the yellow anchor
      aligned to the fragment's yellow anchor. Leave unmatched workspace objects and the
      catalog unchanged.

    Key insight:
    - Template identity comes from partial multicolor matching, while the yellow corner's
      drift from top-left to another corner directly encodes the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    catalog_bbox, templates = _extract_templates(g)
    planned: dict[tuple[int, int], int] = {}

    for component in _workspace_components(g, catalog_bbox):
        matches = []
        for template_index, template in enumerate(templates):
            match = _match_component(g, component, template)
            if match is None:
                continue
            matches.append((template_index, match))

        assert len(matches) <= 1, "each workspace fragment must match at most one template"
        if not matches:
            continue

        _template_index, match = matches[0]
        top_r, top_c = match["top_left"]
        rotated = match["rotated"]
        for rr_raw, cc_raw in np.argwhere(rotated != BG):
            rr = int(rr_raw)
            cc = int(cc_raw)
            target = (top_r + rr, top_c + cc)
            value = int(rotated[rr, cc])
            assert planned.get(target, value) == value, "overlapping template stamps must agree"
            planned[target] = value

    out = g.copy()
    for (r, c), value in planned.items():
        out[r, c] = value
    return out


def _generate_template(size: int, existing: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(200):
        body, line, accent = random.sample(TEMPLATE_PALETTE, 3)
        template = np.full((size, size), int(body), dtype=int)

        row = random.randrange(1, size)
        col = random.randrange(1, size)
        template[row, :] = int(line)
        template[:, col] = int(line)

        block_h = random.randint(1, min(2, size - 1))
        block_w = random.randint(1, min(2, size - 1))
        block_r = random.randint(0, size - block_h)
        block_c = random.randint(0, size - block_w)
        template[block_r : block_r + block_h, block_c : block_c + block_w] = int(accent)

        if size >= 4 and random.random() < 0.75:
            extra_choices = [color for color in TEMPLATE_PALETTE if color not in (body, line, accent)]
            extra = random.choice(extra_choices)
            er = random.randrange(size)
            ec = random.randrange(size)
            template[er, ec] = int(extra)

        template[0, 0] = YELLOW
        if int(np.sum(template == YELLOW)) != 1:
            continue
        if len(set(map(int, template.ravel()))) < 4:
            continue
        if any(np.array_equal(template, other) for other in existing):
            continue
        return template
    return None


def _catalog_layout(templates: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    gap = random.randint(*CATALOG_GAP_RANGE)
    widths = [int(template.shape[1]) for template in templates]
    heights = [int(template.shape[0]) for template in templates]
    catalog_h = max(heights) + 2
    catalog_w = sum(widths) + gap * (len(templates) - 1) + 2
    catalog = np.full((catalog_h, catalog_w), CATALOG_BG, dtype=int)

    positions = []
    c = 1
    for template in templates:
        slack = catalog_h - 1 - int(template.shape[0])
        r = 1 if slack <= 1 else random.randint(1, slack - 1)
        positions.append((r, c))
        paste(catalog, template, r, c)
        c += int(template.shape[1]) + gap
    return catalog, positions


def _grow_fragment_mask(shape: tuple[int, int], turns: int, target: int) -> np.ndarray:
    h, w = map(int, shape)
    start = {
        0: (0, 0),
        1: (0, w - 1),
        2: (h - 1, w - 1),
        3: (h - 1, 0),
    }[int(turns)]

    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < int(target):
        candidates = []
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    candidates.append((rr, cc))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            break
        rr, cc = random.choice(candidates)
        mask[rr, cc] = True
        frontier.add((rr, cc))
    return mask


def _fragment_patch(template: np.ndarray, turns: int) -> tuple[np.ndarray, int]:
    rotated = _rotate_cw(template, turns)
    total = int(np.count_nonzero(rotated))
    target = random.randint(math.ceil(total * 0.5), total - 1)
    mask = _grow_fragment_mask(rotated.shape, turns, target)
    patch = np.where(mask, rotated, BG).astype(int)
    return patch, target


def _reserve_box(reserved: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    box = np.ones(shape, dtype=int)
    top_r, top_c = random_free_location(reserved, box, transparent=BG, padding=1)
    paste(reserved, box, top_r, top_c)
    return int(top_r), int(top_c)


def _distractor_patch() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        patch = np.zeros((h, w), dtype=int)
        start = (random.randrange(h), random.randrange(w))
        coords = {start}
        target = random.randint(max(2, (h * w) // 2), h * w)
        while len(coords) < target:
            r, c = random.choice(tuple(coords))
            choices = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    choices.append((rr, cc))
            coords.add(random.choice(choices))
        colors = random.sample([color for color in TEMPLATE_PALETTE if color != YELLOW], random.randint(2, 3))
        for r, c in coords:
            patch[r, c] = random.choice(colors)
        if np.count_nonzero(patch) >= 2:
            return patch
    raise RuntimeError("failed to build distractor")


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)

    templates: list[np.ndarray] = []
    sizes = [random.randint(*TEMPLATE_SIZE_RANGE) for _ in range(template_count)]
    for size in sizes:
        template = _generate_template(size, templates)
        if template is None:
            return None
        templates.append(template)

    catalog, _positions = _catalog_layout(templates)
    cat_h, cat_w = map(int, catalog.shape)
    if cat_h >= h - 4 or cat_w >= w - 4:
        return None

    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    paste(inp, catalog, 0, 0)
    paste(expected, catalog, 0, 0)

    reserved = np.zeros((h, w), dtype=int)
    paste(reserved, np.ones((cat_h, cat_w), dtype=int), 0, 0)

    turns_list = [random.randrange(4) for _ in range(template_count)]
    if all(turn == 0 for turn in turns_list):
        turns_list[random.randrange(template_count)] = random.choice((1, 2, 3))

    fragment_specs = []
    for template, turns in zip(templates, turns_list):
        rotated = _rotate_cw(template, turns)
        patch, visible_count = _fragment_patch(template, turns)
        top_r, top_c = _reserve_box(reserved, rotated.shape)
        paste(inp, patch, top_r, top_c, transparent=BG)
        paste(expected, rotated, top_r, top_c)
        fragment_specs.append(
            {
                "template": template,
                "rotated": rotated,
                "turns": turns,
                "top_left": (top_r, top_c),
                "visible": visible_count,
            }
        )

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    distractors = []
    for _ in range(distractor_count):
        patch = _distractor_patch()
        try:
            top_r, top_c = _reserve_box(reserved, patch.shape)
        except ValueError:
            return None
        paste(inp, patch, top_r, top_c, transparent=BG)
        paste(expected, patch, top_r, top_c, transparent=BG)
        distractors.append((patch, top_r, top_c))

    try:
        valid = _validate_instance(inp, expected, templates, fragment_specs, distractors)
    except (AssertionError, ValueError):
        return None
    if not valid:
        return None
    return inp, expected


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    templates: list[np.ndarray],
    fragment_specs: list[dict],
    distractors: list[tuple[np.ndarray, int, int]],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    catalog_bbox, extracted = _extract_templates(inp)
    if len(extracted) != len(templates):
        return False

    components = _workspace_components(inp, catalog_bbox)
    matched = []
    unmatched = []
    for component in components:
        hits = []
        for index, template in enumerate(extracted):
            match = _match_component(inp, component, template)
            if match is not None:
                hits.append((index, match))
        if len(hits) == 1:
            matched.append(hits[0])
        elif len(hits) == 0:
            unmatched.append(component)
        else:
            return False

    if len(matched) != len(fragment_specs):
        return False
    if len(unmatched) != len(distractors):
        return False
    if not any(int(spec["turns"]) != 0 for spec in fragment_specs):
        return False

    for spec in fragment_specs:
        total = int(np.count_nonzero(spec["rotated"]))
        if not (math.ceil(total * 0.5) <= int(spec["visible"]) < total):
            return False

    for component in unmatched:
        patch = crop_bbox(inp, component["bbox"])
        if np.any(patch == YELLOW):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid puzzle instance")
