import random
from collections import Counter

import numpy as np


DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

RETRY_LIMIT = 400
GRID_SIDE_RANGE = (14, 30)
TEMPLATE_HEIGHT_RANGE = (4, 7)
TEMPLATE_WIDTH_RANGE = (4, 7)
MARKER_COUNT_VALUES = [2, 3, 4]
MARKER_COUNT_WEIGHTS = [2, 3, 5]
MIN_BODY_AREA = 6
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 300
GROWTH_TRIES = 80
COLOR_POOL = list(range(10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = DIR8

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _tight_non_bg_crop(patch: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(patch != bg)
    return _crop_bbox(patch, _bbox(coords))


def _corner_dir(key_pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    kr, kc = key_pos
    h, w = shape
    assert kr in (0, h - 1) and kc in (0, w - 1)
    return (1 if kr == 0 else -1, 1 if kc == 0 else -1)


def _rotations(template: np.ndarray, key_color: int, bg: int) -> list[dict]:
    out = []
    for k in range(4):
        rotated = _tight_non_bg_crop(np.rot90(template, k=k), bg)
        key_locs = np.argwhere(rotated == key_color)
        assert key_locs.shape == (1, 2)
        key_pos = tuple(map(int, key_locs[0]))
        out.append(
            {
                "grid": rotated,
                "key_pos": key_pos,
                "dir": _corner_dir(key_pos, rotated.shape),
            }
        )
    return out


def _find_template(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for coords in _components(grid != bg, connectivity=8):
        if coords.shape[0] <= 2:
            continue
        bbox = _bbox(coords)
        patch = _crop_bbox(grid, bbox)
        colors = [int(grid[r, c]) for r, c in coords]
        counts = Counter(colors)
        if len(counts) != 2:
            continue
        singleton_colors = [color for color, count in counts.items() if count == 1]
        if len(singleton_colors) != 1:
            continue
        key_color = int(singleton_colors[0])
        body_color = int(next(color for color in counts if color != key_color))
        key_global = tuple(map(int, coords[np.array(colors) == key_color][0]))
        r0, c0, r1, c1 = bbox
        if key_global not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        candidates.append(
            {
                "bbox": bbox,
                "template": patch,
                "key_color": key_color,
                "body_color": body_color,
                "area": int(coords.shape[0]),
            }
        )

    assert len(candidates) == 1, f"expected one template, found {len(candidates)}"
    return candidates[0]


def _find_markers(grid: np.ndarray, bg: int, key_color: int, body_color: int) -> list[dict]:
    markers = []
    for coords in _components(grid != bg, connectivity=8):
        if coords.shape[0] != 2:
            continue
        (r1, c1), (r2, c2) = map(tuple, coords.tolist())
        if abs(r1 - r2) != 1 or abs(c1 - c2) != 1:
            continue
        colors = [int(grid[r1, c1]), int(grid[r2, c2])]
        if Counter(colors) != Counter({key_color: 1, body_color: 1}):
            continue
        if int(grid[r1, c1]) == key_color:
            anchor, pointer = (r1, c1), (r2, c2)
        else:
            anchor, pointer = (r2, c2), (r1, c1)
        markers.append(
            {
                "anchor": anchor,
                "direction": (pointer[0] - anchor[0], pointer[1] - anchor[1]),
            }
        )
    markers.sort(key=lambda item: (item["anchor"][0], item["anchor"][1]))
    assert markers, "expected at least one marker"
    return markers


def _solve_layout(grid: np.ndarray) -> tuple[np.ndarray, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    template_info = _find_template(g, bg)
    template = template_info["template"]
    key_color = template_info["key_color"]
    body_color = template_info["body_color"]
    variants = _rotations(template, key_color, bg)
    markers = _find_markers(g, bg, key_color, body_color)

    out = g.copy()
    for marker in markers:
        matches = [variant for variant in variants if variant["dir"] == marker["direction"]]
        assert len(matches) == 1, "marker orientation must match exactly one rotation"
        variant = matches[0]
        kr, kc = variant["key_pos"]
        top = marker["anchor"][0] - kr
        left = marker["anchor"][1] - kc
        h, w = variant["grid"].shape
        assert 0 <= top and 0 <= left and top + h <= g.shape[0] and left + w <= g.shape[1]
        _paste_non_bg(out, variant["grid"], top, left, bg)

    return out, template_info, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the keyed template to each diagonal marker orientation and stamp it with the key on the anchor.

    Input structure:
    - The grid has a dominant background, one complete two-color template sprite, and several separate two-pixel markers.
    - Inside the template, exactly one corner pixel has the key color and all other template pixels use the body color.
    - Each marker is a diagonal pair made of one key-colored anchor pixel and one body-colored pointing pixel.

    Rule:
    - Extract the unique full template that contains the corner key pixel.
    - For every marker, rotate the template by 0/90/180/270 so the template key lands on the anchor and the template opens toward the pointing pixel's diagonal quadrant.
    - Stamp the full rotated template at that location, absorbing the original two marker pixels, while keeping the source template unchanged.

    Key insight:
    - The marker's key-colored pixel fixes translation and the diagonally adjacent body-colored pixel fixes which keyed corner orientation to use.
    """
    return _solve_layout(grid)[0]


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    )


def _sample_template() -> np.ndarray | None:
    h = random.randint(*TEMPLATE_HEIGHT_RANGE)
    w = random.randint(*TEMPLATE_WIDTH_RANGE)
    target_area = random.randint(MIN_BODY_AREA, max(MIN_BODY_AREA, (h * w) // 2 + 2))

    mask = np.zeros((h, w), dtype=bool)
    mask[1, 1] = True
    while int(mask.sum()) < target_area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) != (0, 0) and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True

    for goal in [(h - 1, random.randint(1, w - 1)), (random.randint(1, h - 1), w - 1)]:
        gr, gc = goal
        if mask[gr, gc]:
            continue
        rr, cc = 1, 1
        while rr != gr or cc != gc:
            step_options = []
            if rr < gr:
                step_options.append((rr + 1, cc))
            elif rr > gr:
                step_options.append((rr - 1, cc))
            if cc < gc:
                step_options.append((rr, cc + 1))
            elif cc > gc:
                step_options.append((rr, cc - 1))
            rr, cc = random.choice(step_options)
            if (rr, cc) != (0, 0):
                mask[rr, cc] = True

    if not mask[h - 1].any() or not mask[:, w - 1].any():
        return None

    template = np.zeros((h, w), dtype=int)
    template[mask] = 1
    template[0, 0] = 2

    if template[1, 1] != 1:
        return None
    if any(np.array_equal(template, np.rot90(template, k=k)) for k in (1, 2, 3)):
        return None
    return template


def _place_boxes(shapes: list[tuple[int, int]], height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(shapes)
    for idx in order:
        h, w = shapes[idx]
        if height - h - PLACEMENT_MARGIN < PLACEMENT_MARGIN or width - w - PLACEMENT_MARGIN < PLACEMENT_MARGIN:
            return None
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(PLACEMENT_MARGIN, height - h - PLACEMENT_MARGIN)
            left = random.randint(PLACEMENT_MARGIN, width - w - PLACEMENT_MARGIN)
            candidate = (top, left, top + h - 1, left + w - 1)
            if all(
                existing is None or not _boxes_too_close(candidate, existing, PLACEMENT_GAP)
                for existing in boxes
            ):
                boxes[idx] = candidate
                break
        else:
            return None
    return [box for box in boxes if box is not None]


def _render_expected(
    inp: np.ndarray,
    template: np.ndarray,
    variants: list[dict],
    marker_specs: list[dict],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for spec in marker_specs:
        variant = variants[spec["orientation"]]
        top, left = spec["bbox"][:2]
        _paste_non_bg(out, variant["grid"], top, left, bg)
    return out


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    template: np.ndarray,
    variants: list[dict],
    marker_specs: list[dict],
    bg: int,
    key_color: int,
    body_color: int,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    try:
        solved, template_info, markers = _solve_layout(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False
    if template_info["key_color"] != key_color or template_info["body_color"] != body_color:
        return False
    if len(markers) != len(marker_specs):
        return False
    if len({spec["orientation"] for spec in marker_specs}) < 2:
        return False

    used = set()
    for spec in marker_specs:
        variant = variants[spec["orientation"]]
        key_r, key_c = variant["key_pos"]
        anchor = (spec["bbox"][0] + key_r, spec["bbox"][1] + key_c)
        pointer = (anchor[0] + variant["dir"][0], anchor[1] + variant["dir"][1])
        used.add(anchor)
        if int(inp[anchor]) != key_color or int(inp[pointer]) != body_color:
            return False
        rel_r = key_r + variant["dir"][0]
        rel_c = key_c + variant["dir"][1]
        if int(variant["grid"][rel_r, rel_c]) != body_color:
            return False

    if len(used) != len(marker_specs):
        return False

    component_sizes = sorted(int(comp.shape[0]) for comp in _components(inp != bg, connectivity=8))
    if component_sizes != sorted([int(np.count_nonzero(template != bg))] + [2] * len(marker_specs)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    for _ in range(GROWTH_TRIES):
        canonical = _sample_template()
        if canonical is not None:
            break
    else:
        return None

    bg, body_color, key_color = random.sample(COLOR_POOL, 3)
    template = np.full(canonical.shape, bg, dtype=int)
    template[canonical == 1] = body_color
    template[canonical == 2] = key_color

    original_rotation = random.randrange(4)
    source_template = np.rot90(template, k=original_rotation)
    source_template = _tight_non_bg_crop(source_template, bg)
    variants = _rotations(source_template, key_color, bg)

    marker_count = random.choices(MARKER_COUNT_VALUES, weights=MARKER_COUNT_WEIGHTS, k=1)[0]
    orientations = random.sample(range(4), k=marker_count)

    min_h = max(source_template.shape[0], max(variant["grid"].shape[0] for variant in variants)) + 4
    min_w = max(source_template.shape[1], max(variant["grid"].shape[1] for variant in variants)) + 4
    height = random.randint(max(GRID_SIDE_RANGE[0], min_h), GRID_SIDE_RANGE[1])
    width = random.randint(max(GRID_SIDE_RANGE[0], min_w), GRID_SIDE_RANGE[1])

    shapes = [source_template.shape] + [variants[idx]["grid"].shape for idx in orientations]
    boxes = _place_boxes(shapes, height, width)
    if boxes is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    source_box = boxes[0]
    _paste_non_bg(grid, source_template, source_box[0], source_box[1], bg)

    marker_specs = []
    for bbox, orientation in zip(boxes[1:], orientations):
        variant = variants[orientation]
        kr, kc = variant["key_pos"]
        dr, dc = variant["dir"]
        anchor = (bbox[0] + kr, bbox[1] + kc)
        pointer = (anchor[0] + dr, anchor[1] + dc)
        if int(variant["grid"][kr + dr, kc + dc]) != body_color:
            return None
        grid[anchor] = key_color
        grid[pointer] = body_color
        marker_specs.append({"bbox": bbox, "orientation": orientation})

    expected = _render_expected(grid, source_template, variants, marker_specs, bg)
    if not _validate_scene(grid, expected, source_template, variants, marker_specs, bg, key_color, body_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid posture-stamp input")
