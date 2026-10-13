import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    {"name": "dot", "mask": _mask(["1"]), "placements": ("top", "bottom", "left", "right")},
    {"name": "h2", "mask": _mask(["11"]), "placements": ("top", "bottom")},
    {"name": "h3", "mask": _mask(["111"]), "placements": ("top", "bottom")},
    {"name": "h4", "mask": _mask(["1111"]), "placements": ("top", "bottom")},
    {"name": "v2", "mask": _mask(["1", "1"]), "placements": ("left", "right")},
    {"name": "v3", "mask": _mask(["1", "1", "1"]), "placements": ("left", "right")},
    {"name": "v4", "mask": _mask(["1", "1", "1", "1"]), "placements": ("left", "right")},
    {"name": "l3_tl", "mask": _mask(["11", "10"]), "placements": ("tl",)},
    {"name": "l3_tr", "mask": _mask(["11", "01"]), "placements": ("tr",)},
    {"name": "l3_bl", "mask": _mask(["10", "11"]), "placements": ("bl",)},
    {"name": "l3_br", "mask": _mask(["01", "11"]), "placements": ("br",)},
    {"name": "l4_tl_w", "mask": _mask(["111", "100"]), "placements": ("tl",)},
    {"name": "l4_tr_w", "mask": _mask(["111", "001"]), "placements": ("tr",)},
    {"name": "l4_bl_w", "mask": _mask(["100", "111"]), "placements": ("bl",)},
    {"name": "l4_br_w", "mask": _mask(["001", "111"]), "placements": ("br",)},
    {"name": "l4_tl_t", "mask": _mask(["11", "10", "10"]), "placements": ("tl",)},
    {"name": "l4_tr_t", "mask": _mask(["11", "01", "01"]), "placements": ("tr",)},
    {"name": "l4_bl_t", "mask": _mask(["10", "10", "11"]), "placements": ("bl",)},
    {"name": "l4_br_t", "mask": _mask(["01", "01", "11"]), "placements": ("br",)},
]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    cropped = _crop_mask(mask)
    return cropped.shape[0], cropped.shape[1], tuple(cropped.astype(int).ravel())


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                key = (nr, nc)
                if key not in seen:
                    seen.add(key)
                    q.append(key)
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                }
            )
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = component["coords"] - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _render_component(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    shifted = component["coords"] - np.array([r0, c0], dtype=int)
    out[shifted[:, 0], shifted[:, 1]] = int(component["color"])
    return out


def _extract_box_info(component: dict) -> dict | None:
    mask = _component_mask(component)
    h, w = mask.shape
    if h < 4 or w < 4:
        return None

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None
    if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None

    gap = border & ~mask
    if not gap.any() or not _mask_connected(gap):
        return None

    return {
        "bbox": component["bbox"],
        "color": int(component["color"]),
        "gap_mask": _crop_mask(gap),
        "gap_sig": _shape_signature(gap),
        "content": None,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique key-to-gap chain through hollow lock boxes and crop the final unmatched key.

    Input structure:
    - The background is black.
    - Some monochrome components are hollow rectangular box walls with exactly one connected gap on their border.
    - Every box contains one separate monochrome key-shape inside its interior, and exactly one extra key-shape lies outside all boxes.

    Rule:
    - Normalize each gap by cropping its missing border cells to the smallest bounding box.
    - Start from the only component that is outside every box interior, and repeatedly match its normalized shape to the unique unopened box gap with the same shape.
    - Replace the current key with that box's interior content and stop when no unopened gap matches; output that final key cropped to its own bounding box.

    Key insight:
    - Many terminal-looking boxed contents are distractors, so the answer is not "any unmatched shape" but the endpoint reached by starting from the lone free key and following the forced shape-matching chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    components = _find_components(g, bg)
    boxes = []
    box_by_component: dict[int, dict] = {}
    for component in components:
        box_info = _extract_box_info(component)
        if box_info is not None:
            boxes.append(box_info)
            box_by_component[id(component)] = box_info

    assert boxes, "expected at least one box"
    boxes.sort(key=lambda box: box["bbox"])

    free_candidates = []
    for component in components:
        if id(component) in box_by_component:
            continue
        inside = []
        for box in boxes:
            r0, c0, r1, c1 = box["bbox"]
            coords = component["coords"]
            if np.all((coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)):
                inside.append(box)
        assert len(inside) <= 1, "component cannot belong to multiple boxes"
        if inside:
            box = inside[0]
            assert box["content"] is None, "expected exactly one content per box"
            box["content"] = component
        else:
            free_candidates.append(component)

    assert len(free_candidates) == 1, "expected exactly one free starting key"
    assert all(box["content"] is not None for box in boxes), "every box must contain one key"

    current = free_candidates[0]
    unopened = list(boxes)
    while True:
        current_sig = _shape_signature(_component_mask(current))
        matches = [box for box in unopened if box["gap_sig"] == current_sig]
        assert len(matches) <= 1, "each key should match at most one gap"
        if not matches:
            return _render_component(current, bg)
        box = matches[0]
        unopened.remove(box)
        current = box["content"]


def _render_shape(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full(mask.shape, BLACK, dtype=int)
    out[mask] = int(color)
    return out


def _choose_distinct_templates(count: int, excluded: set[tuple[int, int, tuple[int, ...]]] | None = None) -> list[dict]:
    excluded = set() if excluded is None else set(excluded)
    candidates = [tpl for tpl in SHAPE_LIBRARY if _shape_signature(tpl["mask"]) not in excluded]
    return random.sample(candidates, count)


def _gap_offset(mask: np.ndarray, placement: str, box_h: int, box_w: int) -> tuple[int, int] | None:
    h, w = mask.shape
    if placement == "top":
        if box_w < w + 2:
            return None
        return 0, random.randint(1, box_w - w - 1)
    if placement == "bottom":
        if box_w < w + 2:
            return None
        return box_h - h, random.randint(1, box_w - w - 1)
    if placement == "left":
        if box_h < h + 2:
            return None
        return random.randint(1, box_h - h - 1), 0
    if placement == "right":
        if box_h < h + 2:
            return None
        return random.randint(1, box_h - h - 1), box_w - w
    if placement == "tl":
        if box_h <= h or box_w <= w:
            return None
        return 0, 0
    if placement == "tr":
        if box_h <= h or box_w <= w:
            return None
        return 0, box_w - w
    if placement == "bl":
        if box_h <= h or box_w <= w:
            return None
        return box_h - h, 0
    if placement == "br":
        if box_h <= h or box_w <= w:
            return None
        return box_h - h, box_w - w
    raise ValueError(f"unknown placement {placement}")


def _build_box_patch(
    gap_template: dict,
    content_template: dict,
    wall_color: int,
    content_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    gap_mask = gap_template["mask"]
    content_mask = content_template["mask"]
    gap_h, gap_w = gap_mask.shape
    content_h, content_w = content_mask.shape

    min_h = max(5, content_h + 4, gap_h + 2)
    min_w = max(5, content_w + 4, gap_w + 2)
    if min_h > 10 or min_w > 10:
        return None
    box_h = random.randint(min_h, min(10, min_h + 2))
    box_w = random.randint(min_w, min(10, min_w + 2))

    patch = np.full((box_h, box_w), BLACK, dtype=int)
    patch[0, :] = wall_color
    patch[-1, :] = wall_color
    patch[:, 0] = wall_color
    patch[:, -1] = wall_color

    placement = random.choice(gap_template["placements"])
    offset = _gap_offset(gap_mask, placement, box_h, box_w)
    if offset is None:
        return None
    gr, gc = offset
    patch[gr : gr + gap_h, gc : gc + gap_w][gap_mask] = BLACK

    if not _mask_connected(patch == wall_color):
        return None

    r_min = 2
    c_min = 2
    r_max = box_h - content_h - 2
    c_max = box_w - content_w - 2
    if r_max < r_min or c_max < c_min:
        return None
    cr = random.randint(r_min, r_max)
    cc = random.randint(c_min, c_max)
    patch[cr : cr + content_h, cc : cc + content_w][content_mask] = content_color

    return patch, _render_shape(content_mask, content_color)


def _reserve(blocked: np.ndarray, r: int, c: int, h: int, w: int, padding: int) -> None:
    r0 = max(0, r - padding)
    c0 = max(0, c - padding)
    r1 = min(blocked.shape[0], r + h + padding)
    c1 = min(blocked.shape[1], c + w + padding)
    blocked[r0:r1, c0:c1] = True


def _try_place_scene(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    base = int(np.ceil(np.sqrt(total_area * 1.9)))
    min_side = max(18, max_h + 4, max_w + 4, base)
    if min_side > 30:
        return None

    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = patch.shape
        placed = False
        for _ in range(600):
            r = random.randint(1, h - ph - 1)
            c = random.randint(1, w - pw - 1)
            if blocked[r : r + ph, c : c + pw].any():
                continue
            patch_mask = patch != BLACK
            grid[r : r + ph, c : c + pw][patch_mask] = patch[patch_mask]
            _reserve(blocked, r, c, ph, pw, padding=1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    chain_len = random.randint(2, 4)
    distractor_count = random.randint(1, 2)
    box_count = chain_len + distractor_count

    gap_templates = _choose_distinct_templates(box_count)
    gap_sigs = {_shape_signature(tpl["mask"]) for tpl in gap_templates}
    terminal_templates = _choose_distinct_templates(distractor_count + 1, excluded=gap_sigs)

    wall_colors = random.sample(PALETTE, box_count)
    free_key_color = random.choice(PALETTE)

    content_templates = gap_templates[1:chain_len] + [terminal_templates[0]] + terminal_templates[1:]

    patches = [_render_shape(gap_templates[0]["mask"], free_key_color)]
    final_expected = None
    for i in range(box_count):
        wall_color = wall_colors[i]
        forbidden = {wall_color}
        if i == 0:
            forbidden.add(free_key_color)
        color_choices = [color for color in PALETTE if color not in forbidden]
        content_color = random.choice(color_choices)
        built = _build_box_patch(gap_templates[i], content_templates[i], wall_color, content_color)
        if built is None:
            return None
        patch, expected_content = built
        patches.append(patch)
        if i == chain_len - 1:
            final_expected = expected_content

    assert final_expected is not None
    grid = _try_place_scene(patches)
    if grid is None:
        return None
    return grid, final_expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _generate_scene()
        if generated is None:
            continue
        grid, expected = generated
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError("generate_input failed to build a valid key-chain puzzle")
