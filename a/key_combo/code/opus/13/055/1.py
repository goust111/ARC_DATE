import random
from collections import Counter, defaultdict, deque

import numpy as np


BLUE = 1
GREEN = 3
BLACK = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = mask.astype(np.uint8)
    return arr.shape, tuple(int(x) for x in arr.ravel())


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        coords = np.asarray(cells, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        comps.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": local,
                "key": _mask_key(local),
                "area": int(local.sum()),
            }
        )
    return comps


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, -int(quarter_turns))


def _extract_template(grid: np.ndarray) -> dict:
    blue_components = _components_of_color(grid, BLUE)
    in_corner = []
    for comp in blue_components:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 < 4 and c0 < 4 and r1 < 4 and c1 < 4:
            in_corner.append(comp)
    assert len(in_corner) == 1, "expected exactly one blue template in the top-left 4x4"
    template = in_corner[0]
    assert template["mask"].shape == (3, 3), "template bbox must be exactly 3x3"
    assert 3 <= template["area"] <= 5, "template must have 3-5 pixels"
    for k in (1, 2, 3):
        assert not np.array_equal(template["mask"], _rotate_cw(template["mask"], k)), (
            "template must have no rotational symmetry"
        )
    all_blue = int(np.count_nonzero(grid == BLUE))
    assert all_blue == template["area"], "input should contain only the template in blue"
    return template


def _group_sprites(grid: np.ndarray) -> tuple[dict, list[dict]]:
    sprites = _components_of_color(grid, GREEN)
    assert 5 <= len(sprites) <= 8, "expected 5-8 green sprites"
    for comp in sprites:
        r0, c0, r1, c1 = comp["bbox"]
        assert 3 <= comp["area"] <= 6, "sprite area out of range"
        assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "sprite bbox exceeds 3x3"
        assert not (r0 < 4 and c0 < 4), "sprite must not occupy the template corner"
        assert r0 + 3 <= grid.shape[0] and c0 + 3 <= grid.shape[1], "stamp must fit in grid"
    grouped: dict = defaultdict(list)
    for comp in sprites:
        grouped[comp["key"]].append(comp)
    assert 2 <= len(grouped) <= 3, "expected 2-3 distinct sprite shapes"
    counts = [len(v) for v in grouped.values()]
    assert any(c == 1 for c in counts), "need at least one singleton group"
    assert any(c >= 2 for c in counts), "need at least one repeated group"
    assert len({c % 4 for c in counts}) >= 2, "need multiple rotation outcomes"
    areas_by_key = {key: group[0]["area"] for key, group in grouped.items()}
    shared_area_pairs = 0
    keys = list(grouped)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if areas_by_key[keys[i]] == areas_by_key[keys[j]]:
                shared_area_pairs += 1
    assert shared_area_pairs >= 1, "need at least one same-area different-shape pair"
    for i in range(len(sprites)):
        a = sprites[i]
        ar0, ac0, _ar1, _ac1 = a["bbox"]
        box_a = (ar0, ac0, ar0 + 2, ac0 + 2)
        for j in range(i + 1, len(sprites)):
            b = sprites[j]
            br0, bc0, _br1, _bc1 = b["bbox"]
            box_b = (br0, bc0, br0 + 2, bc0 + 2)
            assert _boxes_separated(box_a, box_b, gap=0), "stamp footprints must not overlap"
    return grouped, sprites


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    template = _extract_template(grid)
    grouped, _sprites = _group_sprites(grid)
    out = np.zeros_like(grid)
    for key, group in grouped.items():
        rot = len(group) % 4
        stamp = _rotate_cw(template["mask"], rot).astype(int) * BLUE
        for comp in group:
            r0, c0, _r1, _c1 = comp["bbox"]
            block = out[r0 : r0 + 3, c0 : c0 + 3]
            assert not np.any((block != 0) & (stamp != 0)), "stamps must not overlap"
            out[r0 : r0 + 3, c0 : c0 + 3] = stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each green sprite with a template stamp rotated by that sprite shape's global frequency.

    Input structure:
    - A black background contains one blue 4-connected template inside the top-left 4x4 area.
    - The template has an exact 3x3 bbox, 3-5 pixels, and no rotational symmetry.
    - Elsewhere there are 5-8 green 4-connected sprites, each 3-6 pixels with bbox at most 3x3.

    Rule:
    - Find the blue template and every green sprite, then normalize each sprite to its tight bbox mask.
    - Group sprites by exact normalized mask and count each group's frequency f.
    - For every sprite, rotate the 3x3 template clockwise by 90*(f mod 4) degrees and stamp that full 3x3 patch at the sprite bbox's top-left corner.
    - Erase the original template and all green sprites; template-zero cells in every stamp become black.

    Key insight:
    - A sprite's output orientation is not local: it is determined only after globally counting how many sprites share its exact shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    return _semantic_output(g)


def _enumerate_normalized_shapes() -> tuple[list[np.ndarray], list[np.ndarray]]:
    sprite_masks: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    template_masks: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for bits in range(1 << 9):
        raw = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                raw[idx // 3, idx % 3] = True
        area = int(raw.sum())
        if not (3 <= area <= 6):
            continue
        crop = _crop_mask(raw)
        if crop.size == 0 or not _is_connected(crop):
            continue
        key = _mask_key(crop)
        sprite_masks.setdefault(key, crop)
        if crop.shape == (3, 3) and area <= 5:
            if all(not np.array_equal(crop, _rotate_cw(crop, k)) for k in (1, 2, 3)):
                template_masks.setdefault(key, crop)
    return list(sprite_masks.values()), list(template_masks.values())


SPRITE_CATALOG, TEMPLATE_CATALOG = _enumerate_normalized_shapes()
SPRITE_BY_KEY = {_mask_key(mask): mask for mask in SPRITE_CATALOG}
SPRITE_BY_AREA: dict[int, list[np.ndarray]] = defaultdict(list)
ROTATED_VARIANTS: dict[tuple[tuple[int, int], tuple[int, ...]], list[np.ndarray]] = defaultdict(list)
for mask in SPRITE_CATALOG:
    SPRITE_BY_AREA[int(mask.sum())].append(mask)
for mask in SPRITE_CATALOG:
    key = _mask_key(mask)
    seen = set()
    for k in (1, 2, 3):
        rot = _crop_mask(_rotate_cw(mask, k))
        rot_key = _mask_key(rot)
        if rot_key != key and rot_key in SPRITE_BY_KEY and rot_key not in seen:
            ROTATED_VARIANTS[key].append(SPRITE_BY_KEY[rot_key])
            seen.add(rot_key)


COUNT_PATTERNS = [
    (1, 4),
    (1, 1, 3),
    (1, 2, 2),
    (1, 2, 3),
    (1, 1, 4),
    (1, 2, 4),
    (1, 3, 3),
    (1, 3, 4),
]
GRID_HEIGHT_RANGE = (12, 16)
GRID_WIDTH_RANGE = (15, 20)
GEN_RETRY_LIMIT = 400
ROTATION_PAIR_PROB = 0.45


def _choose_same_area_pair() -> list[np.ndarray]:
    areas = [area for area, masks in SPRITE_BY_AREA.items() if len(masks) >= 2]
    area = random.choice(areas)
    if random.random() < ROTATION_PAIR_PROB:
        random.shuffle(SPRITE_BY_AREA[area])
        for base in SPRITE_BY_AREA[area]:
            mates = ROTATED_VARIANTS.get(_mask_key(base), [])
            mates = [m for m in mates if int(m.sum()) == area]
            if mates:
                return [base, random.choice(mates)]
    return random.sample(SPRITE_BY_AREA[area], 2)


def _choose_shapes(n_types: int) -> list[np.ndarray]:
    chosen = _choose_same_area_pair()
    used = {_mask_key(mask) for mask in chosen}
    while len(chosen) < n_types:
        candidate = random.choice(SPRITE_CATALOG)
        key = _mask_key(candidate)
        if key in used:
            continue
        chosen.append(candidate)
        used.add(key)
    random.shuffle(chosen)
    return chosen


def _place_template(grid: np.ndarray, template: np.ndarray) -> None:
    r0, c0 = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    patch = template.astype(int) * BLUE
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch


def _random_sprite_items() -> list[dict]:
    counts = list(random.choice(COUNT_PATTERNS))
    shapes = _choose_shapes(len(counts))
    items = []
    pairs = list(zip(shapes, counts))
    random.shuffle(pairs)
    for mask, count in pairs:
        for _ in range(count):
            items.append({"mask": mask.copy(), "key": _mask_key(mask)})
    random.shuffle(items)
    return items


def _valid_footprint(top_left: tuple[int, int], occupied: list[tuple[int, int, int, int]]) -> bool:
    r0, c0 = top_left
    box = (r0, c0, r0 + 2, c0 + 2)
    if not _boxes_separated(box, (0, 0, 3, 3), gap=1):
        return False
    return all(_boxes_separated(box, other, gap=1) for other in occupied)


def _place_sprites(grid: np.ndarray, items: list[dict]) -> bool:
    h, w = grid.shape
    occupied: list[tuple[int, int, int, int]] = []
    order = sorted(items, key=lambda item: item["mask"].sum(), reverse=True)
    for item in order:
        mask = item["mask"]
        sh, sw = mask.shape
        candidates = []
        for r0 in range(h - 2):
            for c0 in range(w - 2):
                if not _valid_footprint((r0, c0), occupied):
                    continue
                if np.any(grid[r0 : r0 + sh, c0 : c0 + sw] != BLACK):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + sh, c0 : c0 + sw][mask] = GREEN
        item["placed_bbox"] = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        occupied.append((r0, c0, r0 + 2, c0 + 2))
    return True


def _validate_example(grid: np.ndarray) -> np.ndarray | None:
    try:
        expected = _semantic_output(grid)
    except AssertionError:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    groups, sprites = _group_sprites(grid)
    template = _extract_template(grid)
    for key, group in groups.items():
        rot = len(group) % 4
        stamp = _rotate_cw(template["mask"], rot).astype(int) * BLUE
        for comp in group:
            r0, c0, _r1, _c1 = comp["bbox"]
            if not np.array_equal(expected[r0 : r0 + 3, c0 : c0 + 3], stamp):
                return None
    green_cells = int(np.count_nonzero(grid == GREEN))
    if green_cells <= 0 or int(np.count_nonzero(expected == BLUE)) <= 0:
        return None
    key_counts = Counter(comp["key"] for comp in sprites)
    if len(key_counts) == 2 and len(set(c % 4 for c in key_counts.values())) < 2:
        return None
    return expected


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)
    template = random.choice(TEMPLATE_CATALOG)
    _place_template(grid, template)
    items = _random_sprite_items()
    if not _place_sprites(grid, items):
        return None
    return grid if _validate_example(grid) is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
