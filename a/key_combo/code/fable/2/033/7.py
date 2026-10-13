from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
LEGEND_SLOT = 5
PANEL_GAP = 1
PANEL_BORDER = 1
GRID_MIN = 23
GRID_MAX = 30
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    pts = np.asarray(coords, dtype=int)
    assert pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] > 0
    r0, c0, r1, c1 = bbox_coords(pts)
    shifted = pts - np.array([r0, c0], dtype=int)
    key = tuple(sorted((int(r), int(c)) for r, c in shifted))
    return key, (int(r1 - r0 + 1), int(c1 - c0 + 1))


def _transform_key(
    key: tuple[tuple[int, int], ...], dims: tuple[int, int], rotation: int, flip: bool
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    h, w = dims
    out: list[tuple[int, int]] = []
    for r, c in key:
        if rotation == 0:
            rr, cc = r, c
            h2, w2 = h, w
        elif rotation == 1:
            rr, cc = w - 1 - c, r
            h2, w2 = w, h
        elif rotation == 2:
            rr, cc = h - 1 - r, w - 1 - c
            h2, w2 = h, w
        else:
            rr, cc = c, h - 1 - r
            h2, w2 = w, h
        if flip:
            cc = w2 - 1 - cc
        out.append((int(rr), int(cc)))
    return _normalize_coords(out)


def _canonical_key(key: tuple[tuple[int, int], ...], dims: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    variants = []
    for rotation in range(4):
        for flip in (False, True):
            transformed, _ = _transform_key(key, dims, rotation, flip)
            variants.append(transformed)
    return min(variants)


def _mask_from_key(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in key) + 1
    w = max(c for _, c in key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[r, c] = True
    return mask


def _all_variant_masks(mask: np.ndarray) -> list[np.ndarray]:
    key, dims = _normalize_coords(np.argwhere(mask))
    variants: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for rotation in range(4):
        for flip in (False, True):
            transformed, _ = _transform_key(key, dims, rotation, flip)
            variants.setdefault(transformed, _mask_from_key(transformed))
    return list(variants.values())


def _corner_panel_bbox(
    grid: np.ndarray, corner: str, color: int
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    if corner == "tl":
        width = 0
        while width < w and int(grid[0, width]) == color:
            width += 1
        height = 0
        while height < h and int(grid[height, 0]) == color:
            height += 1
        bbox = (0, 0, height - 1, width - 1)
    elif corner == "tr":
        width = 0
        while width < w and int(grid[0, w - 1 - width]) == color:
            width += 1
        height = 0
        while height < h and int(grid[height, w - 1]) == color:
            height += 1
        bbox = (0, w - width, height - 1, w - 1)
    elif corner == "bl":
        width = 0
        while width < w and int(grid[h - 1, width]) == color:
            width += 1
        height = 0
        while height < h and int(grid[h - 1 - height, 0]) == color:
            height += 1
        bbox = (h - height, 0, h - 1, width - 1)
    else:
        width = 0
        while width < w and int(grid[h - 1, w - 1 - width]) == color:
            width += 1
        height = 0
        while height < h and int(grid[h - 1 - height, w - 1]) == color:
            height += 1
        bbox = (h - height, w - width, h - 1, w - 1)

    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 < r0 or c1 < c0:
        return None
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(region[0, :] == color):
        return None
    if not np.all(region[-1, :] == color):
        return None
    if not np.all(region[:, 0] == color):
        return None
    if not np.all(region[:, -1] == color):
        return None
    if not np.any(region != color):
        return None
    return bbox


def _detect_panel(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for corner, (r, c) in {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }.items():
        color = int(grid[r, c])
        if color == bg:
            continue
        bbox = _corner_panel_bbox(grid, corner, color)
        if bbox is not None:
            candidates.append((bbox, color))
    assert len(candidates) == 1, "expected one corner legend panel"
    return candidates[0]


def _extract_legend(
    grid: np.ndarray, panel_bbox: tuple[int, int, int, int], panel_color: int, bg: int
) -> list[dict]:
    panel = crop_bbox(grid, panel_bbox)
    colors = sorted(set(map(int, panel.ravel())) - {int(panel_color)})
    assert len(colors) >= 3, "expected several legend entries"
    entries = []
    seen_canonical = set()
    for color in colors:
        comps = components_mask(panel == color)
        assert len(comps) == 1, "legend color should form one component"
        key, dims = _normalize_coords(comps[0])
        canon = _canonical_key(key, dims)
        assert canon not in seen_canonical, "legend classes must be distinct up to D4"
        seen_canonical.add(canon)
        coords = np.asarray(key, dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in key:
            patch[r - r0, c - c0] = int(color)
        entries.append(
            {
                "color": int(color),
                "area": int(len(key)),
                "key": key,
                "dims": dims,
                "canonical": canon,
                "patch": patch,
            }
        )
    return entries


def _extract_gray_sprites(
    grid: np.ndarray, panel_bbox: tuple[int, int, int, int], bg: int
) -> tuple[int, list[dict]]:
    work = grid.copy()
    r0, c0, r1, c1 = panel_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    colors = sorted(set(map(int, work.ravel())) - {int(bg)})
    assert len(colors) == 1, "expected only gray sprites outside the panel"
    sprite_color = int(colors[0])
    comps = components_mask(work == sprite_color)
    assert comps, "expected gray sprites"
    sprites = []
    for comp in comps:
        key, dims = _normalize_coords(comp)
        sprites.append(
            {
                "area": int(len(key)),
                "pose": key,
                "dims": dims,
                "canonical": _canonical_key(key, dims),
            }
        )
    return sprite_color, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count gray sprites by legend class up to rotation/reflection and return the legend copy whose vote count equals its area.

    Input structure:
    - A dominant background surrounds a single corner legend panel with its own solid panel color on the border.
    - Inside the panel, several colored 4-connected legend shapes sit on the panel color and define the classes.
    - Outside the panel, many separated monochrome gray sprites appear on the background in mixed rotations and reflections.

    Rule:
    - Detect the corner panel from the unique non-background corner block.
    - Normalize each legend shape and each gray sprite by rotation/reflection equivalence, then count gray sprites per legend class.
    - Find the unique legend class whose gray-sprite count is exactly equal to that legend shape's own cell count.
    - Output that legend shape in its panel orientation and legend color, tight-cropped on the main background.

    Key insight:
    - The winning class is not the most frequent one; you must first merge differently oriented gray copies into the same class, then test the self-referential condition count = area, and finally return the panel's original colored orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    panel_bbox, panel_color = _detect_panel(g, bg)
    legend = _extract_legend(g, panel_bbox, panel_color, bg)
    _sprite_color, sprites = _extract_gray_sprites(g, panel_bbox, bg)

    counts = Counter(sprite["canonical"] for sprite in sprites)
    winners = [entry for entry in legend if counts[entry["canonical"]] == entry["area"]]
    assert len(winners) == 1, "expected a unique vote=area legend class"
    return winners[0]["patch"]


SHAPE_LIBRARY = {
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
    ],
    5: [
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    ],
}


def _layout_options(n: int) -> list[tuple[int, int]]:
    options = []
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        if rows * cols >= n:
            options.append((rows, cols))
    return [rc for rc in options if max(rc) <= 4]


def _panel_size(rows: int, cols: int) -> tuple[int, int]:
    height = PANEL_BORDER * 2 + rows * LEGEND_SLOT + (rows - 1) * PANEL_GAP
    width = PANEL_BORDER * 2 + cols * LEGEND_SLOT + (cols - 1) * PANEL_GAP
    return height, width


def _choose_shapes(rng: random.Random) -> tuple[list[np.ndarray], int]:
    target_area = rng.choice((3, 4, 5))
    target_choices = SHAPE_LIBRARY[target_area]
    target_idx, twin_idx = rng.sample(range(len(target_choices)), 2)
    selected = [target_choices[target_idx], target_choices[twin_idx]]

    legend_count = rng.choice((4, 5))
    while len(selected) < legend_count:
        area = rng.choice(sorted(SHAPE_LIBRARY))
        mask = rng.choice(SHAPE_LIBRARY[area])
        key, dims = _normalize_coords(np.argwhere(mask))
        canon = _canonical_key(key, dims)
        if any(_canonical_key(*_normalize_coords(np.argwhere(existing))) == canon for existing in selected):
            continue
        selected.append(mask)
    rng.shuffle(selected)
    target_mask = target_choices[target_idx]
    for idx, mask in enumerate(selected):
        if np.array_equal(mask, target_mask):
            return selected, idx
    raise AssertionError("target mask not found")


def _variant_pool(mask: np.ndarray) -> list[np.ndarray]:
    return _all_variant_masks(mask)


def _build_panel(
    rng: random.Random, bg: int, panel_color: int, legend_colors: list[int], masks: list[np.ndarray]
) -> tuple[np.ndarray, list[dict]]:
    rows, cols = rng.choice(_layout_options(len(masks)))
    panel_h, panel_w = _panel_size(rows, cols)
    panel = np.full((panel_h, panel_w), int(panel_color), dtype=int)
    order = list(range(len(masks)))
    rng.shuffle(order)
    legend_entries: list[dict] = [None] * len(masks)  # type: ignore[assignment]
    for slot_id, mask_id in enumerate(order):
        mask = masks[mask_id]
        variants = _variant_pool(mask)
        placed_mask = variants[rng.randrange(len(variants))]
        mh, mw = placed_mask.shape
        slot_r = slot_id // cols
        slot_c = slot_id % cols
        sr0 = PANEL_BORDER + slot_r * (LEGEND_SLOT + PANEL_GAP)
        sc0 = PANEL_BORDER + slot_c * (LEGEND_SLOT + PANEL_GAP)
        rr = sr0 + rng.randint(0, LEGEND_SLOT - mh)
        cc = sc0 + rng.randint(0, LEGEND_SLOT - mw)
        color = legend_colors[mask_id]
        patch = np.full((mh, mw), int(bg), dtype=int)
        patch[placed_mask] = int(color)
        panel[rr : rr + mh, cc : cc + mw][placed_mask] = int(color)
        key, dims = _normalize_coords(np.argwhere(placed_mask))
        legend_entries[mask_id] = {
            "color": int(color),
            "area": int(placed_mask.sum()),
            "canonical": _canonical_key(key, dims),
            "pose": key,
            "patch": patch,
            "variants": variants,
        }
    return panel, legend_entries


def _expanded_slice(top: int, left: int, h: int, w: int, grid_h: int, grid_w: int) -> tuple[slice, slice]:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid_h, top + h + 1)
    c1 = min(grid_w, left + w + 1)
    return slice(r0, r1), slice(c0, c1)


def _stamp_panel(grid: np.ndarray, panel: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    gh, gw = grid.shape
    ph, pw = panel.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, gw - pw
    elif corner == "bl":
        r0, c0 = gh - ph, 0
    else:
        r0, c0 = gh - ph, gw - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = panel
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _place_sprites(
    rng: random.Random,
    grid: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
    class_specs: list[dict],
) -> tuple[bool, list[dict]]:
    gh, gw = grid.shape
    occupied = np.zeros((gh, gw), dtype=bool)
    pr0, pc0, pr1, pc1 = panel_bbox
    rs, cs = _expanded_slice(pr0, pc0, pr1 - pr0 + 1, pc1 - pc0 + 1, gh, gw)
    occupied[rs, cs] = True

    placements: list[dict] = []
    items = []
    for class_id, spec in enumerate(class_specs):
        variants = spec["variant_cycle"]
        for i, mask in enumerate(variants):
            items.append((int(mask.shape[0] * mask.shape[1]), rng.random(), class_id, i, mask))
    items.sort(reverse=True)

    for _score, _tie, class_id, pose_id, mask in items:
        mh, mw = mask.shape
        positions = []
        for r0 in range(1, gh - mh):
            for c0 in range(1, gw - mw):
                rs, cs = _expanded_slice(r0, c0, mh, mw, gh, gw)
                if occupied[rs, cs].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return False, []
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + mh, c0 : c0 + mw][mask] = GREY
        rs, cs = _expanded_slice(r0, c0, mh, mw, gh, gw)
        occupied[rs, cs] = True
        placements.append({"class_id": class_id, "pose_id": pose_id, "mask": mask, "top": r0, "left": c0})
    return True, placements


def _choose_counts(rng: random.Random, entries: list[dict], target_idx: int) -> list[int] | None:
    areas = [int(entry["area"]) for entry in entries]
    target_area = areas[target_idx]
    twins = [idx for idx, area in enumerate(areas) if area == target_area and idx != target_idx]
    if not twins:
        return None

    counts = [0] * len(entries)
    counts[target_idx] = target_area

    twin_idx = rng.choice(twins)
    twin_options = [value for value in range(1, 8) if value != areas[twin_idx] and value != target_area]
    counts[twin_idx] = rng.choice(twin_options)

    other_ids = [idx for idx in range(len(entries)) if idx not in (target_idx, twin_idx)]
    high_ids = [idx for idx in other_ids if any(v > target_area and v != areas[idx] for v in range(1, 8))]
    if high_ids:
        high_idx = rng.choice(high_ids)
        high_options = [value for value in range(target_area + 1, 8) if value != areas[high_idx]]
        counts[high_idx] = rng.choice(high_options)

    for idx in other_ids:
        if counts[idx]:
            continue
        options = [value for value in range(1, 8) if value != areas[idx]]
        counts[idx] = rng.choice(options)

    if sum(count == area for count, area in zip(counts, areas)) != 1:
        return None
    if max(counts) <= target_area:
        return None
    return counts


def _prepare_class_specs(rng: random.Random, entries: list[dict], counts: list[int], target_idx: int) -> list[dict] | None:
    specs = []
    for idx, (entry, count) in enumerate(zip(entries, counts)):
        variants = entry["variants"]
        if count > 1 and len(variants) > 1:
            picks = [variants[rng.randrange(len(variants))] for _ in range(count)]
            if idx == target_idx:
                a, b = rng.sample(range(len(variants)), 2)
                picks[0] = variants[a]
                picks[1] = variants[b]
        else:
            picks = [variants[0] for _ in range(count)]
        specs.append({"variant_cycle": picks, "canonical": entry["canonical"], "area": entry["area"]})
    target_pose_set = {
        _normalize_coords(np.argwhere(mask))[0] for mask in specs[target_idx]["variant_cycle"]
    }
    if len(target_pose_set) < 2:
        return None
    return specs


def _analyze_instance(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    panel_bbox, panel_color = _detect_panel(grid, bg)
    legend = _extract_legend(grid, panel_bbox, panel_color, bg)
    sprite_color, sprites = _extract_gray_sprites(grid, panel_bbox, bg)
    counts = Counter(sprite["canonical"] for sprite in sprites)
    pose_sets: dict[tuple[tuple[int, int], ...], set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    for sprite in sprites:
        pose_sets[sprite["canonical"]].add(sprite["pose"])
    winners = [entry for entry in legend if counts[entry["canonical"]] == entry["area"]]
    return {
        "bg": bg,
        "panel_bbox": panel_bbox,
        "panel_color": panel_color,
        "sprite_color": sprite_color,
        "legend": legend,
        "counts": counts,
        "pose_sets": pose_sets,
        "winners": winners,
    }


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    masks, target_idx = _choose_shapes(rng)

    panel_color = rng.choice([c for c in PALETTE if c not in (bg, GREY)])
    usable = [c for c in PALETTE if c not in (bg, GREY, panel_color)]
    legend_colors = rng.sample(usable, len(masks))

    panel, entries = _build_panel(rng, bg, panel_color, legend_colors, masks)
    counts = _choose_counts(rng, entries, target_idx)
    if counts is None:
        return None
    class_specs = _prepare_class_specs(rng, entries, counts, target_idx)
    if class_specs is None:
        return None

    panel_h, panel_w = panel.shape
    min_h = max(GRID_MIN, panel_h + 8)
    min_w = max(GRID_MIN, panel_w + 8)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    grid_h = rng.randint(min_h, GRID_MAX)
    grid_w = rng.randint(min_w, GRID_MAX)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    panel_bbox = _stamp_panel(grid, panel, corner)
    ok, _placements = _place_sprites(rng, grid, panel_bbox, class_specs)
    if not ok:
        return None

    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != bg:
        return None

    analysis = _analyze_instance(grid)
    if len(analysis["winners"]) != 1:
        return None
    winner = analysis["winners"][0]
    if winner["canonical"] != entries[target_idx]["canonical"]:
        return None
    target_area = winner["area"]
    same_area = [entry for entry in analysis["legend"] if entry["area"] == target_area]
    if len(same_area) < 2:
        return None
    if len(analysis["pose_sets"][winner["canonical"]]) < 2:
        return None

    expected = entries[target_idx]["patch"]
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid vote-equals-area ARC input")
