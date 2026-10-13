from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, h, w):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_component_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        out[r - r0, c - c0] = True
    return out


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _hole_masks(obj_mask: np.ndarray) -> list[np.ndarray]:
    h, w = obj_mask.shape
    bg_mask = ~obj_mask
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c, h, w):
                    if bg_mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            if not touches_border:
                holes.append(_crop_component_mask(pts))
    return holes


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[tuple[int, int, bytes], int]:
    legend: dict[tuple[int, int, bytes], int] = {}
    for pts in _components((grid != int(bg)) & (grid != GREY)):
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        key = _shape_key(_crop_component_mask(pts))
        assert key not in legend, "legend shapes must be unique after normalization"
        legend[key] = int(color)
    assert legend, "expected at least one colored legend shape"
    return legend


def _majority_color(colors: list[int]) -> int:
    counts = Counter(colors)
    return max(counts, key=lambda color: (counts[color], -colors.index(color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each internal hole shape against the legend, recolor grey objects by the
    most frequent matched legend color, discard hole-free objects, then pack the
    surviving objects left-to-right in ascending order of hole count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend = _legend_mapping(g, bg)

    packed: list[tuple[int, np.ndarray]] = []
    for pts in _components(g == GREY):
        obj_mask = _crop_component_mask(pts)
        holes = _hole_masks(obj_mask)
        if not holes:
            continue
        matched = [legend[_shape_key(hole)] for hole in holes if _shape_key(hole) in legend]
        if not matched:
            continue
        color = _majority_color(matched)
        sprite = np.full(obj_mask.shape, int(bg), dtype=int)
        sprite[obj_mask] = int(color)
        packed.append((len(holes), sprite))

    if not packed:
        return np.full((1, 1), int(bg), dtype=int)

    packed.sort(key=lambda item: item[0])
    out_h = max(sprite.shape[0] for _, sprite in packed)
    out_w = sum(sprite.shape[1] for _, sprite in packed) + len(packed) - 1
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for _hole_count, sprite in packed:
        h, w = sprite.shape
        out[:h, col : col + w] = sprite
        col += w + 1
    return out


BASE_HOLE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _build_hole_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for base in BASE_HOLE_SHAPES:
        for k in range(4):
            rotated = np.rot90(base, k)
            for flipped in (rotated, np.fliplr(rotated)):
                key = _shape_key(flipped)
                if key in seen:
                    continue
                seen.add(key)
                library.append(np.array(flipped, dtype=bool))
    return library


HOLE_LIBRARY = _build_hole_library()


def _row_layouts(n: int) -> list[list[int]]:
    layouts = {
        1: [[1]],
        2: [[2], [1, 1]],
        3: [[3], [1, 2], [2, 1], [1, 1, 1]],
        4: [[2, 2], [1, 3], [3, 1], [2, 1, 1], [1, 1, 2], [1, 1, 1, 1]],
    }
    return layouts[n]


def _make_hollow_object(bg: int, hole_masks: list[np.ndarray]) -> np.ndarray:
    holes = [np.array(mask, copy=True) for mask in hole_masks]
    random.shuffle(holes)

    layout = random.choice(_row_layouts(len(holes)))
    rows: list[list[np.ndarray]] = []
    idx = 0
    for count in layout:
        rows.append(holes[idx : idx + count])
        idx += count

    col_gap = random.randint(1, 2)
    row_gap = random.randint(1, 2)
    row_heights = [max(mask.shape[0] for mask in row) for row in rows]
    row_widths = [
        sum(mask.shape[1] for mask in row) + col_gap * (len(row) - 1)
        for row in rows
    ]
    inner_w = max(row_widths) + random.randint(0, 2)
    content_h = sum(row_heights) + row_gap * (len(rows) - 1)
    inner_h = content_h + random.randint(0, 2)

    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)

    sprite = np.full((pad_top + inner_h + pad_bottom, pad_left + inner_w + pad_right), GREY, dtype=int)
    y = pad_top + random.randint(0, inner_h - content_h)
    for row, row_h, row_w in zip(rows, row_heights, row_widths, strict=False):
        x = pad_left + random.randint(0, inner_w - row_w)
        for mask in row:
            hole_h, hole_w = mask.shape
            y_off = y + random.randint(0, row_h - hole_h)
            sprite[y_off : y_off + hole_h, x : x + hole_w][mask] = int(bg)
            x += hole_w + col_gap
        y += row_h + row_gap
    return sprite


def _make_solid_distractor() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 6)
    return np.full((h, w), GREY, dtype=int)


def _sample_legend_entries(bg: int) -> list[dict]:
    count = random.randint(3, 4)
    shapes = random.sample(HOLE_LIBRARY, count)
    colors = random.sample([c for c in range(10) if c not in (bg, GREY)], count)
    entries = []
    for shape, color in zip(shapes, colors, strict=False):
        entries.append({"mask": np.array(shape, copy=True), "color": int(color)})
    return entries


def _hole_plan(hole_count: int, legend_count: int, force_mixed: bool) -> list[int]:
    if hole_count == 1:
        return [random.randrange(legend_count)]

    if hole_count == 2 or not force_mixed:
        choice = random.randrange(legend_count)
        return [choice] * hole_count

    majority = random.randrange(legend_count)
    others = [i for i in range(legend_count) if i != majority]
    if hole_count == 3:
        plan = [majority, majority, random.choice(others)]
    else:
        if len(others) >= 2 and random.random() < 0.5:
            other_a, other_b = random.sample(others, 2)
            plan = [majority, majority, other_a, other_b]
        else:
            plan = [majority, majority, majority, random.choice(others)]
    random.shuffle(plan)
    return plan


def _recolor_sprite(sprite: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.full(sprite.shape, int(bg), dtype=int)
    out[sprite == GREY] = int(color)
    return out


def _pack_expected(bg: int, objects: list[dict]) -> np.ndarray:
    ordered = sorted(objects, key=lambda item: item["hole_count"])
    out_h = max(obj["output"].shape[0] for obj in ordered)
    out_w = sum(obj["output"].shape[1] for obj in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), int(bg), dtype=int)
    col = 0
    for obj in ordered:
        sprite = obj["output"]
        h, w = sprite.shape
        out[:h, col : col + w] = sprite
        col += w + 1
    return out


def _legend_band_width(entries: list[dict], gaps: list[int]) -> int:
    return sum(entry["mask"].shape[1] for entry in entries) + sum(gaps)


def _place_objects(
    grid_h: int,
    grid_w: int,
    top_start: int,
    sprites: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    occ[:top_start, :] = True
    placements: list[tuple[int, int]] = []
    order = sorted(range(len(sprites)), key=lambda i: sprites[i].shape[0] * sprites[i].shape[1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}

    for idx in order:
        sprite = sprites[idx]
        h, w = sprite.shape
        if h > grid_h - top_start or w > grid_w:
            return None
        success = False
        for _ in range(80):
            r = random.randint(top_start, grid_h - h)
            c = random.randint(0, grid_w - w)
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(grid_h, r + h + 1)
            c1 = min(grid_w, c + w + 1)
            if occ[r0:r1, c0:c1].any():
                continue
            occ[r:r + h, c:c + w] = True
            placed[idx] = (int(r), int(c))
            success = True
            break
        if not success:
            return None

    for i in range(len(sprites)):
        placements.append(placed[i])
    return placements


def _color_is_unique_bg(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)
    if not order or int(order[0][1]) != int(bg):
        return False
    if len(order) > 1 and int(order[0][0]) == int(order[1][0]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != GREY])
    legend_entries = _sample_legend_entries(bg)
    legend_count = len(legend_entries)

    keep_count = random.randint(2, 4)
    hole_counts = random.sample([1, 2, 3, 4], keep_count)
    mixed_candidates = [i for i, hole_count in enumerate(hole_counts) if hole_count >= 3]
    mixed_index = random.choice(mixed_candidates) if mixed_candidates and random.random() < 0.5 else None

    kept_objects: list[dict] = []
    for idx, hole_count in enumerate(hole_counts):
        plan = _hole_plan(hole_count, legend_count, idx == mixed_index)
        hole_masks = [legend_entries[legend_idx]["mask"] for legend_idx in plan]
        sprite = _make_hollow_object(bg, hole_masks)
        colors = [legend_entries[legend_idx]["color"] for legend_idx in plan]
        final_color = _majority_color(colors)
        kept_objects.append(
            {
                "input": sprite,
                "output": _recolor_sprite(sprite, bg, final_color),
                "hole_count": int(hole_count),
            }
        )

    distractors: list[np.ndarray] = []
    if random.random() < 0.5:
        for _ in range(random.randint(1, 2)):
            distractors.append(_make_solid_distractor())

    expected = _pack_expected(bg, kept_objects)

    legend_gaps = [random.randint(1, 2) for _ in range(len(legend_entries) - 1)]
    band_h = max(entry["mask"].shape[0] for entry in legend_entries) + random.randint(1, 2)
    object_top = band_h + random.randint(1, 2)
    legend_w = _legend_band_width(legend_entries, legend_gaps)

    object_sprites = [obj["input"] for obj in kept_objects] + distractors
    max_obj_h = max(sprite.shape[0] for sprite in object_sprites)
    max_obj_w = max(sprite.shape[1] for sprite in object_sprites)
    total_bbox_area = sum(sprite.shape[0] * sprite.shape[1] for sprite in object_sprites)

    min_w = max(legend_w + 2, max_obj_w + 2, int(np.ceil(np.sqrt(total_bbox_area * 2.0))))
    max_w = min(30, max(min_w, 20) + 6)
    min_w = max(min_w, 20)
    if min_w > 30:
        return None
    grid_w = random.randint(min_w, max_w)

    rough_h = object_top + max_obj_h + 2
    fill_h = object_top + int(np.ceil(total_bbox_area / max(1, grid_w - 2))) + 3
    min_h = max(rough_h, fill_h, 18)
    max_h = min(30, min_h + 6)
    if min_h > 30:
        return None
    grid_h = random.randint(min_h, max_h)

    placements = _place_objects(grid_h, grid_w, object_top, object_sprites)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    start_col = random.randint(1, grid_w - legend_w - 1)
    col = start_col
    for i, entry in enumerate(legend_entries):
        mask = entry["mask"]
        color = entry["color"]
        h, w = mask.shape
        row = random.randint(0, band_h - h)
        block = grid[row : row + h, col : col + w]
        block[mask] = int(color)
        col += w
        if i < len(legend_gaps):
            col += legend_gaps[i]

    for sprite, (r, c) in zip(object_sprites, placements, strict=False):
        h, w = sprite.shape
        grid[r : r + h, c : c + w] = sprite

    if not _color_is_unique_bg(grid, bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
