import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 7]
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for coords in _components_mask(g == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _classify_scene(grid: np.ndarray) -> tuple[dict[int, dict], dict[int, dict]]:
    bg = int(get_bg(grid))
    objects: dict[int, dict] = {}
    lines: dict[int, dict] = {}

    for comp in _find_components(grid, bg):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height == 1 and int(comp["area"]) == width and width >= 3:
            assert color not in lines, "each color must have exactly one horizontal line"
            lines[color] = {
                "color": color,
                "row": int(r0),
                "c0": int(c0),
                "length": width,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": comp["coords"],
            }
        else:
            assert color not in objects, "each color must have exactly one left object"
            objects[color] = {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": comp["coords"],
            }

    assert objects and lines and len(objects) == len(lines) and len(lines) % 2 == 1
    assert set(objects) == set(lines)
    return objects, lines


def _apply_local_toggle(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, depth: int) -> None:
    if depth <= 0:
        return
    r0, c0, r1, c1 = bbox
    width = int(c1 - c0 + 1)
    depth = min(int(depth), width)
    region = out[r0 : r1 + 1, c0 : c0 + depth]
    fg = region == int(color)
    bg = region == BG
    region[fg] = BG
    region[bg] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use right-side horizontal segments to encode how deeply each same-color left object should be locally stripe-erased.

    Input structure:
    - The background is black.
    - Each non-black color appears exactly twice: one irregular 4-connected object on the left and one horizontal line segment on the right.
    - The right-side line count is odd, so sorting those segments by row gives a unique median segment.

    Rule:
    - Identify the horizontal segments and pick the median-by-row segment as the reference; let its length be L_med.
    - For each color, compare its segment length L_orig to L_med and compute depth d = max(0, L_med - L_orig).
    - In the same-color left object's local bbox, toggle the first d columns from the left: object pixels become black and black pixels become the object color.
    - Leave the rest of every object unchanged, and remove all right-side line segments.

    Key insight:
    - The segments are not output objects; after aligning them conceptually to the median reference, only their length deficit matters, and that deficit becomes each object's local stripe inversion depth.
    """
    g = np.asarray(grid, dtype=int)
    objects, lines = _classify_scene(g)
    ordered_lines = sorted(lines.values(), key=lambda item: (int(item["row"]), int(item["c0"])))
    median = ordered_lines[len(ordered_lines) // 2]
    l_med = int(median["length"])

    out = g.copy()
    for line in ordered_lines:
        r0, c0, _r1, c1 = line["bbox"]
        out[r0, c0 : c1 + 1] = BG

    ordered_objects = sorted(objects.values(), key=lambda item: (item["bbox"][0], item["bbox"][1]))
    for obj in ordered_objects:
        color = int(obj["color"])
        depth = max(0, int(l_med - lines[color]["length"]))
        _apply_local_toggle(out, obj["bbox"], color, depth)

    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_interval_object() -> tuple[np.ndarray, list[int]] | None:
    for _ in range(200):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        intervals: list[tuple[int, int]] = []

        length = random.randint(2, h)
        top = random.randint(0, h - length)
        intervals.append((int(top), int(top + length - 1)))

        for _c in range(1, w):
            prev_top, prev_bot = intervals[-1]
            candidates: list[tuple[tuple[int, int], float]] = []
            for cur_len in range(1, h + 1):
                cur_top_max = h - cur_len
                for cur_top in range(cur_top_max + 1):
                    cur_bot = cur_top + cur_len - 1
                    if max(cur_top, prev_top) > min(cur_bot, prev_bot):
                        continue
                    weight = 1.0
                    if cur_len < h:
                        weight += 1.5
                    if cur_top != prev_top or cur_bot != prev_bot:
                        weight += 1.0
                    candidates.append(((int(cur_top), int(cur_bot)), weight))
            if not candidates:
                break
            choices = [cand for cand, _weight in candidates]
            weights = [weight for _cand, weight in candidates]
            intervals.append(random.choices(choices, weights=weights, k=1)[0])
        else:
            mask = np.zeros((h, w), dtype=bool)
            for c, (top, bot) in enumerate(intervals):
                mask[top : bot + 1, c] = True
            mask = _tight_crop(mask)
            mh, mw = mask.shape
            area = int(mask.sum())
            if mw < 3 or mh < 2 or not (8 <= area <= 20):
                continue
            if bool(mask.all()):
                continue

            valid_depths = []
            for depth in range(1, mw + 1):
                stripe = mask[:, :depth]
                filled = int(stripe.sum())
                if 0 < filled < stripe.size:
                    valid_depths.append(int(depth))
            if not valid_depths:
                continue
            return mask.astype(bool), valid_depths
    return None


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows, r0 + h + pad)
    cc1 = min(cols, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _choose_object_specs(count: int) -> list[dict] | None:
    specs = []
    for color in random.sample(PALETTE, count):
        made = _make_interval_object()
        if made is None:
            return None
        mask, valid_depths = made
        specs.append(
            {
                "color": int(color),
                "mask": mask,
                "valid_depths": valid_depths,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "area": int(mask.sum()),
            }
        )
    specs.sort(key=lambda item: (max(item["h"], item["w"]), item["area"]), reverse=True)
    return specs


def _place_objects(rows: int, cols: int, specs: list[dict]) -> list[dict] | None:
    split = cols // 2
    left_limit = split - 1
    occ = np.zeros((rows, cols), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        if w + 2 > left_limit:
            return None
        candidates = []
        for r0 in range(1, rows - h):
            for c0 in range(1, left_limit - w):
                if _can_place_bbox(occ, r0, c0, h, w, pad=1):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occ[r0 : r0 + h, c0 : c0 + w] = True
        placed.append({**spec, "r0": int(r0), "c0": int(c0), "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))})

    return placed


def _choose_line_lengths(objects: list[dict]) -> tuple[int, dict[int, int], dict[int, int]] | None:
    for _ in range(200):
        l_med = random.randint(4, 7)
        shortable = [obj for obj in objects if any(depth <= l_med - 3 for depth in obj["valid_depths"])]
        if not shortable:
            continue

        count = len(objects)
        short_target = 1 if count == 3 else random.choice([1, 2])
        short_target = min(short_target, len(shortable))
        if short_target == 0:
            continue

        deep_shortable = [obj for obj in shortable if any(depth >= 2 and depth <= l_med - 3 for depth in obj["valid_depths"])]
        short_colors: list[int] = []
        if deep_shortable and random.random() < 0.65:
            chosen = random.choice(deep_shortable)
            short_colors.append(int(chosen["color"]))

        remaining_short_pool = [obj for obj in shortable if int(obj["color"]) not in short_colors]
        random.shuffle(remaining_short_pool)
        for obj in remaining_short_pool:
            if len(short_colors) >= short_target:
                break
            short_colors.append(int(obj["color"]))
        short_colors = short_colors[:short_target]
        if not short_colors:
            continue

        remaining_colors = [int(obj["color"]) for obj in objects if int(obj["color"]) not in short_colors]
        if not remaining_colors:
            continue
        long_target = 1 if count == 3 else random.choice([1, 2])
        long_target = min(long_target, len(remaining_colors))
        if long_target == 0:
            continue
        long_colors = set(random.sample(remaining_colors, long_target))

        lengths: dict[int, int] = {}
        depths: dict[int, int] = {}
        by_color = {int(obj["color"]): obj for obj in objects}

        for color in short_colors:
            obj = by_color[color]
            choices = [depth for depth in obj["valid_depths"] if depth <= l_med - 3]
            if not choices:
                break
            weighted = []
            for depth in choices:
                weight = 1.0
                if depth >= 2:
                    weight += 1.0
                weighted.append(weight)
            depth = random.choices(choices, weights=weighted, k=1)[0]
            depths[color] = int(depth)
            lengths[color] = int(l_med - depth)
        else:
            for color in remaining_colors:
                if color in long_colors:
                    lengths[color] = random.randint(l_med + 1, 8)
                else:
                    lengths[color] = int(l_med)
                depths[color] = max(0, int(l_med - lengths[color]))

            if not any(length > l_med for length in lengths.values()):
                continue
            if not any(depth > 0 for depth in depths.values()):
                continue
            if count > 3 and random.random() < 0.6 and not any(depth >= 2 for depth in depths.values()):
                continue
            return int(l_med), lengths, depths
    return None


def _place_lines(rows: int, cols: int, lengths: dict[int, int]) -> list[dict] | None:
    split = cols // 2
    right_start = split + 1
    max_len = max(int(v) for v in lengths.values())
    if right_start + max_len >= cols - 1:
        return None

    count = len(lengths)
    candidate_rows = list(range(1, rows - 1))
    good_rows = []
    for row in candidate_rows:
        if all(abs(row - other) >= 2 for other in good_rows):
            good_rows.append(int(row))
    if len(good_rows) < count:
        return None

    chosen_rows = sorted(random.sample(good_rows, count))
    colors = list(lengths)
    random.shuffle(colors)
    row_order = dict(zip(colors, chosen_rows))

    possible_starts = {}
    for color, length in lengths.items():
        starts = list(range(right_start, cols - int(length) - 1))
        if not starts:
            return None
        possible_starts[int(color)] = starts

    lines = []
    used_starts = set()
    for color in colors:
        starts = possible_starts[color][:]
        random.shuffle(starts)
        chosen_start = None
        for start in starts:
            if len(used_starts) < 2 or start not in used_starts or len(starts) == 1:
                chosen_start = int(start)
                break
        if chosen_start is None:
            chosen_start = int(starts[0])
        used_starts.add(int(chosen_start))
        lines.append(
            {
                "color": int(color),
                "row": int(row_order[color]),
                "c0": int(chosen_start),
                "length": int(lengths[color]),
                "bbox": (int(row_order[color]), int(chosen_start), int(row_order[color]), int(chosen_start + lengths[color] - 1)),
            }
        )

    if len({line["c0"] for line in lines}) == 1:
        return None
    return lines


def _render_input(rows: int, cols: int, objects: list[dict], lines: list[dict]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for obj in objects:
        patch = _mask_to_patch(obj["mask"], obj["color"])
        r0, c0 = obj["r0"], obj["c0"]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w][patch != 0] = int(obj["color"])
    for line in lines:
        r0, c0, _r1, c1 = line["bbox"]
        grid[r0, c0 : c1 + 1] = int(line["color"])
    return grid


def _expected_from_spec(inp: np.ndarray, objects: list[dict], lines: list[dict], l_med: int) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    line_by_color = {int(line["color"]): line for line in lines}
    for line in lines:
        r0, c0, _r1, c1 = line["bbox"]
        out[r0, c0 : c1 + 1] = BG
    for obj in sorted(objects, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        color = int(obj["color"])
        depth = max(0, int(l_med - line_by_color[color]["length"]))
        _apply_local_toggle(out, obj["bbox"], color, depth)
    return out


def _semantic_check(inp: np.ndarray, out: np.ndarray, objects: list[dict], lines: list[dict], l_med: int) -> bool:
    parsed_objects, parsed_lines = _classify_scene(inp)
    if set(parsed_objects) != {int(obj["color"]) for obj in objects}:
        return False
    if set(parsed_lines) != {int(line["color"]) for line in lines}:
        return False

    ordered_lines = sorted(lines, key=lambda item: (item["row"], item["c0"]))
    if ordered_lines[len(ordered_lines) // 2]["length"] != int(l_med):
        return False
    if not any(int(line["length"]) > int(l_med) for line in lines):
        return False
    if not any(int(line["length"]) < int(l_med) for line in lines):
        return False

    changed_object = False
    for obj in objects:
        color = int(obj["color"])
        depth = max(0, int(l_med - next(line["length"] for line in lines if int(line["color"]) == color)))
        r0, c0, r1, c1 = obj["bbox"]
        if depth > int(c1 - c0 + 1):
            return False
        if depth > 0:
            stripe = inp[r0 : r1 + 1, c0 : c0 + depth]
            if not (np.any(stripe == color) and np.any(stripe == BG)):
                return False
            if not np.array_equal(inp[r0 : r1 + 1, c0 + depth : c1 + 1], out[r0 : r1 + 1, c0 + depth : c1 + 1]):
                return False
            if np.array_equal(inp[r0 : r1 + 1, c0 : c0 + depth], out[r0 : r1 + 1, c0 : c0 + depth]):
                return False
            changed_object = True
    if not changed_object:
        return False

    cleared = np.array(inp, copy=True)
    for line in lines:
        r0, c0, _r1, c1 = line["bbox"]
        cleared[r0, c0 : c1 + 1] = BG
    if np.array_equal(cleared, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([3, 5])
    objects = _choose_object_specs(count)
    if objects is None:
        return None

    max_obj_h = max(int(obj["h"]) for obj in objects)
    rows = random.randint(max(18, 2 * count + max_obj_h + 6), 28)
    cols = random.randint(22, 30)

    objects = _place_objects(rows, cols, objects)
    if objects is None:
        return None

    chosen = _choose_line_lengths(objects)
    if chosen is None:
        return None
    l_med, lengths, _depths = chosen

    lines = _place_lines(rows, cols, lengths)
    if lines is None:
        return None

    inp = _render_input(rows, cols, objects, lines)
    expected = _expected_from_spec(inp, objects, lines, l_med)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _semantic_check(inp, expected, objects, lines, l_med):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
