from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _count_holes(mask: np.ndarray) -> int:
    bg_comps = _components(~mask)
    h, w = map(int, mask.shape)
    holes = 0
    for comp in bg_comps:
        touches_edge = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_edge:
            holes += 1
    return holes


def _find_rule_strip(grid: np.ndarray) -> tuple[int, np.ndarray]:
    h = int(grid.shape[0])
    start = h
    for r in range(h - 1, -1, -1):
        row = grid[r]
        colors = set(map(int, np.unique(row)))
        if colors.issubset({BG, MAROON}) and bool(np.any(row == MAROON)):
            start = r
        else:
            break
    assert start < h, "expected a maroon rule strip at the bottom"
    return start, grid[start:]


def _count_runs(line: np.ndarray) -> int:
    runs = 0
    inside = False
    for value in line.tolist():
        if bool(value) and not inside:
            runs += 1
            inside = True
        elif not bool(value):
            inside = False
    return runs


def _rule_mapping(strip: np.ndarray) -> list[int]:
    h, w = map(int, strip.shape)
    assert h >= 1 and w % 4 == 0, "rule strip width must be divisible by four"
    seg_w = w // 4
    maroon_cols = np.any(strip == MAROON, axis=0)
    return [
        _count_runs(maroon_cols[i * seg_w : (i + 1) * seg_w])
        for i in range(4)
    ]


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components((grid != BG) & (grid != MAROON))
    objects: list[dict[str, object]] = []
    for comp in comps:
        color = int(grid[comp[0][0], comp[0][1]])
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
            assert int(grid[r, c]) == color, "objects must be monochrome"
        objects.append(
            {
                "color": color,
                "top": int(r0),
                "left": int(c0),
                "mask": mask,
                "holes": min(_count_holes(mask), 3),
            }
        )
    objects.sort(key=lambda obj: (int(obj["top"]), int(obj["left"]), int(obj["color"])))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse the bottom maroon rule strip, count holes inside each foreground object,
    and keep/delete/stack the object based on the decoded copy count.
    """
    g = np.asarray(grid, dtype=int)
    rule_top, strip = _find_rule_strip(g)
    copies_by_holes = _rule_mapping(strip)
    objects = _extract_objects(g[:rule_top])

    out = np.zeros_like(g)
    for obj in objects:
        copies = int(copies_by_holes[int(obj["holes"])])
        if copies == 0:
            continue
        mask = np.asarray(obj["mask"], dtype=bool)
        h, w = map(int, mask.shape)
        left = int(obj["left"])
        for idx in range(copies):
            top = int(obj["top"]) + idx * (h + 1)
            if top < 0 or top + h > rule_top:
                break
            patch = out[top : top + h, left : left + w]
            patch[mask] = int(obj["color"])
    return out


def _boxes_touch(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _normalize_mask(out)


def _add_tabs(mask: np.ndarray, max_tabs: int = 2) -> np.ndarray:
    base = np.zeros((mask.shape[0] + 8, mask.shape[1] + 8), dtype=bool)
    base[4 : 4 + mask.shape[0], 4 : 4 + mask.shape[1]] = mask
    for _ in range(random.randint(0, max_tabs)):
        rows = np.where(np.any(base, axis=1))[0]
        cols = np.where(np.any(base, axis=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        side = random.choice(["top", "bottom", "left", "right"])
        depth = random.randint(1, 2)
        if side in {"top", "bottom"}:
            length = random.randint(1, min(3, c1 - c0 + 1))
            start = random.randint(c0, c1 - length + 1)
            if side == "top" and r0 - depth >= 0:
                base[r0 - depth : r0, start : start + length] = True
            if side == "bottom" and r1 + depth < base.shape[0]:
                base[r1 + 1 : r1 + 1 + depth, start : start + length] = True
        else:
            length = random.randint(1, min(3, r1 - r0 + 1))
            start = random.randint(r0, r1 - length + 1)
            if side == "left" and c0 - depth >= 0:
                base[start : start + length, c0 - depth : c0] = True
            if side == "right" and c1 + depth < base.shape[1]:
                base[start : start + length, c1 + 1 : c1 + 1 + depth] = True
    return _normalize_mask(base)


def _make_solid_object() -> np.ndarray:
    for _ in range(120):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        mask = np.ones((h, w), dtype=bool)
        if h >= 3 and w >= 3 and random.random() < 0.75:
            notch_h = random.randint(1, h - 1)
            notch_w = random.randint(1, w - 1)
            corner = random.choice(["tl", "tr", "bl", "br"])
            if corner == "tl":
                mask[:notch_h, :notch_w] = False
            elif corner == "tr":
                mask[:notch_h, w - notch_w :] = False
            elif corner == "bl":
                mask[h - notch_h :, :notch_w] = False
            else:
                mask[h - notch_h :, w - notch_w :] = False
        mask = _add_tabs(mask, max_tabs=2)
        mask = _random_transform(mask)
        if _count_holes(mask) == 0:
            return mask
    raise RuntimeError("failed to build a zero-hole object")


def _make_holey_object(hole_count: int) -> np.ndarray:
    for _ in range(160):
        orient = random.choice(["h", "v"])
        if orient == "h":
            hole_h = random.randint(1, 2)
            hole_ws = [random.randint(1, 2) for _ in range(hole_count)]
            walls = [random.randint(1, 2) for _ in range(max(0, hole_count - 1))]
            top_pad = random.randint(1, 2)
            bottom_pad = random.randint(1, 2)
            left_pad = random.randint(1, 2)
            right_pad = random.randint(1, 2)
            h = top_pad + hole_h + bottom_pad
            w = left_pad + sum(hole_ws) + sum(walls) + right_pad
            mask = np.ones((h, w), dtype=bool)
            row0 = top_pad
            col = left_pad
            for idx, hole_w in enumerate(hole_ws):
                mask[row0 : row0 + hole_h, col : col + hole_w] = False
                col += hole_w
                if idx < len(walls):
                    col += walls[idx]
        else:
            hole_w = random.randint(1, 2)
            hole_hs = [random.randint(1, 2) for _ in range(hole_count)]
            walls = [random.randint(1, 2) for _ in range(max(0, hole_count - 1))]
            top_pad = random.randint(1, 2)
            bottom_pad = random.randint(1, 2)
            left_pad = random.randint(1, 2)
            right_pad = random.randint(1, 2)
            h = top_pad + sum(hole_hs) + sum(walls) + bottom_pad
            w = left_pad + hole_w + right_pad
            mask = np.ones((h, w), dtype=bool)
            col0 = left_pad
            row = top_pad
            for idx, hole_h in enumerate(hole_hs):
                mask[row : row + hole_h, col0 : col0 + hole_w] = False
                row += hole_h
                if idx < len(walls):
                    row += walls[idx]
        mask = _add_tabs(mask, max_tabs=2)
        mask = _random_transform(mask)
        if _count_holes(mask) == hole_count:
            return mask
    raise RuntimeError(f"failed to build a {hole_count}-hole object")


def _build_object(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return _make_solid_object()
    return _make_holey_object(hole_count)


def _choose_rule_counts() -> list[int]:
    while True:
        counts = [random.randint(0, 3) for _ in range(4)]
        if len(set(counts)) < 2:
            continue
        if not any(v == 0 for v in counts):
            continue
        if not any(v >= 2 for v in counts):
            continue
        return counts


def _choose_hole_types(rule_counts: list[int], object_count: int) -> list[int]:
    holes_all = [0, 1, 2, 3]
    if object_count >= 4:
        holes = holes_all[:]
    else:
        required: list[int] = []
        delete_types = [h for h, copies in enumerate(rule_counts) if copies == 0]
        copy_types = [h for h, copies in enumerate(rule_counts) if copies >= 2]
        if delete_types:
            required.append(random.choice(delete_types))
        for hole in copy_types:
            if hole not in required:
                required.append(hole)
                break
        pool = holes_all[:]
        random.shuffle(pool)
        for hole in pool:
            if hole not in required:
                required.append(hole)
            if len(required) == object_count:
                break
        holes = required[:object_count]
    random.shuffle(holes)
    return holes


def _sample_bar_columns(seg_w: int, count: int) -> list[int]:
    if count == 0:
        return []
    choices = [
        cols
        for cols in __import__("itertools").combinations(range(seg_w), count)
        if all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1))
    ]
    return list(random.choice(choices))


def _place_objects(
    objects: list[dict[str, object]],
    work_h: int,
    width: int,
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    reserved_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(
        range(len(objects)),
        key=lambda idx: (
            -int(objects[idx]["reserve_h"]) * int(objects[idx]["width"]),
            -int(objects[idx]["height"]),
        ),
    )
    for idx in order:
        obj = dict(objects[idx])
        reserve_h = int(obj["reserve_h"])
        height = int(obj["height"])
        width_obj = int(obj["width"])
        max_top = work_h - reserve_h
        max_left = width - width_obj
        if max_top < 0 or max_left < 0:
            return None
        success = False
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            box = (top, left, top + reserve_h - 1, left + width_obj - 1)
            if any(_boxes_touch(box, other, gap=1) for other in reserved_boxes):
                continue
            obj["top"] = int(top)
            obj["left"] = int(left)
            reserved_boxes.append(box)
            placed.append(obj)
            success = True
            break
        if not success:
            return None
    placed.sort(key=lambda obj: int(obj["index"]))
    return placed


def _render_candidate(
    objects: list[dict[str, object]],
    work_h: int,
    strip_h: int,
    width: int,
    rule_counts: list[int],
) -> np.ndarray:
    grid = np.zeros((work_h + strip_h, width), dtype=int)
    for obj in objects:
        mask = np.asarray(obj["mask"], dtype=bool)
        top = int(obj["top"])
        left = int(obj["left"])
        h, w = map(int, mask.shape)
        patch = grid[top : top + h, left : left + w]
        patch[mask] = int(obj["color"])
    seg_w = width // 4
    for hole_count, copies in enumerate(rule_counts):
        for col in _sample_bar_columns(seg_w, copies):
            actual_c = hole_count * seg_w + int(col)
            grid[work_h:, actual_c] = MAROON
    return grid


def _validate_candidate(
    grid: np.ndarray,
    placed_objects: list[dict[str, object]],
    rule_counts: list[int],
) -> bool:
    try:
        rule_top, strip = _find_rule_strip(grid)
        parsed_rule = _rule_mapping(strip)
        parsed_objects = _extract_objects(grid[:rule_top])
        out = main(grid)
    except Exception:
        return False

    if parsed_rule != list(rule_counts):
        return False
    if len(parsed_objects) != len(placed_objects):
        return False
    if not np.all(out[rule_top:] == BG):
        return False

    parsed_by_color = {int(obj["color"]): obj for obj in parsed_objects}
    for obj in placed_objects:
        color = int(obj["color"])
        if color not in parsed_by_color:
            return False
        parsed = parsed_by_color[color]
        if int(parsed["holes"]) != int(obj["holes"]):
            return False
        copies = int(rule_counts[int(obj["holes"])])
        expected_pixels = int(np.count_nonzero(np.asarray(obj["mask"], dtype=bool))) * copies
        if int(np.count_nonzero(out == color)) != expected_pixels:
            return False

    surviving = [int(rule_counts[int(obj["holes"])]) for obj in placed_objects]
    if all(v == 1 for v in surviving):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seg_w = random.choice([5, 6])
        width = 4 * seg_w
        strip_h = random.randint(2, 3)
        rule_counts = _choose_rule_counts()
        object_count = random.randint(3, 4)
        hole_types = _choose_hole_types(rule_counts, object_count)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8], object_count)

        objects: list[dict[str, object]] = []
        try:
            for index, (hole_count, color) in enumerate(zip(hole_types, colors, strict=False)):
                mask = _build_object(hole_count)
                height, width_obj = map(int, mask.shape)
                copies = int(rule_counts[hole_count])
                reserve_h = height if copies == 0 else copies * height + (copies - 1)
                objects.append(
                    {
                        "index": int(index),
                        "holes": int(hole_count),
                        "color": int(color),
                        "mask": mask,
                        "height": int(height),
                        "width": int(width_obj),
                        "reserve_h": int(reserve_h),
                    }
                )
        except RuntimeError:
            continue

        max_reserve = max(int(obj["reserve_h"]) for obj in objects)
        min_work_h = max(14, max_reserve + 1)
        max_work_h = 24
        if min_work_h > max_work_h:
            continue
        work_h = random.randint(min_work_h, max_work_h)

        placed = _place_objects(objects, work_h, width)
        if placed is None:
            continue

        grid = _render_candidate(placed, work_h, strip_h, width, rule_counts)
        if _validate_candidate(grid, placed, rule_counts):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
