from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("up", "right", "down", "left")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop(g: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _ring_area(h: int, w: int) -> int:
    return 2 * (h + w) - 4


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == GREY)
    objects = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        sprite = _crop(grid, (r0, c0, r1, c1))
        border = np.zeros_like(sprite, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if int(np.count_nonzero(sprite == GREY)) != _ring_area(h, w):
            continue
        if not np.all(sprite[border] == GREY):
            continue
        interior = sprite[1:-1, 1:-1].copy()
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sprite": sprite,
                "interior": interior,
            }
        )
    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def _repeat_axis(interior: np.ndarray) -> str | None:
    h, w = interior.shape
    axes: list[str] = []
    if w >= 2:
        half = w // 2
        if half > 0 and np.array_equal(interior[:, :half], interior[:, w - half :]):
            axes.append("lr")
    if h >= 2:
        half = h // 2
        if half > 0 and np.array_equal(interior[:half, :], interior[h - half :, :]):
            axes.append("ud")
    if len(axes) == 1:
        return axes[0]
    return None


def _first_half(interior: np.ndarray, axis: str) -> np.ndarray:
    h, w = interior.shape
    if axis == "lr":
        return interior[:, : w // 2].copy()
    return interior[: h // 2, :].copy()


def _singleton_marker(first_half: np.ndarray) -> tuple[int, int, int]:
    counts = Counter(map(int, first_half.ravel()))
    cells = [(r, c) for r in range(first_half.shape[0]) for c in range(first_half.shape[1]) if counts[int(first_half[r, c])] == 1]
    if len(cells) != 1:
        raise ValueError("expected exactly one singleton marker")
    r, c = cells[0]
    return r, c, int(first_half[r, c])


def _decode_direction_and_depth(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[str, int]:
    r, c = pos
    h, w = shape
    top = 2 * r <= h - 1
    left = 2 * c <= w - 1
    if top and left:
        return "up", r + 1
    if top and not left:
        return "right", w - c
    if not top and not left:
        return "down", h - r
    return "left", c + 1


def _recolor_interior(interior: np.ndarray, direction: str, depth: int, color: int) -> np.ndarray:
    out = interior.copy()
    h, w = out.shape
    depth = int(depth)
    if direction in {"up", "down"}:
        depth = min(depth, h)
        rows = slice(0, depth) if direction == "up" else slice(h - depth, h)
        mask = out[rows, :] != BLACK
        out[rows, :][mask] = color
    else:
        depth = min(depth, w)
        cols = slice(0, depth) if direction == "left" else slice(w - depth, w)
        mask = out[:, cols] != BLACK
        out[:, cols][mask] = color
    return out


def _inspect(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    objects = _extract_objects(grid)
    repeated: list[dict] = []
    normal: list[dict] = []
    for obj in objects:
        axis = _repeat_axis(obj["interior"])
        if axis is None:
            item = dict(obj)
            colors = set(map(int, item["interior"].ravel())) - {BLACK}
            item["colors"] = colors
            normal.append(item)
            continue
        first = _first_half(obj["interior"], axis)
        mr, mc, color = _singleton_marker(first)
        direction, depth = _decode_direction_and_depth((mr, mc), first.shape)
        item = dict(obj)
        item.update(
            {
                "axis": axis,
                "first_half": first,
                "marker_pos": (mr, mc),
                "marker_color": color,
                "direction": direction,
                "depth": depth,
            }
        )
        repeated.append(item)
    return repeated, normal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor ordinary bordered rectangles using direction-and-depth commands encoded in repeated ones.

    Input structure:
    - The grid is a black background containing several non-touching rectangles with grey borders.
    - Each rectangle encloses an interior pattern; some interiors repeat exactly across left/right halves or top/bottom halves.
    - Every repeating rectangle has one singleton-colored marker pixel in its first half, and each marker color appears in exactly one non-repeating rectangle.

    Rule:
    - Detect every grey rectangular frame and inspect only its interior.
    - Classify rectangles whose interior has exactly one valid equal-half repeat axis; keep the left half for left/right repeats or the top half for top/bottom repeats.
    - Find the singleton marker in that first half, decode its quadrant as up/right/down/left, convert its row or column offset from that side into a repaint depth, and locate the unique non-repeating rectangle containing the same color.
    - Repaint that target rectangle's interior non-black pixels by rows or columns from the decoded side for the decoded depth; keep the grey borders and the black background unchanged.

    Key insight:
    - Repetition identifies command sources rather than outputs: the unrepeated half contains a direction marker whose color selects a target object and whose position determines how deeply to recolor it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    repeated, normal = _inspect(g)

    if not repeated or not normal:
        return out

    for src in repeated:
        matches = [obj for obj in normal if src["marker_color"] in obj["colors"]]
        if len(matches) != 1:
            raise ValueError("marker color must match exactly one normal object")
        target = matches[0]
        r0, c0, r1, c1 = target["bbox"]
        new_interior = _recolor_interior(target["interior"], src["direction"], src["depth"], src["marker_color"])
        out[r0 + 1 : r1, c0 + 1 : c1] = new_interior
    return out


def _is_nonrepeat(interior: np.ndarray) -> bool:
    return _repeat_axis(interior) is None


def _wrap(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    sprite = np.full((h + 2, w + 2), GREY, dtype=int)
    sprite[1:-1, 1:-1] = interior
    return sprite


def _colors_without(*excluded: int) -> list[int]:
    banned = set(excluded)
    return [c for c in COLORS if c not in banned]


def _make_repeat_object(marker_color: int, forced_direction: str) -> dict:
    for _ in range(400):
        axis = random.choice(["lr", "ud"])
        if axis == "lr":
            full_h = random.randint(2, 6)
            half_w = random.randint(2, 3)
            middle_w = 0 if half_w == 3 else random.randint(0, 1)
            first_shape = (full_h, half_w)
            full_shape = (full_h, 2 * half_w + middle_w)
        else:
            full_w = random.randint(2, 6)
            half_h = random.randint(2, 3)
            middle_h = 0 if half_h == 3 else random.randint(0, 1)
            first_shape = (half_h, full_w)
            full_shape = (2 * half_h + middle_h, full_w)

        fh, fw = first_shape
        candidates = [(r, c) for r in range(fh) for c in range(fw) if _decode_direction_and_depth((r, c), first_shape)[0] == forced_direction]
        if not candidates:
            continue
        marker_r, marker_c = random.choice(candidates)
        marker_direction, depth = _decode_direction_and_depth((marker_r, marker_c), first_shape)
        assert marker_direction == forced_direction

        base_a, base_b = random.sample(_colors_without(marker_color), 2)
        first = np.full(first_shape, base_a, dtype=int)
        area = fh * fw
        slots = [(r, c) for r in range(fh) for c in range(fw) if (r, c) != (marker_r, marker_c)]
        random.shuffle(slots)
        alt_count = random.randint(2, min(len(slots), max(2, area // 2)))
        for r, c in slots[:alt_count]:
            first[r, c] = base_b
        first[marker_r, marker_c] = marker_color

        if axis == "lr":
            pieces = [first]
            if full_shape[1] % 2 == 1:
                middle = np.random.choice([base_a, base_b], size=(fh, 1)).astype(int)
                pieces.append(middle)
            pieces.append(first.copy())
            interior = np.concatenate(pieces, axis=1)
        else:
            pieces = [first]
            if full_shape[0] % 2 == 1:
                middle = np.random.choice([base_a, base_b], size=(1, fw)).astype(int)
                pieces.append(middle)
            pieces.append(first.copy())
            interior = np.concatenate(pieces, axis=0)

        if _repeat_axis(interior) != axis:
            continue
        check = _first_half(interior, axis)
        try:
            mr, mc, mc_color = _singleton_marker(check)
        except ValueError:
            continue
        if (mr, mc, mc_color) != (marker_r, marker_c, marker_color):
            continue
        sprite = _wrap(interior)
        return {
            "kind": "repeat",
            "sprite": sprite,
            "interior": interior,
            "axis": axis,
            "marker_color": marker_color,
            "direction": forced_direction,
            "depth": depth,
            "first_shape": first_shape,
        }
    raise RuntimeError("failed to build repeat object")


def _stripe_slice(shape: tuple[int, int], direction: str, depth: int) -> tuple[slice, slice]:
    h, w = shape
    if direction == "up":
        return slice(0, min(depth, h)), slice(0, w)
    if direction == "down":
        return slice(max(0, h - depth), h), slice(0, w)
    if direction == "left":
        return slice(0, h), slice(0, min(depth, w))
    return slice(0, h), slice(max(0, w - depth), w)


def _make_normal_object(marker_color: int, direction: str, depth: int, forbid: set[int]) -> dict:
    for _ in range(500):
        min_h = max(2, depth if direction in {"up", "down"} else 2)
        min_w = max(2, depth if direction in {"left", "right"} else 2)
        h = random.randint(min_h, 6)
        w = random.randint(min_w, 6)
        if direction in {"up", "down"} and h == depth and random.random() < 0.75:
            h = min(6, h + 1)
        if direction in {"left", "right"} and w == depth and random.random() < 0.75:
            w = min(6, w + 1)

        fillers = random.sample(_colors_without(marker_color, *forbid), 2)
        palette = [marker_color, fillers[0], fillers[1]]
        interior = np.random.choice(palette, size=(h, w), p=[0.28, 0.38, 0.34]).astype(int)

        hole_budget = 0
        if h * w >= 9 and random.random() < 0.75:
            hole_budget = random.randint(1, min(2, h * w // 4))
        if hole_budget:
            cells = [(r, c) for r in range(h) for c in range(w)]
            random.shuffle(cells)
            for r, c in cells[:hole_budget]:
                interior[r, c] = BLACK

        stripe = _stripe_slice((h, w), direction, depth)
        stripe_cells = interior[stripe]
        if not np.any((stripe_cells != BLACK) & (stripe_cells != marker_color)):
            continue
        if not np.any(interior == marker_color):
            continue
        if not np.any(interior == fillers[0]) or not np.any(interior == fillers[1]):
            continue
        if _repeat_axis(interior) is not None:
            continue

        sprite = _wrap(interior)
        return {
            "kind": "normal",
            "sprite": sprite,
            "interior": interior,
            "colors": set(map(int, interior.ravel())) - {BLACK},
        }
    raise RuntimeError("failed to build normal object")


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _box_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _corridor_box(rep_box: tuple[int, int, int, int], tgt_box: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int]:
    rr0, rc0, rr1, rc1 = rep_box
    tr0, tc0, tr1, tc1 = tgt_box
    if direction == "up":
        return tr1 + 1, max(rc0, tc0), rr0 - 1, min(rc1, tc1)
    if direction == "down":
        return rr1 + 1, max(rc0, tc0), tr0 - 1, min(rc1, tc1)
    if direction == "left":
        return max(rr0, tr0), tc1 + 1, min(rr1, tr1), rc0 - 1
    return max(rr0, tr0), rc1 + 1, min(rr1, tr1), tc0 - 1


def _place_pair(
    grid_shape: tuple[int, int],
    repeat_sprite: np.ndarray,
    target_sprite: np.ndarray,
    direction: str,
    placed: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    h, w = grid_shape
    rh, rw = repeat_sprite.shape
    th, tw = target_sprite.shape
    for _ in range(500):
        gap = random.randint(2, 5)
        if direction == "up":
            c = random.randint(1, w - max(rw, tw) - 1)
            rep_r = random.randint(th + gap + 1, h - rh - 1)
            tgt_r = rep_r - gap - th
            rep_pos = (rep_r, c)
            tgt_pos = (tgt_r, c)
        elif direction == "down":
            c = random.randint(1, w - max(rw, tw) - 1)
            rep_r = random.randint(1, h - rh - gap - th - 1)
            tgt_r = rep_r + rh + gap
            rep_pos = (rep_r, c)
            tgt_pos = (tgt_r, c)
        elif direction == "left":
            r = random.randint(1, h - max(rh, th) - 1)
            rep_c = random.randint(tw + gap + 1, w - rw - 1)
            tgt_c = rep_c - gap - tw
            rep_pos = (r, rep_c)
            tgt_pos = (r, tgt_c)
        else:
            r = random.randint(1, h - max(rh, th) - 1)
            rep_c = random.randint(1, w - rw - gap - tw - 1)
            tgt_c = rep_c + rw + gap
            rep_pos = (r, rep_c)
            tgt_pos = (r, tgt_c)

        rep_box = (rep_pos[0], rep_pos[1], rep_pos[0] + rh - 1, rep_pos[1] + rw - 1)
        tgt_box = (tgt_pos[0], tgt_pos[1], tgt_pos[0] + th - 1, tgt_pos[1] + tw - 1)
        if any(_boxes_touch(rep_box, box, pad=1) or _boxes_touch(tgt_box, box, pad=1) for box in placed):
            continue
        corridor = _corridor_box(rep_box, tgt_box, direction)
        if corridor[0] <= corridor[2] and corridor[1] <= corridor[3]:
            if any(_box_intersects(corridor, box) for box in placed):
                continue
        return rep_pos, tgt_pos, rep_box, tgt_box
    raise RuntimeError("failed to place pair")


def _expected_from_pairs(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        r0, c0, r1, c1 = pair["target_box"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        out[r0 + 1 : r1, c0 + 1 : c1] = _recolor_interior(interior, pair["direction"], pair["depth"], pair["marker_color"])
    return out


def _semantic_audit(grid: np.ndarray, pairs: list[dict], expected: np.ndarray) -> None:
    repeated, normal = _inspect(grid)
    assert len(repeated) == len(normal) == len(pairs)
    pair_map = {(pair["repeat_box"], pair["target_box"]): pair for pair in pairs}

    seen_targets = set()
    for src in repeated:
        matches = [obj for obj in normal if src["marker_color"] in obj["colors"]]
        assert len(matches) == 1
        target = matches[0]
        key = (src["bbox"], target["bbox"])
        assert key in pair_map
        meta = pair_map[key]
        assert src["direction"] == meta["direction"]
        assert src["depth"] == meta["depth"]
        seen_targets.add(target["bbox"])
    assert len(seen_targets) == len(pairs)
    assert np.array_equal(main(grid), expected)


def _generate_case() -> np.ndarray:
    pair_count = random.choice([2, 3])
    directions = random.sample(DIRS, pair_count)
    marker_colors = random.sample(COLORS, pair_count)
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    pairs: list[dict] = []
    placed: list[tuple[int, int, int, int]] = []

    for marker_color, direction in zip(marker_colors, directions):
        repeat_obj = _make_repeat_object(marker_color, direction)
        forbid = set(marker_colors) - {marker_color}
        normal_obj = _make_normal_object(marker_color, direction, repeat_obj["depth"], forbid)
        rep_pos, tgt_pos, rep_box, tgt_box = _place_pair((h, w), repeat_obj["sprite"], normal_obj["sprite"], direction, placed)

        _paste(grid, repeat_obj["sprite"], rep_pos[0], rep_pos[1])
        _paste(grid, normal_obj["sprite"], tgt_pos[0], tgt_pos[1])
        placed.extend([rep_box, tgt_box])
        pairs.append(
            {
                "repeat_box": rep_box,
                "target_box": tgt_box,
                "direction": direction,
                "depth": repeat_obj["depth"],
                "marker_color": marker_color,
            }
        )

    expected = _expected_from_pairs(grid, pairs)
    if np.array_equal(expected, grid):
        raise RuntimeError("degenerate case with no visible repaint")
    _semantic_audit(grid, pairs, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _generate_case()
        except Exception:
            continue
    raise RuntimeError("failed to generate valid ARC input")
