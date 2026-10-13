import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
TRANSFORM_KINDS = ("hflip", "vflip", "rot90")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.max(counts))
    winners = [int(v) for v, c in zip(values, counts) if int(c) == best]
    assert len(winners) == 1
    return winners[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _l_patterns(height: int, width: int) -> list[np.ndarray]:
    patterns = []

    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    patterns.append(mask)

    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[:, -1] = True
    patterns.append(mask)

    mask = np.zeros((height, width), dtype=bool)
    mask[-1, :] = True
    mask[:, 0] = True
    patterns.append(mask)

    mask = np.zeros((height, width), dtype=bool)
    mask[-1, :] = True
    mask[:, -1] = True
    patterns.append(mask)

    return patterns


def _classify_hole(hole_mask: np.ndarray) -> str:
    holes = _components(hole_mask)
    assert len(holes) == 1
    coords = holes[0]
    rows = coords[:, 0]
    cols = coords[:, 1]

    if len(set(map(int, rows))) == 1:
        span = int(cols.max() - cols.min() + 1)
        if coords.shape[0] >= 2 and span == coords.shape[0]:
            return "hflip"

    if len(set(map(int, cols))) == 1:
        span = int(rows.max() - rows.min() + 1)
        if coords.shape[0] >= 2 and span == coords.shape[0]:
            return "vflip"

    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    local = hole_mask[r0 : r1 + 1, c0 : c1 + 1]
    if height >= 2 and width >= 2 and coords.shape[0] == height + width - 1:
        for pattern in _l_patterns(height, width):
            if np.array_equal(local, pattern):
                return "rot90"

    raise AssertionError("unrecognized instruction hole")


def _apply_transform(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.fliplr(patch)
    if kind == "vflip":
        return np.flipud(patch)
    if kind == "rot90":
        return np.rot90(patch, 3)
    raise AssertionError(f"unknown transform kind: {kind}")


def _centered_start(start: int, old_len: int, new_len: int) -> int:
    delta = old_len - new_len
    assert delta % 2 == 0
    return int(start + delta // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each colored rectangle according to the black hole embedded inside it.
    Horizontal line holes trigger a left-right flip, vertical line holes trigger an
    up-down flip, and L-shaped holes trigger a clockwise 90-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    out = np.full_like(g, bg)
    components = _components(g != bg)
    assert components

    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1
        color = next(iter(colors))

        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == color) | (patch == bg))
        assert np.all(patch[0, :] == color)
        assert np.all(patch[-1, :] == color)
        assert np.all(patch[:, 0] == color)
        assert np.all(patch[:, -1] == color)

        hole_mask = patch == bg
        assert hole_mask.any()
        assert not hole_mask[0, :].any()
        assert not hole_mask[-1, :].any()
        assert not hole_mask[:, 0].any()
        assert not hole_mask[:, -1].any()

        kind = _classify_hole(hole_mask)
        transformed = _apply_transform(patch, kind)
        new_h, new_w = transformed.shape
        old_h, old_w = patch.shape
        nr0 = _centered_start(int(r0), int(old_h), int(new_h))
        nc0 = _centered_start(int(c0), int(old_w), int(new_w))
        nr1 = nr0 + new_h
        nc1 = nc0 + new_w

        assert 0 <= nr0 and nr1 <= g.shape[0]
        assert 0 <= nc0 and nc1 <= g.shape[1]

        dest = out[nr0:nr1, nc0:nc1]
        assert not np.any((dest != bg) & (transformed != bg))
        out[nr0:nr1, nc0:nc1] = transformed

    return out


# --------------- generator ---------------

BG_COLOR = 0
FINAL_H_MAX = 30
FINAL_W_MAX = 30
OBJECT_COUNT_RANGE = (3, 5)
RECT_SIDE_MIN = 5
RECT_SIDE_MAX = 9
PLACEMENT_GAP_RANGE = (1, 3)
ROW_SLACK_MAX = 2
MAX_GENERATION_ATTEMPTS = 400


def _balanced_kinds(count: int) -> list[str]:
    base = count // len(TRANSFORM_KINDS)
    remainder = count % len(TRANSFORM_KINDS)
    counts = {kind: base for kind in TRANSFORM_KINDS}
    extras = list(TRANSFORM_KINDS)
    random.shuffle(extras)
    for idx in range(remainder):
        counts[extras[idx]] += 1

    kinds: list[str] = []
    for kind in TRANSFORM_KINDS:
        kinds.extend([kind] * counts[kind])
    random.shuffle(kinds)
    return kinds


def _sample_rot_dims(limit: int) -> tuple[int, int]:
    choices = []
    for h in range(RECT_SIDE_MIN, limit + 1):
        for w in range(RECT_SIDE_MIN, limit + 1):
            if h % 2 == w % 2:
                choices.append((h, w))
    assert choices
    return random.choice(choices)


def _horizontal_hole(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    row = random.randint(1, height - 2)
    length = random.randint(2, width - 3)
    starts = [
        c0
        for c0 in range(1, width - length)
        if 2 * c0 != width - length
    ]
    assert starts
    c0 = random.choice(starts)
    mask[row, c0 : c0 + length] = True
    return mask


def _vertical_hole(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    col = random.randint(1, width - 2)
    length = random.randint(2, height - 3)
    starts = [
        r0
        for r0 in range(1, height - length)
        if 2 * r0 != height - length
    ]
    assert starts
    r0 = random.choice(starts)
    mask[r0 : r0 + length, col] = True
    return mask


def _l_hole(height: int, width: int) -> np.ndarray:
    hole_h = random.randint(2, height - 2)
    hole_w = random.randint(2, width - 2)
    top = random.randint(1, height - hole_h - 1)
    left = random.randint(1, width - hole_w - 1)
    pattern = random.choice(_l_patterns(hole_h, hole_w))
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + hole_h, left : left + hole_w] = pattern
    return mask


def _hole_mask(kind: str, height: int, width: int) -> np.ndarray:
    if kind == "hflip":
        return _horizontal_hole(height, width)
    if kind == "vflip":
        return _vertical_hole(height, width)
    if kind == "rot90":
        return _l_hole(height, width)
    raise AssertionError(f"unexpected kind: {kind}")


def _effect_box(kind: str, height: int, width: int) -> tuple[int, int, int, int]:
    if kind != "rot90":
        return height, width, 0, 0
    side = max(height, width)
    return side, side, (side - height) // 2, (side - width) // 2


def _build_object(kind: str, color: int, size_limit: int) -> dict:
    if kind == "rot90":
        height, width = _sample_rot_dims(size_limit)
    else:
        height = random.randint(RECT_SIDE_MIN, size_limit)
        width = random.randint(RECT_SIDE_MIN, size_limit)

    patch = np.full((height, width), color, dtype=int)
    hole = _hole_mask(kind, height, width)
    patch[hole] = BG_COLOR

    comps = _components(patch != BG_COLOR)
    assert len(comps) == 1
    assert _classify_hole(hole) == kind

    effect_h, effect_w, pad_r, pad_c = _effect_box(kind, height, width)
    return {
        "kind": kind,
        "color": color,
        "patch": patch,
        "effect_h": effect_h,
        "effect_w": effect_w,
        "pad_r": pad_r,
        "pad_c": pad_c,
    }


def _positive_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _layout_objects(objects: list[dict]) -> tuple[int, int] | None:
    rows = []
    row_count = random.randint(1, min(3, len(objects)))
    shuffled = list(objects)
    random.shuffle(shuffled)
    group_sizes = _positive_partition(len(shuffled), row_count)

    start = 0
    max_width = 0
    for size in group_sizes:
        group = shuffled[start : start + size]
        start += size
        random.shuffle(group)

        row_height = max(obj["effect_h"] for obj in group) + random.randint(0, ROW_SLACK_MAX)
        gaps = [random.randint(*PLACEMENT_GAP_RANGE) for _ in range(len(group) - 1)]
        row_width = sum(obj["effect_w"] for obj in group) + sum(gaps)
        offsets = [random.randint(0, row_height - obj["effect_h"]) for obj in group]
        rows.append((group, row_height, row_width, gaps, offsets))
        max_width = max(max_width, row_width)

    v_gaps = [random.randint(*PLACEMENT_GAP_RANGE) for _ in range(len(rows) - 1)]
    total_height = sum(row[1] for row in rows) + sum(v_gaps)
    if total_height > FINAL_H_MAX or max_width > FINAL_W_MAX:
        return None

    cur_r = 0
    for row_idx, (group, row_height, row_width, gaps, offsets) in enumerate(rows):
        cur_c = random.randint(0, max_width - row_width)
        for obj_idx, (obj, row_offset) in enumerate(zip(group, offsets)):
            obj["effect_origin"] = (cur_r + row_offset, cur_c)
            cur_c += obj["effect_w"]
            if obj_idx < len(gaps):
                cur_c += gaps[obj_idx]
        cur_r += row_height
        if row_idx < len(v_gaps):
            cur_r += v_gaps[row_idx]

    return total_height, max_width


def _expected_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BG_COLOR, dtype=int)
    for obj in objects:
        patch = obj["patch"]
        kind = obj["kind"]
        r0, c0 = obj["origin"]
        transformed = _apply_transform(patch, kind)
        nr0 = _centered_start(r0, patch.shape[0], transformed.shape[0])
        nc0 = _centered_start(c0, patch.shape[1], transformed.shape[1])
        dest = out[nr0 : nr0 + transformed.shape[0], nc0 : nc0 + transformed.shape[1]]
        if np.any((dest != BG_COLOR) & (transformed != BG_COLOR)):
            return np.empty((0, 0), dtype=int)
        out[nr0 : nr0 + transformed.shape[0], nc0 : nc0 + transformed.shape[1]] = transformed
    return out


def _gen_attempt(return_objects: bool = False):
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    size_limit = RECT_SIDE_MAX if object_count <= 4 else RECT_SIDE_MAX - 1
    colors = random.sample(range(1, 10), object_count)
    objects = [_build_object(kind, color, size_limit) for kind, color in zip(_balanced_kinds(object_count), colors)]

    layout = _layout_objects(objects)
    if layout is None:
        return None
    base_h, base_w = layout

    colored_sizes = [int(np.count_nonzero(obj["patch"] != BG_COLOR)) for obj in objects]
    total_colored = sum(colored_sizes)
    max_color = max(colored_sizes)
    candidates = [
        (h, w)
        for h in range(base_h, FINAL_H_MAX + 1)
        for w in range(base_w, FINAL_W_MAX + 1)
        if h * w - total_colored > max_color
    ]
    if not candidates:
        return None

    grid_h, grid_w = random.choice(candidates)
    row_shift = random.randint(0, grid_h - base_h)
    col_shift = random.randint(0, grid_w - base_w)
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    for obj in objects:
        eff_r, eff_c = obj["effect_origin"]
        r0 = row_shift + eff_r + obj["pad_r"]
        c0 = col_shift + eff_c + obj["pad_c"]
        obj["origin"] = (r0, c0)
        patch = obj["patch"]
        h, w = patch.shape
        block = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((block != BG_COLOR) & (patch != BG_COLOR)):
            return None
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    if _most_common_color(grid) != BG_COLOR:
        return None
    if len(_components(grid != BG_COLOR)) != object_count:
        return None

    expected = _expected_output(grid.shape, objects)
    if expected.size == 0 or np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    if return_objects:
        return grid, objects
    return grid


def _generate_case(return_objects: bool = False):
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(return_objects=return_objects)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")


def generate_input() -> np.ndarray:
    result = _generate_case(return_objects=False)
    assert isinstance(result, np.ndarray)
    return result
