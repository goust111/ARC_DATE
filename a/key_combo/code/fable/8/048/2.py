from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
KEY_COLOR = 8
OBJECT_COLOR = 5
VALUE_COLORS = [1, 2, 3, 4, 6, 7, 9]
HOLE_COUNTS = [1, 2, 3, 4, 5]

MAX_OBJECT_SHAPE_ATTEMPTS = 80
MAX_PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _count_enclosed_holes(local_object: np.ndarray) -> int:
    h, w = local_object.shape
    bg_mask = ~local_object
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_boundary = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_boundary:
                holes += 1
    return holes


def _component_hole_count(grid: np.ndarray, coords: np.ndarray) -> int:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_enclosed_holes(local)


def _parse_key_columns(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    entries: list[dict] = []
    used_heights: set[int] = set()
    for c in range(w):
        if int(grid[0, c]) != KEY_COLOR:
            continue
        end = 0
        while end < h and int(grid[end, c]) != BG:
            end += 1
        if end < 2:
            continue
        if not np.all(grid[: end - 1, c] == KEY_COLOR):
            continue
        value = int(grid[end - 1, c])
        if value in (BG, KEY_COLOR):
            continue
        if end < h and not np.all(grid[end:, c] == BG):
            continue
        height = end - 1
        if height in used_heights:
            raise AssertionError("duplicate key height")
        used_heights.add(height)
        entries.append({"col": c, "height": height, "value": value})
    if not entries:
        raise AssertionError("expected at least one key column")
    return sorted(entries, key=lambda item: item["col"])


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray holed object by looking up its enclosed-hole count in the left key columns.

    Input structure:
    - Black background fills the canvas.
    - On the left are several hanging key columns: each starts at the top row, has a teal code segment, and ends with one value-color cell.
    - On the right are separate gray connected objects with enclosed black holes and also edge notches that stay open to the outside.

    Rule:
    - Read every left key column as height -> color, where the key is the number of teal cells above the value cell.
    - For each gray object, count only its enclosed holes; open notches do not count.
    - If that hole count appears among the keys, recolor the whole gray object with the mapped value color; otherwise erase the object to black.
    - Remove all key columns in the output, while holes and notches remain black.

    Key insight:
    - The left columns are a lookup table, not an ordered palette, and the object's true key is its enclosed-hole count rather than every visible cavity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    key_entries = _parse_key_columns(g)
    lookup = {entry["height"]: entry["value"] for entry in key_entries}

    out = np.full_like(g, BG)
    gray_components = _components_from_mask(g == OBJECT_COLOR)
    assert gray_components, "expected gray objects"
    for coords in gray_components:
        holes = _component_hole_count(g, coords)
        color = lookup.get(holes)
        if color is None:
            continue
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _slot_layout(hole_count: int) -> tuple[int, int]:
    options = {
        1: [(1, 1), (1, 2)],
        2: [(1, 2), (2, 1), (2, 2)],
        3: [(2, 2)],
        4: [(2, 2), (2, 3)],
        5: [(2, 3)],
    }
    return random.choice(options[hole_count])


def _random_hole_size() -> tuple[int, int]:
    return random.choice([(1, 1), (1, 1), (1, 2), (2, 1), (2, 2)])


def _hole_slot_origins(slot_rows: int, slot_cols: int, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + 3 * r, left + 3 * c) for r in range(slot_rows) for c in range(slot_cols)]


def _add_notches(patch: np.ndarray, anchors: list[tuple[int, int]]) -> None:
    h, w = patch.shape
    n_notches = random.randint(1, 3)
    for edge in random.sample(["top", "bottom", "left", "right"], k=n_notches):
        if edge in {"top", "bottom"}:
            depth = random.randint(1, 2)
            width = random.randint(1, 2)
            base_c = random.choice(anchors)[1] + random.randint(0, 1)
            c0 = max(1, min(w - width - 1, base_c))
            rows = slice(0, depth) if edge == "top" else slice(h - depth, h)
            patch[rows, c0 : c0 + width] = BG
        else:
            depth = random.randint(1, 2)
            width = random.randint(1, 2)
            base_r = random.choice(anchors)[0] + random.randint(0, 1)
            r0 = max(1, min(h - width - 1, base_r))
            cols = slice(0, depth) if edge == "left" else slice(w - depth, w)
            patch[r0 : r0 + width, cols] = BG


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = patch
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    return np.array(out, dtype=int, copy=True)


def _make_object_patch(hole_count: int) -> np.ndarray | None:
    for _ in range(MAX_OBJECT_SHAPE_ATTEMPTS):
        slot_rows, slot_cols = _slot_layout(hole_count)
        top = 2
        bottom = 2
        left = 2
        right = 2
        h = top + 3 * (slot_rows - 1) + 2 + bottom
        w = left + 3 * (slot_cols - 1) + 2 + right
        patch = np.full((h, w), OBJECT_COLOR, dtype=int)
        anchors = _hole_slot_origins(slot_rows, slot_cols, top, left)
        chosen = random.sample(anchors, hole_count)
        for sr, sc in chosen:
            hh, ww = _random_hole_size()
            dr = 0 if hh == 2 else random.randint(0, 1)
            dc = 0 if ww == 2 else random.randint(0, 1)
            patch[sr + dr : sr + dr + hh, sc + dc : sc + dc + ww] = BG
        _add_notches(patch, chosen)
        patch = _random_transform(patch)
        comps = _components_from_mask(patch == OBJECT_COLOR)
        if len(comps) != 1:
            continue
        if _count_enclosed_holes(patch == OBJECT_COLOR) != hole_count:
            continue
        if patch.shape[0] > 14 or patch.shape[1] > 14:
            continue
        if np.count_nonzero(patch == BG) <= hole_count:
            continue
        return patch
    return None


def _plan_counts() -> tuple[dict[int, int], list[int]]:
    key_count = random.randint(2, 4)
    key_heights = random.sample(HOLE_COUNTS, key_count)
    value_colors = random.sample(VALUE_COLORS, key_count)
    lookup = dict(zip(key_heights, value_colors))

    dup = random.choice(key_heights)
    other_choices = [v for v in key_heights if v != dup]
    other = random.choice(other_choices) if other_choices else dup
    unmatched_options = [v for v in HOLE_COUNTS if v not in lookup]
    unmatched = random.choice(unmatched_options)

    object_counts = [dup, dup, other, unmatched]
    if random.random() < 0.35:
        extras = key_heights + unmatched_options
        object_counts.append(random.choice(extras))
    random.shuffle(object_counts)
    return lookup, object_counts


def _place_objects(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
    min_col: int,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].size, reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph + 1):
            for c0 in range(min_col, w - pw + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + ph + 1)
                cc1 = min(w, c0 + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True

    return [pos for pos in positions if pos is not None]


def _key_layout(lookup: dict[int, int]) -> list[dict]:
    start_col = random.randint(0, 1)
    order = list(lookup.items())
    random.shuffle(order)
    entries = []
    for idx, (height, value) in enumerate(order):
        c = start_col + 2 * idx
        entries.append({"col": c, "height": height, "value": value})
    return entries


def _paint_keys(inp: np.ndarray, entries: list[dict]) -> None:
    for entry in entries:
        c = entry["col"]
        height = entry["height"]
        inp[:height, c] = KEY_COLOR
        inp[height, c] = entry["value"]


def _manual_output(
    shape: tuple[int, int],
    lookup: dict[int, int],
    objects: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in objects:
        color = lookup.get(obj["hole_count"])
        if color is None:
            continue
        patch = obj["patch"]
        mask = patch == OBJECT_COLOR
        r0 = obj["r0"]
        c0 = obj["c0"]
        view = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[mask] = color
    return out


def _validate_scene(inp: np.ndarray, lookup: dict[int, int], objects: list[dict], expected: np.ndarray) -> bool:
    try:
        parsed = _parse_key_columns(inp)
        parsed_lookup = {entry["height"]: entry["value"] for entry in parsed}
        if parsed_lookup != lookup:
            return False
        comps = _components_from_mask(inp == OBJECT_COLOR)
        if len(comps) != len(objects):
            return False
        actual_counts = sorted(_component_hole_count(inp, comp) for comp in comps)
        planned_counts = sorted(obj["hole_count"] for obj in objects)
        if actual_counts != planned_counts:
            return False
        matched_counts = [obj["hole_count"] for obj in objects if obj["hole_count"] in lookup]
        unmatched_counts = [obj["hole_count"] for obj in objects if obj["hole_count"] not in lookup]
        if not matched_counts or not unmatched_counts:
            return False
        if len(matched_counts) == len(set(matched_counts)):
            return False
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False
        for obj in objects:
            patch = obj["patch"]
            mask = patch == OBJECT_COLOR
            r0 = obj["r0"]
            c0 = obj["c0"]
            view = actual[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
            if obj["hole_count"] in lookup:
                if not np.all(view[mask] == lookup[obj["hole_count"]]):
                    return False
            else:
                if np.any(view[mask] != BG):
                    return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    lookup, object_counts = _plan_counts()
    patches = []
    for hole_count in object_counts:
        patch = _make_object_patch(hole_count)
        if patch is None:
            return None
        patches.append(patch)

    max_key_height = max(lookup) + 1
    min_height = max(20, max(max_key_height + 2, max(p.shape[0] for p in patches) + 4))

    key_entries = _key_layout(lookup)
    right_start = max(entry["col"] for entry in key_entries) + random.randint(3, 5)
    min_width = max(24, right_start + max(p.shape[1] for p in patches) + 2)
    if min_height > 30 or min_width > 30:
        return None

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        h = random.randint(min_height, 30)
        w = random.randint(min_width, 30)
        positions = _place_objects((h, w), patches, right_start)
        if positions is None:
            continue

        inp = np.full((h, w), BG, dtype=int)
        _paint_keys(inp, key_entries)
        objects = []
        for patch, hole_count, (r0, c0) in zip(patches, object_counts, positions):
            view = inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
            mask = patch == OBJECT_COLOR
            view[mask] = OBJECT_COLOR
            objects.append({"patch": patch, "hole_count": hole_count, "r0": r0, "c0": c0})

        expected = _manual_output(inp.shape, lookup, objects)
        if _validate_scene(inp, lookup, objects, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-column hole-lookup puzzle")
