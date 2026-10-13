from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_points(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        pts.add((int(r), int(c0)))
        pts.add((int(r), int(c1)))
    for c in range(c0, c1 + 1):
        pts.add((int(r0), int(c)))
        pts.add((int(r1), int(c)))
    return pts


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY, AZURE):
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(pts) != _frame_points(r0, c0, r1, c1):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_non_bg = {int(v) for v in inner.reshape(-1) if int(v) != BG}
            if not inner_non_bg:
                kind = "empty"
            elif inner_non_bg <= {GREY}:
                kind = "sieve"
            else:
                kind = "container"
            frames.append(
                {
                    "color": color,
                    "kind": kind,
                    "bbox": (r0, c0, r1, c1),
                    "inner": inner.copy(),
                }
            )
    return frames


def _find_pairs(grid: np.ndarray) -> list[dict[str, object]]:
    frames = _find_frames(grid)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    pairs: list[dict[str, object]] = []
    for color, items in by_color.items():
        containers = [item for item in items if item["kind"] == "container"]
        sieves = [item for item in items if item["kind"] == "sieve"]
        assert len(containers) == 1, f"expected one container for frame color {color}"
        assert len(sieves) == 1, f"expected one sieve for frame color {color}"
        pairs.append({"color": color, "container": containers[0], "sieve": sieves[0]})

    assert pairs, "expected at least one container/sieve pair"
    pairs.sort(key=lambda pair: tuple(int(v) for v in pair["container"]["bbox"][:2]))
    return pairs


def _find_output_slots(grid: np.ndarray) -> list[tuple[slice, slice]]:
    lines: list[tuple[int, int, int]] = []
    for pts in _components(grid == AZURE):
        r0, c0, r1, c1 = _bbox(pts)
        if c0 != c1:
            continue
        if set(pts) != {(r, c0) for r in range(r0, r1 + 1)}:
            continue
        lines.append((r0, r1, c0))

    assert lines, "expected azure slot separators"

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r0, r1, col in lines:
        groups[(r0, r1)].append(int(col))

    row_span = max(groups, key=lambda key: (key[1], len(groups[key])))
    cols = sorted(groups[row_span])
    assert len(cols) >= 2, "need at least two boundary lines to form one slot"

    widths = [cols[i + 1] - cols[i] - 1 for i in range(len(cols) - 1)]
    assert all(width > 0 for width in widths), "slot widths must be positive"
    assert len(set(widths)) == 1, "all output slots must share one width"

    r0, r1 = row_span
    return [(slice(r0, r1 + 1), slice(cols[i] + 1, cols[i + 1])) for i in range(len(cols) - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each container with the same-colored sieve frame, keep only container pixels
    under the sieve's grey holes, and project those filtered crops into the bottom slots.

    Pair order is the containers' reading order (top-to-bottom, then left-to-right),
    matching the output slots' left-to-right order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pairs = _find_pairs(g)
    slots = _find_output_slots(g)
    assert len(pairs) == len(slots), "pair count and slot count must match"

    out = np.zeros_like(g)
    slot_shape = g[slots[0]]
    for pair, slot in zip(pairs, slots):
        container = np.asarray(pair["container"]["inner"], dtype=int)
        sieve = np.asarray(pair["sieve"]["inner"], dtype=int)
        assert container.shape == sieve.shape, "container and sieve inner sizes must match"
        assert container.shape == slot_shape.shape, "slot size must match frame inner size"
        out[slot] = np.where(sieve == GREY, container, BG)
    return out


def _make_pattern(rng: random.Random, h: int, w: int, palette: list[int]) -> np.ndarray:
    colors = rng.sample(palette, k=rng.randint(2, min(4, len(palette))))
    mode = rng.choice(("cells", "rows", "cols", "blocks"))

    if mode == "cells":
        arr = np.array([[rng.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)
    elif mode == "rows":
        row_colors = [rng.choice(colors) for _ in range(h)]
        arr = np.repeat(np.array(row_colors, dtype=int)[:, None], w, axis=1)
        for _ in range(rng.randint(1, max(1, h * w // 4))):
            arr[rng.randrange(h), rng.randrange(w)] = rng.choice(colors)
    elif mode == "cols":
        col_colors = [rng.choice(colors) for _ in range(w)]
        arr = np.repeat(np.array(col_colors, dtype=int)[None, :], h, axis=0)
        for _ in range(rng.randint(1, max(1, h * w // 4))):
            arr[rng.randrange(h), rng.randrange(w)] = rng.choice(colors)
    else:
        arr = np.empty((h, w), dtype=int)
        split_r = rng.randint(1, h - 1)
        split_c = rng.randint(1, w - 1)
        block_colors = [rng.choice(colors) for _ in range(4)]
        arr[:split_r, :split_c] = block_colors[0]
        arr[:split_r, split_c:] = block_colors[1]
        arr[split_r:, :split_c] = block_colors[2]
        arr[split_r:, split_c:] = block_colors[3]
        for _ in range(rng.randint(0, max(1, h * w // 5))):
            arr[rng.randrange(h), rng.randrange(w)] = rng.choice(colors)

    if len({int(v) for v in arr.reshape(-1)}) == 1:
        r, c = rng.randrange(h), rng.randrange(w)
        alternatives = [color for color in colors if color != int(arr[r, c])]
        arr[r, c] = rng.choice(alternatives)
    return arr


def _make_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mode = rng.choice(("random", "row_stripes", "col_stripes", "checker", "cross"))
    mask = np.zeros((h, w), dtype=bool)

    if mode == "random":
        density = rng.uniform(0.25, 0.65)
        mask = np.array([[rng.random() < density for _ in range(w)] for _ in range(h)], dtype=bool)
    elif mode == "row_stripes":
        parity = rng.randrange(2)
        for r in range(h):
            if r % 2 == parity:
                mask[r, :] = True
        for _ in range(rng.randint(0, max(1, w // 2))):
            mask[rng.randrange(h), rng.randrange(w)] = True
    elif mode == "col_stripes":
        parity = rng.randrange(2)
        for c in range(w):
            if c % 2 == parity:
                mask[:, c] = True
        for _ in range(rng.randint(0, max(1, h // 2))):
            mask[rng.randrange(h), rng.randrange(w)] = True
    elif mode == "checker":
        offset = rng.randrange(2)
        for r in range(h):
            for c in range(w):
                mask[r, c] = (r + c + offset) % 2 == 0
    else:
        center_r, center_c = h // 2, w // 2
        mask[center_r, :] = True
        mask[:, center_c] = True
        if h == w and rng.random() < 0.5:
            for i in range(min(h, w)):
                mask[i, i] = True
        for _ in range(rng.randint(0, max(1, (h + w) // 4))):
            mask[rng.randrange(h), rng.randrange(w)] = True

    if not bool(mask.any()):
        mask[rng.randrange(h), rng.randrange(w)] = True
    if bool(mask.all()):
        mask[rng.randrange(h), rng.randrange(w)] = False
    return mask


def _slot_bounds(slot_left: int, slot_row: int, inner_h: int, inner_w: int, pair_count: int) -> list[tuple[slice, slice]]:
    return [
        (
            slice(slot_row, slot_row + inner_h),
            slice(slot_left + 1 + i * (inner_w + 1), slot_left + 1 + i * (inner_w + 1) + inner_w),
        )
        for i in range(pair_count)
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 3)
    inner_h = rng.randint(3, 5)
    inner_w = rng.randint(3, 5)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if pair_count == 1:
        cols = rng.choice((1, 2))
    elif pair_count == 2:
        cols = rng.choice((2, 3))
    else:
        cols = 3
    rows = (2 * pair_count + cols - 1) // cols

    gap_x = rng.randint(1, 2)
    gap_y = rng.randint(1, 2)
    margin_x = rng.randint(1, 2)
    margin_y = rng.randint(1, 2)
    top_pad = rng.randint(0, 2)
    gap_to_slots = rng.randint(1, 3)

    top_w = cols * outer_w + (cols - 1) * gap_x + 2 * margin_x
    top_h = rows * outer_h + (rows - 1) * gap_y + 2 * margin_y
    slot_total_w = pair_count * inner_w + (pair_count + 1)
    min_w = max(top_w, slot_total_w + 2)
    if min_w > 30:
        return None

    width = rng.randint(min_w, min(30, min_w + 3))
    height = top_pad + top_h + gap_to_slots + inner_h
    if height > 30:
        return None

    slot_left = rng.randint(0, width - slot_total_w)
    slot_row = height - inner_h
    layout_left = rng.randint(0, width - top_w)
    base_r = top_pad + margin_y
    base_c = layout_left + margin_x

    cell_positions = []
    for idx in range(rows * cols):
        rr = idx // cols
        cc = idx % cols
        cell_positions.append((base_r + rr * (outer_h + gap_y), base_c + cc * (outer_w + gap_x)))
    selected_positions = rng.sample(cell_positions, k=2 * pair_count)

    frame_colors = rng.sample(list(FRAME_COLORS), k=pair_count)
    pattern_pool = [color for color in FRAME_COLORS if color not in frame_colors]

    pair_specs: list[dict[str, object]] = []
    objects: list[dict[str, object]] = []
    for color in frame_colors:
        pattern = _make_pattern(rng, inner_h, inner_w, pattern_pool)
        mask = _make_mask(rng, inner_h, inner_w)
        pair_specs.append({"color": color, "pattern": pattern, "mask": mask})
        objects.append({"kind": "container", "color": color, "pattern": pattern})
        objects.append({"kind": "sieve", "color": color, "mask": mask})

    rng.shuffle(objects)
    grid = np.zeros((height, width), dtype=int)
    slots = _slot_bounds(slot_left, slot_row, inner_h, inner_w, pair_count)
    for k in range(pair_count + 1):
        grid[slot_row:, slot_left + k * (inner_w + 1)] = AZURE

    pair_by_color = {int(spec["color"]): spec for spec in pair_specs}
    for obj, (r0, c0) in zip(objects, selected_positions):
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        color = int(obj["color"])
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if obj["kind"] == "container":
            inner[:, :] = np.asarray(obj["pattern"], dtype=int)
            pair_by_color[color]["container_pos"] = (r0, c0)
        else:
            mask = np.asarray(obj["mask"], dtype=bool)
            inner[mask] = GREY
            pair_by_color[color]["sieve_pos"] = (r0, c0)

    expected = np.zeros_like(grid)
    ordered_specs = sorted(
        pair_specs,
        key=lambda spec: tuple(int(v) for v in spec["container_pos"]),
    )
    for spec, slot in zip(ordered_specs, slots):
        expected[slot] = np.where(np.asarray(spec["mask"], dtype=bool), np.asarray(spec["pattern"], dtype=int), BG)

    if int(expected.sum()) == 0:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
