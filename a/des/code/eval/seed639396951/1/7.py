from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))

# Generator layout controls.
GROUP_GAP = 2
OUTER_MARGIN = 1
GRID_MAX = 42
EXTRA_SLACK_MAX = 6
MAX_GENERATION_ATTEMPTS = 300
ROW_PATTERNS = {
    1: [(1,)],
    2: [(2,), (1, 1)],
    3: [(3,), (2, 1), (1, 2)],
    4: [(4,), (2, 2), (3, 1), (1, 3)],
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(local, border)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only odd-numbered layers in each nested rectangular ring group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rings: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for comp in _components(g != BLACK):
        assert _is_rectangular_ring(comp)
        rings.append((comp, _bbox(comp)))

    bboxes = [box for _, box in rings]
    out = g.copy()
    for comp, box in rings:
        # A ring's layer index is exactly one plus the number of containing rings.
        depth = 1 + sum(_contains(other, box) for other in bboxes)
        if depth % 2 == 0:
            out[comp[:, 0], comp[:, 1]] = BLACK
    return out


def _draw_ring(arr: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    arr[r0, c0 : c1 + 1] = color
    arr[r1, c0 : c1 + 1] = color
    arr[r0 : r1 + 1, c0] = color
    arr[r0 : r1 + 1, c1] = color


def _group_profile(num_groups: int) -> tuple[tuple[int, ...], int]:
    if num_groups == 1:
        return (2, 3, 4, 5), 27
    if num_groups == 2:
        return (2, 3, 4, 5), 19
    if num_groups == 3:
        return (2, 3, 4), 17
    return (2, 3, 4), 15


def _build_group(num_groups: int) -> tuple[np.ndarray, np.ndarray] | None:
    layer_options, outer_max = _group_profile(num_groups)
    layers = random.choice(layer_options)
    inner_max = outer_max - 4 * (layers - 1)
    if inner_max < 3:
        return None

    inner_h = random.randint(3, inner_max)
    inner_w = random.randint(3, inner_max)
    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)
    colors = random.sample(PALETTE, layers)

    inp = np.zeros((outer_h, outer_w), dtype=int)
    out = np.zeros_like(inp)
    for layer_idx, color in enumerate(colors):
        box = (
            2 * layer_idx,
            2 * layer_idx,
            outer_h - 1 - 2 * layer_idx,
            outer_w - 1 - 2 * layer_idx,
        )
        _draw_ring(inp, box, int(color))
        if layer_idx % 2 == 0:
            _draw_ring(out, box, int(color))
    return inp, out


def _compositions(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _pack_groups(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    patterns = list(ROW_PATTERNS[len(groups)])
    random.shuffle(patterns)

    for pattern in patterns:
        indices = list(range(len(groups)))
        random.shuffle(indices)

        rows: list[list[int]] = []
        cursor = 0
        for row_size in pattern:
            rows.append(indices[cursor : cursor + row_size])
            cursor += row_size

        row_widths: list[int] = []
        row_heights: list[int] = []
        for row in rows:
            widths = [groups[idx][0].shape[1] for idx in row]
            heights = [groups[idx][0].shape[0] for idx in row]
            row_widths.append(sum(widths) + GROUP_GAP * max(0, len(row) - 1))
            row_heights.append(max(heights))

        content_w = max(row_widths)
        content_h = sum(row_heights) + GROUP_GAP * max(0, len(rows) - 1)
        min_h = content_h + 2 * OUTER_MARGIN
        min_w = content_w + 2 * OUTER_MARGIN
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        h = min_h + random.randint(0, min(EXTRA_SLACK_MAX, GRID_MAX - min_h))
        w = min_w + random.randint(0, min(EXTRA_SLACK_MAX, GRID_MAX - min_w))
        vertical_slack = _compositions(h - min_h, len(rows) + 1)

        positions: list[tuple[int, int] | None] = [None] * len(groups)
        row_top = OUTER_MARGIN + vertical_slack[0]
        for row_idx, row in enumerate(rows):
            row_h = row_heights[row_idx]
            row_w = row_widths[row_idx]
            start_c = OUTER_MARGIN + random.randint(0, w - 2 * OUTER_MARGIN - row_w)
            c = start_c
            for idx in row:
                patch_h, patch_w = groups[idx][0].shape
                offset_r = random.randint(0, row_h - patch_h)
                positions[idx] = (row_top + offset_r, c)
                c += patch_w + GROUP_GAP
            row_top += row_h
            if row_idx < len(rows) - 1:
                row_top += GROUP_GAP + vertical_slack[row_idx + 1]

        return h, w, [(int(r), int(c)) for r, c in positions if r is not None and c is not None]

    return None


def _gen_attempt() -> np.ndarray | None:
    num_groups = random.choice((1, 2, 3, 4))
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_groups):
        group = _build_group(num_groups)
        if group is None:
            return None
        groups.append(group)

    packed = _pack_groups(groups)
    if packed is None:
        return None

    h, w, positions = packed
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    for (group_in, group_out), (r, c) in zip(groups, positions):
        gh, gw = group_in.shape
        inp[r : r + gh, c : c + gw] = group_in
        expected[r : r + gh, c : c + gw] = group_out

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
