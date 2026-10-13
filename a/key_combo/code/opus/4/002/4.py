from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FG = 3
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GROUP_LEN_MIN = 3
GROUP_LEN_MAX = 8
SHIFT_ATTEMPTS = 80
GEN_ATTEMPTS = 200
MAX_INPUT_WIDTH = 30


TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "widths": (2, 2, 5, 4),
        "packed": (
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1),
            (0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
    },
    {
        "widths": (2, 2, 2, 5, 5),
        "packed": (
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1),
            (1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0),
        ),
    },
    {
        "widths": (5, 2, 2),
        "packed": (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 1),
            (0, 0, 0, 0, 0, 1, 0, 0, 1),
            (0, 0, 0, 0, 0, 1, 1, 1, 1),
            (0, 0, 0, 0, 0, 0, 1, 1, 0),
            (1, 1, 0, 0, 1, 1, 1, 1, 1),
            (0, 1, 1, 1, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
    },
    {
        "widths": (2, 2, 3),
        "packed": (
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 1, 1, 0, 0, 0),
            (1, 0, 0, 1, 0, 0, 0),
            (1, 1, 1, 1, 0, 0, 0),
            (0, 1, 1, 0, 0, 0, 0),
            (1, 1, 1, 1, 1, 0, 1),
            (0, 0, 0, 0, 1, 1, 1),
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0),
        ),
    },
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    return sum(
        0 <= r + dr < h and 0 <= c + dc < w and bool(mask[r + dr, c + dc])
        for dr, dc in DIR4
    )


def _fg_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted({int(v) for v in grid.ravel() if int(v) != bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    return int(colors[0])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _enclosed_background_regions(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    enclosed: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                enclosed.append(np.array(coords, dtype=int))
    return enclosed


def _fill_enclosed_regions(grid: np.ndarray, bg: int, fill_color: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for coords in _enclosed_background_regions(out, bg):
        out[coords[:, 0], coords[:, 1]] = int(fill_color)
    return out


def _column_groups(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    nonempty = np.any(grid != int(bg), axis=0)
    groups: list[np.ndarray] = []
    c = 0
    while c < grid.shape[1]:
        if not nonempty[c]:
            c += 1
            continue
        c0 = c
        while c < grid.shape[1] and nonempty[c]:
            c += 1
        groups.append(np.array(grid[:, c0:c], dtype=int, copy=True))
    assert groups, "expected at least one foreground group"
    return groups


def _endpoint_row(group: np.ndarray, bg: int, side: str) -> int:
    h, w = group.shape
    c = 0 if side == "left" else w - 1
    rows: list[int] = []
    mask = group != int(bg)
    for r in range(h):
        if not mask[r, c]:
            continue
        if _degree(mask, r, c) == 1:
            rows.append(r)
    assert len(rows) == 1, f"expected one {side} endpoint"
    return int(rows[0])


def _shift_group(group: np.ndarray, bg: int, dr: int) -> np.ndarray:
    shifted = np.full_like(group, int(bg))
    coords = np.argwhere(group != int(bg))
    nr = coords[:, 0] + int(dr)
    assert int(nr.min()) >= 0 and int(nr.max()) < group.shape[0], "shifted group left bounds"
    shifted[nr, coords[:, 1]] = group[coords[:, 0], coords[:, 1]]
    return shifted


def _pack_groups(groups: list[np.ndarray], bg: int, *, align: bool) -> np.ndarray:
    h = int(groups[0].shape[0])
    out_w = sum(int(group.shape[1]) for group in groups)
    out = np.full((h, out_w), int(bg), dtype=int)
    c = 0
    first = np.array(groups[0], dtype=int, copy=True)
    out[:, : first.shape[1]] = first
    prev_right = _endpoint_row(first, bg, "right")
    c += first.shape[1]
    for group in groups[1:]:
        shifted = np.array(group, dtype=int, copy=True)
        if align:
            dr = prev_right - _endpoint_row(shifted, bg, "left")
            shifted = _shift_group(shifted, bg, dr)
        out[:, c : c + shifted.shape[1]] = shifted
        prev_right = _endpoint_row(shifted, bg, "right")
        c += shifted.shape[1]
    return out


def _is_simple_group(group01: np.ndarray) -> bool:
    mask = group01.astype(bool)
    coords = np.argwhere(mask)
    if not (GROUP_LEN_MIN <= len(coords) <= GROUP_LEN_MAX):
        return False
    comps = _component_coords(mask)
    if len(comps) != 1:
        return False
    degrees = [_degree(mask, int(r), int(c)) for r, c in coords]
    if any(d not in (1, 2) for d in degrees) or degrees.count(1) != 2:
        return False
    left = [r for r in range(group01.shape[0]) if mask[r, 0] and _degree(mask, r, 0) == 1]
    right = [
        r
        for r in range(group01.shape[0])
        if mask[r, group01.shape[1] - 1] and _degree(mask, r, group01.shape[1] - 1) == 1
    ]
    return len(left) == 1 and len(right) == 1 and not _enclosed_background_regions(group01, 0)


def _bad_degree_score(group01: np.ndarray) -> int:
    mask = group01.astype(bool)
    score = 0
    for r, c in np.argwhere(mask):
        score += max(0, _degree(mask, int(r), int(c)) - 2)
    h, w = group01.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if int(group01[r : r + 2, c : c + 2].sum()) == 4:
                score += 1
    return int(score)


def _split_packed(packed01: np.ndarray, widths: tuple[int, ...]) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    c = 0
    for width in widths:
        groups.append(np.array(packed01[:, c : c + int(width)], dtype=int, copy=True))
        c += int(width)
    assert c == packed01.shape[1]
    return groups


def _render_separated(groups: list[np.ndarray], separators: list[int]) -> np.ndarray:
    width = sum(int(group.shape[1]) for group in groups) + sum(separators)
    canvas = np.full((groups[0].shape[0], width), BG, dtype=int)
    c = 0
    for i, group in enumerate(groups):
        patch = np.where(group != BG, FG, BG)
        canvas[:, c : c + patch.shape[1]] = patch
        c += patch.shape[1]
        if i < len(separators):
            c += separators[i]
    return canvas


def _valid_shifts(group: np.ndarray) -> list[int]:
    rows = np.argwhere(group != BG)[:, 0]
    lo = -int(rows.min())
    hi = int(group.shape[0] - 1 - rows.max())
    return list(range(lo, hi + 1))


def _materialize_template() -> tuple[list[np.ndarray], np.ndarray]:
    spec = random.choice(TEMPLATES)
    packed01 = np.array(spec["packed"], dtype=int)
    widths = tuple(int(w) for w in spec["widths"])
    if random.random() < 0.5:
        packed01 = np.flipud(packed01)
    if random.random() < 0.5:
        packed01 = np.fliplr(packed01)
        widths = tuple(reversed(widths))
    if packed01.shape[0] == 8 and random.random() < 0.6:
        top = random.randint(0, 1)
        bottom = 1 - top
        packed01 = np.pad(packed01, ((top, bottom), (0, 0)), constant_values=0)
    groups = _split_packed(packed01, widths)
    for group in groups:
        assert _is_simple_group(group)
    expected01 = _pack_groups(groups, 0, align=True)
    assert np.array_equal(expected01, packed01)
    holes = _enclosed_background_regions(packed01, 0)
    assert len(holes) == 1 and len(holes[0]) >= 2
    return groups, packed01


def _validate_instance(groups: list[np.ndarray], packed01: np.ndarray, inp: np.ndarray) -> None:
    packed = np.where(packed01 != 0, FG, BG)
    expected = _fill_enclosed_regions(packed, BG, FG)
    actual = main(inp)
    assert np.array_equal(actual, expected), "solver/output mismatch"
    naive = _pack_groups(groups, BG, align=False)
    naive_filled = _fill_enclosed_regions(naive, BG, FG)
    assert not np.array_equal(naive_filled, expected), "endpoint alignment must matter"
    assert not np.array_equal(expected, packed), "flood fill must change the packed contour"
    assert len(_enclosed_background_regions(packed, BG)) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack separated path fragments by matching endpoint rows, then fill the bounded compartment they create.

    Input structure:
    - The grid is a short horizontal strip with black background.
    - A single green color forms 3 to 5 path-like column groups separated by all-black columns.
    - Each group is a 4-connected open segment and has one degree-1 endpoint on its left edge and one on its right edge.

    Rule:
    - Remove the separator columns and keep the first group at its original vertical position.
    - For each later group, shift it vertically so its left endpoint lands on the previous packed group's right endpoint row, then place it immediately to the right.
    - After all groups are packed, treat the green pixels as barriers and flood-fill every black 4-connected region that does not touch the grid border with green.

    Key insight:
    - No isolated fragment encloses anything on its own; the enclosed region appears only after the endpoint-aligned packing step, so the solver must first reconstruct the combined contour and only then fill its interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    fg = _fg_color(g, bg)
    groups = _column_groups(g, bg)
    packed = _pack_groups(groups, bg, align=True)
    return _fill_enclosed_regions(packed, bg, fg)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        base_groups, packed01 = _materialize_template()
        separators = [random.randint(1, 3) for _ in range(len(base_groups) - 1)]
        if sum(int(group.shape[1]) for group in base_groups) + sum(separators) > MAX_INPUT_WIDTH:
            continue

        for _shift_try in range(SHIFT_ATTEMPTS):
            shifted_groups = [np.where(base_groups[0] != 0, FG, BG)]
            any_nonzero = False
            for group01 in base_groups[1:]:
                group = np.where(group01 != 0, FG, BG)
                shifts = _valid_shifts(group)
                nonzero = [dr for dr in shifts if dr != 0]
                if nonzero and random.random() < 0.85:
                    dr = random.choice(nonzero)
                else:
                    dr = random.choice(shifts)
                any_nonzero |= dr != 0
                shifted_groups.append(_shift_group(group, BG, dr))
            if not any_nonzero:
                continue

            inp = _render_separated(shifted_groups, separators)
            try:
                _validate_instance(shifted_groups, packed01, inp)
            except AssertionError:
                continue
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
