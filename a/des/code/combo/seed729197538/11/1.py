from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
MAROON = 9
FAMILY_COLOR_POOL = (0, 1, 2, 3, 4, 6, 7)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == MAROON)]
    assert len(rows) == 1, "expected one full-width maroon separator"
    return int(rows[0])


def _parse_families(legend: np.ndarray) -> list[tuple[int, frozenset[int]]]:
    h, w = legend.shape
    used = np.zeros_like(legend, dtype=bool)
    blocks: list[tuple[int, int, int]] = []

    for r in range(h - 1):
        for c in range(w - 1):
            if used[r : r + 2, c : c + 2].any():
                continue
            color = int(legend[r, c])
            if color == AZURE:
                continue
            if np.all(legend[r : r + 2, c : c + 2] == color):
                blocks.append((int(r), int(c), color))
                used[r : r + 2, c : c + 2] = True

    blocks.sort()
    assert blocks, "expected at least one legend family"

    families: list[tuple[int, frozenset[int]]] = []
    for r, c, ancestor in blocks:
        band = legend[r : r + 2, c + 2 :]
        members = {int(v) for v in np.unique(band) if int(v) != AZURE}
        members.discard(ancestor)
        assert members, "every family must list at least one member color"
        families.append((ancestor, frozenset(members)))
    return families


def _adjacent_border(work: np.ndarray, comp: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = work.shape
    border: set[tuple[int, int]] = set()
    for r, c in comp:
        for nr, nc in _neighbors4(r, c, h, w):
            if int(work[nr, nc]) == GREY:
                border.add((int(nr), int(nc)))
    return border


def main(grid: np.ndarray) -> np.ndarray:
    """Infer each enclosed region's family from the bottom legend and repaint its grey border."""
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    families = _parse_families(g[sep + 1 :, :])
    work = g[:sep, :]

    out = g.copy()
    non_grey = work != GREY
    for comp in _components(non_grey):
        if any(
            r == 0 or c == 0 or r == work.shape[0] - 1 or c == work.shape[1] - 1
            for r, c in comp
        ):
            continue

        fragment_colors = {int(work[r, c]) for r, c in comp if int(work[r, c]) != AZURE}
        assert fragment_colors, "each enclosed region must contain at least one colored fragment"
        matches = [ancestor for ancestor, members in families if fragment_colors <= members]
        assert len(matches) == 1, "fragment colors must identify a unique family"

        for r, c in _adjacent_border(work, comp):
            out[r, c] = matches[0]
    return out


def _make_family_specs() -> list[dict[str, tuple[int, ...] | int]]:
    colors = random.sample(FAMILY_COLOR_POOL, len(FAMILY_COLOR_POOL))
    family_count = random.choice((2, 3))
    ancestors = colors[:family_count]
    member_pool = colors[family_count:]
    sizes = [2, 3] if family_count == 2 else [1, 1, 2]
    random.shuffle(sizes)

    families: list[dict[str, tuple[int, ...] | int]] = []
    idx = 0
    for ancestor, size in zip(ancestors, sizes, strict=False):
        members = tuple(member_pool[idx : idx + size])
        idx += size
        families.append({"ancestor": int(ancestor), "members": members})
    random.shuffle(families)
    return families


def _rectangle_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 7)
    return np.ones((h, w), dtype=bool)


def _notched_mask() -> np.ndarray:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    mask = np.ones((h, w), dtype=bool)
    notch_h = random.randint(1, h - 2)
    notch_w = random.randint(1, w - 2)
    corner = random.randrange(4)

    if corner == 0:
        mask[:notch_h, :notch_w] = False
    elif corner == 1:
        mask[:notch_h, w - notch_w :] = False
    elif corner == 2:
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False
    return mask


def _random_region_mask() -> np.ndarray:
    return _rectangle_mask() if random.random() < 0.5 else _notched_mask()


def _outline_from_interior(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    interior = np.zeros((h + 2, w + 2), dtype=bool)
    interior[1 : h + 1, 1 : w + 1] = mask
    border = np.zeros_like(interior)

    for r, c in np.argwhere(interior):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c, h + 2, w + 2):
            if not interior[nr, nc]:
                border[nr, nc] = True

    assert np.any(interior) and np.any(border)
    assert not np.any(interior & border)
    return interior, border


def _place_patch(
    canvas: np.ndarray,
    top: int,
    left: int,
    border: np.ndarray,
    fragments: list[tuple[int, tuple[int, int]]],
) -> None:
    ph, pw = border.shape
    canvas[top : top + ph, left : left + pw][border] = GREY
    for color, (r, c) in fragments:
        canvas[top + r, left + c] = int(color)


def _family_assignments(family_count: int, region_count: int) -> list[int]:
    choices = random.sample(range(family_count), 2)
    while len(choices) < region_count:
        choices.append(random.randrange(family_count))
    random.shuffle(choices)
    return choices


def _legend_layout(families: list[dict[str, tuple[int, ...] | int]]) -> tuple[int, int, int]:
    ancestor_col = random.randint(1, 3)
    member_start = ancestor_col + random.randint(3, 5)
    max_members = max(len(family["members"]) for family in families)  # type: ignore[arg-type]
    min_width = member_start + 2 * max_members + 1
    return ancestor_col, member_start, min_width


def _draw_legend(
    grid: np.ndarray,
    sep_row: int,
    families: list[dict[str, tuple[int, ...] | int]],
    ancestor_col: int,
    member_start: int,
) -> None:
    grid[sep_row, :] = MAROON
    base = sep_row + 1
    for i, family in enumerate(families):
        r0 = base + 3 * i
        ancestor = int(family["ancestor"])
        members = list(family["members"])  # type: ignore[arg-type]
        random.shuffle(members)
        grid[r0 : r0 + 2, ancestor_col : ancestor_col + 2] = ancestor
        for j, color in enumerate(members):
            rr = r0 + random.randint(0, 1)
            cc = member_start + 2 * j
            grid[rr, cc] = int(color)


def _gen_attempt() -> np.ndarray | None:
    families = _make_family_specs()
    ancestor_col, member_start, min_width = _legend_layout(families)
    width = random.randint(max(18, min_width), 26)

    legend_height = 3 * len(families) - 1
    main_height = random.randint(10, 17)
    total_height = main_height + 1 + legend_height
    grid = np.full((total_height, width), AZURE, dtype=int)

    sep_row = main_height
    _draw_legend(grid, sep_row, families, ancestor_col, member_start)

    region_count = random.randint(2, 4)
    assignments = _family_assignments(len(families), region_count)
    occupied = np.zeros((main_height, width), dtype=bool)

    expected = grid.copy()
    expected[sep_row, :] = MAROON

    for family_idx in assignments:
        family = families[family_idx]
        interior, border = _outline_from_interior(_random_region_mask())
        ph, pw = interior.shape
        if ph + 2 > main_height or pw + 2 > width:
            return None

        positions = [
            (r, c)
            for r in range(1, main_height - ph)
            for c in range(1, width - pw)
            if not occupied[max(0, r - 1) : min(main_height, r + ph + 1), max(0, c - 1) : min(width, c + pw + 1)].any()
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        occupied[top : top + ph, left : left + pw] = True

        cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        members = list(family["members"])  # type: ignore[arg-type]
        subset_size = random.randint(1, len(members))
        active_colors = random.sample(members, subset_size)
        fragment_count = random.randint(subset_size, min(5, len(cells)))
        fragment_cells = random.sample(cells, fragment_count)

        fragments: list[tuple[int, tuple[int, int]]] = []
        for color, cell in zip(active_colors, fragment_cells, strict=False):
            fragments.append((int(color), cell))
        for cell in fragment_cells[subset_size:]:
            fragments.append((int(random.choice(active_colors)), cell))

        _place_patch(grid, top, left, border, fragments)
        _place_patch(expected, top, left, border, fragments)
        expected[top : top + ph, left : left + pw][border] = int(family["ancestor"])

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
