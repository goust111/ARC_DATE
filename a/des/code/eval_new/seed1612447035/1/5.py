from __future__ import annotations

import random

import numpy as np


GREY = 5
MAROON = 9
VISIBLE_COLORS = [0, 1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400

REFLECT = {
    "\\": {
        (0, 1): (1, 0),
        (1, 0): (0, 1),
        (0, -1): (-1, 0),
        (-1, 0): (0, -1),
    },
    "/": {
        (0, 1): (-1, 0),
        (-1, 0): (0, 1),
        (0, -1): (1, 0),
        (1, 0): (0, -1),
    },
}

TRANSFORMS = ("id", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "diag", "anti")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _parse_palette(row: np.ndarray, bg: int) -> dict[frozenset[int], int]:
    mapping: dict[frozenset[int], int] = {}
    c = 0
    while c < row.shape[0]:
        if int(row[c]) == bg:
            c += 1
            continue

        start = c
        while c < row.shape[0] and int(row[c]) != bg:
            c += 1
        block = [int(v) for v in row[start:c].tolist()]
        assert len(block) == 3, "palette blocks must be 1x3"

        left, mixed, right = block
        key = frozenset((left, right))
        assert len(key) == 2, "each palette rule must combine two different colors"
        assert key not in mapping, "palette pair mapping must be unambiguous"
        mapping[key] = mixed

    assert mapping, "expected at least one palette rule"
    return mapping


def _source_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top + on_bottom + on_left + on_right == 1, "sources must be non-corner edge pixels"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def _mirror_orientations(body: np.ndarray) -> dict[tuple[int, int], str]:
    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(body == GREY)}
    mirrors: dict[tuple[int, int], str] = {}
    for r, c in grey_cells:
        assert 0 < r < body.shape[0] - 1 and 0 < c < body.shape[1] - 1, "mirrors must stay interior"
        partners = []
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = r + dr, c + dc
            if (rr, cc) in grey_cells:
                partners.append((rr, cc))
        assert len(partners) == 1, "each grey pixel must belong to exactly one diagonal mirror pair"
        rr, cc = partners[0]
        mirrors[(r, c)] = "\\" if (rr - r) == (cc - c) else "/"
    return mirrors


def _trace_beams(
    body: np.ndarray,
    bg: int,
    sources: dict[tuple[int, int], int],
    mirrors: dict[tuple[int, int], str],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    bg_hits: dict[tuple[int, int], set[int]] = {}
    target_hits: dict[tuple[int, int], set[int]] = {}

    for (r0, c0), color in sources.items():
        dr, dc = _source_direction(body.shape, r0, c0)
        r, c = r0, c0
        seen: set[tuple[int, int, int, int]] = set()

        while True:
            state = (r, c, dr, dc)
            assert state not in seen, "beam path must not loop"
            seen.add(state)

            nr, nc = r + dr, c + dc
            if not _in_bounds(body.shape, nr, nc):
                break
            if (nr, nc) in sources:
                break
            if (nr, nc) in mirrors:
                dr, dc = REFLECT[mirrors[(nr, nc)]][(dr, dc)]
                continue

            cell = int(body[nr, nc])
            if cell == bg:
                bg_hits.setdefault((nr, nc), set()).add(color)
            elif cell == MAROON:
                target_hits.setdefault((nr, nc), set()).add(color)
            else:
                raise AssertionError("main area may only contain sources, mirrors, targets, and background")
            r, c = nr, nc

    return bg_hits, target_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge-emitted colored beams through grey diagonal mirrors and recolor maroon targets via a bottom palette.

    Input structure:
    - bg is the most common color.
    - The last row is a palette row made of 1x3 color blocks, with one bg cell separating adjacent blocks; each block means left-color + right-color = middle-color.
    - Above the palette is the main area, containing colored edge source pixels, 2-pixel grey diagonal mirrors, and interior single-cell maroon targets.

    Rule:
    - Remove the palette row in the output.
    - Each edge source emits one beam orthogonally inward. When the next cell is a grey "\" mirror, the beam turns right<->down or left<->up; when it is a grey "/" mirror, it turns right<->up or left<->down. Mirror cells and source cells are never overwritten.
    - Every traversed bg cell becomes the source color. A maroon target hit by one beam becomes that beam color; a target hit by two different beam colors is recolored with the palette entry for that unordered color pair; an unhit target stays maroon.

    Key insight:
    - The task combines two visible structures: discrete ray tracing controlled by diagonal grey mirror pairs, and a bottom-row lookup table that resolves two-color target overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a palette row"
    bg = _get_bg(g)

    palette = _parse_palette(g[-1], bg)
    body = g[:-1].copy()
    h, w = body.shape

    sources: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            if r not in {0, h - 1} and c not in {0, w - 1}:
                continue
            value = int(body[r, c])
            if value in {bg, GREY, MAROON}:
                continue
            sources[(r, c)] = value
    assert sources, "expected at least one source on the main-area boundary"

    mirrors = _mirror_orientations(body)
    bg_hits, target_hits = _trace_beams(body, bg, sources, mirrors)

    out = body.copy()
    for (r, c), colors in bg_hits.items():
        assert len(colors) == 1, "background beam overlaps are not allowed in valid inputs"
        out[r, c] = next(iter(colors))

    for r_raw, c_raw in np.argwhere(body == MAROON):
        r = int(r_raw)
        c = int(c_raw)
        colors = target_hits.get((r, c), set())
        if not colors:
            continue
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
            continue
        assert len(colors) == 2, "targets may be crossed by at most two beam colors"
        key = frozenset(colors)
        assert key in palette, "missing palette rule for a two-color target overlap"
        out[r, c] = palette[key]

    return out


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "flip_lr":
        return np.fliplr(arr)
    if name == "flip_ud":
        return np.flipud(arr)
    if name == "diag":
        return arr.T
    if name == "anti":
        return np.rot90(arr, 2).T
    raise ValueError(name)


def _build_palette_row(
    width: int,
    bg: int,
    source_a: int,
    source_b: int,
    mix_color: int,
    extra_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    blocks = []
    if rng.random() < 0.5:
        blocks.append([source_a, mix_color, source_b])
    else:
        blocks.append([source_b, mix_color, source_a])

    max_blocks = min(2, (width + 1) // 4)
    if max_blocks >= 2 and len(extra_colors) >= 3 and rng.random() < 0.75:
        left, mid, right = extra_colors[:3]
        if rng.random() < 0.5:
            blocks.append([left, mid, right])
        else:
            blocks.append([right, mid, left])

    rng.shuffle(blocks)
    needed = 3 * len(blocks) + (len(blocks) - 1)
    assert needed <= width, "palette row must fit in the canvas"

    row = np.full(width, bg, dtype=int)
    start = rng.randint(0, width - needed)
    c = start
    for idx, block in enumerate(blocks):
        row[c : c + 3] = np.array(block, dtype=int)
        c += 3
        if idx + 1 < len(blocks):
            c += 1
    return row


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(9, 16)
    width = rng.randint(12, 18)

    color_pool = VISIBLE_COLORS[:]
    rng.shuffle(color_pool)
    bg = color_pool[0]
    source_a, source_b, mix_color = color_pool[1:4]
    extras = color_pool[4:]

    c_mix = rng.randint(2, width - 5)
    c_b = rng.randint(c_mix + 3, width - 2)
    r_mix = rng.randint(3, height - 4)
    r_a = rng.randint(1, r_mix - 2)

    a_source = (r_a, 0)
    b_source = (0, c_b)
    mix_target = (r_mix, c_mix)

    a_path = {(r_a, c) for c in range(1, c_mix + 1)}
    a_path |= {(r, c_mix) for r in range(r_a + 1, height)}
    b_path = {(r, c_b) for r in range(1, r_mix + 1)}
    b_path |= {(r_mix, c) for c in range(0, c_b)}
    if (a_path & b_path) != {mix_target}:
        return None

    mirrors = {
        (r_a, c_mix + 1),
        (r_a + 1, c_mix + 2),
        (r_mix + 1, c_b),
        (r_mix + 2, c_b - 1),
    }
    if any(r in {0, height - 1} or c in {0, width - 1} for r, c in mirrors):
        return None
    if mirrors & (a_path | b_path | {a_source, b_source}):
        return None

    a_candidates = [(r_a, c) for c in range(1, c_mix)]
    a_candidates += [(r, c_mix) for r in range(r_a + 1, height - 1) if r != r_mix]
    b_candidates = [(r, c_b) for r in range(1, r_mix)]
    b_candidates += [(r_mix, c) for c in range(1, c_b) if c != c_mix]
    if not a_candidates or not b_candidates:
        return None

    a_target = rng.choice(a_candidates)
    b_target = rng.choice(b_candidates)

    feature_cells = {a_source, b_source, mix_target, a_target, b_target} | mirrors
    free_targets = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            pos = (r, c)
            if pos in feature_cells or pos in a_path or pos in b_path:
                continue
            free_targets.append(pos)
    spaced_targets = [
        pos
        for pos in free_targets
        if all(abs(pos[0] - rr) + abs(pos[1] - cc) > 1 for rr, cc in feature_cells)
    ]
    if spaced_targets:
        free_targets = spaced_targets
    if not free_targets:
        return None
    untouched = rng.choice(free_targets)

    body = np.full((height, width), bg, dtype=int)
    body[a_source] = source_a
    body[b_source] = source_b
    for cell in mirrors:
        body[cell] = GREY
    for cell in (mix_target, a_target, b_target, untouched):
        body[cell] = MAROON

    labels = np.zeros((height, width), dtype=int)
    labels[mix_target] = 1
    labels[a_target] = 2
    labels[b_target] = 3
    labels[untouched] = 4

    transform = rng.choice(TRANSFORMS)
    body = _apply_transform(body, transform)
    labels = _apply_transform(labels, transform)

    palette = _build_palette_row(body.shape[1], bg, source_a, source_b, mix_color, extras, rng)
    grid = np.vstack([body, palette[None, :]])

    out = main(grid)
    if out.shape != body.shape:
        return None

    mix_pos = np.argwhere(labels == 1)
    a_pos = np.argwhere(labels == 2)
    b_pos = np.argwhere(labels == 3)
    untouched_pos = np.argwhere(labels == 4)
    if not (mix_pos.shape[0] == a_pos.shape[0] == b_pos.shape[0] == untouched_pos.shape[0] == 1):
        return None

    mr, mc = map(int, mix_pos[0])
    ar, ac = map(int, a_pos[0])
    br, bc = map(int, b_pos[0])
    ur, uc = map(int, untouched_pos[0])
    if int(out[mr, mc]) != mix_color:
        return None
    if int(out[ar, ac]) != source_a:
        return None
    if int(out[br, bc]) != source_b:
        return None
    if int(out[ur, uc]) != MAROON:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-mixing ARC input")
