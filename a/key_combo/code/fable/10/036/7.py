from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _enclosed_chambers(grid: np.ndarray, wall: int) -> list[np.ndarray]:
    h, w = grid.shape
    passable = grid != wall
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return _components(enclosed)


def _full_plus_centers(grid: np.ndarray, chamber_mask: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not chamber_mask[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            ok = True
            for dr, dc in PLUS_OFFSETS[1:]:
                nr, nc = r + dr, c + dc
                if not chamber_mask[nr, nc] or int(grid[nr, nc]) != color:
                    ok = False
                    break
            if ok:
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a complete 5-cell plus only as a chamber key, then repaint keyed chambers with their dot color.

    Input structure:
    - The most common color is background, and the largest non-background color forms 1-cell-thick walls around several closed chambers.
    - Inside each chamber there may be scattered dots, or one 5-cell orthogonal plus together with dots of another color.
    - Chambers without a full plus may instead contain only dots or incomplete plus fragments.

    Rule:
    - Find each enclosed chamber by treating wall cells as barriers and collecting non-wall regions not connected to the outer border.
    - If a chamber contains a full orthogonal plus whose center and four orthogonal neighbors share one color, repaint every non-wall cell in that chamber to the chamber's other interior dot color.
    - If a chamber has no full plus, erase every interior colored mark in that chamber back to the background; walls stay unchanged.

    Key insight:
    - The plus color is never the paint color: the plus only unlocks the chamber, and the actual fill color comes from the separate scattered dots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"
    wall = max(colors, key=lambda color: int(np.count_nonzero(g == color)))

    out = g.copy()
    for chamber in _enclosed_chambers(g, wall):
        chamber_mask = _coords_mask(g.shape, chamber)
        plus_centers = _full_plus_centers(g, chamber_mask, bg)
        assert len(plus_centers) <= 1, "each chamber must contain at most one full plus"
        if plus_centers:
            pr, pc = plus_centers[0]
            plus_color = int(g[pr, pc])
            chamber_colors = {
                int(g[r, c])
                for r, c in chamber
                if int(g[r, c]) not in (bg, wall, plus_color)
            }
            assert len(chamber_colors) == 1, "a keyed chamber must have exactly one dot color"
            fill_color = chamber_colors.pop()
            out[chamber_mask] = fill_color
        else:
            erase_mask = chamber_mask & (g != bg)
            out[erase_mask] = bg

    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _draw_wall_box(grid: np.ndarray, box: tuple[int, int, int, int], wall: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall


def _candidate_cells(inner_box: tuple[int, int, int, int], require_margin: bool = False) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = inner_box
    if require_margin:
        return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _would_create_plus(points: set[tuple[int, int]], pos: tuple[int, int]) -> bool:
    test = set(points)
    test.add(pos)
    for r, c in test:
        if all((r + dr, c + dc) in test for dr, dc in PLUS_OFFSETS):
            return True
    return False


def _place_safe_dots(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    color: int,
    count_lo: int,
    count_hi: int,
) -> set[tuple[int, int]] | None:
    free = [cell for cell in cells if cell not in reserved]
    random.shuffle(free)
    dots: set[tuple[int, int]] = set()
    target = random.randint(count_lo, min(count_hi, len(free)))
    for cell in free:
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) <= 1 for rr, cc in dots):
            continue
        if _would_create_plus(dots, cell):
            continue
        dots.add(cell)
        if len(dots) == target:
            break
    if len(dots) < target:
        return None
    for r, c in dots:
        grid[r, c] = color
    return dots


def _make_key_chamber(
    grid: np.ndarray,
    inner_box: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> dict | None:
    cells = _candidate_cells(inner_box)
    centers = _candidate_cells(inner_box, require_margin=True)
    if not centers:
        return None
    plus_color, dot_color = random.sample([c for c in PALETTE if c not in (bg, wall)], 2)
    center = random.choice(centers)
    plus_cells = {(center[0] + dr, center[1] + dc) for dr, dc in PLUS_OFFSETS}
    for r, c in plus_cells:
        grid[r, c] = plus_color
    dots = _place_safe_dots(grid, cells, plus_cells, dot_color, 2, 5)
    if dots is None:
        return None
    inner_area = (inner_box[2] - inner_box[0] + 1) * (inner_box[3] - inner_box[1] + 1)
    if inner_area - len(plus_cells) - len(dots) < 4:
        return None
    return {
        "kind": "key",
        "plus_color": plus_color,
        "dot_color": dot_color,
        "center": center,
        "plus_cells": plus_cells,
        "dots": dots,
    }


def _make_dots_chamber(
    grid: np.ndarray,
    inner_box: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> dict | None:
    cells = _candidate_cells(inner_box)
    dot_color = random.choice([c for c in PALETTE if c not in (bg, wall)])
    dots = _place_safe_dots(grid, cells, set(), dot_color, 2, 5)
    if dots is None:
        return None
    return {"kind": "dots", "dot_color": dot_color, "dots": dots}


def _make_broken_chamber(
    grid: np.ndarray,
    inner_box: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> dict | None:
    centers = _candidate_cells(inner_box, require_margin=True)
    cells = _candidate_cells(inner_box)
    if not centers:
        return None
    frag_color, dot_color = random.sample([c for c in PALETTE if c not in (bg, wall)], 2)
    center = random.choice(centers)
    arm_offsets = list(PLUS_OFFSETS[1:])
    random.shuffle(arm_offsets)
    keep_count = random.choice((2, 3))
    fragment = {(center[0], center[1])}
    for dr, dc in arm_offsets[:keep_count]:
        fragment.add((center[0] + dr, center[1] + dc))
    for r, c in fragment:
        grid[r, c] = frag_color
    dots = _place_safe_dots(grid, cells, fragment, dot_color, 1, 4)
    if dots is None:
        return None
    return {
        "kind": "broken",
        "frag_color": frag_color,
        "dot_color": dot_color,
        "center": center,
        "fragment": fragment,
        "dots": dots,
    }


def _place_boxes(height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w in sorted(sizes, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(1, height - outer_h):
            for c0 in range(1, width - outer_w):
                box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                if any(_boxes_conflict(box, other, pad=1) for other in placed):
                    continue
                candidates.append(box)
        if not candidates:
            return None
        placed.append(random.choice(candidates))
    random.shuffle(placed)
    return placed


def _build_expected(inp: np.ndarray, chambers: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["inner_box"]
        if chamber["kind"] == "key":
            out[r0 : r1 + 1, c0 : c1 + 1] = chamber["dot_color"]
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _validate_semantics(inp: np.ndarray, chambers: list[dict], bg: int, wall: int) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    expected = _build_expected(inp, chambers, bg)
    if not np.array_equal(out, expected):
        return False

    detected = _enclosed_chambers(inp, wall)
    if len(detected) != len(chambers):
        return False

    total_filled = 0
    total_erased = 0
    broken_seen = False
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["inner_box"]
        inner_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        inner_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if chamber["kind"] == "key":
            if chamber["plus_color"] == chamber["dot_color"]:
                return False
            if len(set(map(int, inner_out.ravel()))) != 1:
                return False
            if int(inner_out[0, 0]) != chamber["dot_color"]:
                return False
            total_filled += int(np.count_nonzero(inner_in == bg))
        else:
            if not np.all(inner_out == bg):
                return False
            total_erased += int(np.count_nonzero(inner_in != bg))
            if chamber["kind"] == "broken":
                broken_seen = True

    plus_total = 0
    for comp in detected:
        mask = _coords_mask(inp.shape, comp)
        plus_total += len(_full_plus_centers(inp, mask, bg))

    return (
        plus_total == sum(chamber["kind"] == "key" for chamber in chambers)
        and total_filled >= 4
        and total_erased >= 3
        and broken_seen
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(19, 28)
    width = random.randint(19, 28)
    bg, wall = random.sample(PALETTE, 2)
    chamber_count = 3 if random.random() < 0.7 else 4
    kinds = ["key", "dots", "broken"] + [random.choice(("key", "dots", "broken")) for _ in range(chamber_count - 3)]
    random.shuffle(kinds)

    sizes = []
    for _ in range(chamber_count):
        outer_h = random.randint(6, 9)
        outer_w = random.randint(6, 9)
        sizes.append((outer_h, outer_w))
    boxes = _place_boxes(height, width, sizes)
    if boxes is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    chambers: list[dict] = []
    for kind, box in zip(kinds, boxes):
        _draw_wall_box(inp, box, wall)
        r0, c0, r1, c1 = box
        inner_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        chamber_grid = inp.copy()
        if kind == "key":
            meta = _make_key_chamber(chamber_grid, inner_box, bg, wall)
        elif kind == "dots":
            meta = _make_dots_chamber(chamber_grid, inner_box, bg, wall)
        else:
            meta = _make_broken_chamber(chamber_grid, inner_box, bg, wall)
        if meta is None:
            return None
        inp = chamber_grid
        meta["box"] = box
        meta["inner_box"] = inner_box
        chambers.append(meta)

    if not _validate_semantics(inp, chambers, bg, wall):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-and-plus ARC input")
