from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

GEN_RETRIES = 400
BASE_SIZE_RANGE = (3, 5)
HALF_HEIGHT_RANGE = (5, 8)
WIDTH_RANGE = (6, 10)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_irregular_component(cells: set[tuple[int, int]]) -> bool:
    if len(cells) < 3:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    if r0 == r1 or c0 == c1:
        return False
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return bbox_area > len(cells)


def _components4(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: set[tuple[int, int]] = set()
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _xor_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    half = g.shape[0] // 2
    top_fg = g[:half] == BLUE
    bottom_fg = g[half:] == GREEN
    return top_fg ^ bottom_fg


def _line_mask_from_xor(xor_mask: np.ndarray) -> np.ndarray:
    h, w = xor_mask.shape
    lines = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(xor_mask):
        r = int(r)
        c = int(c)
        if r == 0 or r == h - 1:
            lines[:, c] = True
        if c == 0 or c == w - 1:
            lines[r, :] = True
    return lines


def _expected_output_from_xor(xor_mask: np.ndarray) -> np.ndarray:
    h, w = xor_mask.shape
    out = np.zeros((h, w), dtype=int)
    lines = _line_mask_from_xor(xor_mask)
    out[lines] = YELLOW
    out[xor_mask] = RED
    for r, c in np.argwhere(xor_mask):
        r = int(r)
        c = int(c)
        if r in (0, h - 1) or c in (0, w - 1):
            continue
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if out[rr, cc] == BLACK:
                    out[rr, cc] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the stacked half-masks, turn their XOR into red markers, then branch by marker boundary status.

    Input structure:
    - The input has even height and a black background.
    - The top half contains blue foreground pixels, and the bottom half contains green foreground pixels.
    - The two halves are the same size and are aligned cell-by-cell.

    Rule:
    - Compute the XOR of the top-half blue mask and bottom-half green mask.
    - Create a half-height output grid with red markers at every XOR-positive position.
    - For each red marker on the output boundary, draw a full yellow line through its row or column according to which boundary it touches; corner markers draw both lines.
    - For each red marker strictly inside the output, fill black cells in its clipped 3x3 Chebyshev neighborhood with orange, without overwriting red markers or yellow lines.

    Key insight:
    - The XOR mask is only an intermediate marker field; each marker's location type (boundary vs. interior) determines whether it becomes a spanning line source or a local halo source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0

    half = h // 2
    top = g[:half]
    bottom = g[half:]
    assert set(np.unique(top)).issubset({BLACK, BLUE})
    assert set(np.unique(bottom)).issubset({BLACK, GREEN})

    xor_mask = (top == BLUE) ^ (bottom == GREEN)
    return _expected_output_from_xor(xor_mask)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _boundary_kind(pos: tuple[int, int], h: int, w: int) -> str:
    r, c = pos
    if (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        return "corner"
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    return "interior"


def _add_if_compatible(
    candidates: list[tuple[int, int]],
    taken: list[tuple[int, int]],
    picked: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if picked is not None:
        candidates = [picked] + [p for p in candidates if p != picked]
    for pos in candidates:
        if all(_chebyshev(pos, other) >= 2 for other in taken):
            taken.append(pos)
            return pos
    return None


def _interaction_candidates(boundary_pos: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = boundary_pos
    out: set[tuple[int, int]] = set()
    if r in (0, h - 1):
        for rr in range(1, h - 1):
            for cc in range(max(1, c - 1), min(w - 2, c + 1) + 1):
                out.add((rr, cc))
    if c in (0, w - 1):
        for cc in range(1, w - 1):
            for rr in range(max(1, r - 1), min(h - 2, r + 1) + 1):
                out.add((rr, cc))
    return random.sample(list(out), k=len(out)) if out else []


def _sample_xor_positions(h: int, w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []

    if random.random() < 0.35:
        primary = random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
    else:
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            primary = (0, random.randint(1, w - 2))
        elif edge == "bottom":
            primary = (h - 1, random.randint(1, w - 2))
        elif edge == "left":
            primary = (random.randint(1, h - 2), 0)
        else:
            primary = (random.randint(1, h - 2), w - 1)
    positions.append(primary)

    if _boundary_kind(primary, h, w) != "corner" and random.random() < 0.65:
        if primary[0] in (0, h - 1):
            secondary_pool = (
                [(r, 0) for r in range(1, h - 1)]
                + [(r, w - 1) for r in range(1, h - 1)]
            )
        else:
            secondary_pool = (
                [(0, c) for c in range(1, w - 1)]
                + [(h - 1, c) for c in range(1, w - 1)]
            )
        random.shuffle(secondary_pool)
        _add_if_compatible(secondary_pool, positions)

    if random.random() < 0.3:
        extra_boundary = (
            [(0, c) for c in range(w)]
            + [(h - 1, c) for c in range(w)]
            + [(r, 0) for r in range(1, h - 1)]
            + [(r, w - 1) for r in range(1, h - 1)]
        )
        random.shuffle(extra_boundary)
        _add_if_compatible(extra_boundary, positions)

    internal_count = random.randint(2, 4)
    if random.random() < 0.6:
        interaction_pool = _interaction_candidates(primary, h, w)
        if not _add_if_compatible(interaction_pool, positions):
            return None

    interior_pool = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(interior_pool)
    while len([p for p in positions if _boundary_kind(p, h, w) == "interior"]) < internal_count:
        if not _add_if_compatible(interior_pool, positions):
            return None

    if len(positions) > 8:
        return None
    boundary_count = sum(_boundary_kind(p, h, w) != "interior" for p in positions)
    interior_count = len(positions) - boundary_count
    if boundary_count < 1 or interior_count < 2:
        return None
    return positions


def _seed_options(target: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = target
    options: list[tuple[int, int]] = []
    if r == 0:
        options.append((1, c))
    if r == h - 1:
        options.append((h - 2, c))
    if c == 0:
        options.append((r, 1))
    if c == w - 1:
        options.append((r, w - 2))
    if 0 < r < h - 1 and 0 < c < w - 1:
        options.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    dedup: list[tuple[int, int]] = []
    seen = set()
    for pos in options:
        if pos not in seen and _in_bounds((h, w), pos[0], pos[1]):
            dedup.append(pos)
            seen.add(pos)
    random.shuffle(dedup)
    return dedup


def _cell_available(
    cell: tuple[int, int],
    target: tuple[int, int],
    shape: tuple[int, int],
    reserved_targets: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c):
        return False
    if cell == target or cell in reserved_targets or cell in occupied:
        return False
    for other in reserved_targets:
        if other != target and abs(other[0] - r) + abs(other[1] - c) == 1:
            return False
    return True


def _grow_base_component(
    target: tuple[int, int],
    shape: tuple[int, int],
    reserved_targets: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = shape
    for seed in _seed_options(target, h, w):
        if not _cell_available(seed, target, shape, reserved_targets, occupied):
            continue
        for _ in range(80):
            target_size = random.randint(*BASE_SIZE_RANGE)
            cells = {seed}
            while len(cells) < target_size:
                frontier = set()
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        candidate = (nr, nc)
                        if candidate in cells:
                            continue
                        if _cell_available(candidate, target, shape, reserved_targets, occupied):
                            frontier.add(candidate)
                if not frontier:
                    break
                cells.add(random.choice(list(frontier)))
            if not _is_irregular_component(cells):
                continue
            if all(
                all(abs(rr - pr) + abs(cc - pc) != 1 for pr, pc in occupied)
                for rr, cc in cells
            ):
                return cells
    return None


def _assign_sides(count: int) -> list[bool]:
    labels = [True] * count
    false_count = random.randint(1, count - 1)
    false_indices = random.sample(range(count), k=false_count)
    for idx in false_indices:
        labels[idx] = False
    random.shuffle(labels)
    return labels


def _build_input_from_markers(
    h: int,
    w: int,
    markers: list[tuple[int, int]],
) -> np.ndarray | None:
    top = np.zeros((h, w), dtype=int)
    bottom = np.zeros((h, w), dtype=int)
    reserved_targets = set(markers)
    occupied: set[tuple[int, int]] = set()
    assignments = _assign_sides(len(markers))

    order = sorted(
        range(len(markers)),
        key=lambda i: (_boundary_kind(markers[i], h, w) == "interior", markers[i]),
    )

    bases: list[set[tuple[int, int]] | None] = [None] * len(markers)
    for idx in order:
        target = markers[idx]
        base = _grow_base_component(target, (h, w), reserved_targets, occupied)
        if base is None:
            return None
        bases[idx] = base
        occupied |= base
        occupied.add(target)

    for idx, target in enumerate(markers):
        base = bases[idx]
        assert base is not None
        for r, c in base:
            top[r, c] = BLUE
            bottom[r, c] = GREEN
        if assignments[idx]:
            top[target] = BLUE
        else:
            bottom[target] = GREEN

    grid = np.vstack([top, bottom])
    return grid


def _analyze_sample(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    half = g.shape[0] // 2
    top = g[:half] == BLUE
    bottom = g[half:] == GREEN
    xor_mask = top ^ bottom
    overlap = top & bottom
    out = main(g)
    lines = _line_mask_from_xor(xor_mask)

    markers = [tuple(map(int, p)) for p in np.argwhere(xor_mask)]
    boundary = [p for p in markers if _boundary_kind(p, half, g.shape[1]) != "interior"]
    interior = [p for p in markers if _boundary_kind(p, half, g.shape[1]) == "interior"]
    top_only = int(np.sum(top & ~bottom))
    bottom_only = int(np.sum(bottom & ~top))

    interaction = False
    for r, c in interior:
        rr0, rr1 = max(0, r - 1), min(half, r + 2)
        cc0, cc1 = max(0, c - 1), min(g.shape[1], c + 2)
        local_lines = lines[rr0:rr1, cc0:cc1].copy()
        local_lines[r - rr0, c - cc0] = False
        if np.any(local_lines):
            interaction = True
            break

    return {
        "xor_mask": xor_mask,
        "output": out,
        "boundary_markers": boundary,
        "interior_markers": interior,
        "top_only": top_only,
        "bottom_only": bottom_only,
        "overlap_count": int(np.sum(overlap)),
        "orange_count": int(np.sum(out == ORANGE)),
        "yellow_count": int(np.sum(out == YELLOW)),
        "corner_count": sum(_boundary_kind(p, half, g.shape[1]) == "corner" for p in boundary),
        "vertical_line_markers": sum(p[0] in (0, half - 1) for p in boundary),
        "horizontal_line_markers": sum(p[1] in (0, g.shape[1] - 1) for p in boundary),
        "line_halo_interaction": interaction,
    }


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] % 2 != 0 or max(g.shape) > 30:
        return False
    half = g.shape[0] // 2
    top = g[:half]
    bottom = g[half:]
    if not set(np.unique(top)).issubset({BLACK, BLUE}):
        return False
    if not set(np.unique(bottom)).issubset({BLACK, GREEN}):
        return False

    top_mask = top == BLUE
    bottom_mask = bottom == GREEN
    if np.sum(top_mask) == 0 or np.sum(bottom_mask) == 0:
        return False

    top_comps = _components4(top_mask)
    bottom_comps = _components4(bottom_mask)
    if not top_comps or not bottom_comps:
        return False
    if not all(_is_irregular_component(comp) for comp in top_comps + bottom_comps):
        return False

    xor_mask = top_mask ^ bottom_mask
    overlap = top_mask & bottom_mask
    markers = [tuple(map(int, p)) for p in np.argwhere(xor_mask)]
    if not (3 <= len(markers) <= 8):
        return False
    boundary = [p for p in markers if _boundary_kind(p, half, g.shape[1]) != "interior"]
    interior = [p for p in markers if _boundary_kind(p, half, g.shape[1]) == "interior"]
    if len(boundary) < 1 or len(interior) < 2:
        return False
    if np.sum(overlap) == 0:
        return False
    if np.sum(top_mask & ~bottom_mask) == 0 or np.sum(bottom_mask & ~top_mask) == 0:
        return False

    expected = _expected_output_from_xor(xor_mask)
    actual = main(g)
    if not np.array_equal(actual, expected):
        return False

    if np.sum(actual == YELLOW) == 0 or np.sum(actual == ORANGE) == 0:
        return False
    if not np.array_equal(actual == RED, xor_mask):
        return False
    return True


def _attempt() -> np.ndarray | None:
    h = random.randint(*HALF_HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    markers = _sample_xor_positions(h, w)
    if markers is None:
        return None
    grid = _build_input_from_markers(h, w, markers)
    if grid is None or not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 3")
