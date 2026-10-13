from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_COLORS = tuple(color for color in range(1, 10) if color != GREY)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = DIR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
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
                coords.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _enclosed_black(mask: np.ndarray) -> np.ndarray:
    """Return black cells enclosed by a grey mask using 4-connected background."""
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    return bg & ~seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group sparse grey contour fragments by 8-connectivity, count each group, and fill its black interior from the top legend.

    Input structure:
    - The background is black.
    - The top row contains a left-to-right color-bar legend made of distinct 1x1 colored cells, with one all-black spacer row below it.
    - The main area contains only grey contour fragments. Each intended contour is one 8-connected grey component and may look sparse or diagonally connected.

    Rule:
    - Read the non-black legend cells on the top row from left to right.
    - In the main area, split grey pixels into 8-connected components and count the grey pixels N in each component.
    - For each component, look inside its bounding box for 4-connected black regions that do not touch the box boundary; these are the interior regions enclosed by that grey contour.
    - Fill every such enclosed black region with the color of the N-th legend cell, while keeping the legend and the grey contour pixels unchanged.

    Key insight:
    - The sparse fragments already form closed boundaries in the 8-connected-vs-4-connected sense: a grey 8-connected loop can trap black cells even when the loop uses diagonal contacts, so counting the component identifies the fill color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or g.shape[0] < 3:
        return out

    legend = [int(color) for color in g[0].tolist() if int(color) != BLACK]
    if not legend:
        return out

    main_area = g[2:]
    non_black = [int(color) for color in np.unique(main_area) if int(color) != BLACK]
    if not non_black:
        return out

    grey = max(non_black, key=lambda color: int(np.count_nonzero(main_area == color)))
    for comp in _components(main_area == grey, connectivity=8):
        count = int(comp.shape[0])
        if not (1 <= count <= len(legend)):
            continue

        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        enclosed = _enclosed_black(local)
        if not enclosed.any():
            continue

        rr, cc = np.where(enclosed)
        out[2 + r0 + rr, c0 + cc] = int(legend[count - 1])

    return out


def _mask_from_text(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    return np.asarray([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _dihedral_variants(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            trimmed = _trim(candidate)
            key = (tuple(map(int, trimmed.shape)), trimmed.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(trimmed.copy())
    return tuple(variants)


BASE_PATTERNS = {
    4: (
        """
        010
        101
        010
        """,
    ),
    5: (
        """
        110
        101
        010
        """,
    ),
    6: (
        """
        111
        101
        010
        """,
        """
        1110
        0101
        0010
        """,
    ),
    7: (
        """
        111
        101
        110
        """,
        """
        1110
        1101
        0010
        """,
    ),
    8: (
        """
        111
        101
        111
        """,
        """
        00100
        01010
        10001
        01010
        00100
        """,
    ),
}


def _build_pattern_library() -> dict[int, tuple[np.ndarray, ...]]:
    library: dict[int, tuple[np.ndarray, ...]] = {}
    for count, texts in BASE_PATTERNS.items():
        variants: list[np.ndarray] = []
        seen: set[tuple[tuple[int, int], bytes]] = set()
        for text in texts:
            for variant in _dihedral_variants(_mask_from_text(text)):
                key = (tuple(map(int, variant.shape)), variant.tobytes())
                if key not in seen:
                    seen.add(key)
                    variants.append(variant.copy())
        library[int(count)] = tuple(variants)
    return library


PATTERN_LIBRARY = _build_pattern_library()


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        r0 = max(0, rr - 1)
        r1 = min(occupied.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(occupied.shape[1], cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[2 + rr, cc] = GREY
        occupied[rr, cc] = True


def _sample_layout() -> tuple[np.ndarray, list[int], list[np.ndarray]] | None:
    object_count = random.randint(1, 3)
    counts = random.sample((4, 5, 6, 7, 8), object_count)
    masks = [random.choice(PATTERN_LIBRARY[count]).copy() for count in counts]
    masks.sort(key=lambda mask: (mask.shape[0] * mask.shape[1], mask.shape[0], mask.shape[1]), reverse=True)

    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]
    min_h = 2 + max(heights) + 2
    min_w = max(8, max(widths)) + 2
    packed_h = sum(heights) + 2 * (len(masks) - 1) + 2
    packed_w = sum(widths) + 2 * (len(masks) - 1) + 2
    grid_h = random.randint(max(10, min_h, min(30, packed_h // 2)), 30)
    grid_w = random.randint(max(10, min_w, min(30, packed_w // 2)), 30)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    legend = random.sample(list(LEGEND_COLORS), len(LEGEND_COLORS))
    legend_left = random.randint(0, grid_w - len(legend))
    grid[0, legend_left : legend_left + len(legend)] = np.asarray(legend, dtype=int)

    occupied = np.zeros((grid_h - 2, grid_w), dtype=bool)
    for mask in masks:
        h, w = mask.shape
        candidates: list[tuple[int, int]] = []
        for top in range(grid_h - 2 - h + 1):
            for left in range(grid_w - w + 1):
                if _can_place(occupied, mask, top, left):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _stamp(grid, occupied, mask, top, left)

    return grid, counts, masks


def _valid_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(grid[1] == BLACK):
        return False

    legend = [int(color) for color in grid[0].tolist() if int(color) != BLACK]
    if len(legend) != len(LEGEND_COLORS) or len(set(legend)) != len(legend):
        return False

    main_area = grid[2:]
    if not np.all((main_area == BLACK) | (main_area == GREY)):
        return False

    comps = _components(main_area == GREY, connectivity=8)
    if not (1 <= len(comps) <= 3):
        return False
    for comp in comps:
        if not (4 <= comp.shape[0] <= len(legend)):
            return False

    out = main(grid)
    return bool((out != grid).any())


def generate_input() -> np.ndarray:
    for _ in range(300):
        candidate = _sample_layout()
        if candidate is None:
            continue
        grid, _counts, _masks = candidate
        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
