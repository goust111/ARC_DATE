from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = np.random.default_rng()


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _symmetry_flags(mask: np.ndarray) -> dict[str, bool]:
    return {
        "vertical": bool(np.array_equal(mask, np.fliplr(mask))),
        "horizontal": bool(np.array_equal(mask, np.flipud(mask))),
        "rot180": bool(np.array_equal(mask, np.rot90(mask, 2))),
        "diag": bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T)),
        "anti_diag": bool(
            mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.flipud(np.fliplr(mask)).T)
        ),
    }


def _is_summary_symmetric(mask: np.ndarray) -> bool:
    flags = _symmetry_flags(mask)
    return flags["vertical"] or flags["horizontal"] or flags["rot180"]


def _component_masks(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or visited[r, c]:
                continue
            visited[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = coords - np.array([r0, c0])
            mask[rel[:, 0], rel[:, 1]] = True
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords,
                    "mask": mask,
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every symmetric sprite by copying its color onto the main diagonal in reading order.

    Input structure:
    - A black background contains one grey start marker at the top-left corner and several small monochrome 4-connected sprites.
    - Each sprite has a distinct non-black, non-grey color and occupies a compact bounding box away from the summary corner.
    - Some sprites are symmetric inside their own bounding boxes under vertical reflection, horizontal reflection, or 180-degree rotation.

    Rule:
    - Keep the entire input grid unchanged.
    - Detect every monochrome 4-connected sprite other than the grey marker and test its bbox mask for vertical, horizontal, or 180-degree rotational symmetry.
    - Sort the symmetric sprites by the reading order of their bbox top-left corners, then place the k-th selected sprite's color at position (k, k) on the main diagonal while preserving the grey marker at (0, 0).

    Key insight:
    - The diagonal summary is not tied to fixed colors; it is an ordered index of whichever sprites are structurally symmetric, using spatial reading order to decide the diagonal sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g[0, 0]) == GREY, "expected the grey start marker at (0, 0)"

    out = g.copy()
    components = _component_masks(g)
    symmetric = [comp for comp in components if _is_summary_symmetric(comp["mask"])]  # type: ignore[arg-type]
    symmetric.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))  # type: ignore[index]

    for idx, comp in enumerate(symmetric, start=1):
        assert idx < g.shape[0] and idx < g.shape[1], "summary diagonal exceeds the grid"
        out[idx, idx] = int(comp["color"])  # type: ignore[index]
    out[0, 0] = GREY
    return out


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    arr = np.array(list(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _coords_to_mask(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    arr = _normalize_coords(coords)
    mask = np.zeros((int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _random_connected_coords(area: int, max_span: int = 5) -> np.ndarray | None:
    cells = {(0, 0)}
    for _ in range(area * 8):
        if len(cells) == area:
            break
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if cand in cells:
                    continue
                rows = [p[0] for p in cells] + [cand[0]]
                cols = [p[1] for p in cells] + [cand[1]]
                if max(rows) - min(rows) + 1 <= max_span and max(cols) - min(cols) + 1 <= max_span:
                    frontier.append(cand)
        if not frontier:
            return None
        cells.add(frontier[int(_RNG.integers(len(frontier)))])
    if len(cells) != area:
        return None
    return _normalize_coords(cells)


def _random_asymmetric_mask() -> np.ndarray:
    for _ in range(500):
        area = int(_RNG.integers(3, 9))
        coords = _random_connected_coords(area)
        if coords is None:
            continue
        mask = _coords_to_mask(map(tuple, coords))
        flags = _symmetry_flags(mask)
        if any(flags.values()):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric sprite")


def _make_vertical(seed: np.ndarray, width: int) -> np.ndarray:
    h = seed.shape[0]
    mask = np.zeros((h, width), dtype=bool)
    half = seed.shape[1]
    mask[:, :half] |= seed
    for c in range(half):
        mask[:, width - 1 - c] |= seed[:, c]
    return mask


def _make_horizontal(seed: np.ndarray, height: int) -> np.ndarray:
    w = seed.shape[1]
    mask = np.zeros((height, w), dtype=bool)
    half = seed.shape[0]
    mask[:half, :] |= seed
    for r in range(half):
        mask[height - 1 - r, :] |= seed[r, :]
    return mask


def _random_symmetric_mask(mode: str) -> np.ndarray:
    for _ in range(800):
        if mode == "vertical":
            h = int(_RNG.integers(2, 6))
            w = int(_RNG.integers(2, 6))
            seed = _RNG.random((h, (w + 1) // 2)) < _RNG.uniform(0.35, 0.7)
            mask = _make_vertical(seed, w)
        elif mode == "horizontal":
            h = int(_RNG.integers(2, 6))
            w = int(_RNG.integers(2, 6))
            seed = _RNG.random(((h + 1) // 2, w)) < _RNG.uniform(0.35, 0.7)
            mask = _make_horizontal(seed, h)
        elif mode == "rot180":
            h = int(_RNG.integers(2, 6))
            w = int(_RNG.integers(2, 6))
            seed = _RNG.random((h, w)) < _RNG.uniform(0.25, 0.55)
            mask = seed | np.rot90(seed, 2)
        else:
            raise ValueError(mode)

        if not mask.any():
            continue
        mask = _crop_mask(mask)
        area = int(mask.sum())
        if not 3 <= area <= 8:
            continue
        if not _is_connected(mask):
            continue
        if not _is_summary_symmetric(mask):
            continue
        return mask
    raise RuntimeError(f"failed to generate a symmetric sprite for mode={mode}")


def _place_sprite(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    min_corner_gap: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    sh, sw = mask.shape
    if sh > h - min_corner_gap or sw > w - min_corner_gap:
        return None

    candidates: list[tuple[int, int]] = []
    for r0 in range(min_corner_gap, h - sh + 1):
        for c0 in range(min_corner_gap, w - sw + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + sh + 1)
            cc1 = min(w, c0 + sw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = candidates[int(_RNG.integers(len(candidates)))]
    rr, cc = np.argwhere(mask).T
    grid[r0 + rr, c0 + cc] = color
    occupied[r0 : r0 + sh, c0 : c0 + sw] = True
    return r0, c0


def _build_input() -> np.ndarray | None:
    object_count = int(_RNG.integers(5, 7))
    symmetric_count = int(_RNG.integers(2, 4))
    min_side = 15 if object_count == 5 else 16
    h = int(_RNG.integers(min_side, 19))
    w = int(_RNG.integers(min_side, 19))

    grid = np.zeros((h, w), dtype=int)
    grid[0, 0] = GREY
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, 0] = True

    colors = list(map(int, _RNG.choice(OBJECT_COLORS, size=object_count, replace=False)))
    modes = list(_RNG.choice(np.array(["vertical", "horizontal", "rot180"], dtype=object), size=symmetric_count, replace=True))
    masks = [_random_symmetric_mask(str(mode)) for mode in modes]
    masks.extend(_random_asymmetric_mask() for _ in range(object_count - symmetric_count))

    order = list(_RNG.permutation(object_count))
    sprites = [(colors[i], masks[i]) for i in order]
    sprites.sort(key=lambda item: int(item[1].sum()), reverse=True)

    for color, mask in sprites:
        if _place_sprite(grid, occupied, mask, color, min_corner_gap=4) is None:
            return None

    components = _component_masks(grid)
    if len(components) != object_count:
        return None

    symmetric_components = [comp for comp in components if _is_summary_symmetric(comp["mask"])]  # type: ignore[arg-type]
    if len(symmetric_components) != symmetric_count:
        return None

    for comp in components:
        area = int(np.asarray(comp["mask"]).sum())  # type: ignore[arg-type]
        if not 3 <= area <= 8:
            return None
        r0, c0, _, _ = comp["bbox"]  # type: ignore[index]
        if r0 < 4 or c0 < 4:
            return None

    out = main(grid)
    if int(out[0, 0]) != GREY:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
