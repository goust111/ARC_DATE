from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _crop_component(component: dict, bg: int = 0) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = component["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(component["color"])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Sort scattered monochrome sprites by pixel count, flip every even-ranked one horizontally, and stack them vertically.

    Input structure:
    - The grid uses black background cells and contains several small monochrome 4-connected sprites.
    - Each sprite is spatially separated from the others and has its own color.
    - Sprite pixel counts are distinct, so the size ordering is unambiguous.

    Rule:
    - Identify every monochrome 4-connected sprite and count its foreground pixels.
    - Sort the sprites from smallest area to largest area, then crop each one to its bounding box.
    - Keep odd-ranked sprites unchanged, horizontally mirror even-ranked sprites, and stack them from top to bottom with one background row between neighbors.
    - Center every cropped sprite inside the output width, using the widest cropped sprite as the output width and putting any extra odd column on the right.

    Key insight:
    - Pixel count is not only a statistic: it determines the rank order, and rank parity then decides whether a sprite is mirrored before packing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = 0 if np.any(g == 0) else int(np.bincount(g.ravel()).argmax())
    components = _find_components(g, bg=bg)
    assert components, "expected at least one foreground sprite"

    ordered = sorted(components, key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    patches: list[np.ndarray] = []
    for index, component in enumerate(ordered, start=1):
        patch = _crop_component(component, bg=bg)
        if index % 2 == 0:
            patch = np.fliplr(patch)
        patches.append(patch)

    out_h = sum(int(p.shape[0]) for p in patches) + len(patches) - 1
    out_w = max(int(p.shape[1]) for p in patches)
    out = np.full((out_h, out_w), bg, dtype=int)

    r0 = 0
    for i, patch in enumerate(patches):
        left_pad = (out_w - patch.shape[1]) // 2
        out[r0 : r0 + patch.shape[0], left_pad : left_pad + patch.shape[1]] = patch
        r0 += patch.shape[0]
        if i != len(patches) - 1:
            r0 += 1

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_shape(area: int, color: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            base_r, base_c = random.choice(tuple(cells))
            candidates = []
            for nr, nc in _neighbors4(base_r, base_c):
                if (nr, nc) not in cells:
                    candidates.append((nr, nc))
            if not candidates:
                break
            cells.add(random.choice(candidates))

        if len(cells) != area:
            continue

        cells = _normalize_cells(cells)
        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)
        h = max_r + 1
        w = max_c + 1
        if h > 4 or w > 4 or w < 2:
            continue

        patch = np.zeros((h, w), dtype=int)
        for r, c in cells:
            patch[r, c] = color

        if not np.any(patch[:, 0]) or not np.any(patch[:, -1]) or not np.any(patch[0, :]) or not np.any(patch[-1, :]):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue

        k = random.randrange(4)
        patch = np.rot90(patch, k)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if patch.shape[0] > 4 or patch.shape[1] > 4:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    return None


def _coords_of_patch(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != 0)
    return [(int(r + r0), int(c + c0)) for r, c in coords]


def _can_place(coords: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    return all(not forbidden[r, c] for r, c in coords)


def _mark_forbidden(forbidden: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_patch(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    placements = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(placements)

    for r0, c0 in placements:
        coords = _coords_of_patch(patch, r0, c0)
        if _can_place(coords, forbidden):
            for r, c in coords:
                grid[r, c] = int(patch[patch != 0][0])
            _mark_forbidden(forbidden, coords)
            return True
    return False


def _valid_generated_scene(grid: np.ndarray, expected_count: int) -> bool:
    components = _find_components(grid, bg=0)
    if len(components) != expected_count:
        return False
    areas = [int(comp["area"]) for comp in components]
    if len(set(areas)) != len(areas):
        return False
    if any(not (3 <= area <= 8) for area in areas):
        return False
    if any((comp["bbox"][2] - comp["bbox"][0] + 1) > 4 or (comp["bbox"][3] - comp["bbox"][1] + 1) > 4 for comp in components):
        return False
    if any(np.array_equal(_crop_component(comp), np.fliplr(_crop_component(comp))) for comp in components):
        return False
    return max(main(grid).shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(11, 16)
        w = random.randint(16, 22)
        count = random.randint(4, 5)
        areas = random.sample(range(3, 9), count)
        colors = random.sample(range(1, 10), count)

        patches: list[np.ndarray] = []
        ok = True
        for area, color in zip(areas, colors):
            patch = _random_shape(area, color)
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue

        order = list(range(count))
        random.shuffle(order)
        grid = np.zeros((h, w), dtype=int)
        forbidden = np.zeros((h, w), dtype=bool)

        placed = True
        for idx in order:
            if not _place_patch(grid, forbidden, patches[idx]):
                placed = False
                break
        if not placed:
            continue

        if _valid_generated_scene(grid, count):
            return grid

    raise RuntimeError("failed to generate a valid input")
