from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
COLORS = tuple(c for c in range(10) if c != AZURE)
PACK_GAP = 1
GENERATION_ATTEMPTS = 500


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or int(grid[sr, sc]) == int(bg):
                continue
            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in deltas:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _boundary_count(coords: np.ndarray) -> int:
    cells = {(int(r), int(c)) for r, c in coords}
    boundary = 0
    for r, c in cells:
        if (
            (r - 1, c) not in cells
            or (r + 1, c) not in cells
            or (r, c - 1) not in cells
            or (r, c + 1) not in cells
        ):
            boundary += 1
    return boundary


def _object_patch(grid: np.ndarray, obj: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    color = int(obj["color"])
    for r, c in obj["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _paste(out: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> None:
    mask = patch != int(bg)
    h, w = map(int, patch.shape)
    assert 0 <= row and 0 <= col
    assert row + h <= out.shape[0] and col + w <= out.shape[1]
    target = out[row : row + h, col : col + w]
    assert np.all(target[mask] == int(bg)), "packed objects must not overlap"
    target[mask] = patch[mask]


def _sort_key(obj: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    return int(obj["area"]), r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort foreground objects by boundary complexity and move each class to its corner.

    Input structure:
    - The grid background is Azure.
    - Several separated monochrome 4-connected foreground objects are scattered through the grid.
    - Foreground object colors are distinct, and objects may be thin paths, one-pixel rings, or thicker solid shapes.

    Rule:
    - For each object, count area A and boundary pixels B, where a pixel is boundary when at least one 4-neighbor is outside the object.
    - Objects with B equal to A have no interior pixels and are packed at the upper-left corner.
    - Objects with B less than A have at least one interior pixel and are packed at the lower-right corner.
    - Within each class, objects are ordered by original area from small to large, with the smallest object closest to its corner, and all original positions are cleared.

    Key insight:
    Boundary count separates one-pixel-wide or hollow-looking objects from shapes with interior pixels; the relocation depends on structure, not color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = AZURE

    hollow: list[dict] = []
    solid: list[dict] = []
    for obj in _components_by_color(g, bg):
        area = int(obj["area"])
        boundary = _boundary_count(obj["coords"])
        obj = dict(obj)
        obj["patch"] = _object_patch(g, obj, bg)
        obj["boundary"] = int(boundary)
        if boundary == area:
            hollow.append(obj)
        else:
            assert boundary < area, "boundary count must not exceed area"
            solid.append(obj)

    hollow.sort(key=_sort_key)
    solid.sort(key=_sort_key)

    out = np.full((height, width), bg, dtype=int)

    top = 0
    for obj in hollow:
        patch = obj["patch"]
        _paste(out, patch, top, 0, bg)
        top += int(patch.shape[0]) + PACK_GAP

    bottom = height
    for obj in solid:
        patch = obj["patch"]
        bottom -= int(patch.shape[0])
        _paste(out, patch, bottom, width - int(patch.shape[1]), bg)
        bottom -= PACK_GAP

    assert max(out.shape) <= 30
    return out


def _trim(patch: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(patch != int(bg))
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.array(patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _orient(patch: np.ndarray, bg: int) -> np.ndarray:
    result = np.array(patch, dtype=int, copy=True)
    result = np.rot90(result, random.randrange(4))
    if random.random() < 0.5:
        result = np.fliplr(result)
    if random.random() < 0.5:
        result = np.flipud(result)
    return _trim(result, bg)


def _patch_area(patch: np.ndarray, bg: int) -> int:
    return int(np.count_nonzero(patch != int(bg)))


def _patch_boundary_count(patch: np.ndarray, bg: int) -> int:
    coords = np.argwhere(patch != int(bg))
    return _boundary_count(coords)


def _is_connected_patch(patch: np.ndarray, bg: int) -> bool:
    coords = np.argwhere(patch != int(bg))
    if coords.size == 0:
        return False
    cells = {(int(r), int(c)) for r, c in coords}
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _hollow_patch(color: int, bg: int) -> np.ndarray:
    mode = random.choice(("line", "corner", "zigzag", "ring"))
    if mode == "line":
        length = random.randint(3, 8)
        patch = np.full((1, length), bg, dtype=int)
        patch[0, :] = int(color)
    elif mode == "corner":
        h = random.randint(2, 6)
        w = random.randint(2, 6)
        patch = np.full((h, w), bg, dtype=int)
        patch[:, 0] = int(color)
        patch[-1, :] = int(color)
    elif mode == "zigzag":
        w = random.randint(3, 7)
        patch = np.full((3, w), bg, dtype=int)
        for c in range(w):
            if c % 2 == 0:
                patch[0, c] = int(color)
                patch[1, c] = int(color)
            else:
                patch[1, c] = int(color)
                patch[2, c] = int(color)
    else:
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        patch = np.full((h, w), bg, dtype=int)
        patch[0, :] = int(color)
        patch[-1, :] = int(color)
        patch[:, 0] = int(color)
        patch[:, -1] = int(color)
    patch = _orient(patch, bg)
    assert _is_connected_patch(patch, bg)
    assert _patch_boundary_count(patch, bg) == _patch_area(patch, bg)
    return patch


def _solid_patch(color: int, bg: int) -> np.ndarray:
    for _ in range(100):
        mode = random.choice(("rectangle", "notched", "fat_l"))
        if mode == "rectangle":
            h = random.randint(3, 6)
            w = random.randint(3, 6)
            patch = np.full((h, w), int(color), dtype=int)
        elif mode == "notched":
            h = random.randint(4, 7)
            w = random.randint(4, 7)
            patch = np.full((h, w), int(color), dtype=int)
            notch = random.choice(("tl", "tr", "bl", "br"))
            if notch == "tl":
                patch[0, 0] = bg
                patch[0, 1] = bg
            elif notch == "tr":
                patch[0, -1] = bg
                patch[1, -1] = bg
            elif notch == "bl":
                patch[-1, 0] = bg
                patch[-2, 0] = bg
            else:
                patch[-1, -1] = bg
                patch[-1, -2] = bg
        else:
            h = random.randint(4, 7)
            w = random.randint(4, 7)
            patch = np.full((h, w), bg, dtype=int)
            arm = random.randint(2, min(h, w) - 1)
            patch[:, :arm] = int(color)
            patch[-arm:, :] = int(color)

        patch = _orient(patch, bg)
        area = _patch_area(patch, bg)
        boundary = _patch_boundary_count(patch, bg)
        if _is_connected_patch(patch, bg) and boundary < area:
            return patch
    raise RuntimeError("failed to create a solid patch")


def _sample_class_patches(
    count: int,
    colors: list[int],
    make_patch,
    used_areas: set[int] | None = None,
) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    areas = set() if used_areas is None else set(used_areas)
    for color in colors[:count]:
        for _ in range(200):
            patch = make_patch(int(color), AZURE)
            area = _patch_area(patch, AZURE)
            if area not in areas:
                patches.append(patch)
                areas.add(area)
                break
        else:
            return []
    return patches


def _packed_stack_height(patches: list[np.ndarray]) -> int:
    if not patches:
        return 0
    return sum(int(p.shape[0]) for p in patches) + PACK_GAP * (len(patches) - 1)


def _would_pack_cleanly(hollow: list[np.ndarray], solid: list[np.ndarray], height: int, width: int) -> bool:
    if any(p.shape[0] > height or p.shape[1] > width for p in hollow + solid):
        return False
    return _packed_stack_height(hollow) + _packed_stack_height(solid) + PACK_GAP <= height


def _place_random(grid: np.ndarray, patch: np.ndarray, occupied: np.ndarray) -> bool:
    height, width = map(int, grid.shape)
    ph, pw = map(int, patch.shape)
    if ph + 2 > height or pw + 2 > width:
        return False

    positions = [
        (r, c)
        for r in range(1, height - ph)
        for c in range(1, width - pw)
    ]
    random.shuffle(positions)
    for r, c in positions:
        rr0 = max(0, r - 1)
        cc0 = max(0, c - 1)
        rr1 = min(height, r + ph + 1)
        cc1 = min(width, c + pw + 1)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            continue
        mask = patch != AZURE
        target = grid[r : r + ph, c : c + pw]
        if bool(np.any(target[mask] != AZURE)):
            continue
        target[mask] = patch[mask]
        occupied[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    n_hollow = random.randint(1, 3)
    n_solid = random.randint(1, 3)
    total = n_hollow + n_solid
    colors = random.sample(COLORS, total)

    hollow = _sample_class_patches(n_hollow, colors[:n_hollow], _hollow_patch)
    if not hollow:
        return None
    solid = _sample_class_patches(n_solid, colors[n_hollow:], _solid_patch)
    if not solid:
        return None

    hollow.sort(key=lambda p: _patch_area(p, AZURE))
    solid.sort(key=lambda p: _patch_area(p, AZURE))
    if not _would_pack_cleanly(hollow, solid, height, width):
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    patches = hollow + solid
    random.shuffle(patches)
    for patch in patches:
        if not _place_random(grid, patch, occupied):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None

    comps = _components_by_color(grid, AZURE)
    if len(comps) != total:
        return None
    branch_counts = [0, 0]
    for obj in comps:
        area = int(obj["area"])
        boundary = _boundary_count(obj["coords"])
        branch_counts[0 if boundary == area else 1] += 1
    if min(branch_counts) == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
