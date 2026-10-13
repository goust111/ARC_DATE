from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _neighbors(r: int, c: int, height: int, width: int, connectivity: int):
    deltas4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    deltas8 = deltas4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    deltas = deltas4 if connectivity == 4 else deltas8
    for dr, dc in deltas:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors(rr, cc, height, width, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
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


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == height - 1)
        or np.any(cols == 0)
        or np.any(cols == width - 1)
    )


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _enclosed_region(local_ring_mask: np.ndarray) -> np.ndarray | None:
    hole_mask = ~np.asarray(local_ring_mask, dtype=bool)
    enclosed: list[np.ndarray] = []
    for comp in _components_mask(hole_mask, connectivity=4):
        if not _touches_edge(comp, hole_mask.shape):
            enclosed.append(comp)
    if len(enclosed) != 1:
        return None
    return enclosed[0]


def _find_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for comp in _components_mask(grid != int(bg), connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        colors = {int(grid[int(r), int(c)]) for r, c in comp}
        if len(comp) <= 1:
            continue
        if height > 5 or width > 5:
            continue
        if not (2 <= len(colors) <= 4):
            continue
        if not _touches_edge(comp, grid.shape):
            continue
        sprites.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "colors": colors,
                "patch": _extract_patch(grid, comp, bg),
            }
        )
    assert sprites, "expected at least one edge sprite"
    return sprites


def _find_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings: list[dict] = []
    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg)}):
        for comp in _components_mask(grid == int(color), connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if max(height, width) <= 5:
                continue

            local = np.zeros((height, width), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            enclosed = _enclosed_region(local)
            if enclosed is None:
                continue

            global_interior = enclosed + np.array([[r0, c0]], dtype=int)
            signature = {
                int(grid[int(r), int(c)])
                for r, c in global_interior
                if int(grid[int(r), int(c)]) != int(bg)
            }
            if not signature:
                continue

            interior_mask = np.zeros_like(grid, dtype=bool)
            interior_mask[global_interior[:, 0], global_interior[:, 1]] = True
            rings.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "interior_bbox": _bbox(global_interior),
                    "interior_mask": interior_mask,
                    "signature": signature,
                }
            )
    assert rings, "expected at least one closed ring"
    return rings


def _solve_assignment(rings: list[dict], sprites: list[dict]) -> dict[int, int]:
    candidates = {
        idx: [
            j
            for j, sprite in enumerate(sprites)
            if rings[idx]["signature"].issubset(sprite["colors"])
        ]
        for idx in range(len(rings))
    }
    assert all(candidates.values()), "every ring must match at least one sprite"

    order = sorted(candidates, key=lambda idx: len(candidates[idx]))
    mapping: dict[int, int] = {}
    used: set[int] = set()
    solutions: list[dict[int, int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(mapping.copy())
            return
        ring_idx = int(order[pos])
        for sprite_idx in candidates[ring_idx]:
            if sprite_idx in used:
                continue
            used.add(int(sprite_idx))
            mapping[ring_idx] = int(sprite_idx)
            dfs(pos + 1)
            used.remove(int(sprite_idx))
            mapping.pop(ring_idx, None)

    dfs(0)
    assert len(solutions) == 1, "expected a unique ring-sprite assignment"
    return solutions[0]


def _paste_centered(out: np.ndarray, interior_mask: np.ndarray, patch: np.ndarray, bg: int) -> None:
    rows, cols = np.nonzero(interior_mask)
    assert rows.size > 0
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    inner_h = int(r1 - r0 + 1)
    inner_w = int(c1 - c0 + 1)
    patch_h, patch_w = map(int, patch.shape)

    top = int(r0 + (inner_h - patch_h) // 2)
    left = int(c0 + (inner_w - patch_w) // 2)

    out[interior_mask] = int(bg)
    for pr in range(patch_h):
        gr = int(top + pr)
        if not (0 <= gr < out.shape[0]):
            continue
        for pc in range(patch_w):
            gc = int(left + pc)
            if not (0 <= gc < out.shape[1]):
                continue
            color = int(patch[pr, pc])
            if color == int(bg):
                continue
            if interior_mask[gr, gc]:
                out[gr, gc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each ring's source sprite from the interior color signature, then center
    the matched sprite inside the enclosed region while keeping the ring intact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    rings = _find_rings(g, bg)
    sprites = _find_sprites(g, bg)
    assignment = _solve_assignment(rings, sprites)

    out = g.copy()
    for ring_idx, ring in enumerate(rings):
        sprite = sprites[assignment[ring_idx]]
        _paste_centered(out, ring["interior_mask"], sprite["patch"], bg)
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 30)
SPRITE_COUNT_RANGE = (1, 3)
SPRITE_SIDE_RANGE = (2, 5)
RING_INNER_SIDE_RANGE = (3, 8)
GENERATION_ATTEMPTS = 400


def _dilate(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    height, width = map(int, mask.shape)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(height, height - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(width, width - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(height, height + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(width, width + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _can_place(occupied: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> bool:
    height, width = map(int, occupied.shape)
    patch_h, patch_w = map(int, patch_mask.shape)
    if r0 < 0 or c0 < 0 or r0 + patch_h > height or c0 + patch_w > width:
        return False
    dilated = _dilate(occupied)
    return not bool(np.any(dilated[r0 : r0 + patch_h, c0 : c0 + patch_w] & patch_mask))


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    patch_h, patch_w = map(int, patch.shape)
    region = grid[r0 : r0 + patch_h, c0 : c0 + patch_w]
    mask = patch != int(bg)
    region[mask] = patch[mask]


def _spanning_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    steps = ["D"] * (height - 1) + ["R"] * (width - 1)
    random.shuffle(steps)

    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    if int(mask.sum()) > int(target_area):
        return None

    while int(mask.sum()) < int(target_area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors(int(rr), int(cc), height, width, 8):
                if not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True
    return mask


def _make_sprite_patch(colors: list[int], bg: int) -> np.ndarray | None:
    for _ in range(80):
        height = random.randint(*SPRITE_SIDE_RANGE)
        width = random.randint(*SPRITE_SIDE_RANGE)
        min_area = max(len(colors), height + width - 1)
        if min_area > height * width:
            continue
        target_area = random.randint(min_area, height * width)
        mask = _spanning_mask(height, width, target_area)
        if mask is None:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)
        patch = np.full((height, width), int(bg), dtype=int)
        for color, (r, c) in zip(colors, coords):
            patch[r, c] = int(color)
        for r, c in coords[len(colors) :]:
            patch[r, c] = int(random.choice(colors))
        return patch
    return None


def _edge_positions(grid_shape: tuple[int, int], patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, grid_shape)
    patch_h, patch_w = map(int, patch_shape)
    positions = set()
    for c in range(width - patch_w + 1):
        positions.add((0, int(c)))
        positions.add((height - patch_h, int(c)))
    for r in range(height - patch_h + 1):
        positions.add((int(r), 0))
        positions.add((int(r), width - patch_w))
    positions_list = list(positions)
    random.shuffle(positions_list)
    return positions_list


def _sample_signature(colors: list[int], private_color: int) -> list[int]:
    size = random.randint(1, min(3, len(colors)))
    extras = [color for color in colors if color != int(private_color)]
    chosen = [int(private_color)]
    if size > 1:
        chosen.extend(random.sample(extras, size - 1))
    random.shuffle(chosen)
    return chosen


def _choose_inner_dims(
    sprite_shape: tuple[int, int], signature_size: int, crop_case: bool
) -> tuple[int, int] | None:
    sprite_h, sprite_w = map(int, sprite_shape)
    candidates = []
    for inner_h in range(RING_INNER_SIDE_RANGE[0], RING_INNER_SIDE_RANGE[1] + 1):
        for inner_w in range(RING_INNER_SIDE_RANGE[0], RING_INNER_SIDE_RANGE[1] + 1):
            if (inner_h - 2) * (inner_w - 2) < int(signature_size):
                continue
            if crop_case:
                if inner_h < sprite_h or inner_w < sprite_w:
                    candidates.append((int(inner_h), int(inner_w)))
            else:
                if inner_h >= sprite_h and inner_w >= sprite_w:
                    candidates.append((int(inner_h), int(inner_w)))
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_dot_positions(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, inner_h - 1) for c in range(1, inner_w - 1)]
    if len(cells) < count:
        return None
    random.shuffle(cells)
    picked: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - pr), abs(cell[1] - pc)) > 1 for pr, pc in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked
    return random.sample(cells, count)


def _render_expected(grid: np.ndarray, rings: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for ring in rings:
        interior_mask = np.zeros_like(out, dtype=bool)
        r0 = int(ring["inner_r0"])
        c0 = int(ring["inner_c0"])
        inner_h = int(ring["inner_h"])
        inner_w = int(ring["inner_w"])
        interior_mask[r0 : r0 + inner_h, c0 : c0 + inner_w] = True
        _paste_centered(out, interior_mask, ring["sprite_patch"], bg)
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg = random.randrange(10)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    all_colors = [color for color in range(10) if color != bg]
    private_colors = random.sample(all_colors, n_sprites)
    shared_pool = [color for color in all_colors if color not in private_colors]
    ring_colors = random.sample(all_colors, n_sprites)

    sprite_specs = []
    for private_color in private_colors:
        n_colors = random.randint(2, min(4, 1 + len(shared_pool)))
        extras = random.sample(shared_pool, n_colors - 1)
        colors = [int(private_color)] + [int(color) for color in extras]
        patch = _make_sprite_patch(colors, bg)
        if patch is None:
            return None
        sprite_specs.append(
            {
                "private_color": int(private_color),
                "colors": colors,
                "patch": patch,
            }
        )

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = list(range(n_sprites))
    order.sort(
        key=lambda idx: int(np.count_nonzero(sprite_specs[idx]["patch"] != int(bg))),
        reverse=True,
    )
    for idx in order:
        patch = sprite_specs[idx]["patch"]
        patch_mask = patch != int(bg)
        placed = False
        for r0, c0 in _edge_positions((height, width), patch.shape):
            if _can_place(occupied, patch_mask, r0, c0):
                _paste(grid, patch, r0, c0, bg)
                occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch_mask
                sprite_specs[idx]["placement"] = (int(r0), int(c0))
                placed = True
                break
        if not placed:
            return None

    rings_meta: list[dict] = []
    for idx, sprite in enumerate(sprite_specs):
        signature = _sample_signature(sprite["colors"], sprite["private_color"])
        crop_case = bool(random.randrange(2))
        dims = _choose_inner_dims(sprite["patch"].shape, len(signature), crop_case)
        if dims is None:
            return None
        inner_h, inner_w = dims
        ring_h = int(inner_h + 2)
        ring_w = int(inner_w + 2)
        ring_patch = np.full((ring_h, ring_w), int(bg), dtype=int)
        ring_patch[0, :] = int(ring_colors[idx])
        ring_patch[-1, :] = int(ring_colors[idx])
        ring_patch[:, 0] = int(ring_colors[idx])
        ring_patch[:, -1] = int(ring_colors[idx])
        ring_mask = ring_patch != int(bg)
        ring_footprint = np.ones((ring_h, ring_w), dtype=bool)

        candidates = [
            (r0, c0)
            for r0 in range(1, height - ring_h)
            for c0 in range(1, width - ring_w)
        ]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            if not _can_place(occupied, ring_footprint, r0, c0):
                continue
            dot_positions = _sample_dot_positions(inner_h, inner_w, len(signature))
            if dot_positions is None:
                continue
            _paste(grid, ring_patch, r0, c0, bg)
            occupied[r0 : r0 + ring_h, c0 : c0 + ring_w] |= ring_footprint
            for color, (dr, dc) in zip(signature, dot_positions):
                grid[r0 + dr + 1, c0 + dc + 1] = int(color)
                occupied[r0 + dr + 1, c0 + dc + 1] = True
            rings_meta.append(
                {
                    "ring_color": int(ring_colors[idx]),
                    "inner_r0": int(r0 + 1),
                    "inner_c0": int(c0 + 1),
                    "inner_h": int(inner_h),
                    "inner_w": int(inner_w),
                    "signature": tuple(map(int, signature)),
                    "sprite_patch": sprite["patch"],
                }
            )
            placed = True
            break
        if not placed:
            return None

    counts = Counter(map(int, grid.reshape(-1)))
    bg_count = int(counts[int(bg)])
    if any(bg_count <= int(count) for color, count in counts.items() if int(color) != int(bg)):
        return None

    expected = _render_expected(grid, rings_meta, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
