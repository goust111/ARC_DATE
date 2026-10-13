from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common(arr: np.ndarray) -> int:
    return int(Counter(map(int, arr.reshape(-1))).most_common(1)[0][0])


def _compact_line(
    line: np.ndarray,
    *,
    base: int,
    mover: int,
    obstacle: int,
    toward_start: bool,
) -> np.ndarray:
    out = np.full(line.shape, int(base), dtype=int)
    out[line == obstacle] = int(obstacle)

    start = 0
    n = int(line.shape[0])
    while start < n:
        while start < n and int(line[start]) == obstacle:
            start += 1
        if start >= n:
            break
        end = start
        while end < n and int(line[end]) != obstacle:
            end += 1
        mover_count = int(np.count_nonzero(line[start:end] == mover))
        if toward_start:
            out[start : start + mover_count] = int(mover)
        else:
            out[end - mover_count : end] = int(mover)
        start = end
    return out


def _compact_patch(patch: np.ndarray, *, base: int, mover: int, obstacle: int) -> np.ndarray:
    h, w = map(int, patch.shape)
    out = np.empty_like(patch)

    if w > h:
        left_count = int(np.count_nonzero(patch[:, : w // 2] == mover))
        right_count = int(np.count_nonzero(patch[:, (w + 1) // 2 :] == mover))
        assert left_count != right_count
        toward_start = left_count > right_count
        for r in range(h):
            out[r, :] = _compact_line(
                patch[r, :],
                base=base,
                mover=mover,
                obstacle=obstacle,
                toward_start=toward_start,
            )
    else:
        top_count = int(np.count_nonzero(patch[: h // 2, :] == mover))
        bottom_count = int(np.count_nonzero(patch[(h + 1) // 2 :, :] == mover))
        assert top_count != bottom_count
        toward_start = top_count > bottom_count
        for c in range(w):
            out[:, c] = _compact_line(
                patch[:, c],
                base=base,
                mover=mover,
                obstacle=obstacle,
                toward_start=toward_start,
            )
    return out


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for coords in _components4(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        h, w = map(int, patch.shape)
        assert patch.size == len(coords), "each component must be a solid rectangle"
        assert h != w, "containers must be non-square"
        base = _most_common(patch)
        other_colors = sorted(int(x) for x in np.unique(patch) if int(x) != base)
        assert len(other_colors) == 2, "each container needs one mover color and one obstacle color"
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "base": int(base),
                "other_colors": other_colors,
            }
        )
    assert len(containers) >= 2, "expected multiple containers"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact each rectangular container's moving pixels along its long axis toward the heavier half.

    Input structure:
    - Black background surrounds several separated solid rectangular containers, each filled by one base color.
    - Every container is non-square and also contains exactly two non-base roles: one shared obstacle color and one mover color unique to that container.
    - Containers do not touch, so each rectangle can be isolated by its non-black connected component.

    Rule:
    - Detect the unique non-base color that appears in multiple containers; that is the obstacle color and it stays fixed.
    - For each container, choose the compaction axis from the rectangle's long side, then compare mover counts in the two halves of that axis, ignoring the middle row or column when the length is odd.
    - Within each row of a wide container or each column of a tall container, compact the mover pixels toward the heavier half, but only inside obstacle-delimited segments; emptied cells revert to the container's base color.

    Key insight:
    - Direction is not global: the rectangle's aspect ratio selects the axis, and the mover majority inside that one rectangle selects the destination side while the shared color reveals which pixels are immovable blockers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = _find_containers(g)
    color_to_containers: dict[int, set[int]] = {}
    for idx, container in enumerate(containers):
        for color in container["other_colors"]:
            color_to_containers.setdefault(int(color), set()).add(idx)

    shared = [color for color, owners in color_to_containers.items() if len(owners) > 1]
    assert len(shared) == 1, "expected exactly one shared obstacle color"
    obstacle = int(shared[0])

    out = g.copy()
    for container in containers:
        movers = [color for color in container["other_colors"] if int(color) != obstacle]
        assert len(movers) == 1
        mover = int(movers[0])
        r0, c0, r1, c1 = container["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _compact_patch(
            container["patch"],
            base=int(container["base"]),
            mover=mover,
            obstacle=obstacle,
        )
    return out


def _make_barriers(patch: np.ndarray, obstacle: int) -> None:
    h, w = map(int, patch.shape)
    if w > h:
        bar_count = 1 if w < 9 else random.randint(1, 2)
        for col in sorted(random.sample(range(1, w - 1), bar_count)):
            if random.random() < 0.65:
                r0, r1 = 0, h
            else:
                span = random.randint(1, h)
                r0 = random.randint(0, h - span)
                r1 = r0 + span
            patch[r0:r1, col] = int(obstacle)
    else:
        bar_count = 1 if h < 9 else random.randint(1, 2)
        for row in sorted(random.sample(range(1, h - 1), bar_count)):
            if random.random() < 0.65:
                c0, c1 = 0, w
            else:
                span = random.randint(1, w)
                c0 = random.randint(0, w - span)
                c1 = c0 + span
            patch[row, c0:c1] = int(obstacle)


def _half_cell_sets(
    patch: np.ndarray, *, base: int, target_low: bool
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, patch.shape)
    target: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []
    center: list[tuple[int, int]] = []

    if w > h:
        left_end = w // 2
        right_start = (w + 1) // 2
        for r in range(h):
            for c in range(w):
                if int(patch[r, c]) == base:
                    if c < left_end:
                        (target if target_low else other).append((r, c))
                    elif c >= right_start:
                        (other if target_low else target).append((r, c))
                    else:
                        center.append((r, c))
    else:
        top_end = h // 2
        bottom_start = (h + 1) // 2
        for r in range(h):
            for c in range(w):
                if int(patch[r, c]) == base:
                    if r < top_end:
                        (target if target_low else other).append((r, c))
                    elif r >= bottom_start:
                        (other if target_low else target).append((r, c))
                    else:
                        center.append((r, c))
    return target, other, center


def _add_movers(patch: np.ndarray, mover: int, *, base: int, target_low: bool) -> bool:
    target_cells, other_cells, center_cells = _half_cell_sets(
        patch,
        base=base,
        target_low=target_low,
    )
    if not target_cells:
        return False

    options: list[tuple[int, int]] = []
    for other_count in range(min(3, len(other_cells)) + 1):
        min_target = other_count + 1
        max_target = min(len(target_cells), other_count + 4)
        for target_count in range(min_target, max_target + 1):
            if target_count + other_count >= 2:
                options.append((target_count, other_count))
    if not options:
        if len(target_cells) == 1:
            options = [(1, 0)]
        else:
            return False

    target_count, other_count = random.choice(options)
    center_count = random.randint(0, min(2, len(center_cells)))

    chosen = random.sample(target_cells, target_count)
    if other_count:
        chosen += random.sample(other_cells, other_count)
    if center_count:
        chosen += random.sample(center_cells, center_count)

    for r, c in chosen:
        patch[r, c] = int(mover)
    return True


def _make_container(base: int, mover: int, obstacle: int, *, wide: bool, target_low: bool) -> np.ndarray | None:
    if wide:
        h = random.randint(4, 6)
        w = random.randint(max(h + 1, 7), 10)
    else:
        w = random.randint(4, 6)
        h = random.randint(max(w + 1, 7), 10)

    patch = np.full((h, w), int(base), dtype=int)
    _make_barriers(patch, obstacle)
    if np.count_nonzero(patch == obstacle) == 0:
        return None
    if not _add_movers(patch, mover, base=base, target_low=target_low):
        return None

    counts = Counter(map(int, patch.reshape(-1)))
    ranked = counts.most_common()
    if len(ranked) != 3:
        return None
    if int(ranked[0][0]) != base or ranked[0][1] <= ranked[1][1]:
        return None

    transformed = _compact_patch(patch, base=base, mover=mover, obstacle=obstacle)
    if np.array_equal(transformed, patch):
        return None
    return patch


def _place_patches(patches: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    for patch in random.sample(patches, len(patches)):
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(250):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc1 = min(width, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _container_orientations(count: int) -> list[bool]:
    flags = [bool(random.getrandbits(1)) for _ in range(count)]
    if count >= 2 and (all(flags) or not any(flags)) and random.random() < 0.6:
        flags[random.randrange(count)] = not flags[0]
    return flags


def _gen_attempt() -> np.ndarray | None:
    container_count = random.randint(2, 4)
    colors = random.sample(PALETTE, 1 + 2 * container_count)
    obstacle = colors[0]
    bases = colors[1 : 1 + container_count]
    movers = colors[1 + container_count :]

    wide_flags = _container_orientations(container_count)
    patches: list[np.ndarray] = []
    for base, mover, wide in zip(bases, movers, wide_flags):
        target_low = bool(random.getrandbits(1))
        patch = _make_container(base, mover, obstacle, wide=wide, target_low=target_low)
        if patch is None:
            return None
        patches.append(patch)

    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    height = random.randint(max(18, max_h + 4), 30)
    width = random.randint(max(18, max_w + 4), 30)

    grid = _place_patches(patches, height, width)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if len(_find_containers(grid)) != container_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
