from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop to an inclusive bbox."""
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(grid: np.ndarray, *, bg: int, connectivity: int = 8) -> list[dict]:
    """
    Find foreground connected components, following the same (r, c) / bbox conventions as arc_utils.
    """
    g = np.asarray(grid)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    if int(connectivity) == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_stripe_patch(patch: np.ndarray, bg: int) -> bool:
    h, w = map(int, patch.shape)
    if min(h, w) != 1 or max(h, w) < 2:
        return False
    if np.any(patch == int(bg)):
        return False
    return len({int(v) for v in np.unique(patch)}) == 2


def _is_tile_patch(patch: np.ndarray, bg: int) -> bool:
    h, w = map(int, patch.shape)
    if h != w or h < 2:
        return False
    if np.any(patch == int(bg)):
        return False
    return len({int(v) for v in np.unique(patch)}) >= 2


def _extract_roles(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=8)
    if len(comps) != 2:
        raise ValueError("expected exactly two foreground components")

    stripe_patch = None
    tile_patch = None
    orientation = None
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        if _is_stripe_patch(patch, bg):
            if stripe_patch is not None:
                raise ValueError("expected a unique stripe component")
            stripe_patch = patch
            orientation = "horizontal" if patch.shape[0] == 1 else "vertical"
        elif _is_tile_patch(patch, bg):
            if tile_patch is not None:
                raise ValueError("expected a unique tile component")
            tile_patch = patch

    if stripe_patch is None or tile_patch is None or orientation is None:
        raise ValueError("could not identify stripe and tile")
    return tile_patch, stripe_patch, orientation


def _stripe_sequence(stripe_patch: np.ndarray, orientation: str) -> list[int]:
    if orientation == "horizontal":
        seq = stripe_patch[0, :]
    else:
        seq = stripe_patch[:, 0]
    values = [int(v) for v in seq]
    if len(set(values)) != 2:
        raise ValueError("stripe must contain exactly two colors")
    return values


def _build_output(tile: np.ndarray, stripe_values: list[int], orientation: str) -> np.ndarray:
    first_color = int(stripe_values[0])
    mirrored = np.fliplr(tile) if orientation == "horizontal" else np.flipud(tile)
    parts = [tile if int(v) == first_color else mirrored for v in stripe_values]
    return np.hstack(parts) if orientation == "horizontal" else np.vstack(parts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a two-color stripe into a mirrored tile array.

    Input structure:
    - The background is a single dominant color.
    - There are exactly two disconnected foreground objects: one filled multicolor square tile and one 1-pixel-wide two-color stripe.
    - The stripe is strictly horizontal or strictly vertical, and the two objects do not touch.

    Rule:
    - Read the stripe in geometric order: left-to-right for a horizontal stripe, top-to-bottom for a vertical stripe.
    - Let the stripe's first color be A and its other color be B.
    - If the stripe is horizontal, place tile copies from left to right and use a left-right mirror whenever the stripe cell is B; if the stripe is vertical, stack tile copies from top to bottom and use an up-down mirror whenever the stripe cell is B.

    Key insight:
    - The same stripe carries two independent signals at once: its orientation chooses both the tiling axis and the mirror axis, while its color sequence chooses which slots use the original tile versus its mirror.
    """
    tile, stripe, orientation = _extract_roles(np.asarray(grid, dtype=int))
    sequence = _stripe_sequence(stripe, orientation)
    return _build_output(tile, sequence, orientation)


def _sample_tile(size: int, colors: list[int], orientation: str) -> np.ndarray:
    for _ in range(400):
        tile = _RNG.choice(colors, size=(size, size), replace=True).astype(int)
        if set(map(int, np.unique(tile))) != set(map(int, colors)):
            continue
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        if np.array_equal(tile, np.flipud(tile)):
            continue
        if orientation == "horizontal" and np.array_equal(tile, np.fliplr(tile)):
            continue
        if orientation == "vertical" and np.array_equal(tile, np.flipud(tile)):
            continue
        return tile
    raise RuntimeError("failed to sample a non-symmetric tile")


def _sample_stripe_values(length: int, colors: list[int]) -> list[int]:
    a, b = map(int, colors)
    for _ in range(400):
        seq = [a]
        seq.extend(int(_RNG.choice(colors)) for _ in range(length - 1))
        if len(set(seq)) != 2:
            continue
        if seq.count(a) == len(seq) or seq.count(b) == len(seq):
            continue
        has_run = any(seq[i] == seq[i - 1] for i in range(1, len(seq)))
        if has_run or _RNG.random() < 0.35:
            return seq
    raise RuntimeError("failed to sample a two-color stripe")


def _bbox_too_close(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
    *,
    gap: int = 1,
) -> bool:
    r0a, c0a, r1a, c1a = map(int, box1)
    r0b, c0b, r1b, c1b = map(int, box2)
    separated = (
        r1a + gap < r0b
        or r1b + gap < r0a
        or c1a + gap < c0b
        or c1b + gap < c0a
    )
    return not separated


def _tile_position(H: int, W: int, size: int) -> tuple[int, int]:
    corner = int(_RNG.integers(0, 4))
    r_margin = int(_RNG.integers(1, 3))
    c_margin = int(_RNG.integers(1, 3))
    if corner in (0, 1):
        r0 = r_margin
    else:
        r0 = H - size - r_margin
    if corner in (0, 2):
        c0 = c_margin
    else:
        c0 = W - size - c_margin
    return int(r0), int(c0)


def _stripe_position(
    H: int,
    W: int,
    length: int,
    orientation: str,
    tile_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    h = 1 if orientation == "horizontal" else length
    w = length if orientation == "horizontal" else 1
    for _ in range(400):
        r0 = int(_RNG.integers(1, H - h))
        c0 = int(_RNG.integers(1, W - w))
        stripe_box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bbox_too_close(tile_box, stripe_box, gap=1):
            return r0, c0
    raise RuntimeError("failed to place stripe")


def _render_input(
    H: int,
    W: int,
    tile: np.ndarray,
    stripe_values: list[int],
    orientation: str,
    tile_pos: tuple[int, int],
    stripe_pos: tuple[int, int],
) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    tr, tc = tile_pos
    s = int(tile.shape[0])
    grid[tr : tr + s, tc : tc + s] = tile
    sr, sc = stripe_pos
    if orientation == "horizontal":
        grid[sr, sc : sc + len(stripe_values)] = np.array(stripe_values, dtype=int)
    else:
        grid[sr : sr + len(stripe_values), sc] = np.array(stripe_values, dtype=int)
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        tile, stripe, orientation = _extract_roles(grid)
        stripe_values = _stripe_sequence(stripe, orientation)
        out = main(grid)
    except Exception:
        return False

    if int(max(grid.shape)) > 30 or int(max(out.shape)) > 30:
        return False
    if np.array_equal(tile, np.fliplr(tile)) or np.array_equal(tile, np.flipud(tile)):
        return False
    if len(set(stripe_values)) != 2:
        return False

    all_original = _build_output(tile, [stripe_values[0]] * len(stripe_values), orientation)
    all_mirrored = _build_output(
        tile,
        [next(int(v) for v in stripe_values if int(v) != stripe_values[0])] * len(stripe_values),
        orientation,
    )
    if np.array_equal(out, all_original):
        return False
    if np.array_equal(out, all_mirrored):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    orientation = str(_RNG.choice(["horizontal", "vertical"]))
    tile_size = int(_RNG.choice([3, 4]))
    stripe_len = int(_RNG.integers(3, 7))
    if tile_size * stripe_len > 25:
        return None

    tile_color_count = int(_RNG.integers(2, 5))
    colors = list(range(1, 10))
    chosen = list(map(int, _RNG.choice(colors, size=tile_color_count + 2, replace=False)))
    tile_colors = chosen[:tile_color_count]
    stripe_colors = chosen[tile_color_count:]

    tile = _sample_tile(tile_size, tile_colors, orientation)
    stripe_values = _sample_stripe_values(stripe_len, stripe_colors)

    min_h = max(10, tile_size + 5, (stripe_len if orientation == "vertical" else 1) + 4)
    min_w = max(10, tile_size + 5, (stripe_len if orientation == "horizontal" else 1) + 4)
    H = int(_RNG.integers(min_h, 17))
    W = int(_RNG.integers(min_w, 17))

    tile_pos = _tile_position(H, W, tile_size)
    tr, tc = tile_pos
    tile_box = (tr, tc, tr + tile_size - 1, tc + tile_size - 1)
    stripe_pos = _stripe_position(H, W, stripe_len, orientation, tile_box)

    grid = _render_input(H, W, tile, stripe_values, orientation, tile_pos, stripe_pos)
    expected = _build_output(tile, stripe_values, orientation)

    if not np.array_equal(main(grid), expected):
        return None
    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-tile stripe puzzle input")
