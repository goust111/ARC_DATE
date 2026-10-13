from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RECOLOR_PATTERNS = (
    ("W", "B", "B", "B"),
    ("W", "W", "B", "B"),
    ("W", "W", "W", "B"),
    ("W", "W", "W", "W"),
    ("W", "W", "O", "B"),
    ("W", "W", "W", "O"),
    ("W", "W", "O", "P"),
)

# Generator configuration.
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
OBJECT_COUNT_RANGE = (2, 4)
BBOX_H_RANGE = (4, 7)
BBOX_W_RANGE = (4, 7)
MIN_BG_FRACTION = 0.45
MAX_OBJECT_BUILD_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _corner_samples(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    return [
        int(grid[r0, c0]),
        int(grid[r0, c1]),
        int(grid[r1, c0]),
        int(grid[r1, c1]),
    ]


def _winner_from_corner_samples(samples: list[int], bg: int) -> int | None:
    foreground = [color for color in samples if color != bg]
    if not foreground:
        return None
    counts: dict[int, int] = {}
    for color in foreground:
        counts[color] = counts.get(color, 0) + 1
    best = max(counts.values())
    for color in foreground:
        if counts[color] == best:
            return int(color)
    raise AssertionError("unreachable")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by the most frequent non-background bbox-corner color.

    If all four bbox corners are background, erase the object instead.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    out = g.copy()
    for coords in _components(g != bg):
        winner = _winner_from_corner_samples(_corner_samples(g, _bbox(coords)), bg)
        fill = bg if winner is None else int(winner)
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _sample_distinct_colors(rng: np.random.Generator, bg: int, count: int) -> list[int]:
    palette = [color for color in range(10) if color != bg]
    picks = rng.choice(palette, size=count, replace=False)
    return [int(color) for color in np.atleast_1d(picks)]


def _build_mask(h: int, w: int, pattern: tuple[str, str, str, str], rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    cross_r = int(rng.integers(1, h - 1))
    cross_c = int(rng.integers(1, w - 1))

    mask[:, cross_c] = True
    mask[cross_r, :] = True

    active = [symbol != "B" for symbol in pattern]
    if active[0]:
        mask[0, : cross_c + 1] = True
        mask[: cross_r + 1, 0] = True
    if active[1]:
        mask[0, cross_c:] = True
        mask[: cross_r + 1, w - 1] = True
    if active[2]:
        mask[h - 1, : cross_c + 1] = True
        mask[cross_r:, 0] = True
    if active[3]:
        mask[h - 1, cross_c:] = True
        mask[cross_r:, w - 1] = True

    forbidden = {
        (0, 0) if pattern[0] == "B" else None,
        (0, w - 1) if pattern[1] == "B" else None,
        (h - 1, 0) if pattern[2] == "B" else None,
        (h - 1, w - 1) if pattern[3] == "B" else None,
    }
    forbidden.discard(None)

    growth_steps = int(rng.integers(0, 1 + (h * w) // 3))
    for _ in range(growth_steps):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr, nc = int(rr + dr), int(cc + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if bool(mask[nr, nc]) or (nr, nc) in forbidden:
                    continue
                frontier.append((int(nr), int(nc)))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True

    return mask


def _color_object(
    mask: np.ndarray,
    pattern: tuple[str, str, str, str],
    bg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int | None]:
    h, w = mask.shape
    patch = np.full((h, w), bg, dtype=int)
    corner_cells = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))

    if all(symbol == "B" for symbol in pattern):
        palette = _sample_distinct_colors(rng, bg, 3)
        winner = None
    else:
        present_symbols = [symbol for symbol in ("W", "O", "P") if symbol in pattern]
        palette = _sample_distinct_colors(rng, bg, max(3, len(present_symbols) + 1))
        symbol_colors = {
            symbol: int(palette[index])
            for index, symbol in enumerate(present_symbols)
        }
        winner = int(symbol_colors["W"])

    coords = np.argwhere(mask)
    fill_choices = rng.choice(palette, size=len(coords), replace=True)
    patch[coords[:, 0], coords[:, 1]] = fill_choices.astype(int)

    if winner is not None:
        for (r, c), symbol in zip(corner_cells, pattern):
            if symbol != "B":
                patch[r, c] = int(symbol_colors[symbol])

    occupied = [(int(r), int(c)) for r, c in coords]
    non_corners = [cell for cell in occupied if cell not in set(corner_cells)]
    used_colors = {int(patch[r, c]) for r, c in occupied}
    if len(used_colors) < 2 and non_corners:
        rr, cc = non_corners[0]
        current = int(patch[rr, cc])
        alternatives = [color for color in palette if int(color) != current]
        patch[rr, cc] = int(alternatives[0])

    return patch, winner


def _build_object_patch(
    rng: np.random.Generator,
    bg: int,
    branch: str,
) -> np.ndarray | None:
    for _ in range(MAX_OBJECT_BUILD_ATTEMPTS):
        h = int(rng.integers(BBOX_H_RANGE[0], BBOX_H_RANGE[1] + 1))
        w = int(rng.integers(BBOX_W_RANGE[0], BBOX_W_RANGE[1] + 1))

        if branch == "erase":
            pattern = ("B", "B", "B", "B")
        else:
            pattern = list(RECOLOR_PATTERNS[int(rng.integers(len(RECOLOR_PATTERNS)))])
            rng.shuffle(pattern)
            pattern = tuple(pattern)

        mask = _build_mask(h, w, pattern, rng)
        patch, expected_winner = _color_object(mask, pattern, bg, rng)

        patch = np.rot90(patch, int(rng.integers(0, 4)))
        if bool(rng.integers(0, 2)):
            patch = np.flipud(patch)
        if bool(rng.integers(0, 2)):
            patch = np.fliplr(patch)
        patch = np.ascontiguousarray(patch)

        comps = _components(patch != bg)
        if len(comps) != 1:
            continue
        if _bbox(comps[0]) != (0, 0, patch.shape[0] - 1, patch.shape[1] - 1):
            continue
        if len({int(patch[r, c]) for r, c in comps[0]}) < 2:
            continue
        if _winner_from_corner_samples(_corner_samples(patch, (0, 0, patch.shape[0] - 1, patch.shape[1] - 1)), bg) != expected_winner:
            continue
        return patch
    return None


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> bool:
    h, w = patch.shape
    grid_h, grid_w = grid.shape
    if h > grid_h or w > grid_w:
        return False

    positions = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
    rng.shuffle(positions)
    mask = patch != bg

    for r0, c0 in positions:
        rr0 = max(0, r0 - 1)
        rr1 = min(grid_h, r0 + h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(grid_w, c0 + w + 1)
        if bool(reserved[rr0:rr1, cc0:cc1].any()):
            continue

        target = grid[r0 : r0 + h, c0 : c0 + w]
        target[mask] = patch[mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return True

    return False


def _instance_ok(grid: np.ndarray, bg: int, expected_count: int) -> bool:
    if _dominant_color(grid) != bg:
        return False
    if float(np.count_nonzero(grid == bg)) / float(grid.size) < MIN_BG_FRACTION:
        return False

    comps = _components(grid != bg)
    if len(comps) != expected_count:
        return False

    has_erase = False
    has_recolor = False
    for coords in comps:
        if len({int(grid[r, c]) for r, c in coords}) < 2:
            return False
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1 or c0 == c1:
            return False
        winner = _winner_from_corner_samples(_corner_samples(grid, (r0, c0, r1, c1)), bg)
        if winner is None:
            has_erase = True
        else:
            has_recolor = True

    if not (has_erase and has_recolor):
        return False

    out = main(grid)
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    min_h = max(GRID_H_RANGE[0], 8 + 2 * object_count)
    min_w = max(GRID_W_RANGE[0], 8 + 2 * object_count)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = int(rng.integers(min_h, GRID_H_RANGE[1] + 1))
    grid_w = int(rng.integers(min_w, GRID_W_RANGE[1] + 1))
    bg = int(rng.integers(0, 10))

    branches = ["erase", "recolor"]
    for _ in range(object_count - 2):
        branches.append("erase" if bool(rng.integers(0, 2)) else "recolor")
    rng.shuffle(branches)

    patches: list[np.ndarray] = []
    for branch in branches:
        patch = _build_object_patch(rng, bg, branch)
        if patch is None:
            return None
        patches.append(patch)

    patches.sort(key=lambda patch: patch.size, reverse=True)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    for patch in patches:
        if not _place_patch(grid, reserved, patch, bg, rng):
            return None

    if not _instance_ok(grid, bg, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
