from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg  # noqa: E402


BG = 0
TRANSFORMS = ("lr", "ud", "rot180", "diag", "anti")
SEPARATOR_ORIENTS = ("h", "v")
MAX_RETRIES = 400

_transform_bag: list[str] = []
_orient_bag: list[str] = []


def _take_from_bag(bag: list[str], values: tuple[str, ...]) -> str:
    if not bag:
        bag.extend(values)
        random.shuffle(bag)
    return bag.pop()


def _anti_transpose(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2).T


def _apply_transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "lr":
        return np.fliplr(grid)
    if name == "ud":
        return np.flipud(grid)
    if name == "rot180":
        return np.rot90(grid, 2)
    if name == "diag":
        return grid.T.copy()
    if name == "anti":
        return _anti_transpose(grid)
    raise ValueError(f"unknown transform: {name}")


def _transform_results(grid: np.ndarray) -> dict[str, np.ndarray]:
    out = {
        "lr": np.fliplr(grid),
        "ud": np.flipud(grid),
        "rot180": np.rot90(grid, 2),
    }
    if grid.shape[0] == grid.shape[1]:
        out["diag"] = grid.T.copy()
        out["anti"] = _anti_transpose(grid)
    return out


def _symmetry_matches(grid: np.ndarray) -> list[str]:
    matches: list[str] = []
    for name, transformed in _transform_results(grid).items():
        if transformed.shape == grid.shape and np.array_equal(grid, transformed):
            matches.append(name)
    return matches


def _find_separator(grid: np.ndarray) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    candidates: list[tuple[str, int, int]] = []

    for r in range(g.shape[0]):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != bg and 0 < r < g.shape[0] - 1:
            candidates.append(("h", int(r), int(vals[0])))

    for c in range(g.shape[1]):
        vals = np.unique(g[:, c])
        if vals.size == 1 and int(vals[0]) != bg and 0 < c < g.shape[1] - 1:
            candidates.append(("v", int(c), int(vals[0])))

    assert len(candidates) == 1, "expected exactly one full separator line"
    return candidates[0]


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orient, axis, _sep = _find_separator(grid)
    if orient == "h":
        first = grid[:axis, :]
        second = grid[axis + 1 :, :]
    else:
        first = grid[:, :axis]
        second = grid[:, axis + 1 :]

    assert first.size > 0 and second.size > 0, "separator must split two non-empty panels"
    area_first = int(first.shape[0] * first.shape[1])
    area_second = int(second.shape[0] * second.shape[1])
    assert area_first != area_second, "expected one larger and one smaller panel"
    big, small = (first, second) if area_first > area_second else (second, first)
    return np.asarray(big, dtype=int), np.asarray(small, dtype=int)


def _big_pattern(panel: np.ndarray) -> np.ndarray:
    bg = int(get_bg(panel))
    mask = panel != bg
    b = bbox(mask)
    assert b is not None, "big panel must contain a foreground pattern"
    pattern = crop_bbox(panel, b)
    colors = {int(v) for v in np.unique(pattern) if int(v) != bg}
    assert len(colors) >= 2, "big pattern must be multicolor"
    return pattern


def _infer_transform(grid: np.ndarray) -> str:
    big_panel, _small_panel = _split_panels(np.asarray(grid, dtype=int))
    pattern = _big_pattern(big_panel)
    matches = _symmetry_matches(pattern)
    assert len(matches) == 1, "big panel pattern must have exactly one valid symmetry"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the large panel's unique self-symmetry as an instruction and apply that same transform to the small panel.

    Input structure:
    - One full separator row or column splits the grid into a larger panel and a smaller panel.
    - The large panel contains one multicolor foreground pattern on a background.
    - The smaller panel contains one asymmetric monochrome sprite on its own background; when the instruction is a diagonal transpose, this panel is square.

    Rule:
    - Crop the large panel to its foreground pattern and test which one of these five transforms leaves it unchanged: left-right mirror, up-down mirror, 180-degree rotation, main-diagonal transpose, anti-diagonal transpose.
    - Exactly one of those five symmetries is valid, and that symmetry is the instruction.
    - Output only the small panel, transformed as a whole in the small panel's own coordinate frame by that same instruction.

    Key insight:
    - The large panel is not content to copy; its unique self-symmetry encodes the operation that must be transferred to the separate sprite panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    big_panel, small_panel = _split_panels(g)
    instruction = _infer_transform(g)
    if instruction in {"diag", "anti"}:
        assert small_panel.shape[0] == small_panel.shape[1], "transpose instructions require a square small panel"
    return _apply_transform(small_panel, instruction)


def _orbit_partner(shape: tuple[int, int], rc: tuple[int, int], transform: str) -> tuple[int, int]:
    r, c = rc
    h, w = shape
    if transform == "lr":
        return r, w - 1 - c
    if transform == "ud":
        return h - 1 - r, c
    if transform == "rot180":
        return h - 1 - r, w - 1 - c
    if transform == "diag":
        return c, r
    if transform == "anti":
        return w - 1 - c, h - 1 - r
    raise ValueError(f"unknown transform: {transform}")


def _fill_orbits(shape: tuple[int, int], transform: str, palette: list[int]) -> np.ndarray:
    h, w = shape
    grid = np.empty((h, w), dtype=int)
    seen: set[tuple[int, int]] = set()
    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)

    for rc in coords:
        if rc in seen:
            continue
        partner = _orbit_partner(shape, rc, transform)
        orbit = {rc, partner}
        color = random.choice(palette)
        for rr, cc in orbit:
            grid[rr, cc] = color
            seen.add((rr, cc))
    return grid


def _no_extra_nonbg_full_line(grid: np.ndarray, bg: int) -> bool:
    for row in grid:
        vals = np.unique(row)
        if vals.size == 1 and int(vals[0]) != bg:
            return False
    for col in grid.T:
        vals = np.unique(col)
        if vals.size == 1 and int(vals[0]) != bg:
            return False
    return True


def _make_large_pattern(
    transform: str,
    palette: list[int],
    panel_shape: tuple[int, int],
) -> np.ndarray | None:
    max_h = panel_shape[0] - 2
    max_w = panel_shape[1] - 2
    for _ in range(MAX_RETRIES):
        if transform in {"diag", "anti"}:
            max_n = min(8, max_h, max_w)
            if max_n < 4:
                return None
            n = random.randint(4, max_n)
            shape = (n, n)
        else:
            if max_h < 4 or max_w < 5:
                return None
            h = random.randint(4, min(8, max_h))
            w = random.randint(5, min(9, max_w))
            if h == w:
                continue
            shape = (h, w)

        pattern = _fill_orbits(shape, transform, palette)
        if len(set(map(int, pattern.flat))) < 2:
            continue
        if _symmetry_matches(pattern) != [transform]:
            continue
        return pattern
    return None


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _grow_connected_mask(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < count:
        mask[random.choice(_frontier(mask))] = True
    return mask


def _valid_small_panel(panel: np.ndarray, sprite_color: int, target: str) -> bool:
    bg = int(get_bg(panel))
    assert bg == BG
    fg_colors = {int(v) for v in np.unique(panel) if int(v) != bg}
    if fg_colors != {int(sprite_color)}:
        return False

    comps = find_components(panel, bg=bg, connectivity=4, monochromatic=False)
    if len(comps) != 1:
        return False

    if np.any(np.all(panel == sprite_color, axis=1)) or np.any(np.all(panel == sprite_color, axis=0)):
        return False

    transforms = _transform_results(panel)
    if target not in transforms:
        return False
    target_out = transforms[target]
    if np.array_equal(target_out, panel):
        return False

    for name, transformed in transforms.items():
        if np.array_equal(panel, transformed):
            return False
        if name != target and np.array_equal(target_out, transformed):
            return False
    return True


def _make_small_panel(shape: tuple[int, int], sprite_color: int, target: str) -> np.ndarray | None:
    h, w = shape
    area = h * w
    lo = max(4, area // 4)
    hi = max(lo, min(area - 2, (area * 2) // 5 + 1))

    for _ in range(MAX_RETRIES):
        count = random.randint(lo, hi)
        mask = _grow_connected_mask(h, w, count)
        panel = np.full((h, w), BG, dtype=int)
        panel[mask] = int(sprite_color)
        if _valid_small_panel(panel, sprite_color, target):
            return panel
    return None


def _embed_pattern(panel_shape: tuple[int, int], pattern: np.ndarray) -> np.ndarray | None:
    ph, pw = pattern.shape
    h, w = panel_shape
    if ph + 2 > h or pw + 2 > w:
        return None

    for _ in range(MAX_RETRIES):
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        panel = np.full((h, w), BG, dtype=int)
        panel[top : top + ph, left : left + pw] = pattern
        if int(get_bg(panel)) != BG:
            continue
        if not _no_extra_nonbg_full_line(panel, BG):
            continue
        return panel
    return None


def _choose_sizes(transform: str, orient: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(MAX_RETRIES):
        if orient == "h":
            if transform in {"diag", "anti"}:
                side = random.randint(6, 9)
                small_shape = (side, side)
                big_shape = (random.randint(side + 2, min(16, 29 - side)), side)
            else:
                width = random.randint(8, 13)
                small_h = random.randint(5, min(9, 29 - width - 3))
                big_h_min = max(small_h + 2, 8)
                big_h_max = min(18, 29 - small_h)
                if big_h_min > big_h_max:
                    continue
                small_shape = (small_h, width)
                big_shape = (random.randint(big_h_min, big_h_max), width)
        else:
            if transform in {"diag", "anti"}:
                side = random.randint(6, 9)
                small_shape = (side, side)
                big_shape = (side, random.randint(side + 2, min(16, 29 - side)))
            else:
                height = random.randint(8, 13)
                small_w = random.randint(5, min(9, 29 - height - 3))
                big_w_min = max(small_w + 2, 8)
                big_w_max = min(18, 29 - small_w)
                if big_w_min > big_w_max:
                    continue
                small_shape = (height, small_w)
                big_shape = (height, random.randint(big_w_min, big_w_max))

        if max(*big_shape, *small_shape) <= 30 and big_shape[0] * big_shape[1] > small_shape[0] * small_shape[1]:
            return big_shape, small_shape
    return None


def _assemble_grid(big_panel: np.ndarray, small_panel: np.ndarray, sep_color: int, orient: str) -> np.ndarray:
    if orient == "h":
        h = big_panel.shape[0] + 1 + small_panel.shape[0]
        w = big_panel.shape[1]
        grid = np.full((h, w), BG, dtype=int)
        grid[: big_panel.shape[0], :] = big_panel
        grid[big_panel.shape[0], :] = int(sep_color)
        grid[big_panel.shape[0] + 1 :, :] = small_panel
    else:
        h = big_panel.shape[0]
        w = big_panel.shape[1] + 1 + small_panel.shape[1]
        grid = np.full((h, w), BG, dtype=int)
        grid[:, : big_panel.shape[1]] = big_panel
        grid[:, big_panel.shape[1]] = int(sep_color)
        grid[:, big_panel.shape[1] + 1 :] = small_panel
    return grid


def _validate_instance(grid: np.ndarray, target: str, sprite_color: int) -> bool:
    try:
        inferred = _infer_transform(grid)
        if inferred != target:
            return False

        big_panel, small_panel = _split_panels(grid)
        if target in {"diag", "anti"} and small_panel.shape[0] != small_panel.shape[1]:
            return False

        big_pattern = _big_pattern(big_panel)
        if _symmetry_matches(big_pattern) != [target]:
            return False

        if not _valid_small_panel(small_panel, sprite_color, target):
            return False

        expected = _apply_transform(small_panel, target)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        wrong_outputs = [
            _apply_transform(small_panel, name)
            for name in _transform_results(small_panel)
            if name != target
        ]
        if any(np.array_equal(expected, wrong) for wrong in wrong_outputs):
            return False

        if int(get_bg(grid)) != BG:
            return False
    except Exception:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    target = _take_from_bag(_transform_bag, TRANSFORMS)
    orient = _take_from_bag(_orient_bag, SEPARATOR_ORIENTS)

    size_pair = _choose_sizes(target, orient)
    if size_pair is None:
        return None
    big_shape, small_shape = size_pair

    available = [c for c in range(1, 10)]
    sep_color = random.choice(available)
    available.remove(sep_color)
    sprite_color = random.choice(available)
    available.remove(sprite_color)
    palette_size = random.randint(3, min(4, len(available)))
    pattern_palette = random.sample(available, palette_size)

    pattern = _make_large_pattern(target, pattern_palette, big_shape)
    if pattern is None:
        return None
    big_panel = _embed_pattern(big_shape, pattern)
    if big_panel is None:
        return None

    small_panel = _make_small_panel(small_shape, sprite_color, target)
    if small_panel is None:
        return None
    if not _no_extra_nonbg_full_line(small_panel, BG):
        return None

    grid = _assemble_grid(big_panel, small_panel, sep_color, orient)
    if max(grid.shape) > 30:
        return None
    if not _validate_instance(grid, target, sprite_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-instruction ARC input")
