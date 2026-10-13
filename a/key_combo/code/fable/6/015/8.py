from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paint_coords


MAX_GENERATION_ATTEMPTS = 500


def _rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2)


def _paint_mirrored_rect(
    half: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    color: int,
) -> None:
    w = int(half.shape[1])
    half[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    mc0 = w - 1 - c1
    mc1 = w - 1 - c0
    half[r0 : r1 + 1, mc0 : mc1 + 1] = int(color)


def _paint_mirrored_points(half: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    w = int(half.shape[1])
    mirrored: list[tuple[int, int]] = []
    for r, c in coords:
        mirrored.append((int(r), int(c)))
        mirrored.append((int(r), w - 1 - int(c)))
    paint_coords(half, mirrored, int(color))


def _generate_symmetric_half(height: int, width: int, bg: int, fg_colors: list[int]) -> np.ndarray | None:
    half = np.full((height, width), int(bg), dtype=int)
    support_w = (width + 1) // 2
    motif_count = random.randint(6, 10)

    for _ in range(motif_count):
        color = random.choice(fg_colors)
        kind = random.choice(["rect", "hbar", "vbar", "dots"])
        if kind == "dots":
            n_points = random.randint(1, 3)
            coords = [(random.randrange(height), random.randrange(support_w)) for _ in range(n_points)]
            _paint_mirrored_points(half, coords, color)
            continue

        max_h = min(3, height - 1)
        max_w = min(2, support_w)
        r0 = random.randrange(height - 1)
        c0 = random.randrange(support_w)
        if kind == "rect":
            r1 = min(height - 1, r0 + random.randint(0, max_h - 1))
            c1 = min(support_w - 1, c0 + random.randint(0, max_w - 1))
        elif kind == "hbar":
            r1 = r0
            c1 = min(support_w - 1, c0 + random.randint(0, max(1, max_w) - 1))
        else:
            r1 = min(height - 1, r0 + random.randint(0, max(1, max_h) - 1))
            c1 = c0
        _paint_mirrored_rect(half, r0, r1, c0, c1, color)

    non_bg = int(np.sum(half != int(bg)))
    if non_bg < max(8, width):
        return None
    if non_bg > (height * width * 2) // 3:
        return None

    counts = Counter(map(int, half.flatten()))
    for color in fg_colors:
        if counts[int(color)] < 2:
            return None
    return half


def _random_left_rect(height: int, width: int) -> tuple[int, int, int, int]:
    max_h = max(1, min(4, height // 2))
    max_w = max(1, min(3, max(1, width // 2)))
    rect_h = random.randint(1, max_h)
    rect_w = random.randint(1, max_w)
    r0 = random.randint(0, height - rect_h)
    left_limit = max(0, (width // 2) - rect_w)
    c0 = random.randint(0, left_limit)
    return r0, r0 + rect_h - 1, c0, c0 + rect_w - 1


def _rotated_rect(rect: tuple[int, int, int, int], height: int, width: int) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = rect
    return height - 1 - r1, height - 1 - r0, width - 1 - c1, width - 1 - c0


def _paint_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, r1, c0, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _find_marker_info(grid: np.ndarray, sep_col: int) -> tuple[int, str]:
    colors, counts = np.unique(grid, return_counts=True)
    for color, count in zip(colors.tolist(), counts.tolist()):
        if int(count) != 2:
            continue
        coords = np.argwhere(grid == int(color))
        if coords.shape != (2, 2):
            continue
        same_half = np.all(coords[:, 1] < sep_col) or np.all(coords[:, 1] > sep_col)
        aligned = int(coords[0, 0]) == int(coords[1, 0]) or int(coords[0, 1]) == int(coords[1, 1])
        if same_half and aligned:
            return int(color), "left" if int(coords[0, 1]) < sep_col else "right"
    raise AssertionError("failed to identify marker color")


def _infer_occluder_color(grid: np.ndarray, sep_col: int, marker_color: int) -> int:
    bg = int(get_bg(grid))
    sep_color = int(grid[0, sep_col])
    excluded = {bg, sep_color, int(marker_color)}

    best_color: int | None = None
    best_score: tuple[int, int] | None = None
    colors, counts = np.unique(grid, return_counts=True)
    for color, count in zip(colors.tolist(), counts.tolist()):
        color = int(color)
        if color in excluded:
            continue
        comps = find_components(
            grid,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        if len(comps) < 2:
            continue

        total_area = 0
        valid = True
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            area = int(comp["area"])
            if c0 <= sep_col <= c1:
                valid = False
                break
            if area != (r1 - r0 + 1) * (c1 - c0 + 1):
                valid = False
                break
            total_area += area
        if not valid:
            continue

        score = (total_area, int(count))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color

    if best_color is not None:
        return best_color

    fallback: list[tuple[int, int]] = []
    for color, count in zip(colors.tolist(), counts.tolist()):
        color = int(color)
        if color not in excluded:
            fallback.append((int(count), color))
    if not fallback:
        raise AssertionError("failed to infer occluder color")
    return max(fallback)[1]


def _reconstruct_half(selected: np.ndarray, other: np.ndarray, corrupted: set[int]) -> np.ndarray:
    height, width = map(int, selected.shape)
    out = np.empty_like(selected)
    for r in range(height):
        rr = height - 1 - r
        for c in range(width):
            mc = width - 1 - c
            track = [
                int(selected[r, c]),
                int(other[rr, mc]),
                int(selected[r, mc]),
                int(other[rr, c]),
            ]
            visible = [value for value in track if value not in corrupted]
            if not visible:
                raise AssertionError("repair track has no visible source")
            out[r, c] = Counter(visible).most_common(1)[0][0]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Repair the marked half by combining 180-degree correspondence across halves
    with within-half horizontal mirror fallback, then crop to that half.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and grid.size > 0, "expected a non-empty 2D grid"

    height, total_width = map(int, grid.shape)
    assert total_width % 2 == 1, "expected one centered separator column"
    sep_col = total_width // 2
    assert np.all(grid[:, sep_col] == grid[0, sep_col]), "separator must be a solid column"

    marker_color, marked_half = _find_marker_info(grid, sep_col)
    occluder_color = _infer_occluder_color(grid, sep_col, marker_color)
    corrupted = {int(marker_color), int(occluder_color)}

    left = grid[:, :sep_col]
    right = grid[:, sep_col + 1 :]
    selected = left if marked_half == "left" else right
    other = right if marked_half == "left" else left
    repaired = _reconstruct_half(selected, other, corrupted)
    assert repaired.shape == (height, sep_col)
    return repaired


def _choose_palette() -> tuple[int, int, int, int, list[int]]:
    colors = random.sample(Color.ALL_COLORS, k=7)
    bg, sep, occ, marker = map(int, colors[:4])
    fg = [int(color) for color in colors[4:]]
    return bg, sep, occ, marker, fg


def _pick_marker_positions(
    clean_half: np.ndarray,
    occ_mask: np.ndarray,
    other_occ_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    height, width = map(int, clean_half.shape)
    for _ in range(60):
        row = random.randrange(height)
        cols = list(range(width))
        random.shuffle(cols)
        chosen: list[tuple[int, int]] = []
        for c in cols:
            rr = height - 1 - row
            cc = width - 1 - c
            mc = width - 1 - c
            if occ_mask[row, c]:
                continue
            if other_occ_mask[rr, cc] and occ_mask[row, mc]:
                continue
            chosen.append((row, c))
            if len(chosen) == 2:
                return chosen
    for _ in range(60):
        col = random.randrange(width)
        rows = list(range(height))
        random.shuffle(rows)
        chosen = []
        for r in rows:
            rr = height - 1 - r
            cc = width - 1 - col
            mc = width - 1 - col
            if occ_mask[r, col]:
                continue
            if other_occ_mask[rr, cc] and occ_mask[r, mc]:
                continue
            chosen.append((r, col))
            if len(chosen) == 2:
                return chosen
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(8, 14)
    half_width = random.randint(6, 12)
    if 2 * half_width + 1 > 30:
        return None

    bg, sep, occ, marker, fg_colors = _choose_palette()
    clean_left = _generate_symmetric_half(height, half_width, bg, fg_colors)
    if clean_left is None:
        return None
    clean_right = _rot180(clean_left)

    marked_half = random.choice(["left", "right"])
    clean_selected = clean_left if marked_half == "left" else clean_right
    clean_other = clean_right if marked_half == "left" else clean_left

    occ_selected = np.zeros((height, half_width), dtype=bool)
    occ_other = np.zeros((height, half_width), dtype=bool)

    collision_rect = _random_left_rect(height, half_width)
    _paint_rect(occ_selected, collision_rect)
    _paint_rect(occ_other, _rotated_rect(collision_rect, height, half_width))

    for _ in range(random.randint(1, 2)):
        extra = _random_left_rect(height, half_width)
        _paint_rect(occ_selected, extra)
    for _ in range(random.randint(1, 2)):
        extra = _random_left_rect(height, half_width)
        _paint_rect(occ_other, extra)

    if int(occ_selected.sum()) < 3 or int(occ_other.sum()) < 3:
        return None
    if int(occ_selected.sum()) > (height * half_width) // 3:
        return None
    if int(occ_other.sum()) > (height * half_width) // 3:
        return None

    markers = _pick_marker_positions(clean_selected, occ_selected, occ_other)
    if markers is None:
        return None

    selected_obs = clean_selected.copy()
    other_obs = clean_other.copy()
    selected_obs[occ_selected] = int(occ)
    other_obs[occ_other] = int(occ)
    for r, c in markers:
        selected_obs[r, c] = int(marker)

    expected = clean_selected.copy()

    if marked_half == "left":
        left_obs, right_obs = selected_obs, other_obs
    else:
        left_obs, right_obs = other_obs, selected_obs

    full = np.full((height, 2 * half_width + 1), int(bg), dtype=int)
    full[:, :half_width] = left_obs
    full[:, half_width] = int(sep)
    full[:, half_width + 1 :] = right_obs

    if not np.array_equal(clean_right, _rot180(clean_left)):
        return None
    if int(np.sum(full == int(marker))) != 2:
        return None
    if np.any(full[:, half_width] != int(sep)):
        return None

    counts = Counter(map(int, full.flatten()))
    occ_count = counts[int(occ)]
    for color, count in counts.items():
        if color in {int(bg), int(sep), int(marker), int(occ)}:
            continue
        if occ_count <= int(count):
            return None

    inferred_marker, inferred_side = _find_marker_info(full, half_width)
    if inferred_marker != int(marker) or inferred_side != marked_half:
        return None
    if _infer_occluder_color(full, half_width, marker) != int(occ):
        return None
    if not np.array_equal(main(full), expected):
        return None
    return full, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC case")
