import random
from collections import Counter, defaultdict

import numpy as np


AXIS_LR = "lr"
AXIS_UD = "ud"

# --------------- solver helpers ---------------


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, grid.flatten())))


def _mirror_coord(coord: tuple[int, int], shape: tuple[int, int], axis: str) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    if axis == AXIS_LR:
        return r, w - 1 - c
    return h - 1 - r, c


def _mirror_local(coord: tuple[int, int], shape: tuple[int, int], axis: str) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    if axis == AXIS_LR:
        return r, w - 1 - c
    return h - 1 - r, c


def _crop_half(grid: np.ndarray, axis: str, side: int) -> np.ndarray:
    if axis == AXIS_LR:
        half = grid.shape[1] // 2
        return grid[:, side * half : (side + 1) * half].copy()
    half = grid.shape[0] // 2
    return grid[side * half : (side + 1) * half, :].copy()


def _global_from_local(local: tuple[int, int], half_shape: tuple[int, int], axis: str, side: int) -> tuple[int, int]:
    r, c = local
    hh, hw = half_shape
    if axis == AXIS_LR:
        return r, c if side == 0 else hw + c
    return (r if side == 0 else hh + r), c


def _iter_pair_coords(shape: tuple[int, int], axis: str):
    h, w = shape
    if axis == AXIS_LR:
        assert w % 2 == 0
        half_w = w // 2
        for r in range(h):
            for c in range(half_w):
                yield (r, c), (r, w - 1 - c)
    else:
        assert h % 2 == 0
        half_h = h // 2
        for r in range(half_h):
            for c in range(w):
                yield (r, c), (h - 1 - r, c)


def _marker_side(marker_positions: list[tuple[int, int]], shape: tuple[int, int], axis: str) -> int | None:
    if axis == AXIS_LR:
        half = shape[1] // 2
        sides = {0 if c < half else 1 for _, c in marker_positions}
    else:
        half = shape[0] // 2
        sides = {0 if r < half else 1 for r, _ in marker_positions}
    if len(sides) != 1:
        return None
    return sides.pop()


def _same_line(marker_positions: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in marker_positions}
    cols = {c for _, c in marker_positions}
    return len(rows) == 1 or len(cols) == 1


def _half_placeholder_counts(grid: np.ndarray, color: int, axis: str) -> tuple[int, int]:
    if axis == AXIS_LR:
        half = grid.shape[1] // 2
        left = int(np.count_nonzero(grid[:, :half] == color))
        right = int(np.count_nonzero(grid[:, half:] == color))
        return left, right
    half = grid.shape[0] // 2
    top = int(np.count_nonzero(grid[:half, :] == color))
    bottom = int(np.count_nonzero(grid[half:, :] == color))
    return top, bottom


def _candidate_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = _colors_present(g)
    counts = Counter(map(int, g.flatten()))
    solutions: list[dict] = []

    for axis in (AXIS_LR, AXIS_UD):
        if axis == AXIS_LR and w % 2 != 0:
            continue
        if axis == AXIS_UD and h % 2 != 0:
            continue

        marker_colors = [color for color in colors if counts[color] == 2]
        for marker_color in marker_colors:
            marker_positions = [tuple(map(int, pos)) for pos in np.argwhere(g == marker_color)]
            if not _same_line(marker_positions):
                continue
            side = _marker_side(marker_positions, g.shape, axis)
            if side is None:
                continue
            marker_set = set(marker_positions)

            for placeholder_color in colors:
                if placeholder_color == marker_color or counts[placeholder_color] < 2:
                    continue
                c0, c1 = _half_placeholder_counts(g, placeholder_color, axis)
                if c0 == 0 or c1 == 0:
                    continue

                valid = True
                for a, b in _iter_pair_coords(g.shape, axis):
                    aw = a in marker_set or int(g[a]) == placeholder_color
                    bw = b in marker_set or int(g[b]) == placeholder_color
                    if aw and bw:
                        valid = False
                        break
                    if not aw and not bw and int(g[a]) != int(g[b]):
                        valid = False
                        break
                if not valid:
                    continue

                for pos in marker_positions:
                    mirrored = _mirror_coord(pos, g.shape, axis)
                    if mirrored in marker_set or int(g[mirrored]) == placeholder_color:
                        valid = False
                        break
                if not valid:
                    continue

                solutions.append(
                    {
                        "axis": axis,
                        "side": side,
                        "marker_color": int(marker_color),
                        "placeholder_color": int(placeholder_color),
                        "marker_positions": tuple(sorted(marker_positions)),
                    }
                )
    return solutions


def _find_unique_solution(grid: np.ndarray) -> dict:
    solutions = _candidate_solutions(grid)
    assert len(solutions) == 1, f"expected unique interpretation, got {len(solutions)}"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the marked half of a mirrored pattern and repair that half from the discarded mirror side.

    Input structure:
    - The grid contains a full pattern that becomes left-right symmetric or top-bottom symmetric
      once one placeholder damage color and two aligned marker pixels are ignored.
    - The two same-color markers lie in the same half, sharing one row or one column.
    - Both halves may contain placeholder damage cells, but no mirror pair is damaged on both sides.

    Rule:
    - Determine the reflection axis and keep only the half containing the two markers.
    - Crop that half.
    - Inside the kept crop, replace every placeholder cell and both marker cells by the color
      at their mirror positions in the discarded half.

    Key insight:
    - The markers choose which half to output, but the other half is only a repair reference:
      you must copy back just the mirrored counterparts of damaged cells instead of flipping an entire half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    spec = _find_unique_solution(g)
    axis = spec["axis"]
    side = spec["side"]
    placeholder_color = spec["placeholder_color"]
    marker_positions = set(spec["marker_positions"])

    out = _crop_half(g, axis, side)
    half_shape = out.shape

    for local_r in range(half_shape[0]):
        for local_c in range(half_shape[1]):
            global_pos = _global_from_local((local_r, local_c), half_shape, axis, side)
            if global_pos in marker_positions or int(g[global_pos]) == placeholder_color:
                mirror_pos = _mirror_coord(global_pos, g.shape, axis)
                repaired = int(g[mirror_pos])
                assert repaired != placeholder_color
                out[local_r, local_c] = repaired

    return out


# --------------- generator (o5x) ---------------

PALETTE = tuple(range(10))
GEN_BG = 0
GEN_ATTEMPTS = 400
HALF_MIN = 5
HALF_MAX = 15
MIN_NON_BG = 14
MAX_FILL_RATIO = 0.58
SHAPE_TYPES = ("rect", "rect", "bar", "ell")


def _reflect_half(half: np.ndarray, axis: str) -> np.ndarray:
    return np.fliplr(half).copy() if axis == AXIS_LR else np.flipud(half).copy()


def _is_axis_symmetric(grid: np.ndarray, axis: str) -> bool:
    if axis == AXIS_LR and grid.shape[1] % 2 != 0:
        return False
    if axis == AXIS_UD and grid.shape[0] % 2 != 0:
        return False
    for a, b in _iter_pair_coords(grid.shape, axis):
        if int(grid[a]) != int(grid[b]):
            return False
    return True


def _shape_patch(shape_type: str, height: int, width: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    if shape_type == "rect":
        patch[:, :] = True
    elif shape_type == "bar":
        if height >= width:
            patch[:, : max(1, min(width, 2))] = True
        else:
            patch[: max(1, min(height, 2)), :] = True
    else:
        patch[:, 0] = True
        patch[-1, :] = True
        if width > 2 and height > 2 and random.random() < 0.5:
            patch[:, -1] = True
    return patch


def _paint_shape(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    view[patch] = color


def _free_positions(occ: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = occ.shape
    ph, pw = patch.shape
    out = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if np.any(occ[top : top + ph, left : left + pw] & patch):
                continue
            out.append((top, left))
    return out


def _generate_half_pattern(height: int, width: int, palette: list[int], bg: int) -> np.ndarray | None:
    half = np.full((height, width), bg, dtype=int)
    occ = np.zeros_like(half, dtype=bool)

    n_shapes = random.randint(4, 7)
    for _ in range(n_shapes):
        for _ in range(25):
            shape_type = random.choice(SHAPE_TYPES)
            ph = random.randint(1, min(4, height))
            pw = random.randint(1, min(5, width))
            if ph * pw < 3:
                continue
            patch = _shape_patch(shape_type, ph, pw)
            if int(patch.sum()) < 3:
                continue
            positions = _free_positions(occ, patch)
            if not positions:
                continue
            top, left = random.choice(positions)
            color = random.choice(palette)
            _paint_shape(half, patch, top, left, color)
            occ[top : top + ph, left : left + pw] |= patch
            break

    non_bg = int(np.count_nonzero(half != bg))
    if non_bg < MIN_NON_BG or non_bg > int(height * width * MAX_FILL_RATIO):
        return None
    if len(_colors_present(half)) < 3:
        return None
    row_hits = np.count_nonzero(np.any(half != bg, axis=1))
    col_hits = np.count_nonzero(np.any(half != bg, axis=0))
    if row_hits < 3 or col_hits < 3:
        return None
    if not any(np.count_nonzero(half[r] != bg) >= 2 for r in range(height)):
        return None
    if not any(np.count_nonzero(half[:, c] != bg) >= 2 for c in range(width)):
        return None
    return half


def _pair_ids_by_line(half: np.ndarray, bg: int, alignment: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    coords = [tuple(map(int, pos)) for pos in np.argwhere(half != bg)]
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    index = 0 if alignment == "col" else 1
    for r, c in coords:
        groups[(c if alignment == "col" else r)].append((r, c))

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if alignment == "row" and a[1] != b[1]:
                    pairs.append((a, b))
                if alignment == "col" and a[0] != b[0]:
                    pairs.append((a, b))
    random.shuffle(pairs)
    return pairs


def _visible_counts_ok(grid: np.ndarray, marker_color: int, placeholder_color: int, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    for color, count in counts.items():
        if color in {bg, marker_color, placeholder_color}:
            continue
        if count == 2:
            return False
    return True


def _assemble_input(
    kept_half: np.ndarray,
    axis: str,
    side: int,
    placeholder_color: int,
    marker_color: int,
    marker_pair_ids: tuple[tuple[int, int], tuple[int, int]],
    kept_damage_ids: list[tuple[int, int]],
    discarded_damage_ids: list[tuple[int, int]],
) -> np.ndarray:
    kept = kept_half.copy()
    discarded = _reflect_half(kept_half, axis)

    for r, c in kept_damage_ids:
        kept[r, c] = placeholder_color
    for pair_id in discarded_damage_ids:
        rr, cc = _mirror_local(pair_id, kept_half.shape, axis)
        discarded[rr, cc] = placeholder_color
    for r, c in marker_pair_ids:
        kept[r, c] = marker_color

    if axis == AXIS_LR:
        return np.concatenate([kept, discarded], axis=1) if side == 0 else np.concatenate([discarded, kept], axis=1)
    return np.concatenate([kept, discarded], axis=0) if side == 0 else np.concatenate([discarded, kept], axis=0)


def _other_half_as_output_shape(grid: np.ndarray, axis: str, side: int) -> np.ndarray:
    other = _crop_half(grid, axis, 1 - side)
    return np.fliplr(other) if axis == AXIS_LR else np.flipud(other)


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    axis: str,
    side: int,
    marker_color: int,
    placeholder_color: int,
) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(_candidate_solutions(inp)) != 1:
        return False

    kept_crop = _crop_half(inp, axis, side)
    if np.array_equal(kept_crop, expected):
        return False
    if np.array_equal(_other_half_as_output_shape(inp, axis, side), expected):
        return False

    cleaned = kept_crop.copy()
    cleaned[(cleaned == marker_color) | (cleaned == placeholder_color)] = GEN_BG
    if np.array_equal(cleaned, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice((AXIS_LR, AXIS_UD))
    side = random.randint(0, 1)
    alignment = random.choice(("row", "col"))

    if axis == AXIS_LR:
        height = random.randint(HALF_MIN, HALF_MAX)
        width = random.randint(HALF_MIN, HALF_MAX)
    else:
        height = random.randint(HALF_MIN, HALF_MAX)
        width = random.randint(HALF_MIN, HALF_MAX)

    specials = random.sample(PALETTE[1:], 5)
    marker_color = specials[0]
    placeholder_color = specials[1]
    palette = specials[2:]

    kept_half = _generate_half_pattern(height, width, palette, GEN_BG)
    if kept_half is None:
        return None

    pristine_full = (
        np.concatenate([kept_half, _reflect_half(kept_half, axis)], axis=1)
        if axis == AXIS_LR
        else np.concatenate([kept_half, _reflect_half(kept_half, axis)], axis=0)
    )
    if _is_axis_symmetric(pristine_full, AXIS_UD if axis == AXIS_LR else AXIS_LR):
        return None

    marker_candidates = _pair_ids_by_line(kept_half, GEN_BG, alignment)
    if not marker_candidates:
        return None

    non_bg_ids = {tuple(map(int, pos)) for pos in np.argwhere(kept_half != GEN_BG)}
    for marker_pair in marker_candidates:
        used_pair_ids = set(marker_pair)
        remaining = sorted(non_bg_ids - used_pair_ids)
        if len(remaining) < 2:
            continue

        max_kept = min(3, len(remaining) - 1)
        kept_damage_n = random.randint(1, max_kept)
        kept_damage_ids = random.sample(remaining, kept_damage_n)
        used_pair_ids.update(kept_damage_ids)
        remaining = sorted(non_bg_ids - used_pair_ids)
        if not remaining:
            continue
        discarded_damage_n = random.randint(1, min(3, len(remaining)))
        discarded_damage_ids = random.sample(remaining, discarded_damage_n)

        inp = _assemble_input(
            kept_half,
            axis,
            side,
            placeholder_color,
            marker_color,
            marker_pair,
            kept_damage_ids,
            discarded_damage_ids,
        )

        if not _visible_counts_ok(inp, marker_color, placeholder_color, GEN_BG):
            continue
        if not _semantic_checks(inp, kept_half, axis, side, marker_color, placeholder_color):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
