from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SEED_PALETTE = (BLUE, RED, GREEN, ORANGE)
SIDES = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _marker_column(region: np.ndarray) -> int:
    h, w = region.shape
    candidates = []
    for c in range(w):
        if all(np.unique(np.delete(region[r], c)).size == 1 for r in range(h)):
            candidates.append(c)
    assert len(candidates) == 1, "expected one marker column in the upper band"
    return int(candidates[0])


def _diffuse_upper(region: np.ndarray) -> tuple[np.ndarray, list[int], list[str], int]:
    h, _w = region.shape
    marker_c = _marker_column(region)

    row_colors: list[int] = []
    seed_rows: list[int] = []
    seed_colors: list[int] = []
    for r in range(h):
        outside = np.delete(region[r], marker_c)
        vals = np.unique(outside)
        assert vals.size == 1, "each upper-band row must be monochrome outside the marker column"
        color = int(vals[0])
        row_colors.append(color)
        if color != BLACK:
            seed_rows.append(r)
            seed_colors.append(color)

    assert 2 <= len(seed_rows) <= 3, "expected 2 or 3 upper seed bars"
    assert len(set(seed_colors)) == len(seed_colors), "upper seed colors must be distinct"

    out = np.empty_like(region)
    lookup_colors: list[int] = []
    row_kinds: list[str] = []
    seed_set = set(seed_rows)

    for r in range(h):
        if r in seed_set:
            color = row_colors[r]
            out[r, :] = color
            out[r, marker_c] = YELLOW
            lookup_colors.append(color)
            row_kinds.append("seed")
            continue

        nearest_distance = min(abs(r - sr) for sr in seed_rows)
        nearest = [seed_colors[i] for i, sr in enumerate(seed_rows) if abs(r - sr) == nearest_distance]
        unique_nearest = set(nearest)
        if len(unique_nearest) == 1:
            color = next(iter(unique_nearest))
            out[r, :] = color
            out[r, marker_c] = PINK
            lookup_colors.append(color)
            row_kinds.append("filled")
        else:
            out[r, :] = PINK
            lookup_colors.append(PINK)
            row_kinds.append("tie")

    return out, lookup_colors, row_kinds, marker_c


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    return int(rows[0])


def _find_lower_object_and_marker(
    lower: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    marker_pts = np.argwhere(lower == RED)
    assert marker_pts.shape[0] == 1, "expected one red marker pixel in the lower band"
    marker = tuple(map(int, marker_pts[0]))

    comps = _components(lower == GREY)
    assert len(comps) == 1, "expected one grey 4-connected lower object"
    coords = comps[0]
    bbox = _bbox_from_coords(coords)
    obj_mask = np.zeros_like(lower, dtype=bool)
    obj_mask[coords[:, 0], coords[:, 1]] = True
    return marker, obj_mask, coords, bbox


def _mirror_side(
    marker: tuple[int, int],
    bbox: tuple[int, int, int, int],
    obj_mask: np.ndarray,
) -> str:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    sides: list[str] = []
    if mc == c0 - 1 and r0 <= mr <= r1 and obj_mask[mr, c0]:
        sides.append("left")
    if mc == c1 + 1 and r0 <= mr <= r1 and obj_mask[mr, c1]:
        sides.append("right")
    if mr == r0 - 1 and c0 <= mc <= c1 and obj_mask[r0, mc]:
        sides.append("up")
    if mr == r1 + 1 and c0 <= mc <= c1 and obj_mask[r1, mc]:
        sides.append("down")
    assert len(sides) == 1, "red marker must identify exactly one object side"
    return sides[0]


def _mirrored_bbox(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = _bbox_h(bbox)
    w = _bbox_w(bbox)
    if side == "left":
        return r0, c0 - w, r1, c0 - 1
    if side == "right":
        return r0, c1 + 1, r1, c1 + w
    if side == "up":
        return r0 - h, c0, r0 - 1, c1
    if side == "down":
        return r1 + 1, c0, r1 + h, c1
    raise ValueError(f"unknown side: {side}")


def _mirror_coords(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if side == "left":
        mirrored = np.column_stack((coords[:, 0], 2 * c0 - 1 - coords[:, 1]))
    elif side == "right":
        mirrored = np.column_stack((coords[:, 0], 2 * c1 + 1 - coords[:, 1]))
    elif side == "up":
        mirrored = np.column_stack((2 * r0 - 1 - coords[:, 0], coords[:, 1]))
    elif side == "down":
        mirrored = np.column_stack((2 * r1 + 1 - coords[:, 0], coords[:, 1]))
    else:
        raise ValueError(f"unknown side: {side}")
    return mirrored.astype(int)


def _inside_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse the upper seed bars by nearest row, then color a mirrored lower copy from the spectral row it lands on.

    Input structure:
    - A full-width grey separator row splits the grid into an upper band and a lower band.
    - In the upper band, one marker column differs from otherwise monochrome rows: black rows have yellow there, and 2 or 3 seed-bar rows have pink there and a distinct bar color elsewhere.
    - In the lower band, one grey 4-connected object is adjacent to one red marker pixel on exactly one outer side.

    Rule:
    - Find the separator row. In the upper band, recolor each non-seed row by the nearest seed-bar color; if two different seed colors are equally near, make that whole row pink.
    - Keep each seed row in its own bar color but change its marker-column cell to yellow; unique-nearest filled rows keep a pink marker-column cell.
    - In the lower band, reflect the grey object across the outer edge of the side indicated by the red marker, leaving the original object unchanged.
    - Recolor the mirrored copy uniformly by the diffused upper-band color at the copy bbox's center row index, mapping lower local row k to upper local row k and clamping to the last upper row when needed; if the copy covers the red marker, the copy wins.

    Key insight:
    - The mirror operation chooses both where the lower copy appears and which upper spectral row is consulted for its output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected one ARC-sized 2D grid"

    sep_r = _find_separator_row(g)
    assert 1 <= sep_r < g.shape[0] - 1, "separator must leave both an upper and a lower band"
    upper = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]

    out = g.copy()
    upper_out, lookup_colors, _row_kinds, _marker_c = _diffuse_upper(upper)
    out[:sep_r, :] = upper_out

    marker, obj_mask, obj_coords, obj_bbox = _find_lower_object_and_marker(lower)
    side = _mirror_side(marker, obj_bbox, obj_mask)
    copy_coords = _mirror_coords(obj_coords, obj_bbox, side)
    copy_bbox = _bbox_from_coords(copy_coords)
    assert _inside_bbox(copy_bbox, *lower.shape), "mirrored copy must stay inside the lower band"

    center_local_row = (copy_bbox[0] + copy_bbox[2]) // 2
    mapped_upper_row = min(center_local_row, len(lookup_colors) - 1)
    copy_color = int(lookup_colors[mapped_upper_row])

    for rr, cc in copy_coords:
        out[sep_r + 1 + int(rr), int(cc)] = copy_color

    return out


def _row_kinds_for_seed_rows(height: int, seed_rows: tuple[int, ...]) -> list[str]:
    kinds: list[str] = []
    seed_set = set(seed_rows)
    for r in range(height):
        if r in seed_set:
            kinds.append("seed")
            continue
        nearest_distance = min(abs(r - sr) for sr in seed_rows)
        tied = sum(abs(r - sr) == nearest_distance for sr in seed_rows) >= 2
        kinds.append("tie" if tied else "filled")
    return kinds


def _sample_upper_band(height: int, width: int) -> tuple[np.ndarray, list[int], list[str]]:
    choices: list[tuple[int, ...]] = []
    for n_seeds in (2, 3):
        for seed_rows in itertools.combinations(range(height), n_seeds):
            if any(seed_rows[i + 1] - seed_rows[i] < 2 for i in range(len(seed_rows) - 1)):
                continue
            kinds = _row_kinds_for_seed_rows(height, seed_rows)
            if "tie" not in kinds:
                continue
            if "filled" not in kinds:
                continue
            choices.append(seed_rows)

    assert choices, "failed to find upper-band seed placements"
    seed_rows = random.choice(choices)
    seed_colors = random.sample(list(SEED_PALETTE), len(seed_rows))

    upper = np.zeros((height, width), dtype=int)
    upper[:, 0] = YELLOW
    for row, color in zip(seed_rows, seed_colors):
        upper[row, :] = color
        upper[row, 0] = PINK

    upper_out, lookup_colors, row_kinds, marker_c = _diffuse_upper(upper)
    assert marker_c == 0, "generator expects the first column to be the marker column"
    assert "tie" in row_kinds
    assert len(set(color for color in lookup_colors if color != PINK)) >= 2
    assert np.any(upper_out != upper)
    return upper, lookup_colors, row_kinds


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _bbox_is_full(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_shape(h: int, w: int) -> np.ndarray:
    min_area = max(5, h + w - 2)
    removable = h * w - min_area
    for _ in range(120):
        mask = np.ones((h, w), dtype=bool)
        target_remove = random.randint(1 if removable > 0 else 0, max(0, min(4, removable)))
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        removed = 0
        for r, c in cells:
            if removed >= target_remove:
                break
            mask[r, c] = False
            if not _bbox_is_full(mask) or not _is_connected(mask):
                mask[r, c] = True
                continue
            removed += 1
        if removed == target_remove and _bbox_is_full(mask) and _is_connected(mask):
            return mask
    return np.ones((h, w), dtype=bool)


def _boundary_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    if side == "right":
        return [(r, w - 1) for r in range(h) if mask[r, w - 1]]
    if side == "up":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "down":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    raise ValueError(f"unknown side: {side}")


def _marker_from_boundary(
    bbox: tuple[int, int, int, int],
    side: str,
    boundary_cell: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, _r1, _c1 = bbox
    rr, cc = boundary_cell
    if side == "left":
        return r0 + rr, c0 - 1
    if side == "right":
        return r0 + rr, c0 + cc + 1
    if side == "up":
        return r0 - 1, c0 + cc
    if side == "down":
        return r0 + rr + 1, c0 + cc
    raise ValueError(f"unknown side: {side}")


def _build_lower_band(height: int, width: int, row_kinds: list[str]) -> np.ndarray | None:
    for _ in range(220):
        side = random.choice(SIDES)
        h = random.choice([3, 5])
        if h > height:
            continue
        if side in ("up", "down") and 2 * h > height:
            continue

        w_max = min(5, width)
        if w_max < 3:
            continue
        w = random.randint(3, w_max)
        if side in ("left", "right") and 2 * w > width:
            continue

        shape = _random_shape(h, w)
        candidates: list[tuple[tuple[int, int, int, int], int]] = []
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                copy_bbox = _mirrored_bbox(bbox, side)
                if not _inside_bbox(copy_bbox, height, width):
                    continue
                center_row = (copy_bbox[0] + copy_bbox[2]) // 2
                if row_kinds[min(center_row, len(row_kinds) - 1)] == "tie":
                    continue
                candidates.append((bbox, center_row))

        if not candidates:
            continue

        bbox, center_row = random.choice(candidates)
        boundary = _boundary_cells(shape, side)
        if not boundary:
            continue
        marker = _marker_from_boundary(bbox, side, random.choice(boundary))

        lower = np.zeros((height, width), dtype=int)
        for rr, cc in np.argwhere(shape):
            lower[bbox[0] + int(rr), bbox[1] + int(cc)] = GREY
        lower[int(marker[0]), int(marker[1])] = RED

        marker2, obj_mask2, obj_coords2, bbox2 = _find_lower_object_and_marker(lower)
        side2 = _mirror_side(marker2, bbox2, obj_mask2)
        copy_coords2 = _mirror_coords(obj_coords2, bbox2, side2)
        copy_bbox2 = _bbox_from_coords(copy_coords2)
        mapped_kind = row_kinds[min((copy_bbox2[0] + copy_bbox2[2]) // 2, len(row_kinds) - 1)]
        if mapped_kind == "tie":
            continue
        assert side2 == side
        assert center_row == (copy_bbox2[0] + copy_bbox2[2]) // 2
        return lower

    return None


def _validate_generated_case(grid: np.ndarray) -> None:
    sep_r = _find_separator_row(grid)
    upper = grid[:sep_r, :]
    lower = grid[sep_r + 1 :, :]

    upper_out, lookup_colors, row_kinds, marker_c = _diffuse_upper(upper)
    assert marker_c == 0
    assert "tie" in row_kinds
    assert len(set(color for color in lookup_colors if color != PINK)) >= 2

    marker, obj_mask, obj_coords, obj_bbox = _find_lower_object_and_marker(lower)
    side = _mirror_side(marker, obj_bbox, obj_mask)
    copy_coords = _mirror_coords(obj_coords, obj_bbox, side)
    copy_bbox = _bbox_from_coords(copy_coords)
    center_row = (copy_bbox[0] + copy_bbox[2]) // 2
    assert row_kinds[min(center_row, len(row_kinds) - 1)] != "tie"

    out = main(grid)
    expected_color = lookup_colors[min(center_row, len(lookup_colors) - 1)]
    assert np.any(upper_out != upper), "upper diffusion must change the upper band"
    assert np.any(out[sep_r + 1 :, :] != lower), "lower mirror must change the lower band"
    assert np.all(out[sep_r + 1 + copy_coords[:, 0], copy_coords[:, 1]] == expected_color)
    assert np.all(out[sep_r + 1 + obj_coords[:, 0], obj_coords[:, 1]] == GREY)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        width = random.randint(12, 18)
        upper_h = random.randint(7, 10)
        lower_h = random.randint(7, upper_h)

        upper, _lookup_colors, row_kinds = _sample_upper_band(upper_h, width)
        lower = _build_lower_band(lower_h, width, row_kinds)
        if lower is None:
            continue

        grid = np.vstack(
            [
                upper,
                np.full((1, width), GREY, dtype=int),
                lower,
            ]
        ).astype(int)

        try:
            _validate_generated_case(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
