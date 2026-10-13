from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
SHAPE_RETRIES = 220
CHAMBER_RETRIES = 220


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _divider_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [int(i) for i in range(grid.shape[0]) if bool(np.all(grid[i, :] == color))]
    return [int(i) for i in range(grid.shape[1]) if bool(np.all(grid[:, i] == color))]


def _bands(dividers: list[int], n: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(set(dividers)), n]
    bands = []
    for left, right in zip(bounds, bounds[1:]):
        start, end = left + 1, right - 1
        if start <= end:
            bands.append((int(start), int(end)))
    return bands


def _chambers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rows = _bands(_divider_indices(grid, GREY, axis=0), grid.shape[0])
    cols = _bands(_divider_indices(grid, GREY, axis=1), grid.shape[1])
    out = []
    for r0, r1 in rows:
        for c0, c1 in cols:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(patch == GREY):
                out.append((r0, c0, r1, c1))
    return out


def _valid_axes(visible: np.ndarray, unknown: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = visible.shape
    coords = np.argwhere(visible)
    if coords.size == 0:
        return []
    candidates: list[tuple[int, int, int]] = []
    for x2 in range(2 * w - 1):
        added = set()
        matched = 0
        ok = True
        for r, c in coords:
            mc = int(x2 - c)
            if not (0 <= mc < w):
                ok = False
                break
            if visible[r, mc]:
                matched += 1
            elif unknown[r, mc]:
                added.add((int(r), mc))
            else:
                ok = False
                break
        if ok:
            candidates.append((int(x2), int(len(added)), int(matched)))
    return candidates


def _select_axis(visible: np.ndarray, unknown: np.ndarray) -> int:
    candidates = _valid_axes(visible, unknown)
    if not candidates:
        raise AssertionError("no valid symmetry axis")
    center = visible.shape[1] - 1
    candidates.sort(key=lambda item: (item[1], -item[2], abs(item[0] - center), item[0]))
    return int(candidates[0][0])


def _complete_visible(visible: np.ndarray, x2: int) -> np.ndarray:
    out = visible.copy()
    h, w = visible.shape
    for r, c in np.argwhere(visible):
        mc = int(x2 - c)
        if 0 <= mc < w:
            out[r, mc] = True
    return out


def _solve_patch(patch: np.ndarray) -> dict:
    red_mask = patch == RED
    visible = (patch != BLACK) & (patch != RED)
    colors = np.unique(patch[visible])
    if colors.size == 0:
        return {
            "color": None,
            "count": 0,
        }
    color = int(colors[0])
    axis = _select_axis(visible, red_mask)
    completed = _complete_visible(visible, axis)
    return {
        "color": color,
        "count": int(completed.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase red occluders, restore each chamber's left-right symmetry, then keep and flood-fill only the largest restored chamber.

    Input structure:
    - Grey full rows and columns form a lattice that splits the canvas into rectangular chambers.
    - A subset of chambers contains one non-grey pattern color plus one red filled rectangle occluder.
    - Other chambers are black; grey lattice cells are not part of any chamber interior.

    Rule:
    - In each chamber, treat red as an occluding rectangle and erase it to black.
    - From the remaining non-black pixels, detect the chamber-local left-right mirror axis and mirror the visible pixels across it to restore the full symmetric pattern.
    - Count restored pattern pixels per chamber, keep the chamber or chambers with the maximum count, fill each winning chamber completely with its pattern color, and clear all other patterned chambers to black.

    Key insight:
    - The red rectangle marks "unknown" cells rather than true absence, so symmetry must be inferred while allowing mirrors to land inside the erased rectangle; only the restored size reveals which chamber survives.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == RED] = BLACK

    chamber_infos = []
    for bbox in _chambers(g):
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        info = _solve_patch(patch)
        chamber_infos.append((bbox, info))

    counts = [info["count"] for _, info in chamber_infos if info["color"] is not None]
    if not counts:
        return out
    best = max(counts)

    for (r0, c0, r1, c1), info in chamber_infos:
        if info["color"] is None:
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        elif info["count"] == best:
            out[r0 : r1 + 1, c0 : c1 + 1] = int(info["color"])
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out


def _mask_from_rows(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_TEMPLATES: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {
    (4, 4, 6): [
        (
            _mask_from_rows("1111", "0110", "0000", "0000"),
            _mask_from_rows("0000", "1100", "1100", "0000"),
        )
    ],
    (4, 4, 8): [
        (
            _mask_from_rows("0000", "1001", "1111", "0110"),
            _mask_from_rows("0000", "1100", "1100", "0000"),
        )
    ],
    (5, 4, 6): [
        (
            _mask_from_rows("0000", "0000", "0000", "1111", "0110"),
            _mask_from_rows("0000", "0000", "1100", "1100", "0000"),
        )
    ],
    (5, 4, 8): [
        (
            _mask_from_rows("0000", "0000", "1001", "1111", "0110"),
            _mask_from_rows("0000", "0000", "1100", "1100", "0000"),
        )
    ],
    (4, 5, 4): [
        (
            _mask_from_rows("00100", "01110", "00000", "00000"),
            _mask_from_rows("00000", "11000", "11000", "00000"),
        )
    ],
    (4, 5, 5): [
        (
            _mask_from_rows("00000", "00100", "00100", "01110"),
            _mask_from_rows("00000", "00000", "11000", "11000"),
        )
    ],
    (4, 5, 7): [
        (
            _mask_from_rows("00000", "00100", "01110", "01110"),
            _mask_from_rows("00000", "11000", "11000", "00000"),
        )
    ],
    (5, 5, 4): [
        (
            _mask_from_rows("00000", "00000", "01110", "00100", "00000"),
            _mask_from_rows("00000", "11000", "11000", "00000", "00000"),
        )
    ],
    (5, 5, 5): [
        (
            _mask_from_rows("01110", "01010", "00000", "00000", "00000"),
            _mask_from_rows("00000", "11000", "11000", "00000", "00000"),
        )
    ],
    (5, 5, 7): [
        (
            _mask_from_rows("00000", "00100", "01110", "01110", "00000"),
            _mask_from_rows("00000", "11000", "11000", "00000", "00000"),
        )
    ],
}


def _orbits(h: int, w: int) -> list[list[tuple[int, int]]]:
    x2 = w - 1
    res: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range((w + 1) // 2):
            mc = int(x2 - c)
            if mc == c:
                res.append([(r, c)])
            else:
                res.append([(r, c), (r, mc)])
    return res


def _mask_from_orbit_ids(h: int, w: int, orbits: list[list[tuple[int, int]]], chosen: set[int]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for idx in chosen:
        for r, c in orbits[idx]:
            mask[r, c] = True
    return mask


def _shape_ok(mask: np.ndarray, expected_x2: int) -> bool:
    if int(mask.sum()) < 4 or len(_components(mask)) != 1:
        return False
    rows = {int(r) for r, _ in np.argwhere(mask)}
    cols = {int(c) for _, c in np.argwhere(mask)}
    if len(rows) < 2 or len(cols) < 2:
        return False
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    axes = _valid_axes(mask, np.zeros_like(mask, dtype=bool))
    return len(axes) == 1 and int(axes[0][0]) == expected_x2


def _generate_shape(h: int, w: int, target_area: int) -> np.ndarray | None:
    expected_x2 = w - 1
    orbit_list = _orbits(h, w)
    weights = [len(cells) for cells in orbit_list]
    start_choices = [i for i, wt in enumerate(weights) if wt <= target_area]

    for _ in range(SHAPE_RETRIES):
        chosen = {random.choice(start_choices)}
        area = sum(weights[i] for i in chosen)
        while area < target_area:
            mask = _mask_from_orbit_ids(h, w, orbit_list, chosen)
            frontier = []
            for idx, cells in enumerate(orbit_list):
                if idx in chosen or weights[idx] > target_area - area:
                    continue
                if any(
                    0 <= nr < h
                    and 0 <= nc < w
                    and mask[nr, nc]
                    for r, c in cells
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                ):
                    frontier.append(idx)
            if not frontier:
                break
            idx = random.choice(frontier)
            chosen.add(idx)
            area += weights[idx]
        if area != target_area:
            continue
        mask = _mask_from_orbit_ids(h, w, orbit_list, chosen)
        if _shape_ok(mask, expected_x2):
            return mask
    return None


def _occluder_candidates(shape: np.ndarray) -> list[dict]:
    h, w = shape.shape
    expected_x2 = w - 1
    out = []
    for rh, rw in ((2, 2), (2, 3), (3, 2)):
        if rh > h or rw > w:
            continue
        for r0 in range(h - rh + 1):
            for c0 in range(w - rw + 1):
                rect = np.zeros((h, w), dtype=bool)
                rect[r0 : r0 + rh, c0 : c0 + rw] = True
                hidden = shape & rect
                hidden_count = int(hidden.sum())
                if hidden_count == 0 or hidden_count == int(shape.sum()):
                    continue
                visible = shape & ~rect
                visible_count = int(visible.sum())
                if not (3 <= visible_count <= 8):
                    continue
                if len(_components(visible)) != 1:
                    continue
                axes = _valid_axes(visible, rect)
                if len(axes) != 1 or int(axes[0][0]) != expected_x2:
                    continue
                if any(int(x2) == expected_x2 for x2, _, _ in _valid_axes(visible, np.zeros_like(rect))):
                    continue
                if not np.array_equal(_complete_visible(visible, expected_x2), shape):
                    continue
                out.append(
                    {
                        "rect": rect,
                        "hidden": hidden_count,
                        "visible": visible_count,
                    }
                )
    out.sort(key=lambda item: (-item["hidden"], item["visible"]))
    return out


def _pattern_candidate(h: int, w: int, color: int, forbidden_areas: set[int]) -> dict | None:
    area_options = [4, 6, 8] if w % 2 == 0 else [4, 5, 6, 7, 8]
    area_options = [area for area in area_options if area not in forbidden_areas]
    if not area_options:
        return None

    for _ in range(CHAMBER_RETRIES):
        target_area = random.choice(area_options)
        shape = _generate_shape(h, w, target_area)
        if shape is None:
            continue
        candidates = _occluder_candidates(shape)
        if not candidates:
            continue
        top_hidden = candidates[0]["hidden"]
        preferred = [cand for cand in candidates if cand["hidden"] == top_hidden]
        occ = random.choice(preferred)

        patch = np.zeros((h, w), dtype=int)
        patch[shape] = int(color)
        patch[occ["rect"]] = RED
        return {
            "patch": patch,
            "full_area": int(shape.sum()),
            "visible_area": int(occ["visible"]),
        }
    return None


def _build_lattice(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((h, w), GREY, dtype=int)
    chambers = []
    r = 1
    for rh in row_heights:
        c = 1
        for cw in col_widths:
            grid[r : r + rh, c : c + cw] = BLACK
            chambers.append((r, c, r + rh - 1, c + cw - 1))
            c += cw + 1
        r += rh + 1
    return grid, chambers


def _selected_indices(chambers: list[tuple[int, int, int, int]], count: int) -> list[int] | None:
    if count > len(chambers):
        return None
    even = [i for i, (_, c0, _, c1) in enumerate(chambers) if (c1 - c0 + 1) % 2 == 0]
    odd = [i for i, (_, c0, _, c1) in enumerate(chambers) if (c1 - c0 + 1) % 2 == 1]
    if not even or not odd:
        return None
    for _ in range(80):
        chosen = sorted(random.sample(range(len(chambers)), count))
        even_count = sum((chambers[i][3] - chambers[i][1] + 1) % 2 == 0 for i in chosen)
        odd_count = count - even_count
        if count == 3 and even_count >= 1 and odd_count >= 1:
            return chosen
        if count == 4 and (
            (even_count >= 2 and odd_count >= 2) or (even_count >= 1 and odd_count >= 3)
        ):
            return chosen
    return None


def _template_patch(h: int, w: int, area: int, color: int) -> tuple[np.ndarray, int]:
    shape, rect = random.choice(_TEMPLATES[(h, w, area)])
    shape = shape.copy()
    rect = rect.copy()
    if random.random() < 0.5:
        shape = np.flipud(shape)
        rect = np.flipud(rect)
    patch = np.zeros((h, w), dtype=int)
    patch[shape] = int(color)
    patch[rect] = RED
    visible_area = int((shape & ~rect).sum())
    return patch, visible_area


def _assign_areas(chosen: list[int], chambers: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    even = [idx for idx in chosen if (chambers[idx][3] - chambers[idx][1] + 1) % 2 == 0]
    odd = [idx for idx in chosen if (chambers[idx][3] - chambers[idx][1] + 1) % 2 == 1]
    random.shuffle(even)
    random.shuffle(odd)

    if len(chosen) == 3:
        if len(even) >= 2 and len(odd) >= 1:
            return [(even.pop(), 8), (odd.pop(), 7), (even.pop(), 6)]
        return [(even.pop(), 8), (odd.pop(), 7), (odd.pop(), 5)]

    if len(even) >= 2 and len(odd) >= 2:
        return [(even.pop(), 8), (odd.pop(), 7), (even.pop(), 6), (odd.pop(), 5)]
    return [(even.pop(), 8), (odd.pop(), 7), (odd.pop(), 5), (odd.pop(), 4)]


def _generate_attempt() -> np.ndarray | None:
    n_rows = random.choice([2, 3])
    n_cols = 3
    row_heights = [random.choice([4, 5]) for _ in range(n_rows)]
    while True:
        col_widths = [random.choice([4, 5]) for _ in range(n_cols)]
        if 4 in col_widths and 5 in col_widths:
            break

    grid, chambers = _build_lattice(row_heights, col_widths)
    pattern_count = random.choice([3, 4])
    chosen = _selected_indices(chambers, pattern_count)
    if chosen is None:
        return None

    assignments = _assign_areas(chosen, chambers)
    colors = random.sample(PALETTE, len(assignments))
    records = []
    for (chamber_idx, area), color in zip(assignments, colors):
        r0, c0, r1, c1 = chambers[chamber_idx]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        key = (h, w, area)
        if key not in _TEMPLATES:
            return None
        patch, visible_area = _template_patch(h, w, area, color)
        records.append((chamber_idx, area, visible_area, patch, int(color)))

    full_counts = [area for _, area, _, _, _ in records]
    visible_counts = [visible for _, _, visible, _, _ in records]
    if len(set(full_counts)) != len(full_counts):
        return None
    if int(np.argmax(full_counts)) == int(np.argmax(visible_counts)):
        return None

    for chamber_idx, _, _, patch, _ in records:
        r0, c0, r1, c1 = chambers[chamber_idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    try:
        out = main(grid)
    except Exception:
        return None

    winners = {i for i, val in enumerate(full_counts) if val == max(full_counts)}
    expected = np.full_like(grid, GREY)
    for r0, c0, r1, c1 in chambers:
        expected[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for local_idx, (chamber_idx, _, _, _, color) in enumerate(records):
        r0, c0, r1, c1 = chambers[chamber_idx]
        if local_idx in winners:
            expected[r0 : r1 + 1, c0 : c1 + 1] = color
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
