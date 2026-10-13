from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
RANK_COLORS = (ORANGE, PINK, YELLOW)

GENERATION_RETRIES = 500
GRID_ROWS = (15, 20)
GRID_COLS = (15, 20)
WINDOW_RANDOM_TRIALS = 700


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _fill_mask_from_component(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    axis_left = int(coords[:, 1].max())
    filled = np.zeros(shape, dtype=bool)
    # Match bf32578f: each row's rightmost half-outline pixel is the hidden inner boundary.
    for r in np.unique(coords[:, 0]):
        row_cols = coords[coords[:, 0] == r, 1]
        inner_left = int(row_cols.max())
        right_boundary = 2 * axis_left + 1 - inner_left
        if inner_left + 1 < right_boundary:
            filled[int(r), inner_left + 1 : right_boundary] = True
    return filled


def _shape_infos_from_grid(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for coords in _components(grid == RED):
        mask = _fill_mask_from_component(coords, grid.shape)
        area = int(mask.sum())
        if area == 0:
            continue
        bbox = _bbox_from_mask(mask)
        assert bbox is not None
        infos.append(
            {
                "mask": mask,
                "area": area,
                "bbox": bbox,
            }
        )
    return infos


def _shape_intersects_window(mask: np.ndarray, window: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = window
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _rank_shape_ids(shape_infos: list[dict], ids: Iterable[int]) -> list[int]:
    return sorted(
        list(ids),
        key=lambda idx: (-int(shape_infos[idx]["area"]), int(shape_infos[idx]["bbox"][0]), int(shape_infos[idx]["bbox"][1])),
    )


def _render_output(
    shape: tuple[int, int],
    shape_infos: list[dict],
    selected_ids: Iterable[int],
    blue_coords: Iterable[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for info in shape_infos:
        out[info["mask"]] = GREEN
    for rank, idx in enumerate(_rank_shape_ids(shape_infos, selected_ids)):
        if rank >= len(RANK_COLORS):
            break
        out[shape_infos[idx]["mask"]] = int(RANK_COLORS[rank])
    for r, c in blue_coords:
        out[int(r), int(c)] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each mirrored half-outline, fill its interior, and recolor only the window-selected filled shapes by area rank.

    Input structure:
    - The grid has a black background, 3-4 separated red half-outlines, and exactly 4 blue marker pixels.
    - Each red component is the left half of a closed shape whose vertical symmetry axis sits immediately to the right of that component's rightmost column.
    - The 4 blue markers uniquely realize the minimum row, maximum row, minimum column, and maximum column among blue pixels.

    Rule:
    - For every red component, treat its row-wise rightmost red pixel as the hidden inner boundary, mirror that boundary across the half-column axis, erase the red outline, and fill only the strict interior span in green.
    - Use the blue extremal markers to define a selection window: rows `min_row+1..max_row-1` and columns `min_col+1..max_col-1`.
    - Rank only the filled shapes that intersect that window by descending green area, breaking ties by the lexicographic `(top_row, left_col)` of the filled shape's bbox; recolor ranks 1, 2, 3 to orange, pink, yellow, while all other filled shapes stay green.

    Key insight:
    - The blue pixels do not draw a frame; they only define which reconstructed shapes are visible to the ranking step, so moving the window can change both the selected set and the winning color assignment.
    """
    g = np.asarray(grid, dtype=int)
    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    rows = [r for r, _ in blue_coords]
    cols = [c for _, c in blue_coords]
    window = (min(rows) + 1, min(cols) + 1, max(rows) - 1, max(cols) - 1)

    shape_infos = _shape_infos_from_grid(g)
    selected = [
        idx
        for idx, info in enumerate(shape_infos)
        if _shape_intersects_window(info["mask"], window)
    ]
    return _render_output(g.shape, shape_infos, selected, blue_coords)


def _make_half_shape() -> dict:
    for _ in range(200):
        h = random.randint(3, 6)
        half_w = random.randint(2, 4)
        axis = half_w - 1
        mid_rows = h - 2
        widths = [random.randint(1, axis)]
        for _ in range(mid_rows - 1):
            prev = widths[-1]
            choices = [cand for cand in (prev - 1, prev, prev + 1) if 1 <= cand <= axis]
            widths.append(random.choice(choices))
        if random.random() < 0.5:
            widths.reverse()
        area = 2 * sum(widths)
        if area < 4:
            continue

        inner_lefts = [axis] + [axis - width for width in widths] + [axis]
        red_mask = np.zeros((h, half_w), dtype=bool)
        fill_mask = np.zeros((h, 2 * axis + 1), dtype=bool)
        for r, inner_left in enumerate(inner_lefts):
            red_mask[r, : inner_left + 1] = True
            right_boundary = 2 * axis + 1 - inner_left
            if inner_left + 1 < right_boundary:
                fill_mask[r, inner_left + 1 : right_boundary] = True

        fill_bbox = _bbox_from_mask(fill_mask)
        assert fill_bbox is not None
        return {
            "red_mask": red_mask,
            "fill_mask": fill_mask,
            "h": h,
            "half_w": half_w,
            "full_w": fill_mask.shape[1],
            "area": int(fill_mask.sum()),
            "fill_bbox_local": fill_bbox,
        }
    raise RuntimeError("failed to build half-shape")


def _random_free_rect(occupied: np.ndarray, h: int, w: int, *, border: int = 1, padding: int = 1) -> tuple[int, int] | None:
    H, W = occupied.shape
    candidates: list[tuple[int, int]] = []
    r_stop = H - h - border
    c_stop = W - w - border
    if r_stop < border or c_stop < border:
        return None
    for r0 in range(border, r_stop + 1):
        for c0 in range(border, c_stop + 1):
            rr0 = max(0, r0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(W, c0 + w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_shapes(H: int, W: int, shapes: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((H, W), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(
        range(len(shapes)),
        key=lambda idx: shapes[idx]["h"] * shapes[idx]["full_w"],
        reverse=True,
    )
    for idx in order:
        shp = shapes[idx]
        loc = _random_free_rect(occupied, int(shp["h"]), int(shp["full_w"]), border=1, padding=1)
        if loc is None:
            return None
        r0, c0 = loc
        positions[idx] = (r0, c0)
        rr0 = max(0, r0 - 1)
        rr1 = min(H, r0 + int(shp["h"]) + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(W, c0 + int(shp["full_w"]) + 1)
        occupied[rr0:rr1, cc0:cc1] = True
    return [(int(r0), int(c0)) for r0, c0 in positions if r0 is not None and c0 is not None]


def _assemble_scene(H: int, W: int, shapes: list[dict], positions: list[tuple[int, int]]) -> tuple[np.ndarray, list[dict], np.ndarray]:
    grid = np.full((H, W), BACKGROUND, dtype=int)
    shape_infos: list[dict] = []
    union_fill = np.zeros((H, W), dtype=bool)
    for shp, (r0, c0) in zip(shapes, positions):
        h = int(shp["h"])
        half_w = int(shp["half_w"])
        full_w = int(shp["full_w"])

        red_patch = np.full((h, half_w), BACKGROUND, dtype=int)
        red_patch[shp["red_mask"]] = RED
        grid[r0 : r0 + h, c0 : c0 + half_w] = np.maximum(grid[r0 : r0 + h, c0 : c0 + half_w], red_patch)

        mask = np.zeros((H, W), dtype=bool)
        mask[r0 : r0 + h, c0 : c0 + full_w] = shp["fill_mask"]
        union_fill |= mask
        bbox = _bbox_from_mask(mask)
        assert bbox is not None
        shape_infos.append({"mask": mask, "area": int(mask.sum()), "bbox": bbox})

    return grid, shape_infos, union_fill


def _choose_markers(window: tuple[int, int, int, int], blocked: np.ndarray) -> list[tuple[int, int]] | None:
    H, W = blocked.shape
    r0, c0, r1, c1 = window
    if not (1 <= r0 <= r1 <= H - 2 and 1 <= c0 <= c1 <= W - 2):
        return None

    top = [(r0 - 1, c) for c in range(c0, c1 + 1) if not blocked[r0 - 1, c]]
    bottom = [(r1 + 1, c) for c in range(c0, c1 + 1) if not blocked[r1 + 1, c]]
    left = [(r, c0 - 1) for r in range(r0, r1 + 1) if not blocked[r, c0 - 1]]
    right = [(r, c1 + 1) for r in range(r0, r1 + 1) if not blocked[r, c1 + 1]]
    if not top or not bottom or not left or not right:
        return None

    return [
        random.choice(top),
        random.choice(bottom),
        random.choice(left),
        random.choice(right),
    ]


def _window_candidates(
    shape_infos: list[dict],
    blocked: np.ndarray,
    desired_selected: int,
) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int]], tuple[int, ...]]]:
    H, W = blocked.shape
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int]], tuple[int, ...]]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def consider(window: tuple[int, int, int, int]) -> None:
        if window in seen:
            return
        seen.add(window)
        selected = tuple(
            idx
            for idx, info in enumerate(shape_infos)
            if _shape_intersects_window(info["mask"], window)
        )
        if len(selected) != desired_selected or len(selected) >= len(shape_infos):
            return
        markers = _choose_markers(window, blocked)
        if markers is None:
            return
        candidates.append((window, markers, selected))

    for _ in range(WINDOW_RANDOM_TRIALS):
        r0 = random.randint(1, H - 2)
        r1 = random.randint(r0, H - 2)
        c0 = random.randint(1, W - 2)
        c1 = random.randint(c0, W - 2)
        consider((r0, c0, r1, c1))

    if candidates:
        return candidates

    for r0 in range(1, H - 1):
        for r1 in range(r0, H - 1):
            for c0 in range(1, W - 1):
                for c1 in range(c0, W - 1):
                    consider((r0, c0, r1, c1))
                    if len(candidates) >= 400:
                        return candidates
    return candidates


def _blue_window(blue_coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in blue_coords]
    cols = [c for _, c in blue_coords]
    return min(rows) + 1, min(cols) + 1, max(rows) - 1, max(cols) - 1


def _validate_scene(
    inp: np.ndarray,
    expected_window: tuple[int, int, int, int],
    expected_selected: tuple[int, ...],
) -> bool:
    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(inp == BLUE)]
    if len(blue_coords) != 4:
        return False

    rows = [r for r, _ in blue_coords]
    cols = [c for _, c in blue_coords]
    if sum(r == min(rows) for r in rows) != 1:
        return False
    if sum(r == max(rows) for r in rows) != 1:
        return False
    if sum(c == min(cols) for c in cols) != 1:
        return False
    if sum(c == max(cols) for c in cols) != 1:
        return False
    if _blue_window(blue_coords) != expected_window:
        return False

    red_comps = _components(inp == RED)
    if len(red_comps) < 3 or len(red_comps) > 4:
        return False

    shape_infos = _shape_infos_from_grid(inp)
    if len(shape_infos) != len(red_comps):
        return False
    selected = tuple(
        idx
        for idx, info in enumerate(shape_infos)
        if _shape_intersects_window(info["mask"], expected_window)
    )
    return selected == expected_selected and len(selected) >= 2


def _all_shapes_selected_output(shape_infos: list[dict], blue_coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    return _render_output(shape, shape_infos, range(len(shape_infos)), blue_coords)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        H = random.randint(*GRID_ROWS)
        W = random.randint(*GRID_COLS)
        n_shapes = random.choice((3, 4))
        desired_selected = 2 if n_shapes == 3 else random.choice((2, 3))

        shapes = [_make_half_shape() for _ in range(n_shapes)]
        positions = _place_shapes(H, W, shapes)
        if positions is None:
            continue

        grid, shape_infos, union_fill = _assemble_scene(H, W, shapes, positions)
        blocked = (grid != BACKGROUND) | union_fill
        candidates = _window_candidates(shape_infos, blocked, desired_selected)
        if not candidates:
            continue

        prefer_tie = desired_selected == 3 and random.random() < 0.4
        candidates.sort(
            key=lambda item: (
                0
                if (len({shape_infos[idx]["area"] for idx in item[2]}) < len(item[2])) == prefer_tie
                else 1,
                random.random(),
            )
        )

        for window, markers, selected in candidates:
            inp = grid.copy()
            for r, c in markers:
                inp[r, c] = BLUE

            expected = _render_output(inp.shape, shape_infos, selected, markers)
            if np.array_equal(expected, _all_shapes_selected_output(shape_infos, markers, inp.shape)):
                continue
            if not np.array_equal(main(inp), expected):
                continue
            if not _validate_scene(inp, window, selected):
                continue
            return inp

    raise RuntimeError("failed to generate valid input")
