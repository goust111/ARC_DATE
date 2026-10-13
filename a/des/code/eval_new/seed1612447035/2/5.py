import random
from collections import deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
PROJECTOR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_wall(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]

    if len(row_candidates) == 1 and not col_candidates:
        return ("horizontal", int(row_candidates[0][0]), int(row_candidates[0][1]))
    if len(col_candidates) == 1 and not row_candidates:
        return ("vertical", int(col_candidates[0][0]), int(col_candidates[0][1]))
    raise AssertionError("expected exactly one full-length wall")


def _extract_components(grid: np.ndarray, forbidden: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            rows = sorted({rr for rr, _ in coords})
            cols = sorted({cc for _, cc in coords})
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "rows": rows,
                    "cols": cols,
                }
            )

    return components


def _wall_overlays(grid: np.ndarray) -> tuple[str, int, int, list[dict[str, object]]]:
    bg = _most_common_color(grid)
    orientation, wall_index, wall_color = _find_wall(grid, bg)
    components = _extract_components(grid, forbidden={bg, wall_color})
    assert components, "expected at least one projector"

    overlays: list[dict[str, object]] = []
    if orientation == "vertical":
        left_side = all(max(comp["cols"]) < wall_index for comp in components)
        right_side = all(min(comp["cols"]) > wall_index for comp in components)
        assert left_side != right_side, "projectors must all lie on one side of the wall"
        for comp in components:
            cols = comp["cols"]
            if left_side:
                distance = wall_index - max(cols) - 1
            else:
                distance = min(cols) - wall_index - 1
            assert distance >= 0
            overlays.append(
                {
                    "color": int(comp["color"]),
                    "distance": int(distance),
                    "cover": set(int(r) for r in comp["rows"]),
                }
            )
    else:
        above_side = all(max(comp["rows"]) < wall_index for comp in components)
        below_side = all(min(comp["rows"]) > wall_index for comp in components)
        assert above_side != below_side, "projectors must all lie on one side of the wall"
        for comp in components:
            rows = comp["rows"]
            if above_side:
                distance = wall_index - max(rows) - 1
            else:
                distance = min(rows) - wall_index - 1
            assert distance >= 0
            overlays.append(
                {
                    "color": int(comp["color"]),
                    "distance": int(distance),
                    "cover": set(int(c) for c in comp["cols"]),
                }
            )

    return orientation, int(wall_index), int(wall_color), overlays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the wall with the final visible colors of overlapping perpendicular projector shadows.

    Input structure:
    - The background is the dominant color.
    - A single grey row or column spans the full grid and acts as the wall.
    - Several colored projector blocks sit entirely on one side of that wall, while the opposite side is empty background.

    Rule:
    - Detect whether the wall is horizontal or vertical, then project every block straight toward the wall.
    - A block covers the wall cells aligned with its rows when the wall is vertical, or aligned with its columns when the wall is horizontal.
    - For each covered wall cell, use the color of the closest covering block; if the closest distance is tied, keep the original grey wall color. Uncovered wall cells also stay grey.

    Key insight:
    - The wall color is determined independently at each aligned wall position by comparing projector-to-wall distances, so overlap resolution depends on geometric depth rather than draw order or color id.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orientation, wall_index, wall_color, overlays = _wall_overlays(g)
    out = g.copy()

    if orientation == "vertical":
        for r in range(g.shape[0]):
            cover = [item for item in overlays if r in item["cover"]]
            if not cover:
                continue
            best = min(int(item["distance"]) for item in cover)
            winners = [item for item in cover if int(item["distance"]) == best]
            out[r, wall_index] = int(winners[0]["color"]) if len(winners) == 1 else wall_color
    else:
        for c in range(g.shape[1]):
            cover = [item for item in overlays if c in item["cover"]]
            if not cover:
                continue
            best = min(int(item["distance"]) for item in cover)
            winners = [item for item in cover if int(item["distance"]) == best]
            out[wall_index, c] = int(winners[0]["color"]) if len(winners) == 1 else wall_color

    return out


def _bboxes_overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    return not (
        int(a["r1"]) < int(b["r0"])
        or int(b["r1"]) < int(a["r0"])
        or int(a["c1"]) < int(b["c0"])
        or int(b["c1"]) < int(a["c0"])
    )


def _render_canonical_scene(
    rows: int,
    cols: int,
    wall_col: int,
    projectors: list[dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), BG_COLOR, dtype=int)
    inp[:, wall_col] = WALL_COLOR
    for projector in projectors:
        inp[
            int(projector["r0"]) : int(projector["r1"]) + 1,
            int(projector["c0"]) : int(projector["c1"]) + 1,
        ] = int(projector["color"])

    out = inp.copy()
    for r in range(rows):
        cover = [p for p in projectors if int(p["r0"]) <= r <= int(p["r1"])]
        if not cover:
            continue
        best = min(int(p["distance"]) for p in cover)
        winners = [p for p in cover if int(p["distance"]) == best]
        out[r, wall_col] = int(winners[0]["color"]) if len(winners) == 1 else WALL_COLOR

    return inp, out


def _transform_grid(grid: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return grid.copy()
    if mode == 1:
        return np.fliplr(grid)
    if mode == 2:
        return np.flipud(grid)
    if mode == 3:
        return np.flipud(np.fliplr(grid))
    if mode == 4:
        return grid.T.copy()
    if mode == 5:
        return np.fliplr(grid.T)
    if mode == 6:
        return np.flipud(grid.T)
    return np.flipud(np.fliplr(grid.T))


def _gen_attempt() -> np.ndarray | None:
    projector_count = random.randint(3, 5)
    extra_count = projector_count - 3
    colors = random.sample(PROJECTOR_COLORS, projector_count)

    near_h = random.randint(2, 3)
    far_h = random.randint(2, 3)
    single_h = random.randint(1, 3)
    extra_heights = [random.randint(1, 3) for _ in range(extra_count)]

    near_w = random.randint(1, 3)
    far_w = random.randint(1, 3)
    single_w = random.randint(1, 3)
    extra_widths = [random.randint(1, 3) for _ in range(extra_count)]

    overlap = random.randint(1, min(near_h, far_h) - 1)
    near_dist = random.randint(0, 2)
    far_dist = near_dist + near_w + random.randint(0, 2)
    single_dist = random.randint(0, far_dist + 1)
    extra_dists = [random.randint(0, far_dist + 2) for _ in range(extra_count)]

    top_margin = random.randint(1, 4)
    between_pair_and_single = random.randint(1, 4)
    between_extras = [random.randint(1, 3) for _ in range(extra_count)]
    bottom_margin = random.randint(1, 4)

    if random.random() < 0.5:
        far_r0 = top_margin
        near_r0 = far_r0 + far_h - overlap
    else:
        near_r0 = top_margin
        far_r0 = near_r0 + near_h - overlap

    far_r1 = far_r0 + far_h - 1
    near_r1 = near_r0 + near_h - 1
    row_cursor = max(far_r1, near_r1) + 1 + between_pair_and_single

    single_r0 = row_cursor
    single_r1 = single_r0 + single_h - 1
    row_cursor = single_r1 + 1

    extra_rows: list[tuple[int, int]] = []
    for gap, height in zip(between_extras, extra_heights):
        row_cursor += gap
        r0 = row_cursor
        r1 = r0 + height - 1
        extra_rows.append((r0, r1))
        row_cursor = r1 + 1

    rows = row_cursor + bottom_margin
    if rows > 30:
        return None

    max_left_span = max(
        far_dist + far_w,
        near_dist + near_w,
        single_dist + single_w,
        *[dist + width for dist, width in zip(extra_dists, extra_widths)],
    )
    wall_col = max_left_span + random.randint(1, 5)
    cols = wall_col + 1 + random.randint(2, 6)
    if cols > 30:
        return None

    projectors: list[dict[str, int]] = []
    raw_specs = [
        ("far", far_r0, far_r1, far_w, far_dist, colors[0]),
        ("near", near_r0, near_r1, near_w, near_dist, colors[1]),
        ("single", single_r0, single_r1, single_w, single_dist, colors[2]),
    ]
    for idx, (r0, r1) in enumerate(extra_rows):
        raw_specs.append((f"extra{idx}", r0, r1, extra_widths[idx], extra_dists[idx], colors[3 + idx]))

    for _name, r0, r1, width, distance, color in raw_specs:
        c1 = wall_col - distance - 1
        c0 = c1 - width + 1
        if c0 < 0:
            return None
        projectors.append(
            {
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "distance": int(distance),
                "color": int(color),
            }
        )

    for i in range(len(projectors)):
        for j in range(i + 1, len(projectors)):
            if _bboxes_overlap(projectors[i], projectors[j]):
                return None

    inp, expected = _render_canonical_scene(rows, cols, wall_col, projectors)
    mode = random.randint(0, 7)
    inp = _transform_grid(inp, mode)
    expected = _transform_grid(expected, mode)

    try:
        pred = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(pred, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
