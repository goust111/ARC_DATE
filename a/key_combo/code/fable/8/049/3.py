from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in DIR4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _paint(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def _scene(grid: np.ndarray) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _get_bg(g)
    result: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        objs = []
        for coords in _components(g == color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            objs.append(
                {
                    "coords": coords,
                    "bbox": box,
                    "area": int(len(coords)),
                    "touches_bottom": bool(np.any(coords[:, 0] == h - 1)),
                    "center_r": float(coords[:, 0].mean()),
                    "center_c": float(coords[:, 1].mean()),
                    "bar_top": int(r0),
                    "bar_col": float((c0 + c1) / 2.0),
                }
            )
        result[color] = {"components": objs}
    return result


def _split_roles(color_scene: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    comps = list(color_scene["components"])
    bars = [comp for comp in comps if comp["touches_bottom"]]
    assert len(bars) == 1, "each color must have exactly one bottom bar"
    bar = bars[0]
    field = [comp for comp in comps if comp is not bar]
    rank = _bbox_h(bar["bbox"])
    assert field, "each bar color must also appear in upper objects"
    areas = [int(obj["area"]) for obj in field]
    assert len(areas) == len(set(areas)), "same-color object areas must be distinct"
    assert 1 <= rank <= len(field), "bar height must be a valid rank"
    bar["rank"] = rank
    return bar, field


def _pick_by_strategy(
    field: list[dict[str, object]],
    bar: dict[str, object],
    strategy: str,
) -> dict[str, object]:
    rank = int(bar["rank"])
    if strategy == "area_rank":
        ordered = sorted(field, key=lambda obj: (-int(obj["area"]), obj["bbox"][0], obj["bbox"][1]))
        return ordered[rank - 1]
    if strategy == "largest_only":
        return max(field, key=lambda obj: (int(obj["area"]), -obj["bbox"][0], -obj["bbox"][1]))
    if strategy == "smallest_only":
        return min(field, key=lambda obj: (int(obj["area"]), obj["bbox"][0], obj["bbox"][1]))
    if strategy == "bbox_height_rank":
        ordered = sorted(
            field,
            key=lambda obj: (-_bbox_h(obj["bbox"]), -int(obj["area"]), obj["bbox"][0], obj["bbox"][1]),
        )
        return ordered[rank - 1]
    if strategy == "bbox_area_rank":
        ordered = sorted(
            field,
            key=lambda obj: (
                -_bbox_h(obj["bbox"]) * _bbox_w(obj["bbox"]),
                -int(obj["area"]),
                obj["bbox"][0],
                obj["bbox"][1],
            ),
        )
        return ordered[rank - 1]
    if strategy == "top_rank":
        ordered = sorted(field, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], -int(obj["area"])))
        return ordered[rank - 1]
    if strategy == "left_rank":
        ordered = sorted(field, key=lambda obj: (obj["bbox"][1], obj["bbox"][0], -int(obj["area"])))
        return ordered[rank - 1]
    if strategy == "nearest_bar_rank":
        ordered = sorted(
            field,
            key=lambda obj: (
                abs(float(obj["center_c"]) - float(bar["bar_col"])),
                abs(float(obj["center_r"]) - float(bar["bar_top"])),
                obj["bbox"][0],
                obj["bbox"][1],
            ),
        )
        return ordered[rank - 1]
    raise ValueError(f"unknown strategy: {strategy}")


def _solve_with_strategy(grid: np.ndarray, strategy: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)
    for color, color_scene in _scene(g).items():
        bar, field = _split_roles(color_scene)
        chosen = _pick_by_strategy(field, bar, strategy)
        _paint(out, chosen["coords"], color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the same-color object whose area rank is named by the bottom bar height.

    Input structure:
    - The background is black.
    - Along the bottom edge sit several disconnected monochrome vertical bars, one per foreground color.
    - Above the bars, each bar color has several disconnected objects, and objects of the same color have pairwise distinct pixel counts.

    Rule:
    - For each foreground color, identify its unique bottom-touching bar and read the bar height as a 1-based rank.
    - Sort that color's upper objects by pixel count from largest to smallest, keep only the object at the bar's rank, and erase the others.
    - Erase every bar as well; the output keeps the selected objects in place on the original canvas.

    Key insight:
    - The histogram is not measuring object size directly: each bar height is an ordinal index that must be applied after ranking same-color objects by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_with_strategy(g, "area_rank")


def _random_connected_patch(area: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        h = max(rows) + 1
        w = max(cols) + 1
        if not (2 <= h <= 5 and 2 <= w <= 5):
            continue
        patch = np.zeros((h, w), dtype=int)
        for r, c in norm:
            patch[r, c] = 1
        return patch
    raise RuntimeError(f"failed to build connected patch for area {area}")


def _sample_counts_and_ranks(colors: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    while True:
        counts = {color: random.randint(2, 4) for color in colors}
        if max(counts.values()) >= 3 and sum(counts.values()) <= 12:
            break

    middle_color = random.choice([color for color in colors if counts[color] >= 3])
    remaining = [color for color in colors if color != middle_color]
    largest_color, smallest_color = random.sample(remaining, 2)

    ranks: dict[int, int] = {}
    for color in colors:
        count = counts[color]
        if color == largest_color:
            ranks[color] = 1
        elif color == smallest_color:
            ranks[color] = count
        elif color == middle_color:
            ranks[color] = random.randint(2, count - 1)
        else:
            ranks[color] = random.randint(1, count)
    return counts, ranks


def _sample_bar_columns(width: int, n_colors: int) -> list[int]:
    for _ in range(200):
        cols = sorted(random.sample(range(1, width - 1), n_colors))
        if all(b - a >= 3 for a, b in zip(cols, cols[1:])):
            return cols
    raise RuntimeError("failed to sample separated bar columns")


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    color: int,
    field_bottom: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    h_lim, w_lim = grid.shape
    for _ in range(400):
        r0 = random.randint(1, field_bottom - h + 1)
        c0 = random.randint(1, w_lim - w - 1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h_lim, r0 + h + 1)
        cc1 = min(w_lim, c0 + w + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        grid[r0 : r0 + h, c0 : c0 + w][patch == 1] = color
        blocked[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _expected_from_specs(shape: tuple[int, int], placed_specs: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in placed_specs:
        if spec["keep"]:
            _paint(out, spec["coords"], int(spec["color"]))
    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = random.randint(3, 4)
    colors = random.sample(PALETTE, n_colors)
    counts, ranks = _sample_counts_and_ranks(colors)
    max_rank = max(ranks.values())

    height = random.randint(max(18, max_rank + 12), 28)
    width = random.randint(20, 30)
    field_bottom = height - max_rank - 4
    if field_bottom < 8:
        return None

    bar_columns = _sample_bar_columns(width, n_colors)
    color_to_bar_col = dict(zip(random.sample(colors, len(colors)), bar_columns))

    object_specs: list[dict[str, object]] = []
    for color in colors:
        count = counts[color]
        rank = ranks[color]
        areas = random.sample(range(4, 12), count)
        target_area = sorted(areas, reverse=True)[rank - 1]
        for area in areas:
            object_specs.append(
                {
                    "color": color,
                    "area": area,
                    "patch": _random_connected_patch(area),
                    "keep": area == target_area,
                    "bar_rank": rank,
                    "bar_col": color_to_bar_col[color],
                }
            )

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    placed_specs: list[dict[str, object]] = []
    order = sorted(
        object_specs,
        key=lambda spec: (-(spec["patch"].shape[0] * spec["patch"].shape[1]), -int(spec["area"])),
    )
    for spec in order:
        patch = spec["patch"]
        placed = _place_patch(grid, blocked, patch, int(spec["color"]), field_bottom)
        if placed is None:
            return None
        r0, c0 = placed
        coords = np.argwhere(patch == 1) + np.array([r0, c0])
        placed_specs.append(
            {
                "color": int(spec["color"]),
                "area": int(spec["area"]),
                "coords": coords,
                "bbox": _bbox(coords),
                "keep": bool(spec["keep"]),
                "bar_rank": int(spec["bar_rank"]),
                "bar_col": int(spec["bar_col"]),
            }
        )

    for color in colors:
        rank = ranks[color]
        col = color_to_bar_col[color]
        grid[height - rank : height, col] = color

    expected = _expected_from_specs(grid.shape, placed_specs)
    return grid, expected


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    scene = _scene(grid)
    if len(scene) < 3:
        return False

    ranks = []
    middle_present = False
    for color_scene in scene.values():
        bar, field = _split_roles(color_scene)
        rank = int(bar["rank"])
        ranks.append(rank)
        if 1 < rank < len(field):
            middle_present = True
        if len(field) < 2:
            return False
        if any(comp["bbox"][2] >= grid.shape[0] - 1 for comp in field):
            return False

    if 1 not in ranks or not middle_present:
        return False
    if not any(int(bar["rank"]) == len(_split_roles(color_scene)[1]) for color_scene in scene.values() for bar in [_split_roles(color_scene)[0]]):
        return False

    wrong_strategies = [
        "largest_only",
        "smallest_only",
        "bbox_height_rank",
        "bbox_area_rank",
        "top_rank",
        "left_rank",
        "nearest_bar_rank",
    ]
    for strategy in wrong_strategies:
        if np.array_equal(_solve_with_strategy(grid, strategy), expected):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _build_attempt()
        if built is None:
            continue
        grid, expected = built
        if _semantic_checks(grid, expected):
            return grid
    raise RuntimeError("failed to generate a non-degenerate ranking-histogram sample")
