from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIDE_RANGE = (15, 20)
MAX_GENERATION_ATTEMPTS = 800
MAX_SHAPE_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mirror_vertical(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)


def _mirror_horizontal(mask: np.ndarray) -> np.ndarray:
    return np.flipud(mask)


def _axis_stats(mask: np.ndarray, axis: str) -> dict:
    h, w = mask.shape
    overlap = 0
    if axis == "vertical":
        side_counts = {"left": 0, "right": 0}
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            mc = w - 1 - cc
            if mask[rr, mc]:
                overlap += 1
            elif mc < cc:
                side_counts["left"] += 1
            elif mc > cc:
                side_counts["right"] += 1
    else:
        side_counts = {"top": 0, "bottom": 0}
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            mr = h - 1 - rr
            if mask[mr, cc]:
                overlap += 1
            elif mr < rr:
                side_counts["top"] += 1
            elif mr > rr:
                side_counts["bottom"] += 1

    mismatch = int(sum(side_counts.values()))
    imbalance = abs(max(side_counts.values()) - min(side_counts.values()))
    return {
        "axis": axis,
        "overlap": int(overlap),
        "mismatch": mismatch,
        "imbalance": imbalance,
        "side_counts": side_counts,
    }


def _analyze_sprite(mask: np.ndarray) -> dict:
    vertical = _axis_stats(mask, "vertical")
    horizontal = _axis_stats(mask, "horizontal")
    ranked = sorted(
        [vertical, horizontal],
        key=lambda item: (item["overlap"], -item["mismatch"], item["imbalance"]),
        reverse=True,
    )
    best = ranked[0]
    other = ranked[1]
    defect_side = max(best["side_counts"], key=lambda side: (best["side_counts"][side], side))
    return {
        "axis": best["axis"],
        "defect_side": defect_side,
        "scores": {
            "vertical": vertical["overlap"],
            "horizontal": horizontal["overlap"],
        },
        "side_counts": best["side_counts"],
        "mismatch": best["mismatch"],
        "margin": int(best["overlap"] - other["overlap"]),
    }


def _complete_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "vertical":
        return mask | _mirror_vertical(mask)
    return mask | _mirror_horizontal(mask)


def _moved_top_left(
    top_left: tuple[int, int],
    size: tuple[int, int],
    defect_side: str,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0 = top_left
    h, w = size
    grid_h, grid_w = grid_shape
    if defect_side == "left":
        return r0, grid_w - w
    if defect_side == "right":
        return r0, 0
    if defect_side == "top":
        return grid_h - h, c0
    return 0, c0


def _render_mask(out: np.ndarray, mask: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    h, w = mask.shape
    dst = out[r0 : r0 + h, c0 : c0 + w]
    assert np.all(dst[mask] == out[0, 0]) or np.all(dst[mask] == BG)
    dst[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each nearly symmetric colored object away from its defective side and then complete it by reflection.

    Input structure:
    - The grid has a black background with 3-4 separated monochrome foreground objects.
    - Each foreground color appears in exactly one 4-connected object.
    - Every object is almost symmetric inside its own bounding box about either a vertical or a horizontal axis, with 1-2 missing cells on one side.

    Rule:
    - Process foreground colors independently, ordered by the top-left corner of their bounding boxes.
    - For each color, choose the better of vertical or horizontal bbox-centered reflection by the larger mirrored-pixel overlap, then identify the defect side as the side missing mirrored counterparts.
    - Slide the whole object in the direction opposite the defect until its intact side reaches the corresponding grid edge, then fill the missing reflected cells at the new location.

    Key insight:
    - The same asymmetry signal serves twice: it tells you which way the object must move, and it also tells you which reflected cells must be restored after the move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    objects = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        bbox = _bbox_coords(coords)
        objects.append((bbox[0], bbox[1], int(color), bbox))
    objects.sort()

    out = np.full_like(g, bg)
    for _r, _c, color, bbox in objects:
        r0, c0, r1, c1 = bbox
        sprite = g[r0 : r1 + 1, c0 : c1 + 1] == color
        info = _analyze_sprite(sprite)
        completed = _complete_mask(sprite, info["axis"])
        dst_top_left = _moved_top_left((r0, c0), sprite.shape, info["defect_side"], g.shape)

        dst = out[
            dst_top_left[0] : dst_top_left[0] + sprite.shape[0],
            dst_top_left[1] : dst_top_left[1] + sprite.shape[1],
        ]
        assert np.all(dst[completed] == bg)
        dst[completed] = color

    return out


def _random_shortest_path(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        cells.add((r, c))
    return cells


def _mirror_half_to_vertical(half: np.ndarray, full_w: int) -> np.ndarray:
    h, _ = half.shape
    full = np.zeros((h, full_w), dtype=bool)
    for r, c in np.argwhere(half):
        rr = int(r)
        cc = int(c)
        full[rr, cc] = True
        full[rr, full_w - 1 - cc] = True
    return full


def _build_vertical_base() -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        half_w = (w + 1) // 2

        top = (0, random.randint(0, half_w - 1))
        bottom = (h - 1, random.randint(0, half_w - 1))
        left = (random.randint(0, h - 1), 0)
        axis = (random.randint(0, h - 1), half_w - 1)

        order = [top] + random.sample([left, axis], 2) + [bottom]
        cells: set[tuple[int, int]] = set()
        for a, b in zip(order, order[1:]):
            cells.update(_random_shortest_path(a, b))

        extra_steps = random.randint(0, 2)
        for _extra in range(extra_steps):
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < half_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        half = np.zeros((h, half_w), dtype=bool)
        for r, c in cells:
            half[r, c] = True
        full = _mirror_half_to_vertical(half, w)
        if not (5 <= int(full.sum()) <= 10):
            continue
        if not _is_connected(full):
            continue
        if not _touches_all_bbox_sides(full):
            continue
        if np.array_equal(full, _mirror_horizontal(full)):
            continue
        info = _analyze_sprite(full)
        if info["axis"] != "vertical":
            continue
        if info["scores"]["vertical"] <= info["scores"]["horizontal"]:
            continue
        return full
    return None


def _build_complete_mask(axis: str) -> np.ndarray | None:
    if axis == "vertical":
        return _build_vertical_base()
    for _ in range(MAX_SHAPE_ATTEMPTS):
        base = _build_vertical_base()
        if base is None:
            continue
        full = base.T
        info = _analyze_sprite(full)
        if info["axis"] != "horizontal":
            continue
        if info["scores"]["horizontal"] <= info["scores"]["vertical"]:
            continue
        return full
    return None


def _candidate_defect_cells(mask: np.ndarray, defect_side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    if defect_side in {"left", "right"}:
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            mc = w - 1 - cc
            if not mask[rr, mc]:
                continue
            if defect_side == "left" and cc < mc:
                cells.append((rr, cc))
            if defect_side == "right" and cc > mc:
                cells.append((rr, cc))
    else:
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            mr = h - 1 - rr
            if not mask[mr, cc]:
                continue
            if defect_side == "top" and rr < mr:
                cells.append((rr, cc))
            if defect_side == "bottom" and rr > mr:
                cells.append((rr, cc))
    return cells


def _make_incomplete_mask(defect_side: str, missing_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    axis = "vertical" if defect_side in {"left", "right"} else "horizontal"
    for _ in range(MAX_SHAPE_ATTEMPTS):
        full = _build_complete_mask(axis)
        if full is None:
            continue
        candidates = _candidate_defect_cells(full, defect_side)
        random.shuffle(candidates)
        if len(candidates) < missing_count:
            continue

        for removed in itertools.combinations(candidates, missing_count):
            incomplete = full.copy()
            for r, c in removed:
                incomplete[r, c] = False
            if not (4 <= int(incomplete.sum()) <= 8):
                continue
            if not _is_connected(incomplete):
                continue
            if incomplete.shape != _crop_mask(incomplete).shape:
                continue
            info = _analyze_sprite(incomplete)
            expected_axis = axis
            if info["axis"] != expected_axis:
                continue
            if info["margin"] <= 0:
                continue
            if info["defect_side"] != defect_side:
                continue
            if info["side_counts"][defect_side] != missing_count:
                continue
            other_sides = [side for side in info["side_counts"] if side != defect_side]
            if any(info["side_counts"][side] != 0 for side in other_sides):
                continue
            if not np.array_equal(_complete_mask(incomplete, axis), full):
                continue
            return full, incomplete
    return None


def _choose_sides(n_objects: int) -> list[str]:
    lr = random.choice(["left", "right"])
    tb = random.choice(["top", "bottom"])
    remaining = [side for side in ["left", "right", "top", "bottom"] if side not in {lr, tb}]
    sides = [lr, tb]
    if n_objects == 3:
        sides.append(random.choice(remaining))
    else:
        sides.extend(remaining)
    random.shuffle(sides)
    return sides


def _cells_from_mask(mask: np.ndarray, top_left: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r0, c0 = top_left
    return frozenset((r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask))


def _chebyshev_halo(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    halo = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((r + dr, c + dc))
    return frozenset(halo)


def _all_candidates(spec: dict, grid_shape: tuple[int, int]) -> list[dict]:
    grid_h, grid_w = grid_shape
    mask = spec["input_mask"]
    h, w = mask.shape
    side = spec["defect_side"]
    candidates = []
    for r0 in range(grid_h - h + 1):
        if side == "top" and r0 > grid_h - h - 2:
            continue
        if side == "bottom" and r0 < 2:
            continue
        for c0 in range(grid_w - w + 1):
            if side == "left" and c0 > grid_w - w - 2:
                continue
            if side == "right" and c0 < 2:
                continue
            in_cells = _cells_from_mask(mask, (r0, c0))
            out_top_left = _moved_top_left((r0, c0), mask.shape, side, grid_shape)
            out_cells = _cells_from_mask(spec["full_mask"], out_top_left)
            candidates.append(
                {
                    "input_top_left": (r0, c0),
                    "output_top_left": out_top_left,
                    "in_cells": in_cells,
                    "in_halo": _chebyshev_halo(in_cells),
                    "out_cells": out_cells,
                    "out_halo": _chebyshev_halo(out_cells),
                }
            )
    random.shuffle(candidates)
    return candidates


def _search_layout(specs: list[dict], grid_shape: tuple[int, int]) -> list[dict] | None:
    candidate_lists = [_all_candidates(spec, grid_shape) for spec in specs]
    order = sorted(range(len(specs)), key=lambda idx: len(candidate_lists[idx]))
    placed: dict[int, dict] = {}

    def dfs(
        pos: int,
        input_halo: frozenset[tuple[int, int]],
        output_halo: frozenset[tuple[int, int]],
    ) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for candidate in candidate_lists[idx]:
            if not candidate["in_cells"].isdisjoint(input_halo):
                continue
            if not candidate["out_cells"].isdisjoint(output_halo):
                continue
            placed[idx] = candidate
            if dfs(
                pos + 1,
                frozenset(set(input_halo) | set(candidate["in_halo"])),
                frozenset(set(output_halo) | set(candidate["out_halo"])),
            ):
                return True
            del placed[idx]
        return False

    if not dfs(0, frozenset(), frozenset()):
        return None
    return [placed[i] for i in range(len(specs))]


def _render_input(specs: list[dict], placements: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for spec, placement in zip(specs, placements):
        r0, c0 = placement["input_top_left"]
        h, w = spec["input_mask"].shape
        view = grid[r0 : r0 + h, c0 : c0 + w]
        view[spec["input_mask"]] = spec["color"]
    return grid


def _render_expected_output(specs: list[dict], placements: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec, placement in zip(specs, placements):
        r0, c0 = placement["output_top_left"]
        h, w = spec["full_mask"].shape
        view = out[r0 : r0 + h, c0 : c0 + w]
        view[spec["full_mask"]] = spec["color"]
    return out


def _render_slide_only(specs: list[dict], placements: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec, placement in zip(specs, placements):
        r0, c0 = placement["output_top_left"]
        h, w = spec["input_mask"].shape
        view = out[r0 : r0 + h, c0 : c0 + w]
        view[spec["input_mask"]] = spec["color"]
    return out


def _render_complete_in_place(specs: list[dict], placements: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec, placement in zip(specs, placements):
        r0, c0 = placement["input_top_left"]
        h, w = spec["full_mask"].shape
        view = out[r0 : r0 + h, c0 : c0 + w]
        view[spec["full_mask"]] = spec["color"]
    return out


def _validate_case(grid: np.ndarray, specs: list[dict], placements: list[dict]) -> bool:
    expected = _render_expected_output(specs, placements, grid.shape)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(_render_slide_only(specs, placements, grid.shape), expected):
        return False
    if np.array_equal(_render_complete_in_place(specs, placements, grid.shape), expected):
        return False

    for spec, placement in zip(specs, placements):
        color = spec["color"]
        coords = np.argwhere(grid == color)
        if len(coords) != int(spec["input_mask"].sum()):
            return False
        bbox = _bbox_coords(coords)
        sprite = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == color
        info = _analyze_sprite(sprite)
        if info["axis"] != spec["axis"]:
            return False
        if info["defect_side"] != spec["defect_side"]:
            return False
        if info["margin"] <= 0:
            return False
        if info["side_counts"][spec["defect_side"]] != spec["missing_count"]:
            return False
        other_sides = [side for side in info["side_counts"] if side != spec["defect_side"]]
        if any(info["side_counts"][side] != 0 for side in other_sides):
            return False

        move_distance = abs(placement["output_top_left"][0] - placement["input_top_left"][0]) + abs(
            placement["output_top_left"][1] - placement["input_top_left"][1]
        )
        if move_distance < 2:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid_h = random.randint(*GRID_SIDE_RANGE)
        grid_w = random.randint(*GRID_SIDE_RANGE)
        n_objects = 4 if random.random() < 0.65 else 3
        sides = _choose_sides(n_objects)
        colors = random.sample(COLOR_POOL, n_objects)

        specs = []
        ok = True
        for color, side in zip(colors, sides):
            missing_count = random.choice([1, 2])
            built = _make_incomplete_mask(side, missing_count)
            if built is None:
                ok = False
                break
            full_mask, input_mask = built
            specs.append(
                {
                    "color": color,
                    "axis": "vertical" if side in {"left", "right"} else "horizontal",
                    "defect_side": side,
                    "missing_count": missing_count,
                    "full_mask": full_mask,
                    "input_mask": input_mask,
                }
            )
        if not ok:
            continue

        placements = _search_layout(specs, (grid_h, grid_w))
        if placements is None:
            continue

        grid = _render_input(specs, placements, (grid_h, grid_w))
        if _background_color(grid) != BG:
            continue
        if not _validate_case(grid, specs, placements):
            continue
        return grid

    raise RuntimeError("failed to generate a valid asymmetric-slide puzzle instance")
