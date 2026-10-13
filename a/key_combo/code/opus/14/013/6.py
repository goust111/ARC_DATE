from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SPRITE_COLORS = (1, 2, 4)
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _parse_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    sprites: list[dict] = []
    bars: list[dict] = []

    for comp in _components_for_color(g, GREY):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if comp["area"] != height * width:
            raise AssertionError("grey component must be a solid line segment")
        if height == 1 and width >= 5:
            comp["orientation"] = "h"
            comp["axis"] = r0
        elif width == 1 and height >= 5:
            comp["orientation"] = "v"
            comp["axis"] = c0
        else:
            raise AssertionError("grey component must be a 1-cell-thick control bar")
        bars.append(comp)

    for color in np.unique(g):
        color = int(color)
        if color in (BG, GREY):
            continue
        for comp in _components_for_color(g, color):
            r0, c0, r1, c1 = comp["bbox"]
            if r0 == r1 or c0 == c1:
                raise AssertionError("sprites must not be degenerate line segments")
            sprites.append(comp)

    return sprites, bars


def _mirror_coords(sprite: dict, bar: dict) -> list[tuple[int, int]]:
    mirrored: list[tuple[int, int]] = []
    if bar["orientation"] == "h":
        axis = int(bar["axis"])
        for r, c in sprite["coords"]:
            mirrored.append((2 * axis - int(r), int(c)))
    else:
        axis = int(bar["axis"])
        for r, c in sprite["coords"]:
            mirrored.append((int(r), 2 * axis - int(c)))
    return mirrored


def _match_jobs(grid: np.ndarray) -> list[dict]:
    sprites, bars = _parse_grid(grid)
    jobs: list[dict] = []
    for bar in bars:
        br0, bc0, br1, bc1 = bar["bbox"]
        if bar["orientation"] == "h":
            candidates = [
                sprite
                for sprite in sprites
                if sprite["bbox"][1] == bc0 and sprite["bbox"][3] == bc1
            ]
        else:
            candidates = [
                sprite
                for sprite in sprites
                if sprite["bbox"][0] == br0 and sprite["bbox"][2] == br1
            ]
        assert len(candidates) == 1, "each control bar must match exactly one sprite"
        sprite = candidates[0]
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        if bar["orientation"] == "h":
            assert sr1 < br0 or sr0 > br0, "matched sprite must lie on one side of the horizontal bar"
        else:
            assert sc1 < bc0 or sc0 > bc0, "matched sprite must lie on one side of the vertical bar"
        mirrored = _mirror_coords(sprite, bar)
        jobs.append(
            {
                "sprite": sprite,
                "bar": bar,
                "mirrored": mirrored,
                "area": int(sprite["area"]),
                "bbox_area": (sr1 - sr0 + 1) * (sc1 - sc0 + 1),
                "sort_anchor": (sr0, sc0, sr1, sc1),
            }
        )
    return jobs


def _render_output_from_jobs(grid: np.ndarray, jobs: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    base = g.copy()
    for job in jobs:
        for r, c in job["bar"]["coords"]:
            base[r, c] = BG

    out = base.copy()
    draw_order = sorted(jobs, key=lambda job: (-job["area"], job["sort_anchor"]))
    for job in draw_order:
        color = int(job["sprite"]["color"])
        for r, c in job["mirrored"]:
            assert 0 <= r < h and 0 <= c < w
            if int(base[r, c]) == BG:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each span-matched sprite across its aligned control bar, then erase the bars.

    Input structure:
    - The background is black.
    - Grey one-cell-thick horizontal or vertical line segments are control bars.
    - Non-black, non-grey 4-connected monochrome components are sprites, and each
      bar matches exactly one sprite by exact span alignment.

    Rule:
    - For a horizontal bar, find the unique sprite with the same column span; for a
      vertical bar, find the unique sprite with the same row span.
    - Reflect every pixel of that sprite across the bar's row or column onto the
      opposite side; original sprites stay unchanged.
    - If reflected products overlap, larger sprites are drawn first and smaller
      sprites are drawn later, so the smaller reflected sprite remains visible.
    - Finally erase all grey control-bar pixels to black.

    Key insight:
    - The grey bar is simultaneously the selector, the reflection axis, and a
      disposable scaffold: span alignment chooses the sprite, the bar's geometry
      defines the mirror, and the bar itself disappears in the output.
    """
    jobs = _match_jobs(np.asarray(grid, dtype=int))
    return _render_output_from_jobs(np.asarray(grid, dtype=int), jobs)


def _dilate8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
        out.add((r, c))
    return out


def _make_horizontal_shape() -> dict:
    width = random.choice((5, 6, 7))
    options: list[tuple[int, int]] = []
    max_extra_h2 = min(3, 8 - width)
    if max_extra_h2 >= 1:
        for extra in range(1, max_extra_h2 + 1):
            options.append((2, extra))
    max_extra_h3 = min(3, 8 - width)
    if width <= 6 and max_extra_h3 >= 2:
        for extra in range(2, max_extra_h3 + 1):
            options.append((3, extra))
    height, extra = random.choice(options)
    mask = np.zeros((height, width), dtype=bool)
    if height == 2:
        base_row = random.randint(0, 1)
        bump_row = 1 - base_row
        mask[base_row, :] = True
        cols = random.sample(list(range(width)), extra)
        mask[bump_row, cols] = True
    else:
        mask[1, :] = True
        top_col = random.randrange(width)
        bottom_col = random.randrange(width)
        mask[0, top_col] = True
        mask[2, bottom_col] = True
        remaining = extra - 2
        free = [(0, c) for c in range(width) if c != top_col] + [(2, c) for c in range(width) if c != bottom_col]
        for rr, cc in random.sample(free, remaining):
            mask[rr, cc] = True
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "orientation": "h",
        "mask": mask,
        "coords": coords,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "area": len(coords),
        "bbox_area": int(mask.shape[0] * mask.shape[1]),
    }


def _make_vertical_shape() -> dict:
    height = random.choice((5, 6, 7))
    options: list[tuple[int, int]] = []
    max_extra_w2 = min(3, 8 - height)
    if max_extra_w2 >= 1:
        for extra in range(1, max_extra_w2 + 1):
            options.append((2, extra))
    max_extra_w3 = min(3, 8 - height)
    if height <= 6 and max_extra_w3 >= 2:
        for extra in range(2, max_extra_w3 + 1):
            options.append((3, extra))
    width, extra = random.choice(options)
    mask = np.zeros((height, width), dtype=bool)
    if width == 2:
        base_col = random.randint(0, 1)
        bump_col = 1 - base_col
        mask[:, base_col] = True
        rows = random.sample(list(range(height)), extra)
        mask[rows, bump_col] = True
    else:
        mask[:, 1] = True
        left_row = random.randrange(height)
        right_row = random.randrange(height)
        mask[left_row, 0] = True
        mask[right_row, 2] = True
        remaining = extra - 2
        free = [(r, 0) for r in range(height) if r != left_row] + [(r, 2) for r in range(height) if r != right_row]
        for rr, cc in random.sample(free, remaining):
            mask[rr, cc] = True
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "orientation": "v",
        "mask": mask,
        "coords": coords,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "area": len(coords),
        "bbox_area": int(mask.shape[0] * mask.shape[1]),
    }


def _make_shape(orientation: str, forbidden_areas: set[int] | None = None) -> dict:
    forbidden_areas = set() if forbidden_areas is None else set(forbidden_areas)
    factories = {"h": _make_horizontal_shape, "v": _make_vertical_shape}
    for _ in range(100):
        shape = factories[orientation]()
        if shape["area"] not in forbidden_areas:
            return shape
    raise RuntimeError("failed to sample a distinct sprite area")


def _place_shape(shape: dict, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape["coords"]]


def _sample_pair(
    grid_shape: tuple[int, int],
    color: int,
    orientation: str,
    used_row_spans: set[tuple[int, int]],
    used_col_spans: set[tuple[int, int]],
    occupied_forbidden: set[tuple[int, int]],
    area_forbidden: set[int] | None = None,
    overlap_target: set[tuple[int, int]] | None = None,
    require_overlap: bool = False,
) -> dict | None:
    h, w = grid_shape
    area_forbidden = set() if area_forbidden is None else set(area_forbidden)
    for _ in range(600):
        shape = _make_shape(orientation, area_forbidden)
        sh = int(shape["height"])
        sw = int(shape["width"])
        gap = random.randint(1, 3)
        if orientation == "h":
            side = random.choice(("above", "below"))
            if side == "above":
                max_top = h - (2 * sh + 2 * gap + 1)
                if max_top < 0:
                    continue
                top = random.randint(0, max_top)
                axis = top + sh + gap
            else:
                min_top = 2 * sh + 2 * gap
                if min_top > h - sh:
                    continue
                top = random.randint(min_top, h - sh)
                axis = top - gap - 1
            left = random.randint(0, w - sw)
            sprite_coords = _place_shape(shape, top, left)
            bar_coords = [(axis, c) for c in range(left, left + sw)]
            bbox = (top, left, top + sh - 1, left + sw - 1)
        else:
            side = random.choice(("left", "right"))
            if side == "left":
                max_left = w - (2 * sw + 2 * gap + 1)
                if max_left < 0:
                    continue
                left = random.randint(0, max_left)
                axis = left + sw + gap
            else:
                min_left = 2 * sw + 2 * gap
                if min_left > w - sw:
                    continue
                left = random.randint(min_left, w - sw)
                axis = left - gap - 1
            top = random.randint(0, h - sh)
            sprite_coords = _place_shape(shape, top, left)
            bar_coords = [(r, axis) for r in range(top, top + sh)]
            bbox = (top, left, top + sh - 1, left + sw - 1)

        row_span = (bbox[0], bbox[2])
        col_span = (bbox[1], bbox[3])
        if row_span in used_row_spans or col_span in used_col_spans:
            continue

        actual = set(sprite_coords) | set(bar_coords)
        if any(cell in occupied_forbidden for cell in actual):
            continue

        sprite = {
            "color": int(color),
            "coords": sprite_coords,
            "coord_set": set(sprite_coords),
            "bbox": bbox,
            "area": int(shape["area"]),
        }
        bar = {
            "color": GREY,
            "coords": bar_coords,
            "coord_set": set(bar_coords),
            "bbox": _bbox(bar_coords),
            "area": len(bar_coords),
            "orientation": orientation,
            "axis": axis,
        }
        mirrored = set(_mirror_coords(sprite, bar))
        if any(not (0 <= r < h and 0 <= c < w) for r, c in mirrored):
            continue
        if mirrored & actual:
            continue
        if overlap_target is not None:
            overlap_size = len(mirrored & overlap_target)
            if require_overlap and not (1 <= overlap_size <= 3):
                continue
            if not require_overlap and overlap_size != 0:
                continue

        return {
            "color": int(color),
            "sprite": sprite,
            "bar": bar,
            "mirrored": mirrored,
            "shape_area": int(shape["area"]),
            "bbox_area": int(shape["bbox_area"]),
        }
    return None


def _render_input(grid_shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for pair in pairs:
        for r, c in pair["sprite"]["coords"]:
            grid[r, c] = int(pair["color"])
        for r, c in pair["bar"]["coords"]:
            grid[r, c] = GREY
    return grid


def _render_expected_from_pairs(grid_shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = _render_input(grid_shape, pairs)
    jobs = []
    for pair in pairs:
        jobs.append(
            {
                "sprite": pair["sprite"],
                "bar": pair["bar"],
                "mirrored": list(pair["mirrored"]),
                "area": pair["shape_area"],
                "sort_anchor": pair["sprite"]["bbox"],
            }
        )
    return _render_output_from_jobs(grid, jobs)


def _matching_is_unique(grid: np.ndarray, pairs: list[dict]) -> bool:
    sprites, bars = _parse_grid(grid)
    if len(sprites) != len(pairs) or len(bars) != len(pairs):
        return False
    for bar in bars:
        br0, bc0, br1, bc1 = bar["bbox"]
        if bar["orientation"] == "h":
            candidates = [sprite for sprite in sprites if sprite["bbox"][1] == bc0 and sprite["bbox"][3] == bc1]
        else:
            candidates = [sprite for sprite in sprites if sprite["bbox"][0] == br0 and sprite["bbox"][2] == br1]
        if len(candidates) != 1:
            return False
    return True


def _validate_pairs(grid_shape: tuple[int, int], pairs: list[dict], overlap_mode: bool) -> bool:
    if len(pairs) not in (2, 3):
        return False
    grid = _render_input(grid_shape, pairs)
    if max(grid.shape) > 30:
        return False
    if not _matching_is_unique(grid, pairs):
        return False

    input_occ = {(r, c) for r, c in np.argwhere(grid != BG)}
    for pair in pairs:
        if pair["mirrored"] & input_occ:
            return False

    overlap_count = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            overlap = pairs[i]["mirrored"] & pairs[j]["mirrored"]
            overlap_count += len(overlap)
            if overlap:
                ai = pairs[i]["shape_area"]
                aj = pairs[j]["shape_area"]
                bi = pairs[i]["bbox_area"]
                bj = pairs[j]["bbox_area"]
                if ai == aj or bi == bj:
                    return False
                if (ai < aj) != (bi < bj):
                    return False

    if overlap_mode:
        if not (1 <= overlap_count <= 3):
            return False
    elif overlap_count != 0:
        return False

    expected = _render_expected_from_pairs(grid_shape, pairs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _orientation_plan(pair_count: int, overlap_mode: bool) -> list[str]:
    if overlap_mode:
        first = random.choice(("h", "v"))
        if pair_count == 2:
            return [first, first]
        return [first, first, "v" if first == "h" else "h"]
    if pair_count == 2:
        return random.choice([["h", "v"], ["v", "h"], ["h", "h"], ["v", "v"]])
    return random.choice(
        [
            ["h", "v", "h"],
            ["v", "h", "v"],
            ["h", "h", "v"],
            ["v", "v", "h"],
        ]
    )


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    overlap_mode = random.random() < 0.4
    grid_shape = (random.randint(14, 18), random.randint(14, 18))
    colors = random.sample(list(SPRITE_COLORS), pair_count)
    orientations = _orientation_plan(pair_count, overlap_mode)

    pairs: list[dict] = []
    used_row_spans: set[tuple[int, int]] = set()
    used_col_spans: set[tuple[int, int]] = set()
    occupied_actual: set[tuple[int, int]] = set()
    occupied_forbidden: set[tuple[int, int]] = set()
    first_pair = _sample_pair(
        grid_shape,
        colors[0],
        orientations[0],
        used_row_spans,
        used_col_spans,
        occupied_forbidden,
    )
    if first_pair is None:
        return None
    pairs.append(first_pair)
    used_row_spans.add((first_pair["sprite"]["bbox"][0], first_pair["sprite"]["bbox"][2]))
    used_col_spans.add((first_pair["sprite"]["bbox"][1], first_pair["sprite"]["bbox"][3]))
    occupied_actual |= set(first_pair["sprite"]["coords"]) | set(first_pair["bar"]["coords"])
    occupied_forbidden = _dilate8(occupied_actual, grid_shape)

    second_pair = _sample_pair(
        grid_shape,
        colors[1],
        orientations[1],
        used_row_spans,
        used_col_spans,
        occupied_forbidden,
        {first_pair["shape_area"]} if overlap_mode else None,
        overlap_target=first_pair["mirrored"] if overlap_mode else None,
        require_overlap=overlap_mode,
    )
    if second_pair is None:
        return None
    pairs.append(second_pair)
    used_row_spans.add((second_pair["sprite"]["bbox"][0], second_pair["sprite"]["bbox"][2]))
    used_col_spans.add((second_pair["sprite"]["bbox"][1], second_pair["sprite"]["bbox"][3]))
    occupied_actual |= set(second_pair["sprite"]["coords"]) | set(second_pair["bar"]["coords"])
    occupied_forbidden = _dilate8(occupied_actual, grid_shape)

    for idx in range(2, pair_count):
        pair = _sample_pair(
            grid_shape,
            colors[idx],
            orientations[idx],
            used_row_spans,
            used_col_spans,
            occupied_forbidden,
            overlap_target=set().union(*(p["mirrored"] for p in pairs)) if not overlap_mode else None,
            require_overlap=False,
        )
        if pair is None:
            return None
        pairs.append(pair)
        used_row_spans.add((pair["sprite"]["bbox"][0], pair["sprite"]["bbox"][2]))
        used_col_spans.add((pair["sprite"]["bbox"][1], pair["sprite"]["bbox"][3]))
        occupied_actual |= set(pair["sprite"]["coords"]) | set(pair["bar"]["coords"])
        occupied_forbidden = _dilate8(occupied_actual, grid_shape)

    if not _validate_pairs(grid_shape, pairs, overlap_mode):
        return None
    return _render_input(grid_shape, pairs)


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
