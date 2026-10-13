import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_bbox_edge(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _count_holes(sprite: np.ndarray, color: int) -> int:
    bg_parts = _components_mask(sprite == 0, connectivity=4)
    holes = 0
    height, width = map(int, sprite.shape)
    for coords in bg_parts:
        if not _touches_bbox_edge(coords, height, width):
            holes += 1
    return holes


def _sprite_axis(sprite: np.ndarray) -> str:
    height, width = map(int, sprite.shape)
    return "H" if width >= height else "V"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clone each monochrome object as many times as the number of enclosed holes it contains.

    Input structure:
    - Black (0) background.
    - Several non-touching 4-connected monochrome objects, each using one non-zero color.
    - Some objects contain one or more enclosed black holes fully surrounded by their own pixels.

    Rule:
    - For each object, count the 4-connected black regions inside its bounding box that do not touch that box boundary.
    - Use the object's bounding-box long axis to choose the clone direction: horizontal when width >= height, otherwise vertical.
    - Keep the original object in place, then place that many full bbox-pattern copies one pixel apart in the chosen direction.

    Key insight:
    - The black holes are not repaired; they encode how many whole-object clones must be appended, while the bbox aspect ratio chooses whether the copies extend rightward or downward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = map(int, g.shape)

    for coords in _components_mask(g != 0, connectivity=4):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        holes = _count_holes(sprite, color)
        sh, sw = map(int, sprite.shape)
        axis = _sprite_axis(sprite)

        for idx in range(1, holes + 1):
            if axis == "H":
                nr = int(r0)
                nc = int(c0 + idx * (sw + 1))
            else:
                nr = int(r0 + idx * (sh + 1))
                nc = int(c0)
            assert 0 <= nr and 0 <= nc and nr + sh <= height and nc + sw <= width, "clone exceeds grid"
            out[nr : nr + sh, nc : nc + sw] = sprite

    return out


def _carve_boundary_notch(rng: random.Random, sprite: np.ndarray) -> None:
    height, width = map(int, sprite.shape)
    if height < 4 or width < 4:
        return

    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        notch_height = rng.randint(1, height - 2)
        notch_width = rng.randint(1, width - 2)
        start = rng.randint(1, width - notch_width - 1)
        rows = slice(0, notch_height) if side == "top" else slice(height - notch_height, height)
        sprite[rows, start : start + notch_width] = 0
    else:
        notch_width = rng.randint(1, width - 2)
        notch_height = rng.randint(1, height - 2)
        start = rng.randint(1, height - notch_height - 1)
        cols = slice(0, notch_width) if side == "left" else slice(width - notch_width, width)
        sprite[start : start + notch_height, cols] = 0


def _make_zero_hole_sprite(rng: random.Random, axis: str, color: int) -> np.ndarray:
    if axis == "H":
        height = rng.randint(3, 6)
        if rng.random() < 0.25:
            width = height
        else:
            width = rng.randint(height, min(9, height + 4))
    else:
        width = rng.randint(3, 6)
        height = rng.randint(width + 1, min(9, width + 4))

    sprite = np.full((height, width), color, dtype=int)
    if rng.random() < 0.7:
        _carve_boundary_notch(rng, sprite)
    assert _count_holes(sprite, color) == 0
    assert _sprite_axis(sprite) == axis
    return sprite


def _make_hole_sprite(rng: random.Random, axis: str, holes: int, color: int) -> np.ndarray:
    assert holes in (1, 2)
    if axis == "H":
        height = rng.randint(5, 7)
        hole_width_cap = 3 if holes == 1 else 2
        hole_widths = [rng.randint(1, hole_width_cap) for _ in range(holes)]
        hole_heights = [rng.randint(1, height - 2) for _ in range(holes)]
        hole_tops = [rng.randint(1, height - hh - 1) for hh in hole_heights]

        base_width = sum(hole_widths) + (holes + 1)
        min_width = max(height, base_width)
        max_width = 10 if holes == 1 else 8
        width = rng.randint(min_width, min(max_width, min_width + 2))

        gaps = [1] * (holes + 1)
        extra = width - (sum(hole_widths) + sum(gaps))
        for _ in range(extra):
            gaps[rng.randrange(len(gaps))] += 1

        sprite = np.full((height, width), color, dtype=int)
        col = gaps[0]
        for idx in range(holes):
            hh = hole_heights[idx]
            ww = hole_widths[idx]
            top = hole_tops[idx]
            sprite[top : top + hh, col : col + ww] = 0
            col += ww + gaps[idx + 1]
    else:
        width = rng.randint(4, 6)
        hole_height_cap = 3 if holes == 1 else 2
        hole_heights = [rng.randint(1, hole_height_cap) for _ in range(holes)]
        hole_widths = [rng.randint(1, width - 2) for _ in range(holes)]
        hole_lefts = [rng.randint(1, width - ww - 1) for ww in hole_widths]

        base_height = sum(hole_heights) + (holes + 1)
        min_height = max(width + 1, base_height)
        max_height = 10 if holes == 1 else 8
        height = rng.randint(min_height, min(max_height, min_height + 2))

        gaps = [1] * (holes + 1)
        extra = height - (sum(hole_heights) + sum(gaps))
        for _ in range(extra):
            gaps[rng.randrange(len(gaps))] += 1

        sprite = np.full((height, width), color, dtype=int)
        row = gaps[0]
        for idx in range(holes):
            hh = hole_heights[idx]
            ww = hole_widths[idx]
            left = hole_lefts[idx]
            sprite[row : row + hh, left : left + ww] = 0
            row += hh + gaps[idx + 1]

    assert _count_holes(sprite, color) == holes
    assert _sprite_axis(sprite) == axis
    return sprite


def _make_sprite(rng: random.Random, axis: str, holes: int, color: int) -> np.ndarray:
    if holes == 0:
        return _make_zero_hole_sprite(rng, axis, color)
    return _make_hole_sprite(rng, axis, holes, color)


def _reserve_shape(sprite: np.ndarray, holes: int) -> tuple[int, int]:
    height, width = map(int, sprite.shape)
    axis = _sprite_axis(sprite)
    if axis == "H":
        return height, (holes + 1) * width + holes
    return (holes + 1) * height + holes, width


def _sample_orientations(rng: random.Random, count: int) -> list[str]:
    if count == 2:
        patterns = [("H", "V"), ("H", "H"), ("V", "V")]
        axes = list(rng.choice(patterns))
    else:
        axes = ["H", "V", rng.choice(["H", "V"])]
    rng.shuffle(axes)
    return axes[:count]


def _sample_holes(rng: random.Random, count: int) -> list[int]:
    if count == 2:
        patterns = [(0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)]
        hole_counts = list(rng.choice(patterns))
    else:
        patterns = [
            [0, 1, 2],
            [0, 1, 1],
            [0, 2, 1],
        ]
        hole_counts = list(rng.choice(patterns))
    rng.shuffle(hole_counts)
    return hole_counts[:count]


def _layout_specs(
    rng: random.Random, specs: list[dict[str, object]]
) -> tuple[int, int, list[tuple[int, int]]] | None:
    count = len(specs)
    indices = list(range(count))

    for _ in range(200):
        rng.shuffle(indices)
        if count == 2:
            row_count = 1 if rng.random() < 0.4 else 2
        else:
            row_count = 2 if rng.random() < 0.85 else 1

        if row_count == 1:
            rows = [indices.copy()]
        else:
            if count == 2:
                rows = [[indices[0]], [indices[1]]]
            else:
                split = rng.choice([1, 2])
                rows = [indices[:split], indices[split:]]

        x_gaps = []
        row_widths = []
        row_heights = []
        for row in rows:
            gaps = [rng.randint(2, 4) for _ in range(max(0, len(row) - 1))]
            x_gaps.append(gaps)
            row_width = sum(int(specs[idx]["reserve_w"]) for idx in row) + sum(gaps)
            row_height = max(int(specs[idx]["reserve_h"]) for idx in row)
            row_widths.append(row_width)
            row_heights.append(row_height)

        y_gaps = [rng.randint(2, 4) for _ in range(max(0, row_count - 1))]
        content_height = sum(row_heights) + sum(y_gaps)
        content_width = max(row_widths)
        min_height = max(20, content_height + 2)
        min_width = max(20, content_width + 2)
        if min_height > 30 or min_width > 30:
            continue

        height = rng.randint(min_height, 30)
        width = rng.randint(min_width, 30)
        placements: list[tuple[int, int] | None] = [None] * count

        extra_y = height - content_height
        top_pad = rng.randint(1, extra_y - 1)
        current_row_top = top_pad

        for row_idx, row in enumerate(rows):
            row_height = row_heights[row_idx]
            row_width = row_widths[row_idx]
            extra_x = width - row_width
            left_pad = rng.randint(1, extra_x - 1)
            current_col = left_pad

            for item_idx, spec_idx in enumerate(row):
                reserve_h = int(specs[spec_idx]["reserve_h"])
                reserve_w = int(specs[spec_idx]["reserve_w"])
                y_offset = rng.randint(0, row_height - reserve_h)
                placements[spec_idx] = (current_row_top + y_offset, current_col)
                current_col += reserve_w
                if item_idx < len(row) - 1:
                    current_col += x_gaps[row_idx][item_idx]

            current_row_top += row_height
            if row_idx < row_count - 1:
                current_row_top += y_gaps[row_idx]

        if all(p is not None for p in placements):
            return height, width, [(int(r), int(c)) for r, c in placements]  # type: ignore[misc]

    return None


def _build_expected(height: int, width: int, specs: list[dict[str, object]], positions: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    for spec, (r0, c0) in zip(specs, positions):
        sprite = np.asarray(spec["sprite"], dtype=int)
        holes = int(spec["holes"])
        sh, sw = map(int, sprite.shape)
        inp[r0 : r0 + sh, c0 : c0 + sw] = sprite
        out[r0 : r0 + sh, c0 : c0 + sw] = sprite

        if spec["axis"] == "H":
            for idx in range(1, holes + 1):
                nc = c0 + idx * (sw + 1)
                out[r0 : r0 + sh, nc : nc + sw] = sprite
        else:
            for idx in range(1, holes + 1):
                nr = r0 + idx * (sh + 1)
                out[nr : nr + sh, c0 : c0 + sw] = sprite

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = 2 if rng.random() < 0.7 else 3
    colors = rng.sample(list(range(1, 10)), count)
    axes = _sample_orientations(rng, count)
    hole_counts = _sample_holes(rng, count)

    specs: list[dict[str, object]] = []
    for color, axis, holes in zip(colors, axes, hole_counts):
        sprite = _make_sprite(rng, axis, holes, color)
        reserve_h, reserve_w = _reserve_shape(sprite, holes)
        specs.append(
            {
                "sprite": sprite,
                "axis": axis,
                "holes": holes,
                "color": color,
                "reserve_h": reserve_h,
                "reserve_w": reserve_w,
            }
        )

    layout = _layout_specs(rng, specs)
    if layout is None:
        return None

    height, width, positions = layout
    inp, expected = _build_expected(height, width, specs, positions)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clone-by-hole ARC input")
