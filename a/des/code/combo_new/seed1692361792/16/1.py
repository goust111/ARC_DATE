from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = list(range(1, 10))
GAP_SIDES = ("top", "right", "bottom", "left")
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(g: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _gap_side_from_component(comp: dict) -> str | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in comp["coords"]:
        mask[int(rr - r0), int(cc - c0)] = True

    if bool(mask[1:-1, 1:-1].any()):
        return None

    sides = {
        "top": mask[0, :],
        "right": mask[:, -1],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
    }
    missing: list[str] = []
    for side, edge in sides.items():
        holes = np.flatnonzero(~edge)
        if len(holes) == 0:
            continue
        if len(holes) < 2:
            return None
        if int(holes[0]) == 0 or int(holes[-1]) == len(edge) - 1:
            return None
        if int(holes[-1] - holes[0] + 1) != len(holes):
            return None
        missing.append(side)

    if len(missing) != 1:
        return None
    return missing[0]


def _extract_objects(g: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for comp in _find_components(g, bg=BG):
        gap_side = _gap_side_from_component(comp)
        if gap_side is None:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        frame_color = int(comp["color"])
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pts = np.argwhere(inner != BG)
        assert len(pts) > 0

        sprite_colors = set(map(int, inner[inner != BG].ravel()))
        assert len(sprite_colors) == 1
        sprite_color = sprite_colors.pop()
        assert sprite_color != frame_color

        sr0, sc0 = pts.min(axis=0)
        sr1, sc1 = pts.max(axis=0)
        assert int(sr0) == 0 and int(sc0) == 0
        sprite = inner[: int(sr1) + 1, : int(sc1) + 1].copy()
        assert int(np.count_nonzero(sprite)) == len(pts)

        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap_side": gap_side,
                "frame_color": frame_color,
                "sprite_color": sprite_color,
                "sprite": sprite,
            }
        )

    assert objects, "no valid gap-frame objects detected"
    return objects


def _rotate_sprite(sprite: np.ndarray, gap_side: str) -> np.ndarray:
    if gap_side == "top":
        return sprite.copy()
    if gap_side == "right":
        return np.rot90(sprite, -1)
    if gap_side == "bottom":
        return np.rot90(sprite, 2)
    if gap_side == "left":
        return np.rot90(sprite, 1)
    raise ValueError(f"unknown gap side: {gap_side}")


def _nest_boxes(sprite: np.ndarray, frame_color: int, sprite_color: int) -> np.ndarray:
    seed = np.asarray(sprite, dtype=int)
    h, w = seed.shape
    n = int(np.count_nonzero(seed))
    assert 1 <= n <= 6

    out = np.full((h + 2 * n, w + 2 * n), BG, dtype=int)
    out[n : n + h, n : n + w] = seed
    for layer in range(n):
        color = frame_color if layer % 2 == 0 else sprite_color
        r0 = n - 1 - layer
        c0 = n - 1 - layer
        r1 = n + h + layer
        c1 = n + w + layer
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _row_groups(objects: list[dict]) -> list[list[dict]]:
    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    rows: list[list[dict]] = []
    current: list[dict] = []
    current_bottom = -1
    for obj in ordered:
        r0, _, r1, _ = obj["bbox"]
        if current and int(r0) > current_bottom:
            rows.append(sorted(current, key=lambda item: item["bbox"][1]))
            current = []
            current_bottom = -1
        current.append(obj)
        current_bottom = max(current_bottom, int(r1))
    if current:
        rows.append(sorted(current, key=lambda item: item["bbox"][1]))
    return rows


def _pack_rows(rows: list[list[np.ndarray]]) -> np.ndarray:
    row_tiles: list[np.ndarray] = []
    for row in rows:
        row_h = max(tile.shape[0] for tile in row)
        row_w = sum(tile.shape[1] for tile in row) + max(0, len(row) - 1)
        strip = np.full((row_h, row_w), BG, dtype=int)
        col = 0
        for tile in row:
            h, w = tile.shape
            strip[:h, col : col + w] = tile
            col += w + 1
        row_tiles.append(strip)

    out_h = sum(tile.shape[0] for tile in row_tiles) + max(0, len(row_tiles) - 1)
    out_w = max(tile.shape[1] for tile in row_tiles)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for tile in row_tiles:
        h, w = tile.shape
        out[row : row + h, :w] = tile
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed sprite by its border-gap direction, then wrap it in count-controlled alternating nested frames.

    Input structure:
    - The background is black.
    - Several isolated one-pixel-thick hollow square frames appear on the grid, and each frame has exactly one contiguous gap on one side.
    - Each frame contains one smaller sprite of a different color, aligned to the top-left corner of the frame interior, while the rest of the interior stays background.

    Rule:
    - Interpret the gap side as a rotation code for the interior sprite: top = unchanged, right = 90 degrees clockwise, bottom = 180 degrees, left = 90 degrees counterclockwise.
    - Count the rotated sprite's foreground pixels as N, then keep the rotated sprite and add N tight nested box layers around it.
    - Color the nested layers alternately with the frame color and the sprite color, starting with the frame color on the innermost added layer.
    - Tight-crop every transformed structure and repack the structures in the same row/column order as the input, with one background cell between neighbors.

    Key insight:
    - The missing border segment is an orientation instruction, and the rotated sprite's own pixel count determines how many alternating layers must be wrapped around it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    transformed_rows: list[list[np.ndarray]] = []
    for row in _row_groups(objects):
        tiles: list[np.ndarray] = []
        for obj in row:
            rotated = _rotate_sprite(obj["sprite"], obj["gap_side"])
            tile = _nest_boxes(rotated, obj["frame_color"], obj["sprite_color"])
            tiles.append(tile)
        transformed_rows.append(tiles)
    return _pack_rows(transformed_rows)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            total += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return total


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = [
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
    ]
    for other in variants:
        if other.shape == mask.shape and np.array_equal(mask, other):
            return False
    return True


def _random_sprite_mask() -> np.ndarray:
    for _ in range(300):
        max_h = random.randint(2, 3)
        max_w = random.randint(2, 3)
        max_area = max_h * max_w
        target = random.randint(2, min(6, max_area))

        mask = np.zeros((max_h, max_w), dtype=bool)
        mask[0, 0] = True
        frontier = {(nr, nc) for nr, nc in _neighbors4(0, 0) if 0 <= nr < max_h and 0 <= nc < max_w}
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < max_h and 0 <= nc < max_w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue
        pts = np.argwhere(mask)
        r1, c1 = pts.max(axis=0)
        cropped = mask[: int(r1) + 1, : int(c1) + 1]
        if cropped.shape == (1, 1):
            continue
        if _component_count(cropped) != 1:
            continue
        if not _is_asymmetric(cropped):
            continue
        return cropped
    raise RuntimeError("failed to build asymmetric sprite")


def _render_input_object(grid: np.ndarray, obj: dict) -> None:
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    side = int(obj["outer_side"])
    frame_color = int(obj["frame_color"])
    sprite_color = int(obj["sprite_color"])
    gap_side = obj["gap_side"]
    gap_start = int(obj["gap_start"])
    gap_len = int(obj["gap_len"])

    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    if gap_side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif gap_side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif gap_side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c1] = BG

    mask = obj["sprite_mask"]
    h, w = mask.shape
    patch = grid[r0 + 1 : r0 + 1 + h, c0 + 1 : c0 + 1 + w]
    patch[mask] = sprite_color


def _expected_output(rows: list[list[dict]]) -> np.ndarray:
    transformed_rows: list[list[np.ndarray]] = []
    for row in rows:
        tiles: list[np.ndarray] = []
        for obj in row:
            sprite = np.where(obj["sprite_mask"], obj["sprite_color"], BG)
            rotated = _rotate_sprite(sprite, obj["gap_side"])
            tiles.append(_nest_boxes(rotated, obj["frame_color"], obj["sprite_color"]))
        transformed_rows.append(tiles)
    return _pack_rows(transformed_rows)


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _make_object_specs(count: int) -> list[dict]:
    colors = random.sample(COLOR_POOL, 2 * count)
    specs: list[dict] = []
    for i in range(count):
        sprite_mask = _random_sprite_mask()
        sprite_h, sprite_w = sprite_mask.shape
        inner_side = max(sprite_h, sprite_w) + random.randint(2, 4)
        outer_side = inner_side + 2
        gap_side = random.choice(GAP_SIDES)
        gap_len = random.randint(2, min(4, outer_side - 2))
        gap_start = random.randint(1, outer_side - gap_len - 1)
        specs.append(
            {
                "frame_color": colors[i],
                "sprite_color": colors[count + i],
                "sprite_mask": sprite_mask,
                "gap_side": gap_side,
                "gap_len": gap_len,
                "gap_start": gap_start,
                "outer_side": outer_side,
            }
        )
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    obj_count = random.choice([1, 2, 2, 3, 3, 4])
    specs = _make_object_specs(obj_count)

    row_count = random.randint(1, min(2, obj_count))
    row_sizes = _random_partition(obj_count, row_count)
    rows: list[list[dict]] = []
    idx = 0
    for size in row_sizes:
        rows.append(specs[idx : idx + size])
        idx += size

    top_margin = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)
    left_margins = [random.randint(0, 3) for _ in rows]
    right_margins = [random.randint(0, 3) for _ in rows]
    row_gaps = [random.randint(1, 3) for _ in range(max(0, len(rows) - 1))]

    row_widths: list[int] = []
    row_heights: list[int] = []
    horizontal_gaps: list[list[int]] = []
    for row in rows:
        gaps = [random.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        horizontal_gaps.append(gaps)
        width = sum(int(obj["outer_side"]) for obj in row) + sum(gaps)
        height = max(int(obj["outer_side"]) for obj in row)
        row_widths.append(width)
        row_heights.append(height)

    input_h = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
    input_w = max(left_margins[i] + row_widths[i] + right_margins[i] for i in range(len(rows)))
    if input_h > 30 or input_w > 30:
        return None

    expected = _expected_output(rows)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    grid = np.full((input_h, input_w), BG, dtype=int)
    cur_r = top_margin
    for row_idx, row in enumerate(rows):
        cur_c = left_margins[row_idx]
        for obj_idx, obj in enumerate(row):
            obj["r0"] = cur_r
            obj["c0"] = cur_c
            _render_input_object(grid, obj)
            cur_c += int(obj["outer_side"])
            if obj_idx < len(row) - 1:
                cur_c += horizontal_gaps[row_idx][obj_idx]
        cur_r += row_heights[row_idx]
        if row_idx < len(rows) - 1:
            cur_r += row_gaps[row_idx]

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
