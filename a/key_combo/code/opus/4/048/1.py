from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components


DIRS = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

HORIZONTAL_DIRS = {"left", "right"}
VERTICAL_DIRS = {"up", "down"}
GENERATION_RETRIES = 400


def _component_colors(grid: np.ndarray, coords: np.ndarray) -> list[int]:
    return sorted({int(grid[r, c]) for r, c in coords if int(grid[r, c]) != Color.BLACK})


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _find_cross(grid: np.ndarray) -> dict:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="dict")
    candidates = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        sprite = crop_bbox(grid, comp["bbox"])
        full_rows = [r for r in range(sprite.shape[0]) if np.all(sprite[r, :] != Color.BLACK)]
        full_cols = [c for c in range(sprite.shape[1]) if np.all(sprite[:, c] != Color.BLACK)]
        if len(full_rows) == 1 and len(full_cols) == 1 and len(_component_colors(grid, comp["coords"])) >= 2:
            candidates.append(
                {
                    "bbox": comp["bbox"],
                    "sprite": sprite,
                    "center_r": r0 + full_rows[0],
                    "center_c": c0 + full_cols[0],
                    "full_row": full_rows[0],
                    "full_col": full_cols[0],
                    "coords": comp["coords"],
                }
            )
    assert len(candidates) == 1, "expected exactly one multicolor cross component"
    return candidates[0]


def _infer_periodic_sequence(line: np.ndarray, center_idx: int) -> tuple[int, ...]:
    center_idx = int(center_idx)
    left_len = center_idx
    right_len = int(line.shape[0] - center_idx - 1)
    assert left_len == right_len and right_len >= 4, "cross arms must be symmetric and long enough"

    radial = [int(line[center_idx])]
    for d in range(1, right_len + 1):
        left = int(line[center_idx - d])
        right = int(line[center_idx + d])
        assert left == right != Color.BLACK, "cross colors must be mirrored around the center"
        radial.append(right)

    valid_periods = []
    for period in (2, 3):
        if all(radial[d] == radial[d % period] for d in range(len(radial))):
            seq = radial[:period]
            if len(set(seq)) == period:
                valid_periods.append(period)
    assert len(valid_periods) == 1, "cross periodic sequence must be uniquely inferable"
    period = valid_periods[0]
    return tuple(radial[:period])


def _find_obstacles(grid: np.ndarray, cross_bbox: tuple[int, int, int, int]) -> list[dict]:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=False, return_format="dict")
    obstacles = []
    for comp in comps:
        if tuple(comp["bbox"]) == tuple(cross_bbox):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        sprite = crop_bbox(grid, comp["bbox"])
        colors = _component_colors(grid, comp["coords"])
        assert len(colors) == 1, "every obstacle must be monochrome in the input"
        color = colors[0]
        assert np.all(sprite == color), "every obstacle must be a solid rectangle"
        assert sprite.shape[0] == 2 and sprite.shape[1] in (2, 3), "obstacles must be 2x2 or 2x3"
        obstacles.append({"bbox": (r0, c0, r1, c1), "color": color})
    assert 2 <= len(obstacles) <= 4, "expected 2 to 4 obstacles"
    return obstacles


def _period_color(seq: tuple[int, ...], distance: int) -> int:
    return int(seq[int(distance) % len(seq)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the cross periodically until obstacles, then recolor each obstacle's path cells with the blocked ghost continuation.

    Input structure:
    - The grid has black background, one multicolor cross, and 2 to 4 solid rectangular obstacle blocks.
    - The cross occupies one horizontal seed row and one vertical seed column that intersect at a single center pixel.
    - Each obstacle is a monochrome 2x2 or 2x3 rectangle whose color does not belong to the cross sequences.

    Rule:
    - Keep the seed cross unchanged and infer the horizontal and vertical periodic color sequences from its center row and center column.
    - Along the center row and center column, repeat those sequences into black cells only.
    - In each direction, stop as soon as the extension reaches an obstacle: do not paint the obstacle and do not continue beyond it.
    - For every obstacle pixel that lies on the center row or center column, repaint that pixel with the sequence color for its exact distance from the cross center; all other obstacle pixels stay unchanged.

    Key insight:
    - The visible periodic extension stops at each obstacle, but the obstacle's intersecting cells are colored by the invisible ghost continuation that would have appeared at those coordinates.
    """
    g = np.asarray(grid, dtype=int)
    cross = _find_cross(g)
    center = (cross["center_r"], cross["center_c"])
    cross_mask = np.zeros(g.shape, dtype=bool)
    for r, c in cross["coords"]:
        cross_mask[int(r), int(c)] = True

    horiz_seq = _infer_periodic_sequence(cross["sprite"][cross["full_row"], :], cross["full_col"])
    vert_seq = _infer_periodic_sequence(cross["sprite"][:, cross["full_col"]], cross["full_row"])
    assert horiz_seq[0] == vert_seq[0], "horizontal and vertical sequences must agree at the center"

    obstacles = _find_obstacles(g, cross["bbox"])
    out = g.copy()
    h, w = g.shape
    cr, cc = center

    for name, (dr, dc) in DIRS.items():
        seq = horiz_seq if name in HORIZONTAL_DIRS else vert_seq
        distance = 1
        while True:
            r = cr + dr * distance
            c = cc + dc * distance
            if not (0 <= r < h and 0 <= c < w):
                break
            if cross_mask[r, c]:
                distance += 1
                continue
            if g[r, c] != Color.BLACK:
                break
            out[r, c] = _period_color(seq, distance)
            distance += 1

    for obstacle in obstacles:
        r0, c0, r1, c1 = obstacle["bbox"]
        if r0 <= cr <= r1:
            for c in range(c0, c1 + 1):
                out[cr, c] = _period_color(horiz_seq, abs(c - cc))
        if c0 <= cc <= c1:
            for r in range(r0, r1 + 1):
                out[r, cc] = _period_color(vert_seq, abs(r - cr))

    return out


def _choose_sequence(period: int, center_color: int) -> tuple[int, ...]:
    palette = [c for c in Color.NOT_BLACK if c != center_color]
    tail = random.sample(palette, k=period - 1)
    return (center_color, *tail)


def _rectangles_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _candidate_bboxes(
    direction: str,
    shape: tuple[int, int],
    center: tuple[int, int],
    horiz_extent: int,
    vert_extent: int,
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    cr, cc = center
    candidates = []
    widths = [2, 3]
    random.shuffle(widths)

    if direction == "left":
        for width in widths:
            max_c0 = cc - horiz_extent - width - 1
            cols = list(range(1, max_c0 + 1))
            random.shuffle(cols)
            rows = [cr - 1, cr]
            random.shuffle(rows)
            for r0 in rows:
                for c0 in cols:
                    candidates.append((r0, c0, r0 + 1, c0 + width - 1))
    elif direction == "right":
        for width in widths:
            min_c0 = cc + horiz_extent + 2
            max_c0 = w - width - 1
            cols = list(range(min_c0, max_c0 + 1))
            random.shuffle(cols)
            rows = [cr - 1, cr]
            random.shuffle(rows)
            for r0 in rows:
                for c0 in cols:
                    candidates.append((r0, c0, r0 + 1, c0 + width - 1))
    elif direction == "up":
        for width in widths:
            max_r0 = cr - vert_extent - 3
            rows = list(range(1, max_r0 + 1))
            random.shuffle(rows)
            cols = [cc - k for k in range(width) if 0 <= cc - k and cc - k + width - 1 < w]
            random.shuffle(cols)
            for r0 in rows:
                for c0 in cols:
                    candidates.append((r0, c0, r0 + 1, c0 + width - 1))
    else:
        for width in widths:
            min_r0 = cr + vert_extent + 2
            max_r0 = h - 3
            rows = list(range(min_r0, max_r0 + 1))
            random.shuffle(rows)
            cols = [cc - k for k in range(width) if 0 <= cc - k and cc - k + width - 1 < w]
            random.shuffle(cols)
            for r0 in rows:
                for c0 in cols:
                    candidates.append((r0, c0, r0 + 1, c0 + width - 1))

    return candidates


def _render_input(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    cr, cc = spec["center"]
    horiz_seq = spec["horiz_seq"]
    vert_seq = spec["vert_seq"]
    horiz_extent = spec["horiz_extent"]
    vert_extent = spec["vert_extent"]

    grid = np.zeros((h, w), dtype=int)
    for d in range(-horiz_extent, horiz_extent + 1):
        grid[cr, cc + d] = _period_color(horiz_seq, abs(d))
    for d in range(-vert_extent, vert_extent + 1):
        color = _period_color(vert_seq, abs(d))
        if d == 0:
            assert grid[cr, cc] == color
        else:
            grid[cr + d, cc] = color

    for obstacle in spec["obstacles"]:
        r0, c0, r1, c1 = obstacle["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(obstacle["color"])
    return grid


def _oracle_output(spec: dict, grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = spec["shape"]
    cr, cc = spec["center"]
    horiz_seq = spec["horiz_seq"]
    vert_seq = spec["vert_seq"]
    horiz_extent = spec["horiz_extent"]
    vert_extent = spec["vert_extent"]

    for c in range(cc + horiz_extent + 1, w):
        hit = next((obs for obs in spec["obstacles"] if obs["direction"] == "right" and obs["bbox"][1] <= c <= obs["bbox"][3]), None)
        if hit is not None:
            break
        out[cr, c] = _period_color(horiz_seq, c - cc)
    for c in range(cc - horiz_extent - 1, -1, -1):
        hit = next((obs for obs in spec["obstacles"] if obs["direction"] == "left" and obs["bbox"][1] <= c <= obs["bbox"][3]), None)
        if hit is not None:
            break
        out[cr, c] = _period_color(horiz_seq, cc - c)
    for r in range(cr - vert_extent - 1, -1, -1):
        hit = next((obs for obs in spec["obstacles"] if obs["direction"] == "up" and obs["bbox"][0] <= r <= obs["bbox"][2]), None)
        if hit is not None:
            break
        out[r, cc] = _period_color(vert_seq, cr - r)
    for r in range(cr + vert_extent + 1, h):
        hit = next((obs for obs in spec["obstacles"] if obs["direction"] == "down" and obs["bbox"][0] <= r <= obs["bbox"][2]), None)
        if hit is not None:
            break
        out[r, cc] = _period_color(vert_seq, r - cr)

    for obstacle in spec["obstacles"]:
        r0, c0, r1, c1 = obstacle["bbox"]
        if obstacle["direction"] in HORIZONTAL_DIRS:
            for c in range(c0, c1 + 1):
                out[cr, c] = _period_color(horiz_seq, abs(c - cc))
        else:
            for r in range(r0, r1 + 1):
                out[r, cc] = _period_color(vert_seq, abs(r - cr))
    return out


def _strip_colors(spec: dict, obstacle: dict) -> list[int]:
    cr, cc = spec["center"]
    if obstacle["direction"] in HORIZONTAL_DIRS:
        c0, c1 = obstacle["bbox"][1], obstacle["bbox"][3]
        return [_period_color(spec["horiz_seq"], abs(c - cc)) for c in range(c0, c1 + 1)]
    r0, r1 = obstacle["bbox"][0], obstacle["bbox"][2]
    return [_period_color(spec["vert_seq"], abs(r - cr)) for r in range(r0, r1 + 1)]


def _validate_spec(spec: dict, grid: np.ndarray, expected: np.ndarray) -> None:
    h, w = spec["shape"]
    cr, cc = spec["center"]
    assert grid.shape == (h, w)
    assert expected.shape == (h, w)
    assert max(grid.shape) <= 30
    assert spec["horiz_seq"][0] == spec["vert_seq"][0]
    assert len(spec["obstacles"]) == len({obs["direction"] for obs in spec["obstacles"]})
    assert len(spec["obstacles"]) >= 2
    assert any(obs["direction"] in HORIZONTAL_DIRS for obs in spec["obstacles"])
    assert any(obs["direction"] in VERTICAL_DIRS for obs in spec["obstacles"])

    cross_bbox = (
        cr - spec["vert_extent"],
        cc - spec["horiz_extent"],
        cr + spec["vert_extent"],
        cc + spec["horiz_extent"],
    )
    for obstacle in spec["obstacles"]:
        r0, c0, r1, c1 = obstacle["bbox"]
        assert r1 - r0 + 1 == 2 and c1 - c0 + 1 in (2, 3)
        assert not _rectangles_touch(cross_bbox, obstacle["bbox"], pad=0)
        if obstacle["direction"] in HORIZONTAL_DIRS:
            assert r0 <= cr <= r1
        else:
            assert c0 <= cc <= c1

    for i, obstacle in enumerate(spec["obstacles"]):
        for other in spec["obstacles"][i + 1 :]:
            assert not _rectangles_touch(obstacle["bbox"], other["bbox"], pad=1)

    cross = _find_cross(grid)
    parsed_h = _infer_periodic_sequence(cross["sprite"][cross["full_row"], :], cross["full_col"])
    parsed_v = _infer_periodic_sequence(cross["sprite"][:, cross["full_col"]], cross["full_row"])
    assert (cross["center_r"], cross["center_c"]) == spec["center"]
    assert parsed_h == tuple(spec["horiz_seq"])
    assert parsed_v == tuple(spec["vert_seq"])

    obstacle_colors = {obs["color"] for obs in spec["obstacles"]}
    cross_colors = set(spec["horiz_seq"]) | set(spec["vert_seq"])
    assert obstacle_colors.isdisjoint(cross_colors)

    strips = [_strip_colors(spec, obstacle) for obstacle in spec["obstacles"]]
    assert any(len(set(strip)) >= 2 for strip in strips)

    for obstacle, strip in zip(spec["obstacles"], strips):
        assert all(color != obstacle["color"] for color in strip)
        r0, c0, r1, c1 = obstacle["bbox"]
        if obstacle["direction"] == "right":
            near_color = strip[0]
            prev_color = expected[cr, c0 - 1]
            assert near_color != int(prev_color)
            assert expected[cr, c1 + 1] == Color.BLACK
        elif obstacle["direction"] == "left":
            near_color = strip[-1]
            prev_color = expected[cr, c1 + 1]
            assert near_color != int(prev_color)
            assert expected[cr, c0 - 1] == Color.BLACK
        elif obstacle["direction"] == "down":
            near_color = strip[0]
            prev_color = expected[r0 - 1, cc]
            assert near_color != int(prev_color)
            assert expected[r1 + 1, cc] == Color.BLACK
        else:
            near_color = strip[-1]
            prev_color = expected[r1 + 1, cc]
            assert near_color != int(prev_color)
            assert expected[r0 - 1, cc] == Color.BLACK

    assert np.array_equal(main(grid), expected)


def _attempt_generate() -> np.ndarray | None:
    center_color = random.choice(Color.NOT_BLACK)
    horiz_period = random.choice([2, 3])
    vert_period = random.choice([2, 3])
    horiz_seq = _choose_sequence(horiz_period, center_color)
    vert_seq = _choose_sequence(vert_period, center_color)

    horiz_extent = random.randint(2 * horiz_period, 2 * horiz_period + 1)
    vert_extent = random.randint(2 * vert_period, 2 * vert_period + 1)

    min_h = 2 * (vert_extent + 4) + 1
    min_w = 2 * (horiz_extent + 5) + 1
    if min_h > 29 or min_w > 29:
        return None

    h_choices = [n for n in range(min_h, min(29, min_h + 4) + 1) if n % 2 == 1]
    w_choices = [n for n in range(min_w, min(29, min_w + 4) + 1) if n % 2 == 1]
    if not h_choices or not w_choices:
        return None
    h = random.choice(h_choices)
    w = random.choice(w_choices)
    center = (h // 2, w // 2)

    directions = random.sample(["left", "right"], k=1) + random.sample(["up", "down"], k=1)
    remaining = [d for d in DIRS if d not in directions]
    extra_count = random.choice([0, 1, 2])
    directions += random.sample(remaining, k=min(extra_count, len(remaining)))
    random.shuffle(directions)

    cross_bbox = (
        center[0] - vert_extent,
        center[1] - horiz_extent,
        center[0] + vert_extent,
        center[1] + horiz_extent,
    )
    occupied = [cross_bbox]

    cross_colors = set(horiz_seq) | set(vert_seq)
    obstacle_palette = [c for c in Color.NOT_BLACK if c not in cross_colors]
    if len(obstacle_palette) < len(directions):
        return None
    obstacle_colors = random.sample(obstacle_palette, k=len(directions))

    obstacles = []
    for direction, color in zip(directions, obstacle_colors):
        placed = None
        for bbox in _candidate_bboxes(direction, (h, w), center, horiz_extent, vert_extent):
            if any(_rectangles_touch(bbox, prior, pad=1) for prior in occupied):
                continue
            placed = {"direction": direction, "color": color, "bbox": bbox}
            occupied.append(bbox)
            obstacles.append(placed)
            break
        if placed is None:
            return None

    if all(obs["bbox"][3] - obs["bbox"][1] + 1 == 2 for obs in obstacles) and horiz_period == vert_period == 2:
        return None

    spec = {
        "shape": (h, w),
        "center": center,
        "horiz_seq": horiz_seq,
        "vert_seq": vert_seq,
        "horiz_extent": horiz_extent,
        "vert_extent": vert_extent,
        "obstacles": obstacles,
    }
    grid = _render_input(spec)
    expected = _oracle_output(spec, grid)
    _validate_spec(spec, grid, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phantom-period obstacle puzzle")
