import hashlib
import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 1
SEED_COLORS = (2, 3, 4)

_GEN_RETRIES = 500
_SEEN_OUTPUTS: set[str] = set()


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox):
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains_bbox(parent, child):
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    return pr0 < cr0 and pc0 < cc0 and cr1 < pr1 and cc1 < pc1


def _seed_inside_bbox(seed, bbox):
    r, c = seed
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _frame_interior_radius(bbox):
    r0, c0, r1, c1 = bbox
    return min((r1 - r0 - 2) // 2, (c1 - c0 - 2) // 2)


def _frame_center(bbox):
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _is_rectangular_frame(coords):
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return False
    coord_set = set(coords)
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return coord_set == expected


def _components_of_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_groups(grid):
    blue_components = _components_of_color(grid, FRAME_COLOR)
    assert 2 <= len(blue_components) <= 9, "expected several blue frame components"
    frames = []
    for comp in blue_components:
        assert _is_rectangular_frame(comp), "every blue component must be a hollow odd-sized rectangular frame"
        bbox = _bbox_from_coords(comp)
        frames.append({"bbox": bbox, "area": _bbox_area(bbox)})

    seed_coords = np.argwhere((grid != BACKGROUND) & (grid != FRAME_COLOR))
    assert 2 <= len(seed_coords) <= 3, "expected 2-3 seed pixels"
    seed_colors = [int(grid[r, c]) for r, c in seed_coords]
    assert len(set(seed_colors)) == len(seed_colors), "seed colors must be distinct"
    assert set(seed_colors) <= set(SEED_COLORS), "unexpected seed color"

    groups = []
    frame_owner_counts = {frame["bbox"]: 0 for frame in frames}
    for rr, cc in seed_coords:
        seed = (int(rr), int(cc))
        color = int(grid[seed])
        containing = [frame["bbox"] for frame in frames if _seed_inside_bbox(seed, frame["bbox"])]
        assert containing, "each seed must lie inside at least one frame"
        ordered = sorted(containing, key=_bbox_area)
        for bbox in containing:
            frame_owner_counts[bbox] += 1
        for child, parent in zip(ordered, ordered[1:]):
            assert _strictly_contains_bbox(parent, child), "containing frames must form one strict nesting chain"
        inner_bbox = ordered[0]
        assert _frame_center(inner_bbox) == seed, "seed must be the center of the innermost frame"
        groups.append(
            {
                "seed": seed,
                "color": color,
                "depth": len(ordered),
                "inner_bbox": inner_bbox,
                "frames": tuple(ordered),
            }
        )

    assert len({group["depth"] for group in groups}) >= 2, "group depths must not all be identical"
    for bbox, count in frame_owner_counts.items():
        assert count == 1, f"frame {bbox} must belong to exactly one seed/group"
    return groups


def _paint_group(out, seed, depth, color, inner_bbox, radius_override=None, metric="manhattan", even_black=True):
    r0, c0, r1, c1 = inner_bbox
    sr, sc = seed
    radius = depth if radius_override is None else radius_override
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if metric == "manhattan":
                dist = abs(r - sr) + abs(c - sc)
            elif metric == "chebyshev":
                dist = max(abs(r - sr), abs(c - sc))
            else:
                raise ValueError(metric)
            if dist > radius:
                continue
            if dist == 0 or dist % 2 == 1 or not even_black:
                out[r, c] = color
            else:
                out[r, c] = BACKGROUND


def _render_output(shape, groups):
    out = np.full(shape, BACKGROUND, dtype=int)
    for group in sorted(groups, key=lambda g: (g["depth"], g["seed"][0], g["seed"][1])):
        _paint_group(out, group["seed"], group["depth"], group["color"], group["inner_bbox"])
    return out


def _naive_max_fill(shape, groups):
    out = np.full(shape, BACKGROUND, dtype=int)
    for group in sorted(groups, key=lambda g: (g["depth"], g["seed"][0], g["seed"][1])):
        radius = _frame_interior_radius(group["inner_bbox"])
        _paint_group(out, group["seed"], group["depth"], group["color"], group["inner_bbox"], radius_override=radius)
    return out


def _naive_square_fill(shape, groups):
    out = np.full(shape, BACKGROUND, dtype=int)
    for group in sorted(groups, key=lambda g: (g["depth"], g["seed"][0], g["seed"][1])):
        _paint_group(out, group["seed"], group["depth"], group["color"], group["inner_bbox"], metric="chebyshev")
    return out


def _naive_no_erase(shape, groups):
    out = np.full(shape, BACKGROUND, dtype=int)
    for group in sorted(groups, key=lambda g: (g["depth"], g["seed"][0], g["seed"][1])):
        _paint_group(out, group["seed"], group["depth"], group["color"], group["inner_bbox"], even_black=False)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the blue frame nests and keep only a seed-centered alternating diamond whose radius equals each nest depth.

    Input structure:
    - The grid is black except for 2-3 disjoint groups of blue hollow rectangular frames.
    - Each group contains 1-3 strictly nested odd-sized blue rectangles.
    - The center cell of each innermost frame holds one distinct seed pixel in red, green, or yellow.

    Rule:
    - For each group, count how many blue frames contain its seed; that nesting count is the expansion depth.
    - Starting from the seed, paint all cells inside the innermost frame whose Manhattan distance from the seed is at most that depth.
    - Distance 0 and every odd layer keep the seed color, while every positive even layer is painted black.
    - Remove all blue frame pixels; if two painted regions overlap, the deeper group takes priority.

    Key insight:
    - The blue rectangles are not output objects: they jointly encode both the diamond radius (nesting depth) and the legal drawing region (the innermost frame interior).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    groups = _parse_groups(g)
    return _render_output(g.shape, groups)


def _rand_choice(rng, seq):
    return seq[rng.randrange(len(seq))]


def _sample_depths(rng, group_count):
    if group_count == 2:
        options = [(1, 2), (1, 3), (2, 3)]
    else:
        options = [
            (1, 2, 3),
            (1, 1, 2),
            (1, 1, 3),
            (1, 2, 2),
            (1, 3, 3),
            (2, 2, 3),
            (2, 3, 3),
        ]
    depths = list(_rand_choice(rng, options))
    rng.shuffle(depths)
    return depths


def _make_local_group(rng, depth, color):
    cap_r = depth + rng.randrange(0, 2)
    cap_c = depth + rng.randrange(0, 2)
    inner_h = 2 * cap_r + 3
    inner_w = 2 * cap_c + 3

    sizes = [(inner_h, inner_w)]
    for _ in range(depth - 1):
        prev_h, prev_w = sizes[-1]
        sizes.append((prev_h + 2 * _rand_choice(rng, (2, 3)), prev_w + 2 * _rand_choice(rng, (2, 3))))

    outer_h, outer_w = sizes[-1]
    center = (outer_h // 2, outer_w // 2)
    frames = []
    for h, w in sizes:
        r0 = center[0] - h // 2
        c0 = center[1] - w // 2
        frames.append((r0, c0, r0 + h - 1, c0 + w - 1))

    return {
        "depth": depth,
        "color": color,
        "patch_shape": (outer_h, outer_w),
        "seed": center,
        "frames": tuple(frames),
        "inner_bbox": frames[0],
        "slack": _frame_interior_radius(frames[0]) - depth,
    }


def _boxes_are_far_enough(a, b, gap):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_groups(rng, specs):
    max_h = max(spec["patch_shape"][0] for spec in specs)
    max_w = max(spec["patch_shape"][1] for spec in specs)
    h_min = max(18, max_h + 2)
    w_min = max(18, max_w + 2)
    if h_min > 30 or w_min > 30:
        return None

    for _ in range(200):
        height = rng.randint(h_min, 30)
        width = rng.randint(w_min, 30)
        occupied = []
        placed = []
        for spec in sorted(specs, key=lambda s: s["patch_shape"][0] * s["patch_shape"][1], reverse=True):
            ph, pw = spec["patch_shape"]
            if ph + 2 > height or pw + 2 > width:
                placed = None
                break
            found = False
            for _ in range(200):
                top = rng.randint(1, height - ph - 1)
                left = rng.randint(1, width - pw - 1)
                bbox = (top, left, top + ph - 1, left + pw - 1)
                if all(_boxes_are_far_enough(bbox, other, 2) for other in occupied):
                    found = True
                    occupied.append(bbox)
                    placed.append((spec, top, left))
                    break
            if not found:
                placed = None
                break
        if placed is not None:
            return height, width, placed
    return None


def _translate_group(local_group, top, left):
    return {
        "depth": local_group["depth"],
        "color": local_group["color"],
        "seed": (local_group["seed"][0] + top, local_group["seed"][1] + left),
        "frames": tuple(
            (r0 + top, c0 + left, r1 + top, c1 + left)
            for (r0, c0, r1, c1) in local_group["frames"]
        ),
        "inner_bbox": tuple(v + delta for v, delta in zip(local_group["inner_bbox"], (top, left, top, left))),
        "slack": local_group["slack"],
    }


def _draw_input(shape, groups):
    grid = np.full(shape, BACKGROUND, dtype=int)
    for group in groups:
        for r0, c0, r1, c1 in group["frames"]:
            grid[r0, c0 : c1 + 1] = FRAME_COLOR
            grid[r1, c0 : c1 + 1] = FRAME_COLOR
            grid[r0 : r1 + 1, c0] = FRAME_COLOR
            grid[r0 : r1 + 1, c1] = FRAME_COLOR
        sr, sc = group["seed"]
        grid[sr, sc] = group["color"]
    return grid


def _signature(arr):
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _validate_generated(grid, groups):
    parsed = _parse_groups(grid)
    assert len(parsed) == len(groups)
    oracle = _render_output(grid.shape, groups)
    solved = main(grid)
    assert np.array_equal(solved, oracle), "main disagrees with constructive oracle"

    parsed_key = sorted((g["seed"], g["color"], g["depth"], g["inner_bbox"]) for g in parsed)
    built_key = sorted((g["seed"], g["color"], g["depth"], g["inner_bbox"]) for g in groups)
    assert parsed_key == built_key, "parser failed to recover constructed groups"

    assert any(group["slack"] > 0 for group in groups), "need at least one roomy innermost frame"
    assert any(group["depth"] >= 2 for group in groups), "need an example with visible black even layers"
    assert not np.array_equal(oracle, _naive_max_fill(grid.shape, groups)), "radius shortcut accidentally works"
    assert not np.array_equal(oracle, _naive_square_fill(grid.shape, groups)), "Chebyshev shortcut accidentally works"
    assert not np.array_equal(oracle, _naive_no_erase(grid.shape, groups)), "ignoring black erasure accidentally works"


def _gen_attempt(rng):
    group_count = 2 if rng.random() < 0.55 else 3
    depths = _sample_depths(rng, group_count)
    colors = rng.sample(list(SEED_COLORS), group_count)
    local_specs = [_make_local_group(rng, depth, color) for depth, color in zip(depths, colors)]
    if not any(spec["slack"] > 0 for spec in local_specs):
        return None
    placement = _place_groups(rng, local_specs)
    if placement is None:
        return None

    height, width, placed = placement
    groups = [_translate_group(spec, top, left) for spec, top, left in placed]
    oracle = _render_output((height, width), groups)
    if np.array_equal(oracle, _naive_max_fill((height, width), groups)):
        return None
    if np.array_equal(oracle, _naive_square_fill((height, width), groups)):
        return None
    if np.array_equal(oracle, _naive_no_erase((height, width), groups)):
        return None
    grid = _draw_input((height, width), groups)
    _validate_generated(grid, groups)

    out_sig = _signature(oracle)
    if out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame diamond sample")
