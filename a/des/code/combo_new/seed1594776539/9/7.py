from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _discrete_median(values: list[int]) -> int:
    ordered = sorted(int(v) for v in values)
    return int(ordered[len(ordered) // 2])


def _scan_order_colors(grid: np.ndarray, bg: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for value in grid.ravel():
        color = int(value)
        if color != int(bg) and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _extract_groups(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for color in _scan_order_colors(grid, int(bg)):
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        groups.append(
            {
                "color": int(color),
                "count": int(len(coords)),
                "center": (_discrete_median(rows), _discrete_median(cols)),
                "coords": coords,
                "first": min(coords),
            }
        )
    return groups


def _draw_diamond(out: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    height, width = out.shape
    cr, cc = map(int, center)
    for dr in range(-int(radius), int(radius) + 1):
        rr = int(cr) + int(dr)
        if not (0 <= rr < height):
            continue
        span = int(radius) - abs(int(dr))
        c0 = max(0, int(cc) - int(span))
        c1 = min(width - 1, int(cc) + int(span))
        out[int(rr), int(c0) : int(c1) + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each color's scattered singleton pixels with a same-color concentric diamond centered on that cluster.

    Input structure:
    - The background is a dominant green field.
    - Several other colors appear only as isolated singleton pixels.
    - Pixels of the same color form a loose cluster, and different color clusters are separated in the input.

    Rule:
    - Group all non-background pixels by color and count each group's size N.
    - For each group, sort its row coordinates and column coordinates separately and take the discrete median on each axis as the visual center.
    - Clear the original singleton pixels, then draw that color's concentric diamond layers centered there: layer k contains cells whose Manhattan distance from the center is exactly k, for k = 0..N-1.
    - If diamonds from different colors overlap, later colors in row-major first-appearance order overwrite earlier ones.

    Key insight:
    - Each color cluster encodes three independent attributes at once: color selects the paint, pixel count sets the diamond radius, and the cluster's median position sets the diamond center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    out = np.full_like(g, int(bg))
    for group in _extract_groups(g, int(bg)):
        _draw_diamond(
            out,
            center=group["center"],
            radius=int(group["count"]) - 1,
            color=int(group["color"]),
        )
    return out


# --------------- generator ---------------

_BG = 3
_PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
_GEN_RETRIES = 400
_PLACEMENT_TRIES = 300
_OVERLAP_PROB = 0.5

_BASE_PATTERNS: dict[int, list[tuple[tuple[int, int], ...]]] = {
    2: [
        ((-2, 0), (0, -2)),
        ((-3, -1), (0, 0)),
        ((-4, 0), (0, 0)),
    ],
    3: [
        ((-2, 0), (0, -2), (2, 2)),
        ((-3, 0), (0, -3), (1, 1)),
        ((-2, -2), (0, 0), (2, 2)),
    ],
    4: [
        ((-3, 0), (-1, -2), (0, 2), (2, 0)),
        ((-4, 0), (-2, -2), (0, 0), (1, 2)),
        ((-4, -1), (-2, 1), (0, -3), (1, 0)),
    ],
    5: [
        ((-3, 0), (-1, -2), (0, 2), (2, 0), (0, -4)),
        ((-4, 0), (-2, -2), (0, 0), (1, 2), (3, -1)),
        ((-4, -1), (-2, 2), (0, -3), (0, 0), (2, 1)),
    ],
}


def _transform_offsets(offsets: tuple[tuple[int, int], ...], transform_id: int) -> tuple[tuple[int, int], ...]:
    transformed: list[tuple[int, int]] = []
    for r, c in offsets:
        if transform_id == 0:
            rr, cc = r, c
        elif transform_id == 1:
            rr, cc = r, -c
        elif transform_id == 2:
            rr, cc = -r, c
        elif transform_id == 3:
            rr, cc = -r, -c
        elif transform_id == 4:
            rr, cc = c, r
        elif transform_id == 5:
            rr, cc = c, -r
        elif transform_id == 6:
            rr, cc = -c, r
        else:
            rr, cc = -c, -r
        transformed.append((int(rr), int(cc)))
    return tuple(sorted(transformed))


def _offset_center(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in offsets]
    cols = [c for _r, c in offsets]
    return _discrete_median(rows), _discrete_median(cols)


def _points_are_nonadjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
                return False
    return True


def _build_pattern_variants() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    variants: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for count, bases in _BASE_PATTERNS.items():
        seen: set[tuple[tuple[int, int], ...]] = set()
        good: list[tuple[tuple[int, int], ...]] = []
        for base in bases:
            for transform_id in range(8):
                variant = _transform_offsets(base, int(transform_id))
                if variant in seen:
                    continue
                if _offset_center(variant) != (0, 0):
                    continue
                if not _points_are_nonadjacent(list(variant)):
                    continue
                seen.add(variant)
                good.append(variant)
        assert good, f"no usable patterns for count {count}"
        variants[int(count)] = good
    return variants


_PATTERN_VARIANTS = _build_pattern_variants()


def _seed_coords(center: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    return [(int(cr) + int(dr), int(cc) + int(dc)) for dr, dc in offsets]


def _coords_in_bounds(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    return all(0 <= int(r) < height and 0 <= int(c) < width for r, c in coords)


def _seeds_clear(existing: set[tuple[int, int]], coords: list[tuple[int, int]]) -> bool:
    for r, c in coords:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if (int(rr), int(cc)) in existing:
                    return False
    return True


def _diamond_overlaps(
    center_a: tuple[int, int], radius_a: int, center_b: tuple[int, int], radius_b: int
) -> bool:
    return abs(int(center_a[0]) - int(center_b[0])) + abs(int(center_a[1]) - int(center_b[1])) <= int(radius_a) + int(radius_b)


def _margin_for(spec: dict[str, object]) -> int:
    radius = int(spec["radius"])
    extent = int(spec["extent"])
    return max(int(radius), int(extent))


def _random_center(height: int, width: int, spec: dict[str, object]) -> tuple[int, int] | None:
    margin = _margin_for(spec)
    if height <= 2 * margin or width <= 2 * margin:
        return None
    return (
        random.randint(int(margin), int(height - 1 - margin)),
        random.randint(int(margin), int(width - 1 - margin)),
    )


def _make_spec(color: int, count: int) -> dict[str, object]:
    offsets = random.choice(_PATTERN_VARIANTS[int(count)])
    extent = max(max(abs(int(r)), abs(int(c))) for r, c in offsets)
    return {
        "color": int(color),
        "count": int(count),
        "radius": int(count) - 1,
        "offsets": offsets,
        "extent": int(extent),
    }


def _place_group(
    height: int,
    width: int,
    spec: dict[str, object],
    placed: list[dict[str, object]],
    forbid_overlap: bool,
) -> dict[str, object] | None:
    used = {coord for group in placed for coord in group["coords"]}
    for _ in range(_PLACEMENT_TRIES):
        center = _random_center(height, width, spec)
        if center is None:
            return None
        coords = _seed_coords(center, spec["offsets"])
        if not _coords_in_bounds(coords, height, width):
            continue
        if not _seeds_clear(used, coords):
            continue
        if any(abs(int(center[0]) - int(other["center"][0])) + abs(int(center[1]) - int(other["center"][1])) < 4 for other in placed):
            continue
        if forbid_overlap and any(
            _diamond_overlaps(center, int(spec["radius"]), other["center"], int(other["radius"]))
            for other in placed
        ):
            continue
        group = dict(spec)
        group["center"] = center
        group["coords"] = coords
        return group
    return None


def _place_overlap_pair(
    height: int,
    width: int,
    spec_a: dict[str, object],
    spec_b: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]] | None:
    min_distance = 4
    max_distance = int(spec_a["radius"]) + int(spec_b["radius"])
    deltas = [
        (dr, dc)
        for dr in range(-max_distance, max_distance + 1)
        for dc in range(-max_distance, max_distance + 1)
        if (dr != 0 or dc != 0)
        and min_distance <= abs(int(dr)) + abs(int(dc)) <= max_distance
    ]
    random.shuffle(deltas)

    for _ in range(_PLACEMENT_TRIES):
        center_a = _random_center(height, width, spec_a)
        if center_a is None:
            return None
        coords_a = _seed_coords(center_a, spec_a["offsets"])
        if not _coords_in_bounds(coords_a, height, width):
            continue
        used = set(coords_a)
        for dr, dc in deltas:
            center_b = (int(center_a[0]) + int(dr), int(center_a[1]) + int(dc))
            coords_b = _seed_coords(center_b, spec_b["offsets"])
            if not _coords_in_bounds(coords_b, height, width):
                continue
            if not _seeds_clear(used, coords_b):
                continue
            group_a = dict(spec_a)
            group_a["center"] = center_a
            group_a["coords"] = coords_a
            group_b = dict(spec_b)
            group_b["center"] = center_b
            group_b["coords"] = coords_b
            return group_a, group_b
    return None


def _render_input(height: int, width: int, groups: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), _BG, dtype=int)
    for group in groups:
        for r, c in group["coords"]:
            grid[int(r), int(c)] = int(group["color"])
    return grid


def _group_pairs(groups: list[dict[str, object]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, left in enumerate(groups):
        for j in range(i + 1, len(groups)):
            right = groups[j]
            if _diamond_overlaps(left["center"], int(left["radius"]), right["center"], int(right["radius"])):
                pairs.append((int(i), int(j)))
    return pairs


def _visible_colors(grid: np.ndarray, colors: list[int]) -> bool:
    return all(bool(np.any(grid == int(color))) for color in colors)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)

    max_groups = 4 if min(height, width) < 24 else 5
    n_groups = random.randint(2, max_groups)
    require_overlap = random.random() < _OVERLAP_PROB

    colors = random.sample(list(_PALETTE), n_groups)
    counts = [random.randint(2, 5) for _ in range(n_groups)]

    overlap_pair: tuple[int, int] | None = None
    if require_overlap:
        overlap_pair = tuple(sorted(random.sample(range(n_groups), 2)))
        counts[overlap_pair[0]] = random.randint(4, 5)
        counts[overlap_pair[1]] = random.randint(4, 5)

    specs = [_make_spec(int(color), int(count)) for color, count in zip(colors, counts)]
    groups: list[dict[str, object]] = []

    if overlap_pair is not None:
        pair = _place_overlap_pair(
            height,
            width,
            specs[overlap_pair[0]],
            specs[overlap_pair[1]],
        )
        if pair is None:
            return None
        temp = {overlap_pair[0]: pair[0], overlap_pair[1]: pair[1]}
        for idx, group in temp.items():
            group["index"] = int(idx)
            groups.append(group)

    for idx, spec in enumerate(specs):
        if overlap_pair is not None and idx in overlap_pair:
            continue
        group = _place_group(
            height=height,
            width=width,
            spec=spec,
            placed=groups,
            forbid_overlap=True,
        )
        if group is None:
            return None
        group["index"] = int(idx)
        groups.append(group)

    groups.sort(key=lambda group: int(group["index"]))
    if not all(_points_are_nonadjacent(list(group["coords"])) for group in groups):
        return None

    overlap_pairs = _group_pairs(groups)
    if require_overlap:
        assert overlap_pair is not None
        if overlap_pairs != [overlap_pair]:
            return None
    elif overlap_pairs:
        return None

    inp = _render_input(height, width, groups)
    out = main(inp)
    if np.array_equal(inp, out):
        return None

    group_colors = [int(group["color"]) for group in groups]
    if not _visible_colors(out, group_colors):
        return None

    extracted = _extract_groups(inp, _BG)
    if sorted(int(group["count"]) for group in extracted) != sorted(counts):
        return None
    if any(int(group["count"]) < 2 or int(group["count"]) > 5 for group in extracted):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {_GEN_RETRIES} attempts")
