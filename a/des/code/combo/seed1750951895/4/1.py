import numpy as np


BG = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _group_pixels(grid: np.ndarray) -> list[dict]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0, "expected foreground pixels"

    diff_counts: dict[int, int] = {}
    sum_counts: dict[int, int] = {}
    for r, c in coords:
        diff_counts[int(r - c)] = diff_counts.get(int(r - c), 0) + 1
        sum_counts[int(r + c)] = sum_counts.get(int(r + c), 0) + 1

    groups: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    for r, c in coords:
        d = int(r - c)
        s = int(r + c)
        on_main = diff_counts[d] > 1
        on_anti = sum_counts[s] > 1
        assert on_main != on_anti, "each pixel must belong unambiguously to one diagonal group"
        key = ("main", d) if on_main else ("anti", s)
        groups.setdefault(key, []).append((int(r), int(c), int(grid[r, c])))

    return [
        {"family": family, "value": value, "pixels": pixels}
        for (family, value), pixels in groups.items()
    ]


def _nearest_corner(group: dict, height: int, width: int) -> str:
    pixels = [(r, c) for r, c, _ in group["pixels"]]
    if group["family"] == "main":
        candidates = {
            "TL": min(r * r + c * c for r, c in pixels),
            "BR": min((height - 1 - r) ** 2 + (width - 1 - c) ** 2 for r, c in pixels),
        }
    else:
        candidates = {
            "TR": min(r * r + (width - 1 - c) ** 2 for r, c in pixels),
            "BL": min((height - 1 - r) ** 2 + c * c for r, c in pixels),
        }
    ordered = sorted(candidates.items(), key=lambda item: item[1])
    assert ordered[0][1] != ordered[1][1], "nearest corner must be unique"
    return ordered[0][0]


def _sort_pixels_from_corner(group: dict, corner: str) -> list[tuple[int, int, int]]:
    pixels = group["pixels"]
    if corner in {"TL", "TR"}:
        return sorted(pixels, key=lambda item: (item[0], item[1]))
    return sorted(pixels, key=lambda item: (-item[0], -item[1]))


def _target_cells(corner: str, length: int, height: int, width: int) -> list[tuple[int, int]]:
    if corner == "TL":
        cells = [(i, i) for i in range(length)]
    elif corner == "BR":
        cells = [(height - 1 - i, width - 1 - i) for i in range(length)]
    elif corner == "TR":
        cells = [(i, width - 1 - i) for i in range(length)]
    else:
        assert corner == "BL"
        cells = [(height - 1 - i, i) for i in range(length)]
    assert all(0 <= r < height and 0 <= c < width for r, c in cells)
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Group isolated pixels by diagonal, then compress each group into a corner-anchored diagonal segment."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    groups = _group_pixels(g)
    out = np.full_like(g, BG)
    painted: set[tuple[int, int]] = set()

    for group in groups:
        corner = _nearest_corner(group, height, width)
        ordered = _sort_pixels_from_corner(group, corner)
        target = _target_cells(corner, len(ordered), height, width)
        for (r, c), (_, _, color) in zip(target, ordered):
            assert (r, c) not in painted, "output segments must not overlap"
            painted.add((r, c))
            out[r, c] = color

    return out


def _line_coords(height: int, width: int, family: str, value: int) -> list[tuple[int, int]]:
    if family == "main":
        coords = []
        for r in range(height):
            c = r - value
            if 0 <= c < width:
                coords.append((r, c))
        return coords

    coords = []
    for r in range(height):
        c = value - r
        if 0 <= c < width:
            coords.append((r, c))
    return coords


def _sample_spaced_positions(length: int, count: int, from_start: bool) -> list[int] | None:
    if length < 2 * count - 1:
        return None

    def sample_forward() -> list[int]:
        positions: list[int] = []
        lo = 0
        for i in range(count):
            max_pos = length - 1 - 2 * (count - 1 - i)
            slack = min(2, max_pos - lo)
            pos = int(lo + RNG.integers(slack + 1))
            positions.append(pos)
            lo = pos + 2
        return positions

    if from_start:
        return sample_forward()

    forward = sample_forward()
    return [length - 1 - p for p in reversed(forward)]


def _choose_lengths(num_groups: int, max_total: int) -> list[int] | None:
    for _ in range(100):
        lengths = [int(RNG.integers(2, 5)) for _ in range(num_groups)]
        if sum(lengths) <= max_total:
            return lengths
    return None


def _min_corner_distance(pixels: list[tuple[int, int]], corner: str, height: int, width: int) -> int:
    if corner == "TL":
        return min(r * r + c * c for r, c in pixels)
    if corner == "BR":
        return min((height - 1 - r) ** 2 + (width - 1 - c) ** 2 for r, c in pixels)
    if corner == "TR":
        return min(r * r + (width - 1 - c) ** 2 for r, c in pixels)
    return min((height - 1 - r) ** 2 + c * c for r, c in pixels)


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(8, 15))
    width = int(RNG.integers(8, 15))
    if min(height, width) < 5:
        return None

    corners = ["TL", "TR", "BR", "BL"]
    num_groups = int(RNG.integers(2, 5))
    chosen_corners = list(RNG.choice(corners, size=num_groups, replace=False))
    lengths = _choose_lengths(num_groups, max_total=min(9, 2 * min(height, width) - 2))
    if lengths is None:
        return None

    plans = []
    for corner, count in zip(chosen_corners, lengths):
        if corner in {"TL", "BR"}:
            family = "main"
            values = list(range(-(width - 1), height))
        else:
            family = "anti"
            values = list(range(height + width - 1))
        RNG.shuffle(values)
        plans.append({"corner": corner, "family": family, "count": count, "values": values})

    pixels_by_group: list[list[tuple[int, int]]] = []
    used_coords: set[tuple[int, int]] = set()
    used_main: set[int] = set()
    used_anti: set[int] = set()

    for plan in plans:
        placed = None
        for value in plan["values"]:
            coords = _line_coords(height, width, plan["family"], value)
            positions = _sample_spaced_positions(
                len(coords),
                plan["count"],
                from_start=plan["corner"] in {"TL", "TR"},
            )
            if positions is None:
                continue
            pixels = [coords[idx] for idx in positions]

            if plan["family"] == "main":
                cross_values = [r + c for r, c in pixels]
                if value in used_main or len(set(cross_values)) != len(cross_values):
                    continue
                if any(s in used_anti for s in cross_values):
                    continue
            else:
                cross_values = [r - c for r, c in pixels]
                if value in used_anti or len(set(cross_values)) != len(cross_values):
                    continue
                if any(d in used_main for d in cross_values):
                    continue

            if any(p in used_coords for p in pixels):
                continue
            if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for r1, c1 in pixels for r2, c2 in used_coords):
                continue

            compat = ("TL", "BR") if plan["family"] == "main" else ("TR", "BL")
            distances = {
                corner: _min_corner_distance(pixels, corner, height, width)
                for corner in compat
            }
            if len(set(distances.values())) != 2:
                continue
            nearest = min(distances, key=distances.get)
            if nearest != plan["corner"]:
                continue

            placed = pixels
            if plan["family"] == "main":
                used_main.add(value)
                used_anti.update(cross_values)
            else:
                used_anti.add(value)
                used_main.update(cross_values)
            used_coords.update(pixels)
            break

        if placed is None:
            return None
        pixels_by_group.append(placed)

    total_pixels = sum(len(group) for group in pixels_by_group)
    colors = list(RNG.choice(COLOR_POOL, size=total_pixels, replace=False))
    grid = np.full((height, width), BG, dtype=int)

    color_idx = 0
    for group_pixels in pixels_by_group:
        for r, c in group_pixels:
            grid[r, c] = colors[color_idx]
            color_idx += 1

    groups = _group_pixels(grid)
    expected_group_count = len(plans)
    if len(groups) != expected_group_count:
        return None

    out = main(grid)
    if np.count_nonzero(out) != total_pixels:
        return None

    occupied = set()
    for group in groups:
        corner = _nearest_corner(group, height, width)
        cells = _target_cells(corner, len(group["pixels"]), height, width)
        if any(cell in occupied for cell in cells):
            return None
        occupied.update(cells)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
