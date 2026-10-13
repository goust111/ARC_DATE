import numpy as np


BACKGROUND = 0
COLOR_VALUES = tuple(range(1, 10))
EVEN_SIZES = tuple(range(8, 29, 2))
MAX_GENERATION_ATTEMPTS = 400


def _ring_coords(height: int, width: int, depth: int) -> list[tuple[int, int]]:
    top = int(depth)
    left = int(depth)
    bottom = int(height) - int(depth) - 1
    right = int(width) - int(depth) - 1
    if top > bottom or left > right:
        return []

    if top == bottom:
        return [(top, c) for c in range(left, right + 1)]
    if left == right:
        return [(r, left) for r in range(top, bottom + 1)]

    coords: list[tuple[int, int]] = []
    coords.extend((top, c) for c in range(left, right + 1))
    coords.extend((r, right) for r in range(top + 1, bottom))
    coords.extend((bottom, c) for c in range(right, left - 1, -1))
    coords.extend((r, left) for r in range(bottom - 1, top, -1))
    return coords


def _cyclic_distance(a: int, b: int, length: int) -> int:
    diff = abs(int(a) - int(b))
    return min(diff, int(length) - diff)


def _solve_ring(
    coords: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    ring_length = len(coords)
    if len(markers) == 1:
        color = int(markers[0][1])
        return {coord: color for coord in coords}

    assignment: dict[tuple[int, int], int] = {}
    marker_indices = [int(idx) for idx, _ in markers]
    marker_colors = [int(color) for _, color in markers]

    for pos, coord in enumerate(coords):
        distances = [_cyclic_distance(pos, idx, ring_length) for idx in marker_indices]
        best = min(distances)
        winner_ids = [i for i, dist in enumerate(distances) if dist == best]
        winner_colors = {marker_colors[i] for i in winner_ids}
        assert len(winner_colors) == 1, "ambiguous nearest marker on ring"
        assignment[coord] = next(iter(winner_colors))
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker across its equal-depth rectangular ring, splitting shared rings by nearest arc distance.

    Input structure:
    - A black background grid contains isolated non-black marker pixels.
    - Each cell belongs to the rectangular ring determined by its minimum distance to the grid border.
    - Some rings contain one marker, some contain several differently colored markers, and some contain none.

    Rule:
    - For every marked ring, recolor the entire ring.
    - A ring with one marker becomes uniformly that marker's color.
    - A ring with multiple markers is partitioned along the ring cycle: each cell takes the color of the marker
      reachable by the shortest walk along that same ring (generator guarantees no conflicting ties).
    - Unmarked rings remain black.

    Key insight:
    - The propagation path is not radial or straight; it is constrained to the closed equal-depth ring, so
      multiple markers on one ring create a circular Voronoi split along perimeter arc length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    out = np.full_like(g, BACKGROUND)

    for depth in range((min(height, width) + 1) // 2):
        coords = _ring_coords(height, width, depth)
        if not coords:
            break
        markers = [
            (idx, int(g[r, c]))
            for idx, (r, c) in enumerate(coords)
            if int(g[r, c]) != BACKGROUND
        ]
        if not markers:
            continue
        for (r, c), color in _solve_ring(coords, markers).items():
            out[r, c] = int(color)

    return out


def _ring_lengths(height: int, width: int) -> dict[int, int]:
    return {
        depth: len(_ring_coords(height, width, depth))
        for depth in range(min(height, width) // 2)
    }


def _odd_gap_sequence(total: int, parts: int, min_gap: int, rng: np.random.Generator) -> list[int] | None:
    remainder = int(total) - int(parts) * int(min_gap)
    if remainder < 0 or remainder % 2 != 0:
        return None

    extras = [0] * int(parts)
    for _ in range(remainder // 2):
        extras[int(rng.integers(0, parts))] += 1

    gaps = [int(min_gap + 2 * extra) for extra in extras]
    rng.shuffle(gaps)
    return gaps


def _sample_marker_indices(ring_length: int, count: int, rng: np.random.Generator) -> list[int] | None:
    ring_length = int(ring_length)
    count = int(count)
    if count == 1:
        return [int(rng.integers(0, ring_length))]

    if count == 2:
        candidates = [gap for gap in range(3, ring_length - 2, 2) if ring_length - gap >= 3]
        if not candidates:
            return None
        start = int(rng.integers(0, ring_length))
        gap = int(rng.choice(candidates))
        return sorted({start, (start + gap) % ring_length})

    if count == 4:
        gaps = _odd_gap_sequence(ring_length, 4, 3, rng)
        if gaps is None:
            return None
        start = int(rng.integers(0, ring_length))
        indices = [start]
        pos = start
        for gap in gaps[:-1]:
            pos = (pos + int(gap)) % ring_length
            indices.append(pos)
        if len(set(indices)) != 4:
            return None
        return sorted(indices)

    return None


def _cyclic_gaps(indices: list[int], ring_length: int) -> list[int]:
    ordered = sorted(int(i) for i in indices)
    gaps = []
    for idx, current in enumerate(ordered):
        nxt = ordered[(idx + 1) % len(ordered)]
        if idx + 1 == len(ordered):
            nxt += int(ring_length)
        gaps.append(int(nxt - current))
    return gaps


def _manhattan_assignment_differs(
    coords: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> bool:
    ring_assignment = _solve_ring(coords, markers)
    marker_cells = [coords[idx] for idx, _ in markers]
    marker_colors = [int(color) for _, color in markers]

    for coord in coords:
        mr, mc = coord
        distances = [abs(mr - sr) + abs(mc - sc) for sr, sc in marker_cells]
        best = min(distances)
        winner_colors = {
            marker_colors[i]
            for i, dist in enumerate(distances)
            if dist == best
        }
        if len(winner_colors) != 1:
            continue
        if ring_assignment[coord] != next(iter(winner_colors)):
            return True
    return False


def _sample_case(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(EVEN_SIZES))
    width = int(rng.choice(EVEN_SIZES))
    ring_lengths = _ring_lengths(height, width)
    ring_depths = list(ring_lengths)
    if len(ring_depths) < 4:
        return None

    multi_depth_candidates = [depth for depth, length in ring_lengths.items() if length >= 6]
    if not multi_depth_candidates:
        return None

    num_marked = int(rng.integers(2, min(4, len(ring_depths) - 1) + 1))
    multi_depth = int(rng.choice(multi_depth_candidates))

    remaining_for_single = [depth for depth in ring_depths if depth != multi_depth]
    if not remaining_for_single:
        return None
    single_depth = int(rng.choice(remaining_for_single))

    chosen = {multi_depth, single_depth}
    extra_pool = [depth for depth in ring_depths if depth not in chosen]
    rng.shuffle(extra_pool)
    while len(chosen) < num_marked:
        chosen.add(int(extra_pool.pop()))

    marked_depths = sorted(chosen)
    ring_marker_counts: dict[int, int] = {}
    multi_ring_length = ring_lengths[multi_depth]
    if multi_ring_length >= 12 and float(rng.random()) < 0.45:
        ring_marker_counts[multi_depth] = 4
    else:
        ring_marker_counts[multi_depth] = 2
    ring_marker_counts[single_depth] = 1

    for depth in marked_depths:
        if depth in ring_marker_counts:
            continue
        length = ring_lengths[depth]
        if length >= 12 and float(rng.random()) < 0.25:
            ring_marker_counts[depth] = 4
        elif length >= 6 and float(rng.random()) < 0.55:
            ring_marker_counts[depth] = 2
        else:
            ring_marker_counts[depth] = 1

    total_markers = sum(ring_marker_counts.values())
    if total_markers > len(COLOR_VALUES):
        return None

    colors = list(rng.choice(np.array(COLOR_VALUES, dtype=int), size=total_markers, replace=False))
    color_cursor = 0

    grid = np.full((height, width), BACKGROUND, dtype=int)
    ring_specs: dict[int, dict[str, object]] = {}

    for depth in marked_depths:
        coords = _ring_coords(height, width, depth)
        count = ring_marker_counts[depth]
        indices = _sample_marker_indices(len(coords), count, rng)
        if indices is None or len(indices) != count:
            return None

        marker_colors = [int(colors[color_cursor + i]) for i in range(count)]
        color_cursor += count
        for idx, color in zip(indices, marker_colors):
            r, c = coords[int(idx)]
            grid[r, c] = int(color)

        ring_specs[depth] = {
            "coords": coords,
            "indices": indices,
            "colors": marker_colors,
        }

    multi_spec = ring_specs[multi_depth]
    if not _manhattan_assignment_differs(
        multi_spec["coords"],  # type: ignore[arg-type]
        list(zip(multi_spec["indices"], multi_spec["colors"])),  # type: ignore[arg-type]
    ):
        return None

    return grid


def _validate_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if any(dim % 2 != 0 for dim in g.shape):
        return False

    height, width = map(int, g.shape)
    ring_lengths = _ring_lengths(height, width)
    marked_rings = []
    single_seen = False
    multi_seen = False
    arc_not_manhattan = False

    out = main(g)
    if out.shape != g.shape or np.array_equal(out, g):
        return False

    new_pixels = int(np.count_nonzero(out)) - int(np.count_nonzero(g))
    if new_pixels <= 0:
        return False

    for depth, ring_length in ring_lengths.items():
        coords = _ring_coords(height, width, depth)
        markers = [
            (idx, int(g[r, c]))
            for idx, (r, c) in enumerate(coords)
            if int(g[r, c]) != BACKGROUND
        ]
        ring_output_colors = {int(out[r, c]) for r, c in coords}

        if not markers:
            if ring_output_colors != {BACKGROUND}:
                return False
            continue

        marked_rings.append(depth)
        marker_colors = {int(color) for _, color in markers}
        if not ring_output_colors.issubset(marker_colors):
            return False

        if len(markers) == 1:
            single_seen = True
            if len(ring_output_colors) != 1:
                return False
        else:
            multi_seen = True
            gaps = _cyclic_gaps([idx for idx, _ in markers], ring_length)
            if any(gap % 2 == 0 for gap in gaps):
                return False
            if len(ring_output_colors) < 2:
                return False
            if _manhattan_assignment_differs(coords, markers):
                arc_not_manhattan = True

    if len(marked_rings) >= len(ring_lengths):
        return False
    return single_seen and multi_seen and arc_not_manhattan


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _sample_case(rng)
        if candidate is not None and _validate_case(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid ring-extension input")
