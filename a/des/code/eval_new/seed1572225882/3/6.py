import numpy as np


CW_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
CCW_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)


def _nearest_edge_distance(r: int, c: int, h: int, w: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _trace_spiral(r: int, c: int, segments: int, clockwise: bool) -> list[tuple[int, int]]:
    dirs = CW_DIRS if clockwise else CCW_DIRS
    path: list[tuple[int, int]] = []
    cr, cc = r, c
    for segment_len in range(1, segments + 1):
        dr, dc = dirs[(segment_len - 1) % 4]
        for _ in range(segment_len):
            cr += dr
            cc += dc
            path.append((cr, cc))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Grow a parity-controlled spiral arm from every colored seed pixel.

    Input structure:
    - Black background with only singleton colored seed pixels.
    - The non-black pixels use 2-4 colors, with each color appearing on several seeds.
    - No other foreground shapes or markers are present.

    Rule:
    - Group seed pixels by color and count the seeds in each color group.
    - Odd-sized color groups draw clockwise spirals: right 1, down 2, left 3, up 4, then repeat directions.
    - Even-sized color groups draw counterclockwise spirals: left 1, down 2, right 3, up 4, then repeat directions.
    - For each seed, let L be its minimum distance to any grid edge; draw exactly L spiral segments from that seed.
    - Paint visited cells with the seed color only when the visited cell is in bounds and still black; out-of-bounds or occupied cells are skipped while the spiral path keeps advancing.

    Key insight:
    - The spiral has two independent controls: color-group parity chooses the handedness, and each seed's nearest-edge distance chooses the number of segments.
    """

    out = grid.copy()
    colors, counts = np.unique(grid[grid != 0], return_counts=True)
    color_counts = {int(color): int(count) for color, count in zip(colors, counts)}
    h, w = grid.shape

    for r, c in np.argwhere(grid != 0):
        color = int(grid[r, c])
        clockwise = color_counts[color] % 2 == 1
        segments = _nearest_edge_distance(int(r), int(c), h, w)
        for rr, cc in _trace_spiral(int(r), int(c), segments, clockwise):
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                out[rr, cc] = color

    return out


def _pick_seed_positions(
    rng: np.random.Generator, h: int, w: int, total: int, spacing: int = 2
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(candidates)

    picks: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - pr), abs(c - pc)) >= spacing for pr, pc in picks):
            picks.append((r, c))
            if len(picks) == total:
                return picks
    return None


def _layout_is_clear(seed_groups: dict[int, list[tuple[int, int]]], h: int, w: int) -> bool:
    seed_set = {
        (r, c)
        for coords in seed_groups.values()
        for r, c in coords
    }
    occupied_arms: set[tuple[int, int]] = set()
    all_lengths: list[int] = []

    for color, coords in seed_groups.items():
        clockwise = len(coords) % 2 == 1
        for r, c in coords:
            segments = _nearest_edge_distance(r, c, h, w)
            if segments <= 0:
                return False
            all_lengths.append(segments)

            local_cells: set[tuple[int, int]] = set()
            for rr, cc in _trace_spiral(r, c, segments, clockwise):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in local_cells:
                    return False
                local_cells.add((rr, cc))
                if (rr, cc) in seed_set:
                    return False
                if (rr, cc) in occupied_arms:
                    return False

            if not local_cells:
                return False
            occupied_arms.update(local_cells)

    return (
        len(all_lengths) > 0
        and len(set(all_lengths)) >= 2
        and max(all_lengths) >= 3
        and len(occupied_arms) >= len(seed_set)
        and len(occupied_arms) <= (3 * h * w) // 5
    )


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(500):
        num_colors = int(rng.integers(2, 5))
        num_odd = num_colors // 2
        if num_colors % 2 == 1 and rng.random() < 0.5:
            num_odd += 1
        parities = [1] * num_odd + [0] * (num_colors - num_odd)
        rng.shuffle(parities)

        counts = [
            int(rng.choice([3, 5]) if parity else rng.choice([2, 4]))
            for parity in parities
        ]
        total_seeds = sum(counts)

        base = min(24, max(10, int(np.ceil(np.sqrt(total_seeds)) * 4)))
        h = int(rng.integers(max(10, base - 2), min(30, base + 6) + 1))
        w = int(rng.integers(max(10, base - 2), min(30, base + 6) + 1))

        positions = _pick_seed_positions(rng, h, w, total_seeds, spacing=2)
        if positions is None:
            continue

        rng.shuffle(positions)
        colors = rng.choice(PALETTE, size=num_colors, replace=False)
        seed_groups: dict[int, list[tuple[int, int]]] = {}

        start = 0
        for color, count in zip(colors, counts):
            seed_groups[int(color)] = positions[start:start + count]
            start += count

        if not _layout_is_clear(seed_groups, h, w):
            continue

        grid = np.zeros((h, w), dtype=int)
        for color, coords in seed_groups.items():
            for r, c in coords:
                grid[r, c] = color
        return grid

    raise RuntimeError("failed to generate a clear spiral-seed layout")
