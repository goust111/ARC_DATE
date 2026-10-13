from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Red is the visibly distinguished pusher and gray is the shared toll-bar color.
PUSHER_COLOR = 2
BAR_COLOR = 5
PALETTE = tuple(range(10))


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, the task's background."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1) bbox using the arc_utils convention."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    """Find monochromatic 4-connected components without external imports."""
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            points: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                points.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(points, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(points),
                }
            )
    return objects


def _solid_rectangle(obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    return int(obj["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _center(obj: dict, axis: int) -> int:
    b = obj["bbox"]
    lo, hi = (b[0], b[2]) if axis == 0 else (b[1], b[3])
    assert (lo + hi) % 2 == 0, "all object centers must lie on grid cells"
    return (lo + hi) // 2


def _paint_square(
    out: np.ndarray, axis: int, along_center: int, perp_lo: int, size: int, color: int
) -> None:
    half = size // 2
    along_lo = along_center - half
    if axis == 1:
        out[perp_lo : perp_lo + size, along_lo : along_lo + size] = color
    else:
        out[along_lo : along_lo + size, perp_lo : perp_lo + size] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Move a pushed queue through gray toll bars, leaving its current head at every bar and packing the remainder at the edge.

    Input structure:
    - On the dominant background, odd-sized solid colored squares and separated gray thin bars share one horizontal or vertical axis.
    - The red square is the uniquely marked pusher at the rear end; every other square has a distinct color, and each gray bar is longer perpendicular to the axis.
    - There are more collectible squares than bars, every bar has a square available at the queue head, and all objects lie ahead of the pusher toward one grid edge.

    Rule:
    - Scan from the pusher toward that edge.  A colored square joins the front of the queue; at a gray bar, remove the current front square and center it at that station.
    - Split each bar across the square-width axis band, shift its two surviving halves one cell outward perpendicular to motion, and leave a one-cell separation from the retained square.
    - Pack the remaining queue against the destination edge in front-to-back order, with the red pusher last.

    Key insight:
    - A station's retained color is the queue head at that moment, so collection order and every preceding toll jointly determine both all station colors and the final edge queue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = g.shape
    bg = _background(g)
    objects = _components(g, bg)
    assert objects, "expected foreground objects"
    assert all(_solid_rectangle(obj) for obj in objects), "all objects must be solid rectangles"

    bars = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if int(obj["color"]) == BAR_COLOR:
            assert min(h, w) == 1 and max(h, w) >= 5, "gray objects must be long thin bars"
            bars.append(obj)
    assert len(bars) >= 2, "expected multiple gray toll bars"

    bar_orientations = {0 if (b["bbox"][2] - b["bbox"][0]) == 0 else 1 for b in bars}
    assert len(bar_orientations) == 1, "all bars must have the same orientation"
    # A horizontal bar is crossed vertically; a vertical bar is crossed horizontally.
    axis = 0 if next(iter(bar_orientations)) == 0 else 1
    perp = 1 - axis

    bar_ids = {id(obj) for obj in bars}
    squares = [obj for obj in objects if id(obj) not in bar_ids]
    assert len(squares) > len(bars), "collectible-square count must exceed bar count"
    pushers = [obj for obj in squares if int(obj["color"]) == PUSHER_COLOR]
    assert len(pushers) == 1, "expected one red pusher"
    pusher = pushers[0]

    pr0, pc0, pr1, pc1 = pusher["bbox"]
    size = pr1 - pr0 + 1
    assert size == pc1 - pc0 + 1 and size % 2 == 1, "pusher must be an odd solid square"
    square_colors = [int(obj["color"]) for obj in squares]
    assert len(square_colors) == len(set(square_colors)), "square colors must be distinct"
    assert BAR_COLOR not in square_colors and bg not in square_colors

    perp_lo = (pr0, pc0)[perp]
    perp_hi = (pr1, pc1)[perp]
    for obj in squares:
        r0, c0, r1, c1 = obj["bbox"]
        assert r1 - r0 + 1 == size and c1 - c0 + 1 == size, "all queue pieces must match the pusher"
        assert (r0, c0)[perp] == perp_lo and (r1, c1)[perp] == perp_hi, "squares must share one axis band"
    for bar in bars:
        b = bar["bbox"]
        assert (b[0], b[1])[perp] <= perp_lo and (b[2], b[3])[perp] >= perp_hi

    pusher_center = _center(pusher, axis)
    other_centers = [_center(obj, axis) for obj in objects if obj is not pusher]
    assert all(c != pusher_center for c in other_centers)
    if all(c > pusher_center for c in other_centers):
        direction = 1
    elif all(c < pusher_center for c in other_centers):
        direction = -1
    else:
        raise AssertionError("the pusher must be at one end of the shared axis")

    events = [obj for obj in objects if obj is not pusher]
    events.sort(key=lambda obj: direction * _center(obj, axis))
    event_centers = [_center(obj, axis) for obj in events]
    assert len(event_centers) == len(set(event_centers)), "events need distinct axis positions"

    queue = [int(pusher["color"])]  # front ... rear
    retained: list[tuple[dict, int]] = []
    for obj in events:
        if id(obj) in bar_ids:
            assert len(queue) > 1, "a toll may not remove the rear pusher"
            retained.append((obj, queue.pop(0)))
        else:
            queue.insert(0, int(obj["color"]))
    assert queue[-1] == PUSHER_COLOR and len(retained) == len(bars)

    out = np.full_like(g, bg)
    for bar, held_color in retained:
        for r, c in bar["coords"]:
            rc = [int(r), int(c)]
            if rc[perp] < perp_lo:
                rc[perp] -= 1
            elif rc[perp] > perp_hi:
                rc[perp] += 1
            else:
                continue
            assert 0 <= rc[0] < height and 0 <= rc[1] < width, "split bar must stay in bounds"
            out[rc[0], rc[1]] = BAR_COLOR
        _paint_square(out, axis, _center(bar, axis), perp_lo, size, held_color)

    along_len = (height, width)[axis]
    for i, color in enumerate(queue):
        center = size // 2 + i * size if direction < 0 else along_len - 1 - size // 2 - i * size
        _paint_square(out, axis, center, perp_lo, size, color)
    return out


def _schedule(n_bars: int, n_blocks: int) -> list[str]:
    """Create causal-chain witnesses, including consecutive and interleaved tolls."""
    if n_bars == 2:
        events = ["block", "block", "bar", "bar", "block"]
    else:
        events = ["block", "block", "bar", "bar", "block", "bar", "block"]
    extras = n_blocks - events.count("block")
    for _ in range(extras):
        # Keep the consecutive-bar witness, but vary where extra collection occurs.
        candidates = [i for i in range(1, len(events) + 1) if not (events[i - 1 : i + 1] == ["bar", "bar"])]
        events.insert(random.choice(candidates), "block")
    assert events.count("bar") == n_bars and events.count("block") == n_blocks
    return events


def _transform(grid: np.ndarray, direction_case: int) -> np.ndarray:
    """Map canonical rightward motion to right, left, up, or down."""
    if direction_case == 0:
        return grid.copy()
    if direction_case == 1:
        return np.fliplr(grid).copy()
    if direction_case == 2:
        return np.rot90(grid, 1).copy()
    return np.rot90(grid, -1).copy()


def _oracle(
    shape: tuple[int, int],
    bg: int,
    row0: int,
    size: int,
    events: list[tuple[str, int, int, int]],
) -> np.ndarray:
    """Render from generator metadata, independently of component recognition."""
    height, width = shape
    out = np.full(shape, bg, dtype=int)
    queue = [PUSHER_COLOR]
    stations: list[tuple[int, int, int]] = []
    for kind, center, color, half_len in events:
        if kind == "block":
            queue.insert(0, color)
        else:
            stations.append((center, queue.pop(0), half_len))

    for center, held, half_len in stations:
        out[row0 - half_len - 1 : row0 - 1, center] = BAR_COLOR
        out[row0 + size + 1 : row0 + size + half_len + 1, center] = BAR_COLOR
        out[row0 : row0 + size, center - size // 2 : center + size // 2 + 1] = held
    for i, color in enumerate(queue):
        c1 = width - i * size
        out[row0 : row0 + size, c1 - size : c1] = color
    return out


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    schedule: list[str],
    events: list[tuple[str, int, int, int]],
    size: int,
) -> bool:
    """Check the stated semantics and anti-shortcut witnesses, not just closure."""
    if not np.array_equal(main(inp), expected) or np.array_equal(inp, expected):
        return False
    if not any(schedule[i] == schedule[i + 1] == "bar" for i in range(len(schedule) - 1)):
        return False
    if schedule.count("bar") >= 3:
        first_bar = schedule.index("bar")
        last_bar = len(schedule) - 1 - schedule[::-1].index("bar")
        if "block" not in schedule[first_bar + 1 : last_bar]:
            return False

    # Independently replay identities: every toll removes the current head and
    # the pusher survives at the edge.  At least one toll is not paired with an
    # immediately preceding block, ruling out a nearest-object shortcut.
    q = [PUSHER_COLOR]
    held: list[int] = []
    block_colors = []
    for kind, _center_col, color, _half_len in events:
        if kind == "block":
            q.insert(0, color)
            block_colors.append(color)
        else:
            if len(q) <= 1:
                return False
            held.append(q.pop(0))
    if q[-1] != PUSHER_COLOR or len(held) != schedule.count("bar"):
        return False
    if len(set(block_colors)) != len(block_colors):
        return False
    for color in [PUSHER_COLOR, *block_colors]:
        if int(np.sum(expected == color)) != size * size:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_bars = random.choice((2, 3))
    # Three stations need a compact one-cell queue to leave visible room for
    # the final packed queue within ARC's 30-cell limit; two-station cases use
    # larger 3x3 pieces so both cross-section shapes are represented.
    size = 3 if n_bars == 2 else 1
    n_blocks = n_bars + (random.choice((1, 2)) if n_bars == 2 else 1)
    schedule = _schedule(n_bars, n_blocks)

    bg = random.choice([c for c in PALETTE if c not in {PUSHER_COLOR, BAR_COLOR}])
    available = [c for c in PALETTE if c not in {bg, PUSHER_COLOR, BAR_COLOR}]
    if len(available) < n_blocks:
        return None
    block_colors = random.sample(available, n_blocks)

    half_lengths = [random.randint(2, 5) for _ in range(n_bars)]
    max_half = max(half_lengths)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    row0 = top_margin + max_half + 1
    height = row0 + size + max_half + 1 + bottom_margin
    if height > 30:
        return None

    rear_margin = random.randint(1, 2)
    pusher_c0 = rear_margin
    cursor = pusher_c0 + size
    event_specs: list[tuple[str, int, int, int]] = []
    block_i = bar_i = 0
    previous = "pusher"
    for kind in schedule:
        gap = random.randint(1, 2)
        if kind == "bar" and previous == "bar":
            gap = size  # retained station squares remain visibly separated
        start = cursor + gap
        if kind == "block":
            center = start + size // 2
            event_specs.append((kind, center, block_colors[block_i], 0))
            block_i += 1
            cursor = start + size
        else:
            center = start
            event_specs.append((kind, center, BAR_COLOR, half_lengths[bar_i]))
            bar_i += 1
            cursor = start + 1
        previous = kind

    remaining_count = n_blocks + 1 - n_bars
    last_station = max(spec[1] for spec in event_specs if spec[0] == "bar")
    min_width = max(cursor + 1, last_station + size // 2 + 2 + remaining_count * size)
    width = min_width + random.randint(0, min(3, 30 - min_width)) if min_width <= 30 else 31
    if width > 30:
        return None

    inp = np.full((height, width), bg, dtype=int)
    inp[row0 : row0 + size, pusher_c0 : pusher_c0 + size] = PUSHER_COLOR
    for kind, center, color, half_len in event_specs:
        if kind == "block":
            inp[row0 : row0 + size, center - size // 2 : center + size // 2 + 1] = color
        else:
            inp[row0 - half_len : row0 + size + half_len, center] = BAR_COLOR

    expected = _oracle(inp.shape, bg, row0, size, event_specs)
    if not _semantic_check(inp, expected, schedule, event_specs, size):
        return None

    direction_case = random.randrange(4)
    transformed_inp = _transform(inp, direction_case)
    transformed_expected = _transform(expected, direction_case)
    if max(transformed_inp.shape) > 30 or not np.array_equal(main(transformed_inp), transformed_expected):
        return None
    return transformed_inp


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate examples in all four motion directions."""
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid toll-queue puzzle")
