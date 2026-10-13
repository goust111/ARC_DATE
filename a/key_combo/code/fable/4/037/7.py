from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the common ARC convention: shape=(rows, cols), coord=(r, c)."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: set[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[set[Coord]]:
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    components: list[set[Coord]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in unseen:
                    unseen.remove(cell)
                    component.add(cell)
                    queue.append(cell)
        components.append(component)
    return components


def _perimeter(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _visible_perimeter(shape: tuple[int, int], top: int, left: int, fh: int, fw: int) -> set[Coord]:
    return {
        (r, c)
        for r, c in _perimeter((top, left, top + fh - 1, left + fw - 1))
        if _in_bounds(shape, r, c)
    }


def _target_candidates(
    shape: tuple[int, int], source: BBox, target_pixels: set[Coord]
) -> list[tuple[str, int, int]]:
    """Enumerate same-size, axis-aligned frames clipped by exactly the relevant edge."""
    h, w = shape
    sr0, sc0, sr1, sc1 = source
    fh, fw = sr1 - sr0 + 1, sc1 - sc0 + 1
    candidates: list[tuple[str, int, int]] = []

    placements: list[tuple[str, int, int]] = []
    placements += [("left", sr0, c0) for c0 in range(1 - fw, 0)]
    placements += [("right", sr0, c0) for c0 in range(w - fw + 1, w)]
    placements += [("up", r0, sc0) for r0 in range(1 - fh, 0)]
    placements += [("down", r0, sc0) for r0 in range(h - fh + 1, h)]

    for direction, tr0, tc0 in placements:
        tr1, tc1 = tr0 + fh - 1, tc0 + fw - 1
        strictly_separate = {
            "left": tc1 < sc0,
            "right": sc1 < tc0,
            "up": tr1 < sr0,
            "down": sr1 < tr0,
        }[direction]
        if not strictly_separate:
            continue
        visible = _visible_perimeter(shape, tr0, tc0, fh, fw)
        if visible and visible == target_pixels:
            candidates.append((direction, tr0, tc0))
    return candidates


def _locate_scene(grid: np.ndarray) -> tuple[int, int, BBox, str, int, int]:
    """Return (background, frame color, source bbox, direction, target top, target left)."""
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    scenes: list[tuple[int, int, BBox, str, int, int]] = []

    for frame_color in set(map(int, g.flat)) - {bg}:
        color_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == frame_color)}
        for component in _components4(g == frame_color):
            box = _bbox(component)
            r0, c0, r1, c1 = box
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            if component != _perimeter(box):
                continue

            target_pixels = color_cells - component
            if not target_pixels:
                continue
            candidates = _target_candidates(g.shape, box, target_pixels)
            for direction, tr0, tc0 in candidates:
                scenes.append((bg, frame_color, box, direction, tr0, tc0))

    assert len(scenes) == 1, "the complete source frame and clipped target frame must be uniquely determined"
    return scenes[0]


def _mirror_relative(ir: int, ic: int, ih: int, iw: int, direction: str) -> Coord:
    if direction in {"left", "right"}:
        return ir, iw - 1 - ic
    return ih - 1 - ir, ic


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the source-frame markers toward a same-size off-canvas target frame, clip them, and erase the source markers.

    Input structure:
    - A black background contains two same-colored, same-size hollow rectangular frames.
    - The source frame is fully visible and contains colored marker pixels; the target frame crosses one canvas edge, so only a clipped frame ring is visible and its visible interior is black.
    - The frames share their row placement when separated left/right, or their column placement when separated above/below.

    Rule:
    - Reconstruct the unique full target frame using the source dimensions and its visible clipped perimeter.
    - Mirror source-interior coordinates horizontally for a left/right target or vertically for an above/below target, copy marker colors into the target interior, and discard off-grid destinations.
    - Replace every source-interior marker with background while preserving both visible frame rings.

    Key insight:
    - The visible bbox of the target is not its coordinate system: its missing off-canvas portion must first be restored from the complete frame, and the frames' relative direction then selects the mirror axis before clipping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, frame_color, source, direction, tr0, tc0 = _locate_scene(g)
    sr0, sc0, sr1, sc1 = source
    fh, fw = sr1 - sr0 + 1, sc1 - sc0 + 1
    ih, iw = fh - 2, fw - 2

    source_perimeter = _perimeter(source)
    non_frame = {tuple(map(int, rc)) for rc in np.argwhere((g != bg) & (g != frame_color))}
    source_interior = {
        (r, c) for r in range(sr0 + 1, sr1) for c in range(sc0 + 1, sc1)
    }
    assert non_frame and non_frame <= source_interior, "all marker pixels must lie inside the complete source frame"

    target_interior = [
        (tr0 + 1 + ir, tc0 + 1 + ic)
        for ir in range(ih)
        for ic in range(iw)
        if _in_bounds(g.shape, tr0 + 1 + ir, tc0 + 1 + ic)
    ]
    assert target_interior and all(int(g[r, c]) == bg for r, c in target_interior), (
        "the visible target interior must initially be background"
    )
    assert all(int(g[r, c]) == frame_color for r, c in source_perimeter)

    markers = [(r - sr0 - 1, c - sc0 - 1, int(g[r, c])) for r, c in sorted(non_frame)]
    out = g.copy()
    for ir, ic, _color in markers:
        out[sr0 + 1 + ir, sc0 + 1 + ic] = bg

    for ir, ic, color in markers:
        mr, mc = _mirror_relative(ir, ic, ih, iw, direction)
        r, c = tr0 + 1 + mr, tc0 + 1 + mc
        if _in_bounds(g.shape, r, c):
            assert int(g[r, c]) == bg, "a copied marker must land in the visible target interior"
            out[r, c] = color
    return out


# ------------------------------ generator ------------------------------

_DIRECTIONS = ("left", "up", "right", "down")
_generation_count = 0
_MAX_ATTEMPTS = 400


def _draw_frame(grid: np.ndarray, top: int, left: int, fh: int, fw: int, color: int) -> None:
    for r, c in _perimeter((top, left, top + fh - 1, left + fw - 1)):
        if _in_bounds(grid.shape, r, c):
            grid[r, c] = color


def _sample_layout(direction: str) -> tuple[tuple[int, int], BBox, int, int, int]:
    """Return shape, source bbox, target top/left packed as three trailing ints (top, left, clip)."""
    fh, fw = random.randint(6, 10), random.randint(6, 10)
    if direction in {"left", "right"}:
        clip = random.randint(2, fw - 3)
        w = random.randint(2 * fw - clip + 5, 30)
        h = random.randint(max(12, fh + 4), min(30, fh + 12))
        sr0 = random.randint(2, h - fh - 2)
        if direction == "left":
            tr0, tc0 = sr0, -clip
            sc0 = random.randint(fw - clip + 2, w - fw - 2)
        else:
            tr0, tc0 = sr0, w - fw + clip
            sc0 = random.randint(2, tc0 - fw - 2)
    else:
        clip = random.randint(2, fh - 3)
        h = random.randint(2 * fh - clip + 5, 30)
        w = random.randint(max(12, fw + 4), min(30, fw + 12))
        sc0 = random.randint(2, w - fw - 2)
        if direction == "up":
            tr0, tc0 = -clip, sc0
            sr0 = random.randint(fh - clip + 2, h - fh - 2)
        else:
            tr0, tc0 = h - fh + clip, sc0
            sr0 = random.randint(2, tr0 - fh - 2)
    source = (sr0, sc0, sr0 + fh - 1, sc0 + fw - 1)
    return (h, w), source, tr0, tc0, clip


def _destination(
    ir: int, ic: int, source: BBox, direction: str, tr0: int, tc0: int, mirrored: bool
) -> Coord:
    fh = source[2] - source[0] + 1
    fw = source[3] - source[1] + 1
    ih, iw = fh - 2, fw - 2
    if mirrored:
        ir, ic = _mirror_relative(ir, ic, ih, iw, direction)
    return tr0 + 1 + ir, tc0 + 1 + ic


def _sample_markers(
    shape: tuple[int, int], source: BBox, direction: str, tr0: int, tc0: int, frame_color: int
) -> dict[Coord, int] | None:
    fh, fw = source[2] - source[0] + 1, source[3] - source[1] + 1
    ih, iw = fh - 2, fw - 2
    cells = [(ir, ic) for ir in range(ih) for ic in range(iw)]
    visible = [p for p in cells if _in_bounds(shape, *_destination(*p, source, direction, tr0, tc0, True))]
    clipped = [p for p in cells if p not in visible]
    if len(visible) < 3 or len(clipped) < 2:
        return None

    n = random.randint(6, min(10, len(cells)))
    chosen = random.sample(visible, 3) + random.sample(clipped, 2)
    remaining = [p for p in cells if p not in chosen]
    chosen += random.sample(remaining, n - len(chosen))

    palette = random.sample([c for c in range(1, 10) if c != frame_color], random.choice((2, 3)))
    colors = [random.choice(palette) for _ in chosen]
    for i, color in enumerate(palette):
        colors[i] = color  # every marker color is represented at a visible destination
    markers = {p: color for p, color in zip(chosen, colors)}

    mirrored_pattern = {
        (*_mirror_relative(ir, ic, ih, iw, direction), color) for (ir, ic), color in markers.items()
    }
    original_pattern = {(ir, ic, color) for (ir, ic), color in markers.items()}
    if mirrored_pattern == original_pattern:
        return None

    mirrored_visible = {
        (*_destination(ir, ic, source, direction, tr0, tc0, True), color)
        for (ir, ic), color in markers.items()
        if _in_bounds(shape, *_destination(ir, ic, source, direction, tr0, tc0, True))
    }
    plain_visible = {
        (*_destination(ir, ic, source, direction, tr0, tc0, False), color)
        for (ir, ic), color in markers.items()
        if _in_bounds(shape, *_destination(ir, ic, source, direction, tr0, tc0, False))
    }
    if mirrored_visible == plain_visible:
        return None
    return markers


def _independent_output(
    inp: np.ndarray,
    source: BBox,
    direction: str,
    tr0: int,
    tc0: int,
    markers: dict[Coord, int],
) -> np.ndarray:
    out = inp.copy()
    sr0, sc0, _sr1, _sc1 = source
    for (ir, ic), color in markers.items():
        out[sr0 + 1 + ir, sc0 + 1 + ic] = 0
        r, c = _destination(ir, ic, source, direction, tr0, tc0, True)
        if _in_bounds(inp.shape, r, c):
            out[r, c] = color
    return out


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    source: BBox,
    direction: str,
    tr0: int,
    tc0: int,
    markers: dict[Coord, int],
    frame_color: int,
) -> bool:
    fh, fw = source[2] - source[0] + 1, source[3] - source[1] + 1
    source_ring = _perimeter(source)
    target_ring = _visible_perimeter(inp.shape, tr0, tc0, fh, fw)
    if not target_ring or target_ring & source_ring:
        return False
    if any(int(inp[r, c]) != frame_color or int(expected[r, c]) != frame_color for r, c in source_ring | target_ring):
        return False

    destinations = [_destination(ir, ic, source, direction, tr0, tc0, True) for ir, ic in markers]
    visible_count = sum(_in_bounds(inp.shape, r, c) for r, c in destinations)
    clipped_count = len(destinations) - visible_count
    if visible_count < 2 or clipped_count < 1:
        return False

    sr0, sc0, _sr1, _sc1 = source
    if any(int(expected[sr0 + 1 + ir, sc0 + 1 + ic]) != 0 for ir, ic in markers):
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        located = _locate_scene(inp)
        if located[3:] != (direction, tr0, tc0):
            return False
        return np.array_equal(main(inp), expected)
    except (AssertionError, ValueError):
        return False


def _generate_attempt(direction: str) -> np.ndarray | None:
    shape, source, tr0, tc0, _clip = _sample_layout(direction)
    frame_color = random.randint(1, 9)
    markers = _sample_markers(shape, source, direction, tr0, tc0, frame_color)
    if markers is None:
        return None

    inp = np.zeros(shape, dtype=int)
    fh, fw = source[2] - source[0] + 1, source[3] - source[1] + 1
    _draw_frame(inp, source[0], source[1], fh, fw, frame_color)
    _draw_frame(inp, tr0, tc0, fh, fw, frame_color)
    for (ir, ic), color in markers.items():
        inp[source[0] + 1 + ir, source[1] + 1 + ic] = color

    expected = _independent_output(inp, source, direction, tr0, tc0, markers)
    if _semantic_check(inp, expected, source, direction, tr0, tc0, markers, frame_color):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Generate varied examples while alternating mirror axes across consecutive calls."""
    global _generation_count
    direction = _DIRECTIONS[_generation_count % len(_DIRECTIONS)]
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt(direction)
        if grid is not None:
            _generation_count += 1
            return grid
    raise RuntimeError(f"failed to generate a valid {direction} example")
