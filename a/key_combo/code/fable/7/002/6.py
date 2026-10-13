from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np


def _append_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.append(str(candidate.parent))
            return
    fallback = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if fallback.exists():
        sys.path.append(str(fallback.parent))
        return
    raise ImportError("arc_utils.py not found")


_append_arc_utils_path()

from arc_utils import Color, bbox_coords, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
ALL_NON_BG = [1, 2, 3, 4, 5, 6, 7, 8, 9]
FRAME_COUNT_RANGE = (3, 5)
CAVITY_SIDE_RANGE = (3, 5)
GRID_SLACK = 5
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 300
SELF_CHECK_VARIANTS = 100


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    pin_pos: tuple[int, int]
    pin_color: int
    cavity: np.ndarray


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _parse_frame_component(grid: np.ndarray, obj: dict, bg: int) -> Frame | None:
    color = int(obj["color"])
    r0, c0, r1, c1 = map(int, obj["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    perimeter = _perimeter_mask(h, w)
    coords = np.asarray(obj["coords"], dtype=int)
    local = coords - np.array([r0, c0], dtype=int)
    if not np.all(perimeter[local[:, 0], local[:, 1]]):
        return None

    perimeter_count = int(perimeter.sum())
    if int(coords.shape[0]) != perimeter_count - 1:
        return None

    same_color_on_perimeter = crop[perimeter] == color
    if int(same_color_on_perimeter.sum()) != perimeter_count - 1:
        return None

    pin_locs = np.argwhere(perimeter & (crop != color))
    if len(pin_locs) != 1:
        return None
    pr, pc = map(int, pin_locs[0])
    pin_color = int(crop[pr, pc])
    if pin_color == bg:
        return None

    cavity = crop[1:-1, 1:-1].copy()
    return Frame(
        color=color,
        bbox=(r0, c0, r1, c1),
        pin_pos=(r0 + pr, c0 + pc),
        pin_color=pin_color,
        cavity=cavity,
    )


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames = []
    for obj in comps:
        frame = _parse_frame_component(g, obj, bg)
        if frame is not None:
            frames.append(frame)

    assert len(frames) >= 2, "expected at least two hollow frames"
    cavity_shapes = {frame.cavity.shape for frame in frames}
    assert len(cavity_shapes) == 1, "all cavities must have the same size"

    frame_colors = [frame.color for frame in frames]
    assert len(set(frame_colors)) == len(frame_colors), "frame colors must be unique"
    color_to_index = {frame.color: i for i, frame in enumerate(frames)}

    successors = {}
    indegree = [0] * len(frames)
    for i, frame in enumerate(frames):
        assert frame.pin_color in color_to_index, "pin color must match another frame color"
        j = color_to_index[frame.pin_color]
        assert j != i or len(frames) == 1, "self-loop is not allowed"
        successors[i] = j
        indegree[j] += 1

    assert all(deg == 1 for deg in indegree), "pins must create indegree 1 for every frame"
    seen = set()
    cur = 0
    for _ in range(len(frames)):
        assert cur not in seen, "graph must not branch before visiting all frames"
        seen.add(cur)
        cur = successors[cur]
    assert len(seen) == len(frames) and cur == 0, "pins must form a single directed cycle"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every frame's cavity content one step along the cycle encoded by off-color wall pins.

    Input structure:
    - The background is black and contains several disjoint 1-cell-thick hollow rectangular frames of distinct colors.
    - Every frame has the same cavity size and contains a small pattern inside that cavity.
    - On each frame wall exactly one cell has a different color; that off-color pin matches the main wall color of another frame.

    Rule:
    - Read each off-color wall pixel as a directed edge from its own frame to the frame whose wall color matches the pin.
    - These edges form one directed cycle over all frames; move each frame's entire cavity crop into the cavity of the frame it points to, preserving all pattern colors and relative positions.
    - Repair every pin back to its own frame color and leave all frame locations unchanged.

    Key insight:
    - The lone wrong-colored wall pixel is not noise to remove first; it is the only reliable routing label, and the cavity contents must be shifted simultaneously along the resulting color-defined cycle.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frames = _extract_frames(g)
    color_to_index = {frame.color: i for i, frame in enumerate(frames)}

    out = g.copy()
    for frame in frames:
        pr, pc = frame.pin_pos
        out[pr, pc] = frame.color
        r0, c0, r1, c1 = frame.bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    for frame in frames:
        dst = frames[color_to_index[frame.pin_color]]
        dr0, dc0, dr1, dc1 = dst.bbox
        out[dr0 + 1 : dr1, dc0 + 1 : dc1] = frame.cavity

    return out


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h - 1):
        coords.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        coords.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _normalize_coords(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(cells), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _random_polyomino(h: int, w: int, area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    coords = _normalize_coords(cells)
    ph = int(coords[:, 0].max()) + 1
    pw = int(coords[:, 1].max()) + 1
    if ph > h or pw > w:
        return _random_polyomino(h, w, area)
    mask = np.zeros((ph, pw), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _pattern_modes(cavity_h: int, cavity_w: int) -> list[np.ndarray]:
    patterns = []

    singleton = np.zeros((1, 1), dtype=bool)
    singleton[0, 0] = True
    patterns.append(singleton)

    if cavity_w >= 2:
        horiz = np.ones((1, random.randint(2, cavity_w)), dtype=bool)
        patterns.append(horiz)
    if cavity_h >= 2:
        vert = np.ones((random.randint(2, cavity_h), 1), dtype=bool)
        patterns.append(vert)

    area_cap = max(2, min(cavity_h * cavity_w - 1, cavity_h + cavity_w))
    poly = _random_polyomino(cavity_h, cavity_w, random.randint(2, area_cap))
    patterns.append(poly)
    return patterns


def _random_cavity_pattern(
    cavity_h: int, cavity_w: int, pattern_colors: list[int], used: set[bytes]
) -> np.ndarray:
    for _ in range(200):
        color = random.choice(pattern_colors)
        sprite = random.choice(_pattern_modes(cavity_h, cavity_w)).copy()
        sh, sw = sprite.shape
        anchors = [
            (0, 0),
            (0, cavity_w - sw),
            (cavity_h - sh, 0),
            (cavity_h - sh, cavity_w - sw),
            (random.randint(0, cavity_h - sh), random.randint(0, cavity_w - sw)),
        ]
        r0, c0 = random.choice(anchors)
        cavity = np.zeros((cavity_h, cavity_w), dtype=int)
        cavity[r0 : r0 + sh, c0 : c0 + sw][sprite] = color
        sig = cavity.tobytes()
        if sig not in used:
            used.add(sig)
            return cavity
    raise RuntimeError("failed to create a unique cavity pattern")


def _place_frames(n_frames: int, outer_h: int, outer_w: int) -> list[tuple[int, int, int, int]] | None:
    rows = (n_frames + 1) // 2
    min_h = rows * outer_h + (rows + 1) * 2
    min_w = 2 * outer_w + 6
    height = random.randint(min_h, min(30, min_h + GRID_SLACK))
    width = random.randint(min_w, min(30, max(min_w, min_w + GRID_SLACK)))

    if height > 30 or width > 30:
        return None

    reserve = np.zeros((height, width), dtype=int)
    block = np.ones((outer_h, outer_w), dtype=int)
    boxes = []
    for _ in range(n_frames):
        try:
            r0, c0 = random_free_location(
                reserve,
                block,
                transparent=0,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(reserve, block, r0, c0, transparent=0)
        boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    return boxes


def _render_case(
    frame_colors: list[int],
    boxes: list[tuple[int, int, int, int]],
    pin_targets: list[int],
    pin_positions: list[tuple[int, int]],
    cavities: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    height = max(r1 for _, _, r1, _ in boxes) + 2
    width = max(c1 for _, _, _, c1 in boxes) + 2
    inp = np.zeros((height, width), dtype=int)

    for i, (r0, c0, r1, c1) in enumerate(boxes):
        color = frame_colors[i]
        inp[r0, c0 : c1 + 1] = color
        inp[r1, c0 : c1 + 1] = color
        inp[r0 : r1 + 1, c0] = color
        inp[r0 : r1 + 1, c1] = color
        pr, pc = pin_positions[i]
        inp[pr, pc] = frame_colors[pin_targets[i]]
        inp[r0 + 1 : r1, c0 + 1 : c1] = cavities[i]

    out = inp.copy()
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        pr, pc = pin_positions[i]
        out[pr, pc] = frame_colors[i]
        out[r0 + 1 : r1, c0 + 1 : c1] = 0
    for i, target in enumerate(pin_targets):
        tr0, tc0, tr1, tc1 = boxes[target]
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = cavities[i]
    return inp, out


def _independent_expected(
    frame_colors: list[int],
    boxes: list[tuple[int, int, int, int]],
    pin_targets: list[int],
    pin_positions: list[tuple[int, int]],
    cavities: list[np.ndarray],
) -> np.ndarray:
    _, out = _render_case(frame_colors, boxes, pin_targets, pin_positions, cavities)
    return out


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    frame_colors: list[int],
    boxes: list[tuple[int, int, int, int]],
    pin_targets: list[int],
    pin_positions: list[tuple[int, int]],
    cavities: list[np.ndarray],
) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert len(set(frame_colors)) == len(frame_colors)
    assert all(np.count_nonzero(cavity) > 0 for cavity in cavities)
    assert len({cavity.tobytes() for cavity in cavities}) == len(cavities)
    assert set(pin_targets) == set(range(len(frame_colors)))

    seen = set()
    cur = 0
    for _ in range(len(frame_colors)):
        assert cur not in seen
        seen.add(cur)
        cur = pin_targets[cur]
    assert len(seen) == len(frame_colors) and cur == 0

    for (r0, c0, r1, c1), (pr, pc) in zip(boxes, pin_positions):
        assert r0 <= pr <= r1 and c0 <= pc <= c1
        assert pr in (r0, r1) or pc in (c0, c1)

    oracle = _independent_expected(frame_colors, boxes, pin_targets, pin_positions, cavities)
    assert np.array_equal(oracle, expected)
    assert not np.array_equal(inp, expected), "rotation should not be a no-op"


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GENERATION_ATTEMPTS):
        n_frames = random.randint(*FRAME_COUNT_RANGE)
        cavity_h = random.randint(*CAVITY_SIDE_RANGE)
        cavity_w = random.randint(*CAVITY_SIDE_RANGE)
        outer_h = cavity_h + 2
        outer_w = cavity_w + 2
        if 2 * (outer_w + 3) > 30:
            continue

        boxes = _place_frames(n_frames, outer_h, outer_w)
        if boxes is None:
            continue

        frame_colors = random.sample(ALL_NON_BG, n_frames)
        pattern_colors = [c for c in ALL_NON_BG if c not in frame_colors]
        if not pattern_colors:
            continue

        used_patterns: set[bytes] = set()
        cavities = [
            _random_cavity_pattern(cavity_h, cavity_w, pattern_colors, used_patterns)
            for _ in range(n_frames)
        ]

        order = list(range(n_frames))
        random.shuffle(order)
        pin_targets = [0] * n_frames
        for i, src in enumerate(order):
            pin_targets[src] = order[(i + 1) % n_frames]

        local_perimeter = _perimeter_coords(outer_h, outer_w)
        pin_locals = random.sample(local_perimeter, n_frames)
        pin_positions = []
        for (r0, c0, _r1, _c1), (lr, lc) in zip(boxes, pin_locals):
            pin_positions.append((r0 + lr, c0 + lc))

        inp, expected = _render_case(frame_colors, boxes, pin_targets, pin_positions, cavities)
        try:
            _validate_generated_case(
                inp, expected, frame_colors, boxes, pin_targets, pin_positions, cavities
            )
            solved = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        return inp, expected

    raise RuntimeError("failed to generate a valid input for puzzle 6")


def generate_input() -> np.ndarray:
    return _generate_case()[0]


if __name__ == "__main__":
    variants = set()
    for _ in range(SELF_CHECK_VARIANTS):
        inp, expected = _generate_case()
        out = main(inp)
        assert isinstance(inp, np.ndarray)
        assert isinstance(out, np.ndarray)
        assert inp.ndim == 2 and out.ndim == 2
        assert max(inp.shape) <= 30 and max(out.shape) <= 30
        assert np.array_equal(out, expected)
        variants.add((inp.shape, inp.tobytes()))
    assert len(variants) >= 20
    print("self-check variants", len(variants))
