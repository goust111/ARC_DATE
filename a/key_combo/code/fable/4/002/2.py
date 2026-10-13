from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 900
GRID_MIN = 22
GRID_MAX = 30
FRAME_SIZE_RANGE = (4, 7)
CHAIN_LENGTH_CHOICES = (3, 4)
CORRIDOR_RANGE = (3, 7)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

SIDE_DELTAS: dict[str, Coord] = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


@dataclass(frozen=True)
class Frame:
    bbox: BBox
    color: int
    gaps: tuple[Coord, ...]


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(b: BBox) -> list[Coord]:
    r0, c0, r1, c1 = b
    out: list[Coord] = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
        out.append((r1, c))
    for r in range(r0 + 1, r1):
        out.append((r, c0))
        out.append((r, c1))
    return out


def _is_corner(p: Coord, b: BBox) -> bool:
    r, c = p
    r0, c0, r1, c1 = b
    return r in (r0, r1) and c in (c0, c1)


def _in_bbox(p: Coord, b: BBox) -> bool:
    r, c = p
    r0, c0, r1, c1 = b
    return r0 <= r <= r1 and c0 <= c <= c1


def _inside_interior(frame: Frame, r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame.bbox
    return r0 < int(r) < r1 and c0 < int(c) < c1


def _side_of_gap(gap: Coord, b: BBox) -> str:
    r, c = gap
    r0, c0, r1, c1 = b
    on_row = r in (r0, r1)
    on_col = c in (c0, c1)
    assert int(on_row) + int(on_col) == 1, "gap must be on exactly one non-corner side"
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    assert c == c1
    return "right"


def _gap_dir(gap: Coord, frame: Frame) -> Coord:
    return SIDE_DELTAS[_side_of_gap(gap, frame.bbox)]


def _inner_door_cell(frame: Frame, gap: Coord) -> Coord:
    dr, dc = _gap_dir(gap, frame)
    return int(gap[0] - dr), int(gap[1] - dc)


def _try_parse_frame(g: np.ndarray, color: int) -> Frame | None:
    coords = np.argwhere(g == int(color))
    if coords.size == 0:
        return None

    b = _bbox(coords)
    r0, c0, r1, c1 = b
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    for r_raw, c_raw in coords:
        r, c = int(r_raw), int(c_raw)
        if not (r in (r0, r1) or c in (c0, c1)):
            return None

    border = _border_cells(b)
    missing: list[Coord] = []
    colored = 0
    for r, c in border:
        if int(g[r, c]) == int(color):
            colored += 1
            continue
        if _is_corner((r, c), b) or int(g[r, c]) != BG:
            return None
        missing.append((r, c))

    if colored != int(coords.shape[0]) or len(missing) not in (1, 2):
        return None
    return Frame(bbox=b, color=int(color), gaps=tuple(sorted(missing)))


def _find_frames(g: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for color in sorted(set(map(int, g.ravel())) - {BG}):
        frame = _try_parse_frame(g, color)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda f: (f.bbox[0], f.bbox[1], f.color))
    assert len(frames) >= 3, "expected several rectangular door frames"
    return frames


def _interior_pattern(g: np.ndarray, frame: Frame) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame.bbox
    cells: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color != BG:
                cells.append((r, c, color))
    return cells


def _source_index(g: np.ndarray, frames: list[Frame]) -> int:
    filled = [i for i, frame in enumerate(frames) if _interior_pattern(g, frame)]
    assert len(filled) == 1, "exactly one frame must contain the carried pattern"
    pattern_colors = {color for _r, _c, color in _interior_pattern(g, frames[filled[0]])}
    assert len(pattern_colors) >= 2, "source pattern must be multicolor"
    return int(filled[0])


def _gap_lookup(frames: list[Frame]) -> dict[Coord, int]:
    out: dict[Coord, int] = {}
    for i, frame in enumerate(frames):
        for gap in frame.gaps:
            assert gap not in out, "two frame gaps overlap"
            out[gap] = i
    return out


def _ray_hit(g: np.ndarray, frames: list[Frame], frame_i: int, gap: Coord) -> tuple[int, Coord] | None:
    dr, dc = _gap_dir(gap, frames[frame_i])
    r, c = gap[0] + dr, gap[1] + dc
    h, w = g.shape
    lookup = _gap_lookup(frames)
    while 0 <= r < h and 0 <= c < w:
        p = (int(r), int(c))
        if p in lookup and lookup[p] != frame_i:
            other_i = lookup[p]
            odr, odc = _gap_dir(p, frames[other_i])
            if (odr, odc) == (-dr, -dc):
                return other_i, p
            return None
        if int(g[r, c]) != BG:
            return None
        r += dr
        c += dc
    return None


def _apply_relay(g: np.ndarray, frames: list[Frame], source_i: int) -> np.ndarray:
    out = g.copy()
    current_i = int(source_i)
    incoming: Coord | None = None
    pattern = _interior_pattern(g, frames[current_i])
    seen: set[tuple[int, Coord | None]] = set()

    for _ in range(len(frames) + 2):
        state = (current_i, incoming)
        assert state not in seen, "door relay must not loop"
        seen.add(state)

        gaps = list(frames[current_i].gaps)
        if incoming is None:
            assert len(gaps) == 1, "source frame must have exactly one outgoing door"
            out_gap = gaps[0]
        else:
            choices = [gap for gap in gaps if gap != incoming]
            if not choices:
                break
            assert len(choices) == 1, "a relay frame must have only one unused outgoing door"
            out_gap = choices[0]

        hit = _ray_hit(g, frames, current_i, out_gap)
        if hit is None:
            break

        next_i, in_gap = hit
        out_cell = _inner_door_cell(frames[current_i], out_gap)
        in_cell = _inner_door_cell(frames[next_i], in_gap)
        dr = int(in_cell[0] - out_cell[0])
        dc = int(in_cell[1] - out_cell[1])
        next_frame = frames[next_i]
        moved: list[tuple[int, int, int]] = []
        for r, c, color in pattern:
            nr = int(r + dr)
            nc = int(c + dc)
            if _inside_interior(next_frame, nr, nc):
                out[nr, nc] = int(color)
                moved.append((nr, nc, int(color)))

        pattern = moved
        current_i = int(next_i)
        incoming = in_gap
    else:
        raise AssertionError("door relay exceeded the number of frames")

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay the source-frame pattern through facing one-cell door gaps and keep every delivered copy.

    Input structure:
    - A black background contains several differently colored one-cell-thick hollow rectangular frames.
    - The source frame and any terminal frame have one non-corner border gap; relay frames have two gaps.
    - Exactly one frame interior contains a multicolor pattern; all other frame interiors are black.
    - A door sees another door only along its outward perpendicular ray through black cells; walls block decoy alignments.

    Rule:
    - Starting with the source frame's single gap, cast each outgoing door ray to find the first facing gap not blocked by a wall.
    - Translate the current pattern by the displacement from the interior cell just behind the outgoing gap to the interior cell just behind the incoming gap.
    - Paste only translated cells that land strictly inside the receiving frame; keep prior copies, frames, gaps, and corridors unchanged.
    - Continue from the receiving frame's other gap until no unused outgoing gap sees a facing door.

    Key insight:
    - The placement is determined by door-to-door offsets, not by centering or matching frame sizes, and visually aligned doors blocked by another frame wall are invalid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    assert 0 in set(map(int, g.ravel())), "black background must be present"

    frames = _find_frames(g)
    source_i = _source_index(g, frames)
    return _apply_relay(g, frames, source_i)


def _side_cells(b: BBox, side: str) -> list[Coord]:
    r0, c0, r1, c1 = b
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    assert side == "right"
    return [(r, c1) for r in range(r0 + 1, r1)]


def _draw_frame(g: np.ndarray, frame: Frame) -> None:
    gaps = set(frame.gaps)
    for r, c in _border_cells(frame.bbox):
        if (r, c) not in gaps:
            g[r, c] = int(frame.color)


def _bbox_in_bounds(b: BBox, shape: tuple[int, int], margin: int = 1) -> bool:
    r0, c0, r1, c1 = b
    h, w = shape
    return margin <= r0 and margin <= c0 and r1 < h - margin and c1 < w - margin


def _too_close(a: BBox, b: BBox, pad: int = 1) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _bbox_clear(b: BBox, frames: list[Frame], shape: tuple[int, int]) -> bool:
    return _bbox_in_bounds(b, shape) and all(not _too_close(b, frame.bbox, 1) for frame in frames)


def _unit_step(a: Coord, b: Coord) -> Coord | None:
    ar, ac = a
    br, bc = b
    if ar == br and ac != bc:
        return 0, 1 if bc > ac else -1
    if ac == bc and ar != br:
        return 1 if br > ar else -1, 0
    return None


def _between_cells(a: Coord, b: Coord) -> list[Coord]:
    step = _unit_step(a, b)
    assert step is not None
    dr, dc = step
    r, c = a[0] + dr, a[1] + dc
    out: list[Coord] = []
    while (r, c) != b:
        out.append((int(r), int(c)))
        r += dr
        c += dc
    return out


def _segment_clear(a: Coord, b: Coord, frames: list[Frame]) -> bool:
    for p in _between_cells(a, b):
        if any(_in_bbox(p, frame.bbox) for frame in frames):
            return False
    return True


def _random_gap(b: BBox, side: str) -> Coord:
    return random.choice(_side_cells(b, side))


def _place_next_frame(
    shape: tuple[int, int],
    frames: list[Frame],
    current: Frame,
    out_gap: Coord,
    color: int,
) -> Frame | None:
    out_side = _side_of_gap(out_gap, current.bbox)
    in_side = OPPOSITE[out_side]
    for _ in range(100):
        h = random.randint(*FRAME_SIZE_RANGE)
        w = random.randint(*FRAME_SIZE_RANGE)
        dist = random.randint(*CORRIDOR_RANGE)

        if in_side in ("left", "right"):
            offset = random.randint(1, h - 2)
            r0 = out_gap[0] - offset
            r1 = r0 + h - 1
            if in_side == "left":
                c0 = out_gap[1] + dist
                c1 = c0 + w - 1
                in_gap = (out_gap[0], c0)
            else:
                c1 = out_gap[1] - dist
                c0 = c1 - w + 1
                in_gap = (out_gap[0], c1)
        else:
            offset = random.randint(1, w - 2)
            c0 = out_gap[1] - offset
            c1 = c0 + w - 1
            if in_side == "top":
                r0 = out_gap[0] + dist
                r1 = r0 + h - 1
                in_gap = (r0, out_gap[1])
            else:
                r1 = out_gap[0] - dist
                r0 = r1 - h + 1
                in_gap = (r1, out_gap[1])

        b = (int(r0), int(c0), int(r1), int(c1))
        if not _bbox_clear(b, frames, shape):
            continue
        if not _segment_clear(out_gap, in_gap, frames):
            continue
        return Frame(bbox=b, color=int(color), gaps=(in_gap,))
    return None


def _near_side_gap(frame: Frame, side: str, near_corner: str) -> Coord:
    cells = _side_cells(frame.bbox, side)
    r0, c0, r1, c1 = frame.bbox
    if near_corner == "top":
        cells.sort(key=lambda p: abs(p[0] - (r0 + 1)))
    elif near_corner == "bottom":
        cells.sort(key=lambda p: abs(p[0] - (r1 - 1)))
    elif near_corner == "left":
        cells.sort(key=lambda p: abs(p[1] - (c0 + 1)))
    else:
        cells.sort(key=lambda p: abs(p[1] - (c1 - 1)))
    return random.choice(cells[: min(3, len(cells))])


def _choose_relay_exit(frame: Frame, in_gap: Coord) -> Coord:
    side = _side_of_gap(in_gap, frame.bbox)
    r0, c0, r1, c1 = frame.bbox
    if side in ("left", "right"):
        top_near = in_gap[0] - r0 <= r1 - in_gap[0]
        preferred = "top" if top_near else "bottom"
        other = "bottom" if top_near else "top"
        out_side = preferred if random.random() < 0.8 else other
        near = side
        return _near_side_gap(frame, out_side, near)

    left_near = in_gap[1] - c0 <= c1 - in_gap[1]
    preferred = "left" if left_near else "right"
    other = "right" if left_near else "left"
    out_side = preferred if random.random() < 0.8 else other
    near = side
    return _near_side_gap(frame, out_side, near)


def _build_chain(shape: tuple[int, int], colors: list[int], chain_len: int) -> list[Frame] | None:
    for _ in range(80):
        h = random.randint(*FRAME_SIZE_RANGE)
        w = random.randint(*FRAME_SIZE_RANGE)
        r0 = random.randint(1, shape[0] - h - 2)
        c0 = random.randint(1, shape[1] - w - 2)
        source_box = (r0, c0, r0 + h - 1, c0 + w - 1)
        source_side = random.choice(list(SIDE_DELTAS))
        source_gap = _random_gap(source_box, source_side)
        frames = [Frame(source_box, int(colors[0]), (source_gap,))]
        out_gap = source_gap

        ok = True
        for i in range(1, chain_len):
            nxt = _place_next_frame(shape, frames, frames[-1], out_gap, int(colors[i]))
            if nxt is None:
                ok = False
                break
            if i < chain_len - 1:
                exit_gap = _choose_relay_exit(nxt, nxt.gaps[0])
                nxt = Frame(nxt.bbox, nxt.color, (nxt.gaps[0], exit_gap))
                out_gap = exit_gap
            frames.append(nxt)
        if ok:
            return frames
    return None


def _place_decoys(
    shape: tuple[int, int],
    frames: list[Frame],
    colors: list[int],
) -> tuple[list[Frame], tuple[Coord, Coord]] | None:
    for _ in range(220):
        blocker = random.choice(frames)
        horizontal = random.random() < 0.5
        br0, bc0, br1, bc1 = blocker.bbox

        if horizontal:
            rows = [
                r
                for r in range(br0 + 1, br1)
                if (r, bc0) not in blocker.gaps and (r, bc1) not in blocker.gaps
            ]
            if not rows:
                continue
            r = random.choice(rows)
            h1 = random.randint(*FRAME_SIZE_RANGE)
            h2 = random.randint(*FRAME_SIZE_RANGE)
            w1 = random.randint(*FRAME_SIZE_RANGE)
            w2 = random.randint(*FRAME_SIZE_RANGE)
            off1 = random.randint(1, h1 - 2)
            off2 = random.randint(1, h2 - 2)
            d1 = random.randint(*CORRIDOR_RANGE)
            d2 = random.randint(*CORRIDOR_RANGE)
            left = (r - off1, bc0 - d1 - w1 + 1, r - off1 + h1 - 1, bc0 - d1)
            right = (r - off2, bc1 + d2, r - off2 + h2 - 1, bc1 + d2 + w2 - 1)
            gap_a = (r, left[3])
            gap_b = (r, right[1])
        else:
            cols = [
                c
                for c in range(bc0 + 1, bc1)
                if (br0, c) not in blocker.gaps and (br1, c) not in blocker.gaps
            ]
            if not cols:
                continue
            c = random.choice(cols)
            h1 = random.randint(*FRAME_SIZE_RANGE)
            h2 = random.randint(*FRAME_SIZE_RANGE)
            w1 = random.randint(*FRAME_SIZE_RANGE)
            w2 = random.randint(*FRAME_SIZE_RANGE)
            off1 = random.randint(1, w1 - 2)
            off2 = random.randint(1, w2 - 2)
            d1 = random.randint(*CORRIDOR_RANGE)
            d2 = random.randint(*CORRIDOR_RANGE)
            top = (br0 - d1 - h1 + 1, c - off1, br0 - d1, c - off1 + w1 - 1)
            bottom = (br1 + d2, c - off2, br1 + d2 + h2 - 1, c - off2 + w2 - 1)
            gap_a = (top[2], c)
            gap_b = (bottom[0], c)
            left, right = top, bottom

        first = Frame(left, int(colors[0]), (gap_a,))
        second = Frame(right, int(colors[1]), (gap_b,))
        if not _bbox_clear(first.bbox, frames, shape):
            continue
        if not _bbox_clear(second.bbox, frames + [first], shape):
            continue

        wall_a = (gap_a[0], bc0) if horizontal else (br0, gap_a[1])
        wall_b = (gap_b[0], bc1) if horizontal else (br1, gap_b[1])
        if not _segment_clear(gap_a, wall_a, frames):
            continue
        if not _segment_clear(gap_b, wall_b, frames):
            continue
        return [first, second], (gap_a, gap_b)
    return None


def _paint_source_pattern(g: np.ndarray, source: Frame, pattern_colors: list[int]) -> bool:
    gap = source.gaps[0]
    dr, dc = _gap_dir(gap, source)
    candidates: list[Coord] = []
    if dc:
        for depth in range(1, 4):
            c = gap[1] - dc * depth
            for lateral in range(-2, 3):
                r = gap[0] + lateral
                if _inside_interior(source, r, c):
                    candidates.append((r, c))
    else:
        for depth in range(1, 4):
            r = gap[0] - dr * depth
            for lateral in range(-2, 3):
                c = gap[1] + lateral
                if _inside_interior(source, r, c):
                    candidates.append((r, c))

    candidates = sorted(set(candidates), key=lambda p: (abs(p[0] - gap[0]) + abs(p[1] - gap[1]), p))
    if len(candidates) < 4:
        return False

    target = random.randint(4, min(8, len(candidates)))
    selected = [candidates[0]]
    selected_set = {candidates[0]}
    while len(selected) < target:
        frontier = [
            p
            for p in candidates
            if p not in selected_set
            and any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in selected_set)
        ]
        if not frontier:
            frontier = [p for p in candidates if p not in selected_set]
        p = random.choice(frontier[: min(6, len(frontier))])
        selected.append(p)
        selected_set.add(p)

    colors = list(pattern_colors)
    random.shuffle(colors)
    for i, (r, c) in enumerate(selected):
        g[r, c] = int(colors[i % len(colors)] if i < len(colors) else random.choice(colors))
    return len({int(g[r, c]) for r, c in selected}) >= 2


def _directed_edges(g: np.ndarray, frames: list[Frame]) -> set[tuple[tuple[int, Coord], tuple[int, Coord]]]:
    edges: set[tuple[tuple[int, Coord], tuple[int, Coord]]] = set()
    for i, frame in enumerate(frames):
        for gap in frame.gaps:
            hit = _ray_hit(g, frames, i, gap)
            if hit is not None:
                edges.add(((i, gap), hit))
    return edges


def _expected_edges(frames: list[Frame], chain_len: int) -> set[tuple[tuple[int, Coord], tuple[int, Coord]]]:
    edges: set[tuple[tuple[int, Coord], tuple[int, Coord]]] = set()
    for i in range(chain_len - 1):
        out_gap = frames[i].gaps[0] if i == 0 else frames[i].gaps[1]
        in_gap = frames[i + 1].gaps[0]
        edges.add(((i, out_gap), (i + 1, in_gap)))
        edges.add(((i + 1, in_gap), (i, out_gap)))
    return edges


def _blocked_pair_valid(g: np.ndarray, frames: list[Frame], pair: tuple[Coord, Coord]) -> bool:
    a, b = pair
    step = _unit_step(a, b)
    if step is None:
        return False
    lookup = _gap_lookup(frames)
    if a not in lookup or b not in lookup:
        return False
    ai = lookup[a]
    bi = lookup[b]
    if _gap_dir(a, frames[ai]) != step or _gap_dir(b, frames[bi]) != (-step[0], -step[1]):
        return False
    for r, c in _between_cells(a, b):
        if int(g[r, c]) != BG:
            return True
    return False


def _known_output_and_steps(
    g: np.ndarray,
    frames: list[Frame],
    chain_len: int,
) -> tuple[np.ndarray, list[tuple[int, int, Coord]]]:
    out = g.copy()
    pattern = _interior_pattern(g, frames[0])
    steps: list[tuple[int, int, Coord]] = []
    for i in range(chain_len - 1):
        out_gap = frames[i].gaps[0] if i == 0 else frames[i].gaps[1]
        in_gap = frames[i + 1].gaps[0]
        out_cell = _inner_door_cell(frames[i], out_gap)
        in_cell = _inner_door_cell(frames[i + 1], in_gap)
        dr = int(in_cell[0] - out_cell[0])
        dc = int(in_cell[1] - out_cell[1])
        moved: list[tuple[int, int, int]] = []
        for r, c, color in pattern:
            nr = r + dr
            nc = c + dc
            if _inside_interior(frames[i + 1], nr, nc):
                out[nr, nc] = color
                moved.append((nr, nc, color))
        steps.append((len(pattern), len(moved), (dr, dc)))
        pattern = moved
    return out, steps


def _validate_generated(
    inp: np.ndarray,
    frames: list[Frame],
    chain_len: int,
    blocked_pair: tuple[Coord, Coord],
) -> bool:
    try:
        parsed = _find_frames(inp)
        expected_signature = sorted((f.color, f.bbox, f.gaps) for f in frames)
        parsed_signature = sorted((f.color, f.bbox, f.gaps) for f in parsed)
        if parsed_signature != expected_signature:
            return False

        if _directed_edges(inp, frames) != _expected_edges(frames, chain_len):
            return False
        if not _blocked_pair_valid(inp, frames, blocked_pair):
            return False

        expected, steps = _known_output_and_steps(inp, frames, chain_len)
        if len(steps) != chain_len - 1 or any(new == 0 for _old, new, _d in steps):
            return False
        if not any(0 < new < old for old, new, _d in steps):
            return False
        if len({d for _old, _new, d in steps}) < 2:
            return False

        outside = np.ones(inp.shape, dtype=bool)
        for frame in frames[:chain_len]:
            r0, c0, r1, c1 = frame.bbox
            outside[r0 + 1 : r1, c0 + 1 : c1] = False
        if not np.array_equal(inp[outside], expected[outside]):
            return False

        for frame in frames[chain_len:]:
            if _interior_pattern(expected, frame):
                return False

        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    chain_len = random.choice(CHAIN_LENGTH_CHOICES)
    pattern_count = random.choice((2, 3))
    needed = chain_len + 2 + pattern_count
    if needed > 9:
        pattern_count = 2

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_colors = palette[: chain_len + 2]
    pattern_colors = palette[chain_len + 2 : chain_len + 2 + pattern_count]

    chain = _build_chain((rows, cols), frame_colors[:chain_len], chain_len)
    if chain is None:
        return None

    decoy = _place_decoys((rows, cols), chain, frame_colors[chain_len : chain_len + 2])
    if decoy is None:
        return None
    decoys, blocked_pair = decoy
    frames = chain + decoys

    inp = np.full((rows, cols), BG, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame)
    if not _paint_source_pattern(inp, chain[0], pattern_colors):
        return None

    if _validate_generated(inp, frames, chain_len, blocked_pair):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-relay pattern input")
