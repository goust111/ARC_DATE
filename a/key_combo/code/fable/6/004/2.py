from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


SIDE_INFO = {
    "T": {"normal": (-1, 0), "opposite": "B"},
    "B": {"normal": (1, 0), "opposite": "T"},
    "L": {"normal": (0, -1), "opposite": "R"},
    "R": {"normal": (0, 1), "opposite": "L"},
}

GEN_ATTEMPTS = 500
SEQ_COLORS = list(range(1, 10))


@dataclass(frozen=True)
class PortInterval:
    side: str
    start: int
    end: int
    physical_seq: tuple[int, ...]
    clockwise_seq: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SpriteSpec:
    sprite: np.ndarray
    body: int
    port_side: str
    intervals: list[PortInterval]


def _side_array(sprite: np.ndarray, side: str) -> np.ndarray:
    if side == "T":
        return sprite[0, :].copy()
    if side == "B":
        return sprite[-1, :].copy()
    if side == "L":
        return sprite[:, 0].copy()
    if side == "R":
        return sprite[:, -1].copy()
    raise ValueError(f"bad side: {side}")


def _clockwise_from_physical(seq: np.ndarray | tuple[int, ...], side: str) -> tuple[int, ...]:
    vals = tuple(int(x) for x in seq)
    if side in ("T", "R"):
        return vals
    return vals[::-1]


def _side_coord(shape: tuple[int, int], side: str, idx: int) -> tuple[int, int]:
    h, w = shape
    idx = int(idx)
    if side == "T":
        return 0, idx
    if side == "B":
        return h - 1, idx
    if side == "L":
        return idx, 0
    if side == "R":
        return idx, w - 1
    raise ValueError(f"bad side: {side}")


def _dominant_color(sprite: np.ndarray) -> int:
    vals, counts = np.unique(sprite, return_counts=True)
    order = np.argsort(counts)[::-1]
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]])
    else:
        assert int(counts[order[0]]) > 0
    return int(vals[order[0]])


def _analyze_sprite(sprite: np.ndarray, bg: int) -> SpriteSpec:
    assert sprite.ndim == 2
    assert sprite.size > 0
    assert not np.any(sprite == bg), "object bbox must be a filled rectangle"
    body = _dominant_color(sprite)
    deviating = [side for side in "TBLR" if np.any(_side_array(sprite, side) != body)]
    assert len(deviating) == 1, "expected exactly one coded edge"
    port_side = deviating[0]
    intervals = _enumerate_intervals(_side_array(sprite, port_side), port_side, body)
    assert intervals, "expected at least one port interval candidate"
    return SpriteSpec(sprite=sprite, body=body, port_side=port_side, intervals=intervals)


def _enumerate_intervals(edge: np.ndarray, side: str, body: int) -> list[PortInterval]:
    n = len(edge)
    out: list[PortInterval] = []
    for start in range(n):
        if np.any(edge[:start] != body):
            continue
        for end in range(start + 2, n + 1):
            if np.any(edge[end:] != body):
                continue
            seg = tuple(int(x) for x in edge[start:end])
            if len(set(seg)) < 2:
                continue
            clockwise = _clockwise_from_physical(seg, side)
            if clockwise == clockwise[::-1]:
                continue
            out.append(
                PortInterval(
                    side=side,
                    start=start,
                    end=end,
                    physical_seq=seg,
                    clockwise_seq=clockwise,
                )
            )
    return out


def _translation_from_match(
    shape_a: tuple[int, int],
    interval_a: PortInterval,
    shape_b: tuple[int, int],
    interval_b: PortInterval,
) -> tuple[int, int]:
    assert interval_b.side == SIDE_INFO[interval_a.side]["opposite"]
    nr, nc = SIDE_INFO[interval_a.side]["normal"]
    first_a = _side_coord(shape_a, interval_a.side, interval_a.start)
    first_b = _side_coord(shape_b, interval_b.side, interval_b.start)
    dr = first_a[0] + nr - first_b[0]
    dc = first_a[1] + nc - first_b[1]
    for k in range(interval_a.length):
        ra, ca = _side_coord(shape_a, interval_a.side, interval_a.start + k)
        rb, cb = _side_coord(shape_b, interval_b.side, interval_b.start + k)
        assert (ra + nr, ca + nc) == (rb + dr, cb + dc)
    return int(dr), int(dc)


def _render_pair(a: np.ndarray, b: np.ndarray, dr: int, dc: int, bg: int) -> np.ndarray:
    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(a != bg):
        coords.append((int(rr), int(cc)))
    for rr, cc in np.argwhere(b != bg):
        coords.append((int(rr) + dr, int(cc) + dc))
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    r1 = max(r for r, _ in coords)
    c1 = max(c for _, c in coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    paste(out, a, -r0, -c0, transparent=bg)
    rr = dr - r0
    cc = dc - c0
    mask_b = b != bg
    target = out[rr : rr + b.shape[0], cc : cc + b.shape[1]]
    assert target.shape == b.shape
    assert not np.any((target != bg) & mask_b), "assembled objects may not overlap"
    paste(out, b, rr, cc, transparent=bg)
    return out


def _find_unique_match(spec_a: SpriteSpec, spec_b: SpriteSpec) -> tuple[PortInterval, PortInterval, tuple[int, int]]:
    matches = []
    same_order_matches = 0
    for cand_a in spec_a.intervals:
        for cand_b in spec_b.intervals:
            if cand_b.side != SIDE_INFO[cand_a.side]["opposite"]:
                continue
            if cand_a.clockwise_seq == cand_b.clockwise_seq:
                same_order_matches += 1
            if cand_a.clockwise_seq != cand_b.clockwise_seq[::-1]:
                continue
            dr, dc = _translation_from_match(spec_a.sprite.shape, cand_a, spec_b.sprite.shape, cand_b)
            try:
                _render_pair(spec_a.sprite, spec_b.sprite, dr, dc, bg=-1)
            except AssertionError:
                continue
            matches.append((cand_a, cand_b, (dr, dc)))
    assert len(matches) == 1, "expected exactly one reversed port match"
    assert same_order_matches == 0, "same-order edge-code matching should fail"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Dock two rectangular objects by matching one edge code to the reverse of the other.

    Input structure:
    A dominant background surrounds two disconnected filled rectangular objects. Each rectangle
    has one dominant body color, and exactly one edge contains a contiguous multicolor code
    segment while the rest of its boundary stays in the body color.

    Rule:
    Read each candidate code segment along its object's boundary in clockwise order. The unique
    valid port pair is the pair of opposite-facing edge segments whose clockwise readings are
    exact reverses of one another. Translate the rectangles without rotating or flipping so
    those port cells face each other and match cell-by-cell, then return the tight crop of the
    assembled result on the original background color.

    Key insight:
    The visible connector is not a single marker but a boundary sequence. When two sides face
    each other, equal facing colors correspond to reversed clockwise boundary readings rather
    than equal readings in the same direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) == 2, "this puzzle's consistent case has exactly two rectangles"
    sprites = [crop_bbox(g, comp["bbox"]) for comp in components]
    specs = [_analyze_sprite(sprite, bg) for sprite in sprites]
    _cand_a, _cand_b, (dr, dc) = _find_unique_match(specs[0], specs[1])
    return _render_pair(specs[0].sprite, specs[1].sprite, dr, dc, bg)


def _paint_port(sprite: np.ndarray, side: str, start: int, seq: tuple[int, ...]) -> None:
    for k, color in enumerate(seq):
        rr, cc = _side_coord(sprite.shape, side, start + k)
        sprite[rr, cc] = int(color)


def _choose_sequence(length: int, body_a: int, body_b: int, hide_mode: str) -> tuple[int, ...]:
    assert length >= 2
    visible_pool = [c for c in SEQ_COLORS if c not in {body_a, body_b}]
    while True:
        seq = [random.choice(visible_pool) for _ in range(length)]
        if hide_mode == "A":
            seq[0] = body_a
            if length >= 4 and random.random() < 0.7:
                seq[-1] = body_a
        elif hide_mode == "B":
            seq[-1] = body_b
            if length >= 4 and random.random() < 0.7:
                seq[0] = body_b
        if len(set(seq)) < 2:
            continue
        if tuple(seq) == tuple(reversed(seq)):
            continue
        if any(x == 0 for x in seq):
            continue
        return tuple(int(x) for x in seq)


def _actual_pair_output(
    sprite_a: np.ndarray,
    side_a: str,
    start_a: int,
    seq: tuple[int, ...],
    sprite_b: np.ndarray,
    side_b: str,
    start_b: int,
    bg: int,
) -> np.ndarray:
    interval_a = PortInterval(
        side=side_a,
        start=start_a,
        end=start_a + len(seq),
        physical_seq=seq,
        clockwise_seq=_clockwise_from_physical(seq, side_a),
    )
    interval_b = PortInterval(
        side=side_b,
        start=start_b,
        end=start_b + len(seq),
        physical_seq=seq,
        clockwise_seq=_clockwise_from_physical(seq, side_b),
    )
    dr, dc = _translation_from_match(sprite_a.shape, interval_a, sprite_b.shape, interval_b)
    return _render_pair(sprite_a, sprite_b, dr, dc, bg)


def _random_rect_position(h: int, w: int, H: int, W: int) -> tuple[int, int]:
    return random.randint(0, H - h), random.randint(0, W - w)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _scatter_input(sprite_a: np.ndarray, sprite_b: np.ndarray, bg: int) -> np.ndarray | None:
    out_h = random.randint(max(sprite_a.shape[0], sprite_b.shape[0]) + 8, 24)
    out_w = random.randint(max(sprite_a.shape[1], sprite_b.shape[1]) + 8, 24)
    for _ in range(120):
        r1, c1 = _random_rect_position(sprite_a.shape[0], sprite_a.shape[1], out_h, out_w)
        r2, c2 = _random_rect_position(sprite_b.shape[0], sprite_b.shape[1], out_h, out_w)
        b1 = (r1, c1, r1 + sprite_a.shape[0] - 1, c1 + sprite_a.shape[1] - 1)
        b2 = (r2, c2, r2 + sprite_b.shape[0] - 1, c2 + sprite_b.shape[1] - 1)
        if not _boxes_separated(b1, b2, gap=1):
            continue
        grid = np.full((out_h, out_w), bg, dtype=int)
        paste(grid, sprite_a, r1, c1, transparent=bg)
        paste(grid, sprite_b, r2, c2, transparent=bg)
        if get_bg(grid) != bg:
            continue
        return grid
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray, bool] | None:
    bg = random.choice(SEQ_COLORS)
    bodies = random.sample([c for c in SEQ_COLORS if c != bg], 2)
    body_a, body_b = bodies
    horizontal = random.random() < 0.5
    if horizontal:
        h1 = random.randint(5, 9)
        h2 = random.randint(5, 9)
        w1 = random.randint(4, 8)
        w2 = random.randint(4, 8)
        side_a, side_b = random.choice([("R", "L"), ("L", "R")])
        side_len = min(h1, h2)
    else:
        h1 = random.randint(4, 8)
        h2 = random.randint(4, 8)
        w1 = random.randint(5, 9)
        w2 = random.randint(5, 9)
        side_a, side_b = random.choice([("B", "T"), ("T", "B")])
        side_len = min(w1, w2)
    if side_len < 5:
        return None
    seq_len = random.randint(3, side_len - 2)
    max_start_a = len(_side_array(np.zeros((h1, w1), dtype=int), side_a)) - seq_len - 1
    max_start_b = len(_side_array(np.zeros((h2, w2), dtype=int), side_b)) - seq_len - 1
    if max_start_a < 1 or max_start_b < 1:
        return None
    start_a = random.randint(1, max_start_a)
    start_b = random.randint(1, max_start_b)

    hide_mode = random.choices(["A", "B", "none"], weights=[4, 4, 3])[0]
    seq = _choose_sequence(seq_len, body_a, body_b, hide_mode)

    sprite_a = np.full((h1, w1), body_a, dtype=int)
    sprite_b = np.full((h2, w2), body_b, dtype=int)
    _paint_port(sprite_a, side_a, start_a, seq)
    _paint_port(sprite_b, side_b, start_b, seq)

    expected = _actual_pair_output(sprite_a, side_a, start_a, seq, sprite_b, side_b, start_b, bg)
    if max(expected.shape) > 30:
        return None
    inp = _scatter_input(sprite_a, sprite_b, bg)
    if inp is None:
        return None
    return inp, expected, hide_mode != "none"


def _validate_case(inp: np.ndarray, expected: np.ndarray, require_hidden_ambiguity: bool) -> bool:
    bg = get_bg(inp)
    components = find_components(inp, bg=bg, connectivity=4, monochromatic=False)
    if len(components) != 2:
        return False
    sprites = [crop_bbox(inp, comp["bbox"]) for comp in components]
    try:
        specs = [_analyze_sprite(sprite, bg) for sprite in sprites]
        _match = _find_unique_match(specs[0], specs[1])
    except AssertionError:
        return False
    if require_hidden_ambiguity and max(len(spec.intervals) for spec in specs) < 2:
        return False
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, inp):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected, hidden = built
        if _validate_case(inp, expected, hidden):
            return inp
    raise RuntimeError("failed to generate a valid reversed-port ARC sample")
