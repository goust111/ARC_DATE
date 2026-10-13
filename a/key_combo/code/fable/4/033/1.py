from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]


@dataclass(frozen=True)
class _Chain:
    """A recovered chain of full rectangle components."""

    h: int
    w: int
    direction: int
    anchors: tuple[Coord, ...]
    gap: Coord


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue: deque[Coord] = deque([(sr, sc)])
        points: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            points.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(points, dtype=int))
    return result


def _solid_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Describe every solid component as (top, left, height, width)."""
    rectangles: list[tuple[int, int, int, int]] = []
    for coords in _components(mask):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if len(coords) == h * w:
            rectangles.append((int(r0), int(c0), h, w))
    return rectangles


def _recover_chains(mask: np.ndarray) -> list[_Chain]:
    """Infer rectangle size, diagonal direction, and the one internal gap per chain."""
    rectangles = _solid_rectangles(mask)
    shape_counts = Counter((h, w) for _r, _c, h, w in rectangles if h >= 2 and w >= 2)
    candidate_shapes = {shape for shape, count in shape_counts.items() if count >= 4}

    found: list[_Chain] = []
    for h, w in candidate_shapes:
        shape_anchors = [(r, c) for r, c, hh, ww in rectangles if (hh, ww) == (h, w)]
        for direction in (-1, 1):
            # c*h - direction*r*w is invariant under the stipulated (h, +/-w) stride.
            lines: dict[int, list[Coord]] = defaultdict(list)
            for r, c in shape_anchors:
                lines[c * h - direction * r * w].append((r, c))
            for anchors in lines.values():
                anchors.sort()
                if len(anchors) < 4:
                    continue
                jumps: list[int] = []
                valid = True
                for (r0, c0), (r1, c1) in zip(anchors, anchors[1:]):
                    dr, dc = r1 - r0, c1 - c0
                    if dr <= 0 or dr % h or dc != direction * (dr // h) * w:
                        valid = False
                        break
                    jumps.append(dr // h)
                if not valid or jumps.count(2) != 1 or any(jump not in (1, 2) for jump in jumps):
                    continue
                gap_index = jumps.index(2)
                r, c = anchors[gap_index]
                gap = (r + h, c + direction * w)
                found.append(_Chain(h, w, direction, tuple(anchors), gap))

    # A valid puzzle has exactly one maximally supported line for each of two sizes.
    if found:
        best_by_shape: dict[tuple[int, int], list[_Chain]] = defaultdict(list)
        for chain in found:
            best_by_shape[(chain.h, chain.w)].append(chain)
        reduced: list[_Chain] = []
        for chains in best_by_shape.values():
            best_len = max(len(chain.anchors) for chain in chains)
            reduced.extend(chain for chain in chains if len(chain.anchors) == best_len)
        found = reduced
    return found


def _box_count(mask: np.ndarray, anchor: Coord, h: int, w: int) -> int:
    r, c = anchor
    assert 0 <= r and 0 <= c and r + h <= mask.shape[0] and c + w <= mask.shape[1]
    return int(mask[r : r + h, c : c + w].sum())


def main(grid: np.ndarray) -> np.ndarray:
    """Complete only the missing diagonal-chain rectangle whose slot contains a small foreground residue.

    Input structure:
    - Black is the background and all foreground pixels have one shared color.
    - Solid rectangles have exactly two distinct sizes.  Each size forms its own
      diagonal chain, stepping by that rectangle's (height, width); end stamps
      may be clipped by the canvas.
    - Each chain has one internal missing stamp.  One slot is empty, while the
      other retains exactly one or two foreground pixels.

    Rule:
    - Recover the two families from their full rectangle components and infer
      each diagonal direction and missing anchor using its own size as stride.
    - Fill the entire missing rectangle only when its inferred box contains the
      one- or two-pixel residue.  Preserve the clean gap and every other pixel.

    Key insight:
    - Rectangle size determines each chain's coordinate stride, but the residue
      is the unique authorization signal selecting which precisely inferred gap
      is restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"
    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert len(colors) == 1, "expected black background and one foreground color"
    foreground = colors[0]
    mask = g == foreground

    chains = _recover_chains(mask)
    assert len(chains) == 2, "the two rectangle chains must be uniquely recoverable"
    assert len({(chain.h, chain.w) for chain in chains}) == 2, "chain sizes must differ"

    counts = [_box_count(mask, chain.gap, chain.h, chain.w) for chain in chains]
    residue_indices = [i for i, count in enumerate(counts) if count in (1, 2)]
    clean_indices = [i for i, count in enumerate(counts) if count == 0]
    assert len(residue_indices) == len(clean_indices) == 1, "one gap must be clean and one must contain residue"

    selected = chains[residue_indices[0]]
    r, c = selected.gap
    out = g.copy()
    out[r : r + selected.h, c : c + selected.w] = foreground
    return out


# ------------------------------- generator -------------------------------

_SHAPE_PAIRS = (
    ((2, 3), (3, 2)),
    ((2, 4), (3, 2)),
    ((2, 3), (4, 2)),
    ((3, 4), (4, 2)),
)
_MAX_ATTEMPTS = 500


@dataclass(frozen=True)
class _Spec:
    h: int
    w: int
    direction: int
    base: Coord
    slots: tuple[tuple[int, Coord], ...]
    full_ts: tuple[int, ...]
    missing_t: int

    @property
    def missing_anchor(self) -> Coord:
        r0, c0 = self.base
        return r0 + self.missing_t * self.h, c0 + self.direction * self.missing_t * self.w


def _intersects(shape: tuple[int, int], anchor: Coord, h: int, w: int) -> bool:
    height, width = shape
    r, c = anchor
    return r < height and c < width and r + h > 0 and c + w > 0


def _inside(shape: tuple[int, int], anchor: Coord, h: int, w: int) -> bool:
    height, width = shape
    r, c = anchor
    return 0 <= r and 0 <= c and r + h <= height and c + w <= width


def _make_spec(shape: tuple[int, int], rect_shape: Coord) -> _Spec | None:
    height, width = shape
    h, w = rect_shape
    direction = random.choice((-1, 1))
    base_r = random.randint(1, h - 1)  # t=-1 is a visibly top-clipped end stamp.
    base_c = random.randint(0, width - w)
    base = (base_r, base_c)
    before = (base_r - h, base_c - direction * w)
    if not _intersects(shape, before, h, w):
        return None

    slots: list[tuple[int, Coord]] = [(-1, before)]
    for t in range(0, 40):
        anchor = (base_r + t * h, base_c + direction * t * w)
        if not _intersects(shape, anchor, h, w):
            break
        slots.append((t, anchor))
    full_ts = tuple(t for t, anchor in slots if t >= 0 and _inside(shape, anchor, h, w))
    if len(full_ts) < 5 or full_ts != tuple(range(len(full_ts))):
        return None
    internal = [t for t in full_ts if t - 1 in full_ts and t + 1 in full_ts]
    if not internal:
        return None
    return _Spec(h, w, direction, base, tuple(slots), full_ts, random.choice(internal))


def _spec_mask(shape: tuple[int, int], spec: _Spec, *, include_gap: bool) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for t, (r, c) in spec.slots:
        if t == spec.missing_t and not include_gap:
            continue
        rr0, cc0 = max(0, r), max(0, c)
        rr1, cc1 = min(height, r + spec.h), min(width, c + spec.w)
        if rr0 < rr1 and cc0 < cc1:
            mask[rr0:rr1, cc0:cc1] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return out


def _choose_specs(shape: tuple[int, int], pair: tuple[Coord, Coord]) -> tuple[_Spec, _Spec] | None:
    first_options = [spec for _ in range(180) if (spec := _make_spec(shape, pair[0])) is not None]
    second_options = [spec for _ in range(180) if (spec := _make_spec(shape, pair[1])) is not None]
    random.shuffle(first_options)
    random.shuffle(second_options)
    for first in first_options:
        first_intended = _spec_mask(shape, first, include_gap=True)
        forbidden = _dilate8(first_intended)
        for second in second_options:
            second_intended = _spec_mask(shape, second, include_gap=True)
            if not np.any(forbidden & second_intended):
                return first, second
    return None


def _paint_residue(grid: np.ndarray, spec: _Spec, color: int, count: int) -> tuple[Coord, ...]:
    r0, c0 = spec.missing_anchor
    cells = [(r, c) for r in range(spec.h) for c in range(spec.w)]
    non_corners = [
        cell for cell in cells
        if cell not in {(0, 0), (0, spec.w - 1), (spec.h - 1, 0), (spec.h - 1, spec.w - 1)}
    ]
    if count == 1:
        local = [random.choice(non_corners or cells)]
    else:
        pairs = [
            (a, b)
            for a in (non_corners or cells)
            for b in cells
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        ]
        local = list(random.choice(pairs))
    absolute = tuple((r0 + r, c0 + c) for r, c in local)
    for r, c in absolute:
        grid[r, c] = color
    return absolute


def _own_stride_is_necessary(mask: np.ndarray, specs: tuple[_Spec, _Spec]) -> bool:
    """Check the missing boxes are flanked under their own, but not the other, stride."""
    for i, spec in enumerate(specs):
        r, c = spec.missing_anchor
        own_neighbors = (
            (r - spec.h, c - spec.direction * spec.w),
            (r + spec.h, c + spec.direction * spec.w),
        )
        if any(_box_count(mask, anchor, spec.h, spec.w) != spec.h * spec.w for anchor in own_neighbors):
            return False
        other = specs[1 - i]
        wrong_neighbors = (
            (r - other.h, c - spec.direction * other.w),
            (r + other.h, c + spec.direction * other.w),
        )
        wrong_full = []
        for anchor in wrong_neighbors:
            if _inside(mask.shape, anchor, spec.h, spec.w):
                wrong_full.append(_box_count(mask, anchor, spec.h, spec.w) == spec.h * spec.w)
            else:
                wrong_full.append(False)
        if all(wrong_full):
            return False
    return True


def _validate_sample(
    inp: np.ndarray,
    expected: np.ndarray,
    specs: tuple[_Spec, _Spec],
    residue_index: int,
    residue_cells: tuple[Coord, ...],
) -> bool:
    """Semantic checks for uniqueness, selection dependence, and non-degeneracy."""
    try:
        chains = _recover_chains(inp != 0)
        if len(chains) != 2:
            return False
        wanted = {(spec.h, spec.w, spec.direction, spec.missing_anchor) for spec in specs}
        got = {(chain.h, chain.w, chain.direction, chain.gap) for chain in chains}
        if got != wanted or not np.array_equal(main(inp), expected):
            return False

        clean_index = 1 - residue_index
        clean = specs[clean_index]
        residue = specs[residue_index]
        if _box_count(inp != 0, clean.missing_anchor, clean.h, clean.w) != 0:
            return False
        if _box_count(inp != 0, residue.missing_anchor, residue.h, residue.w) != len(residue_cells):
            return False
        if not _own_stride_is_necessary(inp != 0, specs):
            return False

        # Removing the residue must destroy the unique authorization signal.
        no_residue = inp.copy()
        for r, c in residue_cells:
            no_residue[r, c] = 0
        try:
            main(no_residue)
            return False
        except AssertionError:
            pass

        # Moving the same visible residue to the decoy must switch the restored chain.
        moved = no_residue.copy()
        rr, rc = residue.missing_anchor
        cr, cc = clean.missing_anchor
        for r, c in residue_cells:
            moved[cr + (r - rr), cc + (c - rc)] = int(inp[r, c])
        alternate = moved.copy()
        alternate[cr : cr + clean.h, cc : cc + clean.w] = int(inp[residue_cells[0]])
        if not np.array_equal(main(moved), alternate):
            return False

        changed = expected != inp
        target = np.zeros(inp.shape, dtype=bool)
        tr, tc = residue.missing_anchor
        target[tr : tr + residue.h, tc : tc + residue.w] = True
        if not changed.any() or np.any(changed & ~target):
            return False
    except (AssertionError, IndexError, ValueError):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(24, 30), random.randint(24, 30))
    pair = random.choice(_SHAPE_PAIRS)
    specs = _choose_specs(shape, pair)
    if specs is None:
        return None

    color = random.randint(1, 9)
    inp = np.zeros(shape, dtype=int)
    for spec in specs:
        inp[_spec_mask(shape, spec, include_gap=False)] = color

    residue_index = random.randrange(2)
    residue_cells = _paint_residue(inp, specs[residue_index], color, random.choice((1, 2)))
    expected = inp.copy()
    selected = specs[residue_index]
    r, c = selected.missing_anchor
    expected[r : r + selected.h, c : c + selected.w] = color

    if _validate_sample(inp, expected, specs, residue_index, residue_cells):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance of the two-chain residue puzzle."""
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique non-degenerate two-chain sample")
