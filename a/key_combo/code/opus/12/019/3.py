from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BORDER_COLORS = [BLUE, GREEN, YELLOW, GREY, PINK, TEAL, MAROON]
GENERATION_RETRIES = 600


@dataclass(frozen=True)
class Source:
    bbox: tuple[int, int, int, int]
    border_color: int
    indicator: tuple[int, int]
    direction: str
    stripe_axis: str
    period: int
    residue: int
    weight: int


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_row_pattern(inner: np.ndarray) -> np.ndarray:
    orange_mask = inner == ORANGE
    row_flags = np.all(inner == ORANGE, axis=1)
    if np.array_equal(orange_mask, np.repeat(row_flags[:, None], inner.shape[1], axis=1)):
        return row_flags
    return np.array([], dtype=bool)


def _full_col_pattern(inner: np.ndarray) -> np.ndarray:
    orange_mask = inner == ORANGE
    col_flags = np.all(inner == ORANGE, axis=0)
    if np.array_equal(orange_mask, np.repeat(col_flags[None, :], inner.shape[0], axis=0)):
        return col_flags
    return np.array([], dtype=bool)


def _infer_period(indices: np.ndarray, length: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for period in (2, 3):
        residues = {int(i) % period for i in indices}
        if len(residues) != 1:
            continue
        residue = next(iter(residues))
        expected = np.array([i for i in range(length) if i % period == residue], dtype=int)
        if np.array_equal(indices, expected):
            candidates.append((period, residue))
    assert len(candidates) == 1, "stripe period must be uniquely inferable"
    return candidates[0]


def _parse_source(grid: np.ndarray, coords: np.ndarray) -> Source:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 5 and w >= 5, "source rectangle must have a real interior pattern"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    perimeter = np.zeros_like(sub, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    reds = np.argwhere((sub == RED) & perimeter)
    assert len(reds) == 1, "each source must have exactly one red border indicator"
    ir, ic = map(int, reds[0])
    assert (ir in (0, h - 1)) ^ (ic in (0, w - 1)), "indicator may not sit on a corner"

    border_colors = sorted(set(map(int, sub[perimeter])) - {RED})
    assert len(border_colors) == 1, "border color must be unique within each source"
    border_color = border_colors[0]
    assert border_color not in (BG, RED, ORANGE)
    assert np.all(np.isin(sub[perimeter], [border_color, RED]))

    inner = sub[1:-1, 1:-1]
    assert inner.size > 0
    assert np.all(np.isin(inner, [BG, ORANGE]))
    assert np.any(inner == ORANGE) and np.any(inner == BG)

    row_flags = _full_row_pattern(inner)
    col_flags = _full_col_pattern(inner)
    horizontal_ok = row_flags.size > 0 and np.count_nonzero(row_flags) >= 2
    vertical_ok = col_flags.size > 0 and np.count_nonzero(col_flags) >= 2
    assert horizontal_ok ^ vertical_ok, "stripe orientation must be unique"

    if ir == 0:
        direction = "up"
    elif ir == h - 1:
        direction = "down"
    elif ic == 0:
        direction = "left"
    else:
        direction = "right"

    if horizontal_ok:
        stripe_axis = "horizontal"
        assert direction in {"up", "down"}, "horizontal row stripes must project vertically"
        indices = np.flatnonzero(row_flags)
        period, residue = _infer_period(indices, inner.shape[0])
    else:
        stripe_axis = "vertical"
        assert direction in {"left", "right"}, "vertical column stripes must project horizontally"
        indices = np.flatnonzero(col_flags)
        period, residue = _infer_period(indices, inner.shape[1])

    return Source(
        bbox=(r0, c0, r1, c1),
        border_color=border_color,
        indicator=(r0 + ir, c0 + ic),
        direction=direction,
        stripe_axis=stripe_axis,
        period=period,
        residue=residue,
        weight=int(np.count_nonzero(inner == ORANGE)),
    )


def _extract_sources(grid: np.ndarray) -> list[Source]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components(g != BG)
    assert 2 <= len(comps) <= 3, "expected 2 or 3 source rectangles"
    sources = [_parse_source(g, comp) for comp in comps]
    weights = [src.weight for src in sources]
    assert len(weights) == len(set(weights)), "source stripe weights must be unique"
    return sorted(sources, key=lambda src: src.bbox)


def _selected_projection_indices(src: Source, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = src.bbox
    if src.stripe_axis == "horizontal":
        if src.direction == "up":
            rows = np.arange(0, r0, dtype=int)
        else:
            rows = np.arange(r1 + 1, h, dtype=int)
        if rows.size == 0:
            return rows
        rel = rows - (r0 + 1)
        return rows[(rel % src.period) == src.residue]

    if src.direction == "left":
        cols = np.arange(0, c0, dtype=int)
    else:
        cols = np.arange(c1 + 1, w, dtype=int)
    if cols.size == 0:
        return cols
    rel = cols - (c0 + 1)
    return cols[(rel % src.period) == src.residue]


def _rectangle_mask(shape: tuple[int, int], sources: list[Source]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for src in sources:
        r0, c0, r1, c1 = src.bbox
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _compose_output(grid: np.ndarray, sources: list[Source]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    rect_mask = _rectangle_mask(g.shape, sources)

    best_weight = np.full(g.shape, -1, dtype=int)
    best_color = np.zeros(g.shape, dtype=int)

    for src in sorted(sources, key=lambda item: item.weight):
        selected = _selected_projection_indices(src, g.shape)
        if src.stripe_axis == "horizontal":
            for row in selected:
                current = src.weight > best_weight[row]
                best_weight[row, current] = src.weight
                best_color[row, current] = src.border_color
        else:
            for col in selected:
                current = src.weight > best_weight[:, col]
                best_weight[current, col] = src.weight
                best_color[current, col] = src.border_color

    proj_mask = (best_weight >= 0) & ~rect_mask
    out[proj_mask] = best_color[proj_mask]

    for src in sources:
        r, c = src.indicator
        dr, dc = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[src.direction]
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = RED
            rr += dr
            cc += dc

    return out


def _projection_overlap_count(sources: list[Source], shape: tuple[int, int]) -> int:
    h, w = shape
    rect_mask = _rectangle_mask(shape, sources)
    cover = np.zeros(shape, dtype=int)
    for src in sources:
        selected = _selected_projection_indices(src, shape)
        if src.stripe_axis == "horizontal":
            for row in selected:
                cover[row, :] += 1
        else:
            for col in selected:
                cover[:, col] += 1
    return int(np.count_nonzero((cover >= 2) & ~rect_mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each indicator-marked stripe source toward the canvas edge and resolve stripe overlaps by source stripe count.

    Input structure:
    - The grid has a black background and 2-3 disjoint framed rectangles.
    - Each rectangle has a one-pixel border of one color, except for exactly one red border indicator pixel.
    - Inside each rectangle, orange stripes occupy either periodic full interior rows or periodic full interior columns, while the other interior cells stay black.

    Rule:
    - For each rectangle, use the side containing the red indicator to determine whether its pattern projects up, down, left, or right.
    - Infer the stripe period from the interior orange rows or columns, then continue that periodic mask from the rectangle toward the chosen canvas edge, painting the projected rows or columns with the rectangle's border color.
    - Draw a red ray from each indicator pixel to the same edge, keep every original rectangle unchanged, and when projected colored regions overlap, keep the one whose source rectangle contains more orange stripe pixels.

    Key insight:
    - The orange stripes play two roles at once: they define both the periodic projection mask and the priority weight that decides which projected color wins in crossings.
    """
    sources = _extract_sources(np.asarray(grid, dtype=int))
    return _compose_output(np.asarray(grid, dtype=int), sources)


def _valid_indices(length: int, period: int) -> list[int]:
    return [offset for offset in range(period) if 2 <= len(range(offset, length, period)) < length]


def _sample_source_spec(kind: str, used_weights: set[int]) -> Source | None:
    for _ in range(120):
        if kind == "horizontal":
            height = random.randint(5, 7)
            width = random.randint(5, 8)
            inner_len = height - 2
            span = width - 2
            direction = random.choice(["up", "down"])
        else:
            height = random.randint(5, 8)
            width = random.randint(5, 7)
            inner_len = width - 2
            span = height - 2
            direction = random.choice(["left", "right"])
        period = random.choice([2, 3])
        valid_residues = _valid_indices(inner_len, period)
        if not valid_residues:
            continue
        residue = random.choice(valid_residues)
        count = len(range(residue, inner_len, period))
        weight = count * span
        if weight in used_weights:
            continue
        return Source(
            bbox=(0, 0, height - 1, width - 1),
            border_color=-1,
            indicator=(-1, -1),
            direction=direction,
            stripe_axis=kind,
            period=period,
            residue=residue,
            weight=weight,
        )
    return None


def _placement_ranges(shape: tuple[int, int], proto: Source) -> tuple[range, range]:
    h, w = shape
    ph = proto.bbox[2] - proto.bbox[0] + 1
    pw = proto.bbox[3] - proto.bbox[1] + 1

    row_lo = 1
    row_hi = h - ph - 1
    col_lo = 1
    col_hi = w - pw - 1
    if proto.direction == "up":
        row_lo = max(row_lo, h // 2 - 1)
    elif proto.direction == "down":
        row_hi = min(row_hi, h // 2 - ph)
    elif proto.direction == "left":
        col_lo = max(col_lo, w // 2 - 1)
    else:
        col_hi = min(col_hi, w // 2 - pw)

    if row_lo > row_hi or col_lo > col_hi:
        return range(0), range(0)
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _line_crosses_other_rectangles(src: Source, others: list[Source], shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = src.indicator
    dr, dc = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[src.direction]
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        for other in others:
            r0, c0, r1, c1 = other.bbox
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                return True
        rr += dr
        cc += dc
    return False


def _place_sources(shape: tuple[int, int], protos: list[Source]) -> list[Source] | None:
    placed: list[Source] = []
    colors = random.sample(BORDER_COLORS, len(protos))
    order = sorted(range(len(protos)), key=lambda idx: 0 if protos[idx].stripe_axis == "horizontal" else 1)
    for color, idx in zip(colors, order):
        proto = protos[idx]
        rows, cols = _placement_ranges(shape, proto)
        if not rows or not cols:
            return None
        candidates: list[Source] = []
        ph = proto.bbox[2] - proto.bbox[0] + 1
        pw = proto.bbox[3] - proto.bbox[1] + 1
        for r0 in rows:
            for c0 in cols:
                r1 = r0 + ph - 1
                c1 = c0 + pw - 1
                ok = True
                for other in placed:
                    or0, oc0, or1, oc1 = other.bbox
                    if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
                        ok = False
                        break
                if not ok:
                    continue
                if proto.direction in {"up", "down"}:
                    indicator = (r0 if proto.direction == "up" else r1, random.randint(c0 + 1, c1 - 1))
                else:
                    indicator = (random.randint(r0 + 1, r1 - 1), c0 if proto.direction == "left" else c1)
                candidate = Source(
                    bbox=(r0, c0, r1, c1),
                    border_color=color,
                    indicator=indicator,
                    direction=proto.direction,
                    stripe_axis=proto.stripe_axis,
                    period=proto.period,
                    residue=proto.residue,
                    weight=proto.weight,
                )
                candidates.append(candidate)
        random.shuffle(candidates)
        chosen = None
        for candidate in candidates:
            if not _line_crosses_other_rectangles(candidate, placed, shape):
                chosen = candidate
                break
        if chosen is None:
            return None
        placed.append(chosen)
    return sorted(placed, key=lambda src: src.bbox)


def _render_input(shape: tuple[int, int], sources: list[Source]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for src in sources:
        r0, c0, r1, c1 = src.bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG
        grid[r0, c0 : c1 + 1] = src.border_color
        grid[r1, c0 : c1 + 1] = src.border_color
        grid[r0 : r1 + 1, c0] = src.border_color
        grid[r0 : r1 + 1, c1] = src.border_color

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if src.stripe_axis == "horizontal":
            for rr in range(inner_h):
                if rr % src.period == src.residue:
                    grid[r0 + 1 + rr, c0 + 1 : c1] = ORANGE
        else:
            for cc in range(inner_w):
                if cc % src.period == src.residue:
                    grid[r0 + 1 : r1, c0 + 1 + cc] = ORANGE

        ir, ic = src.indicator
        grid[ir, ic] = RED
    return grid


def _compare_sources(expected: list[Source], actual: list[Source]) -> bool:
    if len(expected) != len(actual):
        return False
    for lhs, rhs in zip(expected, actual):
        if lhs != rhs:
            return False
    return True


def _sample_prototypes(force_conflict: bool) -> list[Source] | None:
    count = random.choice([2, 3])
    kinds: list[str]
    if force_conflict:
        kinds = ["horizontal", "vertical"]
        if count == 3:
            kinds.append(random.choice(["horizontal", "vertical"]))
    else:
        if count == 2 and random.random() < 0.5:
            kinds = [random.choice(["horizontal", "vertical"])] * 2
        else:
            kinds = [random.choice(["horizontal", "vertical"]) for _ in range(count)]
    used_weights: set[int] = set()
    protos: list[Source] = []
    for kind in kinds:
        proto = _sample_source_spec(kind, used_weights)
        if proto is None:
            return None
        used_weights.add(proto.weight)
        protos.append(proto)
    return protos


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    force_conflict = random.random() < 0.7
    protos = _sample_prototypes(force_conflict)
    if protos is None:
        return None
    sources = _place_sources(shape, protos)
    if sources is None:
        return None

    grid = _render_input(shape, sources)
    try:
        parsed = _extract_sources(grid)
    except AssertionError:
        return None
    if not _compare_sources(sources, parsed):
        return None

    overlap = _projection_overlap_count(sources, shape)
    if force_conflict and overlap == 0:
        return None
    if not force_conflict and len(sources) == 2 and overlap > (shape[0] + shape[1]) // 3:
        return None

    expected = _compose_output(grid, sources)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
