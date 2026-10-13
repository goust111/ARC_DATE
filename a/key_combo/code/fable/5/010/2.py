"""ARC-AGI puzzle generator and solver for "Projection Stamp"."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
HISTOGRAM = 5
N_COLORS = 10
MAX_GENERATION_ATTEMPTS = 800


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(np.asarray(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bounding box in the common ARC (r0, c0, r1, c1) convention."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _contiguous_support(values: np.ndarray) -> tuple[int, int]:
    nonzero = np.flatnonzero(values)
    assert nonzero.size > 0, "each histogram must have a nonzero support"
    start, end = int(nonzero[0]), int(nonzero[-1])
    assert np.array_equal(nonzero, np.arange(start, end + 1)), "nonzero histogram support must be contiguous"
    return start, end


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    """Locate the unique same-colored full row/full column separator pair."""
    h, w = grid.shape
    full_rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) not in (BACKGROUND, HISTOGRAM) and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    full_cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) not in (BACKGROUND, HISTOGRAM) and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    pairs = [(r, c, rc) for r, rc in full_rows for c, cc in full_cols if rc == cc]
    assert len(pairs) == 1, "there must be one visible separator row/column pair"
    sep_r, sep_c, sep_color = pairs[0]
    assert 0 < sep_r < h - 1 and 0 < sep_c < w - 1
    return sep_r, sep_c, sep_color


def _decode_layout(grid: np.ndarray) -> dict:
    """Read the separator, the two stacked histograms, and the canvas."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    sep_r, sep_c, sep_color = _find_separator(g)

    assert bool(np.all(g[:sep_r, :sep_c] == BACKGROUND)), "the unused top-left corner must be black"
    top = g[:sep_r, sep_c + 1 :]
    left = g[sep_r + 1 :, :sep_c]
    canvas = g[sep_r + 1 :, sep_c + 1 :].copy()
    assert set(map(int, np.unique(top))) <= {BACKGROUND, HISTOGRAM}
    assert set(map(int, np.unique(left))) <= {BACKGROUND, HISTOGRAM}
    assert not bool(np.any(canvas == HISTOGRAM)), "gray is reserved for the two histograms"
    assert not bool(np.any(canvas == sep_color)), "separator color is reserved for the separator"

    col_counts = np.sum(top == HISTOGRAM, axis=0).astype(int)
    row_counts = np.sum(left == HISTOGRAM, axis=1).astype(int)

    # The top bars grow from the outer top edge downward; left bars grow from
    # the outer left edge rightward. Checking this rejects arbitrary gray noise.
    for c, count in enumerate(col_counts.tolist()):
        assert bool(np.all(top[:count, c] == HISTOGRAM))
        assert bool(np.all(top[count:, c] == BACKGROUND))
    for r, count in enumerate(row_counts.tolist()):
        assert bool(np.all(left[r, :count] == HISTOGRAM))
        assert bool(np.all(left[r, count:] == BACKGROUND))

    dst_r0, dst_r1 = _contiguous_support(row_counts)
    dst_c0, dst_c1 = _contiguous_support(col_counts)
    row_key = row_counts[dst_r0 : dst_r1 + 1]
    col_key = col_counts[dst_c0 : dst_c1 + 1]
    assert bool(np.all(row_key > 0)) and bool(np.all(col_key > 0))
    assert int(row_key.sum()) == int(col_key.sum()), "row and column projections must count the same pixels"

    return {
        "separator": (sep_r, sep_c, sep_color),
        "canvas": canvas,
        "row_counts": row_counts,
        "col_counts": col_counts,
        "row_key": row_key,
        "col_key": col_key,
        "destination": (dst_r0, dst_c0),
    }


def _canvas_sprites(canvas: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for coords in _components(canvas != BACKGROUND):
        colors = np.unique(canvas[coords[:, 0], coords[:, 1]])
        assert colors.size == 1 and int(colors[0]) != BACKGROUND, "every sprite must be monochrome"
        mask = _component_mask(coords)
        sprites.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "color": int(colors[0]),
                "mask": mask,
                "rows": mask.sum(axis=1).astype(int),
                "cols": mask.sum(axis=0).astype(int),
            }
        )
    assert sprites, "the canvas must contain candidate sprites"
    return sprites


def _projection_match(sprite: dict, row_key: np.ndarray, col_key: np.ndarray) -> bool:
    return bool(np.array_equal(sprite["rows"], row_key) and np.array_equal(sprite["cols"], col_key))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the uniquely projection-matched sprite and stamp it at the histogram-marked canvas location.

    Input structure:
    - A full separator row and column divide black canvas from top and left margin regions.
    - Gray columns in the top margin and gray rows in the left margin are edge-stacked histograms.
    - Several separated, monochrome candidate sprites lie on the black main canvas.

    Rule:
    - Read each gray bar's length and its canvas-aligned position as column and row projections.
    - Find the unique sprite whose bounding-box-relative row and column projections both match those bars.
    - Erase the margins, separator, distractors, and source sprites; translate the matched sprite's bounding box
      to the intersection of the nonzero histogram intervals and return only that stamped canvas.

    Key insight:
    - Each gray bar is dual-use evidence: its length identifies the shape, while its aligned row or column fixes the destination.
    """
    info = _decode_layout(grid)
    canvas = info["canvas"]
    row_key = info["row_key"]
    col_key = info["col_key"]
    matches = [s for s in _canvas_sprites(canvas) if _projection_match(s, row_key, col_key)]
    assert len(matches) == 1, "row and column projections must select exactly one candidate"

    target = matches[0]
    mask = target["mask"]
    dst_r, dst_c = info["destination"]
    assert mask.shape == (len(row_key), len(col_key))
    assert dst_r + mask.shape[0] <= canvas.shape[0] and dst_c + mask.shape[1] <= canvas.shape[1]

    out = np.full_like(canvas, BACKGROUND)
    rr, cc = np.nonzero(mask)
    out[dst_r + rr, dst_c + cc] = int(target["color"])
    return out


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(mask.sum())


def _make_mask(rng: random.Random, h: int, w: int, *, asymmetric: bool) -> np.ndarray | None:
    """Build a connected cropped mask by joining four boundary anchors, then growing it."""
    for _ in range(180):
        mask = np.zeros((h, w), dtype=bool)
        center = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
        anchors = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        mask[center] = True
        for anchor in anchors:
            r, c = anchor
            mask[r, c] = True
            while (r, c) != center:
                choices: list[tuple[int, int]] = []
                if r != center[0]:
                    choices.append((r + (1 if center[0] > r else -1), c))
                if c != center[1]:
                    choices.append((r, c + (1 if center[1] > c else -1)))
                r, c = rng.choice(choices)
                mask[r, c] = True

        max_area = min(h * w - 2, max(int(mask.sum()), int(round(0.72 * h * w))))
        min_area = min(max_area, max(int(mask.sum()), int(round(0.42 * h * w))))
        desired = rng.randint(min_area, max_area)
        while int(mask.sum()) < desired:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not bool(mask[rr, cc]):
                        frontier.add((rr, cc))
            if not frontier:
                break
            mask[rng.choice(tuple(frontier))] = True

        rows = mask.sum(axis=1).astype(int)
        cols = mask.sum(axis=0).astype(int)
        if not _connected(mask) or int(mask.sum()) >= h * w - 1:
            continue
        if not (bool(mask[0].any()) and bool(mask[-1].any()) and bool(mask[:, 0].any()) and bool(mask[:, -1].any())):
            continue
        if asymmetric and (np.array_equal(rows, rows[::-1]) or np.array_equal(cols, cols[::-1])):
            continue
        return mask
    return None


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    """Whether two half-open rectangles have at least `gap` blank rows or columns between them."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return bool(ar1 + gap <= br0 or br1 + gap <= ar0 or ac1 + gap <= bc0 or bc1 + gap <= ac0)


def _place_rect(
    rng: random.Random,
    canvas_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = canvas_shape
    ph, pw = patch_shape
    options: list[tuple[int, int]] = []
    for r in range(1, h - ph):
        for c in range(1, w - pw):
            rect = (r, c, r + ph, c + pw)
            if all(_separated(rect, other) for other in occupied):
                options.append((r, c))
    return rng.choice(options) if options else None


def _semantic_validate(
    inp: np.ndarray,
    expected_mask: np.ndarray,
    expected_color: int,
    destination: tuple[int, int],
    source_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Check stated semantics and anti-shortcut witnesses, not merely solver closure."""
    info = _decode_layout(inp)
    canvas = info["canvas"]
    row_key = info["row_key"]
    col_key = info["col_key"]
    assert np.array_equal(row_key, expected_mask.sum(axis=1))
    assert np.array_equal(col_key, expected_mask.sum(axis=0))
    assert info["destination"] == destination
    assert not np.array_equal(row_key, row_key[::-1])
    assert not np.array_equal(col_key, col_key[::-1])

    sprites = _canvas_sprites(canvas)
    both = [s for s in sprites if _projection_match(s, row_key, col_key)]
    row_only = [
        s for s in sprites
        if np.array_equal(s["rows"], row_key) and not np.array_equal(s["cols"], col_key)
    ]
    col_only = [
        s for s in sprites
        if np.array_equal(s["cols"], col_key) and not np.array_equal(s["rows"], row_key)
    ]
    assert len(both) == 1, "the correct candidate must be unique"
    assert row_only, "a row-only decoy is required so column counts matter"
    assert col_only, "a column-only decoy is required so row counts matter"
    assert both[0]["color"] == expected_color and np.array_equal(both[0]["mask"], expected_mask)

    dr, dc = destination
    dh, dw = expected_mask.shape
    assert not bool(np.any(canvas[dr : dr + dh, dc : dc + dw] != BACKGROUND)), "destination must be visibly empty"
    sr0, sc0, sr1, sc1 = source_bbox
    assert (sr0, sc0, sr1 - sr0 + 1, sc1 - sc0 + 1) != (dr, dc, dh, dw)

    expected = np.zeros_like(canvas)
    rr, cc = np.nonzero(expected_mask)
    expected[dr + rr, dc + cc] = expected_color
    produced = main(inp)
    assert np.array_equal(produced, expected)
    return expected


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    top_margin = rng.randint(5, 7)
    left_margin = rng.randint(5, 7)
    canvas_h = rng.randint(17, 29 - top_margin)
    canvas_w = rng.randint(17, 29 - left_margin)

    target_h = rng.randint(3, min(6, top_margin))
    target_w = rng.randint(3, min(6, left_margin))
    target = _make_mask(rng, target_h, target_w, asymmetric=True)
    if target is None:
        return None

    # Horizontal reflection preserves every row count but changes column counts;
    # vertical reflection does the converse because both projections are asymmetric.
    candidates = [target, np.fliplr(target), np.flipud(target)]
    for _ in range(rng.randint(0, 2)):
        eh = rng.randint(3, min(6, canvas_h - 3))
        ew = rng.randint(3, min(6, canvas_w - 3))
        extra = _make_mask(rng, eh, ew, asymmetric=False)
        if extra is None:
            return None
        erows = extra.sum(axis=1)
        ecols = extra.sum(axis=0)
        if np.array_equal(erows, target.sum(axis=1)) and np.array_equal(ecols, target.sum(axis=0)):
            return None
        candidates.append(extra)

    separator_color = rng.choice([c for c in range(1, N_COLORS) if c != HISTOGRAM])
    palette = [c for c in range(1, N_COLORS) if c not in (HISTOGRAM, separator_color)]
    colors = rng.sample(palette, len(candidates))

    dst_r = rng.randint(1, canvas_h - target_h - 1)
    dst_c = rng.randint(1, canvas_w - target_w - 1)
    occupied = [(dst_r, dst_c, dst_r + target_h, dst_c + target_w)]
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    order = list(range(len(candidates)))
    rng.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}
    for index in order:
        mask = candidates[index]
        loc = _place_rect(rng, canvas.shape, mask.shape, occupied)
        if loc is None:
            return None
        r0, c0 = loc
        placements[index] = loc
        h, w = mask.shape
        occupied.append((r0, c0, r0 + h, c0 + w))
        view = canvas[r0 : r0 + h, c0 : c0 + w]
        view[mask] = colors[index]

    total_h = top_margin + 1 + canvas_h
    total_w = left_margin + 1 + canvas_w
    inp = np.zeros((total_h, total_w), dtype=int)
    inp[top_margin, :] = separator_color
    inp[:, left_margin] = separator_color
    inp[top_margin + 1 :, left_margin + 1 :] = canvas

    for j, count in enumerate(target.sum(axis=0).astype(int).tolist()):
        col = left_margin + 1 + dst_c + j
        inp[:count, col] = HISTOGRAM
    for i, count in enumerate(target.sum(axis=1).astype(int).tolist()):
        row = top_margin + 1 + dst_r + i
        inp[row, :count] = HISTOGRAM

    src_r, src_c = placements[0]
    source_bbox = (src_r, src_c, src_r + target_h - 1, src_c + target_w - 1)
    try:
        expected = _semantic_validate(inp, target, colors[0], (dst_r, dst_c), source_bbox)
    except (AssertionError, ValueError, IndexError):
        return None
    return inp, expected


_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def generate_input() -> np.ndarray:
    """Generate varied, projection-unique inputs with mandatory one-projection decoys."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(random.Random())
        if result is None:
            continue
        inp, out = result
        in_key = (tuple(map(int, inp.shape)), inp.tobytes())
        out_key = (tuple(map(int, out.shape)), out.tobytes())
        # Avoid repeated examples and, more strongly, avoid different inputs that
        # happen to produce the same output during one generation session.
        if in_key in _SEEN_INPUTS or out_key in _SEEN_OUTPUTS:
            continue
        _SEEN_INPUTS.add(in_key)
        _SEEN_OUTPUTS.add(out_key)
        return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
