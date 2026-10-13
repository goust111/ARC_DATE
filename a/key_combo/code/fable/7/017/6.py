from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class _Model:
    n: int
    d: int
    direction: tuple[int, int]
    origin: tuple[int, int]
    color: int
    mask_bytes: bytes

    def mask(self) -> np.ndarray:
        return np.frombuffer(self.mask_bytes, dtype=np.uint8).reshape(self.n, self.n).astype(bool)


def _rot(mask: np.ndarray, turns: int) -> np.ndarray:
    """Rotate clockwise by ``turns`` quarter turns."""
    return np.rot90(mask, -(turns % 4))


def _inside_square(r: int, c: int, top: int, left: int, n: int) -> bool:
    return top <= r < top + n and left <= c < left + n


def _stamp(
    canvas: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    """Opaque clipped stamp: false sprite cells paint background too."""
    h, w = canvas.shape
    n = mask.shape[0]
    r0, r1 = max(0, top), min(h, top + n)
    c0, c1 = max(0, left), min(w, left + n)
    if r0 >= r1 or c0 >= c1:
        return
    sub = mask[r0 - top : r1 - top, c0 - left : c1 - left]
    canvas[r0:r1, c0:c1] = np.where(sub, color, BG)


def _render_first_two(shape, mask, origin, d, direction, color) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    dr, dc = direction
    r0, c0 = origin
    _stamp(out, mask, r0, c0, color)
    _stamp(out, _rot(mask, 1), r0 + dr * d, c0 + dc * d, color)
    return out


def _render_all(shape, mask, origin, d, direction, color) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    h, w = shape
    dr, dc = direction
    r0, c0 = origin
    # A monotone diagonal trajectory can intersect the canvas only in one interval.
    for k in range(80):
        top, left = r0 + k * dr * d, c0 + k * dc * d
        if k >= 2 and (top + mask.shape[0] <= 0 or top >= h or left + mask.shape[0] <= 0 or left >= w):
            break
        _stamp(out, _rot(mask, k), top, left, color)
    else:
        raise AssertionError("stamp trajectory did not leave the canvas")
    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1] and mask[q] and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == int(mask.sum())


def _valid_sprite(mask: np.ndarray) -> bool:
    if not _connected(mask):
        return False
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if mask.sum() < mask.size * 0.25 or mask.sum() > mask.size * 0.68:
        return False
    return not any(np.array_equal(mask, _rot(mask, k)) for k in (1, 2, 3))


def _recover_for_geometry(
    g: np.ndarray,
    n: int,
    d: int,
    direction: tuple[int, int],
    origin: tuple[int, int],
    color: int,
) -> np.ndarray | None:
    """Recover every local sprite cell from the two opaque visible residues."""
    h, w = g.shape
    dr, dc = direction
    a_top, a_left = origin
    b_top, b_left = a_top + dr * d, a_left + dc * d
    values = np.full((n, n), -1, dtype=np.int8)

    for r in range(h):
        for c in range(w):
            local = None
            if _inside_square(r, c, b_top, b_left, n):
                # rotCW(mask)[u,v] == mask[n-1-v,u]
                u, v = r - b_top, c - b_left
                local = (n - 1 - v, u)
            elif _inside_square(r, c, a_top, a_left, n):
                local = (r - a_top, c - a_left)
            elif int(g[r, c]) != BG:
                return None

            if local is not None:
                observed = 1 if int(g[r, c]) == color else 0
                lr, lc = local
                if values[lr, lc] >= 0 and values[lr, lc] != observed:
                    return None
                values[lr, lc] = observed

    if np.any(values < 0):
        return None
    mask = values.astype(bool)
    if not _valid_sprite(mask):
        return None
    if not np.array_equal(_render_first_two(g.shape, mask, origin, d, direction, color), g):
        return None
    return mask


def _models(grid: np.ndarray) -> list[_Model]:
    g = np.asarray(grid, dtype=int)
    colors = [int(x) for x in np.unique(g) if int(x) != BG]
    assert len(colors) == 1, "expected one monochrome ink color on black background"
    color = colors[0]
    h, w = g.shape
    foreground = [tuple(map(int, p)) for p in np.argwhere(g != BG)]
    found: list[_Model] = []

    # Sizes are deliberately inferred, not encoded by canvas dimensions or color IDs.
    for n in range(4, min(8, h, w) + 1):
        for d in range(1, n):
            for direction in DIAGONALS:
                dr, dc = direction
                # The first square is wholly visible; the advancing second
                # square is clipped on exactly one of its two forward edges.
                for r0 in range(0, h - n + 1):
                    r1 = r0 + dr * d
                    for c0 in range(0, w - n + 1):
                        c1 = c0 + dc * d
                        row_clip = not (0 <= r1 and r1 + n <= h)
                        col_clip = not (0 <= c1 and c1 + n <= w)
                        if row_clip == col_clip:
                            continue
                        if row_clip and not (r1 < 0 if dr < 0 else r1 + n > h):
                            continue
                        if col_clip and not (c1 < 0 if dc < 0 else c1 + n > w):
                            continue
                        if any(
                            not (_inside_square(r, c, r0, c0, n) or _inside_square(r, c, r1, c1, n))
                            for r, c in foreground
                        ):
                            continue
                        mask = _recover_for_geometry(g, n, d, direction, (r0, c0), color)
                        if mask is not None:
                            found.append(_Model(n, d, direction, (r0, c0), color, mask.astype(np.uint8).tobytes()))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover two overprinted rotating stamps and continue their diagonal trajectory until it leaves the grid.

    Input structure:
    A black canvas contains one monochrome n-by-n sprite stamped twice.  The
    second opaque stamp is shifted d cells along a diagonal, rotated clockwise
    by ninety degrees, and clipped at the advancing edge; its whole square,
    including empty sprite cells, covers the first stamp.

    Rule:
    Enumerate the square size, diagonal direction, displacement, and first
    origin.  Rotate the second residue back and combine its exposed local cells
    with the unoccluded cells of the first residue; the valid input has exactly
    one complete connected, edge-touching, non-rotationally-symmetric sprite.
    Starting at the second stamp, repeatedly translate by the same diagonal
    vector and rotate clockwise, applying each opaque stamp after the previous
    one, until the next square is wholly outside the canvas.

    Key insight:
    Neither visible residue is the generator: their complementary exposures
    recover it in one local coordinate frame, after which both position and
    orientation must be extrapolated while respecting later-over-earlier ink.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    candidates = _models(g)
    assert len(candidates) == 1, "stamp evidence must determine one geometry and one sprite"
    model = candidates[0]
    return _render_all(g.shape, model.mask(), model.origin, model.d, model.direction, model.color)


def _random_sprite(n: int) -> np.ndarray | None:
    # Grow a connected irregular blob, then punch only removable boundary-safe cells.
    target = random.randint(max(n + 3, int(n * n * 0.30)), int(n * n * 0.58))
    seed = (random.randrange(n), random.randrange(n))
    cells = {seed}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= q[0] < n and 0 <= q[1] < n and q not in cells:
                    frontier.append(q)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((n, n), dtype=bool)
    for p in cells:
        mask[p] = True
    return mask if _valid_sprite(mask) else None


def _coverage_complete(shape, n, origin, d, direction) -> bool:
    h, w = shape
    dr, dc = direction
    a_top, a_left = origin
    b_top, b_left = a_top + dr * d, a_left + dc * d
    seen = np.zeros((n, n), dtype=bool)
    for r in range(h):
        for c in range(w):
            if _inside_square(r, c, b_top, b_left, n):
                u, v = r - b_top, c - b_left
                seen[n - 1 - v, u] = True
            elif _inside_square(r, c, a_top, a_left, n):
                seen[r - a_top, c - a_left] = True
    return bool(seen.all())


def _attempt() -> np.ndarray | None:
    h, w = random.randint(15, 24), random.randint(15, 24)
    n = random.randint(5, 7)
    d = random.randint(2, n - 1)
    direction = random.choice(DIAGONALS)
    dr, dc = direction
    mask = _random_sprite(n)
    if mask is None:
        return None
    color = random.choice(PALETTE)

    # Pick a fully visible first stamp and a second stamp clipped by 1..d
    # cells at one advancing edge. Trying both axes balances horizontal and
    # vertical clipping and lets rotation create complementary evidence.
    origins = []
    for exit_axis in random.sample((0, 1), 2):
        for clip in range(1, d + 1):
            if exit_axis == 0:
                b_top = h - n + clip if dr > 0 else -clip
                row0 = b_top - dr * d
                for b_left in range(0, w - n + 1):
                    col0 = b_left - dc * d
                    origins.append((row0, col0))
            else:
                b_left = w - n + clip if dc > 0 else -clip
                col0 = b_left - dc * d
                for b_top in range(0, h - n + 1):
                    row0 = b_top - dr * d
                    origins.append((row0, col0))
    random.shuffle(origins)

    for origin in origins[:80]:
        r0, c0 = origin
        if not (0 <= r0 <= h - n and 0 <= c0 <= w - n):
            continue
        if not _coverage_complete((h, w), n, origin, d, direction):
            continue
        inp = _render_first_two((h, w), mask, origin, d, direction, color)
        if np.count_nonzero(inp) < 8 or np.count_nonzero(inp == BG) <= inp.size // 2:
            continue

        # Semantic anti-shortcut checks: each source, rotation, translation,
        # continuation, and chronological opaque overlay changes visible cells.
        first_only = np.zeros((h, w), dtype=int)
        _stamp(first_only, mask, r0, c0, color)
        second_only = np.zeros((h, w), dtype=int)
        _stamp(second_only, _rot(mask, 1), r0 + dr * d, c0 + dc * d, color)
        wrong_rot = np.zeros((h, w), dtype=int)
        _stamp(wrong_rot, mask, r0, c0, color)
        _stamp(wrong_rot, mask, r0 + dr * d, c0 + dc * d, color)
        reversed_order = np.zeros((h, w), dtype=int)
        _stamp(reversed_order, _rot(mask, 1), r0 + dr * d, c0 + dc * d, color)
        _stamp(reversed_order, mask, r0, c0, color)
        out = _render_all((h, w), mask, origin, d, direction, color)
        if any(np.array_equal(inp, x) for x in (first_only, second_only, wrong_rot, reversed_order)):
            continue
        if np.array_equal(inp, out):
            continue

        models = _models(inp)
        if len(models) != 1:
            continue
        m = models[0]
        if (m.n, m.d, m.direction, m.origin) != (n, d, direction, origin):
            continue
        if not np.array_equal(m.mask(), mask):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique non-degenerate rotating-stamp input")
