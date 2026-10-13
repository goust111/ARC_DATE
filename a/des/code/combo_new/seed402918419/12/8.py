import random
from collections import deque

import numpy as np


BG = 0
QUADRANTS = ("nw", "ne", "sw", "se")
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CANVAS_SIZE = 30
RETRY_LIMIT = 400


def _quadrant_slice(n: int, name: str) -> tuple[slice, slice]:
    k = n // 2
    if name == "nw":
        return slice(0, k), slice(0, k)
    if name == "ne":
        return slice(0, k), slice(k + 1, n)
    if name == "sw":
        return slice(k + 1, n), slice(0, k)
    if name == "se":
        return slice(k + 1, n), slice(k + 1, n)
    raise ValueError(name)


def _to_base(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "nw":
        return mask
    if name == "ne":
        return np.rot90(mask, 1)
    if name == "se":
        return np.rot90(mask, 2)
    if name == "sw":
        return np.rot90(mask, -1)
    raise ValueError(name)


def _from_base(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "nw":
        return mask
    if name == "ne":
        return np.rot90(mask, -1)
    if name == "se":
        return np.rot90(mask, 2)
    if name == "sw":
        return np.rot90(mask, 1)
    raise ValueError(name)


def _color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "size": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _marker_quadrant(marker_r: int, marker_c: int, r0: int, c0: int, n: int) -> str | None:
    k = n // 2
    r1 = r0 + n - 1
    c1 = c0 + n - 1
    if marker_r == r0 - 1 and c0 <= marker_c <= c1:
        if c0 <= marker_c <= c0 + k - 1:
            return "nw"
        if c0 + k + 1 <= marker_c <= c1:
            return "ne"
    if marker_r == r1 + 1 and c0 <= marker_c <= c1:
        if c0 <= marker_c <= c0 + k - 1:
            return "sw"
        if c0 + k + 1 <= marker_c <= c1:
            return "se"
    if marker_c == c0 - 1 and r0 <= marker_r <= r1:
        if r0 <= marker_r <= r0 + k - 1:
            return "nw"
        if r0 + k + 1 <= marker_r <= r1:
            return "sw"
    if marker_c == c1 + 1 and r0 <= marker_r <= r1:
        if r0 <= marker_r <= r0 + k - 1:
            return "ne"
        if r0 + k + 1 <= marker_r <= r1:
            return "se"
    return None


def _ring_is_clear(grid: np.ndarray, marker: tuple[int, int], r0: int, c0: int, n: int) -> bool:
    h, w = grid.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + n + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + n + 1)
    ring = grid[rr0:rr1, cc0:cc1].copy()
    ring[r0 - rr0 : r0 - rr0 + n, c0 - cc0 : c0 - cc0 + n] = BG
    mr, mc = marker
    ring[mr - rr0, mc - cc0] = BG
    return bool(np.all(ring == BG))


def _candidate_for_marker(
    grid: np.ndarray, marker: tuple[int, int, int], r0: int, c0: int, n: int
) -> dict | None:
    h, w = grid.shape
    if not (0 <= r0 <= h - n and 0 <= c0 <= w - n):
        return None
    mr, mc, marker_color = marker
    missing = _marker_quadrant(mr, mc, r0, c0, n)
    if missing is None:
        return None
    if not _ring_is_clear(grid, (mr, mc), r0, c0, n):
        return None

    square = grid[r0 : r0 + n, c0 : c0 + n]
    colors = [int(color) for color in np.unique(square) if int(color) != BG]
    if len(colors) != 1:
        return None
    fg = colors[0]
    if fg == marker_color:
        return None

    quadrants = {}
    for name in QUADRANTS:
        rs, cs = _quadrant_slice(n, name)
        quadrants[name] = square[rs, cs] == fg

    if bool(quadrants[missing].any()):
        return None

    base = None
    for name in QUADRANTS:
        if name == missing:
            continue
        if not bool(quadrants[name].any()):
            return None
        canonical = _to_base(quadrants[name], name)
        if base is None:
            base = canonical
        elif not np.array_equal(base, canonical):
            return None

    assert base is not None
    predicted = _from_base(base, missing)
    if not bool(predicted.any()):
        return None

    edge_touch = sum(
        [
            bool((square[0, :] != BG).any()),
            bool((square[-1, :] != BG).any()),
            bool((square[:, 0] != BG).any()),
            bool((square[:, -1] != BG).any()),
        ]
    )
    score = (edge_touch, int((square != BG).sum()), -n)
    return {
        "marker": marker,
        "bbox": (r0, c0, n),
        "missing": missing,
        "predicted": predicted,
        "score": score,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each rotationally symmetric square by restoring its missing quadrant in the marker's replacement color.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping odd-sized square regions whose visible pixels use black plus one foreground color.
    - Inside each square, the complete pattern would have fourfold rotational symmetry around the square center, but exactly one quadrant has been erased to black.
    - A single 1x1 marker pixel of a different non-black color sits immediately outside the square along the outer side that corresponds to the missing quadrant.

    Rule:
    - For each marker, identify the adjacent odd square whose three visible quadrants agree under 90-degree rotation.
    - Infer the erased quadrant by rotating one of the present quadrants into the missing position.
    - Paint the inferred foreground pixels in the missing quadrant using the marker color instead of the square's original foreground color, while leaving inferred black pixels black.
    - Erase every marker pixel back to black.

    Key insight:
    - The marker position reveals which quadrant is missing, and the marker color gives the replacement color, while the shape itself must be reconstructed from the other three quadrants' rotational agreement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    markers = []
    for comp in _color_components(g):
        if comp["size"] == 1:
            r, c = comp["coords"][0]
            markers.append((int(r), int(c), int(comp["color"])))

    candidates = []
    for marker in markers:
        best = None
        for n in range(5, min(h, w) + 1, 2):
            for r0 in range(h - n + 1):
                for c0 in range(w - n + 1):
                    cand = _candidate_for_marker(g, marker, r0, c0, n)
                    if cand is not None and (best is None or cand["score"] > best["score"]):
                        best = cand
        if best is not None:
            candidates.append(best)

    used = np.zeros((h, w), dtype=bool)
    for cand in sorted(candidates, key=lambda item: item["score"], reverse=True):
        r0, c0, n = cand["bbox"]
        if bool(used[r0 : r0 + n, c0 : c0 + n].any()):
            continue
        mr, mc, marker_color = cand["marker"]
        rs, cs = _quadrant_slice(n, cand["missing"])
        view = out[r0 : r0 + n, c0 : c0 + n]
        view[rs, cs] = np.where(cand["predicted"], marker_color, BG)
        out[mr, mc] = BG
        used[r0 : r0 + n, c0 : c0 + n] = True
    return out


def _grow_seed(k: int, rng: random.Random) -> np.ndarray:
    min_cells = max(3, k + 1)
    max_cells = max(min_cells, k * k - 1)
    target = rng.randint(min_cells, max_cells)
    mask = np.zeros((k, k), dtype=bool)
    mask[k - 1, k - 1] = True

    while int(mask.sum()) < target:
        cells = np.argwhere(mask)
        rr, cc = map(int, cells[rng.randrange(len(cells))])
        dr, dc = NEIGHBORS4[rng.randrange(4)]
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < k and 0 <= nc < k:
            mask[nr, nc] = True

    rr, cc = map(int, np.argwhere(mask)[rng.randrange(int(mask.sum()))])
    while rr > 0:
        rr -= 1
        mask[rr, cc] = True
    rr, cc = map(int, np.argwhere(mask)[rng.randrange(int(mask.sum()))])
    while cc > 0:
        cc -= 1
        mask[rr, cc] = True

    if bool(mask.all()):
        removable = [(r, c) for r in range(k) for c in range(k) if (r, c) != (k - 1, k - 1)]
        rng.shuffle(removable)
        for r, c in removable:
            trial = mask.copy()
            trial[r, c] = False
            if trial[0, :].any() and trial[:, 0].any():
                mask = trial
                break
    return mask


def _make_full_mask(k: int, seed: np.ndarray, rng: random.Random) -> np.ndarray:
    n = 2 * k + 1
    center = k
    mask = np.zeros((n, n), dtype=bool)
    arm = rng.randint(1, k)
    mask[center, center - arm : center + arm + 1] = True
    mask[center - arm : center + arm + 1, center] = True
    for name in QUADRANTS:
        rs, cs = _quadrant_slice(n, name)
        mask[rs, cs] = _from_base(seed, name)
    return mask


def _marker_cell(r0: int, c0: int, n: int, missing: str, side: str, rng: random.Random) -> tuple[int, int]:
    k = n // 2
    if missing == "nw" and side == "top":
        return r0 - 1, c0 + rng.randrange(k)
    if missing == "nw" and side == "left":
        return r0 + rng.randrange(k), c0 - 1
    if missing == "ne" and side == "top":
        return r0 - 1, c0 + k + 1 + rng.randrange(k)
    if missing == "ne" and side == "right":
        return r0 + rng.randrange(k), c0 + n
    if missing == "sw" and side == "bottom":
        return r0 + n, c0 + rng.randrange(k)
    if missing == "sw" and side == "left":
        return r0 + k + 1 + rng.randrange(k), c0 - 1
    if missing == "se" and side == "bottom":
        return r0 + n, c0 + k + 1 + rng.randrange(k)
    if missing == "se" and side == "right":
        return r0 + k + 1 + rng.randrange(k), c0 + n
    raise ValueError((missing, side))


def _render_object(
    fg_color: int, marker_color: int, n: int, missing: str, side: str, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    k = n // 2
    seed = _grow_seed(k, rng)
    full_mask = _make_full_mask(k, seed, rng)
    inp = np.where(full_mask, fg_color, BG)
    out = inp.copy()
    rs, cs = _quadrant_slice(n, missing)
    inp[rs, cs] = BG
    out[rs, cs] = np.where(_from_base(seed, missing), marker_color, BG)

    marker = _marker_cell(0, 0, n, missing, side, rng)
    return inp, out, marker


def _reserve_bounds(r0: int, c0: int, n: int, marker: tuple[int, int]) -> tuple[int, int, int, int]:
    mr, mc = marker
    rr0 = min(r0, mr) - 2
    rr1 = max(r0 + n - 1, mr) + 2
    cc0 = min(c0, mc) - 2
    cc1 = max(c0 + n - 1, mc) + 2
    return rr0, rr1, cc0, cc1


def _crop_with_margin(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = rng.randint(0, min(3, int(r0)) + 1)
    left = rng.randint(0, min(3, int(c0)) + 1)
    bottom = rng.randint(0, min(3, int(inp.shape[0] - 1 - r1)) + 1)
    right = rng.randint(0, min(3, int(inp.shape[1] - 1 - c1)) + 1)
    rr0 = int(r0 - top)
    rr1 = int(r1 + bottom + 1)
    cc0 = int(c0 - left)
    cc1 = int(c1 + right + 1)
    return inp[rr0:rr1, cc0:cc1], out[rr0:rr1, cc0:cc1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    out = np.zeros_like(inp)
    reserved = np.zeros_like(inp, dtype=bool)

    object_count = rng.choices([1, 2, 3], weights=[2, 4, 3])[0]
    max_size = {1: 13, 2: 11, 3: 9}[object_count]
    size_choices = [size for size in (5, 7, 9, 11, 13) if size <= max_size]
    sizes = sorted([rng.choice(size_choices) for _ in range(object_count)], reverse=True)
    palette = rng.sample(list(range(1, 10)), 2 * object_count)

    color_pairs = [(palette[2 * i], palette[2 * i + 1]) for i in range(object_count)]
    specs = []
    for idx, n in enumerate(sizes):
        fg_color, marker_color = color_pairs[idx]
        missing = rng.choice(list(QUADRANTS))
        side_choices = {
            "nw": ["top", "left"],
            "ne": ["top", "right"],
            "sw": ["bottom", "left"],
            "se": ["bottom", "right"],
        }
        side = rng.choice(side_choices[missing])
        local_inp, local_out, local_marker = _render_object(fg_color, marker_color, n, missing, side, rng)
        specs.append((n, local_inp, local_out, local_marker, marker_color))

    for n, local_inp, local_out, local_marker, marker_color in specs:
        placed = False
        for _ in range(200):
            r0 = rng.randrange(1, CANVAS_SIZE - n)
            c0 = rng.randrange(1, CANVAS_SIZE - n)
            mr, mc = r0 + local_marker[0], c0 + local_marker[1]
            if not (0 <= mr < CANVAS_SIZE and 0 <= mc < CANVAS_SIZE):
                continue
            rr0, rr1, cc0, cc1 = _reserve_bounds(r0, c0, n, (mr, mc))
            if rr0 < 0 or cc0 < 0 or rr1 >= CANVAS_SIZE or cc1 >= CANVAS_SIZE:
                continue
            if bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue

            inp[r0 : r0 + n, c0 : c0 + n] = local_inp
            out[r0 : r0 + n, c0 : c0 + n] = local_out
            inp[mr, mc] = marker_color
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    cropped_inp, cropped_out = _crop_with_margin(inp, out, rng)
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
