from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_gray_square(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h == side_w and coords.shape[0] == side_h * side_w:
            candidates.append((r0, c0, side_h))
    assert len(candidates) == 1, "expected one solid gray square"
    return candidates[0]


def _line_options(grid: np.ndarray, coords: np.ndarray) -> list[dict]:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        orientation = "h"
        length = c1 - c0 + 1
        if coords.shape[0] != length or length < 2:
            return []
        seq = list(map(int, grid[r0, c0 : c1 + 1]))
    elif c0 == c1:
        orientation = "v"
        length = r1 - r0 + 1
        if coords.shape[0] != length or length < 2:
            return []
        seq = list(map(int, grid[r0 : r1 + 1, c0]))
    else:
        return []

    unique = set(seq)
    if len(unique) == 1:
        return [
            {
                "orientation": orientation,
                "body_color": seq[0],
                "body_len": length,
                "next_color": None,
            }
        ]

    if len(unique) != 2 or length < 3:
        return []

    opts: list[dict] = []
    if seq[0] != seq[1] and all(color == seq[1] for color in seq[1:]):
        opts.append(
            {
                "orientation": orientation,
                "body_color": seq[1],
                "body_len": length - 1,
                "next_color": seq[0],
            }
        )
    if seq[-1] != seq[-2] and all(color == seq[0] for color in seq[:-1]):
        opts.append(
            {
                "orientation": orientation,
                "body_color": seq[0],
                "body_len": length - 1,
                "next_color": seq[-1],
            }
        )
    return opts


def _build_solutions(grid: np.ndarray, side: int) -> list[tuple[str, list[dict]]]:
    mask = (grid != BG) & (grid != GREY)
    parsed: list[dict] = []
    for coords in _components(mask):
        options = _line_options(grid, coords)
        if len(options) == 1:
            parsed.append(options[0])

    area = side * side
    solutions: list[tuple[str, list[dict]]] = []
    for orientation in ("h", "v"):
        candidates = [item for item in parsed if item["orientation"] == orientation]
        for n in range(2, len(candidates) + 1):
            for combo in combinations(candidates, n):
                if sum(item["body_len"] for item in combo) != area:
                    continue
                body_colors = [item["body_color"] for item in combo]
                if len(set(body_colors)) != n:
                    continue
                lookup = {item["body_color"]: item for item in combo}
                next_colors = [item["next_color"] for item in combo if item["next_color"] is not None]
                if len(next_colors) != n - 1 or len(set(next_colors)) != n - 1:
                    continue
                if any(color not in lookup for color in next_colors):
                    continue
                if any(item["next_color"] == item["body_color"] for item in combo if item["next_color"] is not None):
                    continue
                heads = [color for color in body_colors if color not in next_colors]
                tails = [item for item in combo if item["next_color"] is None]
                if len(heads) != 1 or len(tails) != 1:
                    continue

                order: list[dict] = []
                cur = heads[0]
                seen: set[int] = set()
                while cur is not None:
                    if cur in seen or cur not in lookup:
                        order = []
                        break
                    seen.add(cur)
                    stripe = lookup[cur]
                    order.append(stripe)
                    cur = stripe["next_color"]
                if len(order) == n and len(seen) == n and order[-1]["next_color"] is None:
                    solutions.append((orientation, order))
    return solutions


def _render_output(side: int, orientation: str, order: list[dict]) -> np.ndarray:
    flat = np.empty(side * side, dtype=int)
    idx = 0
    for stripe in order:
        end = idx + int(stripe["body_len"])
        flat[idx:end] = int(stripe["body_color"])
        idx = end
    if orientation == "h":
        return flat.reshape(side, side)
    return flat.reshape(side, side, order="F")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gray square by following the hidden cap-pointer chain through the stripes.

    Input structure:
    - One solid gray k×k square appears on a black background together with several
      one-cell-wide colored stripes, all horizontal or all vertical.
    - Each stripe body is monochrome and has a distinct body color; all body lengths
      add up to k².
    - Every non-final stripe has one extra end-cap pixel whose color equals the next
      stripe's body color, while one stripe has no cap at all.

    Rule:
    - Ignore non-matching distractors and find the unique stripe family whose body
      lengths sum to the gray square area.
    - Recover the chain head as the body color never referenced by any end-cap, then
      follow end-cap colors from stripe to stripe until the uncapped tail.
    - Output only the gray square's size: fill it continuously with each stripe body
      color repeated by its own body length, using row-major order for horizontal
      stripes and column-major order for vertical stripes.

    Key insight:
    - The visible stripe positions and lengths are decoys; only the cap-color pointer
      chain gives the true loading order, and the lengths then drive a continuous
      wraparound fill inside the square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _r0, _c0, side = _find_gray_square(g)
    solutions = _build_solutions(g, side)
    assert len(solutions) == 1, "expected a unique valid cap-pointer chain"
    orientation, order = solutions[0]
    return _render_output(side, orientation, order)


def _sample_partition(total: int, parts: int, minimum: int, maximum: int) -> list[int] | None:
    remainder = total - parts * minimum
    if remainder < 0:
        return None
    cuts = sorted(random.sample(range(remainder + parts - 1), parts - 1))
    vals: list[int] = []
    prev = -1
    for cut in cuts + [remainder + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    lengths = [minimum + v for v in vals]
    if max(lengths) > maximum:
        return None
    return lengths


def _make_output_from_lengths(side: int, orientation: str, colors: list[int], lengths: list[int]) -> np.ndarray:
    order = [
        {"body_color": color, "body_len": length, "next_color": None}
        for color, length in zip(colors, lengths)
    ]
    return _render_output(side, orientation, order)


def _stripe_patch(orientation: str, body_color: int, body_len: int, next_color: int | None) -> np.ndarray:
    cap_left = random.choice([True, False])
    total = body_len + (0 if next_color is None else 1)
    if orientation == "h":
        patch = np.zeros((1, total), dtype=int)
        if next_color is None:
            patch[0, :] = body_color
        elif cap_left:
            patch[0, 0] = next_color
            patch[0, 1:] = body_color
        else:
            patch[0, :-1] = body_color
            patch[0, -1] = next_color
        return patch

    patch = np.zeros((total, 1), dtype=int)
    if next_color is None:
        patch[:, 0] = body_color
    elif cap_left:
        patch[0, 0] = next_color
        patch[1:, 0] = body_color
    else:
        patch[:-1, 0] = body_color
        patch[-1, 0] = next_color
    return patch


def _distractor_patch(kind: str, orientation: str, colors: list[int]) -> np.ndarray:
    if kind == "block":
        patch = np.zeros((2, 2), dtype=int)
        patch[:, :] = colors[0]
        return patch
    if kind == "elbow":
        patch = np.zeros((2, 2), dtype=int)
        patch[0, :] = colors[0]
        patch[:, 0] = colors[0]
        return patch

    if orientation == "h":
        patch = np.zeros((1, 4), dtype=int)
        patch[0, :] = [colors[0], colors[1], colors[1], colors[0]]
        return patch
    patch = np.zeros((4, 1), dtype=int)
    patch[:, 0] = [colors[0], colors[1], colors[1], colors[0]]
    return patch


def _valid_locations(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, padding: int = 1) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    locs: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(w, c0 + pw + padding)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue
            locs.append((r0, c0))
    return locs


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, padding: int = 1) -> tuple[int, int] | None:
    locs = _valid_locations(grid, reserved, patch, padding=padding)
    if not locs:
        return None
    r0, c0 = random.choice(locs)
    ph, pw = patch.shape
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    rr0 = max(0, r0 - padding)
    rr1 = min(grid.shape[0], r0 + ph + padding)
    cc0 = max(0, c0 - padding)
    cc1 = min(grid.shape[1], c0 + pw + padding)
    reserved[rr0:rr1, cc0:cc1] = True
    return r0, c0


def _spatial_order(meta: list[dict], orientation: str) -> list[int]:
    key = "r" if orientation == "h" else "c"
    return [item["body_color"] for item in sorted(meta, key=lambda item: item[key])]


def _nontrivial_lengths(side: int) -> tuple[int, list[int]] | None:
    area = side * side
    counts = [n for n in range(4, 7) if 2 * n <= area]
    random.shuffle(counts)
    for n in counts:
        for _ in range(100):
            lengths = _sample_partition(area, n, minimum=2, maximum=10)
            if lengths is None:
                continue
            if len(set(lengths)) == n:
                continue
            boundaries = np.cumsum(lengths[:-1])
            if not any(int(x) % side != 0 for x in boundaries):
                continue
            if lengths == sorted(lengths) or lengths == sorted(lengths, reverse=True):
                continue
            return n, lengths
    return None


def _attempt_generate() -> np.ndarray | None:
    side = random.choice([4, 5, 6])
    sampled = _nontrivial_lengths(side)
    if sampled is None:
        return None
    n_stripes, lengths = sampled
    orientation = random.choice(["h", "v"])

    body_colors = random.sample(PALETTE, n_stripes)
    expected = _make_output_from_lengths(side, orientation, body_colors, lengths)

    stripes: list[dict] = []
    patches: list[tuple[str, np.ndarray]] = [("square", np.full((side, side), GREY, dtype=int))]
    for idx, (body_color, body_len) in enumerate(zip(body_colors, lengths)):
        next_color = None if idx == n_stripes - 1 else body_colors[idx + 1]
        patch = _stripe_patch(orientation, body_color, body_len, next_color)
        patches.append((f"stripe:{idx}", patch))

    spare = [color for color in PALETTE if color not in body_colors]
    random.shuffle(spare)
    distractor_count = random.choice([1, 2])
    distractor_specs: list[tuple[str, np.ndarray]] = []
    distractor_kinds = random.sample(["block", "elbow", "broken"], distractor_count)
    for kind in distractor_kinds:
        if kind == "broken":
            if len(spare) < 2:
                return None
            colors = [spare.pop(), spare.pop()]
            patch = _distractor_patch(kind, orientation, colors)
        else:
            if not spare:
                return None
            colors = [spare.pop()]
            patch = _distractor_patch(kind, orientation, colors)
        distractor_specs.append((f"distractor:{kind}", patch))
    patches.extend(distractor_specs)

    max_h = max(patch.shape[0] for _, patch in patches)
    max_w = max(patch.shape[1] for _, patch in patches)
    min_h = max(14, max_h + 4, side + 6)
    min_w = max(14, max_w + 4, side + 6)
    if orientation == "h":
        min_h = max(min_h, 3 * (n_stripes + distractor_count) + side + 1)
        min_w = max(min_w, max_w + 6)
    else:
        min_h = max(min_h, max_h + 6)
        min_w = max(min_w, 3 * (n_stripes + distractor_count) + side + 1)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    random.shuffle(patches)
    stripe_meta: list[dict] = []
    for name, patch in sorted(patches, key=lambda item: item[1].size, reverse=True):
        loc = _place_patch(grid, reserved, patch, padding=1)
        if loc is None:
            return None
        r0, c0 = loc
        if name.startswith("stripe:"):
            idx = int(name.split(":")[1])
            stripe_meta.append({"body_color": body_colors[idx], "r": r0, "c": c0})

    if _spatial_order(stripe_meta, orientation) == body_colors:
        return None

    out = main(grid)
    if not np.array_equal(out, expected):
        return None

    solutions = _build_solutions(grid, side)
    if len(solutions) != 1:
        return None

    # Make sure the valid solution is really the intended one and that the
    # fill cannot be reduced to simple spatial or monotone length ordering.
    solved_orientation, solved_order = solutions[0]
    solved_colors = [item["body_color"] for item in solved_order]
    if solved_orientation != orientation or solved_colors != body_colors:
        return None
    if len(set(lengths)) == len(lengths):
        return None
    if lengths == sorted(lengths) or lengths == sorted(lengths, reverse=True):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-loading input")
