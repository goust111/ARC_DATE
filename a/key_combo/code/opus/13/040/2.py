import math
from collections import deque

import numpy as np


BG = 0
GREY = 5
PAYLOAD_COLORS = (1, 2)

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in _DIR4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    comps = _find_components(g != BG)

    source_coords = None
    target_anchors: list[tuple[int, int]] = []
    for coords in comps:
        values = [int(g[r, c]) for r, c in coords]
        grey_count = sum(v == GREY for v in values)
        if grey_count == 0:
            raise AssertionError("all foreground must belong to the source sprite or isolated grey anchors")
        if grey_count == 1 and len(coords) > 1:
            assert source_coords is None, "expected exactly one source component"
            source_coords = coords
        else:
            assert len(coords) == 1 and values[0] == GREY, "targets must be isolated grey pixels"
            target_anchors.append((int(coords[0, 0]), int(coords[0, 1])))

    assert source_coords is not None, "missing source sprite"
    assert 3 <= len(target_anchors) <= 5, "expected 3-5 target anchors"

    source_anchor_cells = [(int(r), int(c)) for r, c in source_coords if int(g[r, c]) == GREY]
    assert len(source_anchor_cells) == 1, "source sprite must contain exactly one grey anchor"
    source_anchor = source_anchor_cells[0]

    payload = [(int(r), int(c), int(g[r, c])) for r, c in source_coords if int(g[r, c]) != GREY]
    payload_colors = sorted({color for _r, _c, color in payload})
    assert payload_colors == list(PAYLOAD_COLORS), "source sprite must use payload colors 1 and 2"

    return {
        "grid": g,
        "source_coords": source_coords,
        "source_anchor": source_anchor,
        "payload": payload,
        "payload_colors": tuple(payload_colors),
        "target_anchors": sorted(target_anchors),
    }


def _swap_color(color: int, swap: bool) -> int:
    if not swap:
        return color
    if color == PAYLOAD_COLORS[0]:
        return PAYLOAD_COLORS[1]
    if color == PAYLOAD_COLORS[1]:
        return PAYLOAD_COLORS[0]
    raise AssertionError("unexpected payload color")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the anchored two-color sprite onto every isolated grey marker, swapping the payload colors on odd columns.

    Input structure:
    - The grid has black background, one 4-connected multicolor sprite, and 3-5 additional isolated grey anchor pixels.
    - The sprite contains exactly one embedded grey source anchor and exactly two payload colors (blue and red).
    - Target grey pixels are not 4-connected to the sprite and may sit near borders, so stamped copies can be clipped.

    Rule:
    - Treat the grey pixel inside the sprite as the source anchor and every other grey pixel as a target anchor.
    - For each target, translate every non-grey sprite pixel by the anchor offset from source to target, then erase the target grey pixel.
    - If the target anchor column is even, keep the two payload colors; if it is odd, swap blue and red inside that copy.
    - Drop any translated pixels that fall outside the grid, and keep the original source sprite unchanged.

    Key insight:
    - The same target anchor controls both where the copy lands and whether its internal colors stay as-is or are exchanged, so position and recoloring are coupled.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    g = parsed["grid"]
    sr, sc = parsed["source_anchor"]

    out = g.copy()
    for tr, tc in parsed["target_anchors"]:
        out[tr, tc] = BG
        swap = (tc % 2) == 1
        for r, c, color in parsed["payload"]:
            rr = tr + (r - sr)
            cc = tc + (c - sc)
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = _swap_color(color, swap)
    return out


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < area:
        border = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    border.add((nr, nc))
        if not border:
            return mask
        pick = list(border)[int(rng.integers(len(border)))]
        mask[pick] = True
        frontier.add(pick)
    return mask


def _make_sprite(rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(200):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        area = int(rng.integers(5, 9))
        if area > h * w:
            continue
        mask = _random_connected_mask(rng, h, w, area)
        mask = _crop_mask(mask)
        if mask.shape[1] < 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        anchor = coords[int(rng.integers(len(coords)))]
        non_anchor = [rc for rc in coords if rc != anchor]
        if len(non_anchor) < 4:
            continue

        n_blue = int(rng.integers(1, len(non_anchor)))
        n_red = len(non_anchor) - n_blue
        if n_blue == 0 or n_red == 0 or n_blue == n_red:
            continue

        order = non_anchor[:]
        rng.shuffle(order)
        sprite = np.zeros(mask.shape, dtype=int)
        for i, (r, c) in enumerate(order):
            sprite[r, c] = PAYLOAD_COLORS[0] if i < n_blue else PAYLOAD_COLORS[1]
        sprite[anchor] = GREY
        return sprite, anchor
    return None


def _absolute_payload(sprite: np.ndarray, anchor: tuple[int, int], top_left: tuple[int, int]) -> list[tuple[int, int, int]]:
    ar, ac = anchor
    r0, c0 = top_left
    cells = []
    for r, c in np.argwhere(sprite != 0):
        color = int(sprite[r, c])
        if color == GREY:
            continue
        cells.append((int(r0 + r), int(c0 + c), color))
    return cells


def _anchor_distance_ok(points: list[tuple[int, int]], min_dist: float) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if math.hypot(r1 - r2, c1 - c2) <= min_dist:
                return False
    return True


def _copy_cells_for_anchor(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    source_anchor_abs: tuple[int, int],
    target_anchor: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    sr, sc = source_anchor_abs
    tr, tc = target_anchor
    inside: list[tuple[int, int, int]] = []
    outside: list[tuple[int, int, int]] = []
    swap = (tc % 2) == 1
    for r, c in np.argwhere(sprite != 0):
        color = int(sprite[r, c])
        if color == GREY:
            continue
        rr = tr + (int(r) - sr)
        cc = tc + (int(c) - sc)
        rec = (rr, cc, _swap_color(color, swap))
        if _in_bounds(grid_shape, rr, cc):
            inside.append(rec)
        else:
            outside.append(rec)
    return inside, outside


def _candidate_anchors(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    anchor_local: tuple[int, int],
    source_top_left: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    h, w = grid_shape
    ar, ac = anchor_local
    sr_abs = source_top_left[0] + ar
    sc_abs = source_top_left[1] + ac
    source_payload = {(r, c) for r, c, _color in _absolute_payload(sprite, anchor_local, source_top_left)}
    source_anchor_abs = (sr_abs, sc_abs)
    diag_limit = math.hypot(sprite.shape[0], sprite.shape[1])

    out = {"full_even": [], "full_odd": [], "clip_even": [], "clip_odd": []}
    for tr in range(h):
        for tc in range(w):
            if (tr, tc) in source_payload or (tr, tc) == source_anchor_abs:
                continue
            if math.hypot(tr - sr_abs, tc - sc_abs) <= diag_limit:
                continue

            inside, outside = _copy_cells_for_anchor(grid_shape, sprite, source_anchor_abs, (tr, tc))
            inside_cells = {(r, c) for r, c, _color in inside}
            if not inside_cells:
                continue
            if inside_cells & source_payload or source_anchor_abs in inside_cells:
                continue

            key = "full" if not outside else "clip"
            if outside and len(inside) < 2:
                continue
            parity = "even" if tc % 2 == 0 else "odd"
            out[f"{key}_{parity}"].append((tr, tc))
    return out


def _choose_anchor(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    source_anchor_abs: tuple[int, int],
    min_dist: float,
) -> tuple[int, int] | None:
    order = candidates[:]
    rng.shuffle(order)
    for anchor in order:
        trial = [source_anchor_abs, *chosen, anchor]
        if _anchor_distance_ok(trial, min_dist):
            return anchor
    return None


def _build_input(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    source_top_left: tuple[int, int],
    target_anchors: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    r0, c0 = source_top_left
    for r, c in np.argwhere(sprite != 0):
        grid[r0 + int(r), c0 + int(c)] = int(sprite[r, c])
    for tr, tc in target_anchors:
        grid[tr, tc] = GREY
    return grid


def _semantic_checks(inp: np.ndarray) -> None:
    parsed = _parse_grid(inp)
    out = main(inp)
    src_bbox = _bbox(parsed["source_coords"])
    sr0, sc0, sr1, sc1 = src_bbox
    assert np.array_equal(out[sr0 : sr1 + 1, sc0 : sc1 + 1], inp[sr0 : sr1 + 1, sc0 : sc1 + 1]), "source sprite changed"

    anchors = parsed["target_anchors"]
    assert any(c % 2 == 0 for _r, c in anchors), "missing even-column target"
    assert any(c % 2 == 1 for _r, c in anchors), "missing odd-column target"

    sr, sc = parsed["source_anchor"]
    original = {(r - sr, c - sc): color for r, c, color in parsed["payload"]}
    clipped_count = 0
    full_even_patch = None
    full_odd_patch = None

    for tr, tc in anchors:
        expected = {}
        swap = (tc % 2) == 1
        inside_count = 0
        total_count = 0
        for (dr, dc), color in original.items():
            rr = tr + dr
            cc = tc + dc
            total_count += 1
            if _in_bounds(out.shape, rr, cc):
                expected[(rr, cc)] = _swap_color(color, swap)
                inside_count += 1
        if inside_count < total_count:
            clipped_count += 1
        assert out[tr, tc] == BG, "target anchor was not erased"
        for (rr, cc), color in expected.items():
            assert int(out[rr, cc]) == color, "copy color mismatch"

        if inside_count == total_count:
            patch = tuple(sorted((rr - tr, cc - tc, color) for (rr, cc), color in expected.items()))
            if tc % 2 == 0 and full_even_patch is None:
                full_even_patch = patch
            if tc % 2 == 1 and full_odd_patch is None:
                full_odd_patch = patch

    assert clipped_count >= 1, "generator must cover clipping"
    assert full_even_patch is not None and full_odd_patch is not None, "need unclipped examples of both parities"
    assert full_even_patch != full_odd_patch, "odd-column recoloring must visibly differ"


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid_h = int(_RNG.integers(14, 19))
        grid_w = int(_RNG.integers(14, 19))

        made = _make_sprite(_RNG)
        if made is None:
            continue
        sprite, anchor_local = made
        sh, sw = sprite.shape

        row_lo = 1
        row_hi = grid_h - sh - 1
        col_lo = 1
        col_hi = grid_w - sw - 1
        if row_lo > row_hi or col_lo > col_hi:
            continue

        source_top_left = (
            int(_RNG.integers(row_lo, row_hi + 1)),
            int(_RNG.integers(col_lo, col_hi + 1)),
        )
        source_anchor_abs = (source_top_left[0] + anchor_local[0], source_top_left[1] + anchor_local[1])
        diag_limit = math.hypot(sh, sw)
        candidates = _candidate_anchors((grid_h, grid_w), sprite, anchor_local, source_top_left)

        even_full = _choose_anchor(_RNG, candidates["full_even"], [], source_anchor_abs, diag_limit)
        if even_full is None:
            continue
        odd_full = _choose_anchor(_RNG, candidates["full_odd"], [even_full], source_anchor_abs, diag_limit)
        if odd_full is None:
            continue

        clip_pool = candidates["clip_even"] + candidates["clip_odd"]
        clipped = _choose_anchor(_RNG, clip_pool, [even_full, odd_full], source_anchor_abs, diag_limit)
        if clipped is None:
            continue

        target_count = int(_RNG.integers(3, 6))
        chosen = [even_full, odd_full, clipped]
        extra_pool = candidates["full_even"] + candidates["full_odd"] + clip_pool
        extra_pool = [cell for cell in extra_pool if cell not in chosen]
        while len(chosen) < target_count:
            nxt = _choose_anchor(_RNG, extra_pool, chosen, source_anchor_abs, diag_limit)
            if nxt is None:
                break
            chosen.append(nxt)
            extra_pool = [cell for cell in extra_pool if cell != nxt]
        if len(chosen) != target_count:
            continue

        inp = _build_input((grid_h, grid_w), sprite, source_top_left, chosen)
        try:
            _semantic_checks(inp)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
