from __future__ import annotations

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _monochromatic_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr > 0 and not seen[rr - 1, cc] and int(g[rr - 1, cc]) == color:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and not seen[rr + 1, cc] and int(g[rr + 1, cc]) == color:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and not seen[rr, cc - 1] and int(g[rr, cc - 1]) == color:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and not seen[rr, cc + 1] and int(g[rr, cc + 1]) == color:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))

            arr = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": arr})

    return comps


def _frame_candidates(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _monochromatic_components(grid, bg=0):
        coords = comp["coords"]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        rel = coords - np.array([r0, c0])
        mask = np.zeros((h, w), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(mask, border):
            continue

        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center2": (int(r0 + r1), int(c0 + c1)),
                "area": h * w,
            }
        )

    return frames


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _best_chain_for_group(group: list[dict]) -> list[dict]:
    items = sorted(
        group,
        key=lambda item: (
            int(item["area"]),
            -int(item["bbox"][0]),
            -int(item["bbox"][1]),
        ),
        reverse=True,
    )
    memo: dict[int, list[dict]] = {}

    def dfs(i: int) -> list[dict]:
        if i in memo:
            return memo[i]
        best = [items[i]]
        for j in range(i + 1, len(items)):
            if not _contains_bbox(items[i]["bbox"], items[j]["bbox"]):
                continue
            cand = [items[i]] + dfs(j)
            if len(cand) > len(best):
                best = cand
        memo[i] = best
        return best

    best_chain: list[dict] = []
    for i in range(len(items)):
        cand = dfs(i)
        if len(cand) > len(best_chain):
            best_chain = cand
    return best_chain


def _find_decoder_chain(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    grouped: dict[tuple[int, int], list[dict]] = {}
    for frame in _frame_candidates(g):
        grouped.setdefault(frame["center2"], []).append(frame)

    candidates: list[tuple[tuple[int, int, int], list[dict]]] = []
    for group in grouped.values():
        chain = _best_chain_for_group(group)
        if len(chain) < 2:
            continue
        decoder_right = int(chain[0]["bbox"][3])
        if decoder_right + 1 >= g.shape[1]:
            continue
        if _bbox(g[:, decoder_right + 1 :] != 0) is None:
            continue
        score = (
            len(chain),
            -int(chain[0]["bbox"][1]),
            -int(chain[0]["bbox"][0]),
        )
        candidates.append((score, chain))

    assert candidates, "expected a left-side nested decoder and a right-side cipher"
    return max(candidates, key=lambda item: item[0])[1]


def _cipher_bbox(grid: np.ndarray, decoder_right: int) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert decoder_right + 1 < g.shape[1], "cipher must lie to the right of the decoder"
    box = _bbox(g[:, decoder_right + 1 :] != 0)
    assert box is not None, "expected a non-empty cipher region"
    r0, c0, r1, c1 = box
    return int(r0), int(c0 + decoder_right + 1), int(r1), int(c1 + decoder_right + 1)


def _apply_cycle(patch: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.array(patch, copy=True, dtype=int)
    src = np.asarray(patch, dtype=int)
    for i, color in enumerate(colors):
        out[src == int(color)] = int(colors[(i + 1) % len(colors)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the right-side color cipher by cycling colors according to the left-side nested frame order.

    Input structure:
    - The background is black.
    - The left side contains 2 to 4 concentric hollow rectangular frames, one pixel thick, with different colors from outermost to innermost.
    - The right side contains a single rectangular cipher patch made of colored pixels; some of its colors match decoder-frame colors and others do not.

    Rule:
    - Detect the nested frame chain on the left and read its colors from outermost to innermost as C1, C2, ..., Cn.
    - Build the cyclic substitution C1 -> C2, C2 -> C3, ..., Cn -> C1.
    - Crop the right-side cipher rectangle and replace each pixel whose color is in the cycle; all other colors, including black, stay unchanged.

    Key insight:
    - The decoder is not about rectangle size itself: the visible outside-to-inside nesting order defines the color-cycle used to transform the separate cipher patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    chain = _find_decoder_chain(g)
    colors = [int(frame["color"]) for frame in chain]
    decoder_right = int(chain[0]["bbox"][3])
    r0, c0, r1, c1 = _cipher_bbox(g, decoder_right)
    cipher = g[r0 : r1 + 1, c0 : c1 + 1]
    return _apply_cycle(cipher, colors)


_BG = 0
_PALETTE = tuple(range(1, 10))
_LAYER_CHOICES = (2, 3, 4)
_OUTER_MAX = 15
_CIPHER_H_RANGE = (4, 10)
_CIPHER_W_MAX = 9
_MAX_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _draw_frame(canvas: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0, c0 : c1 + 1] = int(color)
    canvas[r1, c0 : c1 + 1] = int(color)
    canvas[r0 : r1 + 1, c0] = int(color)
    canvas[r0 : r1 + 1, c1] = int(color)


def _sample_insets(rng: np.random.Generator, layer_count: int) -> list[int] | None:
    insets = [0]
    inset = 0
    for _ in range(layer_count - 1):
        inset += int(rng.integers(1, 3))
        insets.append(inset)
    if 3 + 2 * insets[-1] > _OUTER_MAX:
        return None
    return insets


def _make_decoder_patch(rng: np.random.Generator, frame_colors: list[int]) -> np.ndarray | None:
    insets = _sample_insets(rng, len(frame_colors))
    if insets is None:
        return None
    min_outer = 3 + 2 * insets[-1]
    outer_h = int(rng.integers(min_outer, _OUTER_MAX + 1))
    outer_w = int(rng.integers(min_outer, _OUTER_MAX + 1))
    patch = np.zeros((outer_h, outer_w), dtype=int)
    for color, inset in zip(frame_colors, insets):
        _draw_frame(
            patch,
            (int(inset), int(inset), int(outer_h - inset - 1), int(outer_w - inset - 1)),
            int(color),
        )
    return patch


def _make_cipher_patch(rng: np.random.Generator, shape: tuple[int, int], frame_colors: list[int]) -> np.ndarray:
    h, w = shape
    frame_colors = [int(c) for c in frame_colors]
    extra_pool = [c for c in _PALETTE if c not in frame_colors]
    extra_count = int(rng.integers(1, min(4, len(extra_pool) + 1)))
    extras = list(map(int, rng.choice(extra_pool, size=extra_count, replace=False)))
    palette = frame_colors + extras

    values = np.asarray(palette + [_BG], dtype=int)
    probs = np.asarray([2] * len(frame_colors) + [2] * len(extras) + [1], dtype=float)
    probs /= probs.sum()
    patch = rng.choice(values, size=(h, w), p=probs).astype(int)

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    non_corner = [(r, c) for r in range(h) for c in range(w) if (r, c) not in corners]
    need = frame_colors + [extras[0], _BG]
    perm = rng.permutation(len(non_corner))
    for color, idx in zip(need, perm[: len(need)]):
        r, c = non_corner[int(idx)]
        patch[r, c] = int(color)

    corner_colors = rng.choice(np.asarray(palette, dtype=int), size=4, replace=True)
    for (r, c), color in zip(sorted(corners), corner_colors):
        patch[r, c] = int(color)

    return patch


def _compose_input(
    rng: np.random.Generator,
    decoder_patch: np.ndarray,
    cipher_patch: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    dec_h, dec_w = decoder_patch.shape
    cip_h, cip_w = cipher_patch.shape

    left_pad_l = int(rng.integers(0, 2))
    left_pad_r = int(rng.integers(0, 2))
    sep = int(rng.integers(1, 5))
    right_pad_l = int(rng.integers(0, 2))
    right_pad_r = int(rng.integers(0, 2))

    width = left_pad_l + dec_w + left_pad_r + sep + right_pad_l + cip_w + right_pad_r
    if width > 30:
        return None

    base_h = max(dec_h, cip_h)
    extra_h = int(rng.integers(0, min(5, 30 - base_h + 1)))
    height = int(base_h + extra_h)
    if height > 30:
        return None

    decoder_r = int(rng.integers(0, height - dec_h + 1))
    decoder_c = int(left_pad_l)
    cipher_r = int(rng.integers(0, height - cip_h + 1))
    cipher_c = int(left_pad_l + dec_w + left_pad_r + sep + right_pad_l)

    grid = np.zeros((height, width), dtype=int)
    grid[decoder_r : decoder_r + dec_h, decoder_c : decoder_c + dec_w] = decoder_patch
    grid[cipher_r : cipher_r + cip_h, cipher_c : cipher_c + cip_w] = cipher_patch
    return grid, (cipher_r, cipher_c, cipher_r + cip_h - 1, cipher_c + cip_w - 1)


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    layer_count = int(rng.choice(_LAYER_CHOICES))
    frame_colors = list(map(int, rng.choice(np.asarray(_PALETTE, dtype=int), size=layer_count, replace=False)))
    decoder_patch = _make_decoder_patch(rng, frame_colors)
    if decoder_patch is None:
        return None

    dec_w = int(decoder_patch.shape[1])
    max_cipher_w = min(_CIPHER_W_MAX, 22 - dec_w)
    if max_cipher_w < 4:
        return None

    cipher_h = int(rng.integers(_CIPHER_H_RANGE[0], _CIPHER_H_RANGE[1]))
    cipher_w = int(rng.integers(4, max_cipher_w + 1))
    cipher_patch = _make_cipher_patch(rng, (cipher_h, cipher_w), frame_colors)
    composed = _compose_input(rng, decoder_patch, cipher_patch)
    if composed is None:
        return None

    grid, cipher_bbox = composed
    expected = _apply_cycle(cipher_patch, frame_colors)

    try:
        chain = _find_decoder_chain(grid)
        parsed_colors = [int(frame["color"]) for frame in chain]
        parsed_bbox = _cipher_bbox(grid, int(chain[0]["bbox"][3]))
    except AssertionError:
        return None

    if len(_frame_candidates(grid)) != layer_count:
        return None
    if parsed_colors != frame_colors:
        return None
    if parsed_bbox != cipher_bbox:
        return None
    if not np.any(cipher_patch == _BG):
        return None
    if not any(np.any(cipher_patch == color) for color in frame_colors):
        return None
    if not any(np.any(cipher_patch == color) for color in range(1, 10) if color not in frame_colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame cipher input")
