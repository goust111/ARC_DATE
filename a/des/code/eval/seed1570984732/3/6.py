from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _is_alternating(bits: np.ndarray) -> bool:
    return bits.size >= 2 and np.any(bits) and np.any(~bits) and np.all(bits[:-1] != bits[1:])


def _piece_mask(piece: dict) -> np.ndarray:
    r0, c0, r1, c1 = piece["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in piece["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _profile_from_bits(bits: np.ndarray) -> tuple[int, ...]:
    return tuple(1 if bool(v) else -1 for v in bits.tolist())


def _analyze_piece(piece: dict) -> dict:
    mask = _piece_mask(piece)
    r0, c0, r1, c1 = piece["bbox"]
    h, w = mask.shape
    candidates: list[dict] = []

    if w >= 3 and np.all(mask[:, : w - 2]):
        outer = mask[:, w - 1]
        inner = mask[:, w - 2]
        if np.array_equal(outer, inner) and _is_alternating(outer):
            candidates.append(
                {
                    "side": "right",
                    "profile": _profile_from_bits(outer),
                    "base_bbox": (r0, c0, r1, c1 - 1),
                }
            )

    if w >= 3 and np.all(mask[:, 2:]):
        outer = mask[:, 0]
        inner = mask[:, 1]
        if np.array_equal(outer, inner) and _is_alternating(outer):
            candidates.append(
                {
                    "side": "left",
                    "profile": _profile_from_bits(outer),
                    "base_bbox": (r0, c0 + 1, r1, c1),
                }
            )

    if h >= 3 and np.all(mask[: h - 2, :]):
        outer = mask[h - 1, :]
        inner = mask[h - 2, :]
        if np.array_equal(outer, inner) and _is_alternating(outer):
            candidates.append(
                {
                    "side": "bottom",
                    "profile": _profile_from_bits(outer),
                    "base_bbox": (r0, c0, r1 - 1, c1),
                }
            )

    if h >= 3 and np.all(mask[2:, :]):
        outer = mask[0, :]
        inner = mask[1, :]
        if np.array_equal(outer, inner) and _is_alternating(outer):
            candidates.append(
                {
                    "side": "top",
                    "profile": _profile_from_bits(outer),
                    "base_bbox": (r0 + 1, c0, r1, c1),
                }
            )

    assert len(candidates) == 1, "each component must have exactly one toothed edge"
    analyzed = dict(piece)
    analyzed.update(candidates[0])
    return analyzed


def _complementary(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return len(a) == len(b) and all(x == -y for x, y in zip(a, b))


def _pair_same_color(group: list[dict]) -> list[tuple[dict, tuple[int, int]]]:
    assert len(group) == 2, "each color must appear on exactly two pieces"
    a, b = group
    sides = {a["side"], b["side"]}

    if sides == {"right", "left"}:
        left = a if a["side"] == "right" else b
        right = b if left is a else a
        assert left["base_bbox"][0] == right["base_bbox"][0]
        assert left["base_bbox"][2] == right["base_bbox"][2]
        assert left["base_bbox"][1] < right["base_bbox"][1]
        assert _complementary(left["profile"], right["profile"])
        gap = right["base_bbox"][1] - left["base_bbox"][3] - 1
        assert gap >= 2 and gap % 2 == 0, "horizontal pieces must meet after equal synchronous motion"
        shift = gap // 2
        return [(left, (0, shift)), (right, (0, -shift))]

    if sides == {"bottom", "top"}:
        top = a if a["side"] == "bottom" else b
        bottom = b if top is a else a
        assert top["base_bbox"][1] == bottom["base_bbox"][1]
        assert top["base_bbox"][3] == bottom["base_bbox"][3]
        assert top["base_bbox"][0] < bottom["base_bbox"][0]
        assert _complementary(top["profile"], bottom["profile"])
        gap = bottom["base_bbox"][0] - top["base_bbox"][2] - 1
        assert gap >= 2 and gap % 2 == 0, "vertical pieces must meet after equal synchronous motion"
        shift = gap // 2
        return [(top, (shift, 0)), (bottom, (-shift, 0))]

    raise AssertionError("same-color pieces must face each other on one axis")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored jigsaw pieces, then slide both members toward each other
    until their alternating protrusions and notches interlock exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = [_analyze_piece(piece) for piece in _find_components(g)]
    if not comps:
        return g.copy()

    by_color: dict[int, list[dict]] = {}
    for piece in comps:
        by_color.setdefault(int(piece["color"]), []).append(piece)

    out = np.full_like(g, BG)
    occupied = np.zeros_like(g, dtype=bool)

    for color in sorted(by_color):
        for piece, (dr, dc) in _pair_same_color(by_color[color]):
            for r, c in piece["coords"]:
                rr = int(r) + dr
                cc = int(c) + dc
                assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "translated piece must stay in bounds"
                assert not occupied[rr, cc], "translated pieces must not overlap"
                occupied[rr, cc] = True
                out[rr, cc] = int(color)

    return out


def _alternating_bits(length: int, rng: np.random.Generator) -> list[int]:
    start = int(rng.integers(0, 2))
    return [(start + i) % 2 for i in range(length)]


def _piece_patch(body_h: int, body_w: int, side: str, protrusions: list[int], color: int) -> np.ndarray:
    if side == "right":
        patch = np.zeros((body_h, body_w + 1), dtype=int)
        patch[:, :body_w] = color
        for r, bit in enumerate(protrusions):
            if bit:
                patch[r, body_w] = color
            else:
                patch[r, body_w - 1] = BG
        return patch

    if side == "left":
        patch = np.zeros((body_h, body_w + 1), dtype=int)
        patch[:, 1:] = color
        for r, bit in enumerate(protrusions):
            if bit:
                patch[r, 0] = color
            else:
                patch[r, 1] = BG
        return patch

    if side == "bottom":
        patch = np.zeros((body_h + 1, body_w), dtype=int)
        patch[:body_h, :] = color
        for c, bit in enumerate(protrusions):
            if bit:
                patch[body_h, c] = color
            else:
                patch[body_h - 1, c] = BG
        return patch

    assert side == "top"
    patch = np.zeros((body_h + 1, body_w), dtype=int)
    patch[1:, :] = color
    for c, bit in enumerate(protrusions):
        if bit:
            patch[0, c] = color
        else:
            patch[1, c] = BG
    return patch


def _patch_coords(patch: np.ndarray) -> np.ndarray:
    return np.argwhere(patch != BG).astype(int)


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = coords.copy()
    shifted[:, 0] += dr
    shifted[:, 1] += dc
    return shifted


def _make_pair(color: int, rng: np.random.Generator) -> dict:
    orientation = "horizontal" if int(rng.integers(0, 2)) == 0 else "vertical"
    body_h = int(rng.integers(4, 8))
    body_w = int(rng.integers(4, 8))

    if orientation == "horizontal":
        profile = _alternating_bits(body_h, rng)
        base_gap = int(rng.choice(np.asarray([4, 6, 8, 10], dtype=int)))
        shift = base_gap // 2

        left_patch = _piece_patch(body_h, body_w, "right", profile, color)
        right_patch = _piece_patch(body_h, body_w, "left", [1 - v for v in profile], color)

        right_c0 = body_w + base_gap - 1
        in_left = _patch_coords(left_patch)
        in_right = _shift_coords(_patch_coords(right_patch), 0, right_c0)
        out_left = _shift_coords(_patch_coords(left_patch), 0, shift)
        out_right = _shift_coords(_patch_coords(right_patch), 0, right_c0 - shift)

        return {
            "box_shape": (body_h, 2 * body_w + base_gap),
            "input_pieces": [(in_left, color), (in_right, color)],
            "output_pieces": [(out_left, color), (out_right, color)],
        }

    profile = _alternating_bits(body_w, rng)
    base_gap = int(rng.choice(np.asarray([4, 6, 8, 10], dtype=int)))
    shift = base_gap // 2

    top_patch = _piece_patch(body_h, body_w, "bottom", profile, color)
    bottom_patch = _piece_patch(body_h, body_w, "top", [1 - v for v in profile], color)

    bottom_r0 = body_h + base_gap - 1
    in_top = _patch_coords(top_patch)
    in_bottom = _shift_coords(_patch_coords(bottom_patch), bottom_r0, 0)
    out_top = _shift_coords(_patch_coords(top_patch), shift, 0)
    out_bottom = _shift_coords(_patch_coords(bottom_patch), bottom_r0 - shift, 0)

    return {
        "box_shape": (2 * body_h + base_gap, body_w),
        "input_pieces": [(in_top, color), (in_bottom, color)],
        "output_pieces": [(out_top, color), (out_bottom, color)],
    }


def _render_pairs(
    pairs: list[dict],
    origins: list[tuple[int, int]],
    use_output: bool,
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    key = "output_pieces" if use_output else "input_pieces"
    for pair, (r0, c0) in zip(pairs, origins):
        for coords, color in pair[key]:
            for r, c in coords:
                rr = r0 + int(r)
                cc = c0 + int(c)
                assert grid[rr, cc] == BG
                grid[rr, cc] = int(color)
    return grid


def _place_pairs(pairs: list[dict], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    workspace = 30
    margin = 1
    occupied = np.zeros((workspace, workspace), dtype=bool)
    order = sorted(range(len(pairs)), key=lambda i: pairs[i]["box_shape"][0] * pairs[i]["box_shape"][1], reverse=True)
    origins: list[tuple[int, int] | None] = [None] * len(pairs)

    for idx in order:
        ph, pw = pairs[idx]["box_shape"]
        if ph > workspace - 2 or pw > workspace - 2:
            return None

        placed = False
        for _ in range(200):
            r0 = int(rng.integers(1, workspace - ph))
            c0 = int(rng.integers(1, workspace - pw))
            rr0 = max(0, r0 - margin)
            rr1 = min(workspace, r0 + ph + margin)
            cc0 = max(0, c0 - margin)
            cc1 = min(workspace, c0 + pw + margin)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            origins[idx] = (r0, c0)
            placed = True
            break

        if not placed:
            return None

    return [origin for origin in origins if origin is not None]


def _crop_with_padding(grid: np.ndarray, expected: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h, w = grid.shape

    top_pad = int(rng.integers(1, min(4, int(r0)) + 1))
    left_pad = int(rng.integers(1, min(4, int(c0)) + 1))
    bottom_pad = int(rng.integers(1, min(4, int(h - 1 - r1)) + 1))
    right_pad = int(rng.integers(1, min(4, int(w - 1 - c1)) + 1))

    rr0 = int(r0) - top_pad
    cc0 = int(c0) - left_pad
    rr1 = int(r1) + bottom_pad
    cc1 = int(c1) + right_pad
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1], expected[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(1, 4))
    colors = [int(v) for v in rng.choice(np.asarray(PALETTE, dtype=int), size=n_pairs, replace=False)]
    pairs = [_make_pair(color, rng) for color in colors]
    origins = _place_pairs(pairs, rng)
    if origins is None:
        return None

    workspace_shape = (30, 30)
    grid = _render_pairs(pairs, origins, use_output=False, shape=workspace_shape)
    expected = _render_pairs(pairs, origins, use_output=True, shape=workspace_shape)
    grid, expected = _crop_with_padding(grid, expected, rng)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jigsaw-pair input")
