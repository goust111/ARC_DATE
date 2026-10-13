from __future__ import annotations

import numpy as np


_BG = 0
_SIDES = ("up", "down", "left", "right")
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()

_ARROW_MASKS = {
    "up": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}
_ARROW_COORDS = {
    name: frozenset(map(tuple, np.argwhere(mask)))
    for name, mask in _ARROW_MASKS.items()
}


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue

        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []

        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in _NEIGHBORS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        components.append(coords)

    return components


def _arrow_direction(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) != 5:
        return None
    mask = _component_mask(coords)
    if mask.shape != (3, 3):
        return None
    rel = frozenset(map(tuple, np.argwhere(mask)))
    for direction, pattern in _ARROW_COORDS.items():
        if rel == pattern:
            return direction
    return None


def _placement_side(
    source_bbox: tuple[int, int, int, int],
    arrow_bbox: tuple[int, int, int, int],
) -> str:
    sr0, sc0, sr1, sc1 = source_bbox
    ar0, ac0, ar1, ac1 = arrow_bbox

    if ar1 < sr0:
        return "up"
    if ar0 > sr1:
        return "down"
    if ac1 < sc0:
        return "left"
    if ac0 > sc1:
        return "right"
    raise AssertionError("arrow must lie strictly above, below, left, or right of its source")


def _mirror_top_left(
    source_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = source_bbox
    h = sr1 - sr0 + 1
    w = sc1 - sc0 + 1

    if side == "up":
        return sr0 - h - 2, sc0
    if side == "down":
        return sr1 + 3, sc0
    if side == "left":
        return sr0, sc0 - w - 2
    assert side == "right"
    return sr0, sc1 + 3


def _stamp(board: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    board[r0 + rows, c0 + cols] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != _BG):
        components = _find_components_of_color(g, color)
        assert len(components) == 2, "each foreground color must form exactly one source and one arrow"

        arrow_comp = None
        arrow_dir = None
        source_comp = None
        for comp in components:
            direction = _arrow_direction(comp)
            if direction is not None:
                assert arrow_comp is None, "a color may not contain two arrows"
                arrow_comp = comp
                arrow_dir = direction
            else:
                assert source_comp is None, "a color may not contain two source objects"
                source_comp = comp

        assert arrow_comp is not None and source_comp is not None
        source_bbox = _bbox(source_comp)
        arrow_bbox = _bbox(arrow_comp)
        side = _placement_side(source_bbox, arrow_bbox)
        source_mask = _component_mask(source_comp)
        mirror_mask = np.flipud(source_mask) if arrow_dir in {"up", "down"} else np.fliplr(source_mask)
        mirror_r0, mirror_c0 = _mirror_top_left(source_bbox, side)

        mh, mw = mirror_mask.shape
        assert 0 <= mirror_r0 and 0 <= mirror_c0
        assert mirror_r0 + mh <= g.shape[0] and mirror_c0 + mw <= g.shape[1]

        pairs.append(
            {
                "color": color,
                "arrow_comp": arrow_comp,
                "mirror_mask": mirror_mask,
                "mirror_r0": mirror_r0,
                "mirror_c0": mirror_c0,
            }
        )

    out = g.copy()
    for pair in pairs:
        for r, c in pair["arrow_comp"]:
            out[r, c] = _BG

    for pair in pairs:
        _stamp(out, pair["mirror_mask"], pair["mirror_r0"], pair["mirror_c0"], pair["color"])

    return out


# --------------- generator ---------------

_GRID_H_RANGE = (17, 29)
_GRID_W_RANGE = (17, 29)
_OBJECT_COUNT_RANGE = (1, 4)
_COLOR_POOL = tuple(range(1, 10))
_DIMENSION_POOL = ((3, 3), (3, 5), (5, 3), (5, 5), (3, 7), (7, 3), (5, 7), (7, 5))
_INPUT_GAP_RANGE = (1, 3)
_SCENE_ATTEMPTS = 400
_PAIR_ATTEMPTS = 120


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _path_to_center(
    start: tuple[int, int],
    end: tuple[int, int],
) -> set[tuple[int, int]]:
    r, c = start
    er, ec = end
    coords = {(r, c)}
    while (r, c) != (er, ec):
        options = []
        if r < er:
            options.append((r + 1, c))
        elif r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        elif c > ec:
            options.append((r, c - 1))
        r, c = options[int(_RNG.integers(len(options)))]
        coords.add((r, c))
    return coords


def _random_source_mask() -> np.ndarray:
    while True:
        h, w = _DIMENSION_POOL[int(_RNG.integers(len(_DIMENSION_POOL)))]
        center = (h // 2, w // 2)
        anchors = [
            (0, int(_RNG.integers(w))),
            (h - 1, int(_RNG.integers(w))),
            (int(_RNG.integers(h)), 0),
            (int(_RNG.integers(h)), w - 1),
        ]

        coords = {center}
        for anchor in anchors:
            coords |= _path_to_center(anchor, center)

        min_area = max(6, len(coords))
        max_area = h * w - 1
        if min_area > max_area:
            continue

        target_area = int(_RNG.integers(min_area, max_area + 1))
        while len(coords) < target_area:
            frontier = set()
            for r, c in coords:
                for dr, dc in _NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            pick = list(frontier)[int(_RNG.integers(len(frontier)))]
            coords.add(pick)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True

        if frozenset(map(tuple, np.argwhere(mask))) in _ARROW_COORDS.values():
            continue
        return mask


def _source_position_candidates(
    H: int,
    W: int,
    h: int,
    w: int,
    side: str,
    gap: int,
) -> list[tuple[int, int]]:
    if side == "up":
        r_start = max(h + 2, gap + 3)
        r_stop = H - h
        c_start = 0
        c_stop = W - w
    elif side == "down":
        r_start = 0
        r_stop = min(H - h - gap - 3, H - 2 * h - 2)
        c_start = 0
        c_stop = W - w
    elif side == "left":
        r_start = 0
        r_stop = H - h
        c_start = max(w + 2, gap + 3)
        c_stop = W - w
    else:
        assert side == "right"
        r_start = 0
        r_stop = H - h
        c_start = 0
        c_stop = min(W - w - gap - 3, W - 2 * w - 2)

    if r_start > r_stop or c_start > c_stop:
        return []

    return [(r, c) for r in range(r_start, r_stop + 1) for c in range(c_start, c_stop + 1)]


def _arrow_top_left(src_r0: int, src_c0: int, h: int, w: int, side: str, gap: int) -> tuple[int, int]:
    if side == "up":
        return src_r0 - gap - 3, src_c0 + w // 2 - 1
    if side == "down":
        return src_r0 + h + gap, src_c0 + w // 2 - 1
    if side == "left":
        return src_r0 + h // 2 - 1, src_c0 - gap - 3
    assert side == "right"
    return src_r0 + h // 2 - 1, src_c0 + w + gap


def _bool_canvas(H: int, W: int, mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    canvas = np.zeros((H, W), dtype=bool)
    rows, cols = np.nonzero(mask)
    canvas[r0 + rows, c0 + cols] = True
    return canvas


def _place_pair(
    H: int,
    W: int,
    color: int,
    side: str,
    arrow_dir: str,
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
) -> dict | None:
    arrow_mask = _ARROW_MASKS[arrow_dir]

    for _ in range(_PAIR_ATTEMPTS):
        source_mask = _random_source_mask()
        h, w = source_mask.shape
        gap = int(_RNG.integers(_INPUT_GAP_RANGE[0], _INPUT_GAP_RANGE[1] + 1))

        candidates = _source_position_candidates(H, W, h, w, side, gap)
        if not candidates:
            continue
        _RNG.shuffle(candidates)

        mirror_mask = np.flipud(source_mask) if arrow_dir in {"up", "down"} else np.fliplr(source_mask)

        for src_r0, src_c0 in candidates:
            arrow_r0, arrow_c0 = _arrow_top_left(src_r0, src_c0, h, w, side, gap)
            mirror_r0, mirror_c0 = _mirror_top_left((src_r0, src_c0, src_r0 + h - 1, src_c0 + w - 1), side)

            source_footprint = _bool_canvas(H, W, source_mask, src_r0, src_c0)
            arrow_footprint = _bool_canvas(H, W, arrow_mask, arrow_r0, arrow_c0)
            mirror_footprint = _bool_canvas(H, W, mirror_mask, mirror_r0, mirror_c0)

            input_pair = source_footprint | arrow_footprint
            output_pair = source_footprint | mirror_footprint

            if np.any(input_blocked & input_pair):
                continue
            if np.any(output_blocked & output_pair):
                continue

            return {
                "color": color,
                "source_mask": source_mask,
                "src_r0": src_r0,
                "src_c0": src_c0,
                "arrow_mask": arrow_mask,
                "arrow_r0": arrow_r0,
                "arrow_c0": arrow_c0,
                "mirror_mask": mirror_mask,
                "mirror_r0": mirror_r0,
                "mirror_c0": mirror_c0,
                "input_pair": input_pair,
                "output_pair": output_pair,
            }

    return None


def _render_scene(H: int, W: int, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((H, W), _BG, dtype=int)
    out = np.full((H, W), _BG, dtype=int)

    for pair in pairs:
        _stamp(inp, pair["source_mask"], pair["src_r0"], pair["src_c0"], pair["color"])
        _stamp(inp, pair["arrow_mask"], pair["arrow_r0"], pair["arrow_c0"], pair["color"])
        _stamp(out, pair["source_mask"], pair["src_r0"], pair["src_c0"], pair["color"])
        _stamp(out, pair["mirror_mask"], pair["mirror_r0"], pair["mirror_c0"], pair["color"])

    return inp, out


def _generate_scene() -> np.ndarray | None:
    H = int(_RNG.integers(_GRID_H_RANGE[0], _GRID_H_RANGE[1] + 1))
    W = int(_RNG.integers(_GRID_W_RANGE[0], _GRID_W_RANGE[1] + 1))
    num_objects = int(_RNG.integers(_OBJECT_COUNT_RANGE[0], _OBJECT_COUNT_RANGE[1] + 1))

    colors = list(_RNG.choice(_COLOR_POOL, size=num_objects, replace=False))
    sides = [_SIDES[int(_RNG.integers(len(_SIDES)))] for _ in range(num_objects)]
    arrow_dirs = [_SIDES[int(_RNG.integers(len(_SIDES)))] for _ in range(num_objects)]

    input_blocked = np.zeros((H, W), dtype=bool)
    output_blocked = np.zeros((H, W), dtype=bool)
    pairs: list[dict] = []

    for color, side, arrow_dir in zip(colors, sides, arrow_dirs):
        pair = _place_pair(H, W, int(color), side, arrow_dir, input_blocked, output_blocked)
        if pair is None:
            return None
        pairs.append(pair)
        input_blocked |= _dilate8(pair["input_pair"])
        output_blocked |= _dilate8(pair["output_pair"])

    inp, expected = _render_scene(H, W, pairs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_SCENE_ATTEMPTS):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
