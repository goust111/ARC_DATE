from __future__ import annotations

import itertools
from collections import deque

import numpy as np


PANEL_COLOR = 8
BG = 0
LEGEND_COLORS = (1, 2, 3, 4)
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _solid_square(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return bool(
        coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1)
        and r1 - r0 == c1 - c0
        and coords.shape[0] in (4, 9)
    )


def _shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for count in range(3, 6):
        for cells in itertools.combinations(range(9), count):
            mask = np.zeros((3, 3), dtype=bool)
            mask.flat[list(cells)] = True
            if len(_components(mask)) != 1:
                continue
            shapes.append(mask)
    return shapes


SHAPE_LIBRARY = _shape_library()


def _parse_panel_and_canvas(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    teal = np.argwhere(grid == PANEL_COLOR)
    assert teal.size > 0, "missing teal legend panel"
    panel_w = int(teal[:, 1].max()) + 1
    assert int(teal[:, 1].min()) == 0
    assert panel_w == 3
    assert not np.any(grid[:, panel_w:] == PANEL_COLOR)

    work_start = panel_w
    if work_start < grid.shape[1] and np.all(grid[:, work_start] == BG):
        work_start += 1
    return grid[:, :panel_w], grid[:, work_start:]


def _parse_legend(panel: np.ndarray) -> dict[int, np.ndarray]:
    h, w = panel.shape
    assert w == 3
    legend: dict[int, np.ndarray] = {}
    r = 0
    while r < h:
        if np.all(panel[r] == BG):
            r += 1
            continue
        assert r + 3 <= h
        slot = panel[r : r + 3]
        colors = set(map(int, np.unique(slot))) - {BG, PANEL_COLOR}
        assert len(colors) == 1
        color = colors.pop()
        mask = slot == color
        assert 3 <= int(mask.sum()) <= 5
        assert len(_components(mask)) == 1
        legend[color] = mask.copy()
        r += 3
        if r < h and np.all(panel[r] == BG):
            r += 1
    assert 3 <= len(legend) <= 4
    return legend


def _parse_blocks(canvas: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    h, w = canvas.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(canvas[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(canvas[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            side = r1 - r0 + 1
            assert side == c1 - c0 + 1
            assert side in (2, 3)
            assert coords.shape[0] == side * side
            assert np.all(canvas[r0 : r1 + 1, c0 : c1 + 1] == color)
            blocks.append({"color": color, "r": r0, "c": c0, "size": side})
    assert 3 <= len(blocks) <= 4
    return blocks


def _distance_key(block: dict, canvas_shape: tuple[int, int]) -> int:
    h, w = canvas_shape
    s = int(block["size"])
    return int((2 * int(block["r"]) + s - h) ** 2 + (2 * int(block["c"]) + s - w) ** 2)


def _scaled_cells(mask: np.ndarray, top: int, left: int, scale: int) -> np.ndarray:
    tile = np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
    coords = np.argwhere(tile)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _render_output(
    legend: dict[int, np.ndarray],
    blocks: list[dict],
    canvas_shape: tuple[int, int],
    *,
    order: str = "distance",
    crop_legend: bool = False,
    force_scale: int | None = None,
) -> np.ndarray:
    h, w = canvas_shape
    out = np.zeros((h, w), dtype=int)

    if order == "distance":
        ordered = sorted(blocks, key=lambda block: _distance_key(block, canvas_shape), reverse=True)
    elif order == "distance-near":
        ordered = sorted(blocks, key=lambda block: _distance_key(block, canvas_shape))
    elif order == "row-major":
        ordered = sorted(blocks, key=lambda block: (int(block["r"]), int(block["c"])))
    elif order == "reverse-row-major":
        ordered = sorted(blocks, key=lambda block: (int(block["r"]), int(block["c"])), reverse=True)
    else:
        raise ValueError(order)

    for block in ordered:
        color = int(block["color"])
        top = int(block["r"])
        left = int(block["c"])
        scale = int(block["size"]) if force_scale is None else int(force_scale)
        mask = legend[color]
        if crop_legend:
            mask = _crop_mask(mask)
        coords = _scaled_cells(mask, top, left, scale)
        assert coords.size > 0
        assert coords[:, 0].max() < h and coords[:, 1].max() < w
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _try_render_output(*args, **kwargs) -> np.ndarray | None:
    try:
        return _render_output(*args, **kwargs)
    except AssertionError:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each code square through the legend, scale the shape by square size, and overlay by center distance.

    Input structure:
    - A teal legend panel occupies the left edge, split into 3x3 slots by black separator rows.
    - Each slot contains one 4-connected single-color shape on teal background; color identifies a legend entry.
    - To the right, the canvas contains 3-4 solid 2x2 or 3x3 color squares on black background.

    Rule:
    - Read each legend slot as a 3x3 shape mask for its color.
    - For every canvas square, look up the mask with the same color and enlarge each mask cell into an s x s block, where s is the square side length.
    - Anchor the enlarged pattern at the square's top-left cell, then draw all patterns onto a blank canvas from farthest to nearest relative to the canvas center, letting later patterns overwrite earlier ones.
    - Remove the legend panel and separator column; output only the composited canvas.

    Key insight:
    The input squares encode both which legend shape to use and how much to scale it, while overlap conflicts are resolved only by distance to the canvas center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panel, canvas = _parse_panel_and_canvas(g)
    legend = _parse_legend(panel)
    blocks = _parse_blocks(canvas)

    assert set(int(block["color"]) for block in blocks).issubset(set(legend))
    dists = [_distance_key(block, canvas.shape) for block in blocks]
    assert len(set(dists)) == len(dists), "distance ties make layer order ambiguous"
    return _render_output(legend, blocks, canvas.shape)


def _pick_shapes(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    for _ in range(200):
        idxs = rng.choice(len(SHAPE_LIBRARY), size=count, replace=False)
        shapes = [SHAPE_LIBRARY[int(i)].copy() for i in idxs]
        if len({shape.tobytes() for shape in shapes}) < count:
            continue
        if not any(_bbox_from_coords(np.argwhere(shape)) != (0, 0, 2, 2) for shape in shapes):
            continue
        if len({int(shape.sum()) for shape in shapes}) == 1 and count >= 3:
            continue
        return shapes
    raise RuntimeError("failed to pick distinct legend shapes")


def _boxes_separated(a: dict, b: dict) -> bool:
    ar0, ac0 = int(a["r"]), int(a["c"])
    asz = int(a["size"])
    br0, bc0 = int(b["r"]), int(b["c"])
    bsz = int(b["size"])
    return (
        ar0 + asz < br0 - 1
        or br0 + bsz < ar0 - 1
        or ac0 + asz < bc0 - 1
        or bc0 + bsz < ac0 - 1
    )


def _place_blocks(
    rng: np.random.Generator,
    colors: list[int],
    sizes: list[int],
    canvas_shape: tuple[int, int],
) -> list[dict] | None:
    h, w = canvas_shape
    for _ in range(600):
        items = list(zip(colors, sizes))
        rng.shuffle(items)
        placed: list[dict] = []
        ok = True
        for color, size in items:
            anchors = [
                (r, c)
                for r in range(h - 3 * size + 1)
                for c in range(w - 3 * size + 1)
            ]
            rng.shuffle(anchors)
            chosen = None
            for r, c in anchors:
                candidate = {"color": int(color), "r": int(r), "c": int(c), "size": int(size)}
                if all(_boxes_separated(candidate, other) for other in placed):
                    chosen = candidate
                    break
            if chosen is None:
                ok = False
                break
            placed.append(chosen)
        if ok:
            return sorted(placed, key=lambda block: int(block["color"]))
    return None


def _overlap_cells(legend: dict[int, np.ndarray], a: dict, b: dict) -> int:
    ac = _scaled_cells(legend[int(a["color"])], int(a["r"]), int(a["c"]), int(a["size"]))
    bc = _scaled_cells(legend[int(b["color"])], int(b["r"]), int(b["c"]), int(b["size"]))
    aset = {tuple(map(int, rc)) for rc in ac}
    bset = {tuple(map(int, rc)) for rc in bc}
    return len(aset & bset)


def _legend_square_render(blocks: list[dict], canvas_shape: tuple[int, int]) -> np.ndarray:
    legend = {int(block["color"]): np.ones((3, 3), dtype=bool) for block in blocks}
    return _render_output(legend, blocks, canvas_shape)


def _semantic_sample_ok(
    legend: dict[int, np.ndarray],
    blocks: list[dict],
    canvas_shape: tuple[int, int],
) -> bool:
    sizes = {int(block["size"]) for block in blocks}
    if sizes != {2, 3}:
        return False

    dists = [_distance_key(block, canvas_shape) for block in blocks]
    if len(set(dists)) != len(dists):
        return False

    correct = _render_output(legend, blocks, canvas_shape)
    if np.count_nonzero(correct) < 12:
        return False

    pair_overlaps = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            overlap = _overlap_cells(legend, blocks[i], blocks[j])
            if overlap > 0:
                pair_overlaps.append((i, j, overlap))
    if not pair_overlaps:
        return False
    if not any(int(blocks[i]["size"]) != int(blocks[j]["size"]) for i, j, _ in pair_overlaps):
        return False

    reverse_layer = _render_output(legend, blocks, canvas_shape, order="distance-near")
    if np.array_equal(correct, reverse_layer):
        return False

    if np.array_equal(correct, _render_output(legend, blocks, canvas_shape, order="row-major")):
        return False
    if np.array_equal(correct, _render_output(legend, blocks, canvas_shape, order="reverse-row-major")):
        return False
    if np.array_equal(correct, _render_output(legend, blocks, canvas_shape, crop_legend=True)):
        return False
    alt_scale2 = _try_render_output(legend, blocks, canvas_shape, force_scale=2)
    if alt_scale2 is not None and np.array_equal(correct, alt_scale2):
        return False
    alt_scale3 = _try_render_output(legend, blocks, canvas_shape, force_scale=3)
    if alt_scale3 is not None and np.array_equal(correct, alt_scale3):
        return False
    if np.array_equal(correct, _legend_square_render(blocks, canvas_shape)):
        return False

    overlap_pixels = sum(overlap for _, _, overlap in pair_overlaps)
    if overlap_pixels > max(1, np.count_nonzero(correct) // 2):
        return False
    return True


def _build_input(
    legend: dict[int, np.ndarray],
    blocks: list[dict],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    h, w = canvas_shape
    panel_h = h
    grid = np.zeros((panel_h, 3 + 1 + w), dtype=int)
    panel = np.full((panel_h, 3), PANEL_COLOR, dtype=int)

    colors = sorted(legend)
    r = 0
    for idx, color in enumerate(colors):
        mask = legend[color]
        slot = np.full((3, 3), PANEL_COLOR, dtype=int)
        slot[mask] = int(color)
        panel[r : r + 3] = slot
        r += 3
        if idx != len(colors) - 1:
            panel[r] = BG
            r += 1
    grid[:, :3] = panel

    canvas = np.zeros(canvas_shape, dtype=int)
    for block in blocks:
        r0 = int(block["r"])
        c0 = int(block["c"])
        s = int(block["size"])
        canvas[r0 : r0 + s, c0 : c0 + s] = int(block["color"])
    grid[:, 4:] = canvas
    return grid


def _oracle_checks(
    grid: np.ndarray,
    legend: dict[int, np.ndarray],
    blocks: list[dict],
    canvas_shape: tuple[int, int],
) -> bool:
    panel, canvas = _parse_panel_and_canvas(grid)
    parsed_legend = _parse_legend(panel)
    parsed_blocks = _parse_blocks(canvas)
    if set(parsed_legend) != set(legend):
        return False
    for color in legend:
        if not np.array_equal(parsed_legend[color], legend[color]):
            return False

    original = sorted((int(b["color"]), int(b["r"]), int(b["c"]), int(b["size"])) for b in blocks)
    parsed = sorted((int(b["color"]), int(b["r"]), int(b["c"]), int(b["size"])) for b in parsed_blocks)
    if original != parsed:
        return False

    expected = _render_output(legend, blocks, canvas_shape)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        count = int(rng.integers(3, 5))
        canvas_shape = (4 * count - 1, int(rng.integers(10 + (count == 4), 15 + (count == 4))))
        colors = [int(c) for c in rng.choice(np.array(LEGEND_COLORS), size=count, replace=False)]
        shapes = _pick_shapes(rng, count)
        legend = {color: shape for color, shape in zip(colors, shapes, strict=True)}

        sizes = [2] * count
        for idx in rng.choice(count, size=int(rng.integers(1, count)), replace=False):
            sizes[int(idx)] = 3
        if set(sizes) != {2, 3}:
            continue

        blocks = _place_blocks(rng, colors, sizes, canvas_shape)
        if blocks is None:
            continue
        if not _semantic_sample_ok(legend, blocks, canvas_shape):
            continue

        grid = _build_input(legend, blocks, canvas_shape)
        if not _oracle_checks(grid, legend, blocks, canvas_shape):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
