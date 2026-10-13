from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sprites: list[dict] = []

    for comp in _components(g != BG):
        r0, c0, r1, c1 = _bbox(comp)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = np.unique(crop[crop != BG])
        assert len(colors) == 1, "each sprite must be monochrome apart from black holes"
        color = int(colors[0])

        row_counts = np.sum(crop == color, axis=1)
        has_tenon = int(row_counts[0]) == 1
        if has_tenon:
            tenon_cols = np.where(crop[0] == color)[0]
            assert len(tenon_cols) == 1
            tenon_col = int(tenon_cols[0])
            body = crop[1:, :].copy()
        else:
            tenon_col = None
            body = crop.copy()

        h, w = body.shape
        assert h >= 3 and w >= 3
        hole_cells = np.argwhere(body == BG)
        assert len(hole_cells) == 1, "each body must contain exactly one black hole"
        hole_row, hole_col = map(int, hole_cells[0])
        assert 0 < hole_row < h - 1 and 0 < hole_col < w - 1, "hole must be interior"

        filled = body.copy()
        filled[hole_row, hole_col] = color
        assert np.all(filled == color), "body must be a solid rectangle except for the hole"

        expected_counts = np.full(h, w, dtype=int)
        expected_counts[hole_row] -= 1
        assert np.array_equal(np.sum(body == color, axis=1), expected_counts)
        if has_tenon:
            assert 0 < tenon_col < w - 1, "tenon must align to a valid interior hole column"

        sprites.append(
            {
                "color": color,
                "body": body,
                "hole_row": hole_row,
                "hole_col": hole_col,
                "tenon_col": tenon_col,
                "bbox": (r0, c0, r1, c1),
            }
        )

    assert len(sprites) >= 2
    dims = {tuple(sprite["body"].shape) for sprite in sprites}
    assert len(dims) == 1, "all rectangles must share one body size"
    return sprites


def _recover_chain(sprites: list[dict]) -> list[int]:
    heads = [i for i, sprite in enumerate(sprites) if sprite["tenon_col"] is None]
    assert len(heads) == 1, "there must be exactly one tenonless head"

    hole_cols = [int(sprite["hole_col"]) for sprite in sprites]
    assert len(set(hole_cols)) == len(hole_cols), "hole columns must identify unique predecessors"

    by_tenon: dict[int, int] = {}
    for i, sprite in enumerate(sprites):
        tenon_col = sprite["tenon_col"]
        if tenon_col is None:
            continue
        assert tenon_col not in by_tenon, "each tenon column must identify one follower"
        by_tenon[int(tenon_col)] = i

    order = [heads[0]]
    used = {heads[0]}
    while len(order) < len(sprites):
        prev = sprites[order[-1]]
        nxt = by_tenon.get(int(prev["hole_col"]))
        assert nxt is not None, "non-tail hole must match exactly one follower tenon"
        assert nxt not in used, "matching must form a single chain, not a cycle"
        order.append(nxt)
        used.add(nxt)

    assert int(sprites[order[-1]]["hole_col"]) not in by_tenon, "tail hole must stay open"
    return order


def _build_output(sprites: list[dict], order: list[int]) -> np.ndarray:
    h, w = sprites[0]["body"].shape
    out = np.zeros((len(order) * h, w), dtype=int)
    for i, sprite_idx in enumerate(order):
        sprite = sprites[sprite_idx]
        out[i * h : (i + 1) * h, :] = sprite["body"]
        if i > 0:
            prev = sprites[order[i - 1]]
            out[(i - 1) * h + int(prev["hole_row"]), int(prev["hole_col"])] = int(sprite["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain same-size holey rectangles by matching each follower tenon column to the previous hole column, then fuse them into one vertical lock joint.

    Input structure:
    - Black background with several separated monochrome sprites.
    - Each sprite contains one same-size solid rectangle with exactly one interior black hole.
    - All but one sprites also have a single same-color tenon cell attached directly above the rectangle.

    Rule:
    - The only tenonless sprite is the chain head and goes first.
    - Repeatedly append the unique unused sprite whose tenon column equals the previous sprite's hole column.
    - Output the rectangles as a vertical concatenation; when a follower is attached, its color fills the previous hole and the tenon row itself disappears.

    Key insight:
    - Position in the input does not determine order; the chain is uniquely encoded by the geometric lock relation between hole columns and tenon columns.
    """
    sprites = _extract_sprites(np.asarray(grid, dtype=int))
    order = _recover_chain(sprites)
    return _build_output(sprites, order)


def _make_sprite(color: int, h: int, w: int, hole_row: int, hole_col: int, tenon_col: int | None) -> np.ndarray:
    body = np.full((h, w), color, dtype=int)
    body[hole_row, hole_col] = BG
    if tenon_col is None:
        return body
    sprite = np.zeros((h + 1, w), dtype=int)
    sprite[1:, :] = body
    sprite[0, tenon_col] = color
    return sprite


def _boxes_apart(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _scatter_sprites(sprites: list[np.ndarray], rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)
    for _ in range(120):
        h = rng.randint(max(12, max_h + 4), 30)
        w = rng.randint(max(12, max_w + 6), 30)
        grid = np.zeros((h, w), dtype=int)
        boxes: list[tuple[int, int, int, int]] = [(-99, -99, -99, -99)] * len(sprites)
        placed: list[tuple[int, int, int, int]] = []
        order = list(range(len(sprites)))
        rng.shuffle(order)
        ok = True
        for idx in order:
            sprite = sprites[idx]
            sh, sw = sprite.shape
            candidates = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)]
            rng.shuffle(candidates)
            chosen = None
            for r0, c0 in candidates:
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if _boxes_apart(box, placed):
                    chosen = box
                    break
            if chosen is None:
                ok = False
                break
            r0, c0, r1, c1 = chosen
            grid[r0 : r1 + 1, c0 : c1 + 1][sprite != BG] = sprite[sprite != BG]
            boxes[idx] = chosen
            placed.append(chosen)
        if ok:
            return grid, boxes
    return None


def _naive_orders_match(chain_ids: list[int], boxes: list[tuple[int, int, int, int]], hole_cols: list[int], tenon_cols: list[int | None]) -> bool:
    ids = list(range(len(chain_ids)))
    naive_orders = [
        sorted(ids, key=lambda i: (boxes[i][0], boxes[i][1])),
        sorted(ids, key=lambda i: (boxes[i][1], boxes[i][0])),
        sorted(ids, key=lambda i: (boxes[i][2], boxes[i][3])),
        sorted(ids, key=lambda i: hole_cols[i]),
        sorted(ids, key=lambda i: (-1 if tenon_cols[i] is None else tenon_cols[i], hole_cols[i])),
    ]
    return any(order == chain_ids for order in naive_orders)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    k = rng.randint(3, 5)
    h = rng.randint(3, min(6, 30 // k))
    w = rng.randint(k + 2, min(9, 30))

    colors = rng.sample(PALETTE, k)
    hole_cols = rng.sample(list(range(1, w - 1)), k)
    hole_rows = [rng.randint(1, h - 2) for _ in range(k)]
    tenon_cols = [None] + hole_cols[:-1]

    sprites = [
        _make_sprite(colors[i], h, w, hole_rows[i], hole_cols[i], tenon_cols[i])
        for i in range(k)
    ]
    expected = _build_output(
        [
            {
                "color": colors[i],
                "body": sprites[i][1:, :] if tenon_cols[i] is not None else sprites[i],
                "hole_row": hole_rows[i],
                "hole_col": hole_cols[i],
                "tenon_col": tenon_cols[i],
            }
            for i in range(k)
        ],
        list(range(k)),
    )

    if hole_cols == sorted(hole_cols) or hole_cols == sorted(hole_cols, reverse=True):
        return None

    scattered = _scatter_sprites(sprites, rng)
    if scattered is None:
        return None
    grid, boxes = scattered

    chain_ids = list(range(k))
    if _naive_orders_match(chain_ids, boxes, hole_cols, tenon_cols):
        return None

    try:
        parsed = _extract_sprites(grid)
        order = _recover_chain(parsed)
        solved = _build_output(parsed, order)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None

    segment_black_counts = [int(np.sum(solved[i * h : (i + 1) * h, :] == BG)) for i in range(k)]
    if segment_black_counts[:-1] != [0] * (k - 1) or segment_black_counts[-1] != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-chain instance")
