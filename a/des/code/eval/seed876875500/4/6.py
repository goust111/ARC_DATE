from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

STAMP_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE]

PAIR_COUNT_CHOICES = (2, 3, 4)
DIM_RANGES = {
    2: {"h": (4, 6), "w": (4, 6), "pair_gap": (1, 3), "frame_gap": (1, 3), "out_extra": (0, 2)},
    3: {"h": (4, 5), "w": (4, 5), "pair_gap": (1, 2), "frame_gap": (1, 2), "out_extra": (0, 2)},
    4: {"h": (4, 5), "w": (4, 4), "pair_gap": (1, 1), "frame_gap": (1, 1), "out_extra": (0, 1)},
}
TOP_RANGE = (1, 6)
LEFT_MARGIN_RANGE = (1, 2)
RIGHT_MARGIN_RANGE = (1, 2)
BOTTOM_MARGIN_RANGE = (1, 2)
GRID_MAX_H = 24
GRID_MAX_W = 30
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _draw_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = int(color)
    grid[r0 + h - 1, c0 : c0 + w] = int(color)
    grid[r0 : r0 + h, c0] = int(color)
    grid[r0 : r0 + h, c0 + w - 1] = int(color)


def _frame_with_inner(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    out = np.full((h + 2, w + 2), BLACK, dtype=int)
    _draw_border(out, 0, 0, h + 2, w + 2, MAROON)
    out[1:-1, 1:-1] = inner
    return out


def _is_filled_rectangle(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return int(coords.shape[0]) == area and bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _find_output_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not bool(np.all(patch[border_mask] == MAROON)):
            continue
        if not bool(np.all(patch[~border_mask] == BLACK)):
            continue
        candidates.append((int(h * w), (int(r0), int(c0), int(r1), int(c1))))
    assert len(candidates) == 1, "expected exactly one maroon output frame"
    return candidates[0][1]


def _extract_pairs(grid: np.ndarray) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    colors = sorted(set(map(int, grid.flatten())) - {BLACK, GREY, MAROON})
    for color in colors:
        for coords in _components(grid == int(color)):
            if not _is_filled_rectangle(grid, int(color), coords):
                continue
            sr0, sc0, sr1, sc1 = _bbox(coords)
            h = int(sr1 - sr0 + 1)
            w = int(sc1 - sc0 + 1)
            tr0 = int(sr0 - h - 1)
            tr1 = int(sr0 - 2)
            if tr0 < 0 or tr1 < tr0:
                continue
            if not bool(np.all(grid[sr0 - 1, sc0 : sc1 + 1] == BLACK)):
                continue
            template = grid[tr0 : tr1 + 1, sc0 : sc1 + 1]
            if template.shape != (h, w):
                continue
            vals = set(map(int, np.unique(template)))
            if not vals.issubset({BLACK, GREY}) or BLACK not in vals or GREY not in vals:
                continue
            if not (
                bool(np.all(template[0, :] == GREY))
                and bool(np.all(template[-1, :] == GREY))
                and bool(np.all(template[:, 0] == GREY))
                and bool(np.all(template[:, -1] == GREY))
            ):
                continue
            pairs.append(
                {
                    "color": int(color),
                    "template_bbox": (int(tr0), int(sc0), int(tr1), int(sc1)),
                    "stamp_bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                    "holes": template == BLACK,
                }
            )
    pairs.sort(key=lambda item: (int(item["template_bbox"][1]), int(item["template_bbox"][0])))
    assert pairs, "expected at least one template-stamp pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each stamp fragment selected by the black holes in the grey template above it,
    then overlay all fragments in left-to-right template order inside the maroon frame.
    """
    g = np.asarray(grid, dtype=int)
    fr0, fc0, fr1, fc1 = _find_output_frame(g)
    out = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    out[1:-1, 1:-1] = BLACK
    inner_h = int(out.shape[0] - 2)
    inner_w = int(out.shape[1] - 2)

    for pair in _extract_pairs(g):
        holes = np.asarray(pair["holes"], dtype=bool)
        h, w = holes.shape
        assert h <= inner_h and w <= inner_w, "output frame must fit every fragment"
        patch = out[1 : 1 + h, 1 : 1 + w]
        patch[holes] = int(pair["color"])
    return out


def _sample_holes(
    h: int,
    w: int,
    forced: set[tuple[int, int]],
) -> np.ndarray:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    assert interior
    forced = set(forced)
    assert forced.issubset(set(interior))

    holes = set(forced)
    remaining = [cell for cell in interior if cell not in holes]
    random.shuffle(remaining)

    min_count = max(len(holes), 2 if len(interior) >= 2 else 1)
    max_count = len(interior) if len(interior) <= 2 else len(interior) - 1
    max_count = max(min_count, max_count)
    target = random.randint(min_count, max_count)
    holes.update(remaining[: target - len(holes)])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in holes:
        mask[int(r), int(c)] = True
    return mask


def _compose_inner(pairs: list[dict[str, object]], out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray]:
    inner = np.full((out_h, out_w), BLACK, dtype=int)
    cover = np.zeros((out_h, out_w), dtype=int)
    for pair in pairs:
        holes = np.asarray(pair["holes"], dtype=bool)
        h, w = holes.shape
        inner[:h, :w][holes] = int(pair["color"])
        cover[:h, :w] += holes.astype(int)
    return inner, cover


def _build_input(
    pairs: list[dict[str, object]],
    out_h: int,
    out_w: int,
    frame_r: int,
    frame_c: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for pair in pairs:
        h = int(pair["h"])
        w = int(pair["w"])
        tr = int(pair["template_r"])
        tc = int(pair["template_c"])

        template = np.full((h, w), GREY, dtype=int)
        template[np.asarray(pair["holes"], dtype=bool)] = BLACK
        grid[tr : tr + h, tc : tc + w] = template

        sr = tr + h + 1
        grid[sr : sr + h, tc : tc + w] = int(pair["color"])

    _draw_border(grid, frame_r, frame_c, out_h + 2, out_w + 2, MAROON)
    return grid


def _generate_case() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    cfg = DIM_RANGES[n_pairs]
    colors = random.sample(STAMP_COLORS, n_pairs)

    pairs: list[dict[str, object]] = []
    for color in colors:
        h = random.randint(*cfg["h"])
        w = random.randint(*cfg["w"])
        pairs.append({"color": int(color), "h": int(h), "w": int(w)})

    overlap_indices = sorted(random.sample(range(n_pairs), random.randint(2, min(3, n_pairs))))
    min_h = min(int(pairs[idx]["h"]) for idx in overlap_indices)
    min_w = min(int(pairs[idx]["w"]) for idx in overlap_indices)
    shared_pool = [(r, c) for r in range(1, min_h - 1) for c in range(1, min_w - 1)]
    assert shared_pool
    shared_count = random.randint(1, min(2, len(shared_pool)))
    shared_cells = random.sample(shared_pool, shared_count)

    forced: list[set[tuple[int, int]]] = [set() for _ in range(n_pairs)]
    for idx in overlap_indices:
        forced[idx].update(shared_cells)
    for idx, pair in enumerate(pairs):
        pair["holes"] = _sample_holes(int(pair["h"]), int(pair["w"]), forced[idx])

    out_h = max(int(pair["h"]) for pair in pairs) + random.randint(*cfg["out_extra"])
    out_w = max(int(pair["w"]) for pair in pairs) + random.randint(*cfg["out_extra"])

    x = random.randint(*LEFT_MARGIN_RANGE)
    for idx, pair in enumerate(pairs):
        pair["template_c"] = int(x)
        pair["template_r"] = int(random.randint(*TOP_RANGE))
        x += int(pair["w"])
        if idx != n_pairs - 1:
            x += random.randint(*cfg["pair_gap"])

    frame_r = random.randint(*TOP_RANGE)
    frame_c = int(x + random.randint(*cfg["frame_gap"]))
    grid_h = max(
        max(int(pair["template_r"]) + 2 * int(pair["h"]) + 1 for pair in pairs),
        int(frame_r + out_h + 2),
    ) + random.randint(*BOTTOM_MARGIN_RANGE)
    grid_w = int(frame_c + out_w + 2 + random.randint(*RIGHT_MARGIN_RANGE))
    if grid_h > GRID_MAX_H or grid_w > GRID_MAX_W:
        return None

    inner, cover = _compose_inner(pairs, out_h, out_w)
    visible_colors = {int(v) for v in np.unique(inner) if int(v) != BLACK}
    expected_colors = {int(pair["color"]) for pair in pairs}
    if visible_colors != expected_colors:
        return None
    if not bool(np.any(cover > 1)):
        return None
    if int(np.count_nonzero(inner)) < n_pairs + 1:
        return None

    grid = _build_input(pairs, out_h, out_w, frame_r, frame_c, grid_h, grid_w)
    expected = _frame_with_inner(inner)

    if len(_extract_pairs(grid)) != n_pairs:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-stamp puzzle")
