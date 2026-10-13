from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, paste  # noqa: E402


BG_COLORS = (1, 5)
FG_COLORS = (2, 3, 4, 6, 7, 8, 9)
SHAPES = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
EDGE_TYPES = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")
GEN_RETRIES = 400


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _bbox_intersects_with_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _touches_for_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[str, ...]:
    r0, c0, r1, c1 = bbox
    H, W = shape
    edges: list[str] = []
    if r0 == 0:
        edges.append("top")
    if r1 == H - 1:
        edges.append("bottom")
    if c0 == 0:
        edges.append("left")
    if c1 == W - 1:
        edges.append("right")
    return tuple(edges)


def _destination_bbox(rect: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    dr = 0
    dc = 0
    touches = set(rect["touches"])
    if "top" in touches:
        dr += h
    if "bottom" in touches:
        dr -= h
    if "left" in touches:
        dc += w
    if "right" in touches:
        dc -= w
    nr0 = r0 + dr
    nc0 = c0 + dc
    nr1 = r1 + dr
    nc1 = c1 + dc
    H, W = shape
    assert 0 <= nr0 <= nr1 < H
    assert 0 <= nc0 <= nc1 < W
    return nr0, nc0, nr1, nc1


def _extract_boundary_rectangles(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    rects: list[dict] = []
    for comp in find_components(grid, bg=-1, connectivity=4, monochromatic=True, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if int(comp["area"]) != h * w or int(comp["area"]) < 4:
            continue
        touches = _touches_for_bbox((r0, c0, r1, c1), (H, W))
        if not touches:
            continue
        if {"top", "bottom"} <= set(touches) or {"left", "right"} <= set(touches):
            continue
        rects.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "touches": touches,
                "area": int(comp["area"]),
            }
        )
    rects.sort(key=lambda rect: (rect["color"], rect["bbox"]))
    return rects


def _infer_periodic_tile(grid: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    coords = np.argwhere(visible_mask)
    assert coords.size > 0
    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            mapping: dict[tuple[int, int], int] = {}
            ok = True
            for r, c in coords:
                key = (int(r) % pr, int(c) % pc)
                value = int(grid[r, c])
                if key in mapping and mapping[key] != value:
                    ok = False
                    break
                mapping[key] = value
            if ok and len(mapping) == pr * pc:
                tile = np.zeros((pr, pc), dtype=int)
                for (rr, cc), value in mapping.items():
                    tile[rr, cc] = value
                return tile
    raise AssertionError("failed to infer a unique periodic tile")


def _paint_background_patch(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    tile: np.ndarray,
) -> None:
    r0, c0, r1, c1 = bbox
    pr, pc = tile.shape
    rows = np.arange(r0, r1 + 1) % pr
    cols = np.arange(c0, c1 + 1) % pc
    out[r0 : r1 + 1, c0 : c1 + 1] = tile[np.ix_(rows, cols)]


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    rects = _extract_boundary_rectangles(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for rect in rects:
        by_color[int(rect["color"])].append(rect)

    moving_colors = [color for color, items in by_color.items() if len(items) > 1]
    assert len(moving_colors) == 1
    moving_color = int(moving_colors[0])

    visible_mask = np.ones((H, W), dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        visible_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    tile = _infer_periodic_tile(g, visible_mask)

    out = g.copy()
    moving_rects = by_color[moving_color]
    for rect in moving_rects:
        _paint_background_patch(out, rect["bbox"], tile)

    for rect in moving_rects:
        r0, c0, r1, c1 = rect["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        nr0, nc0, nr1, nc1 = _destination_bbox(rect, (H, W))
        patch = np.full((h, w), moving_color, dtype=int)
        paste(out, patch, nr0, nc0)

    if return_meta:
        return out, {"rects": rects, "moving_color": moving_color, "tile": tile}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every repeated boundary rectangle inward and repair its old position with the inferred periodic background.

    Input structure:
    - The canvas is covered by a two-color periodic background tile that remains fully visible outside the occluders.
    - Several solid monochrome rectangles touch at least one full grid edge; some touch one edge, others sit in corners.
    - Exactly one foreground color appears as multiple disconnected boundary rectangles, while every other rectangle color appears once.

    Rule:
    - Detect all solid boundary rectangles and identify the color that repeats across multiple rectangles.
    - Infer the smallest periodic background tile from the visible non-rectangle cells before removing anything.
    - Erase every rectangle of the repeated color by restoring its source bbox from the global periodic tile.
    - Translate each erased rectangle inward, one rectangle-size per touched edge: top moves down by its height, bottom up by its height, left right by its width, right left by its width, and corner rectangles apply both shifts.
    - Leave singleton-color rectangles in place; if a moved rectangle lands on one, the moved rectangle overwrites it.

    Key insight:
    - Color multiplicity is the single trigger for both geometry and repair: it selects which rectangles move and simultaneously marks the exact bboxes that must be rebuilt from the background period inferred from the still-visible canvas.
    """
    return _solve(grid)


def _checker_background(H: int, W: int, tile: np.ndarray) -> np.ndarray:
    rows = (np.arange(H)[:, None]) % 2
    cols = (np.arange(W)[None, :]) % 2
    return tile[rows, cols].copy()


def _candidate_rectangles(H: int, W: int) -> list[dict]:
    candidates: list[dict] = []
    for h, w in SHAPES:
        if 2 * h > H and 2 * w > W:
            continue
        if 2 * h <= H:
            for c0 in range(1, W - w):
                candidates.append({"bbox": (0, c0, h - 1, c0 + w - 1), "touches": ("top",)})
                candidates.append({"bbox": (H - h, c0, H - 1, c0 + w - 1), "touches": ("bottom",)})
        if 2 * w <= W:
            for r0 in range(1, H - h):
                candidates.append({"bbox": (r0, 0, r0 + h - 1, w - 1), "touches": ("left",)})
                candidates.append({"bbox": (r0, W - w, r0 + h - 1, W - 1), "touches": ("right",)})
        if 2 * h <= H and 2 * w <= W:
            candidates.extend(
                [
                    {"bbox": (0, 0, h - 1, w - 1), "touches": ("top", "left")},
                    {"bbox": (0, W - w, h - 1, W - 1), "touches": ("top", "right")},
                    {"bbox": (H - h, 0, H - 1, w - 1), "touches": ("bottom", "left")},
                    {"bbox": (H - h, W - w, H - 1, W - 1), "touches": ("bottom", "right")},
                ]
            )

    dedup: dict[tuple[int, int, int, int], dict] = {}
    for cand in candidates:
        bbox = tuple(map(int, cand["bbox"]))
        touches = tuple(cand["touches"])
        rect = {"bbox": bbox, "touches": touches, "corner": len(touches) == 2}
        rect["dst_bbox"] = _destination_bbox(rect, (H, W))
        dedup[bbox] = rect
    return list(dedup.values())


def _pick_candidate(
    candidates: list[dict],
    chosen: list[dict],
    rng: random.Random,
    *,
    corner_rate: float,
    reject_boxes: list[tuple[int, int, int, int]] | None = None,
) -> dict | None:
    reject_boxes = [] if reject_boxes is None else reject_boxes
    valid = []
    for cand in candidates:
        if any(cand["bbox"] == other["bbox"] for other in chosen):
            continue
        if any(_bbox_intersects_with_gap(cand["bbox"], other["bbox"], gap=1) for other in chosen):
            continue
        if any(_bbox_intersects(cand["bbox"], box) for box in reject_boxes):
            continue
        valid.append(cand)
    if not valid:
        return None

    corners = [cand for cand in valid if cand["corner"]]
    sides = [cand for cand in valid if not cand["corner"]]
    use_corners = bool(corners) and (not sides or rng.random() < corner_rate)
    pool = corners if use_corners else sides if sides else corners
    return dict(rng.choice(pool))


def _draw_rectangles(base: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = base.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return out


def _build_expected(inp: np.ndarray, moving_color: int) -> np.ndarray:
    return _solve(inp)


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    all_rects: list[dict],
    moving_color: int,
    tile: np.ndarray,
    overlap_target: bool,
) -> bool:
    got, meta = _solve(inp, return_meta=True)
    if not np.array_equal(got, expected):
        return False
    if int(meta["moving_color"]) != int(moving_color):
        return False

    detected = sorted((rect["color"], rect["bbox"]) for rect in meta["rects"])
    intended = sorted((int(rect["color"]), tuple(rect["bbox"])) for rect in all_rects)
    if detected != intended:
        return False
    if not np.array_equal(meta["tile"], tile):
        return False

    moving = [rect for rect in all_rects if rect["color"] == moving_color]
    stationary = [rect for rect in all_rects if rect["color"] != moving_color]
    overlap_exists = any(
        _bbox_intersects(_destination_bbox(rect, inp.shape), other["bbox"])
        for rect in moving
        for other in stationary
    )
    if overlap_exists != overlap_target:
        return False

    visible_mask = np.ones(inp.shape, dtype=bool)
    for rect in all_rects:
        r0, c0, r1, c1 = rect["bbox"]
        visible_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    if {tuple(map(int, rc)) for rc in np.argwhere(visible_mask)} == set():
        return False
    if {((r % 2), (c % 2)) for r, c in np.argwhere(visible_mask)} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return False
    return True


def _sample_layout(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    H = rng.randint(12, 17)
    W = rng.randint(12, 17)
    tile = np.array(
        [[BG_COLORS[rng.randint(0, 1)], 0], [0, 0]],
        dtype=int,
    )
    tile[0, 1] = BG_COLORS[1] if tile[0, 0] == BG_COLORS[0] else BG_COLORS[0]
    tile[1, 0] = tile[0, 1]
    tile[1, 1] = tile[0, 0]

    overlap_target = rng.random() < 0.4
    num_singletons = rng.randint(1, 3)
    moving_color = rng.choice(FG_COLORS)
    singleton_colors = rng.sample([color for color in FG_COLORS if color != moving_color], num_singletons)

    candidates = _candidate_rectangles(H, W)
    for _ in range(200):
        chosen: list[dict] = []
        moving_rects: list[dict] = []

        first = _pick_candidate(candidates, chosen, rng, corner_rate=0.45)
        if first is None:
            return None
        chosen.append(first)
        moving_rects.append(first)

        second = _pick_candidate(
            candidates,
            chosen,
            rng,
            corner_rate=0.45,
            reject_boxes=[rect["dst_bbox"] for rect in moving_rects],
        )
        if second is None or _bbox_intersects(second["dst_bbox"], first["dst_bbox"]):
            continue
        chosen.append(second)
        moving_rects.append(second)

        # Keep moved rectangles from overwriting each other's source areas.
        if any(
            _bbox_intersects(rect["dst_bbox"], other["bbox"])
            for i, rect in enumerate(moving_rects)
            for j, other in enumerate(moving_rects)
            if i != j
        ):
            continue

        stationary_rects: list[dict] = []
        ok = True
        for _idx in range(num_singletons):
            cand = _pick_candidate(candidates, chosen, rng, corner_rate=0.25)
            if cand is None:
                ok = False
                break
            chosen.append(cand)
            stationary_rects.append(cand)
        if not ok:
            continue

        # Do not allow stationary rectangles to mask moving sources in the input.
        if any(_bbox_intersects(m["bbox"], s["bbox"]) for m in moving_rects for s in stationary_rects):
            continue

        overlap_exists = any(
            _bbox_intersects(m["dst_bbox"], s["bbox"])
            for m in moving_rects
            for s in stationary_rects
        )
        if overlap_exists != overlap_target:
            continue

        base = _checker_background(H, W, tile)
        moving = [{**rect, "color": moving_color} for rect in moving_rects]
        stationary = [
            {**rect, "color": color}
            for rect, color in zip(stationary_rects, singleton_colors)
        ]
        all_rects = moving + stationary
        inp = _draw_rectangles(base, all_rects)
        expected = _build_expected(inp, moving_color)
        if _validate_instance(inp, expected, all_rects, moving_color, tile, overlap_target):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        sample = _sample_layout(rng)
        if sample is None:
            continue
        inp, _expected = sample
        return inp
    raise RuntimeError("failed to generate valid input for repeated-color periodic repair task")
