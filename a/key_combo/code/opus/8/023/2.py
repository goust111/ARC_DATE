from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

try:
    _UTILS = Path("/home/lixinyu/snap/new/solvers")
    if _UTILS.exists() and str(_UTILS) not in sys.path:
        sys.path.append(str(_UTILS))
    from arc_utils import Color, find_components, get_bg, neighbors4  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5

    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if connectivity != 4 or not monochromatic or return_format != "dict":
            raise NotImplementedError("fallback find_components only supports 4-connected monochromatic dict mode")
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        if colors is None:
            colors = sorted(set(map(int, g.ravel())) - {int(bg)})
        out: list[dict] = []
        for color in colors:
            mask = g == int(color)
            seen = np.zeros(g.shape, dtype=bool)
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):
                    if seen[r, c] or not mask[r, c]:
                        continue
                    q = deque([(r, c)])
                    seen[r, c] = True
                    coords = []
                    while q:
                        rr, cc = q.popleft()
                        coords.append((rr, cc))
                        for nr, nc in neighbors4(rr, cc):
                            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    if len(coords) < int(min_area):
                        continue
                    arr = np.array(coords, dtype=int)
                    r0, c0 = arr.min(axis=0)
                    r1, c1 = arr.max(axis=0)
                    out.append(
                        {
                            "color": int(color),
                            "coords": arr,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "area": int(arr.shape[0]),
                        }
                    )
        return out


SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]


def _boundary_openings(local: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = local.shape
    return {
        "up": [(0, c) for c in range(w) if int(local[0, c]) == Color.BLACK],
        "down": [(h - 1, c) for c in range(w) if int(local[h - 1, c]) == Color.BLACK],
        "left": [(r, 0) for r in range(h) if int(local[r, 0]) == Color.BLACK],
        "right": [(r, w - 1) for r in range(h) if int(local[r, w - 1]) == Color.BLACK],
    }


def _reachable_black(local: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    seen = np.zeros(local.shape, dtype=bool)
    q = deque()
    for r, c in seeds:
        if int(local[r, c]) == Color.BLACK and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < local.shape[0] and 0 <= nc < local.shape[1] and not seen[nr, nc] and int(local[nr, nc]) == Color.BLACK:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _analyze_cups(scene: np.ndarray) -> list[dict]:
    cups = find_components(
        scene,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    analyzed = []
    for comp in cups:
        r0, c0, r1, c1 = comp["bbox"]
        local = np.asarray(scene[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        openings = _boundary_openings(local)
        open_sides = [side for side, pts in openings.items() if pts]
        assert len(open_sides) == 1, "each cup must have exactly one opening side"
        side = open_sides[0]
        reachable = _reachable_black(local, openings[side])
        internal_mask = (local == Color.BLACK) & ~reachable
        internal_local = [tuple(map(int, p)) for p in np.argwhere(internal_mask)]
        internal_global = [(r0 + r, c0 + c) for r, c in internal_local]
        opening_global = [(r0 + r, c0 + c) for r, c in openings[side]]
        analyzed.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "wall_coords": [(int(r), int(c)) for r, c in comp["coords"]],
                "opening": opening_global,
                "side": side,
                "direction": SIDE_TO_DIR[side],
                "internal_coords": internal_global,
                "area": int(len(internal_global)),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "wall_area": int(comp["area"]),
            }
        )
    return analyzed


def _ray_cells(scene: np.ndarray, cup: dict) -> tuple[list[tuple[int, int]], bool]:
    h, w = scene.shape
    dr, dc = cup["direction"]
    cells: list[tuple[int, int]] = []
    blocked = False
    for sr, sc in cup["opening"]:
        r, c = int(sr), int(sc)
        while 0 <= r < h and 0 <= c < w:
            if int(scene[r, c]) == Color.GREY:
                blocked = True
                break
            cells.append((r, c))
            r += dr
            c += dc
    return cells, blocked


def _paint_expected(scene: np.ndarray, palette: list[int], cups: list[dict]) -> np.ndarray:
    out = scene.copy()
    ordered = sorted(cups, key=lambda cup: cup["area"])
    for color, cup in zip(palette, ordered):
        for r, c in cup["wall_coords"]:
            out[r, c] = int(color)
        for r, c in cup["internal_coords"]:
            out[r, c] = int(color)
        cup["assigned_color"] = int(color)
    for cup in ordered:
        ray, _ = _ray_cells(scene, cup)
        for r, c in ray:
            out[r, c] = int(cup["assigned_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor cups by enclosed area rank and emit matching rays through their openings.

    Input structure:
    - The grid has black background, 3-4 grey cup-outline objects, and a palette row on the bottom edge.
    - Each cup is a 4-connected grey outline with exactly one boundary opening on one side.
    - Inside each cup, some black cells are enclosed by the cup walls, while black cells reachable from the opening remain exterior.

    Rule:
    - Ignore the bottom palette row while analyzing the cups, but use its non-black pixels from left to right as the ranking palette.
    - For each grey cup, flood 4-connected black cells from the opening side inside its bbox; black cells in the bbox not reached by that flood are the cup's enclosed interior.
    - Count each cup's enclosed interior area, sort cups from smallest area to largest, and recolor that cup's wall plus enclosed interior with the corresponding palette color.
    - From every opening cell, extend a straight ray outward in the opening direction using the assigned color, stopping before any other cup wall; drop the palette row from the output.

    Key insight:
    - The cup color is determined globally by enclosed-area ranking, while the ray direction is determined locally by the opening side, so both the flood-fill and the opening detection are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    assert max(g.shape) <= 30

    palette = [int(v) for v in g[-1] if int(v) != Color.BLACK]
    assert 3 <= len(palette) <= 4
    scene = g[:-1].copy()

    cups = _analyze_cups(scene)
    assert len(cups) == len(palette)
    areas = [cup["area"] for cup in cups]
    assert min(areas) > 0
    assert len(set(areas)) == len(areas)

    return _paint_expected(scene, palette, cups)


def _make_outer_frame(h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), Color.BLACK, dtype=int)
    patch[0, :] = Color.GREY
    patch[-1, :] = Color.GREY
    patch[:, 0] = Color.GREY
    patch[:, -1] = Color.GREY
    return patch


def _carve_gap(patch: np.ndarray, side: str, start: int, gap_width: int) -> None:
    if side == "up":
        patch[0, start : start + gap_width] = Color.BLACK
    elif side == "down":
        patch[-1, start : start + gap_width] = Color.BLACK
    elif side == "left":
        patch[start : start + gap_width, 0] = Color.BLACK
    else:
        patch[start : start + gap_width, -1] = Color.BLACK


def _add_parallel_bar(patch: np.ndarray, side: str) -> None:
    h, w = patch.shape
    if side == "up":
        patch[1, 1 : w - 1] = Color.GREY
    elif side == "down":
        patch[h - 2, 1 : w - 1] = Color.GREY
    elif side == "left":
        patch[1 : h - 1, 1] = Color.GREY
    else:
        patch[1 : h - 1, w - 2] = Color.GREY


def _build_parallel(side: str) -> tuple[np.ndarray, dict] | None:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    if side in ("up", "down") and w < 5:
        return None
    if side in ("left", "right") and h < 5:
        return None
    gap_width = 2 if random.random() < 0.35 else 1
    span = (w - 2) if side in ("up", "down") else (h - 2)
    if span < gap_width:
        return None
    start = random.randint(1, span - gap_width + 1)
    patch = _make_outer_frame(h, w)
    _carve_gap(patch, side, start, gap_width)
    _add_parallel_bar(patch, side)
    cups = _analyze_cups(patch)
    if len(cups) != 1 or cups[0]["side"] != side:
        return None
    area = cups[0]["area"]
    if not (4 <= area <= 20):
        return None
    return patch, {"style": "parallel", "gap_width": gap_width, "area": area}


def _build_divider(side: str) -> tuple[np.ndarray, dict] | None:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    gap_width = 2 if random.random() < 0.35 else 1
    patch = _make_outer_frame(h, w)
    if side in ("up", "down"):
        if w < 6:
            return None
        divider = random.randint(2, w - 3)
        open_left = random.choice([True, False])
        open_span = divider - 1 if open_left else (w - 2 - divider)
        sealed_span = (w - 2 - divider) if open_left else divider - 1
        if open_span < gap_width or sealed_span < 1:
            return None
        start = random.randint(1, divider - gap_width) if open_left else random.randint(divider + 1, w - 1 - gap_width)
        _carve_gap(patch, side, start, gap_width)
        patch[1 : h - 1, divider] = Color.GREY
    else:
        if h < 6:
            return None
        divider = random.randint(2, h - 3)
        open_top = random.choice([True, False])
        open_span = divider - 1 if open_top else (h - 2 - divider)
        sealed_span = (h - 2 - divider) if open_top else divider - 1
        if open_span < gap_width or sealed_span < 1:
            return None
        start = random.randint(1, divider - gap_width) if open_top else random.randint(divider + 1, h - 1 - gap_width)
        _carve_gap(patch, side, start, gap_width)
        patch[divider, 1 : w - 1] = Color.GREY
    cups = _analyze_cups(patch)
    if len(cups) != 1 or cups[0]["side"] != side:
        return None
    area = cups[0]["area"]
    if not (4 <= area <= 20):
        return None
    return patch, {"style": "divider", "gap_width": gap_width, "area": area}


def _build_cup(side: str, style: str) -> dict | None:
    builder = _build_parallel if style == "parallel" else _build_divider
    built = builder(side)
    if built is None:
        return None
    patch, extra = built
    cup = _analyze_cups(patch)[0]
    cup.update(extra)
    cup["patch"] = patch
    return cup


def _rank_signature(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda i: (values[i], i)))


def _cup_bboxes_do_not_touch(placed: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = cand
    for a0, b0, a1, b1 in placed:
        if not (r1 + pad < a0 or a1 + pad < r0 or c1 + pad < b0 or b1 + pad < c0):
            return False
    return True


def _paste_patch(scene: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != Color.BLACK
    scene[top : top + h, left : left + w][mask] = patch[mask]


def _translate_cup(cup: dict, top: int, left: int) -> dict:
    out = {}
    for key, value in cup.items():
        out[key] = value
    r0, c0, r1, c1 = cup["bbox"]
    out["bbox"] = (r0 + top, c0 + left, r1 + top, c1 + left)
    out["wall_coords"] = [(r + top, c + left) for r, c in cup["wall_coords"]]
    out["opening"] = [(r + top, c + left) for r, c in cup["opening"]]
    out["internal_coords"] = [(r + top, c + left) for r, c in cup["internal_coords"]]
    return out


def _make_palette_row(width: int, n: int) -> np.ndarray:
    row = np.full(width, Color.BLACK, dtype=int)
    colors = PALETTE[:n]
    span = 2 * n - 1
    start = random.randint(1, width - span - 1)
    for i, color in enumerate(colors):
        row[start + 2 * i] = int(color)
    return row


def _scene_valid(scene: np.ndarray, cups: list[dict], *, final: bool) -> bool:
    paths: list[set[tuple[int, int]]] = []
    cup_bboxes = [cup["bbox"] for cup in cups]
    blocked_count = 0
    for i, cup in enumerate(cups):
        cells, blocked = _ray_cells(scene, cup)
        blocked_count += int(blocked)
        cell_set = set(cells)
        for r, c in cell_set - set(cup["opening"]):
            for j, (r0, c0, r1, c1) in enumerate(cup_bboxes):
                if i != j and r0 <= r <= r1 and c0 <= c <= c1:
                    return False
        paths.append(cell_set)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if paths[i] & paths[j]:
                return False
    if not final:
        return True
    area_rank = _rank_signature([cup["area"] for cup in cups])
    if area_rank == _rank_signature([cup["bbox_area"] for cup in cups]):
        return False
    if area_rank == _rank_signature([cup["wall_area"] for cup in cups]):
        return False
    if area_rank == _rank_signature([cup["bbox"][1] for cup in cups]):
        return False
    if area_rank == _rank_signature([cup["bbox"][0] for cup in cups]):
        return False
    if len({cup["side"] for cup in cups}) < min(3, len(cups)):
        return False
    if len({cup["style"] for cup in cups}) < 2:
        return False
    if blocked_count > 1:
        return False
    return True


def _generate_scene() -> np.ndarray | None:
    n = random.choice([3, 4, 4])
    scene_h = random.randint(16, 23)
    scene_w = random.randint(18, 24)
    sides = random.sample(["up", "down", "left", "right"], n)
    styles = ["parallel", "divider"] * 2
    random.shuffle(styles)
    styles = styles[:n]

    cups: list[dict] = []
    used_areas: set[int] = set()
    for side, style in zip(sides, styles):
        chosen = None
        for _ in range(120):
            cup = _build_cup(side, style)
            if cup is None:
                continue
            if cup["area"] in used_areas:
                continue
            chosen = cup
            used_areas.add(int(cup["area"]))
            break
        if chosen is None:
            return None
        cups.append(chosen)

    scene = np.full((scene_h, scene_w), Color.BLACK, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_cups: list[dict] = []
    order = sorted(range(n), key=lambda i: cups[i]["patch"].shape[0] * cups[i]["patch"].shape[1], reverse=True)

    for idx in order:
        patch = cups[idx]["patch"]
        h, w = patch.shape
        r_lo, c_lo = 2, 2
        r_hi = scene_h - h - 1
        c_hi = scene_w - w - 2
        if r_hi < r_lo or c_hi < c_lo:
            return None
        positions = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(positions)
        placed = False
        for top, left in positions[:200]:
            box = (top, left, top + h - 1, left + w - 1)
            if not _cup_bboxes_do_not_touch(placed_boxes, box, pad=1):
                continue
            trial = scene.copy()
            _paste_patch(trial, patch, top, left)
            candidate = placed_cups + [_translate_cup(cups[idx], top, left)]
            if len(candidate) == len(placed_cups) + 1 and _scene_valid(trial, candidate, final=False):
                scene = trial
                placed_boxes.append(box)
                placed_cups = candidate
                placed = True
                break
        if not placed:
            return None

    if len(placed_cups) != n or not _scene_valid(scene, placed_cups, final=True):
        return None
    palette_row = _make_palette_row(scene_w, n)
    grid = np.vstack([scene, palette_row])

    analyzed = _analyze_cups(scene)
    if len(analyzed) != n:
        return None
    if sorted(cup["area"] for cup in analyzed) != sorted(cup["area"] for cup in placed_cups):
        return None
    expected = _paint_expected(scene, PALETTE[:n], analyzed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, scene):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cup-ranking input")
