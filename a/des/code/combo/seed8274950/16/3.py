import random
from collections import Counter

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_BRANCHES = ("horizontal", "vertical", "diagonal", "checkerboard")
_MAX_GENERATION_ATTEMPTS = 300
_GEN_COUNTER = 0


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in _NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _component_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objs: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            objs.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return objs


def _is_rect_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    target = np.zeros((h, w), dtype=bool)
    target[0, :] = True
    target[-1, :] = True
    target[:, 0] = True
    target[:, -1] = True
    return bool(np.array_equal(mask, target))


def _is_straight_bar(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return (h == 1 or w == 1) and coords.shape[0] == h * w and max(h, w) >= 2


def _find_structure(grid: np.ndarray) -> tuple[int, dict, dict[str, dict]]:
    bg = _mode_color(grid)
    objs = _component_objects(grid, bg)

    canvas_candidates = [obj for obj in objs if _is_rect_ring(obj["coords"])]
    assert len(canvas_candidates) == 1, "expected one canvas ring"
    canvas = canvas_candidates[0]

    bars = [obj for obj in objs if obj is not canvas]
    assert len(bars) == 4, "expected four frame bars"
    assert all(_is_straight_bar(obj["coords"]) for obj in bars), "frame bars must be straight"

    horizontals = [obj for obj in bars if obj["bbox"][0] == obj["bbox"][2]]
    verticals = [obj for obj in bars if obj["bbox"][1] == obj["bbox"][3]]
    assert len(horizontals) == 2 and len(verticals) == 2, "frame must have two horizontal and two vertical sides"

    horizontals.sort(key=lambda obj: int(obj["bbox"][0]))
    verticals.sort(key=lambda obj: int(obj["bbox"][1]))
    top, bottom = horizontals
    left, right = verticals

    fr0 = int(top["bbox"][0])
    fr1 = int(bottom["bbox"][0])
    fc0 = int(left["bbox"][1])
    fc1 = int(right["bbox"][1])

    assert top["bbox"][1] == fc0 + 1 and top["bbox"][3] == fc1 - 1
    assert bottom["bbox"][1] == fc0 + 1 and bottom["bbox"][3] == fc1 - 1
    assert left["bbox"][0] == fr0 + 1 and left["bbox"][2] == fr1 - 1
    assert right["bbox"][0] == fr0 + 1 and right["bbox"][2] == fr1 - 1

    cr0, cc0, cr1, cc1 = canvas["bbox"]
    assert fr0 < cr0 - 1 and fr1 > cr1 + 1
    assert fc0 < cc0 - 1 and fc1 > cc1 + 1

    return bg, canvas, {"top": top, "right": right, "bottom": bottom, "left": left}


def _classify_pattern(top: int, right: int, bottom: int, left: int) -> str:
    colors = {top, right, bottom, left}
    if len(colors) == 4:
        return "checkerboard"
    if len(colors) != 3:
        raise AssertionError("frame colors must encode exactly one pairing")
    if top == bottom and right != left and top not in {right, left}:
        return "horizontal"
    if left == right and top != bottom and left not in {top, bottom}:
        return "vertical"
    if top == right and bottom != left and top not in {bottom, left}:
        return "diag_tr"
    if bottom == left and top != right and bottom not in {top, right}:
        return "diag_bl"
    raise AssertionError("unrecognized frame-color relation")


def _fill_canvas(
    out: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    top: int,
    right: int,
    bottom: int,
    left: int,
) -> None:
    r0, c0, r1, c1 = canvas_bbox
    kind = _classify_pattern(top, right, bottom, left)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= 1 and inner_w >= 1

    for rr in range(inner_h):
        for cc in range(inner_w):
            if kind == "horizontal":
                color = left if rr % 2 == 0 else right
            elif kind == "vertical":
                color = top if cc % 2 == 0 else bottom
            elif kind == "diag_tr":
                color = left if ((rr + cc) // 2) % 2 == 0 else bottom
            elif kind == "diag_bl":
                color = top if ((rr - cc) % 4) < 2 else right
            else:
                tile = ((rr % 2), (cc % 2))
                color = {
                    (0, 0): top,
                    (0, 1): right,
                    (1, 0): left,
                    (1, 1): bottom,
                }[tile]
            out[r0 + 1 + rr, c0 + 1 + cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the outer frame's side-color relations and paint a pattern inside the inner canvas.

    Input structure:
    - Background fills the grid.
    - One monochrome rectangular canvas ring encloses a background-colored interior.
    - Outside it sits a one-cell frame made of four separate side bars.

    Rule:
    - Read the frame side colors in top, right, bottom, left order.
    - Top=bottom -> horizontal stripes using the left/right colors.
    - Left=right -> vertical stripes using the top/bottom colors.
    - Top=right or bottom=left -> diagonal stripes using the two unpaired colors.
    - Four different side colors -> repeat a 2x2 checkerboard tile using all four side colors.
    """
    g = np.asarray(grid, dtype=int)
    _bg, canvas, frame = _find_structure(g)

    top = int(frame["top"]["color"])
    right = int(frame["right"]["color"])
    bottom = int(frame["bottom"]["color"])
    left = int(frame["left"]["color"])

    out = g.copy()
    _fill_canvas(out, canvas["bbox"], top, right, bottom, left)
    return out


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_frame_bars(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    top: int,
    right: int,
    bottom: int,
    left: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 + 1 : c1] = top
    grid[r1, c0 + 1 : c1] = bottom
    grid[r0 + 1 : r1, c0] = left
    grid[r0 + 1 : r1, c1] = right


def _sample_colors(rng: random.Random, bg: int, count: int) -> list[int]:
    pool = [c for c in range(10) if c != bg]
    return rng.sample(pool, count)


def _gen_attempt(rng: random.Random, branch: str, diag_variant: int) -> np.ndarray | None:
    bg = rng.randrange(10)

    inner_h = rng.randint(4, 10)
    inner_w = rng.randint(4, 10)
    gap_y = rng.randint(1, 3)
    gap_x = rng.randint(1, 3)

    canvas_h = inner_h + 2
    canvas_w = inner_w + 2
    frame_h = canvas_h + 2 * (gap_y + 1)
    frame_w = canvas_w + 2 * (gap_x + 1)

    margin_top = rng.randint(1, 4)
    margin_bottom = rng.randint(1, 4)
    margin_left = rng.randint(1, 4)
    margin_right = rng.randint(1, 4)

    h = frame_h + margin_top + margin_bottom
    w = frame_w + margin_left + margin_right
    grid = np.full((h, w), bg, dtype=int)

    frame_r0 = margin_top
    frame_c0 = margin_left
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    canvas_r0 = frame_r0 + gap_y + 1
    canvas_c0 = frame_c0 + gap_x + 1
    canvas_r1 = canvas_r0 + canvas_h - 1
    canvas_c1 = canvas_c0 + canvas_w - 1

    if branch == "horizontal":
        canvas_color, pair, left, right = _sample_colors(rng, bg, 4)
        top = bottom = pair
    elif branch == "vertical":
        canvas_color, pair, top, bottom = _sample_colors(rng, bg, 4)
        left = right = pair
    elif branch == "diagonal":
        if diag_variant == 0:
            canvas_color, pair, bottom, left = _sample_colors(rng, bg, 4)
            top = right = pair
        else:
            canvas_color, pair, top, right = _sample_colors(rng, bg, 4)
            bottom = left = pair
    else:
        canvas_color, top, right, bottom, left = _sample_colors(rng, bg, 5)

    _draw_ring(grid, (canvas_r0, canvas_c0, canvas_r1, canvas_c1), canvas_color)
    _draw_frame_bars(
        grid,
        (frame_r0, frame_c0, frame_r1, frame_c1),
        top,
        right,
        bottom,
        left,
    )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = random.Random()
    branch = _BRANCHES[_GEN_COUNTER % len(_BRANCHES)]
    diag_variant = (_GEN_COUNTER // len(_BRANCHES)) % 2
    _GEN_COUNTER += 1

    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng, branch, diag_variant)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
