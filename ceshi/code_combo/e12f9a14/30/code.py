# Combo variant of e12f9a14 (index 30)
# combo_spec: D17/1+D20/2
# Name: 距离编码碰撞类型+框架大小决定碰撞

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior: the original task routed collisions by incoming angle.
#    Here every collision is resolved in two stages:
#    1. Compare the two source frames' Manhattan center distance.
#       Near pairs use rule A, far pairs use rule B.
#    2. Compare the two frame sizes.
#       The larger frame has priority; only its color survives the collision.
#    Example: a 7x7 left frame and a 5x5 top frame fire into the same cell.
#    If their center distance is small, the 7x7 beam keeps moving straight east.
#    If the same geometry is moved farther apart, the 7x7 beam bends to the
#    southeast diagonal instead. The smaller beam stops before the collision.
# b) Complexity calibration: only two size classes (5x5 and 7x7 outer frames)
#    and only two distance classes (near/far) are used. That keeps the rule
#    visible while still requiring the solver to combine two global properties.
# c) Original hidden assumptions that no longer hold:
#    - inner blocks are no longer always 2x2, so the old 2x2 detector is invalid;
#    - collision type no longer depends on angle, so the old 45/90 branches are invalid;
#    - beam winners are no longer symmetric, so both colors cannot continue after impact.
#    The new solver therefore detects variable-size framed squares and attaches
#    source metadata (center, size, direction) to each beam.
# d) Placement legality: the generator constructs exactly one colliding pair,
#    derives both frame positions from the intended collision cell, and validates
#    all bboxes stay in-bounds with at least three cells of post-collision runway.
# e) Visual observability: frame size is directly visible in the input, and
#    near/far is visible as tighter vs wider spacing. The output reinforces this:
#    near collisions continue straight; far collisions bend into a diagonal.
# f) Visual cleanliness: using one isolated colliding pair avoids layered
#    crossings and makes the surviving color unambiguous.
# g) Learnability: no hidden ordering by numeric color id is used. Priority comes
#    only from visible frame size, and distance classes are repeated across samples
#    by randomized near/far generation.


NEAR_DISTANCE_THRESHOLD = 12
OUTER_SIZES = (5, 7)
PAIR_GAP_STEPS = {"near": (2, 3), "far": (7, 8)}
GRID_SIDE_RANGE = (21, 30)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
GEN_RETRIES = 200


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    return (0 if dr == 0 else int(np.sign(dr)), 0 if dc == 0 else int(np.sign(dc)))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _side_gap_local(size: int, direction: tuple[int, int]) -> tuple[int, int]:
    h = size // 2
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return (h, size - 1)
    if (dr, dc) == (0, -1):
        return (h, 0)
    if (dr, dc) == (1, 0):
        return (size - 1, h)
    if (dr, dc) == (-1, 0):
        return (0, h)
    raise ValueError(f"unsupported gap direction: {direction}")


def _draw_source(
    grid: np.ndarray,
    center: tuple[int, int],
    outer_size: int,
    bg: int,
    frame_color: int,
    block_color: int,
    direction: tuple[int, int],
) -> None:
    h = outer_size // 2
    r0 = center[0] - h
    c0 = center[1] - h
    patch = np.full((outer_size, outer_size), frame_color, dtype=int)
    patch[1:-1, 1:-1] = block_color
    gr, gc = _side_gap_local(outer_size, direction)
    patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _infer_frame_color(g: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in g.ravel() if int(v) != bg]
    assert vals, "expected non-background objects"
    counts = np.bincount(vals, minlength=10)
    return int(np.argmax(counts))


def _find_sources(g: np.ndarray, bg: int) -> list[dict]:
    frame_color = _infer_frame_color(g, bg)
    inner_mask = (g != bg) & (g != frame_color)
    comps = _component_cells(inner_mask)
    assert comps, "expected at least one framed block"

    sources: list[dict] = []
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        inner_size = r1 - r0 + 1
        assert inner_size == c1 - c0 + 1, "expected square inner blocks"
        assert inner_size in (3, 5), "expected only the two visible size classes"
        color = int(g[r0, c0])
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                assert int(g[rr, cc]) == color, "expected solid monochrome inner block"

        fr0, fc0 = r0 - 1, c0 - 1
        fr1, fc1 = r1 + 1, c1 + 1
        assert fr0 >= 0 and fc0 >= 0 and fr1 < g.shape[0] and fc1 < g.shape[1], "frame must fit"
        outer_size = fr1 - fr0 + 1
        assert outer_size in OUTER_SIZES, "expected supported outer frame size"
        assert outer_size % 2 == 1, "expected odd-sized frame"

        gaps: list[tuple[int, int]] = []
        for rr in range(fr0, fr1 + 1):
            for cc in range(fc0, fc1 + 1):
                if rr not in (fr0, fr1) and cc not in (fc0, fc1):
                    continue
                v = int(g[rr, cc])
                if v == bg:
                    gaps.append((rr, cc))
                else:
                    assert v == frame_color, "expected uniform frame color"
        assert len(gaps) == 1, "expected exactly one gap per frame"
        gap = gaps[0]

        center = (fr0 + outer_size // 2, fc0 + outer_size // 2)
        direction = _normalize_dir(gap[0] - center[0], gap[1] - center[1])
        assert direction in ((0, 1), (0, -1), (1, 0), (-1, 0)), "generator uses centered side gaps"
        sources.append(
            {
                "color": color,
                "frame_color": frame_color,
                "center": center,
                "outer_size": outer_size,
                "gap": gap,
                "dir": direction,
            }
        )

    sources.sort(key=lambda s: (s["center"][0], s["center"][1], s["color"]))
    return sources


class _Beam:
    __slots__ = ("bid", "color", "source", "pos", "direction", "active")

    def __init__(self, bid: int, source: dict):
        self.bid = bid
        self.color = int(source["color"])
        self.source = source
        self.pos = source["gap"]
        self.direction = source["dir"]
        self.active = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays from variable-size framed squares; collisions now depend on frame spacing and frame size.

    Input structure:
    - The background is the dominant color. One shared frame color surrounds one or more monochrome square blocks.
    - Inner blocks are solid 3x3 or 5x5 squares, so the visible outer frames are 5x5 or 7x7.
    - Each frame has exactly one centered side gap, which launches one cardinal ray in the outward direction.

    Rule:
    - Every gap cell becomes the first beam pixel and the ray extends through background cells until edge or collision.
    - When two rays target the same cell, compare their source frame centers.
      Near pairs (small Manhattan distance) use rule A: the larger frame's ray survives and keeps its current direction.
      Far pairs (large Manhattan distance) use rule B: the larger frame's ray survives but turns to the normalized sum of both incoming directions.
    - The smaller frame's ray always stops before the collision. The collision cell is painted by the surviving ray.

    Key insight:
    - Collision geometry alone is no longer enough. The solver must read two global cues from the input:
      frame spacing chooses between straight-vs-bent resolution, and frame size chooses which color wins.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    sources = _find_sources(g, bg)
    beams = {_id: _Beam(_id, src) for _id, src in enumerate(sources)}

    painted_owner: dict[tuple[int, int], int] = {}

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < h and 0 <= cell[1] < w

    def paint(bid: int, cell: tuple[int, int]) -> None:
        out[cell[0], cell[1]] = beams[bid].color
        painted_owner[cell] = bid

    for bid, beam in beams.items():
        assert int(g[beam.pos[0], beam.pos[1]]) == bg, "expected seed on a frame gap"
        assert beam.pos not in painted_owner, "expected unique gap locations"
        paint(bid, beam.pos)

    max_iters = h * w * 8
    for _ in range(max_iters):
        proposals: dict[tuple[int, int], list[int]] = {}
        occupied_hits: list[int] = []
        active = [bid for bid, beam in beams.items() if beam.active]
        if not active:
            break

        for bid in active:
            beam = beams[bid]
            nr = beam.pos[0] + beam.direction[0]
            nc = beam.pos[1] + beam.direction[1]
            nxt = (nr, nc)
            if not in_bounds(nxt) or int(g[nr, nc]) != bg:
                beam.active = False
                continue
            if nxt in painted_owner:
                occupied_hits.append(bid)
                continue
            proposals.setdefault(nxt, []).append(bid)

        if not proposals and not occupied_hits:
            break

        for bid in occupied_hits:
            beams[bid].active = False

        collision_bids: set[int] = set()
        for cell, bidders in proposals.items():
            if len(bidders) < 2:
                continue
            assert len(bidders) == 2, "generator should only create pairwise collisions"
            a, b = sorted(bidders)
            ba = beams[a]
            bb = beams[b]
            sa = ba.source
            sb = bb.source
            dist = _manhattan(sa["center"], sb["center"])
            near = dist <= NEAR_DISTANCE_THRESHOLD

            if sa["outer_size"] > sb["outer_size"]:
                winner, loser = a, b
            elif sb["outer_size"] > sa["outer_size"]:
                winner, loser = b, a
            else:
                winner, loser = (a, b) if sa["center"] <= sb["center"] else (b, a)

            wbeam = beams[winner]
            lbeam = beams[loser]
            win_dir = wbeam.direction
            lose_dir = lbeam.direction

            wbeam.pos = cell
            paint(winner, cell)
            if near:
                wbeam.direction = win_dir
            else:
                turned = _normalize_dir(win_dir[0] + lose_dir[0], win_dir[1] + lose_dir[1])
                assert turned != (0, 0), "expected non-opposite beams"
                wbeam.direction = turned
            wbeam.active = True
            lbeam.active = False

            collision_bids.add(winner)
            collision_bids.add(loser)

        for cell, bidders in proposals.items():
            if len(bidders) != 1:
                continue
            bid = bidders[0]
            if bid in collision_bids or not beams[bid].active:
                continue
            beams[bid].pos = cell
            paint(bid, cell)
    else:
        raise AssertionError("beam simulation did not converge")

    return out


def _collision_layout(
    n: int,
    orientation: str,
    left_or_top_size: int,
    other_size: int,
    steps: int,
    rng: np.random.Generator,
) -> list[dict] | None:
    h1 = left_or_top_size // 2
    h2 = other_size // 2

    for _ in range(60):
        cr = int(rng.integers(4, n - 4))
        cc = int(rng.integers(4, n - 4))

        if orientation == "LT":
            specs = [
                {"center": (cr, cc - (h1 + steps)), "size": left_or_top_size, "dir": (0, 1)},
                {"center": (cr - (h2 + steps), cc), "size": other_size, "dir": (1, 0)},
            ]
            runway_dirs = [(0, 1), (1, 0), (1, 1)]
        elif orientation == "RT":
            specs = [
                {"center": (cr, cc + (h1 + steps)), "size": left_or_top_size, "dir": (0, -1)},
                {"center": (cr - (h2 + steps), cc), "size": other_size, "dir": (1, 0)},
            ]
            runway_dirs = [(0, -1), (1, 0), (1, -1)]
        elif orientation == "LB":
            specs = [
                {"center": (cr, cc - (h1 + steps)), "size": left_or_top_size, "dir": (0, 1)},
                {"center": (cr + (h2 + steps), cc), "size": other_size, "dir": (-1, 0)},
            ]
            runway_dirs = [(0, 1), (-1, 0), (-1, 1)]
        elif orientation == "RB":
            specs = [
                {"center": (cr, cc + (h1 + steps)), "size": left_or_top_size, "dir": (0, -1)},
                {"center": (cr + (h2 + steps), cc), "size": other_size, "dir": (-1, 0)},
            ]
            runway_dirs = [(0, -1), (-1, 0), (-1, -1)]
        else:
            raise ValueError(f"unknown orientation: {orientation}")

        ok = True
        for spec in specs:
            hs = spec["size"] // 2
            r0 = spec["center"][0] - hs
            c0 = spec["center"][1] - hs
            r1 = spec["center"][0] + hs
            c1 = spec["center"][1] + hs
            if r0 < 0 or c0 < 0 or r1 >= n or c1 >= n:
                ok = False
                break
        if not ok:
            continue

        min_runway = min(
            sum(
                1
                for k in range(1, n)
                if 0 <= cr + d[0] * k < n and 0 <= cc + d[1] * k < n
            )
            for d in runway_dirs
        )
        if min_runway < 3:
            continue
        return specs

    return None


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=2, replace=False))
    return bg, frame_color, [int(c) for c in block_colors]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIDE_RANGE))
    bg, frame_color, block_colors = _pick_colors(rng)

    dist_kind = str(rng.choice(["near", "far"]))
    steps = int(rng.choice(PAIR_GAP_STEPS[dist_kind]))
    orientation = str(rng.choice(["LT", "RT", "LB", "RB"]))

    sizes = [5, 7]
    if int(rng.integers(2)) == 1:
        sizes.reverse()

    specs = _collision_layout(n, orientation, sizes[0], sizes[1], steps, rng)
    if specs is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for spec, color in zip(specs, block_colors):
        _draw_source(grid, spec["center"], spec["size"], bg, frame_color, int(color), spec["dir"])

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < 6:
        return None

    sources = _find_sources(grid, bg)
    assert len(sources) == 2, "generator should create one colliding pair"
    dist = _manhattan(sources[0]["center"], sources[1]["center"])
    if dist_kind == "near" and dist > NEAR_DISTANCE_THRESHOLD:
        return None
    if dist_kind == "far" and dist <= NEAR_DISTANCE_THRESHOLD:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused e12f9a14 input")
