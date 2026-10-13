import random
from collections import Counter, deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
ACTIVE_SIDE_CHOICES = (
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
    ("top", "left", "right"),
    ("bottom", "left", "right"),
    ("top", "bottom", "left", "right"),
)
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_COLOR_FALLBACK = 5
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _normalize(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple((int(r), int(c)) for r, c in arr)


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _rotate_norm(coords: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    mask = _coords_to_mask(coords)
    rot = np.rot90(mask, int(k))
    return _normalize(np.argwhere(rot))


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        out.append((start, prev))
        start = prev = idx
    out.append((start, prev))
    return out


def _templates_for_side(side: str, width: int) -> dict[str, tuple[tuple[int, int], ...]]:
    if side == "top":
        base = [(1, c) for c in range(width)]
        return {
            "start": _normalize(base + [(0, 0)]),
            "end": _normalize(base + [(0, width - 1)]),
        }
    if side == "bottom":
        base = [(0, c) for c in range(width)]
        return {
            "start": _normalize(base + [(1, 0)]),
            "end": _normalize(base + [(1, width - 1)]),
        }
    if side == "left":
        base = [(r, 1) for r in range(width)]
        return {
            "start": _normalize(base + [(0, 0)]),
            "end": _normalize(base + [(width - 1, 0)]),
        }
    base = [(r, 0) for r in range(width)]
    return {
        "start": _normalize(base + [(0, 1)]),
        "end": _normalize(base + [(width - 1, 1)]),
    }


def _find_rotation_and_variant(
    piece_norm: tuple[tuple[int, int], ...], side: str, width: int
) -> tuple[int, str]:
    templates = _templates_for_side(side, width)
    for rot in range(4):
        rotated = _rotate_norm(piece_norm, rot)
        for variant, templ in templates.items():
            if rotated == templ:
                return int(rot), variant
    raise AssertionError(f"piece does not fit side={side} width={width}")


def _gap_projection(gap: dict) -> tuple[int, int]:
    return int(gap["span"][0]), int(gap["span"][1])


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _find_gaps(grid: np.ndarray, frame_color: int, bbox_: tuple[int, int, int, int]) -> list[dict]:
    top, left, bottom, right = bbox_
    gaps: list[dict] = []

    top_runs = _runs([c for c in range(left + 1, right) if int(grid[top, c]) != frame_color])
    bottom_runs = _runs([c for c in range(left + 1, right) if int(grid[bottom, c]) != frame_color])
    left_runs = _runs([r for r in range(top + 1, bottom) if int(grid[r, left]) != frame_color])
    right_runs = _runs([r for r in range(top + 1, bottom) if int(grid[r, right]) != frame_color])

    all_runs = {
        "top": top_runs,
        "bottom": bottom_runs,
        "left": left_runs,
        "right": right_runs,
    }

    for side, runs in all_runs.items():
        assert len(runs) <= 1, f"expected at most one {side} gap"
        for start, end in runs:
            width = int(end - start + 1)
            assert width in (2, 3), f"unexpected {side} gap width {width}"
            anchor = (top, start) if side == "top" else None
            if side == "bottom":
                anchor = (bottom, start)
            elif side == "left":
                anchor = (start, left)
            elif side == "right":
                anchor = (start, right)
            gaps.append(
                {
                    "side": side,
                    "span": (int(start), int(end)),
                    "width": width,
                    "anchor": anchor,
                }
            )

    assert 3 <= len(gaps) <= 4, "expected three or four frame gaps"
    return gaps


def _find_pieces(
    grid: np.ndarray, bg: int, frame_color: int, bbox_: tuple[int, int, int, int]
) -> list[dict]:
    top, left, bottom, right = bbox_
    mask = (grid != bg) & (grid != frame_color)
    comps = _components(mask, connectivity=4)
    pieces: list[dict] = []
    for idx, coords in enumerate(comps):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "pieces must be monochrome"
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(coords)
        if r1 < top:
            side = "top"
        elif r0 > bottom:
            side = "bottom"
        elif c1 < left:
            side = "left"
        elif c0 > right:
            side = "right"
        else:
            raise AssertionError("piece must lie outside the frame on a single side")
        norm = _normalize([(int(r - r0), int(c - c0)) for r, c in coords])
        pieces.append(
            {
                "id": int(idx),
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "side": side,
                "norm": norm,
            }
        )
    return pieces


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert colors, "grid must contain foreground colors"
    frame_color = max(colors, key=lambda color: int((g == color).sum()))
    if frame_color == bg:
        frame_color = FRAME_COLOR_FALLBACK

    frame_coords = np.argwhere(g == frame_color)
    assert frame_coords.size > 0, "frame color missing"
    bbox_ = _bbox(frame_coords)
    gaps = _find_gaps(g, frame_color, bbox_)
    pieces = _find_pieces(g, bg, frame_color, bbox_)
    assert len(pieces) == len(gaps), "piece count must match gap count"

    matches = []
    used_piece_ids: set[int] = set()
    for gap in gaps:
        span0, span1 = _gap_projection(gap)
        candidates = []
        for piece in pieces:
            if piece["side"] != gap["side"] or piece["id"] in used_piece_ids:
                continue
            r0, c0, r1, c1 = piece["bbox"]
            if gap["side"] in ("top", "bottom"):
                if _overlaps(c0, c1, span0, span1):
                    candidates.append(piece)
            else:
                if _overlaps(r0, r1, span0, span1):
                    candidates.append(piece)
        assert len(candidates) == 1, f"expected one piece for {gap['side']} gap"
        piece = candidates[0]
        rot, variant = _find_rotation_and_variant(piece["norm"], gap["side"], gap["width"])
        matches.append(
            {
                "gap": gap,
                "piece": piece,
                "rotation": rot,
                "variant": variant,
                "sort_key": tuple(int(v) for v in gap["anchor"]),
            }
        )
        used_piece_ids.add(int(piece["id"]))

    assert len(used_piece_ids) == len(pieces), "all pieces must be used"
    matches.sort(key=lambda item: item["sort_key"])
    return {
        "grid": g,
        "bg": bg,
        "frame_color": frame_color,
        "bbox": bbox_,
        "matches": matches,
    }


def _placed_piece_cells(
    bbox_: tuple[int, int, int, int], gap: dict, variant: str
) -> list[tuple[int, int]]:
    top, left, bottom, right = bbox_
    span0, span1 = _gap_projection(gap)
    side = gap["side"]
    if side == "top":
        extra = (top - 1, span0 if variant == "start" else span1)
        run = [(top, c) for c in range(span0, span1 + 1)]
        return [extra] + run
    if side == "bottom":
        extra = (bottom + 1, span0 if variant == "start" else span1)
        run = [(bottom, c) for c in range(span0, span1 + 1)]
        return [extra] + run
    if side == "left":
        extra = (span0 if variant == "start" else span1, left - 1)
        run = [(r, left) for r in range(span0, span1 + 1)]
        return [extra] + run
    extra = (span0 if variant == "start" else span1, right + 1)
    run = [(r, right) for r in range(span0, span1 + 1)]
    return [extra] + run


def _ray_strips(bbox_: tuple[int, int, int, int], gap: dict) -> list[list[tuple[int, int]]]:
    top, left, bottom, right = bbox_
    span0, span1 = _gap_projection(gap)
    side = gap["side"]
    strips: list[list[tuple[int, int]]] = []
    if side == "top":
        for r in range(top + 1, bottom):
            strips.append([(r, c) for c in range(span0, span1 + 1)])
    elif side == "bottom":
        for r in range(bottom - 1, top, -1):
            strips.append([(r, c) for c in range(span0, span1 + 1)])
    elif side == "left":
        for c in range(left + 1, right):
            strips.append([(r, c) for r in range(span0, span1 + 1)])
    else:
        for c in range(right - 1, left, -1):
            strips.append([(r, c) for r in range(span0, span1 + 1)])
    return strips


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered L-fragments back into a grey frame's side gaps, then project their colors inward.

    Input structure:
    - Background is the dominant color, with one large hollow rectangular frame drawn in the most common non-background color.
    - Three or four sides of the frame contain one contiguous 2- or 3-cell border gap each.
    - Outside the frame sit equally many monochrome L-shaped fragments of size 3 or 4, one fragment per active side.
    - A fragment's row/column projection overlaps the gap on its side, but the fragment may need a 0/90/180/270 rotation before it fits.

    Rule:
    - Detect the frame bbox and read each side gap from the missing border run.
    - Match each outside fragment to the same-side gap whose projection it overlaps, then find the rotation whose long arm spans the gap while the short arm points outward.
    - Remove the scattered fragments from their original locations, insert them into the matching gaps, and keep the rest of the frame unchanged.
    - Process repaired gaps from top to bottom, then left to right: from each repaired border run, cast an inward stripe of the fragment color with the same width as the gap; if a stripe reaches an earlier stripe, overwrite that contact strip and stop.

    Key insight:
    - The fragment's side projection tells you which gap it belongs to, while its chirality under rotation decides which end of the gap gets the outward short arm; the repaired gap then becomes a color ray source for the frame interior.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = int(info["bg"])
    frame_color = int(info["frame_color"])
    bbox_ = info["bbox"]

    out = np.full_like(g, bg)
    out[g == frame_color] = frame_color

    for match in info["matches"]:
        color = int(match["piece"]["color"])
        for r, c in _placed_piece_cells(bbox_, match["gap"], match["variant"]):
            out[r, c] = color

    for match in info["matches"]:
        color = int(match["piece"]["color"])
        for strip in _ray_strips(bbox_, match["gap"]):
            hit_existing = any(int(out[r, c]) != bg for r, c in strip)
            for r, c in strip:
                out[r, c] = color
            if hit_existing:
                break

    return out


def _make_gap_spec(
    rng: np.random.Generator, side: str, bbox_: tuple[int, int, int, int]
) -> dict:
    top, left, bottom, right = bbox_
    width = int(rng.integers(2, 4))
    variant = str(rng.choice(["start", "end"]))
    if side in ("top", "bottom"):
        start_min = left + 1
        start_max = right - width
        start = int(rng.integers(start_min, start_max + 1))
    else:
        start_min = top + 1
        start_max = bottom - width
        start = int(rng.integers(start_min, start_max + 1))
    return {
        "side": side,
        "width": width,
        "start": start,
        "end": int(start + width - 1),
        "variant": variant,
    }


def _carve_gap(grid: np.ndarray, gap: dict) -> None:
    side = gap["side"]
    start = int(gap["start"])
    end = int(gap["end"])
    if side == "top":
        row = int(gap["row"])
        grid[row, start : end + 1] = 0
    elif side == "bottom":
        row = int(gap["row"])
        grid[row, start : end + 1] = 0
    elif side == "left":
        col = int(gap["col"])
        grid[start : end + 1, col] = 0
    else:
        col = int(gap["col"])
        grid[start : end + 1, col] = 0


def _canonical_shape(side: str, width: int, variant: str) -> tuple[tuple[int, int], ...]:
    return _templates_for_side(side, width)[variant]


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    side: str,
    gap: dict,
    bbox_: tuple[int, int, int, int],
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    top, left, bottom, right = bbox_
    ph = max(r for r, _ in shape) + 1
    pw = max(c for _, c in shape) + 1

    if side == "top":
        row_lo, row_hi = 0, top - ph - 1
        col_lo = max(0, int(gap["start"]) - pw + 1)
        col_hi = min(w - pw, int(gap["end"]))
    elif side == "bottom":
        row_lo, row_hi = bottom + 2, h - ph
        col_lo = max(0, int(gap["start"]) - pw + 1)
        col_hi = min(w - pw, int(gap["end"]))
    elif side == "left":
        col_lo, col_hi = 0, left - pw - 1
        row_lo = max(0, int(gap["start"]) - ph + 1)
        row_hi = min(h - ph, int(gap["end"]))
    else:
        col_lo, col_hi = right + 2, w - pw
        row_lo = max(0, int(gap["start"]) - ph + 1)
        row_hi = min(h - ph, int(gap["end"]))

    if row_lo > row_hi or col_lo > col_hi:
        return False

    for _ in range(40):
        base_r = int(rng.integers(row_lo, row_hi + 1))
        base_c = int(rng.integers(col_lo, col_hi + 1))
        pts = [(base_r + r, base_c + c) for r, c in shape]
        if any(occupied[r, c] for r, c in pts):
            continue
        for r, c in pts:
            occupied[r, c] = True
            grid[r, c] = int(color)
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    max_frame_h = min(18, h - 8)
    max_frame_w = min(18, w - 8)
    if max_frame_h < 8 or max_frame_w < 8:
        return None

    frame_h = int(rng.integers(8, max_frame_h + 1))
    frame_w = int(rng.integers(8, max_frame_w + 1))
    top = int(rng.integers(4, h - frame_h - 3))
    left = int(rng.integers(4, w - frame_w - 3))
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    bbox_ = (top, left, bottom, right)

    sides = list(ACTIVE_SIDE_CHOICES[int(rng.integers(0, len(ACTIVE_SIDE_CHOICES)))])
    colors = [int(c) for c in rng.choice(PIECE_COLORS, size=len(sides), replace=False)]

    grid = np.zeros((h, w), dtype=int)
    grid[top, left : right + 1] = FRAME_COLOR_FALLBACK
    grid[bottom, left : right + 1] = FRAME_COLOR_FALLBACK
    grid[top : bottom + 1, left] = FRAME_COLOR_FALLBACK
    grid[top : bottom + 1, right] = FRAME_COLOR_FALLBACK

    gap_specs: list[dict] = []
    for side, color in zip(sides, colors):
        gap = _make_gap_spec(rng, side, bbox_)
        if side == "top":
            gap["row"] = top
        elif side == "bottom":
            gap["row"] = bottom
        elif side == "left":
            gap["col"] = left
        else:
            gap["col"] = right
        gap["color"] = int(color)
        _carve_gap(grid, gap)
        gap_specs.append(gap)

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[top : bottom + 1, left : right + 1] = True
    occupied[top - 1 : bottom + 2, left - 1 : right + 2] |= False
    for gap in gap_specs:
        side = str(gap["side"])
        canonical = _canonical_shape(side, int(gap["width"]), str(gap["variant"]))
        rot = int(rng.integers(0, 4))
        shape = _rotate_norm(canonical, rot)
        ok = _place_shape(
            grid,
            occupied,
            shape,
            side,
            gap,
            bbox_,
            int(gap["color"]),
            rng,
        )
        if not ok:
            return None

    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return None

    if len(info["matches"]) != len(gap_specs):
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if int((out != 0).sum()) <= int((grid != 0).sum()):
        return None
    if len(np.unique(grid)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the frame-ray puzzle")
