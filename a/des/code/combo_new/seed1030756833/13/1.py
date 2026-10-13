import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            seen[sr, sc] = True
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return -1, 0
    if r == r1 and c0 < c < c1:
        return 1, 0
    if c == c0 and r0 < r < r1:
        return 0, -1
    if c == c1 and r0 < r < r1:
        return 0, 1
    raise AssertionError("gap must lie on a non-corner side cell")


def _find_frames(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    frames: list[dict] = []

    for pts in _components(g == GREY):
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert 0 <= r0 < r1 < h and 0 <= c0 < c1 < w, "grey component must span a rectangle"
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rectangle interior must be nonempty"

        border = set(_rect_border_cells(r0, c0, r1, c1))
        comp = {(int(r), int(c)) for r, c in pts}
        assert comp <= border, "grey frame pixels must lie exactly on a rectangle border"
        missing = sorted(border - comp)
        assert len(missing) == 1, "each grey rectangle must have exactly one border gap"

        gap = missing[0]
        assert int(g[gap]) == BG, "the missing border cell must be background"
        direction = _gap_direction(gap, (r0, c0, r1, c1))

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_local = np.argwhere((inner != BG) & (inner != GREY))
        assert marker_local.shape[0] == 1, "each rectangle must contain exactly one colored marker"
        mr, mc = map(int, marker_local[0])
        marker = int(inner[mr, mc])

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "dir": direction,
                "color": marker,
                "marker": (r0 + 1 + mr, c0 + 1 + mc),
            }
        )

    assert frames, "expected at least one grey rectangle frame"
    return sorted(frames, key=lambda item: item["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend colored rays from the one-cell gaps of grey rectangular frames.

    Input structure:
    - The background is black.
    - Grey pixels form several one-pixel-thick rectangular borders, each missing
      exactly one non-corner border cell as a gap.
    - Each rectangle interior contains exactly one non-background, non-grey
      single-pixel marker; different rectangles may use different marker colors.
    - Rectangles do not overlap, though they may share rows or columns.

    Rule:
    - Keep every input pixel unchanged.
    - For each grey rectangle, locate the missing border cell and use its side
      to choose the outward cardinal direction.
    - Read the color of the single marker pixel inside that rectangle.
    - Paint a ray of that marker color starting at the gap and continuing
      outward through black cells until the next cell would hit another object,
      a previously painted ray, or the grid edge.

    Key insight:
    The gap side supplies the ray direction, while the enclosed marker supplies
    the ray color; both independent clues are needed to determine each ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    for frame in _find_frames(g):
        r, c = frame["gap"]
        dr, dc = frame["dir"]
        color = int(frame["color"])
        while 0 <= r < h and 0 <= c < w:
            if int(out[r, c]) != BG:
                break
            out[r, c] = color
            r += dr
            c += dc
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _gap_for_side(rng: random.Random, bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == (-1, 0):
        return r0, rng.randint(c0 + 1, c1 - 1)
    if direction == (1, 0):
        return r1, rng.randint(c0 + 1, c1 - 1)
    if direction == (0, -1):
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _draw_frame(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    marker: tuple[int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    for r, c in _rect_border_cells(r0, c0, r1, c1):
        if (r, c) != gap:
            g[r, c] = GREY
    g[marker] = color


def _changed_inside_any_frame(g: np.ndarray, out: np.ndarray, frames: list[dict]) -> bool:
    changed = out != g
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if bool(changed[r0 + 1 : r1, c0 + 1 : c1].any()):
            return True
    return False


def _valid_generated_grid(g: np.ndarray, expected_count: int) -> bool:
    try:
        frames = _find_frames(g)
        out = main(g)
    except AssertionError:
        return False

    if len(frames) != expected_count:
        return False
    if out.shape != g.shape or max(out.shape) > 30:
        return False
    if not bool((out != g).any()):
        return False
    if not np.array_equal(out[g != BG], g[g != BG]):
        return False
    if _changed_inside_any_frame(g, out, frames):
        return False

    h, w = g.shape
    for frame in frames:
        r, c = frame["gap"]
        dr, dc = frame["dir"]
        color = int(frame["color"])
        if int(out[r, c]) != color:
            return False

        length = 0
        while 0 <= r < h and 0 <= c < w and int(g[r, c]) == BG and int(out[r, c]) == color:
            length += 1
            r += dr
            c += dc
        if length < 2:
            return False

    return True


def _direction_sequence(rng: random.Random, count: int) -> list[tuple[int, int]]:
    dirs = list(DIRECTIONS)
    rng.shuffle(dirs)
    result = [dirs[i % len(dirs)] for i in range(count)]
    rng.shuffle(result)
    return result


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(14, 30)
    width = rng.randint(14, 30)
    count = rng.randint(2, 5)
    grid = np.full((height, width), BG, dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    for direction in _direction_sequence(rng, count):
        placed = False
        for _ in range(120):
            rect_h = rng.randint(4, min(8, height - 4))
            rect_w = rng.randint(4, min(8, width - 4))
            r0 = rng.randint(2, height - rect_h - 2)
            c0 = rng.randint(2, width - rect_w - 2)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if any(not _boxes_separated(bbox, other, pad=1) for other in boxes):
                continue

            gap = _gap_for_side(rng, bbox, direction)
            mr = rng.randint(bbox[0] + 1, bbox[2] - 1)
            mc = rng.randint(bbox[1] + 1, bbox[3] - 1)
            color = rng.choice(MARKER_COLORS)
            _draw_frame(grid, bbox, gap, (mr, mc), color)
            boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    if not _valid_generated_grid(grid, len(boxes)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/13/1 input")
