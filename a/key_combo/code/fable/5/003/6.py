from __future__ import annotations

from collections import Counter, deque
import random
import sys

import numpy as np


_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)

try:
    from arc_utils import get_bg as _arc_get_bg
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader robust.
    _arc_get_bg = None


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]
Patch = dict[str, object]

ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 700
PLACE_ATTEMPTS = 260
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _colors(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    vals = set(map(int, grid.reshape(-1)))
    if exclude:
        vals -= set(map(int, exclude))
    return sorted(vals)


def _bbox(coords: list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(b: BBox) -> int:
    return int(b[2] - b[0] + 1)


def _bbox_w(b: BBox) -> int:
    return int(b[3] - b[1] + 1)


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _inside(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> bool:
    H, W = map(int, shape)
    return 0 <= int(r) and int(r) + int(h) <= H and 0 <= int(c) and int(c) + int(w) <= W


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[Coord]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = (
        ((1, 0), (-1, 0), (0, 1), (0, -1))
        if int(connectivity) == 4
        else tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    )
    comps: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[Coord] = []
        while q:
            r, c = q.popleft()
            comp.append((int(r), int(c)))
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _mask_from_component(coords: list[Coord], b: BBox) -> np.ndarray:
    r0, c0, r1, c1 = b
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[int(r) - r0, int(c) - c0] = True
    return out


def _paint_mask(out: np.ndarray, anchor: Coord, mask: np.ndarray, color: int) -> None:
    r0, c0 = map(int, anchor)
    for dr, dc in np.argwhere(mask):
        out[r0 + int(dr), c0 + int(dc)] = int(color)


def _bbox_cells(b: BBox) -> set[Coord]:
    r0, c0, r1, c1 = b
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _dilate_cells(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    H, W = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out.add((nr, nc))
    return out


def _find_rectangles(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for color in _colors(g, exclude={bg}):
        comps = _components(g == int(color), connectivity=4)
        if len(comps) != 1:
            continue

        coords = comps[0]
        b = _bbox(coords)
        h, w = _bbox_h(b), _bbox_w(b)
        area = len(coords)
        if h < 4 or w < 4 or area < 16:
            continue

        r0, c0, r1, c1 = b
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(window == int(bg)):
            continue
        density = area / float(h * w)
        if density < 0.58:
            continue
        if not (
            np.any(window[0, :] == int(color))
            and np.any(window[-1, :] == int(color))
            and np.any(window[:, 0] == int(color))
            and np.any(window[:, -1] == int(color))
        ):
            continue

        rects.append({"color": int(color), "bbox": b, "area": int(area)})

    assert rects, "expected at least one repaired rectangle"
    return rects


def _sprite_patches(g: np.ndarray, color: int) -> list[Patch]:
    patches: list[Patch] = []
    for coords in _components(g == int(color), connectivity=8):
        b = _bbox(coords)
        h, w = _bbox_h(b), _bbox_w(b)
        if h < 3 or w < 3:
            continue
        patches.append(
            {
                "bbox": b,
                "anchor": (int(b[0]), int(b[1])),
                "mask": _mask_from_component(coords, b),
                "area": int(len(coords)),
            }
        )
    return patches


def _progressions(anchors: list[Coord]) -> list[tuple[Coord, ...]]:
    pts = {tuple(map(int, p)) for p in anchors}
    n = len(pts)
    out: list[tuple[Coord, ...]] = []
    seen: set[tuple[Coord, ...]] = set()
    for start in pts:
        for nxt in pts:
            if nxt == start:
                continue
            step = (nxt[0] - start[0], nxt[1] - start[1])
            seq = tuple((start[0] + k * step[0], start[1] + k * step[1]) for k in range(n))
            if set(seq) == pts and seq not in seen:
                seen.add(seq)
                out.append(seq)
    return out


def _visible_overlap_with_rect(anchor: Coord, mask: np.ndarray, rect: BBox) -> bool:
    ar, ac = map(int, anchor)
    rr0, rc0, rr1, rc1 = rect
    for dr, dc in np.argwhere(mask):
        r, c = ar + int(dr), ac + int(dc)
        if rr0 <= r <= rr1 and rc0 <= c <= rc1:
            return True
    return False


def _parse_trail_for_rect(g: np.ndarray, bg: int, color: int, rect: BBox) -> dict[str, object] | None:
    patches = _sprite_patches(g, int(color))
    if len(patches) < 3:
        return None

    shapes = {tuple(np.asarray(p["mask"]).shape) for p in patches}
    if len(shapes) != 1:
        return None
    sprite_h, sprite_w = next(iter(shapes))

    anchors = [p["anchor"] for p in patches]  # type: ignore[list-item]
    if len(set(anchors)) != len(anchors):
        return None
    by_anchor = {p["anchor"]: p for p in patches}

    prototype = np.zeros((sprite_h, sprite_w), dtype=bool)
    masks = []
    for p in patches:
        m = np.asarray(p["mask"], dtype=bool)
        masks.append(m)
        prototype |= m
    if not prototype.any():
        return None
    if any(np.array_equal(m, prototype) for m in masks):
        return None
    if len({m.tobytes() for m in masks}) != len(masks):
        return None
    for i in range(len(masks)):
        without = np.zeros_like(prototype)
        for j, m in enumerate(masks):
            if i != j:
                without |= m
        if np.array_equal(without, prototype):
            return None

    valid: dict[tuple[Coord, Coord, Coord], dict[str, object]] = {}
    for seq in _progressions(anchors):
        if len(seq) < 3:
            continue

        step = (seq[1][0] - seq[0][0], seq[1][1] - seq[0][1])
        if step == (0, 0):
            continue
        if abs(step[0]) < sprite_h and abs(step[1]) < sprite_w:
            continue

        bboxes = [by_anchor[a]["bbox"] for a in seq]
        touching = [i for i, b in enumerate(bboxes) if _intersects(b, rect)]  # type: ignore[arg-type]
        if len(touching) != 1:
            continue
        term_idx = touching[0]
        if term_idx not in (0, len(seq) - 1):
            continue

        if term_idx == len(seq) - 1:
            move = step
            source = (seq[0][0] - move[0], seq[0][1] - move[1])
            terminal = seq[-1]
        else:
            move = (-step[0], -step[1])
            source = (seq[-1][0] - move[0], seq[-1][1] - move[1])
            terminal = seq[0]

        if not _inside(g.shape, source[0], source[1], sprite_h, sprite_w):
            continue
        if not np.all(g[source[0] : source[0] + sprite_h, source[1] : source[1] + sprite_w] == int(bg)):
            continue

        terminal_mask = np.asarray(by_anchor[terminal]["mask"], dtype=bool)
        if not _visible_overlap_with_rect(terminal, terminal_mask, rect):
            continue

        key = (tuple(map(int, source)), tuple(map(int, move)), tuple(map(int, terminal)))
        valid[key] = {
            "sprite_color": int(color),
            "prototype": prototype,
            "source": tuple(map(int, source)),
            "move": tuple(map(int, move)),
            "terminal": tuple(map(int, terminal)),
            "anchors": tuple(tuple(map(int, a)) for a in seq),
            "copy_masks": tuple(m.copy() for m in masks),
        }

    if len(valid) != 1:
        return None
    return next(iter(valid.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each missing source sprite from complementary damaged copies, then clear and repair the scene.

    Input structure:
    - The dominant color is the background.
    - Each group has one single-color solid rectangle and one straight row or
      column trail of same-color damaged sprite copies; the terminal copy visibly
      overlaps the rectangle.
    - All damaged copies in a trail have the same bounding box, no copy is
      complete, and their normalized union is the unseen complete prototype.
    - The equally spaced source position one step farther from the rectangle is
      empty background.

    Rule:
    - Detect every repaired rectangle from its dense non-background rectangular
      footprint, allowing sprite-colored overwritten cells inside that footprint.
    - For the sprite color whose trail has exactly one terminal bbox intersecting
      that rectangle, align all damaged copies by their bboxes and OR their
      visible cells to reconstruct the complete prototype.
    - Extrapolate one trail step away from the rectangle and draw that complete
      prototype there.
    - Remove all original damaged copies and refill each rectangle bbox with its
      rectangle color, so the output contains only repaired rectangles and new
      source sprites.

    Key insight:
    - The original sprite is absent twice: its shape must be inferred by
      complementary copy union, and its location must be inferred by reversing
      the periodic trail from the rectangle end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    bg = _get_bg(g)
    rects = _find_rectangles(g, bg)
    rect_colors = {int(r["color"]) for r in rects}
    sprite_colors = [c for c in _colors(g, exclude={bg}) if c not in rect_colors]
    assert sprite_colors, "expected damaged sprite colors"

    groups: list[dict[str, object]] = []
    used_sprites: set[int] = set()
    for rect in rects:
        candidates: list[dict[str, object]] = []
        for color in sprite_colors:
            if color in used_sprites:
                continue
            parsed = _parse_trail_for_rect(g, bg, int(color), rect["bbox"])  # type: ignore[arg-type]
            if parsed is None:
                continue
            item = dict(parsed)
            item["rect_color"] = int(rect["color"])
            item["rect_bbox"] = rect["bbox"]
            candidates.append(item)
        assert len(candidates) == 1, "each rectangle must match one unambiguous sprite trail"
        groups.append(candidates[0])
        used_sprites.add(int(candidates[0]["sprite_color"]))

    assert used_sprites == set(sprite_colors), "every sprite color must belong to one group"

    out = np.full_like(g, int(bg))
    for group in groups:
        r0, c0, r1, c1 = group["rect_bbox"]  # type: ignore[misc]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(group["rect_color"])

    for group in groups:
        proto = np.asarray(group["prototype"], dtype=bool)
        sr, sc = group["source"]  # type: ignore[misc]
        for dr, dc in np.argwhere(proto):
            rr, cc = int(sr) + int(dr), int(sc) + int(dc)
            assert int(out[rr, cc]) == int(bg), "new source sprite must not overwrite another output object"
            out[rr, cc] = int(group["sprite_color"])

    return out


# ---------------- generator ----------------


def _make_sprite() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        spine = np.zeros((h, w), dtype=bool)
        spine[h // 2, :] = True
        spine[:, w // 2] = True
        available = [(r, c) for r in range(h) for c in range(w) if not spine[r, c]]
        max_extra = len(available) - 1
        if max_extra < 3:
            continue
        extra_count = random.randint(3, max_extra)
        extras = random.sample(available, extra_count)

        sprite = spine.copy()
        for r, c in extras:
            sprite[r, c] = True
        return sprite, spine
    raise RuntimeError("failed to make sprite")


def _make_damage_masks(sprite: np.ndarray, spine: np.ndarray, copy_count: int) -> list[np.ndarray] | None:
    extras = [tuple(map(int, p)) for p in np.argwhere(sprite & ~spine)]
    if len(extras) < copy_count:
        return None

    for _ in range(80):
        random.shuffle(extras)
        unique = extras[:copy_count]
        optional = extras[copy_count:]

        masks: list[np.ndarray] = []
        for i in range(copy_count):
            m = spine.copy()
            ur, uc = unique[i]
            m[ur, uc] = True
            for r, c in optional:
                if random.random() < 0.45:
                    m[r, c] = True
            masks.append(m)

        for r, c in optional:
            if not any(m[r, c] for m in masks):
                random.choice(masks)[r, c] = True

        for i, m in enumerate(masks):
            if np.array_equal(m, sprite):
                removable = [p for p in optional if m[p] and p not in unique]
                if not removable:
                    removable = [p for p in optional if m[p]]
                if removable:
                    rr, cc = random.choice(removable)
                    masks[i][rr, cc] = False

        if any(np.array_equal(m, sprite) for m in masks):
            continue
        if len({m.tobytes() for m in masks}) != len(masks):
            continue

        union = np.zeros_like(sprite)
        for m in masks:
            union |= m
        if not np.array_equal(union, sprite):
            continue
        essential = True
        for i in range(len(masks)):
            without = np.zeros_like(sprite)
            for j, m in enumerate(masks):
                if i != j:
                    without |= m
            if np.array_equal(without, sprite):
                essential = False
                break
        if essential:
            random.shuffle(masks)
            return masks
    return None


def _terminal_anchor(rect: BBox, sprite_shape: tuple[int, int], direction: Coord) -> Coord | None:
    rr0, rc0, rr1, rc1 = rect
    sh, sw = sprite_shape
    dr, dc = direction
    rect_h, rect_w = rr1 - rr0 + 1, rc1 - rc0 + 1
    if dr == 0:
        if rect_h < sh + 2:
            return None
        row = random.randint(rr0 + 1, rr1 - sh)
        overlap = random.randint(1, min(2, sw - 1, rect_w - 1))
        col = rc0 - sw + overlap if dc == 1 else rc1 - overlap + 1
        return int(row), int(col)
    else:
        if rect_w < sw + 2:
            return None
        col = random.randint(rc0 + 1, rc1 - sw)
        overlap = random.randint(1, min(2, sh - 1, rect_h - 1))
        row = rr0 - sh + overlap if dr == 1 else rr1 - overlap + 1
        return int(row), int(col)


def _step_vector(sprite_shape: tuple[int, int], direction: Coord) -> Coord:
    sh, sw = sprite_shape
    dr, dc = direction
    gap = random.randint(2, 4)
    if dr == 0:
        return 0, int(dc * (sw + gap))
    return int(dr * (sh + gap)), 0


def _mask_has_rect_overlap(anchor: Coord, mask: np.ndarray, rect: BBox) -> bool:
    return _visible_overlap_with_rect(anchor, mask, rect)


def _reserve_cells(
    rect: BBox,
    anchors: list[Coord],
    source: Coord,
    sprite_shape: tuple[int, int],
) -> set[Coord]:
    sh, sw = sprite_shape
    cells = set(_bbox_cells(rect))
    for ar, ac in [*anchors, source]:
        cells.update(_bbox_cells((int(ar), int(ac), int(ar) + sh - 1, int(ac) + sw - 1)))
    return cells


def _render_expected(shape: tuple[int, int], bg: int, groups: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for group in groups:
        r0, c0, r1, c1 = group["rect_bbox"]  # type: ignore[misc]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(group["rect_color"])
    for group in groups:
        _paint_mask(
            out,
            group["source"],  # type: ignore[arg-type]
            np.asarray(group["prototype"], dtype=bool),
            int(group["sprite_color"]),
        )
    return out


def _try_place_group(
    grid: np.ndarray,
    blocked: set[Coord],
    rect_color: int,
    sprite_color: int,
) -> dict[str, object] | None:
    H, W = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        sprite, spine = _make_sprite()
        sh, sw = sprite.shape
        copy_count = random.randint(3, min(5, int(np.count_nonzero(sprite & ~spine))))
        masks = _make_damage_masks(sprite, spine, copy_count)
        if masks is None:
            continue

        rh = random.randint(max(5, sh + 2), 9)
        rw = random.randint(max(5, sw + 2), 10)
        if rh >= H - 2 or rw >= W - 2:
            continue
        rect = (
            random.randint(1, H - rh - 2),
            random.randint(1, W - rw - 2),
            0,
            0,
        )
        rect = (rect[0], rect[1], rect[0] + rh - 1, rect[1] + rw - 1)

        direction = random.choice(DIRECTIONS)
        terminal = _terminal_anchor(rect, (sh, sw), direction)
        if terminal is None:
            continue
        step = _step_vector((sh, sw), direction)
        anchors = [
            (terminal[0] - (copy_count - 1 - i) * step[0], terminal[1] - (copy_count - 1 - i) * step[1])
            for i in range(copy_count)
        ]
        source = (terminal[0] - copy_count * step[0], terminal[1] - copy_count * step[1])

        if not all(_inside(grid.shape, r, c, sh, sw) for r, c in [*anchors, source]):
            continue

        terminal_choices = [i for i, m in enumerate(masks) if _mask_has_rect_overlap(terminal, m, rect)]
        if not terminal_choices:
            continue
        term_i = random.choice(terminal_choices)
        ordered_masks = [masks[i] for i in range(len(masks)) if i != term_i]
        random.shuffle(ordered_masks)
        ordered_masks.append(masks[term_i])

        reserved = _reserve_cells(rect, anchors, source, (sh, sw))
        if reserved & blocked:
            continue
        if any(not (0 <= r < H and 0 <= c < W) for r, c in reserved):
            continue

        grid[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1] = int(rect_color)
        for anchor, mask in zip(anchors, ordered_masks):
            _paint_mask(grid, anchor, mask, int(sprite_color))

        return {
            "rect_bbox": rect,
            "rect_color": int(rect_color),
            "sprite_color": int(sprite_color),
            "prototype": sprite.copy(),
            "copy_masks": tuple(m.copy() for m in ordered_masks),
            "anchors": tuple(anchors),
            "source": tuple(map(int, source)),
            "terminal": tuple(map(int, terminal)),
            "step": tuple(map(int, step)),
            "reserved": reserved,
        }
    return None


def _bg_is_unique(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    return winners == [int(bg)]


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, groups: list[dict[str, object]], bg: int) -> bool:
    if not groups:
        return False
    if np.array_equal(grid, expected):
        return False

    clear_only = grid.copy()
    for group in groups:
        clear_only[clear_only == int(group["sprite_color"])] = int(bg)
        r0, c0, r1, c1 = group["rect_bbox"]  # type: ignore[misc]
        clear_only[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(group["rect_color"])
    if np.array_equal(clear_only, expected):
        return False

    for group in groups:
        proto = np.asarray(group["prototype"], dtype=bool)
        masks = [np.asarray(m, dtype=bool) for m in group["copy_masks"]]  # type: ignore[union-attr]
        union = np.zeros_like(proto)
        for m in masks:
            if np.array_equal(m, proto):
                return False
            union |= m
        if not np.array_equal(union, proto):
            return False
        for i in range(len(masks)):
            without = np.zeros_like(proto)
            for j, m in enumerate(masks):
                if i != j:
                    without |= m
            if np.array_equal(without, proto):
                return False

        sr, sc = group["source"]  # type: ignore[misc]
        sh, sw = proto.shape
        if not np.all(grid[int(sr) : int(sr) + sh, int(sc) : int(sc) + sw] == int(bg)):
            return False

        terminal = group["terminal"]  # type: ignore[assignment]
        term_mask = masks[-1]
        if not _visible_overlap_with_rect(terminal, term_mask, group["rect_bbox"]):  # type: ignore[arg-type]
            return False

        one_copy_source = clear_only.copy()
        _paint_mask(one_copy_source, group["source"], masks[0], int(group["sprite_color"]))  # type: ignore[arg-type]
        if np.array_equal(one_copy_source, expected):
            return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice([1, 2, 2, 3])
    min_side = 18 if group_count == 1 else (23 if group_count == 2 else 27)
    H = random.randint(min_side, 30)
    W = random.randint(min_side, 30)

    bg = random.choice(ARC_COLORS)
    palette = [c for c in ARC_COLORS if c != bg]
    if len(palette) < 2 * group_count:
        return None
    chosen = random.sample(palette, 2 * group_count)
    rect_colors = chosen[:group_count]
    sprite_colors = chosen[group_count:]

    grid = np.full((H, W), int(bg), dtype=int)
    blocked: set[Coord] = set()
    groups: list[dict[str, object]] = []

    for rect_color, sprite_color in zip(rect_colors, sprite_colors):
        group = _try_place_group(grid, blocked, int(rect_color), int(sprite_color))
        if group is None:
            return None
        groups.append(group)
        blocked |= _dilate_cells(group["reserved"], grid.shape)  # type: ignore[arg-type]

    expected = _render_expected(grid.shape, int(bg), groups)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _bg_is_unique(grid, int(bg)):
        return None
    if not _semantic_checks(grid, expected, groups, int(bg)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
