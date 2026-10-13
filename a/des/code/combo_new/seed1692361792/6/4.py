import random

import numpy as np


BG = 5
RED = 2
COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MOVE = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
ROT_DIRS = ("right", "up", "left", "down")
GEN_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
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
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 0].max()), int(arr[:, 1].max())


def _normalized_key(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in np.asarray(coords, dtype=int)))


def _is_rect_frame(coords: list[tuple[int, int]] | np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return expected == set(map(tuple, np.asarray(coords, dtype=int)))


def _rotate_point(r: int, c: int, h: int, w: int, k: int) -> tuple[int, int]:
    for _ in range(k % 4):
        r, c, h, w = w - 1 - c, r, w, h
    return int(r), int(c)


def _arrow_template_map() -> dict[tuple[tuple[int, int], ...], tuple[str, tuple[int, int]]]:
    families: list[tuple[np.ndarray, tuple[int, int]]] = [
        (
            np.array(
                [
                    [0, 0, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                ],
                dtype=bool,
            ),
            (1, 0),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 1],
                    [1, 1, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=bool,
            ),
            (1, 0),
        ),
    ]
    templates: dict[tuple[tuple[int, int], ...], tuple[str, tuple[int, int]]] = {}
    for base_mask, tail in families:
        h, w = base_mask.shape
        for k, direction in enumerate(ROT_DIRS):
            mask = np.rot90(base_mask, k)
            coords = tuple(sorted(map(tuple, np.argwhere(mask))))
            tail_rot = _rotate_point(tail[0], tail[1], h, w, k)
            templates[coords] = (direction, tail_rot)
    return templates


ARROW_TEMPLATES = _arrow_template_map()


def _detect_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components(grid == RED, connectivity=8):
        key = _normalized_key(comp)
        assert key in ARROW_TEMPLATES, "unexpected arrow shape"
        direction, tail_rel = ARROW_TEMPLATES[key]
        r0, c0, _, _ = _bbox(comp)
        arrows.append(
            {
                "coords": np.asarray(comp, dtype=int),
                "direction": direction,
                "tail": (r0 + tail_rel[0], c0 + tail_rel[1]),
            }
        )
    assert arrows, "expected at least one red arrow"
    return arrows


def _find_frames_and_objects(
    grid: np.ndarray, bg: int
) -> tuple[dict[int, list[dict[str, object]]], list[dict[str, object]]]:
    frames_by_color: dict[int, list[dict[str, object]]] = {}
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, RED}):
        for comp in _components(grid == color, connectivity=4):
            coords = np.asarray(comp, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            record = {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
            }
            if _is_rect_frame(coords):
                record["inner_bbox"] = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                frames_by_color.setdefault(color, []).append(record)
            else:
                objects.append(record)
    assert objects, "expected at least one movable object"
    assert frames_by_color, "expected at least one colored frame"
    return frames_by_color, objects


def _object_lookup(objects: list[dict[str, object]]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for r, c in np.asarray(obj["coords"], dtype=int):
            lookup[(int(r), int(c))] = idx
    return lookup


def _object_for_arrow(arrow: dict[str, object], lookup: dict[tuple[int, int], int]) -> int:
    dr, dc = MOVE[str(arrow["direction"])]
    tail_r, tail_c = arrow["tail"]
    probe = (int(tail_r) - dr, int(tail_c) - dc)
    if probe in lookup:
        return lookup[probe]
    candidates: set[int] = set()
    for r, c in np.asarray(arrow["coords"], dtype=int):
        nr, nc = int(r) - dr, int(c) - dc
        if (nr, nc) in lookup:
            candidates.add(lookup[(nr, nc)])
    assert len(candidates) == 1, "arrow must identify exactly one adjacent object"
    return next(iter(candidates))


def _target_frame(
    obj: dict[str, object], direction: str, frames_by_color: dict[int, list[dict[str, object]]]
) -> dict[str, object]:
    color = int(obj["color"])
    r0, c0, r1, c1 = obj["bbox"]
    assert color in frames_by_color, "object color must have a matching frame"
    candidates: list[tuple[int, dict[str, object]]] = []
    for frame in frames_by_color[color]:
        fr0, fc0, fr1, fc1 = frame["bbox"]
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        if direction == "right" and ic0 > c1 and ir0 <= r0 and r1 <= ir1:
            candidates.append((fc0 - c1, frame))
        elif direction == "left" and ic1 < c0 and ir0 <= r0 and r1 <= ir1:
            candidates.append((c0 - fc1, frame))
        elif direction == "down" and ir0 > r1 and ic0 <= c0 and c1 <= ic1:
            candidates.append((fr0 - r1, frame))
        elif direction == "up" and ir1 < r0 and ic0 <= c0 and c1 <= ic1:
            candidates.append((r0 - fr1, frame))
    assert candidates, "no frame found along the arrow direction"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each small object along its red arrow until it settles just inside the matching same-color frame.

    Input structure:
    - The background is grey.
    - The grid contains several one-pixel-thick hollow rectangular frames, each drawn in a solid foreground color and enclosing only grey background.
    - There are also several small 4-connected solid objects, each monochrome and non-rectangular, plus a nearby red arrow marker indicating a cardinal push direction.

    Rule:
    - Read each red arrow to determine its direction and identify the unique object adjacent to the arrow's tail side.
    - Find the nearest frame of the same color that lies ahead in that direction and whose interior can contain the translated object without changing its shape.
    - Translate the object rigidly along the arrow direction until it lands inside the frame and touches the entry-side inner wall, then erase the arrow and clear the object's old position.
    - Keep every frame border unchanged.

    Key insight:
    - Each arrow simultaneously selects which nearby object moves and which same-color frame receives it: direction gives the travel axis, while color matching and straight-line alignment identify the destination container.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames_by_color, objects = _find_frames_and_objects(g, bg)
    arrows = _detect_arrows(g)
    lookup = _object_lookup(objects)

    moves: list[tuple[int, int, int]] = []
    moved: set[int] = set()
    for arrow in arrows:
        obj_idx = _object_for_arrow(arrow, lookup)
        assert obj_idx not in moved, "each object should have exactly one arrow"
        moved.add(obj_idx)
        obj = objects[obj_idx]
        frame = _target_frame(obj, str(arrow["direction"]), frames_by_color)

        r0, c0, r1, c1 = obj["bbox"]
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        dr = dc = 0
        if arrow["direction"] == "right":
            dc = ic0 - c0
        elif arrow["direction"] == "left":
            dc = ic1 - c1
        elif arrow["direction"] == "down":
            dr = ir0 - r0
        else:
            dr = ir1 - r1
        moves.append((obj_idx, dr, dc))

    out = g.copy()
    out[g == RED] = bg
    for obj_idx in moved:
        coords = np.asarray(objects[obj_idx]["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    for obj_idx, dr, dc in moves:
        obj = objects[obj_idx]
        color = int(obj["color"])
        coords = np.asarray(obj["coords"], dtype=int)
        shifted = [(int(r) + dr, int(c) + dc) for r, c in coords]
        assert all(out[r, c] == bg for r, c in shifted), "moved object must land in empty frame interior"
        for r, c in shifted:
            out[r, c] = color

    return out


def _make_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 3)
        w = rng.randint(2, 3)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        target = rng.randint(3, max_area)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            seen = set(cells)
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue

        r0, c0, r1, c1 = _bbox(list(cells))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if np.count_nonzero(mask) == mask.size:
            continue
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        return mask.copy()
    raise RuntimeError("failed to build a non-rectangular object")


def _draw_frame(patch: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, color: int) -> None:
    patch[r0, c0 : c0 + outer_w] = color
    patch[r0 + outer_h - 1, c0 : c0 + outer_w] = color
    patch[r0 : r0 + outer_h, c0] = color
    patch[r0 : r0 + outer_h, c0 + outer_w - 1] = color


def _paint_mask(patch: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.where(mask)
    patch[top + rows, left + cols] = color


def _paint_right_arrow(patch: np.ndarray, tail_r: int, tail_c: int) -> None:
    patch[tail_r, tail_c : tail_c + 3] = RED
    patch[tail_r - 1, tail_c + 2] = RED
    patch[tail_r + 1, tail_c + 2] = RED


def _build_right_pair(rng: random.Random, color: int) -> tuple[np.ndarray, np.ndarray]:
    mask = _make_shape(rng)
    oh, ow = mask.shape
    inner_h = rng.randint(max(oh, 3), min(5, oh + 2))
    inner_w = rng.randint(max(ow, 3), min(5, ow + 2))
    outer_h, outer_w = inner_h + 2, inner_w + 2
    frame_r0 = 1
    final_offset = rng.randint(0, inner_h - oh)
    obj_r0 = frame_r0 + 1 + final_offset
    start_c0 = 1
    shift = ow + 4 + rng.randint(0, 1)
    frame_c0 = shift
    patch_h = outer_h + 2
    patch_w = frame_c0 + outer_w + 1

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = np.full((patch_h, patch_w), BG, dtype=int)
    _draw_frame(inp, frame_r0, frame_c0, outer_h, outer_w, color)
    _draw_frame(out, frame_r0, frame_c0, outer_h, outer_w, color)

    _paint_mask(inp, obj_r0, start_c0, mask, color)
    _paint_mask(out, obj_r0, frame_c0 + 1, mask, color)

    anchor_rows = np.flatnonzero(mask[:, -1])
    anchor_r = int(rng.choice(anchor_rows))
    tail_r = obj_r0 + anchor_r
    tail_c = start_c0 + ow
    _paint_right_arrow(inp, tail_r, tail_c)
    return inp, out


def _rects_conflict(
    r0: int, c0: int, h: int, w: int, placed: list[tuple[int, int, int, int]], pad: int
) -> bool:
    for pr, pc, ph, pw in placed:
        if not (r0 + h + pad <= pr or pr + ph + pad <= r0 or c0 + w + pad <= pc or pc + pw + pad <= c0):
            return True
    return False


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = rng.randint(2, 3)
    colors = rng.sample(COLOR_POOL, pair_count)
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color in colors:
        inp, out = _build_right_pair(rng, color)
        k = rng.randrange(4)
        patches.append((np.rot90(inp, k), np.rot90(out, k)))

    sizes = [(patch.shape[0], patch.shape[1]) for patch, _ in patches]
    total_area = sum((h + 2) * (w + 2) for h, w in sizes)
    span = int(np.ceil(total_area ** 0.5))
    min_h = max(max(h for h, _ in sizes) + 2, min(30, span + 2))
    min_w = max(max(w for _, w in sizes) + 2, min(30, span + 2))
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        order = sorted(range(pair_count), key=lambda idx: patches[idx][0].size, reverse=True)
        placed: list[tuple[int, int, int, int]] = []
        positions: list[tuple[int, int] | None] = [None] * pair_count
        ok = True
        for idx in order:
            patch_h, patch_w = patches[idx][0].shape
            for _ in range(200):
                r0 = rng.randint(0, h - patch_h)
                c0 = rng.randint(0, w - patch_w)
                if not _rects_conflict(r0, c0, patch_h, patch_w, placed, pad=2):
                    placed.append((r0, c0, patch_h, patch_w))
                    positions[idx] = (r0, c0)
                    break
            else:
                ok = False
                break
        if not ok:
            continue

        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        for idx, (patch_in, patch_out) in enumerate(patches):
            r0, c0 = positions[idx]
            ph, pw = patch_in.shape
            inp[r0 : r0 + ph, c0 : c0 + pw] = patch_in
            out[r0 : r0 + ph, c0 : c0 + pw] = patch_out
        return inp, out
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is None:
            continue
        inp, out = case
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid arrow-push sample")
