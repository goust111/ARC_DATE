from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _classify_frame(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 4 or width < 4:
        return None

    on_perimeter = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    if not bool(np.all(on_perimeter)):
        return None

    perimeter_area = 2 * (height + width) - 4
    if int(coords.shape[0]) != perimeter_area:
        return None

    if not bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg)):
        return None

    colors = Counter(int(grid[int(r), int(c)]) for r, c in coords)
    frame_color, frame_count = colors.most_common(1)[0]
    if frame_count != perimeter_area - 1:
        return None
    marker_colors = [color for color, count in colors.items() if color != int(frame_color) and count == 1]
    if len(marker_colors) != 1 or len(colors) != 2:
        return None

    marker_color = int(marker_colors[0])
    marker_pos = next((int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == marker_color)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
        "frame_color": int(frame_color),
        "marker_color": int(marker_color),
        "marker_pos": marker_pos,
    }


def _associate_sprite(sprite_bbox: tuple[int, int, int, int], frames: list[dict]) -> tuple[int, str]:
    r0, c0, r1, c1 = sprite_bbox
    candidates: list[tuple[int, int, str]] = []
    for idx, frame in enumerate(frames):
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        fr0, fc0, fr1, fc1 = frame["bbox"]

        if ic0 <= c0 and c1 <= ic1:
            top_gap = int(fr0 - r1 - 1)
            if top_gap >= 1:
                candidates.append((top_gap, idx, "top"))
            bottom_gap = int(r0 - fr1 - 1)
            if bottom_gap >= 1:
                candidates.append((bottom_gap, idx, "bottom"))

        if ir0 <= r0 and r1 <= ir1:
            left_gap = int(fc0 - c1 - 1)
            if left_gap >= 1:
                candidates.append((left_gap, idx, "left"))
            right_gap = int(c0 - fc1 - 1)
            if right_gap >= 1:
                candidates.append((right_gap, idx, "right"))

    if not candidates:
        raise AssertionError("sprite could not be matched to a frame")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_gap = int(candidates[0][0])
    best = [item for item in candidates if int(item[0]) == best_gap]
    if len(best) != 1:
        raise AssertionError("sprite/frame relation is ambiguous")
    _gap, frame_idx, side = best[0]
    return int(frame_idx), side


def _project_cells(coords: np.ndarray, frame: dict, side: str) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    ir0, ic0, ir1, ic1 = frame["inner_bbox"]
    if side == "top":
        dr, dc = int(ir1 - r1), 0
    elif side == "bottom":
        dr, dc = int(ir0 - r0), 0
    elif side == "left":
        dr, dc = 0, int(ic1 - c1)
    else:
        dr, dc = 0, int(ic0 - c0)
    shifted = coords + np.asarray([dr, dc], dtype=int)
    keep = (
        (shifted[:, 0] >= ir0)
        & (shifted[:, 0] <= ir1)
        & (shifted[:, 1] >= ic0)
        & (shifted[:, 1] <= ic1)
    )
    return shifted[keep]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each exterior sprite into the nearby frame interior.

    The sprite side relative to the frame determines the translation axis, while the
    frame-edge marker determines the output color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    foreground = _components(g != bg)
    frames: list[dict] = []
    sprites: list[dict] = []
    for coords in foreground:
        frame = _classify_frame(g, int(bg), coords)
        if frame is not None:
            frames.append(frame)
        else:
            sprites.append({"coords": coords, "bbox": _bbox(coords)})

    assert frames, "expected at least one frame"
    assert sprites, "expected at least one sprite"

    out = g.copy()
    moves: list[tuple[np.ndarray, dict, str]] = []
    for sprite in sprites:
        frame_idx, side = _associate_sprite(sprite["bbox"], frames)
        moves.append((sprite["coords"], frames[frame_idx], side))
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(bg)

    for coords, frame, side in moves:
        projected = _project_cells(coords, frame, side)
        out[projected[:, 0], projected[:, 1]] = int(frame["marker_color"])

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _make_sprite(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    h = random.randint(2, min(4, max_h))
    w = random.randint(2, min(4, max_w))
    target_area = random.randint(3, min(h * w, 7))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    norm = _normalize(cells)
    hh = max(r for r, _ in norm) + 1
    ww = max(c for _, c in norm) + 1
    mask = np.zeros((hh, ww), dtype=bool)
    for r, c in norm:
        mask[int(r), int(c)] = True
    return mask


def _marker_local(frame_h: int, frame_w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, frame_w - 2)
    if side == "bottom":
        return frame_h - 1, random.randint(1, frame_w - 2)
    if side == "left":
        return random.randint(1, frame_h - 2), 0
    return random.randint(1, frame_h - 2), frame_w - 1


def _build_object(side: str, marker_side: str, marker_color: int, sprite_color: int) -> dict | None:
    inner_h = random.randint(3, 5)
    inner_w = random.randint(3, 5)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    sprite = _make_sprite(inner_h, inner_w)
    if sprite is None:
        return None
    sh, sw = sprite.shape
    gap = random.randint(1, 2)

    if side == "top":
        frame_local = (sh + gap, 0)
        sprite_local = (0, 1 + random.randint(0, inner_w - sw))
        cluster_h, cluster_w = sh + gap + frame_h, frame_w
    elif side == "bottom":
        frame_local = (0, 0)
        sprite_local = (frame_h + gap, 1 + random.randint(0, inner_w - sw))
        cluster_h, cluster_w = frame_h + gap + sh, frame_w
    elif side == "left":
        frame_local = (0, sw + gap)
        sprite_local = (1 + random.randint(0, inner_h - sh), 0)
        cluster_h, cluster_w = frame_h, sw + gap + frame_w
    else:
        frame_local = (0, 0)
        sprite_local = (1 + random.randint(0, inner_h - sh), frame_w + gap)
        cluster_h, cluster_w = frame_h, frame_w + gap + sw

    marker_r, marker_c = _marker_local(frame_h, frame_w, marker_side)
    return {
        "side": side,
        "marker_side": marker_side,
        "marker_color": int(marker_color),
        "sprite_color": int(sprite_color),
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "sprite_mask": sprite,
        "frame_local": tuple(map(int, frame_local)),
        "sprite_local": tuple(map(int, sprite_local)),
        "marker_local": (int(marker_r), int(marker_c)),
        "cluster_h": int(cluster_h),
        "cluster_w": int(cluster_w),
    }


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _place_objects(objects: list[dict], height: int, width: int) -> bool:
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(objects)), key=lambda idx: -(objects[idx]["cluster_h"] * objects[idx]["cluster_w"]))
    for idx in order:
        obj = objects[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, height - obj["cluster_h"] + 1):
            for c0 in range(0, width - obj["cluster_w"] + 1):
                box = (int(r0), int(c0), int(r0 + obj["cluster_h"] - 1), int(c0 + obj["cluster_w"] - 1))
                if all(_boxes_separate(box, other, pad=1) for other in placed_boxes):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        cluster_r0, cluster_c0 = random.choice(candidates)
        obj["cluster_origin"] = (int(cluster_r0), int(cluster_c0))
        placed_boxes.append(
            (
                int(cluster_r0),
                int(cluster_c0),
                int(cluster_r0 + obj["cluster_h"] - 1),
                int(cluster_c0 + obj["cluster_w"] - 1),
            )
        )
    return True


def _sprite_abs_cells(obj: dict) -> np.ndarray:
    cluster_r0, cluster_c0 = obj["cluster_origin"]
    sprite_r0, sprite_c0 = obj["sprite_local"]
    abs_r0 = int(cluster_r0 + sprite_r0)
    abs_c0 = int(cluster_c0 + sprite_c0)
    local = np.argwhere(obj["sprite_mask"])
    return local + np.asarray([abs_r0, abs_c0], dtype=int)


def _frame_abs_bbox(obj: dict) -> tuple[int, int, int, int]:
    cluster_r0, cluster_c0 = obj["cluster_origin"]
    frame_r0, frame_c0 = obj["frame_local"]
    abs_r0 = int(cluster_r0 + frame_r0)
    abs_c0 = int(cluster_c0 + frame_c0)
    return (
        abs_r0,
        abs_c0,
        int(abs_r0 + obj["frame_h"] - 1),
        int(abs_c0 + obj["frame_w"] - 1),
    )


def _render_input(height: int, width: int, objects: list[dict]) -> np.ndarray:
    bg = 0
    frame_color = 5
    grid = np.full((height, width), bg, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = _frame_abs_bbox(obj)
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        mr, mc = obj["marker_local"]
        grid[r0 + mr, c0 + mc] = int(obj["marker_color"])

    for obj in objects:
        sprite_cells = _sprite_abs_cells(obj)
        grid[sprite_cells[:, 0], sprite_cells[:, 1]] = int(obj["sprite_color"])
    return grid


def _expected_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        sprite_cells = _sprite_abs_cells(obj)
        out[sprite_cells[:, 0], sprite_cells[:, 1]] = 0

    for obj in objects:
        sprite_cells = _sprite_abs_cells(obj)
        fr0, fc0, fr1, fc1 = _frame_abs_bbox(obj)
        frame = {"inner_bbox": (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1), "marker_color": obj["marker_color"]}
        projected = _project_cells(sprite_cells, frame, obj["side"])
        out[projected[:, 0], projected[:, 1]] = int(obj["marker_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(22, 30)
    object_count = random.randint(2, 4)

    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(colors)
    marker_colors = colors[:object_count]
    sprite_colors = colors[object_count : 2 * object_count]
    sides = random.sample(list(SIDES), object_count)

    objects: list[dict] = []
    for i in range(object_count):
        marker_side = random.choice(SIDES)
        obj = _build_object(sides[i], marker_side, marker_colors[i], sprite_colors[i])
        if obj is None:
            return None
        objects.append(obj)

    if not _place_objects(objects, height, width):
        return None

    inp = _render_input(height, width, objects)
    expected = _expected_output(inp, objects)

    if _get_bg(inp) != 0:
        return None
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
