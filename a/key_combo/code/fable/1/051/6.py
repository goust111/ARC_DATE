from __future__ import annotations

from collections import Counter, deque
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox_coords, find_components, flood_fill, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).flat)).most_common(1)[0][0])

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        arr = np.asarray(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def flood_fill(grid: np.ndarray, r: int, c: int, color: int, *, connectivity: int = 4) -> None:
        g = np.asarray(grid)
        h, w = map(int, g.shape)
        r0 = int(r)
        c0 = int(c)
        if not (0 <= r0 < h and 0 <= c0 < w):
            return
        old = int(g[r0, c0])
        new = int(color)
        if old == new:
            return
        stack: list[tuple[int, int]] = [(r0, c0)]
        while stack:
            rr, cc = stack.pop()
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if int(g[rr, cc]) != old:
                continue
            g[rr, cc] = new
            stack.append((rr - 1, cc))
            stack.append((rr + 1, cc))
            stack.append((rr, cc - 1))
            stack.append((rr, cc + 1))

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: tuple[int, ...] | list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        h, w = map(int, g.shape)
        allowed = None if colors is None else {int(v) for v in colors}
        seen = np.zeros((h, w), dtype=bool)
        out: list[dict] = []
        for r in range(h):
            for c in range(w):
                if seen[r, c]:
                    continue
                color = int(g[r, c])
                if color == int(bg):
                    seen[r, c] = True
                    continue
                if allowed is not None and color not in allowed:
                    continue
                queue: deque[tuple[int, int]] = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while queue:
                    rr, cc = queue.popleft()
                    if allowed is not None and int(g[rr, cc]) not in allowed:
                        continue
                    if monochromatic and int(g[rr, cc]) != color:
                        continue
                    if not monochromatic and int(g[rr, cc]) == int(bg):
                        continue
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                            val = int(g[nr, nc])
                            if monochromatic:
                                if val == color:
                                    seen[nr, nc] = True
                                    queue.append((nr, nc))
                            else:
                                if val != int(bg) and (allowed is None or val in allowed):
                                    seen[nr, nc] = True
                                    queue.append((nr, nc))
                if len(coords) >= int(min_area):
                    arr = np.array(coords, dtype=int)
                    out.append(
                        {
                            "color": color,
                            "coords": arr,
                            "bbox": bbox_coords(arr),
                            "area": int(arr.shape[0]),
                        }
                    )
        if return_format != "dict":
            raise ValueError("fallback find_components only supports return_format='dict'")
        return out


BG_CHOICES = (0, 1, 2, 3, 4, 6, 7, 8, 9)
FRAME_CHOICES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
FRAME_MIN_H = 3
FRAME_MAX_H = 6
FRAME_MIN_W = 4
FRAME_MAX_W = 5
FRAME_GAP_MARGIN = 1
BOX_MARGIN = 1
EMITTER_COUNT = 4
MIN_COL_GAP = 6
MAX_GENERATION_ATTEMPTS = 500

SIDE_TO_DIR = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

ROTATION_SIDE = {
    "top": {0: "top", 1: "right", 2: "bottom", 3: "left"},
    "right": {0: "right", 1: "bottom", 2: "left", 3: "top"},
    "bottom": {0: "bottom", 1: "left", 2: "top", 3: "right"},
    "left": {0: "left", 1: "top", 2: "right", 3: "bottom"},
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _border_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    return cells


def _emitter_dir(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, pos)
    h, w = map(int, shape)
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise AssertionError("emitter is not on the border")


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = BOX_MARGIN) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0


def _frame_patch(frame: dict, bg: int, frame_color: int) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    patch = np.full((h, w), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    gr, gc = frame["gap"]
    patch[int(gr - r0), int(gc - c0)] = int(bg)
    return patch


def _gap_from_frame(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["gap_side"]
    offset = int(frame["gap_offset"])
    if side == "top":
        return int(r0), int(c0 + offset)
    if side == "bottom":
        return int(r1), int(c0 + offset)
    if side == "left":
        return int(r0 + offset), int(c0)
    return int(r0 + offset), int(c1)


def _enter_seed(frame: dict) -> tuple[int, int]:
    gr, gc = frame["gap"]
    dr, dc = SIDE_TO_DIR[frame["gap_side"]]
    return int(gr + dr), int(gc + dc)


def _wall_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells = {
        (r, c)
        for r in range(int(r0), int(r1) + 1)
        for c in range(int(c0), int(c1) + 1)
        if r in (int(r0), int(r1)) or c in (int(c0), int(c1))
    }
    cells.remove(tuple(map(int, frame["gap"])))
    return cells


def _column_choices(width: int) -> list[int] | None:
    choices = list(range(3, width - 3))
    random.shuffle(choices)
    picked: list[int] = []
    for col in choices:
        if all(abs(col - other) >= MIN_COL_GAP for other in picked):
            picked.append(col)
            if len(picked) == EMITTER_COUNT:
                return sorted(picked)
    return None


def _frame_ok(frame: dict, shape: tuple[int, int], frames: list[dict]) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    h, w = map(int, shape)
    if not (2 <= r0 < r1 < h - 2 and 1 <= c0 < c1 < w - 1):
        return False
    for other in frames:
        if not _boxes_separated(frame["bbox"], other["bbox"]):
            return False
    return True


def _random_bbox_for_column(
    shape: tuple[int, int],
    col: int,
    side: str,
    *,
    row_lo: int,
    row_hi: int,
) -> tuple[tuple[int, int, int, int], int] | None:
    h, w = map(int, shape)
    if row_lo > row_hi:
        return None
    dims: list[tuple[int, int]] = []
    for fh in range(FRAME_MIN_H, FRAME_MAX_H + 1):
        for fw in range(FRAME_MIN_W, FRAME_MAX_W + 1):
            dims.append((fh, fw))
    random.shuffle(dims)
    for fh, fw in dims:
        offsets = list(range(FRAME_GAP_MARGIN, fw - FRAME_GAP_MARGIN))
        random.shuffle(offsets)
        for offset in offsets:
            c0 = int(col - offset)
            c1 = int(c0 + fw - 1)
            if c0 < 1 or c1 >= w - 1:
                continue
            rows = list(range(int(row_lo), int(row_hi) + 1))
            random.shuffle(rows)
            for r0 in rows:
                r1 = int(r0 + fh - 1)
                if r1 >= h - 2:
                    continue
                if side == "bottom" and r1 > row_hi + fh - 1:
                    continue
                return (int(r0), int(c0), int(r1), int(c1)), int(offset)
    return None


def _make_frame(
    bbox: tuple[int, int, int, int],
    gap_side: str,
    gap_offset: int,
    *,
    role: str,
    column: int,
    emitter_side: str,
) -> dict:
    frame = {
        "bbox": tuple(map(int, bbox)),
        "gap_side": str(gap_side),
        "gap_offset": int(gap_offset),
        "role": str(role),
        "column": int(column),
        "emitter_side": str(emitter_side),
    }
    frame["gap"] = _gap_from_frame(frame)
    frame["enter_dir"] = SIDE_TO_DIR[frame["gap_side"]]
    return frame


def _place_enter_frame(shape: tuple[int, int], col: int, emitter_side: str, frames: list[dict]) -> dict | None:
    h, _w = map(int, shape)
    if emitter_side == "top":
        row_lo = 4
        row_hi = max(row_lo, h - FRAME_MAX_H - 4)
    else:
        row_lo = 3
        row_hi = max(row_lo, h - FRAME_MAX_H - 5)
    for _ in range(80):
        sampled = _random_bbox_for_column(shape, col, emitter_side, row_lo=row_lo, row_hi=row_hi)
        if sampled is None:
            return None
        bbox, offset = sampled
        frame = _make_frame(bbox, emitter_side, offset, role="enter", column=col, emitter_side=emitter_side)
        if emitter_side == "top" and frame["gap"][0] < 4:
            continue
        if emitter_side == "bottom" and frame["gap"][0] > h - 5:
            continue
        if _frame_ok(frame, shape, frames):
            return frame
    return None


def _place_block_frame(shape: tuple[int, int], col: int, emitter_side: str, frames: list[dict]) -> dict | None:
    h, _w = map(int, shape)
    if emitter_side == "top":
        row_lo = 4
        row_hi = max(row_lo, h - FRAME_MAX_H - 4)
        forbidden_side = "top"
        side_choices = ["bottom", "left", "right"]
    else:
        row_lo = 3
        row_hi = max(row_lo, h - FRAME_MAX_H - 5)
        forbidden_side = "bottom"
        side_choices = ["top", "left", "right"]
    for _ in range(100):
        sampled = _random_bbox_for_column(shape, col, emitter_side, row_lo=row_lo, row_hi=row_hi)
        if sampled is None:
            return None
        bbox, offset = sampled
        gap_side = random.choice(side_choices)
        r0, c0, r1, c1 = bbox
        if gap_side in ("top", "bottom"):
            gap_offset = random.randint(1, c1 - c0 - 1)
            if gap_side == forbidden_side and c0 + gap_offset == col:
                continue
        else:
            gap_offset = random.randint(1, r1 - r0 - 1)
        frame = _make_frame(bbox, gap_side, gap_offset, role="block", column=col, emitter_side=emitter_side)
        if _frame_ok(frame, shape, frames):
            wall_row = frame["bbox"][0] if emitter_side == "top" else frame["bbox"][2]
            if emitter_side == "top" and wall_row < 4:
                continue
            if emitter_side == "bottom" and wall_row > h - 5:
                continue
            return frame
    return None


def _place_occluded_frames(shape: tuple[int, int], col: int, emitter_side: str, frames: list[dict]) -> tuple[dict, dict] | None:
    h, _w = map(int, shape)
    for _ in range(120):
        enter_frame = _place_enter_frame(shape, col, emitter_side, frames)
        if enter_frame is None:
            return None
        er0, _ec0, er1, _ec1 = enter_frame["bbox"]
        if emitter_side == "top":
            blocker_row_hi = int(er0 - FRAME_MIN_H - 3)
            if blocker_row_hi < 4:
                continue
            blocker = _place_block_frame(shape, col, emitter_side, frames + [enter_frame])
            if blocker is None:
                continue
            br0, _bc0, br1, _bc1 = blocker["bbox"]
            if br1 + 3 > er0:
                continue
        else:
            blocker = _place_block_frame(shape, col, emitter_side, frames + [enter_frame])
            if blocker is None:
                continue
            br0, _bc0, br1, _bc1 = blocker["bbox"]
            if er1 + 3 > br0:
                continue
        blocker["role"] = "occluder"
        enter_frame["role"] = "occluded_target"
        return blocker, enter_frame
    return None


def _rotate_pos(pos: tuple[int, int], shape: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = map(int, pos)
    h, w = map(int, shape)
    t = int(turns) % 4
    if t == 0:
        return r, c
    if t == 1:
        return c, h - 1 - r
    if t == 2:
        return h - 1 - r, w - 1 - c
    return w - 1 - c, r


def _rotate_grid_pair(inp: np.ndarray, out: np.ndarray, turns: int) -> tuple[np.ndarray, np.ndarray]:
    t = int(turns) % 4
    return np.rot90(inp, k=t), np.rot90(out, k=t)


def _render_scene(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(map(int, scene["shape"]))
    bg = int(scene["bg"])
    frame_color = int(scene["frame_color"])
    inp = np.full(shape, bg, dtype=int)
    for frame in scene["frames"]:
        patch = _frame_patch(frame, bg, frame_color)
        r0, c0, _r1, _c1 = frame["bbox"]
        inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    for emitter in scene["emitters"]:
        r, c = emitter["pos"]
        inp[int(r), int(c)] = int(emitter["color"])
    return inp, _expected_from_scene(inp, scene)


def _expected_from_scene(inp: np.ndarray, scene: dict) -> np.ndarray:
    out = np.array(inp, dtype=int)
    frame_by_gap = {tuple(map(int, frame["gap"])): frame for frame in scene["frames"]}
    wall_sets = {id(frame): _wall_cells(frame) for frame in scene["frames"]}
    for emitter in scene["emitters"]:
        color = int(emitter["color"])
        r, c = emitter["pos"]
        dr, dc = emitter["dir"]
        rr = int(r + dr)
        cc = int(c + dc)
        while _in_bounds(out.shape, rr, cc):
            coord = (int(rr), int(cc))
            if any(coord in walls for walls in wall_sets.values()):
                break
            out[rr, cc] = color
            frame = frame_by_gap.get(coord)
            if frame is not None and tuple(frame["enter_dir"]) == (int(dr), int(dc)):
                seed_r, seed_c = _enter_seed(frame)
                flood_fill(out, seed_r, seed_c, color, connectivity=4)
                break
            rr += dr
            cc += dc
    return out


def _aligned_gap_frames(scene: dict, emitter: dict) -> list[tuple[int, dict]]:
    er, ec = emitter["pos"]
    dr, dc = emitter["dir"]
    out: list[tuple[int, dict]] = []
    for frame in scene["frames"]:
        if tuple(frame["enter_dir"]) != (int(dr), int(dc)):
            continue
        gr, gc = frame["gap"]
        if dr != 0 and gc == ec:
            dist = abs(int(gr - er))
            out.append((dist, frame))
        if dc != 0 and gr == er:
            dist = abs(int(gc - ec))
            out.append((dist, frame))
    out.sort(key=lambda item: item[0])
    return out


def _line_frames(scene: dict, emitter: dict) -> list[tuple[int, dict]]:
    er, ec = emitter["pos"]
    dr, dc = emitter["dir"]
    out: list[tuple[int, dict]] = []
    for frame in scene["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        if dr != 0 and c0 <= ec <= c1:
            wall_r = r0 if dr > 0 else r1
            out.append((abs(int(wall_r - er)), frame))
        if dc != 0 and r0 <= er <= r1:
            wall_c = c0 if dc > 0 else c1
            out.append((abs(int(wall_c - ec)), frame))
    out.sort(key=lambda item: item[0])
    return out


def _proxy_ignore_blockers(inp: np.ndarray, scene: dict) -> np.ndarray:
    out = np.array(inp, dtype=int)
    for emitter in scene["emitters"]:
        color = int(emitter["color"])
        r, c = emitter["pos"]
        dr, dc = emitter["dir"]
        aligned = _aligned_gap_frames(scene, emitter)
        if aligned:
            _dist, frame = aligned[0]
            gr, gc = frame["gap"]
            rr = int(r + dr)
            cc = int(c + dc)
            while (rr, cc) != (gr, gc):
                out[rr, cc] = color
                rr += dr
                cc += dc
            out[gr, gc] = color
            seed_r, seed_c = _enter_seed(frame)
            flood_fill(out, seed_r, seed_c, color, connectivity=4)
            continue
        rr = int(r + dr)
        cc = int(c + dc)
        while _in_bounds(out.shape, rr, cc):
            out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def _proxy_fill_first_frame(inp: np.ndarray, scene: dict) -> np.ndarray:
    out = np.array(inp, dtype=int)
    for emitter in scene["emitters"]:
        color = int(emitter["color"])
        r, c = emitter["pos"]
        dr, dc = emitter["dir"]
        line_frames = _line_frames(scene, emitter)
        if not line_frames:
            rr = int(r + dr)
            cc = int(c + dc)
            while _in_bounds(out.shape, rr, cc):
                out[rr, cc] = color
                rr += dr
                cc += dc
            continue
        _dist, frame = line_frames[0]
        if dr > 0:
            stop_r = frame["bbox"][0]
            for rr in range(int(r + dr), int(stop_r)):
                out[rr, c] = color
        elif dr < 0:
            stop_r = frame["bbox"][2]
            for rr in range(int(r + dr), int(stop_r), -1):
                out[rr, c] = color
        elif dc > 0:
            stop_c = frame["bbox"][1]
            for cc in range(int(c + dc), int(stop_c)):
                out[r, cc] = color
        else:
            stop_c = frame["bbox"][3]
            for cc in range(int(c + dc), int(stop_c), -1):
                out[r, cc] = color
        gr, gc = frame["gap"]
        out[gr, gc] = color
        seed_r, seed_c = _enter_seed(frame)
        flood_fill(out, seed_r, seed_c, color, connectivity=4)
    return out


def _nondegenerate(scene: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False
    frames = scene["frames"]
    filled = 0
    dark = 0
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = expected[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            return False
        if np.any(interior != scene["bg"]):
            filled += 1
        else:
            dark += 1
    if filled < 1 or dark < 1:
        return False
    kinds = {emitter["kind"] for emitter in scene["emitters"]}
    if "enter" not in kinds or "through" not in kinds:
        return False
    if not ({"block", "occluded"} & kinds):
        return False
    return True


def _validate_scene(inp: np.ndarray, expected: np.ndarray, scene: dict) -> bool:
    h, w = map(int, inp.shape)
    if h > 30 or w > 30:
        return False
    if int(get_bg(inp)) != int(scene["bg"]):
        return False
    emitter_cells = [tuple(map(int, emitter["pos"])) for emitter in scene["emitters"]]
    if len(set(emitter_cells)) != len(emitter_cells):
        return False
    if not all(pos in set(_border_cells(h, w)) for pos in emitter_cells):
        return False
    if len({int(emitter["color"]) for emitter in scene["emitters"]}) != len(scene["emitters"]):
        return False
    if len({int(inp[pos]) for pos in emitter_cells}) != len(scene["emitters"]):
        return False
    if np.array_equal(_proxy_ignore_blockers(inp, scene), expected):
        return False
    if np.array_equal(_proxy_fill_first_frame(inp, scene), expected):
        return False
    if not _nondegenerate(scene, inp, expected):
        return False
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    return True


def _make_scene() -> dict | None:
    height = random.randint(18, 26)
    width = random.randint(24, 30)
    shape = (height, width)
    cols = _column_choices(width)
    if cols is None:
        return None

    bg = random.choice(BG_CHOICES)
    frame_color_choices = [color for color in FRAME_CHOICES if color != bg]
    frame_color = random.choice(frame_color_choices)
    emitter_colors = random.sample([color for color in range(10) if color not in {bg, frame_color}], EMITTER_COUNT)

    roles = ["enter", "occluded", "through", random.choices(["enter", "block", "through"], weights=[2, 1, 1])[0]]
    random.shuffle(roles)
    sides = [random.choice(["top", "bottom"]) for _ in range(EMITTER_COUNT)]
    if len(set(sides)) == 1:
        sides[random.randrange(EMITTER_COUNT)] = "bottom" if sides[0] == "top" else "top"

    frames: list[dict] = []
    emitters: list[dict] = []
    for idx, (col, color, kind, side) in enumerate(zip(cols, emitter_colors, roles, sides)):
        emitter_pos = (0, col) if side == "top" else (height - 1, col)
        emitter = {
            "pos": emitter_pos,
            "dir": SIDE_TO_DIR[side],
            "color": int(color),
            "kind": str(kind),
            "index": int(idx),
        }
        if kind == "enter":
            frame = _place_enter_frame(shape, col, side, frames)
            if frame is None:
                return None
            frames.append(frame)
        elif kind == "block":
            frame = _place_block_frame(shape, col, side, frames)
            if frame is None:
                return None
            frames.append(frame)
        elif kind == "occluded":
            placed = _place_occluded_frames(shape, col, side, frames)
            if placed is None:
                return None
            blocker, hidden = placed
            frames.extend([blocker, hidden])
        emitters.append(emitter)

    scene = {
        "shape": shape,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "frames": frames,
        "emitters": emitters,
    }
    inp, expected = _render_scene(scene)
    turns = random.randrange(4)
    rinp, rexpected = _rotate_grid_pair(inp, expected, turns)
    rotated_scene = {
        "shape": tuple(map(int, rinp.shape)),
        "bg": int(bg),
        "frame_color": int(frame_color),
        "frames": [],
        "emitters": [],
    }
    old_shape = shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        corners = [
            _rotate_pos((r0, c0), old_shape, turns),
            _rotate_pos((r0, c1), old_shape, turns),
            _rotate_pos((r1, c0), old_shape, turns),
            _rotate_pos((r1, c1), old_shape, turns),
        ]
        rows = [rc[0] for rc in corners]
        cols2 = [rc[1] for rc in corners]
        nr0, nr1 = min(rows), max(rows)
        nc0, nc1 = min(cols2), max(cols2)
        side_rot = ROTATION_SIDE[frame["gap_side"]][turns]
        gap = _rotate_pos(frame["gap"], old_shape, turns)
        if side_rot in ("top", "bottom"):
            gap_offset = int(gap[1] - nc0)
        else:
            gap_offset = int(gap[0] - nr0)
        new_frame = {
            "bbox": (int(nr0), int(nc0), int(nr1), int(nc1)),
            "gap_side": side_rot,
            "gap_offset": int(gap_offset),
            "role": frame["role"],
            "column": frame["column"],
            "emitter_side": ROTATION_SIDE[frame["emitter_side"]][turns],
            "gap": tuple(map(int, gap)),
            "enter_dir": SIDE_TO_DIR[side_rot],
        }
        rotated_scene["frames"].append(new_frame)
    for emitter in emitters:
        new_side = ROTATION_SIDE["top" if emitter["pos"][0] == 0 else "bottom"][turns]
        new_pos = _rotate_pos(emitter["pos"], old_shape, turns)
        rotated_scene["emitters"].append(
            {
                "pos": tuple(map(int, new_pos)),
                "dir": _emitter_dir(new_pos, rinp.shape),
                "color": int(emitter["color"]),
                "kind": emitter["kind"],
                "index": emitter["index"],
                "side": new_side,
            }
        )
    if not _validate_scene(rinp, rexpected, rotated_scene):
        return None
    return {"input": rinp, "scene": rotated_scene}


def _parse_frames(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(frame_color)],
        return_format="dict",
    )
    assert comps, "expected at least one frame"
    frames: list[dict] = []
    for comp in comps:
        coords = {tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)}
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frame must have an interior"
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        assert coords <= border, "frame pixels must lie on the border"
        gaps = list(border - coords)
        assert len(gaps) == 1, "each frame must have exactly one missing border cell"
        gap = tuple(map(int, gaps[0]))
        gr, gc = gap
        if gr == r0:
            assert c0 < gc < c1, "gap must be a non-corner top cell"
            gap_side = "top"
            gap_offset = gc - c0
        elif gr == r1:
            assert c0 < gc < c1, "gap must be a non-corner bottom cell"
            gap_side = "bottom"
            gap_offset = gc - c0
        elif gc == c0:
            assert r0 < gr < r1, "gap must be a non-corner left cell"
            gap_side = "left"
            gap_offset = gr - r0
        else:
            assert gc == c1 and r0 < gr < r1, "gap must be a non-corner right cell"
            gap_side = "right"
            gap_offset = gr - r0
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0 and np.all(interior == bg), "frame interior must be empty background"
        frame = {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "gap_side": gap_side,
            "gap_offset": int(gap_offset),
            "enter_dir": SIDE_TO_DIR[gap_side],
        }
        frames.append(frame)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each edge emitter inward and only light frames whose single gap lies exactly on that ray.

    Input structure:
    - A dominant background contains several one-pixel emitters on the outer border, each with a unique color.
    - The interior contains several non-touching monochrome hollow rectangles of one shared frame color.
    - Every frame is axis-aligned and has exactly one missing non-corner border cell.

    Rule:
    - Each emitter sends a straight ray perpendicular to its border edge, and every background cell on that path is recolored to the emitter color.
    - If the ray reaches a frame's missing border cell from the outside, paint that gap cell, flood-fill the frame interior with the same color, and stop the ray there.
    - If the ray instead meets any solid frame wall, stop immediately before that wall; if it meets nothing, continue to the opposite border.

    Key insight:
    - A frame lights up only when its unique gap is exactly on an incoming ray, so alignment to the gap is both the hit test and the only entrance into the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = int(get_bg(g))

    emitters: list[dict] = []
    emitter_colors: set[int] = set()
    for r, c in _border_cells(h, w):
        color = int(g[r, c])
        if color == bg:
            continue
        assert color not in emitter_colors, "emitter colors must be unique"
        emitter_colors.add(color)
        emitters.append({"pos": (int(r), int(c)), "dir": _emitter_dir((r, c), g.shape), "color": color})
    assert emitters, "expected at least one border emitter"

    non_bg_colors = set(map(int, np.unique(g))) - {bg}
    frame_palette = non_bg_colors - emitter_colors
    assert len(frame_palette) == 1, "expected one shared frame color distinct from emitter colors"
    frame_color = int(next(iter(frame_palette)))
    frames = _parse_frames(g, bg, frame_color)
    gap_to_frame = {tuple(frame["gap"]): frame for frame in frames}

    out = g.copy()
    for emitter in emitters:
        color = int(emitter["color"])
        r, c = emitter["pos"]
        dr, dc = emitter["dir"]
        rr = int(r + dr)
        cc = int(c + dc)
        while _in_bounds(g.shape, rr, cc):
            cell = int(g[rr, cc])
            if cell != bg:
                break
            out[rr, cc] = color
            frame = gap_to_frame.get((rr, cc))
            if frame is not None:
                assert tuple(frame["enter_dir"]) == (int(dr), int(dc)), "gap must be entered from its outside-facing side"
                seed_r, seed_c = _enter_seed(frame)
                flood_fill(out, seed_r, seed_c, color, connectivity=4)
                break
            rr += dr
            cc += dc
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _make_scene()
        if scene is not None:
            return scene["input"]
    raise RuntimeError("failed to generate a valid light-through-gap scene")
