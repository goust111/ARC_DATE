from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg, paste


BG = 0
SIDE = 5
INNER = SIDE - 2
PALETTE = tuple(range(1, 10))
ORTH_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ALL_DIRS = ORTH_DIRS + DIAG_DIRS
MAX_GENERATION_ATTEMPTS = 600


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    side: int

    @property
    def r1(self) -> int:
        return int(self.r0 + self.side - 1)

    @property
    def c1(self) -> int:
        return int(self.c0 + self.side - 1)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return int(self.r0), int(self.c0), int(self.r1), int(self.c1)

    @property
    def inner_slice(self) -> tuple[slice, slice]:
        return slice(self.r0 + 1, self.r1), slice(self.c0 + 1, self.c1)

    def border_cells(self) -> tuple[tuple[int, int], ...]:
        cells: list[tuple[int, int]] = []
        for c in range(self.c0, self.c1 + 1):
            cells.append((self.r0, c))
            cells.append((self.r1, c))
        for r in range(self.r0 + 1, self.r1):
            cells.append((r, self.c0))
            cells.append((r, self.c1))
        return tuple(cells)


@dataclass
class Scene:
    frame_color: int
    cue_color: int
    m: int
    frames: tuple[Frame, ...]
    step_types: tuple[str, ...]
    patches: tuple[np.ndarray, ...]
    cue_cells: tuple[tuple[int, int], ...]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    queue = deque([tuple(map(int, coords[0]))])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ORTH_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_sprite_library() -> tuple[np.ndarray, ...]:
    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for bits in range(1 << (INNER * INNER)):
        mask = np.array([(bits >> idx) & 1 for idx in range(INNER * INNER)], dtype=bool).reshape(INNER, INNER)
        area = int(mask.sum())
        if area < 3 or area > 5:
            continue
        if not _mask_connected(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        key = min(_shape_key(np.rot90(mask, k)) for k in range(4))
        if key in seen:
            continue
        seen.add(key)
        library.append(mask.copy())
    assert len(library) >= 10
    return tuple(library)


SPRITE_LIBRARY = _build_sprite_library()


SCENARIO_QUEUE: deque[dict[str, object]] = deque(
    [
        {"m": 1, "profile": "mixed", "idler": True},
        {"m": 2, "profile": "mixed", "idler": True},
        {"m": 3, "profile": "mixed", "idler": False},
        {"m": 1, "profile": "edge", "idler": False},
        {"m": 3, "profile": "corner", "idler": False},
        {"m": 2, "profile": "edge", "idler": True},
    ]
)


def _prefix(mask: np.ndarray) -> np.ndarray:
    pref = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    pref[1:, 1:] = np.cumsum(np.cumsum(mask.astype(int), axis=0), axis=1)
    return pref


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0])


def _find_hollow_square_frames(grid: np.ndarray, color: int) -> list[Frame]:
    mask = grid == int(color)
    pref = _prefix(mask)
    h, w = grid.shape
    frames: list[Frame] = []
    max_side = min(h, w)
    for side in range(3, max_side + 1):
        border_area = 4 * side - 4
        for r0 in range(h - side + 1):
            r1 = r0 + side - 1
            for c0 in range(w - side + 1):
                c1 = c0 + side - 1
                if _rect_sum(pref, r0, c0, r1, c1) != border_area:
                    continue
                if side > 2 and _rect_sum(pref, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                    continue
                frames.append(Frame(int(r0), int(c0), int(side)))
    frames.sort(key=lambda frame: (frame.r0, frame.c0, frame.side))
    return frames


def _detect_frame_color_and_frames(grid: np.ndarray, bg: int) -> tuple[int, list[Frame]]:
    fits: list[tuple[int, list[Frame]]] = []
    for color in colors_present(grid, exclude={bg}):
        frames = _find_hollow_square_frames(grid, int(color))
        if len(frames) < 2:
            continue
        coverage = np.zeros(grid.shape, dtype=int)
        for frame in frames:
            for r, c in frame.border_cells():
                coverage[r, c] += 1
        if np.any(coverage > 1):
            continue
        if np.array_equal(coverage > 0, grid == int(color)):
            fits.append((int(color), frames))
    assert len(fits) == 1, "expected one color to be fully explained by hollow square frames"
    return fits[0]


def _touch_type(frame_a: Frame, frame_b: Frame) -> str | None:
    border_b = set(frame_b.border_cells())
    for r, c in frame_a.border_cells():
        for dr, dc in ORTH_DIRS:
            if (r + dr, c + dc) in border_b:
                return "edge"
    for r, c in frame_a.border_cells():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if abs(dr) + abs(dc) == 1:
                    continue
                if (r + dr, c + dc) in border_b:
                    return "corner"
    return None


def _ordered_chain(frames: list[Frame], head_index: int) -> tuple[list[int], dict[tuple[int, int], str]]:
    neighbors: dict[int, list[tuple[int, str]]] = {idx: [] for idx in range(len(frames))}
    kinds: dict[tuple[int, int], str] = {}
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            touch = _touch_type(frames[i], frames[j])
            if touch is None:
                continue
            neighbors[i].append((j, touch))
            neighbors[j].append((i, touch))
            kinds[(i, j)] = touch
            kinds[(j, i)] = touch

    degrees = [len(neighbors[idx]) for idx in range(len(frames))]
    assert all(deg <= 2 for deg in degrees), "the frame contacts must form a simple chain"
    assert degrees[head_index] == 1, "the cue must attach to a chain endpoint"
    ends = [idx for idx, deg in enumerate(degrees) if deg == 1]
    assert len(ends) == 2, "the frame graph must have exactly two endpoints"

    visited = {head_index}
    chain = [head_index]
    prev = -1
    cur = head_index
    while True:
        nxt_candidates = [nxt for nxt, _touch in neighbors[cur] if nxt != prev]
        if not nxt_candidates:
            break
        assert len(nxt_candidates) == 1, "the frame graph must stay a path"
        nxt = int(nxt_candidates[0])
        chain.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
    assert len(visited) == len(frames), "the frame graph must be connected"
    return chain, kinds


def _extract_cue(grid: np.ndarray, frames: list[Frame], bg: int) -> tuple[set[tuple[int, int]], int, int]:
    outside_mask = np.ones(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    outside_grid = np.where(outside_mask, grid, int(bg))
    comps = find_components(outside_grid, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) == 1, "expected exactly one outside cue sprite"
    coords = {tuple(map(int, rc)) for rc in np.asarray(comps[0]["coords"], dtype=int)}
    colors = {int(outside_grid[r, c]) for r, c in coords}
    assert len(colors) == 1
    cue_color = colors.pop()
    cue_size = len(coords)
    assert 1 <= cue_size <= 3
    return coords, int(cue_color), int(cue_size)


def _find_head_frame(frames: list[Frame], cue_cells: set[tuple[int, int]]) -> int:
    heads: list[int] = []
    for idx, frame in enumerate(frames):
        for r, c in frame.border_cells():
            if any((r + dr, c + dc) in cue_cells for dr, dc in ORTH_DIRS):
                heads.append(idx)
                break
    assert len(heads) == 1, "the cue must touch exactly one endpoint frame"
    return int(heads[0])


def _rotate_patch(patch: np.ndarray, direction: int, turns: int) -> np.ndarray:
    k = (-int(direction) * int(turns)) % 4
    return np.rot90(patch, k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each frame interior by a cue-sized quarter-turn amount propagated along a contact chain.

    Input structure:
    - A black background contains several same-color one-cell-thick hollow square frames arranged as one contact chain.
    - Adjacent frames touch either along an edge or at a single corner point, and each frame has degree at most two.
    - Most frame cavities contain a small asymmetric sprite, some cavities are empty idlers, and one chain endpoint has an outside monochrome cue sprite touching it.

    Rule:
    - The cue sprite area m in {1,2,3} gives a common rotation amount of m x 90 degrees.
    - The frame touching the cue is the chain head and its cavity rotates clockwise by that amount.
    - Walk along the chain from frame to frame: an edge contact flips the rotation direction, while a corner contact keeps the direction.
    - Empty idler frames still pass the direction onward, but only non-empty cavities change; frames, cue, and layout stay fixed.

    Key insight:
    - The visible clue is split across two structures: cue area gives the angle, while the sequence of edge-versus-corner contacts determines the sign at each later frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    frame_color, frames = _detect_frame_color_and_frames(g, bg)
    cue_cells, cue_color, m = _extract_cue(g, frames, bg)
    assert cue_color != frame_color
    head_index = _find_head_frame(frames, cue_cells)
    chain, kinds = _ordered_chain(frames, head_index)

    out = g.copy()
    direction = 1
    for pos, frame_index in enumerate(chain):
        frame = frames[frame_index]
        rs, cs = frame.inner_slice
        patch = out[rs, cs].copy()
        if np.any(patch != bg):
            out[rs, cs] = _rotate_patch(patch, direction, m)
        if pos + 1 < len(chain):
            touch = kinds[(frame_index, chain[pos + 1])]
            if touch == "edge":
                direction *= -1
    return out


def _touch_kind_from_step(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    assert max(dr, dc) == 1 and (dr != 0 or dc != 0)
    return "corner" if dr == 1 and dc == 1 else "edge"


def _generate_path(length: int, profile: str) -> list[tuple[int, int]] | None:
    required: set[str]
    if profile == "mixed":
        dirs = ALL_DIRS
        required = {"edge", "corner"}
    elif profile == "edge":
        dirs = ORTH_DIRS
        required = {"edge"}
    else:
        dirs = DIAG_DIRS
        required = {"corner"}

    for _ in range(200):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < length:
            cur = path[-1]
            choices = list(dirs)
            random.shuffle(choices)
            placed = False
            for dr, dc in choices:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in used:
                    continue
                if any(max(abs(nxt[0] - prev[0]), abs(nxt[1] - prev[1])) <= 1 for prev in path[:-1]):
                    continue
                span_r = max(p[0] for p in path + [nxt]) - min(p[0] for p in path + [nxt])
                span_c = max(p[1] for p in path + [nxt]) - min(p[1] for p in path + [nxt])
                if (span_r + 1) * SIDE + 4 > 30 or (span_c + 1) * SIDE + 4 > 30:
                    continue
                path.append(nxt)
                used.add(nxt)
                placed = True
                break
            if not placed:
                break
        if len(path) != length:
            continue
        step_types = {_touch_kind_from_step(path[idx], path[idx + 1]) for idx in range(length - 1)}
        if required.issubset(step_types):
            return path
    return None


def _safe_cue_sides(head: tuple[int, int], nxt: tuple[int, int]) -> list[str]:
    dr = int(nxt[0] - head[0])
    dc = int(nxt[1] - head[1])
    sides = ["N", "S", "W", "E"]
    if dr > 0 and "S" in sides:
        sides.remove("S")
    if dr < 0 and "N" in sides:
        sides.remove("N")
    if dc > 0 and "E" in sides:
        sides.remove("E")
    if dc < 0 and "W" in sides:
        sides.remove("W")
    return sides


CUE_OPTIONS: dict[str, dict[int, tuple[tuple[tuple[int, int], ...], ...]]] = {
    "N": {
        1: (((-1, 2),),),
        2: (((-1, 1), (-1, 2)), ((-2, 2), (-1, 2))),
        3: (((-1, 1), (-1, 2), (-2, 2)), ((-1, 1), (-1, 2), (-1, 3))),
    },
    "S": {
        1: (((5, 2),),),
        2: (((5, 2), (5, 3)), ((5, 2), (6, 2))),
        3: (((5, 2), (5, 3), (6, 2)), ((5, 1), (5, 2), (5, 3))),
    },
    "W": {
        1: (((2, -1),),),
        2: (((1, -1), (2, -1)), ((2, -2), (2, -1))),
        3: (((1, -1), (2, -1), (2, -2)), ((1, -1), (2, -1), (3, -1))),
    },
    "E": {
        1: (((2, 5),),),
        2: (((2, 5), (3, 5)), ((2, 5), (2, 6))),
        3: (((2, 5), (3, 5), (2, 6)), ((1, 5), (2, 5), (3, 5))),
    },
}


def _paint_frame(grid: np.ndarray, frame: Frame, color: int) -> None:
    grid[frame.r0, frame.c0 : frame.c1 + 1] = int(color)
    grid[frame.r1, frame.c0 : frame.c1 + 1] = int(color)
    grid[frame.r0 : frame.r1 + 1, frame.c0] = int(color)
    grid[frame.r0 : frame.r1 + 1, frame.c1] = int(color)


def _render_scene(scene: Scene) -> np.ndarray:
    max_r = max([frame.r1 for frame in scene.frames] + [r for r, _ in scene.cue_cells])
    max_c = max([frame.c1 for frame in scene.frames] + [c for _, c in scene.cue_cells])
    grid = np.full((max_r + 2, max_c + 2), BG, dtype=int)
    for frame, patch in zip(scene.frames, scene.patches):
        _paint_frame(grid, frame, scene.frame_color)
        paste(grid, patch, frame.r0 + 1, frame.c0 + 1, transparent=BG)
    for r, c in scene.cue_cells:
        grid[r, c] = int(scene.cue_color)
    return grid


def _scene_output(scene: Scene, *, m_override: int | None = None, mode: str = "true") -> np.ndarray:
    turns = int(scene.m if m_override is None else m_override)
    out = _render_scene(scene)
    direction = 1
    for idx, frame in enumerate(scene.frames):
        patch = scene.patches[idx]
        if np.any(patch != BG):
            rs, cs = frame.inner_slice
            out[rs, cs] = _rotate_patch(patch, direction, turns)
        if idx == len(scene.frames) - 1:
            break
        if mode == "always_edge":
            direction *= -1
        elif mode == "always_corner":
            direction *= 1
        elif mode == "ignore_idlers":
            if np.any(patch != BG) and scene.step_types[idx] == "edge":
                direction *= -1
        else:
            if scene.step_types[idx] == "edge":
                direction *= -1
    return out


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((INNER, INNER), dtype=int)
    patch[mask] = int(color)
    return patch


def _build_scene(path: list[tuple[int, int]], scenario: dict[str, object]) -> Scene | None:
    length = len(path)
    step_types = tuple(_touch_kind_from_step(path[idx], path[idx + 1]) for idx in range(length - 1))
    require_idler = bool(scenario["idler"])

    if require_idler:
        internal = list(range(1, length - 1))
        if not internal:
            return None
        idlers = {random.choice(internal)}
        if length >= 5 and random.random() < 0.35:
            extra = [idx for idx in internal if idx not in idlers]
            if extra:
                idlers.add(random.choice(extra))
    else:
        idlers = set()
    if 0 in idlers or length - 1 in idlers:
        return None

    non_empty = [idx for idx in range(length) if idx not in idlers]
    if len(non_empty) < 2:
        return None
    if bool(scenario["profile"] == "mixed"):
        if not {"edge", "corner"}.issubset(set(step_types)):
            return None

    colors = list(PALETTE)
    random.shuffle(colors)
    frame_color = int(colors.pop())
    cue_color = int(colors.pop())
    sprite_colors = colors[: len(non_empty)]
    shape_choices = random.sample(list(SPRITE_LIBRARY), len(non_empty))

    raw_origins = [(cell_r * SIDE, cell_c * SIDE) for cell_r, cell_c in path]
    head, nxt = path[0], path[1]
    cue_side = random.choice(_safe_cue_sides(head, nxt))
    cue_rel = random.choice(list(CUE_OPTIONS[cue_side][int(scenario["m"])]))
    raw_cue = [(raw_origins[0][0] + dr, raw_origins[0][1] + dc) for dr, dc in cue_rel]

    raw_frames = [Frame(int(r0), int(c0), SIDE) for r0, c0 in raw_origins]
    all_cells = [coord for frame in raw_frames for coord in frame.border_cells()] + raw_cue
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shift_r = 1 - min_r
    shift_c = 1 - min_c
    frames = tuple(Frame(frame.r0 + shift_r, frame.c0 + shift_c, SIDE) for frame in raw_frames)
    cue_cells = tuple((r + shift_r, c + shift_c) for r, c in raw_cue)

    patches: list[np.ndarray] = []
    sprite_iter = iter(zip(sprite_colors, shape_choices))
    for idx in range(length):
        if idx in idlers:
            patches.append(np.zeros((INNER, INNER), dtype=int))
            continue
        color, shape = next(sprite_iter)
        mask = np.rot90(shape, random.randint(0, 3))
        patches.append(_make_patch(mask, int(color)))

    scene = Scene(
        frame_color=frame_color,
        cue_color=cue_color,
        m=int(scenario["m"]),
        frames=frames,
        step_types=step_types,
        patches=tuple(patches),
        cue_cells=cue_cells,
    )
    grid = _render_scene(scene)
    if max(grid.shape) > 30:
        return None
    return scene


def _cue_and_chain_are_clean(scene: Scene) -> bool:
    cue = set(scene.cue_cells)
    cue_touch_frames = 0
    for frame in scene.frames:
        edge_touch = any((r + dr, c + dc) in cue for r, c in frame.border_cells() for dr, dc in ORTH_DIRS)
        diag_touch = any(
            (r + dr, c + dc) in cue
            for r, c in frame.border_cells()
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )
        if edge_touch:
            cue_touch_frames += 1
        elif diag_touch:
            return False
    if cue_touch_frames != 1:
        return False

    for i, frame_a in enumerate(scene.frames):
        for j, frame_b in enumerate(scene.frames):
            if i >= j:
                continue
            touch = _touch_type(frame_a, frame_b)
            if abs(i - j) == 1:
                if touch != scene.step_types[min(i, j)]:
                    return False
            elif touch is not None:
                return False
    return True


def _semantic_checks(scene: Scene) -> bool:
    expected = _scene_output(scene)
    source = _render_scene(scene)
    if np.array_equal(expected, source):
        return False

    for alt_m in (1, 2, 3):
        if alt_m == scene.m:
            continue
        if np.array_equal(expected, _scene_output(scene, m_override=alt_m)):
            return False

    if {"edge", "corner"}.issubset(set(scene.step_types)) and scene.m % 2 == 1:
        if np.array_equal(expected, _scene_output(scene, mode="always_edge")):
            return False
        if np.array_equal(expected, _scene_output(scene, mode="always_corner")):
            return False

    if any(not np.any(scene.patches[idx] != BG) for idx in range(1, len(scene.frames) - 1)) and scene.m % 2 == 1:
        if np.array_equal(expected, _scene_output(scene, mode="ignore_idlers")):
            return False

    if not _cue_and_chain_are_clean(scene):
        return False

    grid = _render_scene(scene)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(scenario: dict[str, object]) -> np.ndarray | None:
    min_len = 4 if bool(scenario["idler"]) or str(scenario["profile"]) == "mixed" else 3
    length = random.randint(min_len, 5)
    path = _generate_path(length, str(scenario["profile"]))
    if path is None:
        return None
    scene = _build_scene(path, scenario)
    if scene is None:
        return None
    if not _semantic_checks(scene):
        return None
    return _render_scene(scene)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scenario = dict(SCENARIO_QUEUE[0])
        grid = _gen_attempt(scenario)
        if grid is not None:
            SCENARIO_QUEUE.rotate(-1)
            return grid
    raise RuntimeError("failed to generate a valid gear-contact-chain instance")
