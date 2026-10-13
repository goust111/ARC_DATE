from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
ORANGE = Color.ORANGE
MAROON = Color.MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """Drop payload pixels along marker-encoded directions and fill matching U-shaped containers.

    Input structure: several orange L-markers (3 of 4 cells in 2x2), one adjacent colored payload pixel each,
    and several maroon U-containers with one open side.

    Rule: infer each marker direction from the missing L corner (TL→down, TR→right, BR→up, BL→left), erase markers,
    move each payload along that direction until either (a) it reaches a container with opposite open side and matching span,
    then fill that container interior with payload color, or (b) it exits the grid edge, then paint its full ray to boundary.

    Key insight: marker geometry gives gravity orientation, and each container works only when its open side is opposite the
    payload trajectory.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    out = g.copy()
    h, w = g.shape

    markers = _find_markers(g)
    if not markers:
        return out

    containers = _find_containers(g)
    for marker_cells, payload_pos, payload_color, gravity in markers:
        # erase marker and payload source
        for r, c in marker_cells:
            out[r, c] = int(BLACK)
        sr, sc = payload_pos
        out[sr, sc] = int(BLACK)

        dr, dc = gravity
        r, c = sr, sc
        captured = False
        while True:
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < h and 0 <= c2 < w):
                break

            if dr != 0 and _is_capture_cell(r2, c2, containers, opposite(gravity), axis="row"):
                container = _pick_container(r2, c2, opposite(gravity), containers)
                if container is not None:
                    _fill_container(out, container, payload_color)
                    captured = True
                    break

            if dc != 0 and _is_capture_cell(r2, c2, containers, opposite(gravity), axis="col"):
                container = _pick_container(r2, c2, opposite(gravity), containers)
                if container is not None:
                    _fill_container(out, container, payload_color)
                    captured = True
                    break

            # check full trajectory when not captured
            out[r2, c2] = payload_color
            r, c = r2, c2

        if captured:
            continue

        # no container: paint all remaining way to boundary
        r, c = r, c
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            out[nr, nc] = payload_color
            r, c = nr, nc

    return out


def _find_markers(g: np.ndarray) -> list[tuple[list[tuple[int, int]], tuple[int, int], int, tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(ORANGE):
                continue
            # BFS to get one 4-neigh orange component
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = [(r, c)]
            while q:
                x, y = q.popleft()
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == int(ORANGE):
                        seen[nx, ny] = True
                        q.append((nx, ny))
                        comp.append((nx, ny))
            if len(comp) != 3:
                continue

            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            present = set(comp)
            corners = {
                (r0, c0): (1, 0),   # TL missing -> down
                (r0, c1): (0, 1),   # TR missing -> right
                (r1, c1): (-1, 0),  # BR missing -> up
                (r1, c0): (0, -1),  # BL missing -> left
            }
            missing_gravity = None
            for corner, gravity in corners.items():
                if corner not in present:
                    missing_gravity = gravity
                    break
            if missing_gravity is None:
                continue

            payload = None
            payload_color = None
            for r2, c2 in comp:
                for dr, dc in dirs:
                    nr, nc = r2 + dr, c2 + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in present:
                        continue
                    v = int(g[nr, nc])
                    if v != int(BLACK) and v != int(ORANGE):
                        payload = (nr, nc)
                        payload_color = v
                        break
                if payload is not None:
                    break
            if payload is None or payload_color is None:
                continue

            out.append((comp, payload, payload_color, missing_gravity))
    return out


def _find_containers(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(MAROON):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == int(MAROON):
                        seen[nx, ny] = True
                        q.append((nx, ny))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(comp) >= bbox_area or len(comp) < 3:
                continue

            top = [(rr, cc) for rr, cc in comp if rr == r0 and c0 <= cc <= c1]
            bot = [(rr, cc) for rr, cc in comp if rr == r1 and c0 <= cc <= c1]
            left = [(rr, cc) for rr, cc in comp if cc == c0 and r0 <= rr <= r1]
            right = [(rr, cc) for rr, cc in comp if cc == c1 and r0 <= rr <= r1]

            open_side = None
            if len(top) == 0:
                open_side = "up"
            elif len(bot) == 0:
                open_side = "down"
            elif len(left) == 0:
                open_side = "left"
            elif len(right) == 0:
                open_side = "right"

            if open_side is None:
                candidates = [
                    (len(top), "up"),
                    (len(bot), "down"),
                    (len(left), "left"),
                    (len(right), "right"),
                ]
                open_side = sorted(candidates, key=lambda t: (t[0], t[1]))[0][1]

            out.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "open": open_side,
                }
            )
    return out


def opposite(v: tuple[int, int]) -> str:
    dr, dc = v
    if dr == 1:
        return "down"
    if dr == -1:
        return "up"
    if dc == 1:
        return "right"
    return "left"


def _is_capture_cell(r: int, c: int, containers: list[dict[str, object]], required_open: str, axis: str) -> bool:
    for cont in containers:
        if cont["open"] != required_open:
            continue
        r0, c0, r1, c1 = cont["bbox"]
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            continue
        if required_open in {"up", "down"}:
            if c0 <= c <= c1:
                return True
        else:
            if r0 <= r <= r1:
                return True
    return False


def _pick_container(r: int, c: int, open_side: str, containers: list[dict[str, object]]) -> tuple[int, int, int, int] | None:
    for cont in containers:
        if cont["open"] != open_side:
            continue
        r0, c0, r1, c1 = cont["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return (r0, c0, r1, c1)
    return None


def _fill_container(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(out[r, c]) == int(MAROON):
                continue
            out[r, c] = color


def generate_input() -> np.ndarray:
    h, w = 14, 14
    g = np.full((h, w), int(BLACK), dtype=int)

    payload_colors = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]
    dirs = [
        (1, 0),   # down
        (0, 1),   # right
        (-1, 0),  # up
        (0, -1),  # left
    ]
    random.shuffle(dirs)
    payload_colors = random.sample(payload_colors, k=random.randint(3, 4))

    selected = list(zip(dirs, payload_colors))
    containers: list[tuple[tuple[int, int, int, int], str]] = []

    # Draw containers first, then payload/markers so markers can touch payloads.
    for i, (gravity, payload_color) in enumerate(selected):
        open_side = _open_side_for_gravity(gravity)
        if open_side is None:
            continue
        if open_side in {"up", "down"}:
            r0 = random.randint(3, 8)
            r1 = min(h - 2, r0 + random.randint(3, 5))
            c0 = random.randint(1, w - 6)
            c1 = random.randint(c0 + 3, min(w - 2, c0 + 5))
            if open_side == "up":
                _draw_u_container(g, r0, c0, r1, c1, "up")
            else:
                _draw_u_container(g, r0, c0, r1, c1, "down")
            containers.append(((r0, c0, r1, c1), open_side))
        else:
            c0 = random.randint(3, 8)
            c1 = min(w - 2, c0 + random.randint(3, 5))
            r0 = random.randint(1, h - 6)
            r1 = random.randint(r0 + 3, min(h - 2, r0 + 5))
            if open_side == "left":
                _draw_u_container(g, r0, c0, r1, c1, "left")
            else:
                _draw_u_container(g, r0, c0, r1, c1, "right")
            containers.append(((r0, c0, r1, c1), open_side))

    for i, (gravity, payload_color) in enumerate(selected):
        dr, dc = gravity
        if dr == 1:  # down, catch by open up
            if random.random() < 0.7:
                container = _choose_container(containers, "up")
                r1, c0, c1 = container[2], max(1, container[1]), min(w - 2, container[3])
                sr = max(0, r1 - 2)
                c = random.randint(c0 + 1, c1 - 1)
            else:
                sr = random.randint(0, h // 2)
                c = random.randint(0, w - 1)
        elif dr == -1:  # up, catch by down
            container = _choose_container(containers, "down")
            sr = min(h - 2, container[0] + 2) if container is not None else random.randint(h // 2, h - 2)
            c = random.randint(0, w - 1)
        elif dc == 1:  # right, catch by left
            container = _choose_container(containers, "left")
            r = random.randint(0, h - 1)
            c = max(0, (container[1] - 2)) if container is not None else random.randint(0, w // 2)
        else:  # left, catch by right
            container = _choose_container(containers, "right")
            r = random.randint(0, h - 1)
            c = min(w - 2, (container[3] + 2)) if container is not None else random.randint(w // 2, w - 1)

        # choose a legal payload position and paint it
        if dc != 0:
            sr, sc = r, c
        else:
            sr, sc = sr, c

        sr = int(np.clip(sr, 0, h - 1))
        sc = int(np.clip(sc, 0, w - 1))
        if g[sr, sc] != int(BLACK):
            sr = max(0, sr - 1)
            if g[sr, sc] != int(BLACK) and sc + 1 < w:
                sc += 1
        g[sr, sc] = int(payload_color)

        # place one L marker so payload is adjacent and missing corner encodes gravity
        base_r = max(0, min(h - 3, sr - 1 if dr < 0 else sr + (1 if dr > 0 else 0)))
        base_c = max(0, min(w - 3, sc - 1 if dc < 0 else sc + (1 if dc > 0 else 0)))
        if dr == 1:
            # missing TL, missing point is (base_r, base_c)
            cells = [(base_r, base_c + 1), (base_r + 1, base_c), (base_r + 1, base_c + 1)]
        elif dc == 1:
            # missing TR
            cells = [(base_r, base_c), (base_r + 1, base_c), (base_r + 1, base_c + 1)]
        elif dr == -1:
            # missing BR
            cells = [(base_r, base_c), (base_r, base_c + 1), (base_r + 1, base_c)]
        else:
            # missing BL
            cells = [(base_r, base_c + 1), (base_r, base_c), (base_r + 1, base_c + 1)]
        for rr, cc in cells:
            g[rr, cc] = int(ORANGE)

    return g


def _open_side_for_gravity(gravity: tuple[int, int]) -> str | None:
    if gravity == (1, 0):
        return "up"
    if gravity == (-1, 0):
        return "down"
    if gravity == (0, 1):
        return "left"
    if gravity == (0, -1):
        return "right"
    return None


def _choose_container(containers: list[tuple[tuple[int, int, int, int], str]], open_side: str) -> tuple[int, int, int, int] | None:
    cands = [b for b, o in containers if o == open_side]
    if not cands:
        return None
    return random.choice(cands)


def _draw_u_container(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, open_side: str) -> None:
    if open_side == "up":
        g[r0 + 1 : r1 + 1, c0] = int(MAROON)
        g[r0 + 1 : r1 + 1, c1] = int(MAROON)
        g[r1, c0 : c1 + 1] = int(MAROON)
    elif open_side == "down":
        g[r0 : r1, c0] = int(MAROON)
        g[r0 : r1, c1] = int(MAROON)
        g[r0, c0 : c1 + 1] = int(MAROON)
    elif open_side == "left":
        g[r0, c0 + 1 : c1 + 1] = int(MAROON)
        g[r1, c0 + 1 : c1 + 1] = int(MAROON)
        g[r0 : r1 + 1, c1] = int(MAROON)
    else:  # right
        g[r0, c0 : c1] = int(MAROON)
        g[r1, c0 : c1] = int(MAROON)
        g[r0 : r1 + 1, c0] = int(MAROON)
